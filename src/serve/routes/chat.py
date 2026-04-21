from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from transformers import StoppingCriteria, StoppingCriteriaList

from internvl.assistant_kv_to_json import assistant_kv_text_to_obj, sanitize_model_text

from ..chat.chat_engine import (
    _auto_score_metrics,
    _compute_score_block,
    _merge_request_generation_config,
    _sse_done,
    _sse_event,
    _strip_leading_hallucinated_score_tail,
)
from ..runtime.env_utils import (
    _cuda_reclaim_after_oom,
    _debug_log_enabled,
    _is_cuda_oom_error,
    _maybe_cuda_reclaim,
    _now_ts,
)
from ..chat.openai_messages import _messages_to_chat_inputs
from ..openai_types import ChatCompletionRequest
from ..runtime.state import STATE, _raise_if_model_unavailable
from ..runtime.vision_input import _pixel_dtype

router = APIRouter()
logger = logging.getLogger(__name__)
_DISCONNECT_CANCEL_STATS_LOCK = threading.Lock()
_DISCONNECT_CANCEL_TRIGGERED_TOTAL = 0
_INVALID_IMAGE_REFUSAL_CN_RE = re.compile(r"(图像|图片).{0,8}(无效|不可见|无法识别|损坏)", re.IGNORECASE)
_LOW_RES_HINT_RE = re.compile(r"(低分辨率|分辨率低|模糊|不清晰|像素化)", re.IGNORECASE)
_TEXT_OVERLAY_HINT_RE = re.compile(r"(文字遮挡|文字覆盖|水印过多|文本过多)", re.IGNORECASE)


class _DisconnectStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        return self._stop_event.is_set()


def _record_disconnect_cancel_stat(*, grace_sec: float, disconnected_for_sec: float, request_elapsed_sec: float) -> None:
    global _DISCONNECT_CANCEL_TRIGGERED_TOTAL
    with _DISCONNECT_CANCEL_STATS_LOCK:
        _DISCONNECT_CANCEL_TRIGGERED_TOTAL += 1
        total = _DISCONNECT_CANCEL_TRIGGERED_TOTAL
    logger.info(
        "推理协作停止触发：reason=client_disconnected total=%s grace_sec=%.2f disconnected_for_sec=%.2f request_elapsed_sec=%.3f",
        total,
        grace_sec,
        disconnected_for_sec,
        request_elapsed_sec,
    )


def _explainable_reason_code_from_text(reason_text: str) -> str:
    if _LOW_RES_HINT_RE.search(reason_text):
        return "low_resolution"
    if _TEXT_OVERLAY_HINT_RE.search(reason_text):
        return "text_overlay_heavy"
    return "unclassified_image_quality_issue"


def _normalize_score_branch_refusal(obj: Dict[str, Any], *, score_metrics: List[str]) -> Dict[str, Any]:
    """
    若服务端已完成图像评分，但模型仍输出“图像无效”拒答，则转为 insufficient_info，
    并落地可解释原因码，避免与已观测图像事实冲突。
    """
    if not score_metrics:
        return obj
    if not isinstance(obj, dict):
        return obj
    if obj.get("task_type") != "refusal":
        return obj
    rr = obj.get("refusal_reason")
    if not isinstance(rr, str) or not _INVALID_IMAGE_REFUSAL_CN_RE.search(rr):
        return obj

    reason_code = _explainable_reason_code_from_text(rr)
    out = dict(obj)
    out["schema_version"] = "1.1"
    out["task_type"] = "insufficient_info"
    out["refusal_reason"] = None
    out["summary"] = "图像已被观测，但当前信息不足以完成有效分析。"

    rb = out.get("reasoning_brief")
    if not isinstance(rb, str) or not rb.strip():
        out["reasoning_brief"] = "服务端评分链路可用，但原始拒答理由与已观测事实冲突，已降级为信息不足处理。"

    limitations = out.get("limitations")
    if not isinstance(limitations, str) or not limitations.strip():
        limitations = "图像已被观测，但当前证据不足以支撑原拒答结论。"
    if "reason_code=" not in limitations:
        limitations = f"{limitations.rstrip()}（reason_code={reason_code}）"
    out["limitations"] = limitations

    meta = out.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    notes = meta.get("notes")
    notes_text = notes if isinstance(notes, str) and notes.strip() else "无"
    marker = f"normalized_refusal_reason={reason_code}"
    if marker not in notes_text:
        notes_text = f"{notes_text}；{marker}" if notes_text != "无" else marker
    meta["notes"] = notes_text
    meta["image_observed"] = True
    out["meta"] = meta
    return out


def _merge_server_aesthetic_scores(obj: Dict[str, Any], server_scores: Dict[str, float]) -> Dict[str, Any]:
    """
    将本请求内 model.score 得到的 IAA/IQA/ISTA 合并进解析后的对象：
    - 仅当 server_scores 非空（至少一项算分成功）时，将 meta.image_observed 置为 true；
    - 仅当 task_type 为 aesthetic_analysis 且 aesthetic 为对象时，用 server_scores 中的键覆盖 aesthetic.scores 对应项。
    """
    if not server_scores or not isinstance(obj, dict):
        return obj
    out = dict(obj)
    meta = out.get("meta")
    if isinstance(meta, dict):
        out["meta"] = {**meta, "image_observed": True}
    if out.get("task_type") != "aesthetic_analysis":
        return out
    ae = out.get("aesthetic")
    if not isinstance(ae, dict):
        return out
    sc = ae.get("scores")
    new_sc: Dict[str, Any] = {"iaa": None, "iqa": None, "ista": None}
    if isinstance(sc, dict):
        for k in new_sc:
            if k in sc:
                new_sc[k] = sc[k]
    for k, v in server_scores.items():
        if k in new_sc:
            new_sc[k] = float(v)
    out["aesthetic"] = {**ae, "scores": new_sc}
    return out


# _debug 体积上限：在 config/runtime.yaml 的 logging.debug_response 中配置，
# 服务启动时由 runtime_yaml 写入 DEBUG_RESPONSE_* 环境变量（仍可直接设环境变量覆盖 YAML）。
def _b64_payload_decoded_byte_len(payload: str) -> int:
    """计算标准 base64 解码后的字节数，不分配解码缓冲区。"""
    p = "".join(payload.split())
    if not p:
        return 0
    pad = 2 if p.endswith("==") else (1 if p.endswith("=") else 0)
    return max(0, len(p) // 4 * 3 - pad)


def _debug_data_url_summary(s: str) -> Optional[str]:
    """
    若为 data:...;base64,...，则仅返回 mime 与解码后字节数，不包含任何 base64 正文。
    """
    sep = ";base64,"
    idx = s.find(sep)
    if idx < 0:
        return None
    head = s[:idx]
    if not head.lower().startswith("data:"):
        return None
    mime = head[5:] if head.startswith("data:") else "unknown"
    payload = s[idx + len(sep) :]
    n = _b64_payload_decoded_byte_len(payload)
    tag = "debug_image" if mime.lower().startswith("image/") else "debug_data"
    return f"[{tag} mime={mime} decoded_bytes={n}]"


def _debug_redact_string(s: str) -> str:
    """压缩/脱敏过长字符串；图片 data URL 仅保留 mime 与解码字节数。"""
    if not s:
        return s
    max_plain = int(os.environ.get("DEBUG_RESPONSE_PLAIN_STRING_MAX", "8192"))
    du = _debug_data_url_summary(s)
    if du is not None:
        return du
    if len(s) > max_plain:
        return s[:max_plain] + f"...<truncated total_len={len(s)}>"
    return s


def _debug_redact_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _debug_redact_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_debug_redact_value(x) for x in obj]
    if isinstance(obj, str):
        return _debug_redact_string(obj)
    return obj


def _debug_model_output_text(raw: str) -> str:
    cap = int(os.environ.get("DEBUG_RESPONSE_MODEL_OUTPUT_MAX_CHARS", "500000"))
    if len(raw) > cap:
        return raw[:cap] + f"...<truncated total_len={len(raw)}>"
    return raw


@router.post("/v1/chat/completions")
async def chat_completions(req: Request):
    request_t0 = time.time()
    body: ChatCompletionRequest = await req.json()
    model_name = body.get("model") or STATE.model_id
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))
    request_max_tokens = body.get("max_tokens")
    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "收到聊天请求：stream=%s messages=%s max_tokens=%s model=%s",
            stream,
            len(messages) if isinstance(messages, list) else -1,
            request_max_tokens,
            model_name,
        )

    _raise_if_model_unavailable()

    if not isinstance(messages, list) or len(messages) == 0:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    try:
        history, question, pixel_values_cpu = await asyncio.to_thread(_messages_to_chat_inputs, messages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    created = _now_ts()
    resp_id = f"chatcmpl-{uuid.uuid4().hex}"
    try:
        generation_config = _merge_request_generation_config(body)
    except HTTPException:
        raise
    if request_max_tokens is not None:
        try:
            parsed_max_tokens = int(request_max_tokens)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="max_tokens must be an integer") from e
        if parsed_max_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_tokens must be > 0")
        generation_config["max_new_tokens"] = parsed_max_tokens

    score_metrics = _auto_score_metrics(pixel_values_cpu)
    auto_score_branch = bool(score_metrics)
    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "请求分支：score_metrics=%s auto_score_branch=%s",
            bool(score_metrics),
            auto_score_branch,
        )

    def _pixels_to_device(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.to(STATE.device, dtype=_pixel_dtype(STATE.device))  # type: ignore[arg-type]

    disconnect_grace_sec = float(os.environ.get("DISCONNECT_CANCEL_GRACE_SEC", "3"))
    server_aesthetic_scores: Dict[str, float] = {}
    prompt_debug: Dict[str, Any] = {
        "question_user_raw": question,
        "question_for_model": question,
        "server_aesthetic_scores": {},
        "auto_score_computed": False,
    }

    def _run_infer_sync(stop_event: threading.Event) -> str:
        with torch.no_grad():
            pixel_values = _pixels_to_device(pixel_values_cpu)
            chunks: List[str] = []
            vf_for_chat: Optional[torch.Tensor] = None
            if score_metrics:
                assert pixel_values is not None
                vals, vf_for_chat = _compute_score_block(
                    pixel_values, generation_config, score_metrics
                )
                server_aesthetic_scores.clear()
                server_aesthetic_scores.update(vals)
                _maybe_cuda_reclaim(stage="after_score_before_chat")
            # 自动评分结果仅服务端合并进 JSON，不向用户消息追加分项或提示；与无评分分支相同传入 question。
            question_eff = question
            prompt_debug["question_for_model"] = question_eff
            prompt_debug["server_aesthetic_scores"] = dict(server_aesthetic_scores)
            prompt_debug["auto_score_computed"] = bool(score_metrics)
            chat_generation_config = dict(generation_config)
            chat_generation_config["stopping_criteria"] = StoppingCriteriaList(
                [_DisconnectStoppingCriteria(stop_event)]
            )
            if not score_metrics:
                out = STATE.model.chat(
                    str(STATE.device),
                    STATE.tokenizer,
                    pixel_values,
                    question,
                    chat_generation_config,
                    history=history or None,
                    return_history=False,
                )
                if STATE.device.type == "cuda":
                    torch.cuda.synchronize(STATE.device)
                return out.strip()
            try:
                out = STATE.model.chat(
                    str(STATE.device),
                    STATE.tokenizer,
                    pixel_values,
                    question_eff,
                    chat_generation_config,
                    history=history or None,
                    return_history=False,
                    visual_features=vf_for_chat,
                )
            finally:
                if vf_for_chat is not None:
                    del vf_for_chat
            out_stripped = out.strip()
            cleaned_out = _strip_leading_hallucinated_score_tail(out_stripped).strip()
            if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "输出清洗（评分分支）：raw_len=%s cleaned_len=%s fallback_to_raw=%s",
                    len(out_stripped),
                    len(cleaned_out),
                    not bool(cleaned_out),
                )
            # 避免误清洗导致正文被清空：若清洗后为空，则回退到原始模型输出。
            chunks.append(cleaned_out if cleaned_out else out_stripped)
            _maybe_cuda_reclaim(stage="after_chat")
            if STATE.device.type == "cuda":
                torch.cuda.synchronize(STATE.device)
            merged: List[str] = []
            if chunks:
                merged.append(chunks[0].rstrip())
            if len(chunks) > 1:
                merged.append(chunks[1].strip())
            return "\n\n".join(s for s in merged if s).strip()

    async def run_infer() -> str:
        # Serialize access to a single model instance to avoid GPU OOM / thread-unsafe kernels.
        async with STATE.lock:
            # Run heavy sync inference work off the event loop, so /health remains responsive.
            stop_event = threading.Event()
            infer_task = asyncio.create_task(asyncio.to_thread(_run_infer_sync, stop_event))
            disconnected_since: Optional[float] = None
            disconnect_stop_logged = False
            while not infer_task.done():
                await asyncio.sleep(0.25)
                if await req.is_disconnected():
                    if disconnected_since is None:
                        disconnected_since = time.time()
                    elif time.time() - disconnected_since >= disconnect_grace_sec:
                        stop_event.set()
                        now = time.time()
                        _record_disconnect_cancel_stat(
                            grace_sec=disconnect_grace_sec,
                            disconnected_for_sec=now - disconnected_since,
                            request_elapsed_sec=now - request_t0,
                        )
                        if (not disconnect_stop_logged) and _debug_log_enabled() and logger.isEnabledFor(
                            logging.DEBUG
                        ):
                            logger.debug(
                                "检测到客户端断连持续超过阈值，触发推理协作停止：grace=%.2fs",
                                disconnect_grace_sec,
                            )
                        disconnect_stop_logged = True
                else:
                    disconnected_since = None
            try:
                return await infer_task
            except BaseException as exc:
                if _is_cuda_oom_error(exc):
                    try:
                        await asyncio.to_thread(_cuda_reclaim_after_oom, "chat_completions")
                    except Exception:
                        logger.exception("OOM 后显存回收失败（已忽略）")
                raise

    raw_text = await run_infer()
    sanitized_text = sanitize_model_text(raw_text)
    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        rt = sanitized_text.strip()
        lines = rt.splitlines()
        logger.debug(
            "chat 路由 KV 解析入参（与 model.chat 解码输出一致）: len=%s line_count=%s starts_BEGIN=%s "
            "starts_brace=%s first_line=%r preview=%r",
            len(rt),
            len(lines),
            rt.startswith("BEGIN_UNIPERCEPT_KV"),
            rt.startswith("{"),
            (lines[0][:220] if lines else ""),
            rt.replace("\n", "⏎")[:420],
        )
    out_obj = assistant_kv_text_to_obj(sanitized_text)
    out_obj = _normalize_score_branch_refusal(out_obj, score_metrics=score_metrics)
    out_obj = _merge_server_aesthetic_scores(out_obj, server_aesthetic_scores)
    if _debug_log_enabled():
        sys_prompt = str(getattr(STATE.model, "system_message", "") or "")
        dbg: Dict[str, Any] = {
            "request": _debug_redact_value(dict(body)),
            "system_prompt": _debug_redact_string(sys_prompt),
            "prompt_to_model": {
                "question_user_raw": _debug_redact_string(str(prompt_debug.get("question_user_raw", ""))),
                "question_for_model": _debug_redact_string(str(prompt_debug.get("question_for_model", ""))),
                "server_aesthetic_scores": _debug_redact_value(dict(prompt_debug.get("server_aesthetic_scores") or {})),
                "auto_score_computed": bool(prompt_debug.get("auto_score_computed")),
            },
            "model_output_text": _debug_model_output_text(sanitized_text),
        }
        out_obj = {**out_obj, "_debug": dbg}
    text = json.dumps(out_obj, ensure_ascii=False)
    repaired_len = len(text.strip())
    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "输出收敛：raw_len=%s json_len=%s",
            len(sanitized_text.strip()),
            repaired_len,
        )
    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "请求完成（统一 SSE 返回）：stream=%s 耗时=%.3fs",
            stream,
            time.time() - request_t0,
        )
    async def one_shot_stream() -> AsyncGenerator[bytes, None]:
        yield _sse_event(
            {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )
        if text:
            yield _sse_event(
                {
                    "id": resp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
            )
        yield _sse_event(
            {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield _sse_done()

    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug("响应协议：统一 SSE 一次性包装，content_len=%s", len(text))
    return StreamingResponse(one_shot_stream(), media_type="text/event-stream")
