from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from transformers import StoppingCriteria, StoppingCriteriaList

from internvl.json_assistant_repair import repair_assistant_json_corruption as _repair_assistant_json_corruption

from ..chat.chat_engine import (
    _auto_score_metrics,
    _compute_score_block,
    _merge_request_generation_config,
    _question_with_json_score_injection,
    _sse_done,
    _sse_event,
    _strip_leading_hallucinated_score_tail,
)
from ..runtime.env_utils import _debug_log_enabled, _maybe_cuda_reclaim, _now_ts
from ..chat.openai_messages import _messages_to_chat_inputs
from ..openai_types import ChatCompletionRequest
from ..runtime.state import STATE, _raise_if_model_unavailable
from ..runtime.vision_input import _pixel_dtype

router = APIRouter()
logger = logging.getLogger(__name__)
_DISCONNECT_CANCEL_STATS_LOCK = threading.Lock()
_DISCONNECT_CANCEL_TRIGGERED_TOTAL = 0


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
    json_score_in_user = bool(score_metrics)
    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "请求分支：score_metrics=%s json_score_in_user=%s",
            bool(score_metrics),
            json_score_in_user,
        )

    def _pixels_to_device(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.to(STATE.device, dtype=_pixel_dtype(STATE.device))  # type: ignore[arg-type]

    disconnect_grace_sec = float(os.environ.get("DISCONNECT_CANCEL_GRACE_SEC", "3"))

    def _run_infer_sync(stop_event: threading.Event) -> str:
        with torch.no_grad():
            pixel_values = _pixels_to_device(pixel_values_cpu)
            chunks: List[str] = []
            score_block = ""
            if score_metrics:
                assert pixel_values is not None
                score_block, _ = _compute_score_block(
                    pixel_values, generation_config, score_metrics
                )
                _maybe_cuda_reclaim(stage="after_score_before_chat")
            question_eff = (
                _question_with_json_score_injection(question, score_block)
                if json_score_in_user and score_block
                else question
            )
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
            out = STATE.model.chat(
                str(STATE.device),
                STATE.tokenizer,
                pixel_values,
                question_eff,
                chat_generation_config,
                history=history or None,
                return_history=False,
            )
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
            return await infer_task

    raw_text = await run_infer()
    text = _repair_assistant_json_corruption(raw_text)
    repaired_len = len(text.strip())
    if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "输出收敛：raw_len=%s repaired_len=%s",
            len(raw_text.strip()),
            repaired_len,
        )
    if not text.strip():
        if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
            logger.debug("输出收敛：repair 后为空，回退到 raw_text")
        text = raw_text
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
