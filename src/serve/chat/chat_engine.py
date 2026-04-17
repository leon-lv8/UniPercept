from __future__ import annotations

import json
import logging
import re
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import HTTPException
from transformers.generation.streamers import TextIteratorStreamer

from internvl.conversation import get_conv_template

from ..runtime.env_utils import _env_bool, _maybe_cuda_reclaim
from ..openai_types import ChatCompletionRequest
from ..runtime.state import STATE, _raise_if_model_unavailable

logger = logging.getLogger(__name__)

_SCORE_ALL_METRICS = ("iaa", "iqa", "ista")
_FAKE_METRIC_LINE = re.compile(r"^\s*(IAA|IQA|ISTA)\s*:\s*\d+\s*$", re.IGNORECASE)

_SCORE_LINE_LABEL_CN = {
    "iaa": "IAA（审美吸引力）",
    "iqa": "IQA（技术画质）",
    "ista": "ISTA（叙事与表达）",
}


def _tokenized_inputs_for_chat(
    question: str,
    pixel_values: Optional[torch.Tensor],
    history: List[Tuple[str, str]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = STATE.tokenizer
    model = STATE.model
    device = STATE.device
    assert tokenizer is not None and model is not None and device is not None

    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    if pixel_values is None:
        num_patches_list: List[int] = []
    else:
        num_patches_list = [1] * len(pixel_values)
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_context_token_id
    template = get_conv_template(model.template)
    template.system_message = getattr(model, "system_message", template.system_message)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    for old_q, old_a in history:
        template.append_message(template.roles[0], old_q)
        template.append_message(template.roles[1], old_a)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    dyn_num_image_token = (
        model._num_image_token_for_pixel_values(pixel_values) if pixel_values is not None else model.num_image_token
    )
    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN + IMG_CONTEXT_TOKEN * dyn_num_image_token * num_patches + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    return input_ids, attention_mask, eos_token_id


def _sse_event(data_obj: dict) -> bytes:
    return f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def _next_stream_chunk(iterator) -> Tuple[bool, str]:
    try:
        return False, next(iterator)
    except StopIteration:
        return True, ""


def _auto_score_metrics(pixel_values_cpu: Optional[torch.Tensor]) -> List[str]:
    if pixel_values_cpu is None:
        return []
    if not _env_bool("AUTO_SCORE_WITH_IMAGE", True):
        return []
    return list(_SCORE_ALL_METRICS)


def _score_desc_for_metric(metric: str) -> str:
    if metric == "iaa":
        return "aesthetics"
    if metric == "iqa":
        return "quality"
    if metric == "ista":
        return "structure and texture richness"
    raise ValueError(f"unknown score metric: {metric!r}")


def _question_is_visual_only_placeholder(q: str) -> bool:
    t = q.strip()
    if not t:
        return True
    return not t.replace("<image>", "").strip()


def _strip_leading_hallucinated_score_tail(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and re.fullmatch(r"score\s*\(\s*\)", lines[i].strip(), flags=re.IGNORECASE):
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    while i < len(lines):
        core = lines[i].rstrip("\r\n")
        if not core.strip():
            break
        if not _FAKE_METRIC_LINE.match(core):
            break
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "".join(lines[i:])


def _compute_score_block(
    pixel_values: torch.Tensor,
    generation_config: Dict[str, Any],
    metrics: List[str],
) -> Tuple[str, Dict[str, float]]:
    if STATE.model is None or STATE.tokenizer is None or STATE.device is None:
        raise RuntimeError("Model is not loaded")
    values: Dict[str, float] = {}
    out_lines: List[str] = []
    with torch.no_grad():
        visual_features = STATE.model.extract_feature(pixel_values)
    for metric in metrics:
        label = _SCORE_LINE_LABEL_CN[metric]
        try:
            s = STATE.model.score(
                str(STATE.device),
                STATE.tokenizer,
                pixel_values,
                generation_config,
                _score_desc_for_metric(metric),
                visual_features=visual_features,
                history=None,
            )
            fv = float(s)
            values[metric] = fv
            out_lines.append(f"{label}: {fv:.2f}/100")
        except Exception:
            logger.exception("model.score failed for metric=%s", metric)
            out_lines.append(f"{label}: （计算失败）")
        finally:
            _maybe_cuda_reclaim(stage=f"score_{metric}")
    del visual_features
    _maybe_cuda_reclaim(stage="score_all_done")
    return "\n".join(out_lines) + "\n", values


def _score_block_for_user_prompt(score_block: str) -> str:
    body = score_block.rstrip()
    return (
        "【以下为服务端已确定的 IAA/IQA/ISTA 评分，请在输出 JSON 的 aesthetic.scores 中"
        "使用 iaa/iqa/ista 三个数字字段填入与下列完全一致的数值（保留一位或两位小数均可，"
        "但必须与下列分数一致）；不得改写或另造分数。】\n"
        f"{body}"
    )


def _question_with_json_score_injection(question: str, score_block: str) -> str:
    suffix = _score_block_for_user_prompt(score_block)
    if not question.strip():
        return suffix
    return f"{question.rstrip()}\n\n{suffix}"


def _merge_request_generation_config(body: ChatCompletionRequest) -> Dict[str, Any]:
    _raise_if_model_unavailable()
    cfg: Dict[str, Any] = dict(STATE.gen_cfg)
    greedy_forced = False
    temp_explicit = "temperature" in body and body.get("temperature") is not None

    if temp_explicit:
        try:
            t = float(body["temperature"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="temperature must be a number") from e
        if t <= 0:
            cfg["do_sample"] = False
            greedy_forced = True
            cfg.pop("temperature", None)
            cfg.pop("top_p", None)
        else:
            cfg["do_sample"] = True
            cfg["temperature"] = t

    if "top_p" in body and body.get("top_p") is not None and not greedy_forced:
        try:
            tp = float(body["top_p"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="top_p must be a number") from e
        cfg["top_p"] = tp
        if tp < 1.0 and not temp_explicit:
            cfg["do_sample"] = True

    if "seed" in body and body.get("seed") is not None:
        try:
            sd = int(body["seed"])  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail="seed must be an integer") from e
        gen = torch.Generator(device=STATE.device)
        gen.manual_seed(sd & 0xFFFFFFFF)
        cfg["generator"] = gen

    return {k: v for k, v in cfg.items() if v is not None}


def _iter_tokens_via_streamer(
    question: str,
    pixel_values: Optional[torch.Tensor],
    generation_config: Dict[str, Any],
    history: List[Tuple[str, str]],
) -> Tuple[TextIteratorStreamer, Thread]:
    if STATE.model is None or STATE.tokenizer is None or STATE.gen_cfg is None or STATE.device is None:
        raise RuntimeError("Model is not loaded")

    input_ids, attention_mask, eos_token_id = _tokenized_inputs_for_chat(question, pixel_values, history)

    streamer = TextIteratorStreamer(
        STATE.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(generation_config)
    generate_kwargs["eos_token_id"] = eos_token_id
    generate_kwargs["streamer"] = streamer

    def _run_generate() -> None:
        with torch.no_grad():
            _ = STATE.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

    gen_thread = Thread(target=_run_generate, daemon=True)
    gen_thread.start()
    return streamer, gen_thread