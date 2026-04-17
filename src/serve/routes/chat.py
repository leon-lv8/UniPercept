from __future__ import annotations

import asyncio
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from internvl.json_assistant_repair import repair_assistant_json_corruption as _repair_assistant_json_corruption

from ..chat.chat_engine import (
    _auto_score_metrics,
    _compute_score_block,
    _iter_tokens_via_streamer,
    _merge_request_generation_config,
    _next_stream_chunk,
    _question_is_visual_only_placeholder,
    _question_with_json_score_injection,
    _sse_done,
    _sse_event,
    _strip_leading_hallucinated_score_tail,
)
from ..runtime.env_utils import _env_bool, _maybe_cuda_reclaim, _now_ts
from ..chat.openai_messages import _messages_to_chat_inputs
from ..openai_types import ChatCompletionRequest, ChatCompletionResponse
from ..runtime.state import STATE, _raise_if_model_unavailable
from ..runtime.vision_input import _pixel_dtype

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body: ChatCompletionRequest = await req.json()
    model_name = body.get("model") or STATE.model_id
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))
    request_max_tokens = body.get("max_tokens")

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
    json_score_in_user = _env_bool("JSON_SCORE_IN_USER_PROMPT", False) and bool(score_metrics)
    skip_chat_after_score = (
        bool(score_metrics)
        and _question_is_visual_only_placeholder(question)
        and not json_score_in_user
    )

    def _pixels_to_device(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.to(STATE.device, dtype=_pixel_dtype(STATE.device))  # type: ignore[arg-type]

    async def run_infer() -> str:
        # Serialize access to a single model instance to avoid GPU OOM / thread-unsafe kernels.
        async with STATE.lock:
            with torch.no_grad():
                pixel_values = _pixels_to_device(pixel_values_cpu)
                chunks: List[str] = []
                score_block = ""
                if score_metrics:
                    assert pixel_values is not None
                    score_block, _ = _compute_score_block(
                        pixel_values, generation_config, score_metrics
                    )
                    if not json_score_in_user:
                        chunks.append(score_block)
                    _maybe_cuda_reclaim(stage="after_score_before_chat")
                question_eff = (
                    _question_with_json_score_injection(question, score_block)
                    if json_score_in_user and score_block
                    else question
                )
                if not score_metrics:
                    out = STATE.model.chat(
                        str(STATE.device),
                        STATE.tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        history=history or None,
                        return_history=False,
                    )
                    if STATE.device.type == "cuda":
                        torch.cuda.synchronize(STATE.device)
                    return out.strip()
                if not skip_chat_after_score:
                    out = STATE.model.chat(
                        str(STATE.device),
                        STATE.tokenizer,
                        pixel_values,
                        question_eff,
                        generation_config,
                        history=history or None,
                        return_history=False,
                    )
                    chunks.append(_strip_leading_hallucinated_score_tail(out.strip()))
                    _maybe_cuda_reclaim(stage="after_chat")
                if STATE.device.type == "cuda":
                    torch.cuda.synchronize(STATE.device)
                merged: List[str] = []
                if chunks:
                    merged.append(chunks[0].rstrip())
                if len(chunks) > 1:
                    merged.append(chunks[1].strip())
                return "\n\n".join(s for s in merged if s).strip()

    if not stream:
        text = _repair_assistant_json_corruption(await run_infer())
        resp: ChatCompletionResponse = {
            "id": resp_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(resp)

    async def event_stream() -> AsyncGenerator[bytes, None]:
        # First chunk (role)
        yield _sse_event(
            {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        # Score block (deterministic) + optional incremental chat via streamer; single lock like before.
        async with STATE.lock:
            pixel_values = _pixels_to_device(pixel_values_cpu)
            score_block = ""
            if score_metrics:
                assert pixel_values is not None
                with torch.no_grad():
                    score_block, _ = _compute_score_block(
                        pixel_values, generation_config, score_metrics
                    )
                _maybe_cuda_reclaim(stage="stream_after_score_before_chat")
            if score_block and not json_score_in_user:
                for line in score_block.splitlines(keepends=True):
                    if not line:
                        continue
                    yield _sse_event(
                        {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": line}, "finish_reason": None}],
                        }
                    )

            if score_block and not skip_chat_after_score and not json_score_in_user:
                yield _sse_event(
                    {
                        "id": resp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": "\n\n"}, "finish_reason": None}],
                    }
                )

            question_eff = (
                _question_with_json_score_injection(question, score_block)
                if json_score_in_user and score_block
                else question
            )
            if not score_metrics or not skip_chat_after_score:
                streamer, gen_thread = _iter_tokens_via_streamer(
                    question_eff, pixel_values, generation_config, history
                )
                try:
                    it = iter(streamer)
                    # 带自动分项时，chat 段先收齐再清洗，避免流式碎片无法去掉开头的 score() 假分块
                    if score_metrics and not skip_chat_after_score:
                        buf: List[str] = []
                        while True:
                            done, piece = await asyncio.to_thread(_next_stream_chunk, it)
                            if done:
                                break
                            if piece:
                                buf.append(piece)
                        cleaned = _repair_assistant_json_corruption(
                            _strip_leading_hallucinated_score_tail("".join(buf))
                        )
                        for line in cleaned.splitlines(keepends=True):
                            if not line:
                                continue
                            yield _sse_event(
                                {
                                    "id": resp_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {"content": line}, "finish_reason": None}],
                                }
                            )
                    else:
                        # 非「先 score 再 chat」的流式路径原先逐 token 下发，未走 JSON 纠错；审美 JSON 常在仅有图无自动分项等场景落此分支
                        buf_tail: List[str] = []
                        while True:
                            done, piece = await asyncio.to_thread(_next_stream_chunk, it)
                            if done:
                                break
                            if piece:
                                buf_tail.append(piece)
                        full_tail = _strip_leading_hallucinated_score_tail("".join(buf_tail))
                        if full_tail.lstrip().startswith("{"):
                            full_tail = _repair_assistant_json_corruption(full_tail)
                        for line in full_tail.splitlines(keepends=True):
                            if not line:
                                continue
                            yield _sse_event(
                                {
                                    "id": resp_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {"content": line}, "finish_reason": None}],
                                }
                            )
                finally:
                    await asyncio.to_thread(gen_thread.join)
                    _maybe_cuda_reclaim(stage="stream_after_chat")
            if STATE.device.type == "cuda":
                torch.cuda.synchronize(STATE.device)

        # Final chunk
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

    return StreamingResponse(event_stream(), media_type="text/event-stream")
