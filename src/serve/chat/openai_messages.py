from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch
from fastapi import HTTPException

from ..runtime.env_utils import _max_images_per_request, _max_prompt_total_chars
from ..runtime.vision_input import _load_stacked_pixel_values_cpu

logger = logging.getLogger(__name__)


def _normalize_trailing_image_placeholder(flat: str, urls: List[str]) -> str:
    if len(urls) != 1 or flat.count("<image>") != 1:
        return flat
    if not flat.rstrip().endswith("<image>"):
        return flat
    body = flat[: flat.rfind("<image>")].rstrip()
    return f"<image>\n{body}" if body else "<image>"


def _flatten_openai_user_content(parts: List[Any], *, strip_images: bool) -> Tuple[str, List[str]]:
    if not isinstance(parts, list):
        raise HTTPException(status_code=400, detail="messages[].content list parts must be objects")
    if len(parts) == 0:
        raise HTTPException(status_code=400, detail="messages[].content must be a non-empty array")
    text_bits: List[str] = []
    urls: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise HTTPException(status_code=400, detail="each messages[].content[] item must be an object")
        ptype = part.get("type")
        if ptype == "text":
            text_bits.append(str(part.get("text", "")))
        elif ptype == "image_url":
            if strip_images:
                continue
            iu = part.get("image_url")
            if isinstance(iu, str):
                url = iu
            elif isinstance(iu, dict):
                url = iu.get("url")
            else:
                url = None
            if not url or not isinstance(url, str):
                continue
            url = url.strip()
            if not url:
                continue
            text_bits.append("<image>")
            urls.append(url)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content part type: {ptype!r}")
    flat = "\n".join(text_bits)
    if not strip_images and urls:
        flat = _normalize_trailing_image_placeholder(flat, urls)
    return flat, urls


def _openai_text_parts(content: Any) -> str:
    if content is None:
        raise HTTPException(status_code=400, detail="messages[].content is required")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                bits.append(str(part.get("text", "")))
        return "\n".join(bits)
    raise HTTPException(status_code=400, detail="messages[].content must be a string or a non-empty array")


def _user_message_flat(m: dict, *, images_allowed: bool) -> Tuple[str, List[str]]:
    content = m.get("content")
    if content is None:
        raise HTTPException(status_code=400, detail="messages[].content is required")
    if isinstance(content, str):
        return content.strip(), []
    if isinstance(content, list):
        flat, urls = _flatten_openai_user_content(content, strip_images=not images_allowed)
        if images_allowed and urls:
            flat = _normalize_trailing_image_placeholder(flat, urls)
        return flat.strip(), urls
    raise HTTPException(status_code=400, detail="messages[].content must be a string or a non-empty array")


def _assistant_message_flat(m: dict) -> str:
    return _openai_text_parts(m.get("content")).strip()


def _collect_client_system_prefix(messages: List[dict]) -> str:
    chunks: List[str] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "system":
            continue
        t = _openai_text_parts(m.get("content")).strip()
        if t:
            chunks.append(t)
    return "\n\n".join(chunks).strip()


def _prompt_chars_total(history: List[Tuple[str, str]], question: str) -> int:
    n = len(question)
    for u, a in history:
        n += len(u) + len(a)
    return n


def _truncate_chat_for_max_chars(
    history: List[Tuple[str, str]], question: str, max_limit: Optional[int]
) -> Tuple[List[Tuple[str, str]], str]:
    if max_limit is None or max_limit <= 0:
        return history, question
    hist = list(history)
    while _prompt_chars_total(hist, question) > max_limit:
        if hist:
            hist.pop(0)
            logger.info("Prompt exceeded MAX_PROMPT_TOTAL_CHARS=%s; dropped oldest chat turn.", max_limit)
            continue
        keep = max_limit - 48
        question = (question[:keep] if keep > 0 else "").rstrip() + "\n... [prompt truncated by MAX_PROMPT_TOTAL_CHARS]"
        logger.info("Prompt exceeded MAX_PROMPT_TOTAL_CHARS=%s; truncated latest user text.", max_limit)
        break
    return hist, question


def _messages_to_chat_inputs(messages: List[dict]) -> Tuple[List[Tuple[str, str]], str, Optional[torch.Tensor]]:
    user_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
    if not user_indices:
        raise HTTPException(status_code=400, detail="messages must contain at least one user message")
    last_user_idx = user_indices[-1]

    system_prefix = _collect_client_system_prefix(messages)
    history: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    first_user_seen = False
    pending_urls: List[str] = []

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        if role == "system":
            continue
        if role == "user":
            text, urls_here = _user_message_flat(m, images_allowed=(i == last_user_idx))
            if not first_user_seen:
                if system_prefix:
                    text = system_prefix + ("\n\n" if text else "") + text
                first_user_seen = True
            if pending_user is None:
                pending_user = text
            else:
                pending_user = pending_user + "\n\n" + text
            if i == last_user_idx:
                pending_urls = urls_here
        else:
            if pending_user is None:
                raise HTTPException(status_code=400, detail="assistant message before any user message")
            history.append((pending_user, _assistant_message_flat(m)))
            pending_user = None

    if pending_user is None:
        raise HTTPException(status_code=400, detail="messages must end with a user message")

    last_question = pending_user

    if not last_question.strip() and not pending_urls:
        raise HTTPException(
            status_code=400,
            detail="Latest user message must include non-empty text or at least one image_url with a non-empty URL",
        )

    max_prompt_limit = _max_prompt_total_chars()
    history, last_question = _truncate_chat_for_max_chars(history, last_question, max_prompt_limit)

    pixel_values_cpu: Optional[torch.Tensor] = None
    if pending_urls:
        mx = _max_images_per_request()
        if len(pending_urls) > mx:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Too many images in the latest user message ({len(pending_urls)}); "
                    f"max is {mx} (set MAX_IMAGES_PER_REQUEST)."
                ),
            )
        try:
            pixel_values_cpu = _load_stacked_pixel_values_cpu(pending_urls)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return history, last_question, pixel_values_cpu
