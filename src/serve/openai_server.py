from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from threading import Thread
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageFile
from transformers import AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Keep import shape aligned with the existing eval script.
sys.path.append("src")
from internvl.conversation import get_conv_template
from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel

from .openai_types import ChatCompletionRequest, ChatCompletionResponse, ModelObject

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    return int(time.time())


def _model_list_entry(model_id: str) -> ModelObject:
    """OpenAI-style model object plus modality hints (InternVL is vision-language)."""
    created = STATE.model_list_created or _now_ts()
    owned_by = os.environ.get("MODEL_OWNED_BY", "unipercept")
    base: ModelObject = {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": owned_by,
    }
    if not _env_bool("MODEL_ADVERTISE_MULTIMODAL", True):
        base["model_type"] = "text"
        base["modalities"] = {"input": ["text"], "output": ["text"]}
        base["capabilities"] = {
            "multimodal": False,
            "vision": False,
            "image_input": False,
            "text_output": True,
        }
        base["supported_input_modalities"] = ["text"]
        base["supported_output_modalities"] = ["text"]
        return base

    base["model_type"] = "multimodal"
    base["modalities"] = {"input": ["text", "image"], "output": ["text"]}
    base["capabilities"] = {
        "multimodal": True,
        "vision": True,
        "image_input": True,
        "text_output": True,
    }
    base["supported_input_modalities"] = ["text", "image"]
    base["supported_output_modalities"] = ["text"]
    return base


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _model_id_suggests_vision_capabilities(model_id: str) -> bool:
    """Many OpenAI-compatible clients only enable image UI when the model id matches a heuristic."""
    m = model_id.lower()
    return any(
        needle in m
        for needle in (
            "-vl",
            "-vision",
            "vl-",
            "vision",
            "gpt-4o",
            "gpt-4-turbo",
            "o1",
            "o3",
            "claude-3",
            "gemini",
            "qwen-vl",
            "internvl",
        )
    )


def _model_catalog_entries() -> List[ModelObject]:
    """Return one or more /v1/models rows; duplicate VL alias helps ID-based clients (e.g. Cursor)."""
    primary_id = STATE.model_id
    primary = _model_list_entry(primary_id)
    if not _env_bool("MODEL_ADVERTISE_MULTIMODAL", True) or not _env_bool("MODEL_AUTO_VL_ALIAS", True):
        return _model_catalog_with_extra_ids([primary])

    if _model_id_suggests_vision_capabilities(primary_id):
        return _model_catalog_with_extra_ids([primary])

    suffix = (os.environ.get("MODEL_VL_ALIAS_SUFFIX") or "-vl").strip() or "-vl"
    vl_id = f"{primary_id}{suffix}"
    vl_entry = _model_list_entry(vl_id)
    if _env_bool("MODEL_LIST_VL_ALIAS_FIRST", True):
        ordered = [vl_entry, primary]
    else:
        ordered = [primary, vl_entry]
    return _model_catalog_with_extra_ids(ordered)


def _model_catalog_with_extra_ids(entries: List[ModelObject]) -> List[ModelObject]:
    seen = {e["id"] for e in entries}
    out = list(entries)
    for raw in os.environ.get("MODEL_LIST_EXTRA_IDS", "").split(","):
        eid = raw.strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        out.append(_model_list_entry(eid))
    return out


def _select_device() -> torch.device:
    prefer_cuda = _env_bool("PREFER_CUDA", True)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device(os.environ.get("CUDA_DEVICE", "cuda:0"))
    return torch.device("cpu")


def _select_dtype(device: torch.device) -> torch.dtype:
    # Match the existing script on GPU; keep CPU safe.
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _load_model(model_path: str, device: torch.device) -> Tuple[InternVLChatModel, object, Dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    dtype = _select_dtype(device)
    model = InternVLChatModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=_env_bool("USE_FLASH_ATTN", True),
    ).to(device).eval()

    gen_cfg = dict(
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "512")),
        do_sample=_env_bool("DO_SAMPLE", False),
        temperature=float(os.environ.get("TEMPERATURE", "0")) if _env_bool("DO_SAMPLE", False) else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # Remove None fields to avoid transformers warnings.
    gen_cfg = {k: v for k, v in gen_cfg.items() if v is not None}
    return model, tokenizer, gen_cfg


def _config_effective_image_size(cfg: Any) -> int:
    return int(getattr(cfg, "force_image_size", None) or cfg.vision_config.image_size)


def _validate_vision_env_against_config(model: InternVLChatModel) -> None:
    raw = os.environ.get("VISION_INPUT_SIZE")
    if not raw:
        return
    want = int(raw.strip())
    expected = _config_effective_image_size(model.config)
    if want != expected:
        msg = (
            f"VISION_INPUT_SIZE={want} must equal the model's effective image_size={expected} "
            f"(force_image_size or vision_config.image_size). Adjust the env var or edit config.json, then restart."
        )
        if _env_bool("STRICT_VISION_INPUT_SIZE", True):
            raise RuntimeError(msg)
        logger.warning("%s", msg)


def _log_model_inference_profile(model: InternVLChatModel, device: torch.device, gen_cfg: Dict) -> None:
    llm_impl = getattr(model.config.llm_config, "attn_implementation", None)
    vit_fa = getattr(model.config.vision_config, "use_flash_attn", None)
    eff = _config_effective_image_size(model.config)
    env_vi = os.environ.get("VISION_INPUT_SIZE")
    eff_dtype = next(model.parameters()).dtype
    logger.info(
        "UniPercept inference profile: device=%s param_dtype=%s max_new_tokens=%s llm_attn_implementation=%s "
        "vision_use_flash_attn=%s config_image_size=%s VISION_INPUT_SIZE=%s",
        device,
        str(eff_dtype),
        gen_cfg.get("max_new_tokens"),
        llm_impl,
        vit_fa,
        eff,
        env_vi if env_vi is not None else f"(unset, using {eff})",
    )


def _max_images_per_request() -> int:
    return max(1, int(os.environ.get("MAX_IMAGES_PER_REQUEST", "8")))


def _max_prompt_total_chars() -> Optional[int]:
    v = int(os.environ.get("MAX_PROMPT_TOTAL_CHARS", "0"))
    return None if v <= 0 else v


def _inference_profile_snapshot() -> Dict[str, Any]:
    if STATE.model is None:
        return {}
    m = STATE.model
    eff = _config_effective_image_size(m.config)
    return {
        "param_dtype": str(next(m.parameters()).dtype),
        "llm_attn_implementation": getattr(m.config.llm_config, "attn_implementation", None),
        "vision_use_flash_attn": getattr(m.config.vision_config, "use_flash_attn", None),
        "config_effective_image_size": eff,
        "vision_input_size_effective": _vision_input_size(),
        "max_new_tokens": STATE.gen_cfg.get("max_new_tokens") if STATE.gen_cfg else None,
        "max_images_per_request": _max_images_per_request(),
        "max_prompt_total_chars": os.environ.get("MAX_PROMPT_TOTAL_CHARS", "0"),
    }


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_vision_transform_cache: Tuple[int, object] = (0, None)


def _vision_input_size() -> int:
    raw = os.environ.get("VISION_INPUT_SIZE")
    if raw:
        return int(raw)
    try:
        cfg = STATE.model.config  # type: ignore[union-attr]
        return int(getattr(cfg, "force_image_size", None) or cfg.vision_config.image_size)
    except Exception:
        return 448


def _build_vision_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _get_vision_transform():
    global _vision_transform_cache
    size = _vision_input_size()
    if _vision_transform_cache[0] != size:
        _vision_transform_cache = (size, _build_vision_transform(size))
    return _vision_transform_cache[1]


def _pixel_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _image_max_bytes() -> int:
    return int(os.environ.get("IMAGE_MAX_BYTES", str(25 * 1024 * 1024)))


def _fetch_image_bytes(url: str) -> bytes:
    url = url.strip()
    if url.startswith("data:"):
        if ";base64," not in url:
            raise ValueError("data: URL must include ;base64,")
        b64 = url.split(";base64,", 1)[1].strip()
        try:
            raw = base64.b64decode(b64, validate=False)
        except binascii.Error as e:
            raise ValueError("invalid base64 in data URL") from e
    elif url.startswith(("http://", "https://")):
        req = Request(url, headers={"User-Agent": "UniPercept-OpenAI-Server/1.0"})
        try:
            with urlopen(req, timeout=float(os.environ.get("IMAGE_FETCH_TIMEOUT_SEC", "60"))) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise ValueError(f"failed to fetch image URL: HTTP {e.code}") from e
        except URLError as e:
            raise ValueError(f"failed to fetch image URL: {e.reason}") from e
    elif url.startswith("file://") and _env_bool("ALLOW_FILE_IMAGE_URL", False):
        path = url[7:]
        if path.startswith("//"):
            path = path[1:]
        workspace = os.path.realpath("/workspace")
        target = os.path.realpath(path)
        if not target.startswith(workspace + os.sep) and target != workspace:
            raise ValueError("file: URL must resolve under /workspace")
        with open(target, "rb") as f:
            raw = f.read()
    else:
        raise ValueError("unsupported image URL (use data:...;base64,... or http(s)://, or enable ALLOW_FILE_IMAGE_URL for file:// under /workspace)")
    if len(raw) > _image_max_bytes():
        raise ValueError(f"image exceeds IMAGE_MAX_BYTES ({_image_max_bytes()})")
    return raw


def _bytes_to_pixel_row(raw: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    t = _get_vision_transform()(img)
    return t


def _load_stacked_pixel_values_cpu(urls: List[str]) -> torch.Tensor:
    if not urls:
        raise ValueError("no image URLs")
    rows = [_bytes_to_pixel_row(_fetch_image_bytes(u)) for u in urls]
    return torch.stack(rows, dim=0)


def _normalize_trailing_image_placeholder(flat: str, urls: List[str]) -> str:
    """If the client sent [text, image], flat is 'question\\n<image>'; align with conversation.py.

    InternVLChatModel.chat() only prepends '<image>\\n' when '<image>' is absent; a trailing
    placeholder leaves visual tokens in the wrong place. Single-image trailing-only case only.
    """
    if len(urls) != 1 or flat.count("<image>") != 1:
        return flat
    if not flat.rstrip().endswith("<image>"):
        return flat
    body = flat[: flat.rfind("<image>")].rstrip()
    return f"<image>\n{body}" if body else "<image>"


def _flatten_openai_user_content(parts: List[Any], *, strip_images: bool) -> Tuple[str, List[str]]:
    """Build user text with optional <image> placeholders + ordered image URLs (OpenAI multimodal).

    strip_images=True: omit image_url parts (text only). Used for non-latest user turns so clients
    like Cherry Studio can resend full history with image_url blocks without 400.

    strip_images=False: include image_url as <image> + URL; skip parts with missing/empty url so
    placeholder resends do not fail the whole request.
    """
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
                # Client may resend empty url for prior turns or failed uploads; skip silently.
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


def _role_line(role: str, content: Any, *, images_allowed: bool) -> Tuple[str, List[str]]:
    """One message -> 'role: ...' line and image URLs (only collected for the latest user message)."""
    if content is None:
        raise HTTPException(status_code=400, detail="messages[].content is required")
    if isinstance(content, str):
        return f"{role}: {content}", []
    if isinstance(content, list):
        flat, urls = _flatten_openai_user_content(content, strip_images=not images_allowed)
        return f"{role}: {flat}", urls
    raise HTTPException(status_code=400, detail="messages[].content must be a string or a non-empty array")


def _messages_to_prompt_and_pixels(messages: List[dict]) -> Tuple[str, Optional[torch.Tensor]]:
    """Match src/eval/conversation.py: optional pixel_values + question with <image> markers."""
    parts: List[str] = []
    pending_urls: List[str] = []

    user_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
    last_user_idx = user_indices[-1] if user_indices else -1

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        allow_img = role == "user" and i == last_user_idx
        line, urls = _role_line(str(role), content, images_allowed=allow_img)
        parts.append(line)
        if urls:
            pending_urls.extend(urls)

    user_lines = [p for p in parts if p.startswith("user:")]
    sys_lines = [p for p in parts if p.startswith("system:")]
    assistant_lines = [p for p in parts if p.startswith("assistant:")]

    prompt_chunks: List[str] = []
    if sys_lines:
        prompt_chunks.append("\n".join(sys_lines))
    if assistant_lines or len(user_lines) > 1:
        prompt_chunks.append("\n".join([*user_lines[:-1], *assistant_lines]))
    if not user_lines:
        raise HTTPException(status_code=400, detail="messages must contain at least one user message")
    last_user_text = user_lines[-1].removeprefix("user: ").strip()
    if not last_user_text and not pending_urls:
        raise HTTPException(
            status_code=400,
            detail="Latest user message must include non-empty text or at least one image_url with a non-empty URL",
        )
    prompt_chunks.append(last_user_text)
    prompt = "\n".join([c for c in prompt_chunks if c.strip()])

    max_prompt_limit = _max_prompt_total_chars()
    if max_prompt_limit is not None and len(prompt) > max_prompt_limit:
        slim_chunks: List[str] = []
        if sys_lines:
            slim_chunks.append("\n".join(sys_lines))
        slim_chunks.append(last_user_text)
        prompt = "\n".join([c for c in slim_chunks if c.strip()])
        if len(prompt) > max_prompt_limit:
            keep = max_prompt_limit - 48
            prompt = (prompt[:keep] if keep > 0 else "").rstrip() + "\n... [prompt truncated by MAX_PROMPT_TOTAL_CHARS]"
        logger.info("Prompt exceeded MAX_PROMPT_TOTAL_CHARS=%s; dropped middle chat history.", max_prompt_limit)

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
    return prompt, pixel_values_cpu


def _tokenized_inputs_for_question(question: str, pixel_values: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Align with InternVLChatModel.chat() tokenization (incl. <image> -> visual token span)."""
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

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
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
    """Call next() in a worker thread without letting StopIteration cross asyncio boundaries.

    asyncio.to_thread(next, it) is unsafe: an exhausted iterator raises StopIteration, which
    cannot be set on a Future (TypeError in asyncio/uvloop).
    """
    try:
        return False, next(iterator)
    except StopIteration:
        return True, ""


class _State:
    model: Optional[InternVLChatModel] = None
    tokenizer: Optional[object] = None
    gen_cfg: Optional[Dict] = None
    device: Optional[torch.device] = None
    model_id: str = "unipercept"
    model_path: str = ""
    model_list_created: int = 0
    lock: asyncio.Lock

    def __init__(self) -> None:
        self.lock = asyncio.Lock()


STATE = _State()

def _iter_tokens_via_streamer(question: str, pixel_values: Optional[torch.Tensor]) -> Tuple["TextIteratorStreamer", Thread]:
    if STATE.model is None or STATE.tokenizer is None or STATE.gen_cfg is None or STATE.device is None:
        raise RuntimeError("Model is not loaded")

    input_ids, attention_mask, eos_token_id = _tokenized_inputs_for_question(question, pixel_values)

    streamer = TextIteratorStreamer(
        STATE.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(STATE.gen_cfg)
    generate_kwargs["eos_token_id"] = eos_token_id
    generate_kwargs["streamer"] = streamer

    def _run_generate() -> None:
        # Note: InternVLChatModel.generate wraps language_model.generate
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.model_id = os.environ.get("MODEL_ID", "unipercept")
    STATE.model_path = os.environ.get("MODEL_PATH", "/models/unipercept")
    STATE.device = _select_device()

    if not os.path.exists(STATE.model_path):
        raise RuntimeError(f"MODEL_PATH does not exist: {STATE.model_path}")

    # Load once at startup so weights live in RAM/GPU memory.
    t0 = time.time()
    model, tokenizer, gen_cfg = _load_model(STATE.model_path, STATE.device)
    _validate_vision_env_against_config(model)
    _log_model_inference_profile(model, STATE.device, gen_cfg)

    # Warmup: a tiny generate to force lazy init / CUDA kernels.
    try:
        with torch.no_grad():
            _ = model.chat(
                str(STATE.device),
                tokenizer,
                None,
                "你好",
                gen_cfg,
                history=None,
                return_history=False,
            )
        if STATE.device.type == "cuda":
            torch.cuda.synchronize(STATE.device)
    except Exception:
        # Warmup failure should not prevent serving; model is still loaded.
        pass

    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.gen_cfg = gen_cfg
    STATE.model_list_created = _now_ts()

    load_s = time.time() - t0
    app.state.load_seconds = load_s
    yield


app = FastAPI(title="UniPercept OpenAI-Compatible Server", lifespan=lifespan)


@app.get("/health")
async def health():
    out: Dict[str, Any] = {
        "status": "ok",
        "model_loaded": STATE.model is not None,
        "device": str(STATE.device),
        "model_id": STATE.model_id,
        "load_seconds": getattr(app.state, "load_seconds", None),
    }
    prof = _inference_profile_snapshot()
    if prof:
        out["inference_profile"] = prof
    return out


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": _model_catalog_entries()}


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body: ChatCompletionRequest = await req.json()
    model_name = body.get("model") or STATE.model_id
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))

    if STATE.model is None or STATE.tokenizer is None or STATE.gen_cfg is None or STATE.device is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not isinstance(messages, list) or len(messages) == 0:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    try:
        prompt, pixel_values_cpu = await asyncio.to_thread(_messages_to_prompt_and_pixels, messages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    created = _now_ts()
    resp_id = f"chatcmpl-{uuid.uuid4().hex}"

    def _pixels_to_device(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.to(STATE.device, dtype=_pixel_dtype(STATE.device))  # type: ignore[arg-type]

    async def run_infer() -> str:
        # Serialize access to a single model instance to avoid GPU OOM / thread-unsafe kernels.
        async with STATE.lock:
            with torch.no_grad():
                pixel_values = _pixels_to_device(pixel_values_cpu)
                out = STATE.model.chat(
                    str(STATE.device),
                    STATE.tokenizer,
                    pixel_values,
                    prompt,
                    dict(STATE.gen_cfg),
                    history=None,
                    return_history=False,
                )
                if STATE.device.type == "cuda":
                    torch.cuda.synchronize(STATE.device)
                return out.strip()

    if not stream:
        text = await run_infer()
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

        # True incremental streaming via transformers streamer.
        async with STATE.lock:
            pixel_values = _pixels_to_device(pixel_values_cpu)
            streamer, gen_thread = _iter_tokens_via_streamer(prompt, pixel_values)
            try:
                it = iter(streamer)
                while True:
                    done, piece = await asyncio.to_thread(_next_stream_chunk, it)
                    if done:
                        break
                    if piece:
                        yield _sse_event(
                            {
                                "id": resp_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                            }
                        )
            finally:
                # Client disconnect/cancel must not release the lock while generate() still runs;
                # otherwise the next request starts a second forward and CUDA OOMs.
                await asyncio.to_thread(gen_thread.join)

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

