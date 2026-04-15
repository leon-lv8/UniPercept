from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import io
import json
import logging
import os
import subprocess
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


def _described(value: Any, description_zh: str) -> Dict[str, str | Any]:
    return {"value": value, "description": description_zh}


_HEALTH_TOP_DESC: Dict[str, str] = {
    "status": "服务健康状态；ok 表示 HTTP 服务可用。",
    "model_loaded": "模型权重是否已完成加载并可处理推理请求。",
    "device": "推理使用的 PyTorch 设备（如 cuda:0 或 cpu）。",
    "model_id": "对外暴露的模型标识（可与 OpenAI 兼容客户端中的 model 字段对应）。",
    "load_seconds": "启动阶段加载模型所耗时间（秒）。",
    "inference_profile": "推理与视觉管线相关的关键配置快照（便于排查性能与显存问题）。",
    "gpu": "通过 nvidia-smi 查询到的 NVIDIA GPU 状态（无 GPU 或命令不可用时见子字段说明）。",
}

_INFERENCE_PROFILE_DESC: Dict[str, str] = {
    "param_dtype": "模型参数的数据类型（例如 bfloat16、float32）。",
    "llm_attn_implementation": "语言模型注意力实现方式（如 flash_attention_2、eager 等）。",
    "vision_use_flash_attn": "视觉编码器（ViT）侧是否启用 FlashAttention 类加速实现。",
    "config_effective_image_size": "配置中生效的输入图像边长（像素），来自 force_image_size 或 vision_config。",
    "vision_input_size_effective": "图像预处理实际使用的边长；可被环境变量 VISION_INPUT_SIZE 覆盖。",
    "max_new_tokens": "单次对话生成时允许的新增 token 数量上限。",
    "max_images_per_request": "单个请求中允许附带的最大图像数量。",
    "max_prompt_total_chars": "环境变量 MAX_PROMPT_TOTAL_CHARS 的原始值；0 表示不按字符数截断提示。",
}

_NVIDIA_SMI_QUERY = (
    "index,name,memory.total,memory.used,memory.free,temperature.gpu,"
    "utilization.gpu,utilization.memory,uuid,pci.bus_id,power.draw,power.limit,driver_version"
)

_NVIDIA_SMI_FIELD_KEYS = [
    "index",
    "name",
    "memory_total_mib",
    "memory_used_mib",
    "memory_free_mib",
    "temperature_gpu_c",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "uuid",
    "pci_bus_id",
    "power_draw_w",
    "power_limit_w",
    "driver_version",
]

_GPU_SECTION_DESC: Dict[str, str] = {
    "nvidia_smi_ok": "是否成功执行 nvidia-smi 并解析到至少一块 GPU。",
    "nvidia_smi_error": "当查询失败时，简要错误信息；成功时为 null。",
    "devices": "各 GPU 的静态与实时指标列表（数值来自 nvidia-smi 查询时刻）。",
}

_NVIDIA_SMI_LEAF_DESC: Dict[str, str] = {
    "index": "GPU 设备索引。",
    "name": "GPU 产品名称/型号。",
    "memory_total_mib": "显存总容量（MiB）。",
    "memory_used_mib": "当前已使用显存（MiB）。",
    "memory_free_mib": "当前空闲显存（MiB）。",
    "temperature_gpu_c": "GPU 核心温度（摄氏度）。",
    "utilization_gpu_pct": "GPU 计算利用率（%）。",
    "utilization_memory_pct": "显存控制器利用率（%）。",
    "uuid": "GPU 唯一标识符（UUID）。",
    "pci_bus_id": "PCI 总线 ID。",
    "power_draw_w": "当前功耗读数（瓦）；部分空闲状态下可能为 [N/A]。",
    "power_limit_w": "功耗上限（瓦）。",
    "driver_version": "NVIDIA 驱动版本号（与具体 GPU 行重复属 nvidia-smi 正常行为）。",
}


def _coerce_gpu_csv_field(key: str, raw: str) -> Any:
    t = raw.strip()
    if t in {"", "[N/A]", "N/A", "[Unknown Error]"}:
        return None
    if key in ("name", "uuid", "pci_bus_id", "driver_version"):
        return t
    if key == "index":
        try:
            return int(t)
        except ValueError:
            return t
    try:
        return float(t) if "." in t else int(t)
    except ValueError:
        return t


def _nvidia_smi_gpu_devices() -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    """Run nvidia-smi once; return (ok, error_message_or_none, list of flat gpu dicts)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_NVIDIA_SMI_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=float(os.environ.get("NVIDIA_SMI_TIMEOUT_SEC", "8")),
            check=False,
        )
    except FileNotFoundError:
        return False, "nvidia-smi 未找到（可能未安装 NVIDIA 驱动或未在 PATH 中）", []
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi 执行超时", []

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        return False, err[:500], []

    lines = [ln.strip() for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
    if not lines:
        return False, "nvidia-smi 无输出（可能无可用 GPU）", []

    devices: List[Dict[str, Any]] = []
    for line in lines:
        row = next(csv.reader(io.StringIO(line)))
        if len(row) < len(_NVIDIA_SMI_FIELD_KEYS):
            continue
        d: Dict[str, Any] = {}
        for i, key in enumerate(_NVIDIA_SMI_FIELD_KEYS):
            d[key] = _coerce_gpu_csv_field(key, row[i])
        devices.append(d)

    if not devices:
        return False, "未能解析 nvidia-smi 的 CSV 输出", []
    return True, None, devices


def _wrap_inference_profile(prof: Dict[str, Any]) -> Dict[str, Dict[str, str | Any]]:
    return {k: _described(v, _INFERENCE_PROFILE_DESC.get(k, f"配置项「{k}」。")) for k, v in prof.items()}


def _wrap_gpu_devices(devices: List[Dict[str, Any]]) -> List[Dict[str, Dict[str, str | Any]]]:
    out: List[Dict[str, Dict[str, str | Any]]] = []
    for dev in devices:
        out.append({k: _described(dev[k], _NVIDIA_SMI_LEAF_DESC[k]) for k in _NVIDIA_SMI_FIELD_KEYS if k in dev})
    return out


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
    ok_smi, smi_err, gpu_devices = await asyncio.to_thread(_nvidia_smi_gpu_devices)
    prof = _inference_profile_snapshot()

    out: Dict[str, Any] = {
        "status": _described("ok", _HEALTH_TOP_DESC["status"]),
        "model_loaded": _described(STATE.model is not None, _HEALTH_TOP_DESC["model_loaded"]),
        "device": _described(str(STATE.device), _HEALTH_TOP_DESC["device"]),
        "model_id": _described(STATE.model_id, _HEALTH_TOP_DESC["model_id"]),
        "load_seconds": _described(getattr(app.state, "load_seconds", None), _HEALTH_TOP_DESC["load_seconds"]),
        "gpu": _described(
            {
                "nvidia_smi_ok": _described(ok_smi, _GPU_SECTION_DESC["nvidia_smi_ok"]),
                "nvidia_smi_error": _described(smi_err, _GPU_SECTION_DESC["nvidia_smi_error"]),
                "devices": _described(_wrap_gpu_devices(gpu_devices), _GPU_SECTION_DESC["devices"]),
            },
            _HEALTH_TOP_DESC["gpu"],
        ),
    }
    if prof:
        out["inference_profile"] = _described(_wrap_inference_profile(prof), _HEALTH_TOP_DESC["inference_profile"])
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

