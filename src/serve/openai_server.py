from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager, suppress
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

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore[assignment]

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


def _configured_system_prompt() -> Tuple[Optional[str], Optional[str]]:
    inline_prompt = os.environ.get("SYSTEM_PROMPT")
    if inline_prompt is not None:
        inline_prompt = inline_prompt.strip()

    file_path = (os.environ.get("SYSTEM_PROMPT_FILE") or "").strip()
    file_prompt: Optional[str] = None
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_prompt = f.read().strip()
        except OSError as e:
            raise RuntimeError(f"Failed to read SYSTEM_PROMPT_FILE: {file_path}") from e

    if inline_prompt:
        return inline_prompt, "SYSTEM_PROMPT"
    if file_prompt:
        return file_prompt, f"SYSTEM_PROMPT_FILE({file_path})"
    return None, None


def _apply_system_prompt_override(model: InternVLChatModel) -> None:
    prompt, source = _configured_system_prompt()
    if not prompt:
        return
    model.system_message = prompt
    logger.info("System prompt override applied from %s", source)


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


def _quantization_flags() -> Tuple[bool, bool]:
    load_in_8bit = _env_bool("LOAD_IN_8BIT", False)
    load_in_4bit = _env_bool("LOAD_IN_4BIT", False)
    if load_in_8bit and load_in_4bit:
        raise RuntimeError("LOAD_IN_8BIT and LOAD_IN_4BIT cannot both be true")
    return load_in_8bit, load_in_4bit


def _quantized_device_map(device: torch.device):
    if device.type != "cuda":
        raise RuntimeError("8bit/4bit quantization requires CUDA device")
    return {"": device.index if device.index is not None else 0}


def _quantization_config(load_in_8bit: bool, load_in_4bit: bool) -> Optional[BitsAndBytesConfig]:
    if not (load_in_8bit or load_in_4bit):
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "bitsandbytes support is not available. Rebuild with INSTALL_BITSANDBYTES=true "
            "or disable LOAD_IN_8BIT/LOAD_IN_4BIT."
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(load_in_4bit=True)


def _load_model(model_path: str, device: torch.device) -> Tuple[InternVLChatModel, object, Dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    dtype = _select_dtype(device)
    load_in_8bit, load_in_4bit = _quantization_flags()
    quantized = load_in_8bit or load_in_4bit
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_flash_attn": _env_bool("USE_FLASH_ATTN", True),
    }
    if quantized:
        model_kwargs["device_map"] = _quantized_device_map(device)
        model_kwargs["quantization_config"] = _quantization_config(load_in_8bit, load_in_4bit)
    model = InternVLChatModel.from_pretrained(
        model_path,
        **model_kwargs,
    )
    if quantized:
        model = model.eval()
    else:
        model = model.to(device).eval()

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
    load_in_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
    load_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    logger.info(
        "UniPercept inference profile: device=%s param_dtype=%s load_in_8bit=%s load_in_4bit=%s "
        "max_new_tokens=%s llm_attn_implementation=%s vision_use_flash_attn=%s config_image_size=%s VISION_INPUT_SIZE=%s",
        device,
        str(eff_dtype),
        load_in_8bit,
        load_in_4bit,
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
        "load_in_8bit": bool(getattr(m, "is_loaded_in_8bit", False)),
        "load_in_4bit": bool(getattr(m, "is_loaded_in_4bit", False)),
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
    "model_loading": "模型是否仍在后台加载（为 true 时推理接口会返回 503）。",
    "model_load_error": "模型加载失败时的错误摘要；成功或未失败时为 null。",
    "device": "推理使用的 PyTorch 设备（如 cuda:0 或 cpu）。",
    "model_id": "对外暴露的模型标识（可与 OpenAI 兼容客户端中的 model 字段对应）。",
    "load_seconds": "启动阶段加载模型所耗时间（秒）。",
    "inference_profile": "推理与视觉管线相关的关键配置快照（便于排查性能与显存问题）。",
    "gpu": "通过 nvidia-smi 查询到的 NVIDIA GPU 状态（无 GPU 或命令不可用时见子字段说明）。",
}

_INFERENCE_PROFILE_DESC: Dict[str, str] = {
    "param_dtype": "模型参数的数据类型（例如 bfloat16、float32）。",
    "load_in_8bit": "是否启用 8bit 量化权重加载（bitsandbytes）。",
    "load_in_4bit": "是否启用 4bit 量化权重加载（bitsandbytes）。",
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


def _openai_text_parts(content: Any) -> str:
    """Flatten OpenAI-style message content to plain text (system/assistant list parts)."""
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
    """User message -> (text with optional <image> markers, image URLs when images_allowed)."""
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
    """Drop oldest (user, assistant) pairs first; then hard-truncate the final user question if needed."""
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
    """Parse OpenAI messages into InternVL chat(history, question) + pixel_values for the latest user only."""
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


def _tokenized_inputs_for_chat(
    question: str,
    pixel_values: Optional[torch.Tensor],
    history: List[Tuple[str, str]],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Match InternVLChatModel.chat() tokenization including multi-turn history."""
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
    model_loading: bool = False
    model_load_error: Optional[str] = None
    lock: asyncio.Lock

    def __init__(self) -> None:
        self.lock = asyncio.Lock()


STATE = _State()


def _model_unavailable_detail() -> str:
    if STATE.model_load_error:
        return f"模型加载失败：{STATE.model_load_error}"
    if STATE.model_loading:
        return "模型正在后台加载中，请稍后再试；也可访问 GET /health 查看 model_loaded、model_loading 等字段。"
    return "模型未就绪，无法处理推理请求。"


def _raise_if_model_unavailable() -> None:
    if (
        STATE.model is not None
        and STATE.tokenizer is not None
        and STATE.gen_cfg is not None
        and STATE.device is not None
    ):
        return
    raise HTTPException(status_code=503, detail=_model_unavailable_detail())

# 带图时固定跑 IAA/IQA/ISTA 三项 score；可用环境变量 AUTO_SCORE_WITH_IMAGE=0 关闭（仅文本对话行为）。
# JSON_SCORE_IN_USER_PROMPT=1：有图且自动打分时，将三行分数注入 user 文案再 chat，assistant 中不再前置分数块（与 JSON 系统提示搭配）。
_SCORE_ALL_METRICS = ("iaa", "iqa", "ista")


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
    """True if, after removing <image>, there is no user text left (score-only or image-only turns)."""
    t = q.strip()
    if not t:
        return True
    return not t.replace("<image>", "").strip()


# 与 score 行展示一致；用于识别并剔除模型在正文开头仿造的分项行（无 /100、多为整数）
_FAKE_METRIC_LINE = re.compile(r"^\s*(IAA|IQA|ISTA)\s*:\s*\d+\s*$", re.IGNORECASE)

_SCORE_LINE_LABEL_CN = {
    "iaa": "IAA（审美吸引力）",
    "iqa": "IQA（技术画质）",
    "ista": "ISTA（叙事与表达）",
}


def _strip_leading_hallucinated_score_tail(text: str) -> str:
    """去掉 chat 开头误生成的 score() 及紧随的整型假分项行（真分值为 xx.xx/100，由服务端单独输出）。"""
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
    """Return display lines (same as before) plus successful metric -> float for callers."""
    if STATE.model is None or STATE.tokenizer is None or STATE.device is None:
        raise RuntimeError("Model is not loaded")
    values: Dict[str, float] = {}
    out_lines: List[str] = []
    for metric in metrics:
        label = _SCORE_LINE_LABEL_CN[metric]
        try:
            s = STATE.model.score(
                str(STATE.device),
                STATE.tokenizer,
                pixel_values,
                generation_config,
                _score_desc_for_metric(metric),
                history=None,
            )
            fv = float(s)
            values[metric] = fv
            out_lines.append(f"{label}: {fv:.2f}/100")
        except Exception:
            logger.exception("model.score failed for metric=%s", metric)
            out_lines.append(f"{label}: （计算失败）")
    # 多行分值，末尾保留换行，便于与后续 chat 正文分隔
    return "\n".join(out_lines) + "\n", values


def _score_block_for_user_prompt(score_block: str) -> str:
    """Trim trailing whitespace; wrap with instruction for JSON aesthetic.scores."""
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


def _repair_assistant_json_corruption(text: str) -> str:
    """Fix common model JSON mistakes (e.g. '\" \"value' after colon) and normalize if parseable."""
    if not text or not text.lstrip().startswith("{"):
        return text
    raw = text
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s).strip()
    if not s.startswith("{"):
        return raw

    def _apply_heuristic_fixes(blob: str) -> str:
        t = blob
        t = t.replace('"weakweaknesses":', '"weaknesses":')
        t = t.replace('"weakweaknesses" :', '"weaknesses":')
        # Unclosed string on one line immediately before next aesthetic key on the following line.
        _ae_str_keys = (
            "subject_and_content",
            "composition",
            "lighting",
            "color",
            "sharpness_and_noise",
            "mood_and_narrative",
            "technical_issues_and_suggestions",
        )
        for a, b in zip(_ae_str_keys, _ae_str_keys[1:]):
            pat = rf'("{re.escape(a)}"\s*:\s*")([^\n]+)(\n\s*"{re.escape(b)}"\s*:)'

            def _close_prev(m: re.Match[str]) -> str:
                pre, mid, suf = m.group(1), m.group(2), m.group(3)
                tail = mid.rstrip()
                # 已正常闭合：…" 结尾，或 …", 结尾（引号在逗号前）
                if tail.endswith('",') or tail.endswith('"'):
                    return m.group(0)
                return f'{pre}{mid}",{suf}'

            t = re.sub(pat, _close_prev, t)
        prev = None
        while prev != t:
            prev = t
            # Fixes `": "` + whitespace + `"` + value (spurious quote); avoid eating closing quote of `" ",` etc.
            t = re.sub(r'(:\s*")\s+"(?=[^,"\]}])', r"\1", t)
        t = t.replace('"sharp_and_noise":', '"sharpness_and_noise":')
        t = re.sub(
            r'"image_observed"\s*:\s*"true"',
            '"image_observed": true',
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r'"image_observed"\s*:\s*"false"',
            '"image_observed": false',
            t,
            flags=re.IGNORECASE,
        )
        return t

    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    fixed = _apply_heuristic_fixes(s)
    try:
        obj = json.loads(fixed)
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        return fixed if fixed != s else raw


def _merge_request_generation_config(body: ChatCompletionRequest) -> Dict[str, Any]:
    """Start from STATE.gen_cfg, then apply OpenAI-style per-request sampling overrides."""
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
) -> Tuple["TextIteratorStreamer", Thread]:
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


def _load_model_worker(app: FastAPI) -> None:
    """Blocking: load weights into GPU/RAM, optional warmup, then publish globals on STATE."""
    t0 = time.time()
    try:
        model, tokenizer, gen_cfg = _load_model(STATE.model_path, STATE.device)  # type: ignore[arg-type]
        _apply_system_prompt_override(model)
        _validate_vision_env_against_config(model)
        _log_model_inference_profile(model, STATE.device, gen_cfg)  # type: ignore[arg-type]

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
            if STATE.device.type == "cuda":  # type: ignore[union-attr]
                torch.cuda.synchronize(STATE.device)
        except Exception:
            pass

        STATE.tokenizer = tokenizer
        STATE.gen_cfg = gen_cfg
        STATE.model_list_created = _now_ts()
        STATE.model = model
        app.state.load_seconds = time.time() - t0
    except Exception as e:
        STATE.model_load_error = str(e)[:2000]
        logger.exception("Model load failed")
    finally:
        STATE.model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.model_id = os.environ.get("MODEL_ID", "unipercept")
    STATE.model_path = os.environ.get("MODEL_PATH", "/models/unipercept")
    STATE.device = _select_device()

    if not os.path.exists(STATE.model_path):
        raise RuntimeError(f"MODEL_PATH does not exist: {STATE.model_path}")

    STATE.model = None
    STATE.tokenizer = None
    STATE.gen_cfg = None
    STATE.model_list_created = 0
    STATE.model_load_error = None
    STATE.model_loading = True
    app.state.load_seconds = None

    async def _run_load() -> None:
        try:
            await asyncio.to_thread(_load_model_worker, app)
        except Exception:
            logger.exception("Background model load task raised")
            if STATE.model_load_error is None:
                STATE.model_load_error = "后台加载任务异常退出（详见服务日志）。"
            STATE.model_loading = False

    app.state.model_load_task = asyncio.create_task(_run_load())

    yield

    t = getattr(app.state, "model_load_task", None)
    if t is not None and not t.done():
        t.cancel()
        with suppress(asyncio.CancelledError):
            await t


app = FastAPI(title="UniPercept OpenAI-Compatible Server", lifespan=lifespan)


@app.get("/health")
async def health():
    ok_smi, smi_err, gpu_devices = await asyncio.to_thread(_nvidia_smi_gpu_devices)
    prof = _inference_profile_snapshot()

    out: Dict[str, Any] = {
        "status": _described("ok", _HEALTH_TOP_DESC["status"]),
        "model_loaded": _described(STATE.model is not None, _HEALTH_TOP_DESC["model_loaded"]),
        "model_loading": _described(STATE.model_loading, _HEALTH_TOP_DESC["model_loading"]),
        "model_load_error": _described(STATE.model_load_error, _HEALTH_TOP_DESC["model_load_error"]),
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
    _raise_if_model_unavailable()
    return {"object": "list", "data": _model_catalog_entries()}


@app.post("/v1/chat/completions")
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
                    await asyncio.to_thread(gen_thread.join)
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

