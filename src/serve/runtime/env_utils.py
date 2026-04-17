from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Optional, Tuple

import torch

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore[assignment]

from .state import STATE

logger = logging.getLogger(__name__)


def _now_ts() -> int:
    return int(time.time())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_cuda_reclaim(stage: str = "") -> None:
    """Best-effort cache reclaim between large inference stages."""
    if STATE.device is None or STATE.device.type != "cuda":
        return
    if not _env_bool("INFERENCE_RECLAIM_BETWEEN_STAGES", True):
        return
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        if stage:
            logger.debug("torch.cuda.ipc_collect skipped at stage=%s", stage)


def _select_device() -> torch.device:
    prefer_cuda = _env_bool("PREFER_CUDA", True)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device(os.environ.get("CUDA_DEVICE", "cuda:0"))
    return torch.device("cpu")


def _select_dtype(device: torch.device) -> torch.dtype:
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


def _quantization_config(load_in_8bit: bool, load_in_4bit: bool) -> Optional[Any]:
    if not (load_in_8bit or load_in_4bit):
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "bitsandbytes support is not available. Rebuild with INSTALL_BITSANDBYTES=true "
            "or disable LOAD_IN_8BIT/LOAD_IN_4BIT."
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=os.environ.get("BNB_4BIT_QUANT_TYPE", "nf4"),
        bnb_4bit_use_double_quant=_env_bool("BNB_4BIT_DOUBLE_QUANT", True),
        bnb_4bit_compute_dtype=(
            torch.float16 if _env_bool("BNB_4BIT_COMPUTE_FP16", True) else torch.bfloat16
        ),
    )


def _max_images_per_request() -> int:
    return max(1, int(os.environ.get("MAX_IMAGES_PER_REQUEST", "8")))


def _max_prompt_total_chars() -> Optional[int]:
    v = int(os.environ.get("MAX_PROMPT_TOTAL_CHARS", "0"))
    return None if v <= 0 else v
