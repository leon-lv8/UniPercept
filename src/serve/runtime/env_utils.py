from __future__ import annotations

import gc
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore[assignment]

from .state import STATE

logger = logging.getLogger(__name__)

_torch = None


def _get_torch():
    """Import torch lazily so heavy CUDA init happens only after early startup setup."""
    global _torch
    if _torch is None:
        import torch as t

        _torch = t
    return _torch


def _now_ts() -> int:
    return int(time.time())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _debug_log_enabled() -> bool:
    return _env_bool("ENABLE_DEBUG_LOG", False)


def _cuda_mem_snapshot() -> Dict[str, Any]:
    if STATE.device is None or STATE.device.type != "cuda":
        return {}
    try:
        torch = _get_torch()
        idx = STATE.device.index if STATE.device.index is not None else torch.cuda.current_device()
        return {
            "device": f"cuda:{idx}",
            "allocated": int(torch.cuda.memory_allocated(idx)),
            "reserved": int(torch.cuda.memory_reserved(idx)),
            "max_reserved": int(torch.cuda.max_memory_reserved(idx)),
        }
    except Exception:
        return {}


def _format_cuda_mem(mem: Dict[str, Any]) -> str:
    if not mem:
        return "(n/a)"

    def hb(n: int) -> str:
        if n < 1024:
            return f"{n}B"
        if n < 1024**2:
            return f"{n / 1024:.2f}KiB"
        if n < 1024**3:
            return f"{n / (1024**2):.2f}MiB"
        return f"{n / (1024**3):.2f}GiB"

    return (
        f"{mem.get('device')} allocated={hb(int(mem.get('allocated', 0)))} "
        f"reserved={hb(int(mem.get('reserved', 0)))} "
        f"max_reserved={hb(int(mem.get('max_reserved', 0)))}"
    )


def _format_cuda_mem_delta(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """Human-readable deltas for reclaim logs (MiB, signed)."""
    if not before or not after:
        return "Δ=(n/a)"

    def mib(n: int) -> float:
        return n / (1024**2)

    da = int(after.get("allocated", 0)) - int(before.get("allocated", 0))
    dr = int(after.get("reserved", 0)) - int(before.get("reserved", 0))

    def signed_mib(delta: int) -> str:
        v = mib(delta)
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}MiB"

    return f"Δallocated={signed_mib(da)} Δreserved={signed_mib(dr)}"


def _cuda_reclaim(stage: str = "", *, force: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reclaim CUDA caches and return (before, after) memory snapshots."""
    before = _cuda_mem_snapshot()
    if STATE.device is None or STATE.device.type != "cuda":
        return before, before
    if not force and not _env_bool("INFERENCE_RECLAIM_BETWEEN_STAGES", True):
        return before, before
    gc.collect()
    torch = _get_torch()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    after = _cuda_mem_snapshot()
    return before, after


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """True when the failure is a CUDA / GPU out-of-memory (English messages included)."""
    torch = _get_torch()
    oom_cls = getattr(torch, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg and ("cuda" in msg or "gpu" in msg):
            return True
    return False


def _cuda_reclaim_after_oom(stage: str = "cuda_oom") -> None:
    """Aggressive reclaim after CUDA OOM so the next request is less likely to fail on fragmentation."""
    logger.warning("CUDA OOM 后执行显存回收：stage=%s", stage)
    try:
        before, after = _cuda_reclaim(stage=stage, force=True)
        if _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "OOM reclaim：前=%s 后=%s",
                _format_cuda_mem(before),
                _format_cuda_mem(after),
            )
    except Exception:
        logger.exception("OOM 后显存回收失败（已忽略）")
        return
    if STATE.device is None or STATE.device.type != "cuda":
        return
    try:
        torch = _get_torch()
        torch.cuda.synchronize(STATE.device)
    except Exception:
        pass


def _configure_debug_logging_from_env() -> None:
    """Enable DEBUG level for project loggers when ENABLE_DEBUG_LOG=true."""
    if not _debug_log_enabled():
        return
    # Disable HF/transformers progress bars so logs stay parseable/searchable.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Docker compose logs often aren't a TTY. Default to colored output when debug is enabled.
    # Users can disable via NO_COLOR=1.
    os.environ.setdefault("LOG_COLOR", "1")
    # Uvicorn configures logging handlers early; if handler levels remain INFO,
    # logger.setLevel(DEBUG) alone won't show any debug records. Promote both.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not (getattr(root, "handlers", None) or []):
        # Uvicorn sometimes configures only uvicorn.* handlers; ensure app logs are visible.
        def _env_truthy(name: str) -> bool:
            v = (os.environ.get(name) or "").strip().lower()
            return v in {"1", "true", "yes", "y", "on"}

        def _should_colorize() -> bool:
            if os.environ.get("NO_COLOR") is not None:
                return False
            # Common conventions: FORCE_COLOR=1 or our LOG_COLOR=1.
            if _env_truthy("FORCE_COLOR") or _env_truthy("LOG_COLOR"):
                return True
            return sys.stdout.isatty()

        class _DimDebugFormatter(logging.Formatter):
            _DIM = "\x1b[2m"
            # Use 256-color light gray for better contrast across themes.
            _FG_GRAY_245 = "\x1b[38;5;245m"
            _RESET = "\x1b[0m"

            def __init__(self) -> None:
                super().__init__("%(levelname)s: %(name)s: %(message)s")

            def format(self, record: logging.LogRecord) -> str:
                s = super().format(record)
                if _should_colorize():
                    if record.levelno == logging.DEBUG:
                        # Make the whole DEBUG line lighter.
                        dim = self._DIM if _env_truthy("DEBUG_LOG_DIM") else ""
                        # DEBUG_LOG_COLOR: 256-color gray index (e.g. 245..250). 247 ~= #9e9e9e (close to #999).
                        raw = (os.environ.get("DEBUG_LOG_COLOR") or "").strip()
                        try:
                            idx = int(raw) if raw else 247
                        except Exception:
                            idx = 247
                        idx = max(232, min(255, idx))
                        fg = f"\x1b[38;5;{idx}m"
                        return f"{fg}{dim}{s}{self._RESET}"
                    if record.levelno == logging.INFO:
                        # INFO_LOG_COLOR: 256-color index. Default 37 (soft cyan).
                        raw = (os.environ.get("INFO_LOG_COLOR") or "").strip()
                        try:
                            idx = int(raw) if raw else 37
                        except Exception:
                            idx = 37
                        idx = max(0, min(255, idx))
                        fg = f"\x1b[38;5;{idx}m"
                        return f"{fg}{s}{self._RESET}"
                return s

        h = logging.StreamHandler(stream=sys.stdout)
        h.setLevel(logging.DEBUG)
        h.setFormatter(_DimDebugFormatter())
        root.addHandler(h)
    for h in list(getattr(root, "handlers", []) or []):
        try:
            h.setLevel(logging.DEBUG)
        except Exception:
            pass

    for uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(uv_name)
        uv.setLevel(logging.DEBUG)
        for h in list(getattr(uv, "handlers", []) or []):
            try:
                h.setLevel(logging.DEBUG)
            except Exception:
                pass

    for logger_name in (
        "serve",
        "serve.runtime",
        "serve.routes",
        "serve.chat",
        "internvl",
    ):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.DEBUG)
        lg.propagate = True
    logger.info("已开启调试日志（ENABLE_DEBUG_LOG=true，日志级别=DEBUG）")


def _maybe_cuda_reclaim(stage: str = "") -> None:
    """Best-effort cache reclaim between large inference stages."""
    if STATE.device is None or STATE.device.type != "cuda":
        return
    if not _env_bool("INFERENCE_RECLAIM_BETWEEN_STAGES", True):
        return
    before, after = _cuda_reclaim(stage=stage, force=False)
    if stage and _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG):
        logger.debug("CUDA reclaim 完成：stage=%s 前=%s 后=%s", stage, _format_cuda_mem(before), _format_cuda_mem(after))


def _force_reclaim_between_towers() -> bool:
    """When true, always run a forced CUDA reclaim between dual tower loads."""
    return _env_bool("FORCE_RECLAIM_BETWEEN_TOWERS", False)


def _select_device() -> Any:
    torch = _get_torch()
    prefer_cuda = _env_bool("PREFER_CUDA", True)
    if prefer_cuda and torch.cuda.is_available():
        return torch.device(os.environ.get("CUDA_DEVICE", "cuda:0"))
    return torch.device("cpu")


def _select_dtype(device: Any) -> Any:
    torch = _get_torch()
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _normalize_quant_mode(raw: Optional[str], *, env_name: str) -> Optional[str]:
    if raw is None:
        return None
    mode = raw.strip().lower()
    if not mode:
        return None
    # Docker / YAML bool false often surfaces as the string "false"; treat as off.
    if mode in ("false", "no", "0"):
        return "none"
    if mode in ("none", "off"):
        return "none"
    if mode in ("8bit", "8-bit"):
        return "8bit"
    if mode in ("4bit", "4-bit"):
        return "4bit"
    raise RuntimeError(
        "Invalid %s=%r; use one of: none, off, 8bit, 4bit "
        "(aliases: 8-bit, 4-bit)." % (env_name, raw)
    )


def _quant_mode_to_flags(mode: str) -> Tuple[bool, bool]:
    if mode == "none":
        return False, False
    if mode == "8bit":
        return True, False
    if mode == "4bit":
        return False, True
    raise RuntimeError(f"Unsupported quant mode: {mode}")


def _assert_no_legacy_quant_env() -> None:
    deprecated = [
        k
        for k in (
            "WEIGHT_QUANT_MODE",
            "WEIGHT_QUANT_SCOPE",
            "BNB_4BIT_QUANT_TYPE",
            "BNB_4BIT_DOUBLE_QUANT",
            "BNB_4BIT_COMPUTE_FP16",
        )
        if os.environ.get(k) is not None
    ]
    if not deprecated:
        return
    raise RuntimeError(
        "Legacy quant env vars are no longer supported: %s. "
        "Use WEIGHT_QUANT_MODE_LLM / WEIGHT_QUANT_MODE_VISION and "
        "LLM_BNB_4BIT_* / VISION_BNB_4BIT_*."
        % ", ".join(deprecated)
    )


def _tower_quant_modes_required() -> Tuple[str, str]:
    _assert_no_legacy_quant_env()
    llm_mode = _normalize_quant_mode(
        os.environ.get("WEIGHT_QUANT_MODE_LLM"),
        env_name="WEIGHT_QUANT_MODE_LLM",
    )
    vision_mode = _normalize_quant_mode(
        os.environ.get("WEIGHT_QUANT_MODE_VISION"),
        env_name="WEIGHT_QUANT_MODE_VISION",
    )
    if llm_mode is None or vision_mode is None:
        missing: List[str] = []
        if llm_mode is None:
            missing.append("WEIGHT_QUANT_MODE_LLM")
        if vision_mode is None:
            missing.append("WEIGHT_QUANT_MODE_VISION")
        raise RuntimeError(
            "Tower quantization config is required. Missing: %s. "
            "Set both to one of: none, 8bit, 4bit."
            % ", ".join(missing)
        )
    return llm_mode, vision_mode


def _tower_quantization_flags(mode: str) -> Tuple[bool, bool]:
    return _quant_mode_to_flags(mode)


def _quantized_device_map(device: Any):
    if device.type != "cuda":
        raise RuntimeError("8bit/4bit quantization requires CUDA device")
    return {"": device.index if device.index is not None else 0}


def _quantization_config(load_in_8bit: bool, load_in_4bit: bool, *, tower: str) -> Optional[Any]:
    if not (load_in_8bit or load_in_4bit):
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "bitsandbytes support is not available. Rebuild with INSTALL_BITSANDBYTES=true "
            "or set WEIGHT_QUANT_MODE_LLM/WEIGHT_QUANT_MODE_VISION=none."
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    if tower not in ("llm", "vision"):
        raise RuntimeError(f"Invalid tower for 4bit config: {tower}")
    prefix = "LLM" if tower == "llm" else "VISION"
    torch = _get_torch()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=os.environ.get(f"{prefix}_BNB_4BIT_QUANT_TYPE", "nf4"),
        bnb_4bit_use_double_quant=_env_bool(f"{prefix}_BNB_4BIT_DOUBLE_QUANT", True),
        bnb_4bit_compute_dtype=(
            torch.float16 if _env_bool(f"{prefix}_BNB_4BIT_COMPUTE_FP16", True) else torch.bfloat16
        ),
    )


_LM_HEAD_MODULE = "language_model.lm_head"


def _bnb_skip_covers_module(skip_prefixes: List[str], module_path: str) -> bool:
    """Mirror HF bitsandbytes prefix rule: skip if key == path or (key + '.') is in path."""
    for key in skip_prefixes:
        if key == module_path or (key + "." in module_path):
            return True
    return False


def _dedupe_str_list_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_bnb_modules_extra() -> List[str]:
    raw = (os.environ.get("BNB_MODULES_TO_NOT_CONVERT") or "").strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


def _weight_quant_scope_normalized(scope: Optional[str]) -> str:
    s = (scope or "both").strip().lower()
    if s in ("", "all"):
        return "both"
    return s


def _quant_modules_to_not_convert_with_scope(
    load_in_8bit: bool,
    load_in_4bit: bool,
    scope: str,
) -> Optional[List[str]]:
    """Compute modules_to_not_convert with explicit scope input (both/llm/vision)."""
    scope = _weight_quant_scope_normalized(scope)
    extras = _parse_bnb_modules_extra()

    if scope in ("both", "all"):
        base: Optional[List[str]] = None
    elif scope in ("llm",):
        base = ["vision_model", "mlp1"]
    elif scope in ("vision", "vit"):
        base = ["language_model"]
    else:
        raise RuntimeError(
            "Invalid quant scope=%r; use one of: both, llm, vision "
            "(aliases: all, vit)." % (scope,)
        )

    if base is None and not extras:
        return None

    merged = _dedupe_str_list_preserve_order((base or []) + extras)

    if not _bnb_skip_covers_module(merged, _LM_HEAD_MODULE):
        merged = _dedupe_str_list_preserve_order(merged + [_LM_HEAD_MODULE])

    return merged


def _max_images_per_request() -> int:
    return max(1, int(os.environ.get("MAX_IMAGES_PER_REQUEST", "8")))


def _max_prompt_total_chars() -> Optional[int]:
    v = int(os.environ.get("MAX_PROMPT_TOTAL_CHARS", "0"))
    return None if v <= 0 else v
