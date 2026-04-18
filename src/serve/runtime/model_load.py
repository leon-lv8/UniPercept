from __future__ import annotations

import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer

from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel

from .env_utils import (
    _assert_no_legacy_quant_env,
    _cuda_reclaim,
    _format_cuda_mem,
    _format_cuda_mem_delta,
    _debug_log_enabled,
    _env_bool,
    _force_reclaim_between_towers,
    _maybe_cuda_reclaim,
    _now_ts,
    _quant_modules_to_not_convert_with_scope,
    _quantization_config,
    _quantized_device_map,
    _select_dtype,
    _tower_quant_modes_required,
    _tower_quantization_flags,
    _max_images_per_request,
)
from .state import STATE
from .vision_input import _vision_input_size

logger = logging.getLogger(__name__)


def _debug_enabled() -> bool:
    return _debug_log_enabled() and logger.isEnabledFor(logging.DEBUG)


def _cuda_synchronize_if_cuda(device: torch.device, *, stage: str) -> None:
    """Drain pending CUDA work before the next heavy load (helps avoid sporadic 'device not ready')."""
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
        if _debug_enabled():
            logger.debug("【加载/CUDA】synchronize：stage=%s device=%s", stage, device)
    except Exception as e:
        logger.warning("【加载/CUDA】synchronize 失败（继续加载）：stage=%s err=%s", stage, e)


def _checkpoint_shard_stats(model_path: str) -> Tuple[int, int]:
    """Return (file_count, total_bytes) for checkpoint shards under model_path."""
    root = Path(model_path)
    if not root.exists():
        return 0, 0
    # Prefer HF safetensors index when present.
    index_file = root / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            raw = json.loads(index_file.read_text(encoding="utf-8"))
            weight_map = raw.get("weight_map", {})
            if isinstance(weight_map, dict):
                shard_names = sorted({str(v) for v in weight_map.values() if isinstance(v, str)})
                total = 0
                count = 0
                for name in shard_names:
                    p = root / name
                    if p.is_file():
                        total += p.stat().st_size
                        count += 1
                return count, total
        except Exception:
            # Fall back to glob scan below.
            pass
    # Fallback: scan common checkpoint files.
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    files = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    total_bytes = sum(p.stat().st_size for p in files if p.is_file())
    return len(files), total_bytes


def _checkpoint_shards(model_path: str) -> List[Tuple[str, int]]:
    """Return ordered list of (shard_filename, size_bytes) when discoverable."""
    root = Path(model_path)
    if not root.exists():
        return []
    index_file = root / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            raw = json.loads(index_file.read_text(encoding="utf-8"))
            weight_map = raw.get("weight_map", {})
            if isinstance(weight_map, dict):
                shard_names = sorted({str(v) for v in weight_map.values() if isinstance(v, str)})
                out: List[Tuple[str, int]] = []
                for name in shard_names:
                    p = root / name
                    out.append((name, int(p.stat().st_size) if p.is_file() else 0))
                return out
        except Exception:
            return []
    return []


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.2f} KiB"
    if n < 1024 ** 3:
        return f"{n / (1024 ** 2):.2f} MiB"
    return f"{n / (1024 ** 3):.2f} GiB"


def _model_component_param_stats(model: InternVLChatModel) -> str:
    def _count_params(mod: Any) -> int:
        if mod is None:
            return 0
        try:
            return sum(int(p.numel()) for p in mod.parameters())
        except Exception:
            return 0

    llm_n = _count_params(getattr(model, "language_model", None))
    vision_n = _count_params(getattr(model, "vision_model", None))
    bridge_n = _count_params(getattr(model, "mlp1", None))
    total_n = _count_params(model)
    return (
        f"language_model={llm_n:,} | vision_model={vision_n:,} | "
        f"mlp1={bridge_n:,} | total={total_n:,}"
    )


def _ensure_internvl_accepts_quant_skip_kwargs() -> None:
    """HF may pass modules_to_not_convert into InternVLChatModel(config, **model_kwargs)."""
    sig = inspect.signature(InternVLChatModel.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return
    raise RuntimeError(
        "InternVLChatModel.__init__ 缺少 **kwargs，无法使用分塔量化（modules_to_not_convert 透传）。"
        "请重新构建 Docker 镜像以包含最新的 src/internvl/model/internvl_chat/modeling_unipercept.py，"
        "或在 compose 中挂载 ./src:/workspace/src 后重启。"
    )


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


def _system_prompt_preview(text: str, max_len: int = 96) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


async def _reload_system_prompt_runtime() -> Dict[str, Any]:
    """Hot-reload SYSTEM_PROMPT_FILE into in-memory model.system_message."""
    if STATE.model is None:
        raise RuntimeError("模型尚未就绪，无法重载系统提示词。")

    prompt, source = _configured_system_prompt()
    if not prompt or not source:
        raise RuntimeError("未读取到系统提示词。请先设置 SYSTEM_PROMPT_FILE 并确保文件内容非空。")

    # Serialize with inference path to avoid mutating model fields during generation.
    async with STATE.lock:
        if STATE.model is None:
            raise RuntimeError("模型尚未就绪，无法重载系统提示词。")
        prev_prompt = str(getattr(STATE.model, "system_message", "") or "")
        STATE.model.system_message = prompt

    changed = prev_prompt != prompt
    if _debug_enabled():
        logger.debug(
            "系统提示词热重载完成：source=%s changed=%s old_len=%s new_len=%s preview=%s",
            source,
            changed,
            len(prev_prompt),
            len(prompt),
            _system_prompt_preview(prompt),
        )
    logger.info("System prompt hot reloaded from %s (changed=%s)", source, changed)
    return {
        "ok": True,
        "source": source,
        "changed": changed,
        "length": len(prompt),
        "preview": _system_prompt_preview(prompt),
        "updated_at": _now_ts(),
    }


def _load_model(model_path: str, device: torch.device) -> Tuple[InternVLChatModel, object, Dict]:
    if _debug_enabled():
        logger.debug("【加载/分词器】开始加载，模型路径=%s，目标设备=%s", model_path, device)
        shard_count, shard_bytes = _checkpoint_shard_stats(model_path)
        logger.debug(
            "【加载/权重】检测到权重分片：数量=%s，累计大小=%s",
            shard_count,
            _human_bytes(shard_bytes),
        )
        shards = _checkpoint_shards(model_path)
        if shards:
            logger.debug("【加载/权重】分片清单（按 index.json 聚合）：")
            for i, (name, size) in enumerate(shards, start=1):
                logger.debug("【加载/权重】  - %d/%d %s (%s)", i, len(shards), name, _human_bytes(size))
    tokenizer_t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if _debug_enabled():
        logger.debug("【加载/分词器】完成，耗时=%.3fs", time.time() - tokenizer_t0)
    dtype = _select_dtype(device)
    llm_mode, vision_mode = _tower_quant_modes_required()
    if _debug_enabled():
        logger.debug(
            "【加载/策略】准备加载：device=%s dtype=%s llm_mode=%s vision_mode=%s",
            device,
            dtype,
            llm_mode,
            vision_mode,
        )
    if _can_use_single_pass_tower_load(llm_mode, vision_mode):
        logger.info("量化配置在双塔上等价，采用单次加载策略。")
        load_in_8bit, load_in_4bit = _tower_quantization_flags(llm_mode)
        if _debug_enabled():
            logger.debug(
                "【加载/策略】单次加载：load_in_8bit=%s load_in_4bit=%s quant_scope=both",
                load_in_8bit,
                load_in_4bit,
            )
        model = _load_model_once(
            model_path,
            device,
            dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            quant_scope="both",
            quant_tower="llm",
        )
    else:
        if _debug_enabled():
            logger.debug("【加载/策略】检测到 llm/vision 配置存在分歧，采用双次加载并合并。")
        model = _load_model_mixed_quant(model_path, device, dtype, llm_mode=llm_mode, vision_mode=vision_mode)

    gen_cfg = dict(
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "512")),
        do_sample=_env_bool("DO_SAMPLE", False),
        temperature=float(os.environ.get("TEMPERATURE", "0")) if _env_bool("DO_SAMPLE", False) else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen_cfg = {k: v for k, v in gen_cfg.items() if v is not None}
    return model, tokenizer, gen_cfg


def _load_model_once(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    load_in_8bit: bool,
    load_in_4bit: bool,
    quant_scope: str,
    quant_tower: str,
) -> InternVLChatModel:
    quantized = load_in_8bit or load_in_4bit
    quant_cfg = _quantization_config(load_in_8bit, load_in_4bit, tower=quant_tower) if quantized else None
    mintc = _quant_modules_to_not_convert_with_scope(load_in_8bit, load_in_4bit, quant_scope) if quantized else None
    if _debug_enabled():
        logger.debug(
            "【加载/塔】开始加载：tower=%s scope=%s model_path=%s 8bit=%s 4bit=%s",
            quant_tower,
            quant_scope,
            model_path,
            load_in_8bit,
            load_in_4bit,
        )
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "use_flash_attn": _env_bool("USE_FLASH_ATTN", True),
    }
    if quantized:
        model_kwargs["device_map"] = _quantized_device_map(device)
        model_kwargs["quantization_config"] = quant_cfg
        if mintc is not None:
            model_kwargs["modules_to_not_convert"] = mintc
    if _debug_enabled():
        qtype = getattr(quant_cfg, "bnb_4bit_quant_type", None) if quant_cfg is not None else None
        qdq = getattr(quant_cfg, "bnb_4bit_use_double_quant", None) if quant_cfg is not None else None
        qdtype = getattr(quant_cfg, "bnb_4bit_compute_dtype", None) if quant_cfg is not None else None
        logger.debug(
            "【加载/塔】参数摘要：tower=%s device_map=%s use_flash_attn=%s low_cpu_mem_usage=%s quantized=%s 4bit_quant_type=%s 4bit_double_quant=%s 4bit_compute_dtype=%s skip_modules=%s",
            quant_tower,
            model_kwargs.get("device_map"),
            model_kwargs.get("use_flash_attn"),
            model_kwargs.get("low_cpu_mem_usage"),
            quantized,
            qtype,
            qdq,
            qdtype,
            ",".join(mintc) if mintc else "(默认)",
        )
    if model_kwargs.get("modules_to_not_convert") is not None:
        _ensure_internvl_accepts_quant_skip_kwargs()
    stage_t0 = time.time()
    model = InternVLChatModel.from_pretrained(
        model_path,
        trust_remote_code=False,
        **model_kwargs,
    )
    if _debug_enabled():
        logger.debug("【加载/塔】权重加载完成：tower=%s 耗时=%.3fs", quant_tower, time.time() - stage_t0)
        logger.debug("【加载/塔】模型参数统计：tower=%s %s", quant_tower, _model_component_param_stats(model))
    if quantized:
        model = model.eval()
        if _debug_enabled():
            before, after = _cuda_reclaim(stage=f"post_load_once_quantized_{quant_tower}", force=False)
            logger.debug(
                "【加载/CUDA】reclaim：stage=%s 前=%s 后=%s",
                f"post_load_once_quantized_{quant_tower}",
                _format_cuda_mem(before),
                _format_cuda_mem(after),
            )
        else:
            _maybe_cuda_reclaim(f"post_load_once_quantized_{quant_tower}")
        return model
    model = model.to(device).eval()
    if _debug_enabled():
        before, after = _cuda_reclaim(stage=f"post_load_once_full_precision_{quant_tower}", force=False)
        logger.debug(
            "【加载/CUDA】reclaim：stage=%s 前=%s 后=%s",
            f"post_load_once_full_precision_{quant_tower}",
            _format_cuda_mem(before),
            _format_cuda_mem(after),
        )
    else:
        _maybe_cuda_reclaim(f"post_load_once_full_precision_{quant_tower}")
    return model


def _load_model_mixed_quant(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    llm_mode: str,
    vision_mode: str,
) -> InternVLChatModel:
    llm_8bit, llm_4bit = _tower_quantization_flags(llm_mode)
    vis_8bit, vis_4bit = _tower_quantization_flags(vision_mode)

    first, second, src = _tower_load_order_effective(llm_mode=llm_mode, vision_mode=vision_mode)
    logger.info(
        "请求分塔加载：llm_mode=%s，vision_mode=%s（加载顺序：%s -> %s，来源=%s）。",
        llm_mode,
        vision_mode,
        first,
        second,
        src,
    )
    if _debug_enabled():
        logger.debug("【加载/双塔】双次加载并合并到同一模型实例（顺序来源=%s）。", src)

    model_llm: Optional[InternVLChatModel] = None
    model_vis: Optional[InternVLChatModel] = None
    try:
        if first == "llm":
            model_llm = _load_model_once(
                model_path,
                device,
                dtype,
                load_in_8bit=llm_8bit,
                load_in_4bit=llm_4bit,
                quant_scope="llm",
                quant_tower="llm",
            )
        else:
            model_vis = _load_model_once(
                model_path,
                device,
                dtype,
                load_in_8bit=vis_8bit,
                load_in_4bit=vis_4bit,
                quant_scope="vision",
                quant_tower="vision",
            )

        if _force_reclaim_between_towers():
            before, after = _cuda_reclaim(stage="between_tower_loads", force=True)
            delta = _format_cuda_mem_delta(before, after)
            if _debug_enabled():
                logger.debug(
                    "【加载/CUDA】reclaim：stage=%s %s 前=%s 后=%s",
                    "between_tower_loads",
                    delta,
                    _format_cuda_mem(before),
                    _format_cuda_mem(after),
                )
            else:
                logger.info(
                    "【加载/CUDA】reclaim（双塔之间强制）：%s 前=%s 后=%s",
                    delta,
                    _format_cuda_mem(before),
                    _format_cuda_mem(after),
                )
        else:
            _maybe_cuda_reclaim("between_tower_loads")

        _cuda_synchronize_if_cuda(device, stage="between_tower_loads_before_second")

        if second == "llm":
            model_llm = _load_model_once(
                model_path,
                device,
                dtype,
                load_in_8bit=llm_8bit,
                load_in_4bit=llm_4bit,
                quant_scope="llm",
                quant_tower="llm",
            )
        else:
            model_vis = _load_model_once(
                model_path,
                device,
                dtype,
                load_in_8bit=vis_8bit,
                load_in_4bit=vis_4bit,
                quant_scope="vision",
                quant_tower="vision",
            )

        assert model_llm is not None and model_vis is not None
        model_llm.vision_model = model_vis.vision_model
        model_llm.mlp1 = model_vis.mlp1
        model_llm.eval()
        if _debug_enabled():
            logger.debug("【加载/双塔】合并完成，语言塔与视觉塔已拼装。")
        return model_llm
    finally:
        if model_vis is not None:
            try:
                # Keep only borrowed modules attached to model_llm; release temporary shell model.
                del model_vis.language_model
            except Exception:
                pass
            del model_vis
        # 双加载结束后主动做一次 CUDA reclaim，并记录前后显存对比（便于观测回收效果）
        before, after = _cuda_reclaim(stage="post_mixed_quant_merge", force=True)
        if _debug_enabled():
            logger.debug(
                "【加载/CUDA】双加载后 reclaim：前=%s 后=%s",
                _format_cuda_mem(before),
                _format_cuda_mem(after),
            )


def _tower_4bit_cfg(prefix: str) -> Tuple[str, bool, bool]:
    return (
        os.environ.get(f"{prefix}_BNB_4BIT_QUANT_TYPE", "nf4"),
        _env_bool(f"{prefix}_BNB_4BIT_DOUBLE_QUANT", True),
        _env_bool(f"{prefix}_BNB_4BIT_COMPUTE_FP16", True),
    )


def _can_use_single_pass_tower_load(llm_mode: str, vision_mode: str) -> bool:
    if llm_mode != vision_mode:
        return False
    # For identical non-4bit modes, one pass is always equivalent.
    if llm_mode != "4bit":
        return True
    # For 4bit, single pass is equivalent only when per-tower 4bit configs match.
    return _tower_4bit_cfg("LLM") == _tower_4bit_cfg("VISION")


def _tower_mode_cost(mode: str) -> int:
    # Rough relative ordering (lower is cheaper).
    # 4bit is usually cheapest, then 8bit, then none/full precision.
    if mode == "4bit":
        return 1
    if mode == "8bit":
        return 2
    return 3


def _auto_tower_load_order(*, llm_mode: str, vision_mode: str) -> str:
    """Choose which tower to load first to reduce peak memory during dual-load.

    Order by quantization tier (not tower name): load the more aggressively quantized
    tower first (4bit, then 8bit, then unquantized), then the other. When both towers
    use the same tier, load vision first (typically smaller than LLM for this model).
    """
    llm_c = _tower_mode_cost(llm_mode)
    vision_c = _tower_mode_cost(vision_mode)
    if llm_c < vision_c:
        return "llm"
    if vision_c < llm_c:
        return "vision"
    return "vision"


def _tower_load_order_effective(*, llm_mode: str, vision_mode: str) -> Tuple[str, str, str]:
    """Return (first, second, source). source is one of: config, auto."""
    raw = (os.environ.get("WEIGHT_TOWER_LOAD_ORDER") or "").strip().lower()
    if raw in {"llm", "llm_first", "llm-first", "language_first", "language-first"}:
        return "llm", "vision", "config"
    if raw in {"vision", "vision_first", "vision-first"}:
        return "vision", "llm", "config"
    if raw in {"", "auto", "default"}:
        first = _auto_tower_load_order(llm_mode=llm_mode, vision_mode=vision_mode)
        return (first, "vision" if first == "llm" else "llm", "auto")
    raise RuntimeError(
        "Invalid WEIGHT_TOWER_LOAD_ORDER=%r; use one of: auto, llm_first, vision_first." % (raw,)
    )


def _config_effective_image_size(cfg: Any) -> int:
    return int(getattr(cfg, "force_image_size", None) or cfg.vision_config.image_size)


def _validate_vision_env_against_config(model: InternVLChatModel) -> None:
    raw = os.environ.get("VISION_INPUT_SIZE")
    if not raw:
        return
    want = int(raw.strip())
    expected = _config_effective_image_size(model.config)
    if want != expected:
        dyn = bool(getattr(model.config, "dynamic_image_size", False) or _env_bool("DYNAMIC_IMAGE_SIZE", False))
        if dyn:
            logger.info(
                "VISION_INPUT_SIZE=%s differs from config image_size=%s but dynamic_image_size is enabled; continuing.",
                want,
                expected,
            )
            return
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
    inference_use_cache = _env_bool("INFERENCE_USE_CACHE", True)
    eff_dtype = next(model.parameters()).dtype
    load_in_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
    load_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    llm_mode, vision_mode = _tower_quant_modes_required()
    mixed_enabled = llm_mode != vision_mode
    mintc_llm = _quant_modules_to_not_convert_with_scope(*_tower_quantization_flags(llm_mode), "llm")
    mintc_vision = _quant_modules_to_not_convert_with_scope(*_tower_quantization_flags(vision_mode), "vision")
    mintc_repr = (
        f"llm:{','.join(mintc_llm) if mintc_llm else '(default)'};"
        f"vision:{','.join(mintc_vision) if mintc_vision else '(default)'}"
    )
    logger.info(
        "UniPercept inference profile: device=%s param_dtype=%s load_in_8bit=%s load_in_4bit=%s "
        "llm_quant_mode_effective=%s vision_quant_mode_effective=%s mixed_quantization_enabled=%s "
        "llm_bnb_4bit_quant_type=%s vision_bnb_4bit_quant_type=%s "
        "bnb_modules_to_not_convert=%s "
        "inference_use_cache=%s max_new_tokens=%s llm_attn_implementation=%s vision_use_flash_attn=%s config_image_size=%s VISION_INPUT_SIZE=%s",
        device,
        str(eff_dtype),
        load_in_8bit,
        load_in_4bit,
        llm_mode,
        vision_mode,
        mixed_enabled,
        os.environ.get("LLM_BNB_4BIT_QUANT_TYPE", "nf4"),
        os.environ.get("VISION_BNB_4BIT_QUANT_TYPE", "nf4"),
        mintc_repr,
        inference_use_cache,
        gen_cfg.get("max_new_tokens"),
        llm_impl,
        vit_fa,
        eff,
        env_vi if env_vi is not None else f"(unset, using {eff})",
    )


def _inference_profile_snapshot() -> Dict[str, Any]:
    if STATE.model is None:
        return {}
    m = STATE.model
    eff = _config_effective_image_size(m.config)
    load_in_8bit = bool(getattr(m, "is_loaded_in_8bit", False))
    load_in_4bit = bool(getattr(m, "is_loaded_in_4bit", False))
    llm_mode, vision_mode = _tower_quant_modes_required()
    mixed_enabled = llm_mode != vision_mode
    mintc_llm = _quant_modules_to_not_convert_with_scope(*_tower_quantization_flags(llm_mode), "llm")
    mintc_vision = _quant_modules_to_not_convert_with_scope(*_tower_quantization_flags(vision_mode), "vision")
    return {
        "param_dtype": str(next(m.parameters()).dtype),
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
        "llm_quant_mode_effective": llm_mode,
        "vision_quant_mode_effective": vision_mode,
        "mixed_quantization_enabled": mixed_enabled,
        "llm_bnb_4bit_quant_type": os.environ.get("LLM_BNB_4BIT_QUANT_TYPE", "nf4"),
        "llm_bnb_4bit_double_quant": _env_bool("LLM_BNB_4BIT_DOUBLE_QUANT", True),
        "llm_bnb_4bit_compute_fp16": _env_bool("LLM_BNB_4BIT_COMPUTE_FP16", True),
        "vision_bnb_4bit_quant_type": os.environ.get("VISION_BNB_4BIT_QUANT_TYPE", "nf4"),
        "vision_bnb_4bit_double_quant": _env_bool("VISION_BNB_4BIT_DOUBLE_QUANT", True),
        "vision_bnb_4bit_compute_fp16": _env_bool("VISION_BNB_4BIT_COMPUTE_FP16", True),
        "bnb_modules_to_not_convert_llm_scope": mintc_llm,
        "bnb_modules_to_not_convert_vision_scope": mintc_vision,
        "llm_attn_implementation": getattr(m.config.llm_config, "attn_implementation", None),
        "vision_use_flash_attn": getattr(m.config.vision_config, "use_flash_attn", None),
        "config_effective_image_size": eff,
        "vision_input_size_effective": _vision_input_size(),
        "inference_use_cache": _env_bool("INFERENCE_USE_CACHE", True),
        "max_new_tokens": STATE.gen_cfg.get("max_new_tokens") if STATE.gen_cfg else None,
        "max_images_per_request": _max_images_per_request(),
        "max_prompt_total_chars": os.environ.get("MAX_PROMPT_TOTAL_CHARS", "0"),
    }


def _load_model_worker(app: FastAPI) -> None:
    t0 = time.time()
    try:
        if _debug_enabled():
            logger.debug(
                "模型后台加载任务启动：model_id=%s model_path=%s device=%s",
                STATE.model_id,
                STATE.model_path,
                STATE.device,
            )
        _assert_no_legacy_quant_env()
        model, tokenizer, gen_cfg = _load_model(STATE.model_path, STATE.device)  # type: ignore[arg-type]
        if _debug_enabled():
            logger.debug("【加载进度 3/4】模型主体加载完成，开始应用运行时配置与预热。")
        _apply_system_prompt_override(model)
        _validate_vision_env_against_config(model)
        _log_model_inference_profile(model, STATE.device, gen_cfg)  # type: ignore[arg-type]

        try:
            warmup_t0 = time.time()
            if _debug_enabled():
                logger.debug("【加载进度 4/4】开始执行模型预热请求。")
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
            if _debug_enabled():
                logger.exception("模型预热失败，继续服务启动流程")
        else:
            if _debug_enabled():
                logger.debug("【加载进度 4/4】模型预热完成，耗时=%.3fs", time.time() - warmup_t0)

        STATE.tokenizer = tokenizer
        STATE.gen_cfg = gen_cfg
        STATE.model_list_created = _now_ts()
        STATE.model = model
        app.state.load_seconds = time.time() - t0
        if _debug_enabled():
            logger.debug(
                "模型加载流程结束：model_id=%s model_path=%s 总耗时=%.3fs",
                STATE.model_id,
                STATE.model_path,
                app.state.load_seconds,
            )
    except Exception as e:
        STATE.model_load_error = str(e)[:2000]
        logger.exception("Model load failed")
    finally:
        STATE.model_loading = False
