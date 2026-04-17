from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer

from internvl.model.internvl_chat.modeling_unipercept import InternVLChatModel

from .env_utils import (
    _env_bool,
    _now_ts,
    _quantization_config,
    _quantization_flags,
    _quantized_device_map,
    _select_dtype,
    _max_images_per_request,
)
from .state import STATE
from .vision_input import _vision_input_size

logger = logging.getLogger(__name__)


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
    logger.info(
        "UniPercept inference profile: device=%s param_dtype=%s load_in_8bit=%s load_in_4bit=%s "
        "inference_use_cache=%s max_new_tokens=%s llm_attn_implementation=%s vision_use_flash_attn=%s config_image_size=%s VISION_INPUT_SIZE=%s",
        device,
        str(eff_dtype),
        load_in_8bit,
        load_in_4bit,
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
    return {
        "param_dtype": str(next(m.parameters()).dtype),
        "load_in_8bit": bool(getattr(m, "is_loaded_in_8bit", False)),
        "load_in_4bit": bool(getattr(m, "is_loaded_in_4bit", False)),
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
