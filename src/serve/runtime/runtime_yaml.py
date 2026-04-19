from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


def _as_env_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _resolve_runtime_config_path() -> Optional[Path]:
    """Resolve YAML path: if RUNTIME_CONFIG_FILE is set, only that path; else try defaults."""
    explicit = os.environ.get("RUNTIME_CONFIG_FILE", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        logger.warning("RUNTIME_CONFIG_FILE is set to %r but file not found; skipping YAML apply", explicit)
        return None

    for raw in (
        Path("/workspace/config/runtime.yaml"),
        Path(__file__).resolve().parents[3] / "config" / "runtime.yaml",
    ):
        if raw.is_file():
            return raw.resolve()

    logger.debug("No runtime YAML found; skipping apply_runtime_yaml_to_environ")
    return None


def _mapping_pairs(data: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Flatten nested dict to (ENV_NAME, value) using explicit paths."""
    out: List[Tuple[str, str]] = []

    def section(name: str) -> Dict[str, Any]:
        v = data.get(name)
        return dict(v) if isinstance(v, Mapping) else {}

    serve = section("serve")
    if "model_id" in serve:
        out.append(("MODEL_ID", _as_env_str(serve["model_id"])))
    if "port" in serve:
        out.append(("PORT", _as_env_str(serve["port"])))
    if "prefer_cuda" in serve:
        out.append(("PREFER_CUDA", _as_env_str(serve["prefer_cuda"])))

    prompt = section("prompt")
    if "system_prompt_file" in prompt:
        out.append(("SYSTEM_PROMPT_FILE", _as_env_str(prompt["system_prompt_file"])))

    logging_cfg = section("logging")
    if "enable_debug" in logging_cfg:
        out.append(("ENABLE_DEBUG_LOG", _as_env_str(logging_cfg["enable_debug"])))
    dr = logging_cfg.get("debug_response")
    if isinstance(dr, Mapping):
        if "plain_string_max" in dr:
            out.append(("DEBUG_RESPONSE_PLAIN_STRING_MAX", _as_env_str(dr["plain_string_max"])))
        if "model_output_max_chars" in dr:
            out.append(("DEBUG_RESPONSE_MODEL_OUTPUT_MAX_CHARS", _as_env_str(dr["model_output_max_chars"])))

    weights = section("weights")
    if "load_order" in weights:
        out.append(("WEIGHT_TOWER_LOAD_ORDER", _as_env_str(weights["load_order"])))
    llm = weights.get("llm")
    if isinstance(llm, Mapping) and "quant_mode" in llm:
        out.append(("WEIGHT_QUANT_MODE_LLM", _as_env_str(llm["quant_mode"])))
    llm_bnb = llm.get("bnb_4bit") if isinstance(llm, Mapping) else None
    if isinstance(llm_bnb, Mapping):
        if "quant_type" in llm_bnb:
            out.append(("LLM_BNB_4BIT_QUANT_TYPE", _as_env_str(llm_bnb["quant_type"])))
        if "double_quant" in llm_bnb:
            out.append(("LLM_BNB_4BIT_DOUBLE_QUANT", _as_env_str(llm_bnb["double_quant"])))
        if "compute_fp16" in llm_bnb:
            out.append(("LLM_BNB_4BIT_COMPUTE_FP16", _as_env_str(llm_bnb["compute_fp16"])))
    vision_q = weights.get("vision")
    if isinstance(vision_q, Mapping) and "quant_mode" in vision_q:
        out.append(("WEIGHT_QUANT_MODE_VISION", _as_env_str(vision_q["quant_mode"])))
    vision_bnb = vision_q.get("bnb_4bit") if isinstance(vision_q, Mapping) else None
    if isinstance(vision_bnb, Mapping):
        if "quant_type" in vision_bnb:
            out.append(("VISION_BNB_4BIT_QUANT_TYPE", _as_env_str(vision_bnb["quant_type"])))
        if "double_quant" in vision_bnb:
            out.append(("VISION_BNB_4BIT_DOUBLE_QUANT", _as_env_str(vision_bnb["double_quant"])))
        if "compute_fp16" in vision_bnb:
            out.append(("VISION_BNB_4BIT_COMPUTE_FP16", _as_env_str(vision_bnb["compute_fp16"])))
    cuda_w = weights.get("cuda")
    if isinstance(cuda_w, Mapping):
        if "force_reclaim_between_towers" in cuda_w:
            out.append(("FORCE_RECLAIM_BETWEEN_TOWERS", _as_env_str(cuda_w["force_reclaim_between_towers"])))
    if "bnb_modules_to_not_convert" in weights:
        bnb_skip = weights["bnb_modules_to_not_convert"]
        if isinstance(bnb_skip, (list, tuple)):
            out.append(
                ("BNB_MODULES_TO_NOT_CONVERT", ",".join(_as_env_str(x) for x in bnb_skip))
            )
        else:
            out.append(("BNB_MODULES_TO_NOT_CONVERT", _as_env_str(bnb_skip)))

    inf = section("inference")
    if "use_cache" in inf:
        out.append(("INFERENCE_USE_CACHE", _as_env_str(inf["use_cache"])))
    if "disconnect_cancel_grace_sec" in inf:
        out.append(("DISCONNECT_CANCEL_GRACE_SEC", _as_env_str(inf["disconnect_cancel_grace_sec"])))

    vision = section("vision")
    if "dynamic_image_size" in vision:
        out.append(("DYNAMIC_IMAGE_SIZE", _as_env_str(vision["dynamic_image_size"])))
    if "strict_input_size" in vision:
        out.append(("STRICT_VISION_INPUT_SIZE", _as_env_str(vision["strict_input_size"])))
    if "input_size" in vision:
        out.append(("VISION_INPUT_SIZE", _as_env_str(vision["input_size"])))

    limits = section("limits")
    if "max_images_per_request" in limits:
        out.append(("MAX_IMAGES_PER_REQUEST", _as_env_str(limits["max_images_per_request"])))
    if "max_new_tokens" in limits:
        out.append(("MAX_NEW_TOKENS", _as_env_str(limits["max_new_tokens"])))
    if "image_max_bytes" in limits:
        out.append(("IMAGE_MAX_BYTES", _as_env_str(limits["image_max_bytes"])))

    return out


def apply_runtime_yaml_to_environ() -> None:
    """Load hierarchical YAML and set os.environ for keys present in the file (YAML wins)."""
    path = _resolve_runtime_config_path()
    if path is None:
        return

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required to load RUNTIME_CONFIG_FILE; install pyyaml or rebuild the server image."
        ) from e

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, Mapping):
        logger.warning("Runtime YAML at %s is not a mapping; skipping", path)
        return
    _validate_no_legacy_weight_keys(raw)

    pairs = _mapping_pairs(raw)
    for key, val in pairs:
        os.environ[key] = val
    logger.info("Applied runtime YAML -> os.environ (%d keys) from %s", len(pairs), path)


def _validate_no_legacy_weight_keys(data: Mapping[str, Any]) -> None:
    weights = data.get("weights")
    if not isinstance(weights, Mapping):
        return
    deprecated = [k for k in ("quant_mode", "quant_scope", "bnb_4bit") if k in weights]
    if not deprecated:
        return
    raise RuntimeError(
        "Legacy weights keys are no longer supported: %s. "
        "Please migrate to weights.llm.quant_mode, weights.vision.quant_mode, "
        "weights.llm.bnb_4bit.*, and weights.vision.bnb_4bit.*."
        % ", ".join(deprecated)
    )
