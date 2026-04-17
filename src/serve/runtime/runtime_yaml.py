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
    if "json_score_in_user_prompt" in prompt:
        out.append(("JSON_SCORE_IN_USER_PROMPT", _as_env_str(prompt["json_score_in_user_prompt"])))

    weights = section("weights")
    if "quant_mode" in weights:
        out.append(("WEIGHT_QUANT_MODE", _as_env_str(weights["quant_mode"])))
    bnb = weights.get("bnb_4bit")
    if isinstance(bnb, Mapping):
        if "quant_type" in bnb:
            out.append(("BNB_4BIT_QUANT_TYPE", _as_env_str(bnb["quant_type"])))
        if "double_quant" in bnb:
            out.append(("BNB_4BIT_DOUBLE_QUANT", _as_env_str(bnb["double_quant"])))
        if "compute_fp16" in bnb:
            out.append(("BNB_4BIT_COMPUTE_FP16", _as_env_str(bnb["compute_fp16"])))

    inf = section("inference")
    if "use_cache" in inf:
        out.append(("INFERENCE_USE_CACHE", _as_env_str(inf["use_cache"])))

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

    pairs = _mapping_pairs(raw)
    for key, val in pairs:
        os.environ[key] = val
    logger.info("Applied runtime YAML -> os.environ (%d keys) from %s", len(pairs), path)
