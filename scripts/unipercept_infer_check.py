#!/usr/bin/env python3
"""Inspect a UniPercept / InternVL checkpoint for VRAM scheme A (no quantization).

Reads config.json only (no GPU). Use for baseline/acceptance: effective image size,
num_image_token estimate, and safe downscale notes.

Usage:
  MODEL_PATH=/path/to/ckpt python scripts/unipercept_infer_check.py
  python scripts/unipercept_infer_check.py /path/to/ckpt --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _num_image_token(data: dict) -> tuple[int, dict]:
    """Match InternVLChatModel __init__ formula (see modeling_unipercept.py)."""
    v = data.get("vision_config") or {}
    force = data.get("force_image_size")
    image_size = int(force) if force is not None else int(v.get("image_size", 448))
    patch_size = int(v.get("patch_size", 14))
    downsample_ratio = float(data.get("downsample_ratio", 0.5))
    if image_size % patch_size != 0:
        raise ValueError(f"image_size {image_size} is not divisible by patch_size {patch_size}")
    n = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
    detail = {
        "effective_image_size": image_size,
        "patch_size": patch_size,
        "downsample_ratio": downsample_ratio,
        "num_image_token": n,
    }
    return n, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="UniPercept infer / VRAM scheme A checkpoint inspector.")
    parser.add_argument("model_path", nargs="?", default=os.environ.get("MODEL_PATH"), help="HuggingFace-style dir with config.json")
    parser.add_argument("--json", action="store_true", help="Single JSON object on stdout")
    args = parser.parse_args()
    if not args.model_path:
        print("Set MODEL_PATH or pass model_path argument.", file=sys.stderr)
        return 2
    cfg_path = os.path.join(args.model_path, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"Missing {cfg_path}", file=sys.stderr)
        return 2
    with open(cfg_path, encoding="utf-8") as f:
        data = json.load(f)

    try:
        _, detail = _num_image_token(data)
    except Exception as e:
        print(f"Failed to parse vision fields: {e}", file=sys.stderr)
        return 2

    eff = detail["effective_image_size"]
    patch = detail["patch_size"]
    out = {
        "model_path": os.path.abspath(args.model_path),
        "config_effective_image_size": eff,
        "patch_size": patch,
        "downsample_ratio": detail["downsample_ratio"],
        "num_image_token": detail["num_image_token"],
        "scheme_a_notes": {
            "VISION_INPUT_SIZE": "Must equal config_effective_image_size unless you edit config.json and restart.",
            "MAX_NEW_TOKENS": "Default in openai_server is 512; lower further to reduce KV cache peak.",
            "MAX_PROMPT_TOTAL_CHARS": "Optional cap; drops middle history then hard-truncates.",
            "MAX_IMAGES_PER_REQUEST": "Default 8; lower to reduce ViT batch peak.",
            "downscale_vision": f"To reduce num_image_token: set force_image_size (and vision_config.image_size) to a "
            f"multiple of {patch} (e.g. {max(patch, eff - patch)}), keep VISION_INPUT_SIZE in sync, reload.",
        },
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print("UniPercept VRAM scheme A — checkpoint summary")
    print(f"  model_path: {out['model_path']}")
    print(f"  config_effective_image_size: {eff}")
    print(f"  patch_size: {patch}")
    print(f"  downsample_ratio: {detail['downsample_ratio']}")
    print(f"  num_image_token (estimate): {detail['num_image_token']}")
    print("Environment (see src/serve/openai_server.py):")
    print("  VISION_INPUT_SIZE must match config_effective_image_size (STRICT_VISION_INPUT_SIZE=true by default).")
    print("  MAX_NEW_TOKENS (default 512), MAX_PROMPT_TOTAL_CHARS, MAX_IMAGES_PER_REQUEST (default 8).")
    print("GET /health returns inference_profile when the server is up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
