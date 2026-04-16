#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${HF_MODEL_ID:-Thunderbolt215215/UniPercept}"
OUT_DIR="${HF_LOCAL_DIR:-/out}"

# 兼容 compose 里传入空 HF_ENDPOINT（会导致 huggingface_hub 组装出非法 URL）
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  unset HF_ENDPOINT
elif [[ ! "${HF_ENDPOINT}" =~ ^https?:// ]]; then
  echo "[hf-download] ERROR: HF_ENDPOINT 必须以 http:// 或 https:// 开头，当前值: ${HF_ENDPOINT}" >&2
  exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[hf-download] hf auth login（非交互）"
  # 新版 hf CLI 中 --add-to-git-credential 是开关参数，不接受 false 值。
  # 这里不写该参数，保持非交互登录且避免将 token 写入 git credential helper。
  hf auth login --token "$HF_TOKEN"
else
  echo "[hf-download] 未设置 HF_TOKEN，跳过登录（仅适用于可匿名拉取的公开仓库）"
fi

echo "[hf-download] hf download ${MODEL_ID} -> ${OUT_DIR}"
mkdir -p "$OUT_DIR"
hf download "$MODEL_ID" --local-dir "$OUT_DIR"
echo "[hf-download] 完成"
