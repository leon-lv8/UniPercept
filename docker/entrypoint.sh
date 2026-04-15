#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-/models/unipercept}"
export MODEL_ID="${MODEL_ID:-unipercept}"
export PORT="${PORT:-8000}"

exec python -m uvicorn src.serve.openai_server:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1

