ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

FROM ${BASE_IMAGE} AS builder

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

ARG INSTALL_FLASH_ATTN=false

# Copy only dependency manifests first to maximize build cache hits.
COPY docker/requirements.runtime.txt /workspace/docker/requirements.runtime.txt
COPY docker/requirements.server.txt /workspace/docker/requirements.server.txt

# Build wheels in a dedicated stage so runtime image stays slim.
# BuildKit cache mount makes repeated builds much faster.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip wheel && \
    mkdir -p /wheels && \
    python -m pip wheel -w /wheels -r /workspace/docker/requirements.runtime.txt && \
    python -m pip wheel -w /wheels -r /workspace/docker/requirements.server.txt && \
    if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then python -m pip wheel -w /wheels "flash_attn>=2.8.3"; fi


FROM ${BASE_IMAGE} AS runtime

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

# System deps (minimal). git is sometimes needed by HF for some repos; keep it lightweight.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (keeps install layer cacheable).
COPY docker/requirements.runtime.txt /workspace/docker/requirements.runtime.txt
COPY docker/requirements.server.txt /workspace/docker/requirements.server.txt

COPY --from=builder /wheels /wheels

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip && \
    python -m pip install --no-index --find-links=/wheels \
      -r /workspace/docker/requirements.runtime.txt \
      -r /workspace/docker/requirements.server.txt

# Copy only runtime source (avoid slow recursive copy of unrelated files).
COPY src /workspace/src
COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh

RUN chmod +x /workspace/docker/entrypoint.sh

EXPOSE 8000

ENV MODEL_PATH=/models/unipercept \
    MODEL_ID=unipercept \
    PREFER_CUDA=true \
    USE_FLASH_ATTN=true

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]

