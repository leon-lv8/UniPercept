ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# Builder uses devel so `pip wheel flash_attn` can compile when no matching prebuilt wheel exists; runtime stays slim (BASE_IMAGE).
ARG BASE_IMAGE_BUILDER=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
ARG INSTALL_GIT=true
ARG INSTALL_BITSANDBYTES=true

FROM ${BASE_IMAGE_BUILDER} AS builder

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/workspace/src

ARG INSTALL_FLASH_ATTN=false
ARG INSTALL_BITSANDBYTES

# Copy only dependency manifests first to maximize build cache hits.
COPY docker/requirements.runtime.txt /workspace/docker/requirements.runtime.txt
COPY docker/requirements.server.txt /workspace/docker/requirements.server.txt

# Build wheels in a dedicated stage so runtime image stays slim.
# BuildKit cache mount makes repeated builds much faster.
# flash_attn: use PIP_CACHE_DIR/TMPDIR under /wheels for that step only — pip's cache mount is a different filesystem and its prebuilt-wheel path triggers EXDEV on rename.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip wheel && \
    mkdir -p /wheels && \
    python -m pip wheel -w /wheels -r /workspace/docker/requirements.runtime.txt && \
    python -m pip wheel -w /wheels -r /workspace/docker/requirements.server.txt && \
    if [ "${INSTALL_BITSANDBYTES}" = "true" ]; then python -m pip wheel -w /wheels "bitsandbytes==0.47.0"; fi && \
    if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then \
      mkdir -p /wheels/.pip-flash-cache /wheels/.tmp-flash && \
      PIP_CACHE_DIR=/wheels/.pip-flash-cache TMPDIR=/wheels/.tmp-flash \
      python -m pip wheel -w /wheels --no-build-isolation "flash_attn>=2.8.3" && \
      rm -rf /wheels/.pip-flash-cache /wheels/.tmp-flash; \
    fi


FROM ${BASE_IMAGE} AS runtime

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/workspace/src

ARG INSTALL_GIT
ARG INSTALL_BITSANDBYTES
ARG INSTALL_FLASH_ATTN=false

# System deps (minimal). git is sometimes needed by HF for some repos; keep it lightweight.
RUN if [ "${INSTALL_GIT}" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy dependency manifests first (keeps install layer cacheable).
COPY docker/requirements.runtime.txt /workspace/docker/requirements.runtime.txt
COPY docker/requirements.server.txt /workspace/docker/requirements.server.txt

COPY --from=builder /wheels /wheels

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip && \
    python -m pip install --no-index --find-links=/wheels \
      -r /workspace/docker/requirements.runtime.txt \
      -r /workspace/docker/requirements.server.txt && \
    if [ "${INSTALL_BITSANDBYTES}" = "true" ]; then \
      python -m pip install --no-index --find-links=/wheels "bitsandbytes==0.47.0"; \
    fi && \
    if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then \
      python -m pip install --no-index --find-links=/wheels "flash_attn>=2.8.3"; \
    fi && \
    rm -rf /wheels

# Copy only runtime source (avoid slow recursive copy of unrelated files).
COPY src /workspace/src
COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh

RUN sed -i 's/\r$//' /workspace/docker/entrypoint.sh && \
    chmod +x /workspace/docker/entrypoint.sh

EXPOSE 8000

ENV MODEL_PATH=/models/unipercept \
    MODEL_ID=unipercept \
    PREFER_CUDA=true \
    USE_FLASH_ATTN=true

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
