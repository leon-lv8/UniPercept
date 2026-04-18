# Hugging Face 模型下载专用镜像：官方 CLI 安装脚本（https://hf.co/cli/install.sh）
# 使用 --no-modify-path 时不改 shell 配置，须把 hf 所在目录加入 PATH（默认 ~/.local/bin）
# 见 https://huggingface.co/docs/huggingface_hub/installation
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://hf.co/cli/install.sh | bash -s -- --no-modify-path \
    && hf version

COPY docker/hf-download/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
