# Hugging Face 模型下载专用镜像：带 Python 的官方 slim 镜像 + install.sh 安装 hf CLI
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
    && curl -LsSf https://hf.co/cli/install.sh | bash -s -- --no-modify-path

COPY docker/hf-download/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
