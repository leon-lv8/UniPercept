#!/bin/bash
# 在 Ubuntu/Debian 上安装 NVIDIA Container Toolkit，并配置 Docker 使用 GPU。
# 默认走国内阿里云镜像；若需官方源：USE_CN_MIRROR=0 sudo ./install-nvidia-docker.sh
set -euo pipefail

OFFICIAL_BASE="https://nvidia.github.io/libnvidia-container"
ALIYUN_BASE="https://mirrors.aliyun.com/libnvidia-container"
USE_CN_MIRROR="${USE_CN_MIRROR:-1}"

if [[ "$USE_CN_MIRROR" == "1" ]]; then
  REPO_BASE="$ALIYUN_BASE"
else
  REPO_BASE="$OFFICIAL_BASE"
fi

. /etc/os-release 2>/dev/null || true
CODENAME="${VERSION_CODENAME:-unknown}"

echo "========================================="
echo "NVIDIA Container Toolkit 一键安装脚本"
echo "发行版: ${PRETTY_NAME:-$(uname -s)} | 代号: ${CODENAME}"
echo "软件源: ${REPO_BASE}"
echo "========================================="

for cmd in curl gpg docker; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "错误: 未找到命令「${cmd}」。请先安装 Docker Engine 与 curl、gnupg，然后再运行本脚本。"
    exit 1
  fi
done

echo "[1/7] 清理可能冲突的旧 NVIDIA 容器仓库配置..."
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo rm -f /etc/apt/keyrings/nvidia-container-toolkit.gpg

echo "[2/7] 添加 GPG 密钥与 APT 源（与官方文档一致）..."
curl -fsSL "${REPO_BASE}/gpgkey" | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 官方 list 使用 $(ARCH)；若使用国内镜像，需把包索引域名替换为 mirrors.aliyun.com
curl -sL "${REPO_BASE}/stable/deb/nvidia-container-toolkit.list" |
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |
  if [[ "$USE_CN_MIRROR" == "1" ]]; then
    sed 's#https://nvidia.github.io/libnvidia-container#https://mirrors.aliyun.com/libnvidia-container#g'
  else
    cat
  fi |
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

echo "[3/7] 更新软件包索引..."
sudo apt-get update -qq

echo "[4/7] 安装 nvidia-container-toolkit..."
sudo apt-get install -y nvidia-container-toolkit

echo "[5/7] 配置 Docker 使用 NVIDIA 运行时..."
sudo nvidia-ctk runtime configure --runtime=docker
if sudo systemctl is-enabled docker >/dev/null 2>&1 || sudo systemctl is-active docker >/dev/null 2>&1; then
  sudo systemctl restart docker
else
  echo "警告: 未检测到 systemd 下的 docker 服务，请手动重启 Docker 使配置生效。"
fi

echo "[6/7] 生成 CDI 规范（可选，供 --device nvidia.com/gpu= 等用法）..."
sudo mkdir -p /etc/cdi
if ! sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml; then
  echo "警告: CDI 生成失败。若尚未安装 NVIDIA 驱动或无 GPU，可忽略；GPU 容器仍可用 --gpus all。"
fi

echo "[7/7] 拉取测试镜像并验证 GPU（首次会下载镜像，请稍候）..."
CUDA_TEST_IMAGE="${CUDA_TEST_IMAGE:-nvidia/cuda:12.0.0-base-ubuntu22.04}"
set +e
docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi -L
VERIFY_RC=$?
set -e

echo "========================================="
if [[ "$VERIFY_RC" -eq 0 ]]; then
  echo "安装与验证成功：Docker 已可访问 GPU。"
else
  echo "验证未通过（退出码 ${VERIFY_RC}）。请确认："
  echo "  1) 已安装并可用的 NVIDIA 驱动（宿主机 nvidia-smi 正常）；"
  echo "  2) Docker 已重启且 daemon.json 中已配置 nvidia 运行时。"
  exit 1
fi
