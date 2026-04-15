#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_HOME_DEFAULT="$HOME/miniconda3"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-unipercept}"
MODEL_REPO="Thunderbolt215215/UniPercept"
MODEL_DIR="$PROJECT_ROOT/ckpt/unipercept"
SMOKE_TEST_IMAGE_URL='https://git.leon-lv.me/server/frontend/vue-website/-/raw/master/src/assets/image/back.jpg?ref_type=heads'
SMOKE_TEST_IMAGE_PATH="$PROJECT_ROOT/test.jpg"
SMOKE_TEST_SCRIPT="$PROJECT_ROOT/scripts/run_smoke_test.py"
PYPI_MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_MIRROR_HOST="pypi.tuna.tsinghua.edu.cn"
NVIDIA_DRIVER_URL="https://cn.download.nvidia.com/XFree86/Linux-x86_64/595.58.03/NVIDIA-Linux-x86_64-595.58.03.run"

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

err() {
  echo "[ERROR] $*" >&2
}

check_repo_root() {
  if [[ ! -f "$PROJECT_ROOT/requirements.txt" || ! -d "$PROJECT_ROOT/src" ]]; then
    err "当前目录不是 UniPercept 项目根目录: $PROJECT_ROOT"
    exit 1
  fi
}

ensure_system_tools() {
  if ! command -v wget >/dev/null 2>&1; then
    warn "未找到 wget，正在安装..."
    sudo apt update
    sudo apt install -y wget
  fi
}

ensure_nvidia_or_optional_install() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "检测到可用 NVIDIA 驱动："
    nvidia-smi | sed -n '1,3p'
    return
  fi

  warn "未检测到可用 nvidia-smi，GPU 推理可能不可用。"
  read -r -p "是否尝试安装 NVIDIA 驱动（需要 sudo，可能重启）? [y/N]: " install_driver
  if [[ "${install_driver:-N}" =~ ^[Yy]$ ]]; then
    local driver_file
    driver_file="$PROJECT_ROOT/$(basename "$NVIDIA_DRIVER_URL")"
    log "下载 NVIDIA 驱动安装包..."
    wget -O "$driver_file" "$NVIDIA_DRIVER_URL"
    chmod +x "$driver_file"
    warn "即将执行驱动安装程序，请按屏幕提示操作。"
    sudo "$driver_file"
    log "驱动安装流程结束，建议执行 reboot 后重新运行本脚本。"
    exit 0
  else
    warn "跳过驱动安装。后续将尝试继续，若 CUDA 不可用将自动回退到 CPU。"
  fi
}

install_miniconda_if_needed() {
  if command -v conda >/dev/null 2>&1; then
    log "已检测到 conda。"
    return
  fi

  local installer="$PROJECT_ROOT/Miniconda3-latest-Linux-x86_64.sh"
  log "未检测到 conda，开始安装 Miniconda 到 $CONDA_HOME_DEFAULT"
  wget -O "$installer" "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  bash "$installer" -b -p "$CONDA_HOME_DEFAULT"
}

init_conda_shell() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    return
  fi

  if [[ -f "$CONDA_HOME_DEFAULT/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_HOME_DEFAULT/etc/profile.d/conda.sh"
  else
    err "找不到 conda 初始化脚本，请检查 Miniconda 安装是否成功。"
    exit 1
  fi
}

ensure_conda_env() {
  if conda env list | awk '{print $1}' | rg -x "$CONDA_ENV_NAME" >/dev/null 2>&1; then
    log "Conda 环境已存在: $CONDA_ENV_NAME"
  else
    log "创建 Conda 环境: $CONDA_ENV_NAME (Python 3.10)"
    conda create -y -n "$CONDA_ENV_NAME" python=3.10
  fi
  conda activate "$CONDA_ENV_NAME"
}

install_python_packages() {
  log "安装/更新 Python 依赖..."
  python -m pip install -U pip
  python -m pip install -i "$PYPI_MIRROR_URL" --trusted-host "$PYPI_MIRROR_HOST" unipercept-reward
  python -m pip install -i "$PYPI_MIRROR_URL" --trusted-host "$PYPI_MIRROR_HOST" "huggingface_hub[cli]"

  read -r -p "是否安装 Flash Attention 扩展 unipercept-reward[flash]（可选）? [y/N]: " install_flash
  if [[ "${install_flash:-N}" =~ ^[Yy]$ ]]; then
    python -m pip install -i "$PYPI_MIRROR_URL" --trusted-host "$PYPI_MIRROR_HOST" "unipercept-reward[flash]" || \
      warn "flash 扩展安装失败，继续使用普通推理。"
  fi
}

hf_login_interactive() {
  log "准备进行 Hugging Face 登录（交互输入 token，不写入脚本）。"
  echo "提示：如没有 token，请访问 https://huggingface.co/settings/tokens 创建 read 权限 token。"
  huggingface-cli login
}

download_model_if_needed() {
  mkdir -p "$MODEL_DIR"
  if [[ -f "$MODEL_DIR/config.json" ]]; then
    log "检测到本地模型已存在，跳过下载: $MODEL_DIR"
    return
  fi

  log "下载模型到: $MODEL_DIR"
  huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
}

download_smoke_image() {
  if [[ -f "$SMOKE_TEST_IMAGE_PATH" ]]; then
    log "测试图片已存在，跳过下载: $SMOKE_TEST_IMAGE_PATH"
    return
  fi
  log "下载测试图片到: $SMOKE_TEST_IMAGE_PATH"
  wget -O "$SMOKE_TEST_IMAGE_PATH" "$SMOKE_TEST_IMAGE_URL"
}

run_smoke_test() {
  if [[ ! -f "$SMOKE_TEST_SCRIPT" ]]; then
    err "未找到验证脚本: $SMOKE_TEST_SCRIPT"
    err "请确认 scripts/run_smoke_test.py 已存在。"
    exit 1
  fi

  log "开始运行推理验证..."
  python "$SMOKE_TEST_SCRIPT" \
    --image "$SMOKE_TEST_IMAGE_PATH" \
    --model-path "$MODEL_DIR"
}

recheck_nvidia_before_python() {
  log "执行 Python 推理前再次检查 NVIDIA 驱动状态..."
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "二次检查通过，nvidia-smi 可用。"
  else
    warn "二次检查未通过：nvidia-smi 不可用。"
    warn "将继续执行推理，运行时会由 Python 脚本自动选择 CPU。"
  fi
}

main() {
  check_repo_root
  ensure_system_tools
  ensure_nvidia_or_optional_install
  install_miniconda_if_needed
  init_conda_shell
  ensure_conda_env
  install_python_packages
  hf_login_interactive
  download_model_if_needed
  download_smoke_image
  recheck_nvidia_before_python
  run_smoke_test
  log "全部完成。你现在可以替换图片路径执行 scripts/run_smoke_test.py 做自己的推理。"
}

main "$@"
