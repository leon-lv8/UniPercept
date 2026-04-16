#!/usr/bin/env bash
# 宿主机 NVIDIA + CUDA 依赖 + Docker GPU（CDI）一站式配置（Ubuntu/Debian，apt）。
# - apt：wget、编译依赖等；可选下载并执行官方 .run 驱动（需交互，可能需 reboot）
# - 可选：安装 nvidia-container-toolkit、生成 /etc/cdi/nvidia.yaml、配置 Docker 运行时并重启 docker
#   （修复 Docker 29+ compose「gpus: all」报错：failed to discover GPU vendor from CDI）
# - 顺序：apt 依赖 → 安装 Docker（若缺失，get.docker.com --mirror Aliyun）→ 合并 registry-mirrors →
#   可选 .run 驱动（缓存目录见下）→ 有 GPU 时配置 CDI → docker compose 构建 hf-download 并下载模型
#
# 用法：bash scripts/setup_host_env.sh
#
# 环境变量：
#   UNIPERCEPT_SETUP_CACHE  缓存根目录（驱动 .run、get-docker.sh 等），默认 ~/.cache/unipercept
#   NVIDIA_DRIVER_URL       官方 .run 下载地址
#   HF_TOKEN                拉取需鉴权的 HF 模型时使用；未设置时在终端会提示输入（可直接回车尝试匿名）
#   HF_MODEL_ID / HF_ENDPOINT  传给 compose 的 hf-download 服务（可选）
#   FORCE_REINSTALL=1       强制重装已存在步骤（默认检测到已安装则跳过）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVIDIA_DRIVER_URL="${NVIDIA_DRIVER_URL:-https://cn.download.nvidia.com/XFree86/Linux-x86_64/595.58.03/NVIDIA-Linux-x86_64-595.58.03.run}"
UNIPERCEPT_SETUP_CACHE="${UNIPERCEPT_SETUP_CACHE:-${HOME}/.cache/unipercept}"
SETUP_DOCKER_DIR="${UNIPERCEPT_SETUP_CACHE}/setup"
SETUP_NVIDIA_DIR="${UNIPERCEPT_SETUP_CACHE}/nvidia"
FORCE_REINSTALL_FLAG="${FORCE_REINSTALL:-0}"

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

err() {
  echo "[ERROR] $*" >&2
}

usage() {
  cat <<EOF
用法: $(basename "$0") [--help]

宿主机一键配置（Docker、镜像加速、可选 NVIDIA 驱动、CDI、HF 模型下载），详见脚本头部注释。
若检测到已有环境，默认跳过对应步骤；可通过 FORCE_REINSTALL=1 强制重装。

环境变量见脚本头部注释。
EOF
}

note_sudo() {
  if [[ "${EUID:-}" -ne 0 ]] && ! sudo -n true 2>/dev/null; then
    log "需要 sudo 权限以安装软件包并（可选）重启 Docker，请输入密码。"
  fi
}

require_debian_apt() {
  if [[ ! -f /etc/debian_version ]]; then
    err "本脚本仅支持 Debian/Ubuntu（apt）。其它发行版请参考："
    err "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    exit 1
  fi
}

ensure_setup_cache_dirs() {
  mkdir -p "$SETUP_DOCKER_DIR" "$SETUP_NVIDIA_DIR"
  log "缓存目录（驱动与安装脚本不落项目根）: $UNIPERCEPT_SETUP_CACHE"
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

is_nonempty_dir() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]
}

all_apt_cuda_build_deps_installed() {
  local pkg
  for pkg in wget curl ca-certificates gnupg build-essential pkg-config python3; do
    dpkg -s "$pkg" >/dev/null 2>&1 || return 1
  done
}

docker_registry_mirrors_ready() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi

  if sudo python3 - <<'PY'
import json
import os
import sys

path = "/etc/docker/daemon.json"
required = {
    "https://docker-registry.657525741.workers.dev",
    "https://tirqxirhruyr.ap-southeast-1.clawcloudrun.com",
}
if not os.path.isfile(path):
    sys.exit(1)
with open(path, encoding="utf-8") as f:
    data = json.load(f)
mirrors = set(data.get("registry-mirrors") or [])
sys.exit(0 if required.issubset(mirrors) else 1)
PY
  then
    return 0
  fi
  return 1
}

nvidia_cdi_ready() {
  command -v nvidia-ctk >/dev/null 2>&1 && [[ -s /etc/cdi/nvidia.yaml ]]
}

prompt_force_reinstall_if_needed() {
  if is_truthy "$FORCE_REINSTALL_FLAG"; then
    log "FORCE_REINSTALL 已启用，将强制执行重装/重配步骤。"
    return 0
  fi

  local has_existing=0
  if command -v docker >/dev/null 2>&1 || driver_already_ok || nvidia_cdi_ready || is_nonempty_dir "$PROJECT_ROOT/ckpt/unipercept"; then
    has_existing=1
  fi
  if [[ "$has_existing" -eq 0 ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    log "检测到已有环境，且当前非交互终端：将按默认策略跳过已存在步骤。"
    return 0
  fi

  read -r -p "检测到已有环境，默认跳过已存在步骤。是否强制重装/重新下载模型? [y/N]: " force_reinstall
  if [[ "${force_reinstall:-N}" =~ ^[Yy]$ ]]; then
    FORCE_REINSTALL_FLAG=1
    log "已开启强制重装。"
  else
    log "保持默认：跳过已存在步骤。"
  fi
}

ensure_apt_cuda_build_deps() {
  if ! is_truthy "$FORCE_REINSTALL_FLAG" && all_apt_cuda_build_deps_installed; then
    log "检测到 apt 依赖已安装，跳过 apt 安装步骤（可用 FORCE_REINSTALL=1 强制执行）。"
    return 0
  fi
  log "通过 apt 安装 CUDA 常用系统依赖（含 curl/gnupg/python3，供 Docker 安装脚本与 daemon.json 合并）..."
  sudo apt-get update
  sudo apt-get install -y \
    wget \
    curl \
    ca-certificates \
    gnupg \
    build-essential \
    pkg-config \
    python3
}

ensure_wget() {
  if ! command -v wget >/dev/null 2>&1; then
    warn "未找到 wget，正在安装..."
    sudo apt-get update
    sudo apt-get install -y wget
  fi
}

driver_already_ok() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

show_driver_status() {
  log "检测到可用 NVIDIA 驱动："
  nvidia-smi | sed -n '1,3p'
}

require_nvidia_smi() {
  if ! driver_already_ok; then
    err "宿主机 nvidia-smi 不可用。请先安装 NVIDIA 驱动后再配置 Docker CDI。"
    exit 1
  fi
  log "宿主机 GPU 状态："
  nvidia-smi | sed -n '1,3p'
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    err "未找到 docker 命令。若本脚本已尝试安装 Docker，请检查上方日志、网络与 sudo；否则请手动安装 Docker Engine。"
    exit 1
  fi
}

docker_cmd_quote_args() {
  local quoted="" a
  for a in "$@"; do
    quoted+=" $(printf '%q' "$a")"
  done
  echo "${quoted# }"
}

# 在未重新登录时尽量可用：直接 docker → sg docker → sudo docker
docker_wrap() {
  local inner
  inner=$(docker_cmd_quote_args "$@")
  if docker info >/dev/null 2>&1; then
    eval "docker $inner"
  elif command -v sg >/dev/null 2>&1 && sg docker -c "docker info" >/dev/null 2>&1; then
    sg docker -c "docker $inner"
  else
    eval "sudo docker $inner"
  fi
}

install_docker_engine_if_missing() {
  if ! is_truthy "$FORCE_REINSTALL_FLAG" && command -v docker >/dev/null 2>&1; then
    log "已检测到 Docker Engine，跳过 get-docker 安装。"
    return 0
  fi
  if is_truthy "$FORCE_REINSTALL_FLAG" && command -v docker >/dev/null 2>&1; then
    warn "已启用强制重装，将重新执行 get-docker 安装。"
  fi

  ensure_setup_cache_dirs
  local installer="$SETUP_DOCKER_DIR/get-docker.sh"
  log "下载 Docker 安装脚本到 $installer（阿里云镜像安装）..."
  curl -fsSL https://get.docker.com -o "$installer"
  log "执行 get-docker.sh --mirror Aliyun（需 sudo）..."
  sudo sh "$installer" --mirror Aliyun

  local target_user="${SUDO_USER:-$USER}"
  if [[ -n "$target_user" && "$target_user" != "root" ]]; then
    log "将用户 $target_user 加入 docker 组（新会话生效；本脚本后续使用 sg/sudo 调用 docker）..."
    sudo usermod -aG docker "$target_user"
  fi

  log "安装完成后可重新登录或执行 newgrp docker，以便当前终端免 sudo 使用 docker。"
  rm -f "$installer" || true
}

merge_docker_daemon_registry_mirrors() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker 未安装，跳过 registry-mirrors 配置。"
    return 0
  fi
  if ! is_truthy "$FORCE_REINSTALL_FLAG" && docker_registry_mirrors_ready; then
    log "检测到 registry-mirrors 已配置，跳过写入 daemon.json。"
    return 0
  fi

  log "合并 /etc/docker/daemon.json 中的 registry-mirrors（保留已有其它配置）..."
  sudo python3 <<'PY'
import json
import os
import shutil
import time

path = "/etc/docker/daemon.json"
mirrors = [
    "https://docker-registry.657525741.workers.dev",
    "https://tirqxirhruyr.ap-southeast-1.clawcloudrun.com",
]
data = {}
if os.path.isfile(path):
    backup = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
existing = data.get("registry-mirrors") or []
merged = []
for m in list(existing) + mirrors:
    if m not in merged:
        merged.append(m)
data["registry-mirrors"] = merged
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = f"{path}.tmp.{os.getpid()}"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
os.replace(tmp, path)
print("registry-mirrors:", merged)
PY

  if systemctl is-active --quiet docker 2>/dev/null; then
    log "重启 Docker 以应用 daemon.json…"
    sudo systemctl restart docker
  else
    warn "docker 服务未处于 active；请手动启动或重启 Docker 后再试。"
  fi
}

install_nvidia_container_toolkit_repo() {
  local keyring="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  local list="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
  log "配置 NVIDIA Container Toolkit apt 源并安装/升级软件包…"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o "$keyring"
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    | sudo tee "$list" >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
}

configure_cdi_and_docker_runtime() {
  sudo mkdir -p /etc/cdi
  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "生成 CDI 规范：/etc/cdi/nvidia.yaml"
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    log "将 NVIDIA 运行时写入 Docker 配置"
    sudo nvidia-ctk runtime configure --runtime=docker
  else
    err "安装后仍未找到 nvidia-ctk，请检查 nvidia-container-toolkit 是否安装成功。"
    exit 1
  fi

  log "重启 Docker 守护进程…"
  if systemctl is-active --quiet docker 2>/dev/null; then
    sudo systemctl restart docker
  else
    warn "docker 服务未处于 active 状态；若你使用 rootless 或其它方式启动 Docker，请自行重启 Docker。"
  fi
}

smoke_test_gpu_container() {
  local img="nvidia/cuda:12.0.0-base-ubuntu22.04"
  log "拉取测试镜像并运行 nvidia-smi（可能需要几分钟）…"
  if docker_wrap run --rm --gpus all "$img" nvidia-smi >/dev/null 2>&1; then
    log "容器内 nvidia-smi 成功，CDI/GPU 配置正常。"
  else
    warn "docker run --gpus all 测试未通过，请查看上方 docker 输出排查。"
  fi
}

configure_docker_nvidia_cdi() {
  require_nvidia_smi
  require_docker

  log "开始安装 NVIDIA Container Toolkit 并配置 Docker CDI…"
  install_nvidia_container_toolkit_repo
  configure_cdi_and_docker_runtime

  log "配置完成。可执行：docker compose --profile gpu up"
  smoke_test_gpu_container
}

# 无 nvidia-smi 时不退出整个流程，仅跳过 CDI
configure_docker_nvidia_cdi_if_gpu() {
  if ! command -v docker >/dev/null 2>&1; then
    warn "Docker 不可用，跳过 NVIDIA CDI 配置。"
    return 0
  fi
  if ! driver_already_ok; then
    warn "nvidia-smi 不可用，跳过 Container Toolkit / CDI；安装驱动并重启后可再次运行本脚本。"
    return 0
  fi
  if ! is_truthy "$FORCE_REINSTALL_FLAG" && nvidia_cdi_ready; then
    log "检测到 nvidia-ctk 与 /etc/cdi/nvidia.yaml 已就绪，跳过 CDI 配置。"
    return 0
  fi
  configure_docker_nvidia_cdi
}

# 未设置 HF_TOKEN 时在终端询问一次（不回显）；已设置或非 TTY 则跳过
prompt_hf_token_if_needed() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "已检测到环境变量 HF_TOKEN，跳过输入。"
    return 0
  fi
  if [[ ! -t 0 ]]; then
    warn "标准输入非终端且未设置 HF_TOKEN，将尝试匿名下载（公开仓库）。"
    return 0
  fi
  log "若模型仓库需要鉴权，请输入 Hugging Face 的 HF_TOKEN；仅公开仓库可直接回车。"
  local _token
  read -r -s -p "HF_TOKEN: " _token
  echo
  export HF_TOKEN="${_token}"
}

hf_download_via_compose() {
  require_docker
  if ! is_truthy "$FORCE_REINSTALL_FLAG" && is_nonempty_dir "$PROJECT_ROOT/ckpt/unipercept"; then
    log "检测到 ./ckpt/unipercept 已有模型文件，跳过 hf-download（可用 FORCE_REINSTALL=1 强制重新下载）。"
    return 0
  fi

  prompt_hf_token_if_needed
  if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "未设置 HF_TOKEN；仅当模型仓库允许匿名下载时可成功。"
  fi
  log "在仓库根目录构建 hf-download 并下载模型到 ./ckpt/unipercept …"
  (
    cd "$PROJECT_ROOT"
    docker_wrap compose build hf-download
    docker_wrap compose --profile download run --rm hf-download
  )
  log "hf-download 流程结束。"
}

optional_install_proprietary_driver() {
  if driver_already_ok; then
    return 0
  fi

  warn "未检测到可用 nvidia-smi，GPU 推理可能不可用。"
  read -r -p "是否尝试安装 NVIDIA 官方 .run 驱动（需要 sudo，可能需重启）? [y/N]: " install_driver
  if [[ ! "${install_driver:-N}" =~ ^[Yy]$ ]]; then
    warn "已跳过 .run 驱动安装。"
    return 1
  fi

  ensure_setup_cache_dirs
  local driver_file="$SETUP_NVIDIA_DIR/$(basename "$NVIDIA_DRIVER_URL")"
  log "下载 NVIDIA 驱动安装包到 $driver_file ..."
  if [[ -f "$driver_file" ]]; then
    log "已存在同名文件，尝试断点续传/覆盖更新..."
  fi
  wget -c -O "$driver_file" "$NVIDIA_DRIVER_URL"
  chmod +x "$driver_file"
  warn "即将执行驱动安装程序，请按屏幕提示操作。"
  sudo "$driver_file"
  log "驱动安装流程结束，建议执行 reboot 后再次运行本脚本或继续环境配置。"
  # 10：告知调用方应中止后续步骤，待重启后再跑
  exit 10
}

main() {
  for _arg in "$@"; do
    if [[ "$_arg" == "-h" || "$_arg" == "--help" ]]; then
      usage
      exit 0
    fi
  done

  note_sudo
  require_debian_apt
  prompt_force_reinstall_if_needed
  ensure_wget
  ensure_apt_cuda_build_deps
  ensure_setup_cache_dirs

  install_docker_engine_if_missing
  merge_docker_daemon_registry_mirrors

  if driver_already_ok; then
    show_driver_status
  else
    if optional_install_proprietary_driver; then
      :
    else
      warn "未安装专有驱动；将继续 Docker 与模型下载（无 GPU 时跳过 CDI）。"
    fi
  fi

  configure_docker_nvidia_cdi_if_gpu

  hf_download_via_compose

  log "宿主机配置流程结束。若刚安装驱动并需重启，请 reboot 后再次执行本脚本以完成 CDI 与 GPU 验证。"
}

main "$@"
