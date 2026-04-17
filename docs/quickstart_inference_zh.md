# UniPercept 小白一键推理指南（不含数据集）

本指南只覆盖一件事：**把 UniPercept 的推理环境搭起来，并成功跑出一张图的 IAA/IQA/ISTA 分数**。  
如果你只是“推理和使用”，不做论文评测，不需要下载 benchmark 数据集。

## 1. 你将得到什么

- 一个可重复执行的一键脚本：`scripts/setup_unipercept_infer.sh`
- 一个最小验证脚本：`scripts/run_smoke_test.py`
- 成功标志：终端打印 3 个分数
  - `Aesthetics (IAA)`
  - `Quality (IQA)`
  - `Structure (ISTA)`

## 2. 使用前准备

- 已有 Ubuntu 机器（建议带 NVIDIA GPU）
- 项目目录存在：`/home/ubuntu/UniPercept`
- 能访问 Hugging Face（脚本会提示你登录）

## 3. 一键安装（推荐）

在项目根目录执行：

```bash
cd /home/ubuntu/UniPercept
chmod +x scripts/setup_unipercept_infer.sh
bash scripts/setup_unipercept_infer.sh
```

脚本会自动完成以下动作：

1. 检查 `nvidia-smi` 是否可用
2. 不存在 conda 时自动安装 Miniconda
3. 创建 `conda` 环境 `unipercept`（Python 3.10）
4. 安装 `unipercept-reward` 与 `huggingface_hub[cli]`（清华源）
5. 交互式执行 `huggingface-cli login`
6. 下载模型到 `ckpt/unipercept`
7. 下载测试图片 `test.jpg`
8. 运行 `scripts/run_smoke_test.py` 输出分数

## 4. 关于显卡驱动（脚本行为）

- 如果 `nvidia-smi` 正常：继续安装推理环境
- 如果 `nvidia-smi` 不可用：脚本会询问你是否安装驱动
  - 输入 `y`：尝试下载并执行 NVIDIA 驱动安装程序（需要 sudo，可能重启）
  - 输入其他：跳过驱动安装，后续尽量继续，可能回退到 CPU

说明：脚本不会自动改 `netplan`，不会默认强制覆盖驱动。

## 5. 如何替换成你自己的图片推理

激活环境后执行：

```bash
cd /home/ubuntu/UniPercept
conda activate unipercept
python scripts/run_smoke_test.py --image /你的图片路径/xxx.jpg --model-path ckpt/unipercept
```

## 6. 手动安装方式（不走一键脚本）

```bash
cd /home/ubuntu/UniPercept

# 1) 创建环境
conda create -n unipercept python=3.10 -y
conda activate unipercept

# 2) 安装依赖（清华源）
pip install -U pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn unipercept-reward
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn "huggingface_hub[cli]"

# 3) 登录 HF（交互输入 token）
huggingface-cli login

# 4) 下载模型
huggingface-cli download Thunderbolt215215/UniPercept --local-dir ckpt/unipercept

# 5) 下载测试图片
wget -O test.jpg "https://git.leon-lv.me/server/frontend/vue-website/-/raw/master/src/assets/image/back.jpg?ref_type=heads"

# 6) 执行推理
python scripts/run_smoke_test.py --image test.jpg --model-path ckpt/unipercept
```

## 7. 常见问题排查

### Q1: `huggingface-cli login` 失败或 401

- 检查 token 是否有 `read` 权限
- 重新登录：

```bash
huggingface-cli logout
huggingface-cli login
```

### Q2: `nvidia-smi` 不可用

- 先确认机器是否有 NVIDIA GPU
- 有 GPU 但命令不可用：需要安装/修复驱动
- 驱动变更后建议重启：`sudo reboot`

### Q3: `flash` 扩展安装失败

- 不影响基本推理，可直接跳过
- 继续使用 `unipercept-reward` 常规模式即可

### Q4: 显存不足（CUDA OOM）

- 先关闭其他占显存进程：`nvidia-smi`
- 改用 CPU（脚本会自动检测并回退）
- 或更换更小图片进行测试
- 服务化部署（`docker compose --profile gpu`）：量化、视觉尺寸、token 上限等由仓库根目录 `config/runtime.yaml`（分层级 YAML）配置；修改后需重新构建或重启容器生效；可通过环境变量 `RUNTIME_CONFIG_FILE` 指向其它 YAML。

```bash
# 示例：改完后重启 GPU 服务
docker compose --profile gpu up --build
```

- 在 `runtime.yaml` 的 `weights.quant_mode` 设为 `8bit` 可开启 8bit 量化（目标：将空载显存降到 10GB 级）；`none` / `off` 为不量化（bf16）。
- 建议同时收紧 `limits` 段（例如 `max_images_per_request: 1`、`max_new_tokens: 256`）以降低峰值显存。

### Q5: 提示找不到模型目录

- 确认 `ckpt/unipercept` 是否存在
- 重新下载：

```bash
huggingface-cli download Thunderbolt215215/UniPercept --local-dir ckpt/unipercept
```

## 8. 成功标志

终端出现以下结构即成功：

```text
==== Inference Result ====
Image: ...
  Aesthetics (IAA): xx.xxxx
  Quality    (IQA): xx.xxxx
  Structure  (ISTA): xx.xxxx
```

这说明你已经具备“直接推理”能力，不需要数据集也可使用。
