# UniPercept：迈向统一的感知层图像理解（美学、质量、结构与纹理）

[![arXiv](https://img.shields.io/badge/arXiv-UniPercept-red?logo=arxiv)](https://arxiv.org/abs/2512.21675)
[![Website](https://img.shields.io/badge/🌎_Website-UniPercept.github.io-blue)](https://thunderbolt215.github.io/Unipercept-project/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20_Model-UniPercept-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/Thunderbolt215215/UniPercept)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-UniPercept--Bench-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/Thunderbolt215215/UniPercept-Bench)
[![PyPI](https://img.shields.io/badge/PyPI-unipercept--reward-3775A9?logo=pypi&logoColor=white)](https://pypi.org/project/unipercept-reward/)

**作者**：Shuo Cao*，Jiayang Li*，Xiaohui Li，Yuandong Pu，Kaiwen Zhu，Yuanting Gao，Siqi Luo，Yi Xin，Qi Qin，Yu Zhou，Xiangyu Chen，Wenlong Zhang，Bin Fu，Yu Qiao，Yihao Liu†  
**机构**：中国科学技术大学、上海人工智能实验室、北京大学  
\* 共同一作，† 通讯作者

如果你觉得这个项目有帮助，欢迎点一个 Star ⭐️！

![Teaser](asserts/img/teaser_v4.jpg)

⭐️ 相关研究：  
- [ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding](https://github.com/thunderbolt215/ArtiMuse)

## 新闻与更新

- [2026-01-04] 📦 **Python 包发布**：现已支持独立包 `unipercept-reward`。你可以通过 `pip install unipercept-reward` 轻松集成感知评分能力。详见下方 **快速开始**。
- [2025-12-29] 🔥 **官方发布**
  - **[技术报告](https://arxiv.org/abs/2512.21675)**
  - **[项目主页](https://thunderbolt215.github.io/Unipercept-project/)**
  - **[UniPercept-Bench](https://huggingface.co/datasets/Thunderbolt215215/UniPercept-Bench)**：一个全面的感知层理解基准，覆盖图像美学评估（IAA）、图像质量评估（IQA）、图像结构与纹理评估（ISTA），并同时支持视觉打分（VR）与视觉问答（VQA）任务。
  - **[UniPercept](https://huggingface.co/Thunderbolt215215/UniPercept)**：一个面向感知层图像理解的强基线 MLLM，通过**领域自适应预训练**与**任务对齐强化学习**优化而成。

## 🚀 快速开始

### 安装

通过 pip 安装：

```bash
pip install unipercept-reward
```

**推荐**：为了启用 **Flash Attention** 以获得更快推理速度和更低显存占用，可安装 `flash` 扩展：

```bash
pip install "unipercept-reward[flash]"
```

### 基础用法

简单推理示例：

```python
from unipercept_reward import UniPerceptRewardInferencer
inferencer = UniPerceptRewardInferencer(device="cuda")

image_paths = [
    "test.png"
]

rewards = inferencer.reward(image_paths=image_paths)

for path, score in zip(image_paths, rewards):
    if score:
        print(f"Image: {path}")
        print(f"  ➤ Aesthetics (IAA): {score['iaa']:.4f}")
        print(f"  ➤ Quality (IQA):    {score['iqa']:.4f}")
        print(f"  ➤ Structure (ISTA): {score['ista']:.4f}")
```

也可以从**本地 checkpoint** 加载模型：

```python
inferencer = UniPerceptRewardInferencer(
    model_path="/path/to/local/checkpoint",
    device="cuda"
)
```

### 输出指标

`.reward()` 方法会为每张图像返回一个字典，包含 3 个感知指标。所有分数范围为 0~100，数值越高表示表现/质量越好。

| Key | 指标名称 | 说明 |
| --- | --- | --- |
| **`iaa`** | 图像美学评估（Image Aesthetics Assessment） | 评估图像的审美质量。 |
| **`iqa`** | 图像质量评估（Image Quality Assessment） | 评估图像质量。 |
| **`ista`** | 图像结构与纹理评估（Image Structure & Texture Assessment） | 评估图像结构与纹理细节的丰富度。 |

## 🌟 摘要

多模态大语言模型（MLLM）在视觉定位、分割、描述等任务中已取得显著进展，但对**感知层**图像特征的理解能力仍有限。本文提出 **UniPercept-Bench**，构建了一个统一框架，用于跨三大核心维度——**美学（Aesthetics）**、**质量（Quality）**、**结构与纹理（Structure and Texture）**——的感知层图像理解。我们建立了分层定义体系并构建了大规模数据集以评估感知层能力。在此基础上，我们提出强基线 **UniPercept**，通过领域自适应预训练与任务对齐强化学习进行训练，实现了在**视觉打分（VR）**与**视觉问答（VQA）**任务上的稳健泛化。UniPercept 在感知层图像理解上优于现有 MLLM，并可作为文本到图像生成的**即插即用奖励模型**。本工作界定了 MLLM 时代的感知层图像理解问题，并通过全面基准与强基线为后续研究提供了坚实基础。

## 📊 UniPercept-Bench

我们提出了系统化的感知图像理解基准 **UniPercept-Bench**：

- **覆盖全面**：涵盖 **3 个领域**（IAA、IQA、ISTA）、**17 个类别**、**43 条准则**。
- **任务完整**：同时支持 **视觉打分（VR）** 与 **视觉问答（VQA）**。

**下载地址**：🤗 [UniPercept-Bench](https://huggingface.co/datasets/Thunderbolt215215/UniPercept-Bench)

![UniPercept-Bench](asserts/img/unipercetp-bench.png)

## 🔍 UniPercept

**UniPercept** 是一个强基线 MLLM，通过领域自适应预训练和任务对齐强化学习训练，可同时处理：
- **视觉打分（VR）**：连续分值预测
- **视觉问答（VQA）**：视觉属性推理

### 🛠️ 环境配置

```bash
conda create -n unipercept python=3.10
conda activate unipercept
cd UniPercept
pip install -r requirements.txt
```

### 📉 评测

请先从 [🤗 UniPercept](https://huggingface.co/Thunderbolt215215/UniPercept) 下载权重，并放入 `ckpt/` 目录。

**视觉打分（VR）**

请下载下列数据集，并放置到对应路径：

| 数据集 | 领域 | 下载 | 路径 |
| :--- | :---: | :---: | :--- |
| **ArtiMuse-10K** | IAA | 🤗 [Link](https://huggingface.co/datasets/Thunderbolt215215/ArtiMuse-10K) | `benchmark/VR/IAA/ArtiMuse-10K/image` |
| **AVA** | IAA | [Link](https://github.com/imfing/ava_downloader) | `benchmark/VR/IAA/AVA/image` |
| **TAD66K** | IAA | [Link](https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment) | `benchmark/VR/IAA/TAD66K/image` |
| **FLICKR-AES** | IAA | [Link](https://github.com/alanspike/personalizedImageAesthetics) | `benchmark/VR/IAA/FLICKR-AES/image` |
| **KonIQ-10K** | IQA | [Link](https://database.mmsp-kn.de/koniq-10k-database.html) | `benchmark/VR/IQA/KonIQ-10K/image` |
| **SPAQ** | IQA | [Link](https://github.com/h4nwei/SPAQ) | `benchmark/VR/IQA/SPAQ/image` |
| **KADID** | IQA | [Link](https://database.mmsp-kn.de/kadid-10k-database.html) | `benchmark/VR/IQA/KADID/image` |
| **PIPAL** | IQA | [Link](https://github.com/HaomingCai/PIPAL-dataset) | `benchmark/VR/IQA/PIPAL/image` |
| **ISTA-10K** | ISTA | 🤗 [Link](https://huggingface.co/datasets/Thunderbolt215215/UniPercept-Bench) | `benchmark/VR/ISTA/ISTA-10K/image` |

完成数据准备后，可在 `src/eval/eval_vr.sh` 中配置目标数据集和设备。结果将保存到 `results/vr`。

```bash
cd UniPercept
bash src/eval/eval_vr.sh
```

**视觉问答（VQA）**

请从 [🤗 UniPercept-Bench](https://huggingface.co/datasets/Thunderbolt215215/UniPercept-Bench) 下载 **UniPercept-Bench-VQA**，并放入 `benchmark/VQA`。  
然后在 `src/eval/eval_vqa.sh` 中配置目标领域。评测结果将保存到 `results/vqa`。

```bash
cd UniPercept
bash src/eval/eval_vqa.sh
```

**交互式图像感知**

你可以与 UniPercept 围绕图像的美学、质量、结构细节等方面进行多轮对话。可参考下列命令运行示例，也可按需自定义，或参考 [InternVL](https://github.com/OpenGVLab/InternVL) 获取更多实现细节。

```bash
cd UniPercept
bash src/eval/conversation.sh
```

我们提供了额外的 Prompt 示例：

<details>
  <summary>Conversation Prompts</summary>

| Key | Prompt 内容 |
| :--- | :--- |
| **IAA-Comprehensive** | `Analyze the aesthetics of this image step by step, providing a comprehensive description without assigning any score.` |
| **IAA-Composition & Design** | `Please evaluate the aesthetic quality of this image from the attribute of Composition & Design.` |
| **IAA-Visual Elements & Structure** | `Please evaluate the aesthetic quality of this image from the attribute of Visual Elements & Structure.` |
| **IAA-Technical Execution** | `Please evaluate the aesthetic quality of this image from the attribute of Technical Execution.` |
| **IAA-Originality & Creativity** | `Please evaluate the aesthetic quality of this image from the attribute of Originality & Creativity.` |
| **IAA-Theme & Communication** | `Please evaluate the aesthetic quality of this image from the attribute of Theme & Communication.` |
| **IAA-Emotion & Viewer Response** | `Please evaluate the aesthetic quality of this image from the attribute of Emotion & Viewer Response.` |
| **IAA-Overall Gestalt** | `Please evaluate the aesthetic quality of this image from the attribute of Overall Gestalt.` |
| **IQA-Comprehensive** | `Evaluate the quality of this image step by step, offering a detailed descriptive analysis rather than a numerical score.` |
| **IQA-Distortion Location** | `Analyze the Distortion Location of this image.` |
| **IQA-Distortion Severity** | `Analyze the Distortion Severity of this image.` |
| **IQA-Distortion Type** | `Analyze the Distortion Type of this image.` |
| **ISTA-Structural Analysis** | `Perform a detailed hierarchical analysis of the image’s texture and structure. For complex scenes, break them down into distinct components and provide the results as structured JSON only, without any explanations.` |

</details>

### 🏆 性能表现

UniPercept 在三大感知领域（IAA、IQA、ISTA）和两类任务（VR、VQA）上，持续优于闭源模型（如 GPT-4o、Gemini-2.5-Pro）及主流开源模型（如 InternVL3、Qwen3-VL）。

<details open>
  <summary>UniPercept-Bench-VR 结果</summary>
  <img src="asserts/img/vr.png" alt="Performance on UniPercept-Bench-VR" width="1000">
</details>

<details>
  <summary>UniPercept-Bench-VQA（IAA）结果</summary>
  <img src="asserts/img/vqa-iaa.png" alt="Performance on UniPercept-Bench-VQA (IAA)" width="1000">
</details>

<details>
  <summary>UniPercept-Bench-VQA（IQA）结果</summary>
  <img src="asserts/img/vqa-iqa.png" alt="Performance on UniPercept-Bench-VQA (IQA)" width="1000">
</details>

<details>
  <summary>UniPercept-Bench-VQA（ISTA）结果</summary>
  <img src="asserts/img/vqa-ista.png" alt="Performance on UniPercept-Bench-VQA (ISTA)" width="1000">
</details>

### 🎨 应用场景

**作为奖励模型（UniPercept As Reward）**

UniPercept 可作为强大的奖励模型用于文生图（T2I）模型后训练。将其奖励信号接入 **FLUX.1-dev** 的训练后，可显著提升图像美学质量、结构丰富性与提示词一致性。

![Reward](asserts/img/reward.png)

**作为评测指标（UniPercept As Metrics）**

UniPercept 可作为感知层指标体系，评估任何图像生成模型输出，覆盖 IAA、IQA、ISTA 三个互补维度。

![Metrics-1](asserts/img/metrics_dpg.png)
![Metrics-2](asserts/img/metric_geneval.png)

### 🖼️ UniPercept 构建的图像档案（Image Profiles）

UniPercept 可进行全面的感知层图像分析，在 IAA、IQA、ISTA 维度上给出准确评分，并输出细粒度、多维度分析结果，最终形成详细图像档案。

![Profile-1](asserts/img/profile1.png)
![Profile-2](asserts/img/profile2.png)
![Profile-3](asserts/img/profile3.png)

## 💐 致谢

本项目基于以下开源工作：

- **[InternVL](https://github.com/OpenGVLab/InternVL)**：我们使用其强大的多模态模型作为基座模型。
- **[VLM-R1](https://github.com/om-ai-lab/VLM-R1)**：我们参考并改造其代码框架以实现任务对齐强化学习。
- **[ArtiMuse](https://github.com/thunderbolt215/ArtiMuse)**：我们采用其 "Token As Score" 策略用于视觉打分。

## ✏️ 引用

如果 UniPercept 对你的研究有帮助，欢迎引用：

```bibtex
@misc{cao2025uniperceptunifiedperceptuallevelimage,
      title={UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture},
      author={Shuo Cao and Jiayang Li and Xiaohui Li and Yuandong Pu and Kaiwen Zhu and Yuanting Gao and Siqi Luo and Yi Xin and Qi Qin and Yu Zhou and Xiangyu Chen and Wenlong Zhang and Bin Fu and Yu Qiao and Yihao Liu},
      year={2025},
      eprint={2512.21675},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.21675},
}

@misc{cao2025artimusefinegrainedimageaesthetics,
      title={ArtiMuse: Fine-Grained Image Aesthetics Assessment with Joint Scoring and Expert-Level Understanding},
      author={Shuo Cao and Nan Ma and Jiayang Li and Xiaohui Li and Lihao Shao and Kaiwen Zhu and Yu Zhou and Yuandong Pu and Jiarui Wu and Jiaquan Wang and Bo Qu and Wenhai Wang and Yu Qiao and Dajuin Yao and Yihao Liu},
      year={2025},
      eprint={2507.14533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.14533},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=thunderbolt215/UniPercept&type=Date)](https://star-history.com/#thunderbolt215/UniPercept&Date)
