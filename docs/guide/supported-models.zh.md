# 支持的模型列表

[← 返回目录](README.md) | [English Version](supported-models.md)

本文档列出 kDiT 支持的所有模型，包括扩散模型、文本编码器和 VAE 模型。

---

## 目录

- [支持的模型列表](#支持的模型列表)
  - [目录](#目录)
  - [概览](#概览)
  - [扩散模型](#扩散模型)
    - [Wan2.2-T2V-14B](#wan22-t2v-14b)
    - [Wan2.2-I2V-14B](#wan22-i2v-14b)
    - [Wan2.2-TI2V-5B](#wan22-ti2v-5b)
    - [Wan2.1-VACE-14B](#wan21-vace-14b)
    - [Qwen-Image-T2I-20B](#qwen-image-t2i-20b)
    - [Qwen-Image-Edit-20B](#qwen-image-edit-20b)
  - [文本编码器](#文本编码器)
    - [T5 文本编码器 (UMT5-XXL)](#t5-文本编码器-umt5-xxl)
    - [Qwen2VL 文本编码器](#qwen2vl-文本编码器)
    - [Qwen2VL 多模态文本编码器](#qwen2vl-多模态文本编码器)
  - [VAE 模型](#vae-模型)
    - [Wan2.1 VAE](#wan21-vae)
    - [Qwen Image VAE](#qwen-image-vae)
  - [模型目录结构](#模型目录结构)
    - [Wan2.2 T2V / I2V（双模型）](#wan22-t2v--i2v双模型)
    - [Wan2.1 VACE（单模型）](#wan21-vace单模型)
    - [Qwen-Image T2I](#qwen-image-t2i)
    - [Qwen-Image Edit](#qwen-image-edit)
  - [模型下载链接](#模型下载链接)
    - [Wan 视频模型](#wan-视频模型)
    - [Qwen 图像模型](#qwen-图像模型)
    - [LoRA 模型](#lora-模型)
  - [硬件要求](#硬件要求)
  - [平台兼容性](#平台兼容性)

---

## 概览

kDiT 支持两大系列生成模型：

| 系列 | 任务 | 模型 | 参数量 |
|------|------|------|--------|
| **Wan 视频** | 视频生成 | Wan2.2-T2V、Wan2.2-I2V、Wan2.2-TI2V、Wan2.1-VACE | 5B–14B |
| **Qwen 图像** | 图像生成 | Qwen-Image-T2I、Qwen-Image-Edit | 20B |

所有模型均使用 **扩散 Transformer（DiT）** 架构，支持：
- FP16 / BF16 推理
- FP8 量化推理（通过 `fp8_e4m3` 线性后端）
- 多卡并行推理（torchrun / Ray）
- LoRA 加速（加载时静态合并）
- 智能缓存策略加速

---

## 扩散模型

### Wan2.2-T2V-14B

| 属性 | 值 |
|------|-----|
| **Pipeline Key** | `Wan2_2_T2V_14B` |
| **Model Key** | [`ModelKey.Wan2_2_T2V_14B`](../kdit/models/model_key.py:71) |
| **任务** | 文生视频 |
| **参数量** | ~14B |
| **架构** | 双模型（高噪声 + 低噪声） |
| **文本编码器** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE |
| **输出** | 视频（多帧） |

**描述：** 从文本提示生成视频。使用双模型架构，高噪声模型处理初始去噪阶段，低噪声模型精炼最终细节。模型边界（默认 0.875）控制切换点。

**支持特性：**
- ✅ FP8 量化
- ✅ torch.compile
- ✅ LoRA（如 Lightning LoRA 4 步生成）
- ✅ 所有缓存策略（TeaCache、EasyCache、MagCache、DCache、DBCache、HybridCache）
- ✅ 所有注意力后端
- ✅ SLG、FETA、实验性参数
- ✅ 多卡推理（torchrun / Ray）

**典型配置：**
```python
from kdit.config import SampleConfig, RuntimeConfig
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=5.0)
runtime_config = RuntimeConfig(size=(720, 480), frame_num=81)
```

---

### Wan2.2-I2V-14B

| 属性 | 值 |
|------|-----|
| **Pipeline Key** | `Wan2_2_I2V_14B` |
| **Model Key** | [`ModelKey.Wan2_2_I2V_14B`](../kdit/models/model_key.py:72) |
| **任务** | 图生视频 |
| **参数量** | ~14B |
| **架构** | 双模型（高噪声 + 低噪声） |
| **文本编码器** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE（编码 + 解码） |
| **输入** | 文本提示 + 参考图像 |
| **输出** | 视频（多帧） |

**描述：** 从文本提示和一张或多张参考图像生成视频。输入图像通过 VAE 编码作为起始帧。支持仅起始图像，或起始 + 结束图像的受控动画。

**支持特性：**
- ✅ FP8 量化
- ✅ torch.compile
- ✅ LoRA（如 Lightning LoRA、Turbo Diffusion LoRA）
- ✅ 所有缓存策略
- ✅ 所有注意力后端
- ✅ SLG、FETA、实验性参数
- ✅ 多卡推理（torchrun / Ray）
- ✅ Turbo Diffusion（12 步特殊 LoRA）

**典型配置：**
```python
from kdit.config import SampleConfig, RuntimeConfig
from kdit.pipelines.context_builders.wan import WanI2VExtraInputs
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=5.0)
runtime_config = RuntimeConfig(size=(720, 480), frame_num=81)
extra_inputs = WanI2VExtraInputs(start_img_path="path/to/image.jpg")
```

---

### Wan2.2-TI2V-5B

| 属性 | 值 |
|------|-----|
| **Pipeline Key** | `Wan2_2_TI2V_5B` |
| **Model Key** | [`ModelKey.Wan2_2_TI2V_5B`](../kdit/models/model_key.py:73) |
| **任务** | 文图生视频 |
| **参数量** | ~5B |
| **架构** | 单模型 |
| **文本编码器** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE |
| **输出** | 视频（多帧） |

**描述：** 轻量级 5B 参数模型，用于文生视频和图生视频。适合 GPU 显存有限的场景。

**支持特性：**
- ✅ FP8 量化
- ✅ torch.compile
- ✅ LoRA
- ✅ 缓存策略
- ✅ 注意力后端
- ✅ 多卡推理

---

### Wan2.1-VACE-14B

| 属性 | 值 |
|------|-----|
| **Pipeline Key** | `Wan2_1_VACE_14B` |
| **Model Key** | [`ModelKey.Wan2_1_VACE_14B`](../kdit/models/model_key.py:74) |
| **任务** | 视频条件控制（VACE） |
| **参数量** | ~14B |
| **架构** | 单模型（或双模型） |
| **文本编码器** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE（编码 + 解码） |
| **输入** | 文本 + 控制视频 + 遮罩 + 参考图像 |
| **输出** | 视频（多帧） |

**描述：** 视频-音频条件编码（VACE）模型，用于受控视频生成。支持多种条件输入，包括带遮罩的控制视频和参考图像。可实现视频修复、扩展、风格迁移和运动控制等任务。

**支持特性：**
- ✅ FP8 量化
- ✅ torch.compile
- ✅ LoRA
- ✅ 所有缓存策略
- ✅ 所有注意力后端
- ✅ SLG、FETA、实验性参数（CFG-Zero-Star、FreSca、TCFG 等）
- ✅ 多卡推理（torchrun / Ray）
- ✅ VACE 条件控制（支持开始/结束百分比）

**典型配置：**
```python
from kdit.config import SampleConfig, RuntimeConfig, KsanaExperimentalConfig, KsanaVideoControlConfig
sample_config = SampleConfig(
    steps=20, cfg_scale=4.0, shift=5.0,
    video_control=KsanaVideoControlConfig(
        experimental=KsanaExperimentalConfig(cfg_zero_star=True),
    ),
)
runtime_config = RuntimeConfig(size=(848, 480), frame_num=81)
```

---

### Qwen-Image-T2I-20B

| 属性 | 值 |
|------|-----|
| **Pipeline Key** | `QwenImage_T2I` |
| **Model Key** | [`ModelKey.QwenImage_T2I`](../kdit/models/model_key.py:75) |
| **任务** | 文生图 |
| **参数量** | ~20B |
| **架构** | 单模型 |
| **文本编码器** | Qwen2VL 文本编码器 |
| **VAE** | Qwen Image VAE |
| **输出** | 图像 |

**描述：** 使用 Qwen-Image 架构的高质量文生图。从文本描述生成精细图像。

**支持特性：**
- ✅ FP8 量化
- ✅ torch.compile
- ✅ LoRA
- ✅ 缓存策略
- ✅ 注意力后端
- ✅ 多卡推理

**典型配置：**
```python
from kdit.config import SampleConfig, RuntimeConfig, SolverType
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=-1, solver=SolverType.FLOWMATCH_EULER)
runtime_config = RuntimeConfig(size=(1024, 1024))
```

---

### Qwen-Image-Edit-20B

| 属性 | 值 |
|------|-----|
| **Pipeline Key** | `QwenImage_Edit` |
| **Model Key** | [`ModelKey.QwenImage_Edit`](../kdit/models/model_key.py:76) |
| **任务** | 图像编辑 |
| **参数量** | ~20B |
| **架构** | 单模型 |
| **文本编码器** | Qwen2VL 多模态文本编码器 |
| **VAE** | Qwen Image VAE |
| **输入** | 文本指令 + 源图像 |
| **输出** | 编辑后的图像 |

**描述：** 使用 Qwen-Image 架构和多模态文本编码的指令式图像编辑。接收源图像和文本指令，生成编辑结果。

**支持特性：**
- ✅ FP8 量化
- ✅ torch.compile
- ✅ LoRA
- ✅ 缓存策略
- ✅ 注意力后端
- ✅ 多卡推理

**典型配置：**
```python
from kdit.config import SampleConfig, RuntimeConfig, SolverType
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=-1, solver=SolverType.FLOWMATCH_EULER)
runtime_config = RuntimeConfig(size=(1024, 1024))
# 通过 img_path 参数传入源图像
```

---

## 文本编码器

| Model Key | 名称 | 架构 | 使用模型 |
|-----------|------|------|----------|
| [`T5TextEncoder`](../kdit/models/model_key.py:61) | UMT5-XXL | T5 编码器 (BF16) | Wan2.2-T2V、Wan2.2-I2V、Wan2.2-TI2V、Wan2.1-VACE |
| [`Qwen2VLTextEncoder`](../kdit/models/model_key.py:62) | Qwen2VL 文本编码器 | Qwen2VL | Qwen-Image-T2I |
| [`Qwen2VLTextEncoderMultimodal`](../kdit/models/model_key.py:63) | Qwen2VL 多模态编码器 | Qwen2VL + 视觉 | Qwen-Image-Edit |

### T5 文本编码器 (UMT5-XXL)

- **文件：** `models_t5_umt5-xxl-enc-bf16.pth`
- **分词器：** `google/umt5-xxl/` 目录
- **精度：** BF16
- **用途：** 为所有 Wan 视频模型编码文本提示

### Qwen2VL 文本编码器

- **目录：** `text_encoder/` + `tokenizer/`
- **用途：** 为 Qwen-Image T2I 编码文本提示

### Qwen2VL 多模态文本编码器

- **目录：** `text_encoder/` + `tokenizer/` + `processor/`
- **用途：** 为 Qwen-Image Edit 编码文本 + 图像输入

---

## VAE 模型

| Model Key | 名称 | 使用模型 | 编码 | 解码 |
|-----------|------|----------|------|------|
| [`VAE_WAN2_1`](../kdit/models/model_key.py:67) | Wan2.1 VAE | Wan2.2-T2V、Wan2.2-I2V、Wan2.2-TI2V、Wan2.1-VACE | ✅ | ✅ |
| [`VAE_WAN2_2`](../kdit/models/model_key.py:68) | Wan2.2 VAE | （预留） | ✅ | ✅ |
| [`QwenImageVAE`](../kdit/models/model_key.py:66) | Qwen Image VAE | Qwen-Image-T2I、Qwen-Image-Edit | ✅ | ✅ |

### Wan2.1 VAE

- **文件：** `Wan2.1_VAE.pth`
- **用途：** 为所有 Wan 视频模型将图像/视频编码到潜空间，以及将潜空间解码回像素
- **潜空间通道数：** 16
- **空间步幅：** 8x

### Qwen Image VAE

- **目录：** `vae/`
- **用途：** 为 Qwen-Image 模型编码/解码图像

---

## 模型目录结构

### Wan2.2 T2V / I2V（双模型）

```
Wan2.2-T2V-14B/                    # 或 Wan2.2-I2V-A14B/
├── google/
│   └── umt5-xxl/                  # T5 分词器文件
├── high_noise_model/
│   ├── config.json
│   └── diffusion_pytorch_model-*.safetensors
├── low_noise_model/
│   ├── config.json
│   └── diffusion_pytorch_model-*.safetensors
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 编码器权重
├── Wan2.1_VAE.pth                     # VAE 权重
└── configuration.json                 # 模型配置
```

### Wan2.1 VACE（单模型）

```
Wan2.1-VACE-14B/
├── google/
│   └── umt5-xxl/                  # T5 分词器文件
├── diffusion_pytorch_model-*.safetensors  # 模型分片
├── diffusion_pytorch_model.safetensors.index.json
├── config.json
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 编码器权重
└── Wan2.1_VAE.pth                     # VAE 权重
```

### Qwen-Image T2I

```
Qwen-Image-T2I/
├── text_encoder/                  # Qwen2VL 文本编码器
├── tokenizer/                     # 分词器文件
├── transformer/                   # DiT 模型权重
└── vae/                          # Qwen Image VAE
```

### Qwen-Image Edit

```
Qwen-Image-Edit/
├── text_encoder/                  # Qwen2VL 多模态编码器
├── tokenizer/                     # 分词器文件
├── processor/                     # 图像处理器
├── transformer/                   # DiT 模型权重
└── vae/                          # Qwen Image VAE
```

---

## 模型下载链接

### Wan 视频模型

| 模型 | HuggingFace | ModelScope |
|------|-------------|------------|
| Wan2.2-T2V-14B | [Wan-AI/Wan2.2-T2V-14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-14B) | [Wan-AI/Wan2.2-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-14B) |
| Wan2.2-I2V-14B | [Wan-AI/Wan2.2-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-720P) | [Wan-AI/Wan2.2-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-14B-720P) |
| Wan2.1-VACE-14B | [Wan-AI/Wan2.1-VACE-14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) | [Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) |

### Qwen 图像模型

| 模型 | HuggingFace |
|------|-------------|
| Qwen-Image-T2I | [Qwen/Qwen-Image-T2I](https://huggingface.co/Qwen/Qwen-Image-T2I) |
| Qwen-Image-Edit | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) |

### LoRA 模型

| LoRA | 用途 | 步数 | 来源 |
|------|------|------|------|
| Lightning LoRA | 快速 T2V/I2V 生成 | 4 步 | 社区 |
| Turbo Diffusion LoRA | 快速 I2V 生成 | 12 步 | 社区 |

> **注意：** 模型下载链接可能会变更。请参考官方模型仓库获取最新版本。

---

## 硬件要求

| 模型 | 最低显存 (FP16) | 最低显存 (FP8) | 推荐配置 |
|------|-----------------|----------------|----------|
| Wan2.2-T2V-14B | ~40GB | ~24GB | 1× A100 80GB 或 2× A6000 |
| Wan2.2-I2V-14B | ~40GB | ~24GB | 1× A100 80GB 或 2× A6000 |
| Wan2.2-TI2V-5B | ~20GB | ~12GB | 1× A6000 或 1× RTX 4090 |
| Wan2.1-VACE-14B | ~40GB | ~24GB | 1× A100 80GB 或 2× A6000 |
| Qwen-Image-T2I-20B | ~50GB | ~30GB | 1× A100 80GB |
| Qwen-Image-Edit-20B | ~50GB | ~30GB | 1× A100 80GB |

> **提示：** 使用 FP8 量化（`linear_backend="fp8_e4m3"`）可显著降低显存需求。多卡配置可进一步分散显存使用。

---

## 平台兼容性

| 平台 | Wan 视频 | Qwen 图像 | torch.compile | 多卡后端 |
|------|----------|-----------|---------------|----------|
| NVIDIA GPU (CUDA) | ✅ | ✅ | ✅ | NCCL |
| 华为 NPU (昇腾) | ✅ | ✅ | ❌ | HCCL |
| Intel XPU | ✅ | ✅ | ❌ | — |
