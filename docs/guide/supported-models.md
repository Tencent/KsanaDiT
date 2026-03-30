# Supported Models

[← Back to Index](README.md) | [中文版](supported-models_cn.md)

This document lists all models supported by kDiT, including diffusion models, text encoders, and VAE models.

---

## Table of Contents

- [Supported Models](#supported-models)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Diffusion Models](#diffusion-models)
    - [Wan2.2-T2V-14B](#wan22-t2v-14b)
    - [Wan2.2-I2V-14B](#wan22-i2v-14b)
    - [Wan2.2-TI2V-5B](#wan22-ti2v-5b)
    - [Wan2.1-VACE-14B](#wan21-vace-14b)
    - [Qwen-Image-T2I-20B](#qwen-image-t2i-20b)
    - [Qwen-Image-Edit-20B](#qwen-image-edit-20b)
  - [Text Encoders](#text-encoders)
    - [T5 Text Encoder (UMT5-XXL)](#t5-text-encoder-umt5-xxl)
    - [Qwen2VL Text Encoder](#qwen2vl-text-encoder)
    - [Qwen2VL Multimodal Text Encoder](#qwen2vl-multimodal-text-encoder)
  - [VAE Models](#vae-models)
    - [Wan2.1 VAE](#wan21-vae)
    - [Qwen Image VAE](#qwen-image-vae)
  - [Model Directory Structure](#model-directory-structure)
    - [Wan2.2 T2V / I2V (Dual-Model)](#wan22-t2v--i2v-dual-model)
    - [Wan2.1 VACE (Single-Model)](#wan21-vace-single-model)
    - [Qwen-Image T2I](#qwen-image-t2i)
    - [Qwen-Image Edit](#qwen-image-edit)
  - [Model Download Links](#model-download-links)
    - [Wan Video Models](#wan-video-models)
    - [Qwen Image Models](#qwen-image-models)
    - [LoRA Models](#lora-models)
  - [Hardware Requirements](#hardware-requirements)
  - [Platform Compatibility](#platform-compatibility)

---

## Overview

kDiT supports two families of generative models:

| Family | Task | Models | Parameters |
|--------|------|--------|------------|
| **Wan Video** | Video Generation | Wan2.2-T2V, Wan2.2-I2V, Wan2.2-TI2V, Wan2.1-VACE | 5B–14B |
| **Qwen Image** | Image Generation | Qwen-Image-T2I, Qwen-Image-Edit | 20B |

All models use a **Diffusion Transformer (DiT)** architecture and support:
- FP16 / BF16 inference
- FP8 quantized inference (via `fp8_e4m3` linear backend)
- Multi-GPU parallel inference (torchrun / Ray)
- LoRA acceleration (static merge at load time)
- Smart caching strategies for speedup

---

## Diffusion Models

### Wan2.2-T2V-14B

| Property | Value |
|----------|-------|
| **Pipeline Key** | `Wan2_2_T2V_14B` |
| **Model Key** | [`ModelKey.Wan2_2_T2V_14B`](../kdit/models/model_key.py:71) |
| **Task** | Text-to-Video |
| **Parameters** | ~14B |
| **Architecture** | Dual-model (high-noise + low-noise) |
| **Text Encoder** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE |
| **Output** | Video (multi-frame) |

**Description:** Generates video from text prompts. Uses a dual-model architecture where a high-noise model handles the initial denoising phase and a low-noise model refines the final details. The model boundary (default 0.875) controls the switching point.

**Supported Features:**
- ✅ FP8 quantization
- ✅ torch.compile
- ✅ LoRA (e.g., Lightning LoRA for 4-step generation)
- ✅ All cache strategies (TeaCache, EasyCache, MagCache, DCache, DBCache, HybridCache)
- ✅ All attention backends
- ✅ SLG, FETA, Experimental args
- ✅ Multi-GPU (torchrun / Ray)

**Typical Configuration:**
```python
from kdit.config import SampleConfig, RuntimeConfig
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=5.0)
runtime_config = RuntimeConfig(size=(720, 480), frame_num=81)
```

---

### Wan2.2-I2V-14B

| Property | Value |
|----------|-------|
| **Pipeline Key** | `Wan2_2_I2V_14B` |
| **Model Key** | [`ModelKey.Wan2_2_I2V_14B`](../kdit/models/model_key.py:72) |
| **Task** | Image-to-Video |
| **Parameters** | ~14B |
| **Architecture** | Dual-model (high-noise + low-noise) |
| **Text Encoder** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE (encode + decode) |
| **Input** | Text prompt + reference image(s) |
| **Output** | Video (multi-frame) |

**Description:** Generates video from a text prompt and one or more reference images. The input image is VAE-encoded as the starting frame. Supports start image only, or start + end image for controlled animation.

**Supported Features:**
- ✅ FP8 quantization
- ✅ torch.compile
- ✅ LoRA (e.g., Lightning LoRA, Turbo Diffusion LoRA)
- ✅ All cache strategies
- ✅ All attention backends
- ✅ SLG, FETA, Experimental args
- ✅ Multi-GPU (torchrun / Ray)
- ✅ Turbo Diffusion (12-step with special LoRA)

**Typical Configuration:**
```python
from kdit.config import SampleConfig, RuntimeConfig
from kdit.pipelines.context_builders.wan import WanI2VExtraInputs
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=5.0)
runtime_config = RuntimeConfig(size=(720, 480), frame_num=81)
extra_inputs = WanI2VExtraInputs(start_img_path="path/to/image.jpg")
```

---

### Wan2.2-TI2V-5B

| Property | Value |
|----------|-------|
| **Pipeline Key** | `Wan2_2_TI2V_5B` |
| **Model Key** | [`ModelKey.Wan2_2_TI2V_5B`](../kdit/models/model_key.py:73) |
| **Task** | Text/Image-to-Video |
| **Parameters** | ~5B |
| **Architecture** | Single model |
| **Text Encoder** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE |
| **Output** | Video (multi-frame) |

**Description:** A lighter 5B parameter model for text-to-video and image-to-video generation. Suitable for scenarios with limited GPU memory.

**Supported Features:**
- ✅ FP8 quantization
- ✅ torch.compile
- ✅ LoRA
- ✅ Cache strategies
- ✅ Attention backends
- ✅ Multi-GPU

---

### Wan2.1-VACE-14B

| Property | Value |
|----------|-------|
| **Pipeline Key** | `Wan2_1_VACE_14B` |
| **Model Key** | [`ModelKey.Wan2_1_VACE_14B`](../kdit/models/model_key.py:74) |
| **Task** | Video Conditioning (VACE) |
| **Parameters** | ~14B |
| **Architecture** | Single model (or dual-model) |
| **Text Encoder** | T5 (UMT5-XXL) |
| **VAE** | Wan2.1 VAE (encode + decode) |
| **Input** | Text + control video + masks + reference image |
| **Output** | Video (multi-frame) |

**Description:** Video-Audio Conditional Encoding (VACE) model for controlled video generation. Supports multiple conditioning inputs including control videos with masks and reference images. Enables tasks like video inpainting, outpainting, style transfer, and motion control.

**Supported Features:**
- ✅ FP8 quantization
- ✅ torch.compile
- ✅ LoRA
- ✅ All cache strategies
- ✅ All attention backends
- ✅ SLG, FETA, Experimental args (CFG-Zero-Star, FreSca, TCFG, etc.)
- ✅ Multi-GPU (torchrun / Ray)
- ✅ VACE conditioning with start/end percentage control

**Typical Configuration:**
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

| Property | Value |
|----------|-------|
| **Pipeline Key** | `QwenImage_T2I` |
| **Model Key** | [`ModelKey.QwenImage_T2I`](../kdit/models/model_key.py:75) |
| **Task** | Text-to-Image |
| **Parameters** | ~20B |
| **Architecture** | Single model |
| **Text Encoder** | Qwen2VL Text Encoder |
| **VAE** | Qwen Image VAE |
| **Output** | Image |

**Description:** High-quality text-to-image generation using the Qwen-Image architecture. Produces detailed images from text descriptions.

**Supported Features:**
- ✅ FP8 quantization
- ✅ torch.compile
- ✅ LoRA
- ✅ Cache strategies
- ✅ Attention backends
- ✅ Multi-GPU

**Typical Configuration:**
```python
from kdit.config import SampleConfig, RuntimeConfig, SolverType
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=-1, solver=SolverType.FLOWMATCH_EULER)
runtime_config = RuntimeConfig(size=(1024, 1024))
```

---

### Qwen-Image-Edit-20B

| Property | Value |
|----------|-------|
| **Pipeline Key** | `QwenImage_Edit` |
| **Model Key** | [`ModelKey.QwenImage_Edit`](../kdit/models/model_key.py:76) |
| **Task** | Image Editing |
| **Parameters** | ~20B |
| **Architecture** | Single model |
| **Text Encoder** | Qwen2VL Multimodal Text Encoder |
| **VAE** | Qwen Image VAE |
| **Input** | Text instruction + source image |
| **Output** | Edited image |

**Description:** Instruction-based image editing using the Qwen-Image architecture with multimodal text encoding. Takes a source image and a text instruction to produce an edited result.

**Supported Features:**
- ✅ FP8 quantization
- ✅ torch.compile
- ✅ LoRA
- ✅ Cache strategies
- ✅ Attention backends
- ✅ Multi-GPU

**Typical Configuration:**
```python
from kdit.config import SampleConfig, RuntimeConfig, SolverType
sample_config = SampleConfig(steps=20, cfg_scale=4.0, shift=-1, solver=SolverType.FLOWMATCH_EULER)
runtime_config = RuntimeConfig(size=(1024, 1024))
# Pass source image via img_path parameter
```

---

## Text Encoders

| Model Key | Name | Architecture | Used By |
|-----------|------|-------------|---------|
| [`T5TextEncoder`](../kdit/models/model_key.py:61) | UMT5-XXL | T5 Encoder (BF16) | Wan2.2-T2V, Wan2.2-I2V, Wan2.2-TI2V, Wan2.1-VACE |
| [`Qwen2VLTextEncoder`](../kdit/models/model_key.py:62) | Qwen2VL Text Encoder | Qwen2VL | Qwen-Image-T2I |
| [`Qwen2VLTextEncoderMultimodal`](../kdit/models/model_key.py:63) | Qwen2VL Multimodal Encoder | Qwen2VL + Vision | Qwen-Image-Edit |

### T5 Text Encoder (UMT5-XXL)

- **File:** `models_t5_umt5-xxl-enc-bf16.pth`
- **Tokenizer:** `google/umt5-xxl/` directory
- **Precision:** BF16
- **Usage:** Encodes text prompts for all Wan video models

### Qwen2VL Text Encoder

- **Directory:** `text_encoder/` + `tokenizer/`
- **Usage:** Encodes text prompts for Qwen-Image T2I

### Qwen2VL Multimodal Text Encoder

- **Directory:** `text_encoder/` + `tokenizer/` + `processor/`
- **Usage:** Encodes text + image inputs for Qwen-Image Edit

---

## VAE Models

| Model Key | Name | Used By | Encode | Decode |
|-----------|------|---------|--------|--------|
| [`VAE_WAN2_1`](../kdit/models/model_key.py:67) | Wan2.1 VAE | Wan2.2-T2V, Wan2.2-I2V, Wan2.2-TI2V, Wan2.1-VACE | ✅ | ✅ |
| [`VAE_WAN2_2`](../kdit/models/model_key.py:68) | Wan2.2 VAE | (Reserved) | ✅ | ✅ |
| [`QwenImageVAE`](../kdit/models/model_key.py:66) | Qwen Image VAE | Qwen-Image-T2I, Qwen-Image-Edit | ✅ | ✅ |

### Wan2.1 VAE

- **File:** `Wan2.1_VAE.pth`
- **Usage:** Encodes images/video to latent space and decodes latents back to pixels for all Wan video models
- **Latent channels:** 16
- **Spatial stride:** 8x

### Qwen Image VAE

- **Directory:** `vae/`
- **Usage:** Encodes/decodes images for Qwen-Image models

---

## Model Directory Structure

### Wan2.2 T2V / I2V (Dual-Model)

```
Wan2.2-T2V-14B/                    # or Wan2.2-I2V-A14B/
├── google/
│   └── umt5-xxl/                  # T5 tokenizer files
├── high_noise_model/
│   ├── config.json
│   └── diffusion_pytorch_model-*.safetensors
├── low_noise_model/
│   ├── config.json
│   └── diffusion_pytorch_model-*.safetensors
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 encoder weights
├── Wan2.1_VAE.pth                     # VAE weights
└── configuration.json                 # Model configuration
```

### Wan2.1 VACE (Single-Model)

```
Wan2.1-VACE-14B/
├── google/
│   └── umt5-xxl/                  # T5 tokenizer files
├── diffusion_pytorch_model-*.safetensors  # Model shards
├── diffusion_pytorch_model.safetensors.index.json
├── config.json
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 encoder weights
└── Wan2.1_VAE.pth                     # VAE weights
```

### Qwen-Image T2I

```
Qwen-Image-T2I/
├── text_encoder/                  # Qwen2VL text encoder
├── tokenizer/                     # Tokenizer files
├── transformer/                   # DiT model weights
└── vae/                          # Qwen Image VAE
```

### Qwen-Image Edit

```
Qwen-Image-Edit/
├── text_encoder/                  # Qwen2VL multimodal encoder
├── tokenizer/                     # Tokenizer files
├── processor/                     # Image processor
├── transformer/                   # DiT model weights
└── vae/                          # Qwen Image VAE
```

---

## Model Download Links

### Wan Video Models

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| Wan2.2-T2V-14B | [Wan-AI/Wan2.2-T2V-14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-14B) | [Wan-AI/Wan2.2-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-14B) |
| Wan2.2-I2V-14B | [Wan-AI/Wan2.2-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-720P) | [Wan-AI/Wan2.2-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-14B-720P) |
| Wan2.1-VACE-14B | [Wan-AI/Wan2.1-VACE-14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) | [Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) |

### Qwen Image Models

| Model | HuggingFace |
|-------|-------------|
| Qwen-Image-T2I | [Qwen/Qwen-Image-T2I](https://huggingface.co/Qwen/Qwen-Image-T2I) |
| Qwen-Image-Edit | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) |

### LoRA Models

| LoRA | Purpose | Steps | HuggingFace |
|------|---------|-------|-------------|
| Lightning LoRA | Fast T2V/I2V generation | 4 steps | Community |
| Turbo Diffusion LoRA | Fast I2V generation | 12 steps | Community |

> **Note:** Model download links may change. Please refer to the official model repositories for the latest versions.

---

## Hardware Requirements

| Model | Min VRAM (FP16) | Min VRAM (FP8) | Recommended |
|-------|-----------------|----------------|-------------|
| Wan2.2-T2V-14B | ~40GB | ~24GB | 1× A100 80GB or 2× A6000 |
| Wan2.2-I2V-14B | ~40GB | ~24GB | 1× A100 80GB or 2× A6000 |
| Wan2.2-TI2V-5B | ~20GB | ~12GB | 1× A6000 or 1× RTX 4090 |
| Wan2.1-VACE-14B | ~40GB | ~24GB | 1× A100 80GB or 2× A6000 |
| Qwen-Image-T2I-20B | ~50GB | ~30GB | 1× A100 80GB |
| Qwen-Image-Edit-20B | ~50GB | ~30GB | 1× A100 80GB |

> **Tip:** Use FP8 quantization (`linear_backend="fp8_e4m3"`) to significantly reduce VRAM requirements. Multi-GPU setups can further distribute memory usage.

---

## Platform Compatibility

| Platform | Wan Video | Qwen Image | torch.compile | Multi-GPU Backend |
|----------|-----------|------------|---------------|-------------------|
| NVIDIA GPU (CUDA) | ✅ | ✅ | ✅ | NCCL |
| Huawei NPU (Ascend) | ✅ | ✅ | ❌ | HCCL |
| Intel XPU | ✅ | ✅ | ❌ | — |
