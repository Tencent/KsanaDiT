# ComfyUI Usage Guide

[中文版](comfyui-usage_cn.md)

## Overview

kDiT integrates with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as a custom node plugin, providing visual workflow design for video and image generation. All kDiT nodes are prefixed with **"kDiT"** in the ComfyUI node menu.

## Installation

### As a ComfyUI Custom Node

```bash
# 1. Navigate to ComfyUI's custom_nodes directory
cd /path/to/ComfyUI/custom_nodes

# 2. Clone the kDiT repository
git clone https://github.com/Tencent/kDiT.git

# 3. Enter the kDiT directory and install via script
cd kDiT
./scripts/install_public.sh
```

The install script automatically detects your platform (GPU/NPU/XPU), installs all dependencies, and sets up ComfyUI custom nodes.

### Model Files

Place model files in the standard ComfyUI directories:

- **Diffusion models**: `ComfyUI/models/diffusion_models/`
- **VAE models**: `ComfyUI/models/vae/`
- **LoRA models**: `ComfyUI/models/loras/`

## Basic Workflow

The typical kDiT workflow in ComfyUI follows this pattern:

```
[Model Loader] → [Generator] → [VAE Decoder] → [Preview/Save]
       ↑              ↑
[Attention Config]  [Text Encoder]
[LoRA Select]       [VAE Encoder]
[Torch Compile]     [Cache Config]
```

### Core Flow

1. **kDiT Model Loader** — Load the diffusion model with optional configs
2. **Text Encoder** — Encode text prompts (use ComfyUI's built-in or WanVideoWrapper text encoder)
3. **kDiT VAE Encoder** — Encode input images/video to latent space (for I2V/Edit tasks)
4. **kDiT Generator** — Run the denoising/sampling process
5. **kDiT VAE Decoder** — Decode latents back to images/video
6. **Preview/Save** — Use ComfyUI's built-in preview or save nodes

## Task-Specific Workflows

### Text-to-Video (T2V)

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [Save Video]
                              ↑
[Text Encoder] ──→ positive/negative
[kDiT VAE Encoder] ──→ image_embeds (empty latent)
```

Key settings:
- **Model**: Select a Wan2.2 T2V model (e.g., `wan2.2_t2v_high_noise_14B_fp16.safetensors`)
- **VAE Encoder**: Set `num_frames`, `width`, `height` for output dimensions
- **Generator**: Configure `steps`, `seed`, `solver_name`, `sample_guide_scale`, `sample_shift`

### Image-to-Video (I2V)

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [Save Video]
                              ↑
[Text Encoder] ──→ positive/negative
[Load Image] ──→ [kDiT VAE Encoder] ──→ image_embeds
```

Key settings:
- **Model**: Select a Wan2.2 I2V model (e.g., `wan2.2_i2v_high_noise_14B_fp16.safetensors`)
- **VAE Encoder**: Connect `start_image` (and optionally `end_image`)
- Optionally use dual models (high + low noise) with `low_noise_model_name`

### VACE (Video Controllable Editing)

```
[kDiT Vace Model Select] ──→ [kDiT Model Loader]
                                      ↓
[kDiT WanVace To Video] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder]
         ↑                        ↑
[Load Video/Image]          [Text Encoder]
[kDiT VAE Loader]
```

Key settings:
- **Vace Model Select**: Choose the VACE diffusion model
- **WanVace To Video**: Configure control video, masks, reference image, strength
- Supports chaining multiple VACE inputs via `prev_vace_embeds`

### Text-to-Image (T2I) — Qwen-Image

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [Save Image]
                              ↑
[Text Encoder] ──→ positive/negative
[kDiT VAE Encoder] ──→ image_embeds (empty latent)
```

Key settings:
- **Model**: Select a Qwen-Image model (e.g., `qwen_image_bf16.safetensors`)
- **sample_shift**: Use `-1` for auto (let pipeline compute)
- **solver_name**: Use `flowmatch_euler` for Qwen models

### Image Editing — Qwen-Image Edit

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [Save Image]
                              ↑
[Text Encoder] ──→ positive/negative
[Load Image] ──→ [kDiT VAE Image Encoder] ──→ image_embeds
```

Key settings:
- **Model**: Select a Qwen-Image-Edit model
- **VAE Image Encoder**: Connect reference images for editing

## Dual Model (High/Low Noise)

For models with separate high-noise and low-noise checkpoints (e.g., Wan2.2):

1. In **kDiT Model Loader**, set `model_name` to the high-noise model
2. Set `low_noise_model_name` to the low-noise model
3. Adjust `model_boundary` (default: 0.875) to control the switching point

For cache and LoRA with dual models:
- Use **kDiT CacheCombine** to set separate caches for each model
- Use **kDiT LoraCombine** to set separate LoRAs for each model

## Video Control Features

### Skip Layer Guidance (SLG)

Speeds up sampling by skipping unconditional inference on specified transformer blocks.

1. Add **kDiT Skip Layer Guidance** node
2. Configure `blocks` (e.g., "10"), `start_percent`, `end_percent`
3. Connect to **kDiT VideoControlConfig** → **kDiT Generator**

### Enhance-A-Video (FETA)

Improves temporal consistency by modulating attention with cross-frame scores.

1. Add **kDiT Enhance-A-Video** node
2. Configure `weight`, `start_percent`, `end_percent`
3. Connect to **kDiT VideoControlConfig** → **kDiT Generator**

### Experimental Sampling Args

Advanced sampling optimizations including:
- **CFG-Zero-Star**: Reduces oversaturation from high CFG
- **FreSca**: Frequency-domain filtering for CFG artifacts
- **TCFG**: Tangent-plane CFG for reduced color shifts
- **RAAG**: Adaptive CFG adjustment
- **Bidirectional Sampling**: Forward + backward temporal sampling
- **TSR**: Temporal Score Rescaling

1. Add **kDiT Experimental Args** node
2. Enable desired features
3. Connect to **kDiT VideoControlConfig** → **kDiT Generator**

## WanVideoWrapper Compatibility

kDiT supports inputs from WanVideoWrapper nodes:

- **kDiT TextEmbConverter**: Converts `WANVIDEOTEXTEMBEDS` to kDiT format (positive + negative)
- **kDiT VideoControlConfig**: Accepts both kDiT and WanVideoWrapper SLG/FETA/Experimental args

## Performance Optimization

### Attention Backends

Use **kDiT AttentionConfig** to select:
- `flash_attn` — Flash Attention (recommended for most cases)
- `sage_attn` — Sage Attention
- `torch_sdpa` — PyTorch SDPA

Use **kDiT RadialSageAttentionConfig** for sparse attention patterns.

Use **kDiT SageSLAttentionConfig** for top-k sparse attention.

### Caching

Connect cache nodes to the Generator's `cache_config` input:
- **kDiT DCache** — Fast step-level caching
- **kDiT DBCache** — Block-level caching with TaylorSeer
- **kDiT TeaCache** / **kDiT EasyCache** / **kDiT MagCache** — Various threshold-based caching
- **kDiT HybridCache** — Combine step + block caching

### torch.compile

Use **kDiT TorchCompile** node connected to Model Loader for JIT compilation acceleration.

### Memory Management

Use **kDiT Empty Torch Cache** node between kDiT nodes and memory-intensive post-processing to free GPU memory.

## Node Reference

For a complete list of all 32 ComfyUI nodes with detailed descriptions, see:

📖 [ComfyUI Node Reference (EN)](comfyui-nodes.md) | [ComfyUI 节点参考 (中文)](comfyui-nodes_cn.md)
