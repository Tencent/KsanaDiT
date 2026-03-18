# ComfyUI Node Reference

[← Back to Index](README.md) | [中文版](comfyui-nodes_cn.md)

This document provides a comprehensive reference for all 32 ComfyUI nodes provided by kDiT. Nodes are organized into 7 functional categories.

---

## Table of Contents

- [ComfyUI Node Reference](#comfyui-node-reference)
  - [Table of Contents](#table-of-contents)
  - [1. Core Nodes](#1-core-nodes)
    - [kDiT Model Loader](#kdit-model-loader)
    - [kDiT Vace Model Select](#kdit-vace-model-select)
    - [kDiT Generator](#kdit-generator)
    - [kDiT EmptyLatent](#kdit-emptylatent)
  - [2. VAE Nodes](#2-vae-nodes)
    - [kDiT VAE Loader](#kdit-vae-loader)
    - [kDiT VAE Encoder](#kdit-vae-encoder)
    - [kDiT VAE Image Encoder](#kdit-vae-image-encoder)
    - [kDiT VAE Decoder](#kdit-vae-decoder)
  - [3. Cache Nodes](#3-cache-nodes)
    - [kDiT HybridCache](#kdit-hybridcache)
    - [kDiT CacheCombine](#kdit-cachecombine)
    - [kDiT DCache](#kdit-dcache)
    - [kDiT DBCache](#kdit-dbcache)
    - [kDiT TeaCache](#kdit-teacache)
    - [kDiT EasyCache](#kdit-easycache)
    - [kDiT MagCache](#kdit-magcache)
    - [kDiT CustomStepCache](#kdit-customstepcache)
  - [4. Attention Nodes](#4-attention-nodes)
    - [kDiT AttentionConfig](#kdit-attentionconfig)
    - [kDiT RadialSageAttentionConfig](#kdit-radialsageattentionconfig)
    - [kDiT SageSLAttentionConfig](#kdit-sageslattentionconfig)
  - [5. LoRA Nodes](#5-lora-nodes)
    - [kDiT LoraSelect](#kdit-loraselect)
    - [kDiT LoraSelectMulti](#kdit-loraselectmulti)
    - [kDiT LoraCombine](#kdit-loracombine)
  - [6. Video Control Nodes](#6-video-control-nodes)
    - [kDiT WanVace To Video](#kdit-wanvace-to-video)
    - [kDiT Skip Layer Guidance](#kdit-skip-layer-guidance)
    - [kDiT Enhance-A-Video](#kdit-enhance-a-video)
    - [kDiT Experimental Args](#kdit-experimental-args)
    - [kDiT VideoControlConfig](#kdit-videocontrolconfig)
  - [7. Utility Nodes](#7-utility-nodes)
    - [kDiT TorchCompile](#kdit-torchcompile)
    - [kDiT TextEmbConverter](#kdit-textembconverter)
    - [kDiT DebugNode](#kdit-debugnode)
    - [kDiT Empty Torch Cache](#kdit-empty-torch-cache)
  - [Node Summary Table](#node-summary-table)

---

## 1. Core Nodes

### kDiT Model Loader

| Property | Value |
|----------|-------|
| **Class** | `KsanaModelLoaderNode` |
| **Display Name** | kDiT Model Loader |
| **Category** | kdit |
| **Source** | [`model_loader.py`](../kdit/adapter/comfyui/nodes/model_loader.py:59) |

Loads a diffusion model for inference. This is the primary entry point for setting up a kDiT pipeline in ComfyUI. Supports dual-model configuration (high/low noise), attention backends, linear backends, LoRA, VACE models, and torch.compile.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | DIFFUSION_MODELS | ✅ | — | Primary diffusion model file |
| `run_dtype` | Enum | ❌ | `float16` | Running dtype: `float16`, `bfloat16` |
| `rms_dtype` | Enum | ❌ | `float` | RMSNorm precision: `float` (fp32) or `half` (fp16/bf16) |
| `linear_backend` | Enum | ❌ | `default` | Linear computation backend (e.g., `default`, `fp8_e4m3`) |
| `attention_config` | KSANA_ATTENTION_CONFIG | ❌ | None | Attention backend configuration |
| `low_noise_model_name` | DIFFUSION_MODELS | ❌ | Empty | Low-noise model for dual-model setup |
| `model_boundary` | FLOAT | ❌ | 0.875 | Timestep boundary for high/low model switching (0–1) |
| `torch_compile_args` | KSANA_TORCH_COMPILE | ❌ | None | Torch compile configuration |
| `lora` | KSANA_LORA | ❌ | None | LoRA weights to merge |
| `vace_model` | KSANA_VACE_MODEL | ❌ | None | VACE model for video conditioning |

**Output:** `KSANA_DIFFUSION_MODEL` — Loaded model ready for generation.

---

### kDiT Vace Model Select

| Property | Value |
|----------|-------|
| **Class** | `KsanaVaceModelSelectNode` |
| **Display Name** | kDiT Vace Model Select |
| **Category** | kdit |
| **Source** | [`model_loader.py`](../kdit/adapter/comfyui/nodes/model_loader.py:25) |

Selects VACE (Video-Audio Conditional Encoding) model files for use with the Model Loader. Supports optional dual-model (high/low noise) configuration.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vace_model` | DIFFUSION_MODELS | ✅ | — | Primary VACE model file |
| `vace_low_model` | DIFFUSION_MODELS | ❌ | Empty | Low-noise VACE model for dual-model setup |

**Output:** `KSANA_VACE_MODEL` — VACE model path(s) for the Model Loader.

---

### kDiT Generator

| Property | Value |
|----------|-------|
| **Class** | `KsanaGeneratorNode` |
| **Display Name** | kDiT Generator |
| **Category** | kdit |
| **Source** | [`generator.py`](../kdit/adapter/comfyui/nodes/generator.py:34) |

The main sampling/denoising node. Takes a loaded model, text embeddings, and image embeddings to generate video or image latents through iterative denoising.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | KSANA_DIFFUSION_MODEL | ✅ | — | Loaded diffusion model |
| `positive` | KSANA_TEXT_ENCODE_OUTPUT | ✅ | — | Positive text conditioning |
| `negative` | KSANA_TEXT_ENCODE_OUTPUT | ✅ | — | Negative text conditioning |
| `image_embeds` | KSANA_VAE_ENCODE_OUTPUT | ✅ | — | Encoded image/video latents |
| `steps` | INT | ✅ | 20 | Number of denoising steps (1–10000) |
| `seed` | INT | ✅ | 42 | Random seed for noise generation |
| `scheduler` | Enum | ✅ | `simple` | Noise schedule type (ComfyUI schedulers) |
| `solver_name` | Enum | ✅ | `unipc` | Sampling algorithm: `unipc`, `euler`, `dpm++` |
| `sample_guide_scale` | FLOAT | ✅ | 4.0 | CFG scale (0–100) |
| `sample_shift` | FLOAT | ✅ | 5.0 | Noise schedule shift (-1 for auto with Qwen) |
| `denoise` | FLOAT | ✅ | 1.0 | Denoising strength (0–1) |
| `latent` | LATENT | ❌ | None | Initial latents for video-to-video |
| `rope_function` | Enum | ❌ | `default` | RoPE implementation: `default`, `comfy` |
| `low_sample_guide_scale` | FLOAT | ❌ | 3.0 | CFG scale for low-noise model |
| `cache_config` | KSANA_CACHE_CONFIG | ❌ | None | Cache strategy configuration |
| `sigmas` | FLOAT | ❌ | None | Custom sigma schedule |
| `add_noise_to_latent` | BOOLEAN | ❌ | False | Add noise to latent before sampling |
| `video_control_config` | KSANA_VIDEO_CONTROL_CONFIG | ❌ | None | Video control (SLG, FETA, Experimental) |
| `vace_embeds` | KSANA_VACE_EMBEDS | ❌ | None | VACE embeddings for video conditioning |

**Output:** `KSANA_GENERATE_OUTPUT` — Denoised latents.

---

### kDiT EmptyLatent

| Property | Value |
|----------|-------|
| **Class** | `EmptyLatentNode` |
| **Display Name** | kDiT EmptyLatent |
| **Category** | kdit |
| **Source** | [`empty_latent.py`](../kdit/adapter/comfyui/nodes/empty_latent.py:28) |

Creates empty latent tensors with specified dimensions for text-to-video/image generation. Handles VAE-compatible latent sizing.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `width` | INT | ✅ | 512 | Output width in pixels |
| `height` | INT | ✅ | 512 | Output height in pixels |
| `num_frames` | INT | ✅ | 1 | Number of video frames (1 for images) |
| `batch_size` | INT | ✅ | 1 | Batch size |

**Output:** `KSANA_VAE_ENCODE_OUTPUT` — Empty latent tensor.

---

## 2. VAE Nodes

### kDiT VAE Loader

| Property | Value |
|----------|-------|
| **Class** | `KsanaVAELoaderNode` |
| **Display Name** | kDiT VAE Loader |
| **Category** | kdit |
| **Source** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:22) |

Loads a VAE model from the ComfyUI VAE directory. Used for encoding images to latents and decoding latents to images/video.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vae_name` | VAE list | ✅ | — | VAE model filename |

**Output:** `KDIT_VAE_MODEL` — Loaded VAE model.

---

### kDiT VAE Encoder

| Property | Value |
|----------|-------|
| **Class** | `KsanaVAEEncodeNode` |
| **Display Name** | kDiT VAE Encoder |
| **Category** | kdit |
| **Source** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:38) |

Encodes video frames into latent space using the kDiT VAE. Used for I2V (image-to-video) workflows where input images need to be converted to latents.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vae` | KDIT_VAE_MODEL | ✅ | — | Loaded VAE model |
| `image` | IMAGE | ✅ | — | Input image(s) to encode |
| `width` | INT | ✅ | 512 | Target width |
| `height` | INT | ✅ | 512 | Target height |
| `num_frames` | INT | ✅ | 1 | Number of frames |
| `batch_size` | INT | ✅ | 1 | Batch size |
| `with_end_image` | BOOLEAN | ❌ | False | Include end frame for I2V |

**Output:** `KSANA_VAE_ENCODE_OUTPUT` — Encoded latent tensor.

---

### kDiT VAE Image Encoder

| Property | Value |
|----------|-------|
| **Class** | `KsanaVAEImageEncodeNode` |
| **Display Name** | kDiT VAE Image Encoder |
| **Category** | kdit |
| **Source** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:63) |

Encodes images for Qwen-Image pipelines (T2I/Edit). Handles multi-image encoding for image editing workflows.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vae` | KDIT_VAE_MODEL | ✅ | — | Loaded VAE model |
| `width` | INT | ✅ | 512 | Target width |
| `height` | INT | ✅ | 512 | Target height |
| `batch_size` | INT | ✅ | 1 | Batch size |
| `image` | IMAGE | ❌ | None | Input image(s) for editing |

**Output:** `KSANA_VAE_ENCODE_OUTPUT` — Encoded image latents.

---

### kDiT VAE Decoder

| Property | Value |
|----------|-------|
| **Class** | `KsanaVAEDecodeNode` |
| **Display Name** | kDiT VAE Decoder |
| **Category** | kdit |
| **Source** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:83) |

Decodes latent tensors back to pixel space (images or video frames) using the kDiT VAE.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vae` | KDIT_VAE_MODEL | ✅ | — | Loaded VAE model |
| `latents` | KSANA_GENERATE_OUTPUT | ✅ | — | Generated latent tensors |

**Output:** `IMAGE` — Decoded images/video frames.

---

## 3. Cache Nodes

Cache nodes implement various caching strategies to accelerate inference by reusing intermediate computation results across denoising steps.

### kDiT HybridCache

| Property | Value |
|----------|-------|
| **Class** | `KsanaHybridCacheNode` |
| **Display Name** | kDiT HybridCache |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:20) |

Combines a step-level cache and a block-level cache into a hybrid caching strategy. This is the recommended way to use caching for maximum speedup.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `step_cache` | KSANA_CACHE_CONFIG | ❌ | None | Step-level cache (e.g., TeaCache, EasyCache) |
| `block_cache` | KSANA_CACHE_CONFIG | ❌ | None | Block-level cache (e.g., DCache, DBCache) |
| `name` | STRING | ❌ | `HybridCache` | Display name for logging |

**Output:** `KSANA_CACHE_CONFIG` — Combined hybrid cache configuration.

---

### kDiT CacheCombine

| Property | Value |
|----------|-------|
| **Class** | `KsanaCacheCombineNode` |
| **Display Name** | kDiT CacheCombine |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:40) |

Combines cache configurations for dual-model setups (high-noise + low-noise models). Each model can have its own independent cache strategy.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `cache` | KSANA_CACHE_CONFIG | ✅ | — | Cache config for the primary (high-noise) model |
| `low_noise_model_cache` | KSANA_CACHE_CONFIG | ❌ | None | Cache config for the low-noise model |

**Output:** `KSANA_CACHE_CONFIG` — Combined cache configuration for both models.

---

### kDiT DCache

| Property | Value |
|----------|-------|
| **Class** | `KsanaDCacheNode` |
| **Display Name** | kDiT DCache |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:63) |

Block-level cache using angular difference detection. Divides transformer blocks into "fast" and "slow" groups with different caching thresholds.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `fast_degree` | FLOAT | ❌ | 45 | Angular threshold for fast blocks (1–90°) |
| `slow_degree` | FLOAT | ❌ | 20 | Angular threshold for slow blocks (1–90°) |
| `fast_force_calc_every_n_step` | INT | ❌ | 1 | Force recalculation interval for fast blocks |
| `slow_force_calc_every_n_step` | INT | ❌ | 5 | Force recalculation interval for slow blocks |
| `name` | STRING | ❌ | `""` | Display name |
| `offload` | BOOLEAN | ❌ | False | Offload cached tensors to CPU |

**Output:** `KSANA_CACHE_CONFIG` — DCache configuration.

---

### kDiT DBCache

| Property | Value |
|----------|-------|
| **Class** | `KsanaDBCacheNode` |
| **Display Name** | kDiT DBCache |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:325) |

Dynamic block-level cache with TaylorSeer prediction. Selectively computes only the most important transformer blocks and predicts the rest using Taylor expansion.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `preset` | Enum | ❌ | `balanced` | Preset: `custom`, `conservative`, `balanced`, `aggressive`, `wan22_high`, `wan22_low` |
| `fn_compute_blocks` | INT | ❌ | 8 | Number of blocks to compute in full-noise phase (0–100) |
| `bn_compute_blocks` | INT | ❌ | 6 | Number of blocks to compute in base-noise phase (0–100) |
| `residual_diff_threshold` | FLOAT | ❌ | 0.12 | Threshold for residual difference detection (0.01–1.0) |
| `max_warmup_steps` | INT | ❌ | 5 | Maximum warmup steps before caching starts |
| `warmup_interval` | INT | ❌ | 1 | Interval between warmup computations |
| `max_cached_steps` | INT | ❌ | -1 | Maximum total cached steps (-1 = unlimited) |
| `max_continuous_cached_steps` | INT | ❌ | -1 | Maximum consecutive cached steps (-1 = unlimited) |
| `enable_separate_cfg` | BOOLEAN | ❌ | True | Separate caching for conditional/unconditional |
| `cfg_compute_first` | BOOLEAN | ❌ | False | Compute CFG branch first |
| `enable_taylorseer` | BOOLEAN | ❌ | True | Enable TaylorSeer prediction |
| `taylorseer_order` | INT | ❌ | 1 | Taylor expansion order (0–4) |
| `num_blocks` | INT | ❌ | 40 | Total number of transformer blocks |

**Output:** `KSANA_CACHE_CONFIG` — DBCache configuration.

---

### kDiT TeaCache

| Property | Value |
|----------|-------|
| **Class** | `KsanaTeaCacheNode` |
| **Display Name** | kDiT TeaCache |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:119) |

Step-level cache that skips entire denoising steps when the predicted change is below a threshold. Uses model-specific presets for optimal performance.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `preset` | Enum | ❌ | `balanced` | Preset: `custom`, `wan21_t2v`, `wan21_i2v_720P`, `wan21_i2v_480P`, `wan22_t2v`, `wan22_i2v`, `fast`, `balanced`, `quality` |
| `threshold` | FLOAT | ❌ | 0.2 | Skip threshold (0.05–0.5). Lower = higher quality |
| `mode` | Enum | ❌ | `t2v_14B` | Model mode: `t2v_14B`, `t2v_1.3B`, `i2v_720P`, `i2v_480P` |
| `start_step` | INT | ❌ | 0 | Step to start caching |
| `end_step` | INT | ❌ | -1 | Step to stop caching (-1 = end) |
| `cache_device` | Enum | ❌ | `main_device` | Cache storage: `main_device`, `offload_device` |
| `verbose` | BOOLEAN | ❌ | False | Enable verbose logging |

**Output:** `KSANA_CACHE_CONFIG` — TeaCache configuration.

---

### kDiT EasyCache

| Property | Value |
|----------|-------|
| **Class** | `KsanaEasyCacheNode` |
| **Display Name** | kDiT EasyCache |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:183) |

Step-level cache with percentage-based step range control. Simpler configuration than TeaCache with intuitive start/end percentage parameters.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `preset` | Enum | ❌ | `balanced` | Preset: `custom`, `wan21_t2v`, `wan21_i2v`, `wan22_t2v`, `wan22_i2v`, `conservative`, `balanced`, `aggressive` |
| `reuse_thresh` | FLOAT | ❌ | 0.05 | Reuse threshold (0.001–2.0). Lower = higher quality |
| `start_percent` | FLOAT | ❌ | 0.2 | Start percentage of steps (0–1) |
| `end_percent` | FLOAT | ❌ | 0.98 | End percentage of steps (0–1) |
| `mode` | Enum | ❌ | `t2v` | Model mode: `t2v`, `i2v` |
| `cache_device` | Enum | ❌ | `main_device` | Cache storage: `main_device`, `offload_device` |
| `verbose` | BOOLEAN | ❌ | False | Enable verbose logging |
| `name` | STRING | ❌ | `""` | Display name |

**Output:** `KSANA_CACHE_CONFIG` — EasyCache configuration.

---

### kDiT MagCache

| Property | Value |
|----------|-------|
| **Class** | `KsanaMagCacheNode` |
| **Display Name** | kDiT MagCache |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:254) |

Step-level cache with magnitude-based change detection and configurable maximum skip steps. Supports retention ratio for partial cache reuse.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `preset` | Enum | ❌ | `balanced` | Preset: `custom`, `conservative`, `balanced`, `aggressive`, `wan22_t2v`, `wan22_i2v` |
| `threshold` | FLOAT | ❌ | 0.04 | Change detection threshold (0.001–0.5) |
| `max_skip_steps` | INT | ❌ | 2 | Maximum consecutive skipped steps (1–10) |
| `retention_ratio` | FLOAT | ❌ | 0.2 | Ratio of cached data to retain (0–1) |
| `mode` | Enum | ❌ | `t2v` | Model mode: `t2v`, `i2v` |
| `cache_device` | Enum | ❌ | `offload_device` | Cache storage: `offload_device`, `main_device` |
| `start_step` | INT | ❌ | 0 | Step to start caching |
| `end_step` | INT | ❌ | -1 | Step to stop caching (-1 = end) |
| `verbose` | BOOLEAN | ❌ | False | Enable verbose logging |

**Output:** `KSANA_CACHE_CONFIG` — MagCache configuration.

---

### kDiT CustomStepCache

| Property | Value |
|----------|-------|
| **Class** | `KsanaCustomStepCacheNode` |
| **Display Name** | kDiT CustomStepCache |
| **Category** | kdit/cache |
| **Source** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:92) |

Manually specify which denoising steps to cache. Provides full control over the caching schedule with optional per-step scale factors.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `steps` | FLOAT (forceInput) | ✅ | — | Step indices to cache (0-indexed) |
| `scales` | FLOAT (forceInput) | ❌ | 1.0 | Scale factors per cached step |
| `name` | STRING | ❌ | `""` | Display name |
| `offload` | BOOLEAN | ❌ | False | Offload cached tensors to CPU |

**Output:** `KSANA_CACHE_CONFIG` — CustomStepCache configuration.

---

## 4. Attention Nodes

### kDiT AttentionConfig

| Property | Value |
|----------|-------|
| **Class** | `KsanaAttentionConfigNode` |
| **Display Name** | kDiT AttentionConfig |
| **Category** | kdit/configs |
| **Source** | [`attn_config.py`](../kdit/adapter/comfyui/nodes/attn_config.py:19) |

Configures the attention backend for the diffusion model. Different backends offer various trade-offs between speed and quality.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `backend` | Enum | ❌ | `flash_attn` | Backend: `flash_attn`, `sage_attn`, `torch_sdpa`, `sage_sla` |

**Output:** `KSANA_ATTENTION_CONFIG` — Attention configuration.

**Available Backends:**
- **`flash_attn`** — Flash Attention 2. Best balance of speed and quality (recommended default)
- **`sage_attn`** — Sage Attention. Faster than Flash with minor quality trade-off
- **`torch_sdpa`** — PyTorch native Scaled Dot-Product Attention. Universal compatibility
- **`sage_sla`** — Sage SL Attention with top-k sparse selection

---

### kDiT RadialSageAttentionConfig

| Property | Value |
|----------|-------|
| **Class** | `KsanaRadialSageAttentionConfigNode` |
| **Display Name** | kDiT RadialSageAttentionConfig |
| **Category** | kdit/configs |
| **Source** | [`attn_config.py`](../kdit/adapter/comfyui/nodes/attn_config.py:42) |

Configures Radial Sage Attention, which implements sparse attention patterns with radial decay for efficient video generation. Particularly effective for long video sequences.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dense_blocks_num` | INT | ❌ | 0 | Number of dense attention blocks at the beginning (0–1000) |
| `dense_attn_steps` | INT | ❌ | 1 | Number of steps using dense attention (0–1000) |
| `decay_factor` | FLOAT | ❌ | 0.2 | Radial decay factor (0.01–0.99). Lower = more sparse |
| `block_size` | Enum | ❌ | 64 | Block size for sparse computation: `64`, `128` |
| `dense_backend` | Enum | ❌ | `sage_attn` | Backend for dense blocks: `flash_attn`, `sage_attn`, `torch_sdpa` |

**Output:** `KSANA_ATTENTION_CONFIG` — Radial Sage Attention configuration.

---

### kDiT SageSLAttentionConfig

| Property | Value |
|----------|-------|
| **Class** | `KsanaSageSLAConfigNode` |
| **Display Name** | kDiT SageSLAttentionConfig |
| **Category** | kdit/configs |
| **Source** | [`attn_config.py`](../kdit/adapter/comfyui/nodes/attn_config.py:110) |

Configures Sage SL Attention with top-k selection for sparse attention computation. Selects only the most important attention entries for computation.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topk` | FLOAT | ✅ | 0.1 | Top-k ratio for sparse attention (0.01–0.99) |
| `dense_backend` | Enum | ✅ | `sage_attn` | Backend for dense computation: `flash_attn`, `sage_attn`, `torch_sdpa` |

**Output:** `KSANA_ATTENTION_CONFIG` — Sage SLA configuration.

---

## 5. LoRA Nodes

### kDiT LoraSelect

| Property | Value |
|----------|-------|
| **Class** | `KsanaLoraSelectNode` |
| **Display Name** | kDiT LoraSelect |
| **Category** | kdit |
| **Source** | [`lora.py`](../kdit/adapter/comfyui/nodes/lora.py:112) |

Selects a single LoRA file for merging into the diffusion model at load time. LoRA weights are statically merged (not dynamically switchable at runtime).

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `lora` | LORAS list | ✅ | — | LoRA file to load |
| `strength` | FLOAT | ❌ | 1.0 | LoRA merge strength (0–10) |

**Output:** `KSANA_LORA` — LoRA configuration.

---

### kDiT LoraSelectMulti

| Property | Value |
|----------|-------|
| **Class** | `KsanaLoraSelectMultiNode` |
| **Display Name** | kDiT LoraSelectMulti |
| **Category** | kdit |
| **Source** | [`lora.py`](../kdit/adapter/comfyui/nodes/lora.py:28) |

Selects up to 5 LoRA files simultaneously with individual strength controls. All LoRAs are merged at model load time.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `lora_1` | LORAS list | ✅ | — | First LoRA file |
| `strength_1` | FLOAT | ❌ | 1.0 | Strength for first LoRA |
| `lora_2` – `lora_5` | LORAS list | ❌ | Empty | Additional LoRA files |
| `strength_2` – `strength_5` | FLOAT | ❌ | 1.0 | Strengths for additional LoRAs |

**Output:** `KSANA_LORA` — Combined LoRA configuration.

---

### kDiT LoraCombine

| Property | Value |
|----------|-------|
| **Class** | `KsanaLoraCombineNode` |
| **Display Name** | kDiT LoraCombine |
| **Category** | kdit |
| **Source** | [`lora.py`](../kdit/adapter/comfyui/nodes/lora.py:141) |

Combines multiple LoRA configurations from separate LoraSelect nodes into a single configuration.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `lora_1` | KSANA_LORA | ✅ | — | First LoRA config |
| `lora_2` | KSANA_LORA | ❌ | None | Second LoRA config |
| `lora_3` | KSANA_LORA | ❌ | None | Third LoRA config |

**Output:** `KSANA_LORA` — Combined LoRA configuration.

---

## 6. Video Control Nodes

### kDiT WanVace To Video

| Property | Value |
|----------|-------|
| **Class** | `KsanaWanVaceToVideoNode` |
| **Display Name** | kDiT WanVace To Video |
| **Category** | kdit/vace |
| **Source** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:39) |

Encodes control video and reference images for Wan VACE (Video-Audio Conditional Encoding) video generation. Supports control video with masks, reference images, and chaining multiple VACE inputs.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `width` | INT | ✅ | 512 | Output video width (16–8192) |
| `height` | INT | ✅ | 512 | Output video height (16–8192) |
| `strength` | FLOAT | ✅ | 1.0 | VACE conditioning strength (0–100) |
| `num_frames` | INT | ✅ | 25 | Number of frames to generate (1–8192) |
| `batch_size` | INT | ✅ | 1 | Batch size (1–4096) |
| `vace_start_percent` | FLOAT | ✅ | 0.0 | Start percentage for VACE application (0–1) |
| `vace_end_percent` | FLOAT | ✅ | 1.0 | End percentage for VACE application (0–1) |
| `vae` | KDIT_VAE_MODEL | ❌ | None | VAE model for encoding (required for operation) |
| `control_video` | IMAGE | ❌ | None | Control video frames |
| `control_masks` | MASK | ❌ | None | Masks for control video |
| `reference_image` | IMAGE | ❌ | None | Reference image for style guidance |
| `prev_vace_embeds` | KSANA_VACE_EMBEDS | ❌ | None | Previous VACE embeddings for chaining |

**Outputs:**
- `KSANA_VACE_EMBEDS` — VACE embeddings (vace_context, vace_scale, metadata)
- `KSANA_VAE_ENCODE_OUTPUT` — Latent tensor for the video

---

### kDiT Skip Layer Guidance

| Property | Value |
|----------|-------|
| **Class** | `KsanaSLGNode` |
| **Display Name** | kDiT Skip Layer Guidance |
| **Category** | kdit/vace |
| **Source** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:309) |

Skip Layer Guidance (SLG): Skips unconditional inference on specified transformer blocks to speed up CFG-based sampling without significant quality loss.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `blocks` | STRING | ✅ | `"10"` | Comma-separated block indices to skip (e.g., `"9,10,11"`) |
| `start_percent` | FLOAT | ✅ | 0.1 | Start percentage of steps (0–1) |
| `end_percent` | FLOAT | ✅ | 1.0 | End percentage of steps (0–1) |

**Output:** `KSANA_SLG_ARGS` — SLG configuration.

---

### kDiT Enhance-A-Video

| Property | Value |
|----------|-------|
| **Class** | `KsanaEnhanceAVideoNode` |
| **Display Name** | kDiT Enhance-A-Video |
| **Category** | kdit/vace |
| **Source** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:370) |

Enhance-A-Video (FETA): Improves temporal consistency by modulating attention with cross-frame scores. Reduces flickering and improves motion smoothness.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `weight` | FLOAT | ✅ | 2.0 | Enhancement weight (0–100). Typical range: 1.0–5.0 |
| `start_percent` | FLOAT | ✅ | 0.0 | Start percentage of steps (0–1) |
| `end_percent` | FLOAT | ✅ | 1.0 | End percentage of steps (0–1) |

**Output:** `KSANA_FETA_ARGS` — FETA configuration.

**Reference:** [Enhance-A-Video](https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video)

---

### kDiT Experimental Args

| Property | Value |
|----------|-------|
| **Class** | `KsanaExperimentalArgsNode` |
| **Display Name** | kDiT Experimental Args |
| **Category** | kdit/vace |
| **Source** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:432) |

Collection of experimental sampling optimizations. Each technique can be independently enabled/disabled.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| **CFG-Zero-Star** | | | | |
| `cfg_zero_star` | BOOLEAN | ✅ | False | Enable CFG-Zero-Star: reduces oversaturation from high CFG |
| `use_zero_init` | BOOLEAN | ✅ | False | Return zero noise for initial steps to stabilize sampling |
| `zero_star_steps` | INT | ✅ | 0 | Number of initial zero-init steps (0–100) |
| **FreSca** | | | | |
| `use_fresca` | BOOLEAN | ✅ | False | Enable frequency-domain filtering to reduce CFG artifacts |
| `fresca_scale_low` | FLOAT | ✅ | 1.0 | Low frequency scale factor (0–10) |
| `fresca_scale_high` | FLOAT | ✅ | 1.25 | High frequency scale factor (0–10) |
| `fresca_freq_cutoff` | INT | ✅ | 20 | Frequency cutoff threshold (0–10000) |
| **TCFG** | | | | |
| `use_tcfg` | BOOLEAN | ✅ | False | Enable tangent-plane CFG projection to reduce color shifts |
| **RAAG** | | | | |
| `raag_alpha` | FLOAT | ✅ | 0.0 | Adaptive CFG adjustment alpha (0 = disabled, 0–10) |
| **Bidirectional** | | | | |
| `bidirectional_sampling` | BOOLEAN | ✅ | False | Forward + backward temporal sampling (doubles compute) |
| **TSR** | | | | |
| `temporal_score_rescaling` | BOOLEAN | ✅ | False | Rescale noise prediction based on temporal statistics |
| `tsr_k` | FLOAT | ✅ | 0.95 | TSR temperature (0–100). Lower = stronger rescaling |
| `tsr_sigma` | FLOAT | ✅ | 1.0 | TSR sigma: how early TSR influences sampling (0–1) |
| **Other** | | | | |
| `video_attention_split_steps` | STRING | ✅ | `""` | Comma-separated step indices for attention split with multiple prompts |

**Output:** `KSANA_EXPERIMENTAL_ARGS` — Experimental configuration.

**References:**
- [CFG-Zero-Star](https://github.com/WeichenFan/CFG-Zero-star)
- [FreSca](https://github.com/WikiChao/FreSca)
- [TCFG](https://arxiv.org/abs/2503.18137)
- [Bidirectional Sampling (WanFM)](https://github.com/ff2416/WanFM)
- [TSR](https://github.com/temporalscorerescaling/TSR)

---

### kDiT VideoControlConfig

| Property | Value |
|----------|-------|
| **Class** | `KsanaVideoControlConfigNode` |
| **Display Name** | kDiT VideoControlConfig |
| **Category** | kdit |
| **Source** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:617) |

Combines video control parameters (SLG, FETA, Experimental) into a single configuration for the Generator node. Also accepts WanVideoWrapper-compatible inputs for cross-plugin compatibility.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `slg_args` | KSANA_SLG_ARGS | ❌ | None | kDiT Skip Layer Guidance config |
| `feta_args` | KSANA_FETA_ARGS | ❌ | None | kDiT Enhance-A-Video config |
| `experimental_args` | KSANA_EXPERIMENTAL_ARGS | ❌ | None | kDiT Experimental Args config |
| `wanvideo_slg_args` | WANVIDEO_SLG_ARGS | ❌ | None | WanVideoWrapper SLG (compatible) |
| `wanvideo_feta_args` | WANVIDEO_FETA_ARGS | ❌ | None | WanVideoWrapper FETA (compatible) |
| `wanvideo_exp_args` | WANVIDEO_EXPERIMENTAL_ARGS | ❌ | None | WanVideoWrapper Experimental (compatible) |

**Output:** `KSANA_VIDEO_CONTROL_CONFIG` — Combined video control configuration.

---

## 7. Utility Nodes

### kDiT TorchCompile

| Property | Value |
|----------|-------|
| **Class** | `KsanaTorchCompileNode` |
| **Display Name** | kDiT TorchCompile |
| **Category** | kdit |
| **Source** | [`torch_compile.py`](../kdit/adapter/comfyui/nodes/torch_compile.py:20) |

Configures `torch.compile()` optimization for the diffusion model. Can significantly speed up inference after an initial compilation warmup. Not supported on NPU.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `backend` | Enum | ❌ | `inductor` | Compile backend |
| `mode` | Enum | ❌ | `max-autotune` | Optimization mode |
| `fullgraph` | BOOLEAN | ❌ | True | Compile as full graph |
| `dynamic` | BOOLEAN | ❌ | False | Enable dynamic shapes |

**Output:** `KSANA_TORCH_COMPILE` — Torch compile configuration.

---

### kDiT TextEmbConverter

| Property | Value |
|----------|-------|
| **Class** | `KsanaTextEmbConverterNode` |
| **Display Name** | kDiT TextEmbConverter |
| **Category** | kdit |
| **Source** | [`converter.py`](../kdit/adapter/comfyui/nodes/converter.py:23) |

Converts text embeddings from other ComfyUI text encoder nodes (e.g., standard CLIP/T5 nodes) into the kDiT-compatible format.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text_emb` | CONDITIONING | ✅ | — | Text embeddings from ComfyUI text encoder |

**Output:** `KSANA_TEXT_ENCODE_OUTPUT` — Converted text embeddings.

---

### kDiT DebugNode

| Property | Value |
|----------|-------|
| **Class** | `KsanaDebugNode` |
| **Display Name** | kDiT DebugNode |
| **Category** | kdit |
| **Source** | [`debug.py`](../kdit/adapter/comfyui/nodes/debug.py:22) |

Passthrough debug node that logs the shape, dtype, and other properties of any input tensor or data structure. Useful for workflow debugging.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | * (any) | ✅ | — | Any data to inspect |
| `name` | STRING | ❌ | `""` | Label for the debug output |

**Output:** Same type as input (passthrough).

---

### kDiT Empty Torch Cache

| Property | Value |
|----------|-------|
| **Class** | `KsanaEmptyTorchCacheNode` |
| **Display Name** | kDiT Empty Torch Cache |
| **Category** | kdit |
| **Source** | [`empty_torch_cache.py`](../kdit/adapter/comfyui/nodes/empty_torch_cache.py:19) |

Passthrough node that frees GPU/NPU memory by emptying the PyTorch CUDA/NPU cache. Connect between heavy operations to reclaim VRAM.

**Inputs:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source` | * (any) | ✅ | — | Any data (passed through) |

**Output:** Same type as input (passthrough, after cache clearing).

---

## Node Summary Table

| # | Display Name | Class Name | Category | Purpose |
|---|-------------|------------|----------|---------|
| 1 | kDiT Model Loader | `KsanaModelLoaderNode` | kdit | Load diffusion model |
| 2 | kDiT Vace Model Select | `KsanaVaceModelSelectNode` | kdit | Select VACE model files |
| 3 | kDiT Generator | `KsanaGeneratorNode` | kdit | Main sampling/denoising |
| 4 | kDiT EmptyLatent | `EmptyLatentNode` | kdit | Create empty latents |
| 5 | kDiT VAE Loader | `KsanaVAELoaderNode` | kdit | Load VAE model |
| 6 | kDiT VAE Encoder | `KsanaVAEEncodeNode` | kdit | Encode video to latents |
| 7 | kDiT VAE Image Encoder | `KsanaVAEImageEncodeNode` | kdit | Encode images to latents |
| 8 | kDiT VAE Decoder | `KsanaVAEDecodeNode` | kdit | Decode latents to pixels |
| 9 | kDiT HybridCache | `KsanaHybridCacheNode` | kdit/cache | Combine step + block cache |
| 10 | kDiT CacheCombine | `KsanaCacheCombineNode` | kdit/cache | Combine caches for dual model |
| 11 | kDiT DCache | `KsanaDCacheNode` | kdit/cache | Block-level angular cache |
| 12 | kDiT DBCache | `KsanaDBCacheNode` | kdit/cache | Dynamic block cache + TaylorSeer |
| 13 | kDiT TeaCache | `KsanaTeaCacheNode` | kdit/cache | Step-level threshold cache |
| 14 | kDiT EasyCache | `KsanaEasyCacheNode` | kdit/cache | Step-level percentage cache |
| 15 | kDiT MagCache | `KsanaMagCacheNode` | kdit/cache | Step-level magnitude cache |
| 16 | kDiT CustomStepCache | `KsanaCustomStepCacheNode` | kdit/cache | Manual step cache |
| 17 | kDiT AttentionConfig | `KsanaAttentionConfigNode` | kdit/configs | Attention backend config |
| 18 | kDiT RadialSageAttentionConfig | `KsanaRadialSageAttentionConfigNode` | kdit/configs | Radial sparse attention |
| 19 | kDiT SageSLAttentionConfig | `KsanaSageSLAConfigNode` | kdit/configs | Top-k sparse attention |
| 20 | kDiT LoraSelect | `KsanaLoraSelectNode` | kdit | Single LoRA selection |
| 21 | kDiT LoraSelectMulti | `KsanaLoraSelectMultiNode` | kdit | Multi-LoRA selection (up to 5) |
| 22 | kDiT LoraCombine | `KsanaLoraCombineNode` | kdit | Combine LoRA configs |
| 23 | kDiT WanVace To Video | `KsanaWanVaceToVideoNode` | kdit/vace | VACE video conditioning |
| 24 | kDiT Skip Layer Guidance | `KsanaSLGNode` | kdit/vace | SLG speed optimization |
| 25 | kDiT Enhance-A-Video | `KsanaEnhanceAVideoNode` | kdit/vace | Temporal consistency (FETA) |
| 26 | kDiT Experimental Args | `KsanaExperimentalArgsNode` | kdit/vace | Experimental sampling opts |
| 27 | kDiT VideoControlConfig | `KsanaVideoControlConfigNode` | kdit | Combine video control args |
| 28 | kDiT TorchCompile | `KsanaTorchCompileNode` | kdit | torch.compile config |
| 29 | kDiT TextEmbConverter | `KsanaTextEmbConverterNode` | kdit | Convert text embeddings |
| 30 | kDiT DebugNode | `KsanaDebugNode` | kdit | Debug inspection |
| 31 | kDiT Empty Torch Cache | `KsanaEmptyTorchCacheNode` | kdit | Free GPU memory |
