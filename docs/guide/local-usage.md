# Local Usage Guide

[中文版](local-usage_cn.md)

## Overview

kDiT provides a Python API centered around the `Pipeline` class for local inference. You can generate videos and images with just a few lines of code.

## Quick Start

### Minimal Example — Text-to-Video

```python
from kdit import Pipeline
from kdit.config import RuntimeConfig, SampleConfig

pipeline = Pipeline.from_models("/path/to/Wan2.2-T2V-A14B")

video = pipeline.generate(
    "A girl skateboarding on a New York street",
    sample_config=SampleConfig(steps=40),
    runtime_config=RuntimeConfig(
        seed=1234,
        size=(720, 480),
        frame_num=81,
        return_frames=True,
    ),
)
print("video shape:", video.shape)
```

### Minimal Example — Text-to-Image

```python
import torch
from kdit import Pipeline
from kdit.config import ModelConfig, RuntimeConfig, SampleConfig, SolverType
from kdit.utils.media import save_image

pipeline = Pipeline.from_models(
    "/path/to/Qwen-Image",
    model_config=ModelConfig(run_dtype=torch.bfloat16),
    offload_device="cpu",
)

image = pipeline.generate(
    "A cute orange cat sitting on a windowsill, sunlight streaming through",
    prompt_negative=" ",
    sample_config=SampleConfig(steps=20, cfg_scale=4.0, solver=SolverType.FLOWMATCH_EULER),
    runtime_config=RuntimeConfig(seed=42, size=(1024, 1024)),
)
save_image(image, "output.png")
```

## Supported Tasks

### 1. Wan2.2 Text-to-Video (T2V)

Generate videos from text prompts using the Wan2.2-T2V model.

```python
pipeline = Pipeline.from_models("/path/to/Wan2.2-T2V-A14B")
video = pipeline.generate(
    "your prompt",
    sample_config=SampleConfig(steps=40),
    runtime_config=RuntimeConfig(seed=1234, size=(720, 480), frame_num=81),
)
```

**Example**: [`examples/local/wan/wan2_2_t2v.py`](../examples/local/wan/wan2_2_t2v.py)

### 2. Wan2.2 Image-to-Video (I2V)

Generate videos from an input image and text prompt.

```python
from kdit.pipelines.context_builders.wan import WanI2VExtraInputs

pipeline = Pipeline.from_models("/path/to/Wan2.2-I2V-A14B")
video = pipeline.generate(
    "your prompt",
    extra_inputs=WanI2VExtraInputs(start_img_path="input.png"),
    sample_config=SampleConfig(steps=40),
    runtime_config=RuntimeConfig(seed=1234, size=(1280, 720), frame_num=81),
)
```

Supports start image, end image, and start+end image modes:
```python
video = pipeline.generate(
    "your prompt",
    extra_inputs=WanI2VExtraInputs(
        start_img_path="start.png",
        end_img_path="end.png",  # optional
    ),
    ...
)
```

**Example**: [`examples/local/wan/wan2_2_i2v.py`](../examples/local/wan/wan2_2_i2v.py)

### 3. Wan2.1 VACE (Video Controllable Editing)

Generate or edit videos with control video and reference image guidance.

```python
from kdit.config import DCacheConfig, KsanaSLGConfig, KsanaFETAConfig, KsanaVideoControlConfig, SolverType
from kdit.utils import load_control_frames
from kdit.utils.vace import VaceConfig

pipeline = Pipeline.from_models("/path/to/Wan2.1-VACE-14B")

# With reference image
reference_image = load_control_frames("reference.png", max_frames=1, target_size=(512, 512))
video_control_config = VaceConfig(reference_image=reference_image, strength=1.0)

video = pipeline.generate(
    "your prompt",
    prompt_negative="bad quality...",
    sample_config=SampleConfig(steps=30, cfg_scale=6.0, shift=8.0, solver=SolverType.UNI_PC),
    runtime_config=RuntimeConfig(seed=1234, size=(512, 512), frame_num=81),
    cache_config=[DCacheConfig()],
    video_control_config=video_control_config,
)
```

**Example**: [`examples/local/wan/wan2_1_vace.py`](../examples/local/wan/wan2_1_vace.py)

### 4. Qwen-Image Text-to-Image (T2I)

Generate images from text prompts.

```python
pipeline = Pipeline.from_models(
    "/path/to/Qwen-Image",
    model_config=ModelConfig(run_dtype=torch.bfloat16),
    offload_device="cpu",
)
image = pipeline.generate(
    "your prompt",
    prompt_negative=" ",
    sample_config=SampleConfig(steps=20, cfg_scale=4.0, solver=SolverType.FLOWMATCH_EULER),
    runtime_config=RuntimeConfig(seed=42, size=(1024, 1024)),
)
```

**Example**: [`examples/local/qwen/qwen_image_t2i.py`](../examples/local/qwen/qwen_image_t2i.py)

### 5. Qwen-Image Edit

Edit images using text instructions and reference images.

```python
from kdit.config import LoraConfig
from kdit.models.model_key import ModelKey

pipeline = Pipeline.from_models(
    "/path/to/Qwen-Image-Edit",
    model_config=ModelConfig(run_dtype=torch.bfloat16),
    pipeline_key=ModelKey.QwenImage_Edit,
    lora_config=LoraConfig(path="/path/to/lora.safetensors", strength=1.0),  # optional
    offload_device="cpu",
)
image = pipeline.generate(
    "the woman and man are hugging together",
    prompt_negative="blur, bad anatomy",
    img_path=["image1.png", "image2.png"],
    sample_config=SampleConfig(steps=40, cfg_scale=4.0, solver=SolverType.FLOWMATCH_EULER),
    runtime_config=RuntimeConfig(seed=321, size=(1024, 1024)),
)
```

**Example**: [`examples/local/qwen/qwen_image_edit.py`](../examples/local/qwen/qwen_image_edit.py)

## Configuration Reference

### ModelConfig

Controls model loading and computation settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_dtype` | `torch.dtype` | `torch.float16` | Runtime data type (float16, bfloat16) |
| `attention_config` | `KsanaAttentionConfig` | Flash Attention | Attention backend configuration |
| `linear_backend` | `KsanaLinearBackend` | `DEFAULT` | Linear layer backend (DEFAULT, FP8_GEMM) |
| `torch_compile_config` | `KsanaTorchCompileConfig` | `None` | torch.compile settings |
| `boundary` | `float` | `0.875` | High/low noise model boundary |

### SampleConfig

Controls the sampling/denoising process.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `int` | `20` | Number of denoising steps |
| `cfg_scale` | `float` | `4.0` | Classifier-Free Guidance scale |
| `shift` | `float` | `5.0` | Noise schedule shift |
| `solver` | `SolverType` | `UNI_PC` | Solver algorithm (UNI_PC, EULER, DPM_PLUS_PLUS, FLOWMATCH_EULER) |
| `sigmas` | `list[float]` | `None` | Custom sigma schedule |
| `video_control` | `KsanaVideoControlConfig` | `None` | Video control config (SLG, FETA, Experimental) |

### RuntimeConfig

Controls runtime behavior and output settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed |
| `size` | `tuple[int, int]` | `(1280, 720)` | Output (width, height) |
| `frame_num` | `int` | `81` | Number of video frames |
| `return_frames` | `bool` | `False` | Return frames as tensor |
| `output_folder` | `str` | `None` | Output directory |
| `save_output` | `bool` | `False` | Auto-save output |
| `rope_function` | `str` | `"default"` | RoPE implementation ("default" or "comfy") |

### DistributedConfig

Controls multi-GPU distributed inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_gpus` | `int` | `1` | Number of GPUs to use |

### LoraConfig

Controls LoRA model loading.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Path to LoRA weights |
| `strength` | `float` | `1.0` | LoRA merge strength |

### Cache Configurations

kDiT provides multiple caching strategies to accelerate inference:

| Cache Type | Class | Description |
|------------|-------|-------------|
| DCache | `DCacheConfig` | Step-level caching with fast/slow degree control |
| DBCache | `DBCacheConfig` | Block-level caching with TaylorSeer prediction |
| TeaCache | `TeaCacheConfig` | Threshold-based step caching with presets |
| EasyCache | `EasyCacheConfig` | Simple threshold-based caching |
| MagCache | `MagCacheConfig` | Magnitude-based caching with retention ratio |
| CustomStepCache | `CustomStepCacheConfig` | User-defined step caching |
| HybridCache | `HybridCacheConfig` | Combines step_cache + block_cache |

Example — HybridCache:
```python
from kdit.config.cache_config import DCacheConfig, DBCacheConfig, HybridCacheConfig

cache_config = HybridCacheConfig(
    step_cache=DCacheConfig(fast_degree=50),
    block_cache=DBCacheConfig(),
)
```

### Video Control Config

For VACE and advanced video generation:

```python
from kdit.config import KsanaSLGConfig, KsanaFETAConfig, KsanaExperimentalConfig, KsanaVideoControlConfig

video_control = KsanaVideoControlConfig(
    slg=KsanaSLGConfig(blocks=[9], start_percent=0.1, end_percent=1.0),
    feta=KsanaFETAConfig(weight=2.0, start_percent=0.0, end_percent=1.0),
    experimental=KsanaExperimentalConfig(cfg_zero_star=True),
)
```

## FP8 Models

kDiT supports FP8 quantized models for reduced memory usage:

```python
model_config = ModelConfig(
    run_dtype=torch.float16,
    attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN),
    linear_backend=KsanaLinearBackend.FP8_GEMM,
    torch_compile_config=KsanaTorchCompileConfig(),
)

pipeline = Pipeline.from_models(
    (high_noise_model_path, low_noise_model_path),
    text_checkpoint_dir=text_dir,
    vae_checkpoint_dir=vae_dir,
    model_config=model_config,
)
```

## Multi-GPU Inference

### Method 1: Auto-detect GPUs

```python
from kdit.utils.distribute import get_gpu_count

pipeline = Pipeline.from_models(
    model_path,
    dist_config=DistributedConfig(num_gpus=get_gpu_count()),
)
```

### Method 2: torchrun

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 your_script.py
```

### Method 3: Environment Variable

```bash
CUDA_VISIBLE_DEVICES=0,1 python your_script.py
```

## Lightning / Fast Generation

Use LoRA models for 4-step fast generation:

```python
from kdit.config import LoraConfig, KsanaTorchCompileConfig

pipeline = Pipeline.from_models(
    "/path/to/Wan2.2-T2V-A14B",
    model_config=ModelConfig(
        run_dtype=torch.float16,
        torch_compile_config=KsanaTorchCompileConfig(mode="max-autotune-no-cudagraphs"),
        boundary=0.9,
    ),
    lora_config=LoraConfig("/path/to/Wan2.2-Lightning/4steps-lora"),
)

video = pipeline.generate(
    "your prompt",
    sample_config=SampleConfig(
        steps=4, cfg_scale=1.0, shift=5.0, solver=SolverType.EULER,
        sigmas=[1.0, 0.9375001, 0.6333333, 0.225, 0.0],
    ),
    runtime_config=RuntimeConfig(seed=1234, size=(1280, 720), frame_num=81),
    cache_config=CustomStepCacheConfig(steps=3, scales=1.1),
)
```

## Turbo Diffusion (Fast I2V)

Use Turbo Diffusion models for 4-step fast I2V:

```python
from kdit.config import KsanaSageSLAConfig

sage_sla_config = KsanaSageSLAConfig(
    dense_attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN),
    topk=0.1,
)

pipeline = Pipeline.from_models(
    (high_model_path, low_model_path),
    text_checkpoint_dir=text_dir,
    vae_checkpoint_dir=vae_dir,
    model_config=ModelConfig(attention_config=sage_sla_config, run_dtype=torch.bfloat16),
)

video = pipeline.generate(
    "your prompt",
    extra_inputs=WanI2VExtraInputs(start_img_path="input.png"),
    sample_config=SampleConfig(steps=4, cfg_scale=1.0, shift=5.0, solver=SolverType.EULER),
    runtime_config=RuntimeConfig(size=(1280, 720), seed=1234, frame_num=81),
)
```
