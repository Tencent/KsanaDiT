# 本地使用指南

[English Version](local-usage.md)

## 概述

kDiT 提供以 `Pipeline` 类为核心的 Python API，用于本地推理。只需几行代码即可生成视频和图像。

## 快速开始

### 最简示例 — 文生视频

```python
from kdit import Pipeline
from kdit.config import RuntimeConfig, SampleConfig

pipeline = Pipeline.from_models("/path/to/Wan2.2-T2V-A14B")

video = pipeline.generate(
    "街头摄影，戴耳机的酷女孩滑板，纽约街头",
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

### 最简示例 — 文生图

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
    "一只可爱的橘猫坐在窗台上，阳光透过窗户洒在它的毛发上",
    prompt_negative=" ",
    sample_config=SampleConfig(steps=20, cfg_scale=4.0, solver=SolverType.FLOWMATCH_EULER),
    runtime_config=RuntimeConfig(seed=42, size=(1024, 1024)),
)
save_image(image, "output.png")
```

## 支持的任务类型

### 1. Wan2.2 文生视频 (T2V)

使用 Wan2.2-T2V 模型从文本提示生成视频。

```python
pipeline = Pipeline.from_models("/path/to/Wan2.2-T2V-A14B")
video = pipeline.generate(
    "你的提示词",
    sample_config=SampleConfig(steps=40),
    runtime_config=RuntimeConfig(seed=1234, size=(720, 480), frame_num=81),
)
```

**示例代码**: [`examples/local/wan/wan2_2_t2v.py`](../examples/local/wan/wan2_2_t2v.py)

### 2. Wan2.2 图生视频 (I2V)

从输入图像和文本提示生成视频。

```python
from kdit.pipelines.context_builders.wan import WanI2VExtraInputs

pipeline = Pipeline.from_models("/path/to/Wan2.2-I2V-A14B")
video = pipeline.generate(
    "你的提示词",
    extra_inputs=WanI2VExtraInputs(start_img_path="input.png"),
    sample_config=SampleConfig(steps=40),
    runtime_config=RuntimeConfig(seed=1234, size=(1280, 720), frame_num=81),
)
```

支持起始图像、结束图像、以及起始+结束图像模式：
```python
video = pipeline.generate(
    "你的提示词",
    extra_inputs=WanI2VExtraInputs(
        start_img_path="start.png",
        end_img_path="end.png",  # 可选
    ),
    ...
)
```

**示例代码**: [`examples/local/wan/wan2_2_i2v.py`](../examples/local/wan/wan2_2_i2v.py)

### 3. Wan2.1 VACE（视频可控编辑）

使用控制视频和参考图像引导生成或编辑视频。

```python
from kdit.config import DCacheConfig, KsanaSLGConfig, KsanaFETAConfig, KsanaVideoControlConfig, SolverType
from kdit.utils import load_control_frames
from kdit.utils.vace import VaceConfig

pipeline = Pipeline.from_models("/path/to/Wan2.1-VACE-14B")

# 使用参考图像
reference_image = load_control_frames("reference.png", max_frames=1, target_size=(512, 512))
video_control_config = VaceConfig(reference_image=reference_image, strength=1.0)

video = pipeline.generate(
    "你的提示词",
    prompt_negative="低质量...",
    sample_config=SampleConfig(steps=30, cfg_scale=6.0, shift=8.0, solver=SolverType.UNI_PC),
    runtime_config=RuntimeConfig(seed=1234, size=(512, 512), frame_num=81),
    cache_config=[DCacheConfig()],
    video_control_config=video_control_config,
)
```

**示例代码**: [`examples/local/wan/wan2_1_vace.py`](../examples/local/wan/wan2_1_vace.py)

### 4. Qwen-Image 文生图 (T2I)

从文本提示生成图像。

```python
pipeline = Pipeline.from_models(
    "/path/to/Qwen-Image",
    model_config=ModelConfig(run_dtype=torch.bfloat16),
    offload_device="cpu",
)
image = pipeline.generate(
    "你的提示词",
    prompt_negative=" ",
    sample_config=SampleConfig(steps=20, cfg_scale=4.0, solver=SolverType.FLOWMATCH_EULER),
    runtime_config=RuntimeConfig(seed=42, size=(1024, 1024)),
)
```

**示例代码**: [`examples/local/qwen/qwen_image_t2i.py`](../examples/local/qwen/qwen_image_t2i.py)

### 5. Qwen-Image 图像编辑

使用文本指令和参考图像进行图像编辑。

```python
from kdit.config import LoraConfig
from kdit.models.model_key import ModelKey

pipeline = Pipeline.from_models(
    "/path/to/Qwen-Image-Edit",
    model_config=ModelConfig(run_dtype=torch.bfloat16),
    pipeline_key=ModelKey.QwenImage_Edit,
    lora_config=LoraConfig(path="/path/to/lora.safetensors", strength=1.0),  # 可选
    offload_device="cpu",
)
image = pipeline.generate(
    "两个人拥抱在一起",
    prompt_negative="模糊, 畸形",
    img_path=["image1.png", "image2.png"],
    sample_config=SampleConfig(steps=40, cfg_scale=4.0, solver=SolverType.FLOWMATCH_EULER),
    runtime_config=RuntimeConfig(seed=321, size=(1024, 1024)),
)
```

**示例代码**: [`examples/local/qwen/qwen_image_edit.py`](../examples/local/qwen/qwen_image_edit.py)

## 配置参考

### ModelConfig

控制模型加载和计算设置。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `run_dtype` | `torch.dtype` | `torch.float16` | 运行时数据类型（float16, bfloat16） |
| `attention_config` | `KsanaAttentionConfig` | Flash Attention | 注意力后端配置 |
| `linear_backend` | `KsanaLinearBackend` | `DEFAULT` | 线性层后端（DEFAULT, FP8_GEMM） |
| `torch_compile_config` | `KsanaTorchCompileConfig` | `None` | torch.compile 设置 |
| `boundary` | `float` | `0.875` | 高/低噪声模型边界值 |

### SampleConfig

控制采样/去噪过程。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `steps` | `int` | `20` | 去噪步数 |
| `cfg_scale` | `float` | `4.0` | 无分类器引导（CFG）强度 |
| `shift` | `float` | `5.0` | 噪声调度偏移 |
| `solver` | `SolverType` | `UNI_PC` | 求解器算法（UNI_PC, EULER, DPM_PLUS_PLUS, FLOWMATCH_EULER） |
| `sigmas` | `list[float]` | `None` | 自定义 sigma 调度 |
| `video_control` | `KsanaVideoControlConfig` | `None` | 视频控制配置（SLG, FETA, Experimental） |

### RuntimeConfig

控制运行时行为和输出设置。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | `int` | `42` | 随机种子 |
| `size` | `tuple[int, int]` | `(1280, 720)` | 输出尺寸（宽, 高） |
| `frame_num` | `int` | `81` | 视频帧数 |
| `return_frames` | `bool` | `False` | 以张量形式返回帧 |
| `output_folder` | `str` | `None` | 输出目录 |
| `save_output` | `bool` | `False` | 自动保存输出 |
| `rope_function` | `str` | `"default"` | RoPE 实现方式（"default" 或 "comfy"） |

### DistributedConfig

控制多卡分布式推理。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_gpus` | `int` | `1` | 使用的 GPU 数量 |

### LoraConfig

控制 LoRA 模型加载。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str` | — | LoRA 权重路径 |
| `strength` | `float` | `1.0` | LoRA 合并强度 |

### 缓存配置

kDiT 提供多种缓存策略以加速推理：

| 缓存类型 | 类名 | 说明 |
|----------|------|------|
| DCache | `DCacheConfig` | 步级缓存，支持快/慢度数控制 |
| DBCache | `DBCacheConfig` | 块级缓存，支持 TaylorSeer 预测 |
| TeaCache | `TeaCacheConfig` | 基于阈值的步级缓存，内置预设 |
| EasyCache | `EasyCacheConfig` | 简单阈值缓存 |
| MagCache | `MagCacheConfig` | 基于幅度的缓存，支持保留比率 |
| CustomStepCache | `CustomStepCacheConfig` | 用户自定义步级缓存 |
| HybridCache | `HybridCacheConfig` | 混合缓存（step_cache + block_cache） |

示例 — HybridCache：
```python
from kdit.config.cache_config import DCacheConfig, DBCacheConfig, HybridCacheConfig

cache_config = HybridCacheConfig(
    step_cache=DCacheConfig(fast_degree=50),
    block_cache=DBCacheConfig(),
)
```

### 视频控制配置

用于 VACE 和高级视频生成：

```python
from kdit.config import KsanaSLGConfig, KsanaFETAConfig, KsanaExperimentalConfig, KsanaVideoControlConfig

video_control = KsanaVideoControlConfig(
    slg=KsanaSLGConfig(blocks=[9], start_percent=0.1, end_percent=1.0),
    feta=KsanaFETAConfig(weight=2.0, start_percent=0.0, end_percent=1.0),
    experimental=KsanaExperimentalConfig(cfg_zero_star=True),
)
```

## FP8 模型

kDiT 支持 FP8 量化模型以减少显存占用：

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

## 多卡推理

### 方式一：自动检测 GPU 数量

```python
from kdit.utils.distribute import get_gpu_count

pipeline = Pipeline.from_models(
    model_path,
    dist_config=DistributedConfig(num_gpus=get_gpu_count()),
)
```

### 方式二：torchrun

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 your_script.py
```

### 方式三：环境变量

```bash
CUDA_VISIBLE_DEVICES=0,1 python your_script.py
```

## Lightning / 快速生成

使用 LoRA 模型实现 4 步快速生成：

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
    "你的提示词",
    sample_config=SampleConfig(
        steps=4, cfg_scale=1.0, shift=5.0, solver=SolverType.EULER,
        sigmas=[1.0, 0.9375001, 0.6333333, 0.225, 0.0],
    ),
    runtime_config=RuntimeConfig(seed=1234, size=(1280, 720), frame_num=81),
    cache_config=CustomStepCacheConfig(steps=3, scales=1.1),
)
```

## Turbo Diffusion（快速 I2V）

使用 Turbo Diffusion 模型实现 4 步快速图生视频：

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
    "你的提示词",
    extra_inputs=WanI2VExtraInputs(start_img_path="input.png"),
    sample_config=SampleConfig(steps=4, cfg_scale=1.0, shift=5.0, solver=SolverType.EULER),
    runtime_config=RuntimeConfig(size=(1280, 720), seed=1234, frame_num=81),
)
```
