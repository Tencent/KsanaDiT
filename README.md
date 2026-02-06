# KsanaDiT

<div align="center">

**High-Performance DiT (Diffusion Transformer) Inference Framework for Video & Image Generation**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

[English](README.md) | [简体中文](README_cn.md)

</div>

## 📖 Introduction

KsanaDiT is a high-performance inference framework specifically designed for Diffusion Transformers (DiT), supporting video generation (T2V/I2V) and image generation (T2I) tasks. The framework provides a rich set of optimization techniques and flexible configuration options, enabling efficient execution of large-scale DiT models on single or multi-GPU environments.

### ✨ Key Features

- 🚀 **High-Performance Inference**: FP8 quantization, QKV Fuse, Torch Compile, and various attention optimizations
- 🎯 **Multiple Attention Backends**: SLA Attention, Flash Attention, Sage Attention, Radial Sage Attention, Torch SDPA
- 🎬 **Multi-Modal Generation**: Text-to-Video (T2V), Image-to-Video (I2V), Video Controllable Editing (Vace), Text-to-Image (T2I)
- 💾 **Smart Caching**: Built-in caching strategies (DBCache, EasyCache (WIP), MagCache (WIP), TeaCache (WIP), CustomStepCache, HybridCache)
- 🔧 **Flexible Configuration**: LoRA support, multiple samplers (Euler, UniPC, DPM++), custom sigma scheduling
- 🌐 **Distributed Support**: Single-GPU, multi-GPU (torchrun), Ray distributed inference, Model Pool management
- 🔌 **ComfyUI Integration**: ComfyUI node support (standalone submodule) for visual workflow design
- 🖥️ **Multi-Platform Support**: GPU, NPU, XPU (WIP)

## 📦 Supported Models

### Video Generation Models

| Model | Type | Parameters | Tasks | Status |
|-------|------|------------|-------|--------|
| Turbo Diffusion | Image-to-Video | 14B | I2V | ✅ |
| Wan2.2-T2V | Text-to-Video | 5B/14B | T2V | ✅ |
| Wan2.2-I2V | Image-to-Video | 14B | I2V | ✅ |
| Wan2.1-Vace | Video Controllable Editing | 14B | Vace | ✅ |

### Image Generation Models

| Model | Type | Parameters | Tasks | Status |
|-------|------|------------|-------|--------|
| Qwen-Image | Text-to-Image | 20B | T2I | ✅ |

## 🛠️ Installation

### Docker

We are actively working on Dockerfiles. Stay tuned!

### Requirements

- **Python**: >= 3.10, < 4.0
- **PyTorch**: >= 2.0
- **GPU Environment**:
  - CUDA >= 12.8
  - Recommended: NVIDIA GPUs
- **NPU Environment**:
  - CANN >= 8.0
  - torch_npu adapter

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/Tencent/KsanaDiT.git
cd KsanaDiT

# Install base dependencies (GPU version by default)
pip install -e .
```

### GPU Accelerated Installation

```bash
# Install GPU optimization dependencies (recommended)
pip install -e ".[gpu]"

# Or install manually
pip install xformers>=0.0.29 flash-attn>=2.6.0 triton>=3.2.0
```

### NPU Environment Installation

```bash
# 1. Install CANN toolkit (refer to official documentation)
# https://www.hiascend.com/software/cann

# 2. Install torch_npu
pip install torch-npu

# 3. Install KsanaDiT (NPU version)
pip install -e ".[npu]"

# 4. Verify NPU environment
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

### Release Installation

Direct installation via wheel packages coming soon.

## 🔌 Interface Support

KsanaDiT provides multiple usage methods to meet different scenario requirements:

### Local Pipeline Mode

Run locally through the Python Pipeline API, suitable for scripted batch generation or integration into your own systems:

```python
from ksana import KsanaPipeline

# Create inference pipeline
pipeline = KsanaPipeline.from_models("path/to/model")

# Generate video/image
result = pipeline.generate(prompt, ...)
```

For detailed usage, refer to [Quick Start](#-quick-start) and the [examples](./examples/) directory.

### ComfyUI Integration

KsanaDiT supports usage as ComfyUI custom nodes, providing a visual workflow experience:

```bash
# 1. Navigate to ComfyUI's custom_nodes directory
cd /path/to/ComfyUI/custom_nodes

# 2. Clone the KsanaDiT repository
git clone https://github.com/Tencent/KsanaDiT.git

# 3. Enter the KsanaDiT directory and install dependencies
cd KsanaDiT
./scripts/install_dev.sh
```

After installation, restart ComfyUI and you will see KsanaDiT-related nodes in the node list. For more ComfyUI usage instructions, refer to [comfyui/README.md](./comfyui/README.md).

## 🚀 Quick Start

For detailed code examples, refer to [examples](./examples/).

### Text-to-Video (T2V)

```python
import torch
from ksana import KsanaPipeline
from ksana.config import (
    KsanaDistributedConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)

# Create inference pipeline
pipeline = KsanaPipeline.from_models(
    "path/to/Wan2.2-T2V-A14B",
    dist_config=KsanaDistributedConfig(num_gpus=1)
)

# Generate video
video = pipeline.generate(
    "Street photography, cool girl with headphones skateboarding, New York streets, graffiti wall background",
    sample_config=KsanaSampleConfig(steps=40),
    runtime_config=KsanaRuntimeConfig(
        seed=1234,
        size=(720, 480),
        frame_num=17,
        return_frames=True,
    ),
)

print(f"Generated video shape: {video.shape}")
```

### Image-to-Video (I2V)

```python
from ksana import KsanaPipeline
from ksana.config import KsanaRuntimeConfig, KsanaSampleConfig

pipeline = KsanaPipeline.from_models("path/to/Wan2.2-I2V-A14B")

video = pipeline.generate(
    "Girl gently waves her fan, blows a breath of fairy air, lightning flies from her hand into the sky and thunder begins",
    img_path="input.png",
    sample_config=KsanaSampleConfig(steps=40),
    runtime_config=KsanaRuntimeConfig(
        seed=1234,
        size=(512, 512),
        frame_num=17,
    ),
)
```

#### Turbo Diffusion

See [run_turbo_diffusion](./examples/wan/wan2_2_i2v.py#L115)

### Text-to-Image (T2I)

```python
import torch
from ksana import KsanaPipeline
from ksana.config import (
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)

pipeline = KsanaPipeline.from_models(
    "path/to/Qwen-Image",
    model_config=KsanaModelConfig(run_dtype=torch.bfloat16),
)

image = pipeline.generate(
    "A cute orange cat sitting on a windowsill, sunlight streaming through the window onto its fur",
    sample_config=KsanaSampleConfig(
        steps=20,
        cfg_scale=4.0,
        solver=KsanaSolverType.FLOWMATCH_EULER,
    ),
    runtime_config=KsanaRuntimeConfig(
        seed=42,
        size=(1024, 1024),
    ),
)
```

## 🎯 Advanced Features

### FP8 Quantized Inference

```python
import torch
from ksana import KsanaPipeline
from ksana.config import (
    KsanaModelConfig,
    KsanaAttentionConfig,
    KsanaAttentionBackend,
    KsanaLinearBackend,
)

model_config = KsanaModelConfig(
    run_dtype=torch.float16,
    attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN),
    linear_backend=KsanaLinearBackend.FP8_GEMM,
)

pipeline = KsanaPipeline.from_models(
    ("high_noise_fp8.safetensors", "low_noise_fp8.safetensors"),
    model_config=model_config,
)
```

### LoRA Accelerated Inference

```python
from ksana import KsanaPipeline
from ksana.config import KsanaLoraConfig, KsanaSampleConfig

pipeline = KsanaPipeline.from_models(
    "path/to/Wan2.2-T2V-A14B",
    lora_config=KsanaLoraConfig("path/to/Wan2.2-Lightning-4steps-lora"),
)

# Fast generation with 4 steps
video = pipeline.generate(
    prompt,
    sample_config=KsanaSampleConfig(
        steps=4,
        cfg_scale=1.0,
        sigmas=[1.0, 0.9375, 0.6333, 0.225, 0.0],
    ),
)
```

### Smart Cache Optimization - *Under Active Development*

```python
from ksana.config.cache_config import (
    DCacheConfig,
    DBCacheConfig,
    KsanaHybridCacheConfig,
)

# Use hybrid caching strategy
cache_config = KsanaHybridCacheConfig(
    step_cache=DCacheConfig(fast_degree=50),
    block_cache=DBCacheConfig(),
)

video = pipeline.generate(
    prompt,
    cache_config=cache_config,
)
```

### Multi-GPU Distributed Inference

```bash
# Method 1: Using CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py

# Method 2: Using torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 your_script.py
```

```python
from ksana import KsanaPipeline
from ksana.config import KsanaDistributedConfig

pipeline = KsanaPipeline.from_models(
    model_path,
    dist_config=KsanaDistributedConfig(num_gpus=4),
)
```

## 📊 Performance Optimization Techniques

### Quantization & Compute Optimization

| Technique | Description | Effect |
|-----------|-------------|--------|
| FP8 GEMM | FP8 quantized matrix multiplication | Reduced memory, improved speed |
| Torchao FP8 Dynamic | Dynamic FP8 quantization | Adaptive precision, balanced quality and performance |
| QKV Fuse | QKV projection fusion | Reduced memory access, improved throughput |
| torch.compile | Graph compilation optimization | 10-30% end-to-end speedup |

### Attention Backends

| Backend | Characteristics | Use Case |
|---------|-----------------|----------|
| Flash Attention | High performance, memory efficient | General recommendation |
| Sage Attention | Optimized attention computation | Long sequences |
| Radial Sage Attention | Radial sparse attention | Very long sequences |
| Torch SDPA | PyTorch native implementation | Compatibility priority |

### Caching Strategies

EasyCache, MagCache, TeaCache - *Under active optimization*

### Samplers

| Sampler | Description | Use Case |
|---------|-------------|----------|
| Euler | Fast sampling | 4-8 step inference |
| UniPC | High-quality sampling | 20-40 step inference |
| DPM++ | Efficient multi-step sampling | General purpose |
| Turbo Diffusion | Ultra-fast sampling | 4-step inference |
| FlowMatch Euler | Flow matching sampling | Image generation |

## 🔧 Configuration

### Environment Variables

```bash
# Log level: debug/info/warn/error
export KSANA_LOGGER_LEVEL=info
```

### Model Configuration

The framework supports model parameter configuration via YAML files, located in the [`ksana/settings/`](ksana/settings/) directory:

- [`qwen/t2i_20b.yaml`](ksana/settings/qwen/t2i_20b.yaml) - Qwen image generation model config
- [`wan/t2v_14b.yaml`](ksana/settings/wan/t2v_14b.yaml) - Wan2.2 T2V model config
- [`wan/i2v_14b.yaml`](ksana/settings/wan/i2v_14b.yaml) - Wan2.2 I2V model config
- [`wan/vace_14b.yaml`](ksana/settings/wan/vace_14b.yaml) - Wan2.1 Vace model config

## 📚 Code Examples

Complete example code is available in the [`examples/`](examples/) directory:

- [`examples/wan/wan2_2_t2v.py`](examples/wan/wan2_2_t2v.py) - Text-to-Video example
- [`examples/wan/wan2_2_i2v.py`](examples/wan/wan2_2_i2v.py) - Image-to-Video example
- [`examples/wan/wan2_1_vace.py`](examples/wan/wan2_1_vace.py) - Video controllable editing example
- [`examples/qwen/qwen_image_t2i.py`](examples/qwen/qwen_image_t2i.py) - Text-to-Image example

## 🧪 Testing

We have comprehensive test coverage. Tests are currently time-consuming; we will continue to streamline them. For developers only.

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/ksana/pipelines/wan2_2_t2v_test.py

# Run GPU tests
bash scripts/ci_tests/ci_ksana_gpus.sh
```

## 🤝 Contributing

We welcome community contributions! Before submitting a PR, please ensure:

1. Code passes all tests
2. Follows project code style (using `black` and `ruff`)
3. Includes necessary documentation and comments
4. Updates relevant README and examples

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code style checks
pre-commit run --all-files

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project benefits from the following excellent open-source projects:

- [Wan-Video](https://github.com/Wan-Video/Wan2.2) - Wan2.2 video generation model
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - ComfyUI integration reference
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) - Video generation optimization techniques
- [Nunchaku](https://github.com/nunchaku-tech/nunchaku) - Quantization optimization solutions
- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) - Inference acceleration solutions

## 📮 Contact

- Bug Reports: [GitHub Issues](https://github.com/tencent/KsanaDiT/issues)
- Feature Requests: [GitHub Discussions](https://github.com/tencent/KsanaDiT/discussions)

## 🗺️ Roadmap

### Completed ✅

- [x] **Multi-Platform Support**: NPU backend support
- [x] **Batch Inference**: Support for batch size > 1, merged cond/uncond
- [x] **Video Editing**: Wan2.1 Vace video controllable editing
- [x] **Advanced Samplers**: DPM++, Turbo Diffusion support
- [x] **Performance Optimization**: QKV Fuse + Dynamic FP8 optimization
- [x] **Memory Optimization**: Pin Manager to resolve OOM issues

### In Progress 🚧

- [ ] Support for more generation models (Qwen Image Edit, Z-Image, Hunyuan, etc.)
- [ ] Memory optimization for longer video generation
- [ ] Additional caching strategies
- [ ] Model quantization toolchain
- [ ] Support for more hardware backends

---

<div align="center">

**If this project helps you, please give us a ⭐️ Star!**

Made with ❤️ by the KsanaDiT Team

</div>
