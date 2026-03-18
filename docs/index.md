# kDiT

<div align="center">

**High-Performance DiT Inference Framework**

</div>

---

## Overview

kDiT is a high-performance inference framework for Diffusion Transformer (DiT) models, developed by Tencent. It features a 6-layer modular architecture designed for maximum throughput and flexibility.

## Key Features

- 🚀 **High Performance** — FP8 quantization, QKV fusion, multi-backend attention (FlashAttention, SageAttention, SDPA)
- 🔧 **Modular Architecture** — 6-layer design: Interaction → Execution → Unit → Cache → Operator → Hardware
- 🎨 **ComfyUI Integration** — 32 custom nodes for visual workflow design
- 🐍 **Python API** — Simple `Pipeline` API for local usage and server deployment
- 📦 **Multi-Model Support** — Wan, Qwen, VACE and more
- ⚡ **Distributed Inference** — Ray-based multi-GPU support with FSDP
- 🔌 **NPU Compatible** — Huawei Ascend NPU support via `torch_npu`

## Quick Start

```python
from kdit import Pipeline

# Load pipeline from model path (auto-detects model type)
pipeline = Pipeline.from_models(
    diffusion_model_path="path/to/model",
    text_encoder_path="path/to/text_encoder",
    vae_path="path/to/vae",
)

# Generate
result = pipeline.generate(prompt="a beautiful sunset over the ocean")
```

## Two Usage Modes

### 🐍 Local Python API

Use kDiT directly in Python scripts with the `Pipeline` API. Best for batch processing, automation, server-side deployment, and custom pipeline development.

📖 [Local Usage Guide](guide/local-usage.md)

### 🎨 ComfyUI Integration

Use kDiT as a ComfyUI custom node plugin with visual workflow design. Best for interactive prototyping, non-programmer friendly usage, and community workflow sharing.

📖 [ComfyUI Usage Guide](guide/comfyui-usage.md)

## Documentation

| Document | Link |
|----------|------|
| Local Usage Guide | [local-usage.md](guide/local-usage.md) |
| ComfyUI Usage Guide | [comfyui-usage.md](guide/comfyui-usage.md) |
| ComfyUI Node Reference | [comfyui-nodes.md](guide/comfyui-nodes.md) |
| Supported Models | [supported-models.md](guide/supported-models.md) |
| Architecture Overview | [architecture.md](architecture.md) |
| API Reference | [api/](api/index.md) |
