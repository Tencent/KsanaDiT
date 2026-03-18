# kDiT

<div align="center">

**高性能 DiT 推理框架**

</div>

---

## 概述

kDiT 是腾讯开发的高性能 Diffusion Transformer (DiT) 推理框架，采用 6 层模块化架构设计，追求极致吞吐和灵活性。

## 核心特性

- 🚀 **高性能** — FP8 量化、QKV 融合、多后端注意力（FlashAttention、SageAttention、SDPA）
- 🔧 **模块化架构** — 6 层设计：交互层 → 执行层 → 单元层 → 缓存层 → 算子层 → 硬件层
- 🎨 **ComfyUI 集成** — 32 个自定义节点，支持可视化工作流设计
- 🐍 **Python API** — 简洁的 `Pipeline` API，支持本地使用和服务端部署
- 📦 **多模型支持** — Wan、Qwen、VACE 等
- ⚡ **分布式推理** — 基于 Ray 的多 GPU 支持，配合 FSDP
- 🔌 **NPU 兼容** — 通过 `torch_npu` 支持华为昇腾 NPU

## 快速开始

```python
from kdit import Pipeline

# 从模型路径加载 Pipeline（自动检测模型类型）
pipeline = Pipeline.from_models(
    diffusion_model_path="path/to/model",
    text_encoder_path="path/to/text_encoder",
    vae_path="path/to/vae",
)

# 生成
result = pipeline.generate(prompt="a beautiful sunset over the ocean")
```

## 两种使用模式

### 🐍 本地 Python API

通过 `Pipeline` API 在 Python 脚本中直接使用 kDiT。适用于批量处理、自动化、服务端部署和自定义 Pipeline 开发。

📖 [本地使用指南](guide/local-usage.md)

### 🎨 ComfyUI 集成

作为 ComfyUI 自定义节点插件使用，支持可视化工作流设计。适用于交互式原型设计、非程序员友好使用和社区工作流分享。

📖 [ComfyUI 使用指南](guide/comfyui-usage.md)

## 文档导航

| 文档 | 链接 |
|------|------|
| 本地使用指南 | [local-usage.md](guide/local-usage.md) |
| ComfyUI 使用指南 | [comfyui-usage.md](guide/comfyui-usage.md) |
| ComfyUI 节点参考 | [comfyui-nodes.md](guide/comfyui-nodes.md) |
| 支持的模型 | [supported-models.md](guide/supported-models.md) |
| 架构概览 | [architecture.md](architecture.md) |
| API 参考 | [api/](api/index.md) |
