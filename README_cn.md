# KsanaDiT

<div align="center">

**高性能视频/图像生成 DiT (Diffusion Transformer) 推理框架**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

[English](README.md) | [简体中文](README_zh.md)

</div>

## 📖 简介

KsanaDiT 是一个专为扩散模型（Diffusion Transformer）设计的高性能推理框架，支持视频生成（T2V/I2V）和图像生成（T2I）任务。框架提供了丰富的优化技术和灵活的配置选项，可在单卡或多卡环境下高效运行大规模 DiT 模型。

### ✨ 核心特性

- 🚀 **高性能推理**: 支持 FP8 量化、QKV Fuse、Torch Compile、以及多种注意力机制优化
- 🎯 **多注意力后端支持**: SLA Attention、Flash Attention、Sage Attention、Radial Sage Attention、Torch SDPA
- 🎬 **多模态生成**: 支持文本生成视频（T2V）、图像生成视频（I2V）、视频可控编辑（Vace）、文本生成图像（T2I）
- 💾 **智能缓存**: 内置多种缓存策略（DBCache、EasyCache(WIP)、MagCache(WIP)、TeaCache(WIP)、CustomStepCache、HybridCache）
- 🔧 **灵活配置**: 支持 LoRA、多种采样器（Euler、UniPC、DPM++）、自定义 Sigma 调度
- 🌐 **分布式支持**: 支持单卡、多卡（torchrun）、Ray 分布式推理、Model Pool 管理
- 🔌 **ComfyUI 集成**: 提供 ComfyUI 节点支持（支持独立submodule），方便可视化工作流
- 🖥️ **多平台支持**: GPU、NPU，XPU(WIP)

## 📦 支持的模型

### 视频生成模型

| 模型 | 类型 | 参数量 | 支持任务 | 状态 |
|------|------|--------|---------|------|
| Turbo Diffusion | 图像生成视频 | 14B | I2V | ✅ |
| Wan2.2-T2V | 文本生成视频 | 5B/14B | T2V | ✅ |
| Wan2.2-I2V | 图像生成视频 | 14B | I2V | ✅ |
| Wan2.1-Vace | 视频可控编辑 | 14B | Vace | ✅ |


### 图像生成模型

| 模型 | 类型 | 参数量 | 支持任务 | 状态 |
|------|------|--------|---------|------|
| Qwen-Image | 文本生成图像 | 20B | T2I | ✅ |

## 🛠️ 安装

### Docker

我们正在抓紧制作Dockerfile，敬请期待。

### 环境要求

- **Python**: >= 3.10, < 4.0
- **PyTorch**: >= 2.0
- **GPU 环境**:
  - CUDA >= 12.8
  - 推荐显卡: NVIDIA
- **NPU 环境**:
  - CANN >= 8.0
  - torch_npu 适配层

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/Tencent/KsanaDiT.git
cd KsanaDiT

# 默认安装基础依赖GPU版本
pip install -e .
```

### GPU 加速安装

```bash
# 安装 GPU 优化依赖（推荐）
pip install -e ".[gpu]"

# 或手动安装
pip install xformers>=0.0.29 flash-attn>=2.6.0 triton>=3.2.0
```

### NPU 环境安装

```bash
# 1. 安装 CANN 工具包（参考官方文档）
# https://www.hiascend.com/software/cann

# 2. 安装 torch_npu
pip install torch-npu

# 3. 安装 KsanaDiT（NPU 版本）
pip install -e ".[npu]"

# 4. 验证 NPU 环境
python -c "import torch_npu; print(torch_npu.npu.is_available())"
```

### 发布安装

直接通过whl包安装，敬请期待。


## 🚀 快速开始
详细的代码可以参考[examples](./examples/)

### 文本生成视频 (T2V)

```python
import torch
from ksana import KsanaPipeline
from ksana.config import (
    KsanaDistributedConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)

# 创建推理管道
pipeline = KsanaPipeline.from_models(
    "path/to/Wan2.2-T2V-A14B",
    dist_config=KsanaDistributedConfig(num_gpus=1)
)

# 生成视频
video = pipeline.generate(
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景",
    sample_config=KsanaSampleConfig(steps=40),
    runtime_config=KsanaRuntimeConfig(
        seed=1234,
        size=(720, 480),
        frame_num=17,
        return_frames=True,
    ),
)

print(f"生成视频形状: {video.shape}")
```

### 图像生成视频 (I2V)

```python
from ksana import KsanaPipeline
from ksana.config import KsanaRuntimeConfig, KsanaSampleConfig

pipeline = KsanaPipeline.from_models("path/to/Wan2.2-I2V-A14B")

video = pipeline.generate(
    "女孩扇子轻微挥动，吹口仙气后，手上的闪电飞到空中开始打雷",
    img_path="input.png",
    sample_config=KsanaSampleConfig(steps=40),
    runtime_config=KsanaRuntimeConfig(
        seed=1234,
        size=(512, 512),
        frame_num=17,
    ),
)
```

#### TurboDiffusion

参考 [run_turbo_diffusion](./examples/wan/wan2_2_i2v.py:L115)

### 文本生成图像 (T2I)

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
    "一只可爱的橘猫坐在窗台上，阳光透过窗户洒在它的毛发上",
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

## 🎯 高级功能

### FP8 量化推理

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

### LoRA 加速推理

```python
from ksana import KsanaPipeline
from ksana.config import KsanaLoraConfig, KsanaSampleConfig

pipeline = KsanaPipeline.from_models(
    "path/to/Wan2.2-T2V-A14B",
    lora_config=KsanaLoraConfig("path/to/Wan2.2-Lightning-4steps-lora"),
)

# 使用 4 步快速生成
video = pipeline.generate(
    prompt,
    sample_config=KsanaSampleConfig(
        steps=4,
        cfg_scale=1.0,
        sigmas=[1.0, 0.9375, 0.6333, 0.225, 0.0],
    ),
)
```

### 智能缓存优化 - *持续优化中*

```python
from ksana.config.cache_config import (
    DCacheConfig,
    DBCacheConfig,
    KsanaHybridCacheConfig,
)

# 使用混合缓存策略
cache_config = KsanaHybridCacheConfig(
    step_cache=DCacheConfig(fast_degree=50),
    block_cache=DBCacheConfig(),
)

video = pipeline.generate(
    prompt,
    cache_config=cache_config,
)
```

### 多 GPU 分布式推理

```bash
# 方式 1: 使用 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py

# 方式 2: 使用 torchrun
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

## 📊 性能优化技术

### 量化与计算优化

| 技术 | 说明 | 效果 |
|------|------|------|
| FP8 GEMM | FP8 量化矩阵乘法 | 显存降低，速度提升 |
| Torchao FP8 Dynamic | 动态 FP8 量化 | 自适应精度，平衡质量与性能 |
| QKV Fuse | QKV 投影融合 | 减少内存访问，提升吞吐 |
| torch.compile | 图编译优化 | 端到端加速 10-30% |

### 注意力机制后端

| 后端 | 特点 | 适用场景 |
|------|------|---------|
| Flash Attention | 高性能、内存高效 | 通用推荐 |
| Sage Attention | 优化的注意力计算 | 长序列 |
| Radial Sage Attention | 径向稀疏注意力 | 超长序列 |
| Torch SDPA | PyTorch 原生实现 | 兼容性优先 |

### 缓存策略

EasyCache，MagCache，TeaCache *持续适配优化中*


### 采样器

| 采样器 | 说明 | 适用场景 |
|--------|------|---------|
| Euler | 快速采样 | 4-8 步推理 |
| UniPC | 高质量采样 | 20-40 步推理 |
| DPM++ | 高效多步采样 | 通用场景 |
| Turbo Diffusion | 极速采样 | 4 步推理 |
| FlowMatch Euler | 流匹配采样 | 图像生成 |

## 🔧 配置说明

### 环境变量

```bash
# 日志级别: debug/info/warn/error
export KSANA_LOGGER_LEVEL=info

```


### 模型配置

框架支持通过 YAML 文件配置模型参数，配置文件位于 [`ksana/settings/`](ksana/settings/) 目录：

- [`qwen/t2i_20b.yaml`](ksana/settings/qwen/t2i_20b.yaml) - Qwen 图像生成模型配置
- [`wan/t2v_14b.yaml`](ksana/settings/wan/t2v_14b.yaml) - Wan2.2 T2V 模型配置
- [`wan/i2v_14b.yaml`](ksana/settings/wan/i2v_14b.yaml) - Wan2.2 I2V 模型配置
- [`wan/vace_14b.yaml`](ksana/settings/wan/vace_14b.yaml) - Wan2.1 Vace 模型配置

## 📚 示例代码

完整示例代码位于 [`examples/`](examples/) 目录：

- [`examples/wan/wan2_2_t2v.py`](examples/wan/wan2_2_t2v.py) - 文本生成视频示例
- [`examples/wan/wan2_2_i2v.py`](examples/wan/wan2_2_i2v.py) - 图像生成视频示例
- [`examples/wan/wan2_1_vace.py`](examples/wan/wan2_1_vace.py) - 视频可控编辑示例
- [`examples/qwen/qwen_image_t2i.py`](examples/qwen/qwen_image_t2i.py) - 文本生成图像示例

## 🧪 测试

我们配置了完备的测试，目前耗时较久，后续我们会持续精简，仅限开发者关注。

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/ksana/pipelines/wan2_2_t2v_test.py

# 运行 GPU 测试
bash scripts/ci_tests/ci_ksana_gpus.sh
```

## 🤝 贡献指南

我们欢迎社区贡献！在提交 PR 之前，请确保：

1. 代码通过所有测试
2. 遵循项目代码风格（使用 `black` 和 `ruff`）
3. 添加必要的文档和注释
4. 更新相关的 README 和示例

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行代码风格检查
pre-commit run --all-files

# 运行测试
pytest tests/
```

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

本项目受益于以下优秀开源项目：

- [Wan-Video](https://github.com/Wan-Video/Wan2.2) - Wan2.2 视频生成模型
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) - ComfyUI 集成参考
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) - 视频生成优化技术
- [Nunchaku](https://github.com/nunchaku-tech/nunchaku) - 量化优化方案
- [TurborDiffusion](https://github.com/thu-ml/TurboDiffusion) - 推理加速方案

## 📮 联系方式

- 问题反馈: [GitHub Issues](https://github.com/tencent/KsanaDiT/issues)
- 功能建议: [GitHub Discussions](https://github.com/tencent/KsanaDiT/discussions)

## 🗺️ 路线图

### 已完成 ✅

- [x] **多平台支持**: NPU后端支持
- [x] **批量推理**: 支持 batch size > 1，合并 cond/uncond
- [x] **视频编辑**: Wan2.1 Vace 视频可控编辑
- [x] **高级采样器**: DPM++、Turbo Diffusion 支持
- [x] **性能优化**: QKV Fuse + Dynamic FP8 优化
- [x] **内存优化**: Pin Manager 解决 OOM 问题

### 进行中 🚧

- [ ] 支持更多生成模型（Qwen Image Edit, Z-Image，Hunyuan 等）
- [ ] 优化内存占用，支持更长视频生成
- [ ] 更多Cache方案
- [ ] 模型量化工具链
- [ ] 支持更多后端硬件


---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️ Star！**

Made with ❤️ by KsanaDiT Team

</div>
