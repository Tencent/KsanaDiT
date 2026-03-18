# ComfyUI 节点参考手册

[← 返回目录](README.md) | [English Version](comfyui-nodes.md)

本文档提供 kDiT 全部 31 个 ComfyUI 节点的完整参考。节点按 7 个功能类别组织。

---

## 目录

- [ComfyUI 节点参考手册](#comfyui-节点参考手册)
  - [目录](#目录)
  - [1. 核心节点](#1-核心节点)
    - [kDiT Model Loader — 模型加载器](#kdit-model-loader--模型加载器)
    - [kDiT Vace Model Select — VACE 模型选择](#kdit-vace-model-select--vace-模型选择)
    - [kDiT Generator — 生成器](#kdit-generator--生成器)
    - [kDiT EmptyLatent — 空潜空间](#kdit-emptylatent--空潜空间)
  - [2. VAE 节点](#2-vae-节点)
    - [kDiT VAE Loader — VAE 加载器](#kdit-vae-loader--vae-加载器)
    - [kDiT VAE Encoder — VAE 编码器](#kdit-vae-encoder--vae-编码器)
    - [kDiT VAE Image Encoder — VAE 图像编码器](#kdit-vae-image-encoder--vae-图像编码器)
    - [kDiT VAE Decoder — VAE 解码器](#kdit-vae-decoder--vae-解码器)
  - [3. 缓存节点](#3-缓存节点)
    - [kDiT HybridCache — 混合缓存](#kdit-hybridcache--混合缓存)
    - [kDiT CacheCombine — 缓存组合](#kdit-cachecombine--缓存组合)
    - [kDiT DCache](#kdit-dcache)
    - [kDiT DBCache](#kdit-dbcache)
    - [kDiT TeaCache](#kdit-teacache)
    - [kDiT EasyCache](#kdit-easycache)
    - [kDiT MagCache](#kdit-magcache)
    - [kDiT CustomStepCache — 自定义步骤缓存](#kdit-customstepcache--自定义步骤缓存)
  - [4. 注意力节点](#4-注意力节点)
    - [kDiT AttentionConfig — 注意力配置](#kdit-attentionconfig--注意力配置)
    - [kDiT RadialSageAttentionConfig — 径向稀疏注意力](#kdit-radialsageattentionconfig--径向稀疏注意力)
    - [kDiT SageSLAttentionConfig — Top-k 稀疏注意力](#kdit-sageslattentionconfig--top-k-稀疏注意力)
  - [5. LoRA 节点](#5-lora-节点)
    - [kDiT LoraSelect — LoRA 选择](#kdit-loraselect--lora-选择)
    - [kDiT LoraSelectMulti — 多 LoRA 选择](#kdit-loraselectmulti--多-lora-选择)
    - [kDiT LoraCombine — LoRA 组合](#kdit-loracombine--lora-组合)
  - [6. 视频控制节点](#6-视频控制节点)
    - [kDiT WanVace To Video — VACE 视频编码](#kdit-wanvace-to-video--vace-视频编码)
    - [kDiT Skip Layer Guidance — 跳层引导](#kdit-skip-layer-guidance--跳层引导)
    - [kDiT Enhance-A-Video — 视频增强](#kdit-enhance-a-video--视频增强)
    - [kDiT Experimental Args — 实验性参数](#kdit-experimental-args--实验性参数)
    - [kDiT VideoControlConfig — 视频控制配置](#kdit-videocontrolconfig--视频控制配置)
  - [7. 工具节点](#7-工具节点)
    - [kDiT TorchCompile — 编译优化](#kdit-torchcompile--编译优化)
    - [kDiT TextEmbConverter — 文本嵌入转换](#kdit-textembconverter--文本嵌入转换)
    - [kDiT DebugNode — 调试节点](#kdit-debugnode--调试节点)
    - [kDiT Empty Torch Cache — 清空显存缓存](#kdit-empty-torch-cache--清空显存缓存)
  - [节点汇总表](#节点汇总表)

---

## 1. 核心节点

### kDiT Model Loader — 模型加载器

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaModelLoaderNode` |
| **显示名称** | kDiT Model Loader |
| **分类** | kdit |
| **源码** | [`model_loader.py`](../kdit/adapter/comfyui/nodes/model_loader.py:59) |

加载扩散模型用于推理。这是在 ComfyUI 中搭建 kDiT 流水线的主要入口。支持双模型配置（高/低噪声）、注意力后端、线性后端、LoRA、VACE 模型和 torch.compile。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model_name` | DIFFUSION_MODELS | ✅ | — | 主扩散模型文件 |
| `run_dtype` | 枚举 | ❌ | `float16` | 运行精度：`float16`、`bfloat16` |
| `rms_dtype` | 枚举 | ❌ | `float` | RMSNorm 精度：`float`（fp32）或 `half`（fp16/bf16） |
| `linear_backend` | 枚举 | ❌ | `default` | 线性计算后端（如 `default`、`fp8_e4m3`） |
| `attention_config` | KSANA_ATTENTION_CONFIG | ❌ | None | 注意力后端配置 |
| `low_noise_model_name` | DIFFUSION_MODELS | ❌ | Empty | 低噪声模型（双模型配置） |
| `model_boundary` | FLOAT | ❌ | 0.875 | 高/低模型切换的时间步边界（0–1） |
| `torch_compile_args` | KSANA_TORCH_COMPILE | ❌ | None | Torch 编译配置 |
| `lora` | KSANA_LORA | ❌ | None | 要合并的 LoRA 权重 |
| `vace_model` | KSANA_VACE_MODEL | ❌ | None | VACE 视频条件模型 |

**输出：** `KSANA_DIFFUSION_MODEL` — 已加载的模型，可用于生成。

---

### kDiT Vace Model Select — VACE 模型选择

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaVaceModelSelectNode` |
| **显示名称** | kDiT Vace Model Select |
| **分类** | kdit |
| **源码** | [`model_loader.py`](../kdit/adapter/comfyui/nodes/model_loader.py:25) |

选择 VACE（视频-音频条件编码）模型文件，供模型加载器使用。支持可选的双模型（高/低噪声）配置。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `vace_model` | DIFFUSION_MODELS | ✅ | — | 主 VACE 模型文件 |
| `vace_low_model` | DIFFUSION_MODELS | ❌ | Empty | 低噪声 VACE 模型（双模型配置） |

**输出：** `KSANA_VACE_MODEL` — VACE 模型路径，供模型加载器使用。

---

### kDiT Generator — 生成器

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaGeneratorNode` |
| **显示名称** | kDiT Generator |
| **分类** | kdit |
| **源码** | [`generator.py`](../kdit/adapter/comfyui/nodes/generator.py:34) |

主采样/去噪节点。接收已加载的模型、文本嵌入和图像嵌入，通过迭代去噪生成视频或图像潜空间表示。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | KSANA_DIFFUSION_MODEL | ✅ | — | 已加载的扩散模型 |
| `positive` | KSANA_TEXT_ENCODE_OUTPUT | ✅ | — | 正向文本条件 |
| `negative` | KSANA_TEXT_ENCODE_OUTPUT | ✅ | — | 负向文本条件 |
| `image_embeds` | KSANA_VAE_ENCODE_OUTPUT | ✅ | — | 编码后的图像/视频潜空间 |
| `steps` | INT | ✅ | 20 | 去噪步数（1–10000） |
| `seed` | INT | ✅ | 42 | 随机种子 |
| `scheduler` | 枚举 | ✅ | `simple` | 噪声调度类型（ComfyUI 调度器） |
| `solver_name` | 枚举 | ✅ | `unipc` | 采样算法：`unipc`、`euler`、`dpm++` |
| `sample_guide_scale` | FLOAT | ✅ | 4.0 | CFG 引导强度（0–100） |
| `sample_shift` | FLOAT | ✅ | 5.0 | 噪声调度偏移（Qwen 模型使用 -1 自动计算） |
| `denoise` | FLOAT | ✅ | 1.0 | 去噪强度（0–1） |
| `latent` | LATENT | ❌ | None | 初始潜空间（用于视频到视频） |
| `rope_function` | 枚举 | ❌ | `default` | RoPE 实现：`default`、`comfy` |
| `low_sample_guide_scale` | FLOAT | ❌ | 3.0 | 低噪声模型的 CFG 强度 |
| `cache_config` | KSANA_CACHE_CONFIG | ❌ | None | 缓存策略配置 |
| `sigmas` | FLOAT | ❌ | None | 自定义 sigma 调度 |
| `add_noise_to_latent` | BOOLEAN | ❌ | False | 采样前向潜空间添加噪声 |
| `video_control_config` | KSANA_VIDEO_CONTROL_CONFIG | ❌ | None | 视频控制配置（SLG、FETA、实验性） |
| `vace_embeds` | KSANA_VACE_EMBEDS | ❌ | None | VACE 嵌入（视频条件控制） |

**输出：** `KSANA_GENERATE_OUTPUT` — 去噪后的潜空间表示。

---

### kDiT EmptyLatent — 空潜空间

| 属性 | 值 |
|------|-----|
| **类名** | `EmptyLatentNode` |
| **显示名称** | kDiT EmptyLatent |
| **分类** | kdit |
| **源码** | [`empty_latent.py`](../kdit/adapter/comfyui/nodes/empty_latent.py:28) |

创建指定尺寸的空潜空间张量，用于文生视频/文生图生成。自动处理 VAE 兼容的潜空间尺寸。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `width` | INT | ✅ | 512 | 输出宽度（像素） |
| `height` | INT | ✅ | 512 | 输出高度（像素） |
| `num_frames` | INT | ✅ | 1 | 视频帧数（图像为 1） |
| `batch_size` | INT | ✅ | 1 | 批次大小 |

**输出：** `KSANA_VAE_ENCODE_OUTPUT` — 空潜空间张量。

---

## 2. VAE 节点

### kDiT VAE Loader — VAE 加载器

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaVAELoaderNode` |
| **显示名称** | kDiT VAE Loader |
| **分类** | kdit |
| **源码** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:22) |

从 ComfyUI VAE 目录加载 VAE 模型。用于将图像编码为潜空间表示，以及将潜空间解码为图像/视频。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `vae_name` | VAE 列表 | ✅ | — | VAE 模型文件名 |

**输出：** `KDIT_VAE_MODEL` — 已加载的 VAE 模型。

---

### kDiT VAE Encoder — VAE 编码器

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaVAEEncodeNode` |
| **显示名称** | kDiT VAE Encoder |
| **分类** | kdit |
| **源码** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:38) |

使用 kDiT VAE 将视频帧编码到潜空间。用于图生视频（I2V）工作流，将输入图像转换为潜空间表示。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `vae` | KDIT_VAE_MODEL | ✅ | — | 已加载的 VAE 模型 |
| `image` | IMAGE | ✅ | — | 输入图像 |
| `width` | INT | ✅ | 512 | 目标宽度 |
| `height` | INT | ✅ | 512 | 目标高度 |
| `num_frames` | INT | ✅ | 1 | 帧数 |
| `batch_size` | INT | ✅ | 1 | 批次大小 |
| `with_end_image` | BOOLEAN | ❌ | False | 是否包含尾帧（I2V 用） |

**输出：** `KSANA_VAE_ENCODE_OUTPUT` — 编码后的潜空间张量。

---

### kDiT VAE Image Encoder — VAE 图像编码器

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaVAEImageEncodeNode` |
| **显示名称** | kDiT VAE Image Encoder |
| **分类** | kdit |
| **源码** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:63) |

为 Qwen-Image 流水线（文生图/图像编辑）编码图像。处理图像编辑工作流中的多图像编码。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `vae` | KDIT_VAE_MODEL | ✅ | — | 已加载的 VAE 模型 |
| `width` | INT | ✅ | 512 | 目标宽度 |
| `height` | INT | ✅ | 512 | 目标高度 |
| `batch_size` | INT | ✅ | 1 | 批次大小 |
| `image` | IMAGE | ❌ | None | 输入图像（编辑用） |

**输出：** `KSANA_VAE_ENCODE_OUTPUT` — 编码后的图像潜空间。

---

### kDiT VAE Decoder — VAE 解码器

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaVAEDecodeNode` |
| **显示名称** | kDiT VAE Decoder |
| **分类** | kdit |
| **源码** | [`vae.py`](../kdit/adapter/comfyui/nodes/vae.py:83) |

使用 kDiT VAE 将潜空间张量解码回像素空间（图像或视频帧）。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `vae` | KDIT_VAE_MODEL | ✅ | — | 已加载的 VAE 模型 |
| `latents` | KSANA_GENERATE_OUTPUT | ✅ | — | 生成的潜空间张量 |

**输出：** `IMAGE` — 解码后的图像/视频帧。

---

## 3. 缓存节点

缓存节点实现多种缓存策略，通过在去噪步骤间复用中间计算结果来加速推理。

### kDiT HybridCache — 混合缓存

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaHybridCacheNode` |
| **显示名称** | kDiT HybridCache |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:20) |

将步级缓存和块级缓存组合为混合缓存策略。这是获得最大加速效果的推荐方式。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `step_cache` | KSANA_CACHE_CONFIG | ❌ | None | 步级缓存（如 TeaCache、EasyCache） |
| `block_cache` | KSANA_CACHE_CONFIG | ❌ | None | 块级缓存（如 DCache、DBCache） |
| `name` | STRING | ❌ | `HybridCache` | 日志显示名称 |

**输出：** `KSANA_CACHE_CONFIG` — 组合后的混合缓存配置。

---

### kDiT CacheCombine — 缓存组合

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaCacheCombineNode` |
| **显示名称** | kDiT CacheCombine |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:40) |

为双模型配置（高噪声 + 低噪声模型）组合缓存配置。每个模型可以有独立的缓存策略。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `cache` | KSANA_CACHE_CONFIG | ✅ | — | 主模型（高噪声）的缓存配置 |
| `low_noise_model_cache` | KSANA_CACHE_CONFIG | ❌ | None | 低噪声模型的缓存配置 |

**输出：** `KSANA_CACHE_CONFIG` — 双模型组合缓存配置。

---

### kDiT DCache

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaDCacheNode` |
| **显示名称** | kDiT DCache |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:63) |

基于角度差异检测的块级缓存。将 Transformer 块分为"快速"和"慢速"两组，使用不同的缓存阈值。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `fast_degree` | FLOAT | ❌ | 45 | 快速块的角度阈值（1–90°） |
| `slow_degree` | FLOAT | ❌ | 20 | 慢速块的角度阈值（1–90°） |
| `fast_force_calc_every_n_step` | INT | ❌ | 1 | 快速块的强制重算间隔 |
| `slow_force_calc_every_n_step` | INT | ❌ | 5 | 慢速块的强制重算间隔 |
| `name` | STRING | ❌ | `""` | 显示名称 |
| `offload` | BOOLEAN | ❌ | False | 将缓存张量卸载到 CPU |

**输出：** `KSANA_CACHE_CONFIG` — DCache 配置。

---

### kDiT DBCache

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaDBCacheNode` |
| **显示名称** | kDiT DBCache |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:325) |

带 TaylorSeer 预测的动态块级缓存。选择性地只计算最重要的 Transformer 块，其余使用泰勒展开预测。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `preset` | 枚举 | ❌ | `balanced` | 预设：`custom`、`conservative`、`balanced`、`aggressive`、`wan22_high`、`wan22_low` |
| `fn_compute_blocks` | INT | ❌ | 8 | 全噪声阶段计算的块数（0–100） |
| `bn_compute_blocks` | INT | ❌ | 6 | 基础噪声阶段计算的块数（0–100） |
| `residual_diff_threshold` | FLOAT | ❌ | 0.12 | 残差差异检测阈值（0.01–1.0） |
| `max_warmup_steps` | INT | ❌ | 5 | 缓存开始前的最大预热步数 |
| `warmup_interval` | INT | ❌ | 1 | 预热计算间隔 |
| `max_cached_steps` | INT | ❌ | -1 | 最大总缓存步数（-1 = 无限） |
| `max_continuous_cached_steps` | INT | ❌ | -1 | 最大连续缓存步数（-1 = 无限） |
| `enable_separate_cfg` | BOOLEAN | ❌ | True | 条件/无条件分别缓存 |
| `cfg_compute_first` | BOOLEAN | ❌ | False | 优先计算 CFG 分支 |
| `enable_taylorseer` | BOOLEAN | ❌ | True | 启用 TaylorSeer 预测 |
| `taylorseer_order` | INT | ❌ | 1 | 泰勒展开阶数（0–4） |
| `num_blocks` | INT | ❌ | 40 | Transformer 块总数 |

**输出：** `KSANA_CACHE_CONFIG` — DBCache 配置。

---

### kDiT TeaCache

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaTeaCacheNode` |
| **显示名称** | kDiT TeaCache |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:119) |

步级缓存，当预测变化低于阈值时跳过整个去噪步骤。使用模型特定的预设以获得最佳性能。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `preset` | 枚举 | ❌ | `balanced` | 预设：`custom`、`wan21_t2v`、`wan21_i2v_720P`、`wan21_i2v_480P`、`wan22_t2v`、`wan22_i2v`、`fast`、`balanced`、`quality` |
| `threshold` | FLOAT | ❌ | 0.2 | 跳过阈值（0.05–0.5），越低质量越高 |
| `mode` | 枚举 | ❌ | `t2v_14B` | 模型模式：`t2v_14B`、`t2v_1.3B`、`i2v_720P`、`i2v_480P` |
| `start_step` | INT | ❌ | 0 | 开始缓存的步骤 |
| `end_step` | INT | ❌ | -1 | 停止缓存的步骤（-1 = 结束） |
| `cache_device` | 枚举 | ❌ | `main_device` | 缓存存储位置：`main_device`、`offload_device` |
| `verbose` | BOOLEAN | ❌ | False | 启用详细日志 |

**输出：** `KSANA_CACHE_CONFIG` — TeaCache 配置。

---

### kDiT EasyCache

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaEasyCacheNode` |
| **显示名称** | kDiT EasyCache |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:183) |

基于百分比步骤范围控制的步级缓存。比 TeaCache 配置更简单，使用直观的开始/结束百分比参数。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `preset` | 枚举 | ❌ | `balanced` | 预设：`custom`、`wan21_t2v`、`wan21_i2v`、`wan22_t2v`、`wan22_i2v`、`conservative`、`balanced`、`aggressive` |
| `reuse_thresh` | FLOAT | ❌ | 0.05 | 复用阈值（0.001–2.0），越低质量越高 |
| `start_percent` | FLOAT | ❌ | 0.2 | 开始百分比（0–1） |
| `end_percent` | FLOAT | ❌ | 0.98 | 结束百分比（0–1） |
| `mode` | 枚举 | ❌ | `t2v` | 模型模式：`t2v`、`i2v` |
| `cache_device` | 枚举 | ❌ | `main_device` | 缓存存储位置：`main_device`、`offload_device` |
| `verbose` | BOOLEAN | ❌ | False | 启用详细日志 |
| `name` | STRING | ❌ | `""` | 显示名称 |

**输出：** `KSANA_CACHE_CONFIG` — EasyCache 配置。

---

### kDiT MagCache

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaMagCacheNode` |
| **显示名称** | kDiT MagCache |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:254) |

基于幅度变化检测的步级缓存，支持可配置的最大跳步数。支持保留比率用于部分缓存复用。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `preset` | 枚举 | ❌ | `balanced` | 预设：`custom`、`conservative`、`balanced`、`aggressive`、`wan22_t2v`、`wan22_i2v` |
| `threshold` | FLOAT | ❌ | 0.04 | 变化检测阈值（0.001–0.5） |
| `max_skip_steps` | INT | ❌ | 2 | 最大连续跳步数（1–10） |
| `retention_ratio` | FLOAT | ❌ | 0.2 | 缓存数据保留比率（0–1） |
| `mode` | 枚举 | ❌ | `t2v` | 模型模式：`t2v`、`i2v` |
| `cache_device` | 枚举 | ❌ | `offload_device` | 缓存存储位置：`offload_device`、`main_device` |
| `start_step` | INT | ❌ | 0 | 开始缓存的步骤 |
| `end_step` | INT | ❌ | -1 | 停止缓存的步骤（-1 = 结束） |
| `verbose` | BOOLEAN | ❌ | False | 启用详细日志 |

**输出：** `KSANA_CACHE_CONFIG` — MagCache 配置。

---

### kDiT CustomStepCache — 自定义步骤缓存

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaCustomStepCacheNode` |
| **显示名称** | kDiT CustomStepCache |
| **分类** | kdit/cache |
| **源码** | [`cache.py`](../kdit/adapter/comfyui/nodes/cache.py:92) |

手动指定要缓存的去噪步骤。提供对缓存调度的完全控制，支持可选的每步缩放因子。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `steps` | FLOAT（强制输入） | ✅ | — | 要缓存的步骤索引（从 0 开始） |
| `scales` | FLOAT（强制输入） | ❌ | 1.0 | 每个缓存步骤的缩放因子 |
| `name` | STRING | ❌ | `""` | 显示名称 |
| `offload` | BOOLEAN | ❌ | False | 将缓存张量卸载到 CPU |

**输出：** `KSANA_CACHE_CONFIG` — CustomStepCache 配置。

---

## 4. 注意力节点

### kDiT AttentionConfig — 注意力配置

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaAttentionConfigNode` |
| **显示名称** | kDiT AttentionConfig |
| **分类** | kdit/configs |
| **源码** | [`attn_config.py`](../kdit/adapter/comfyui/nodes/attn_config.py:19) |

配置扩散模型的注意力后端。不同后端在速度和质量之间提供不同的权衡。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `backend` | 枚举 | ❌ | `flash_attn` | 后端：`flash_attn`、`sage_attn`、`torch_sdpa`、`sage_sla` |

**输出：** `KSANA_ATTENTION_CONFIG` — 注意力配置。

**可用后端：**
- **`flash_attn`** — Flash Attention 2，速度与质量的最佳平衡（推荐默认值）
- **`sage_attn`** — Sage Attention，比 Flash 更快，质量略有损失
- **`torch_sdpa`** — PyTorch 原生缩放点积注意力，通用兼容性
- **`sage_sla`** — Sage SL Attention，带 top-k 稀疏选择

---

### kDiT RadialSageAttentionConfig — 径向稀疏注意力

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaRadialSageAttentionConfigNode` |
| **显示名称** | kDiT RadialSageAttentionConfig |
| **分类** | kdit/configs |
| **源码** | [`attn_config.py`](../kdit/adapter/comfyui/nodes/attn_config.py:42) |

配置径向 Sage 注意力，实现带径向衰减的稀疏注意力模式，用于高效视频生成。对长视频序列特别有效。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dense_blocks_num` | INT | ❌ | 0 | 开头使用密集注意力的块数（0–1000） |
| `dense_attn_steps` | INT | ❌ | 1 | 使用密集注意力的步数（0–1000） |
| `decay_factor` | FLOAT | ❌ | 0.2 | 径向衰减因子（0.01–0.99），越低越稀疏 |
| `block_size` | 枚举 | ❌ | 64 | 稀疏计算的块大小：`64`、`128` |
| `dense_backend` | 枚举 | ❌ | `sage_attn` | 密集块的后端：`flash_attn`、`sage_attn`、`torch_sdpa` |

**输出：** `KSANA_ATTENTION_CONFIG` — 径向 Sage 注意力配置。

---

### kDiT SageSLAttentionConfig — Top-k 稀疏注意力

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaSageSLAConfigNode` |
| **显示名称** | kDiT SageSLAttentionConfig |
| **分类** | kdit/configs |
| **源码** | [`attn_config.py`](../kdit/adapter/comfyui/nodes/attn_config.py:110) |

配置 Sage SL 注意力，使用 top-k 选择进行稀疏注意力计算。只选择最重要的注意力条目进行计算。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `topk` | FLOAT | ✅ | 0.1 | 稀疏注意力的 top-k 比率（0.01–0.99） |
| `dense_backend` | 枚举 | ✅ | `sage_attn` | 密集计算后端：`flash_attn`、`sage_attn`、`torch_sdpa` |

**输出：** `KSANA_ATTENTION_CONFIG` — Sage SLA 配置。

---

## 5. LoRA 节点

### kDiT LoraSelect — LoRA 选择

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaLoraSelectNode` |
| **显示名称** | kDiT LoraSelect |
| **分类** | kdit |
| **源码** | [`lora.py`](../kdit/adapter/comfyui/nodes/lora.py:112) |

选择单个 LoRA 文件，在模型加载时合并到扩散模型中。LoRA 权重是静态合并的（不支持运行时动态切换）。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `lora` | LORAS 列表 | ✅ | — | 要加载的 LoRA 文件 |
| `strength` | FLOAT | ❌ | 1.0 | LoRA 合并强度（0–10） |

**输出：** `KSANA_LORA` — LoRA 配置。

---

### kDiT LoraSelectMulti — 多 LoRA 选择

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaLoraSelectMultiNode` |
| **显示名称** | kDiT LoraSelectMulti |
| **分类** | kdit |
| **源码** | [`lora.py`](../kdit/adapter/comfyui/nodes/lora.py:28) |

同时选择最多 5 个 LoRA 文件，每个有独立的强度控制。所有 LoRA 在模型加载时合并。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `lora_1` | LORAS 列表 | ✅ | — | 第一个 LoRA 文件 |
| `strength_1` | FLOAT | ❌ | 1.0 | 第一个 LoRA 的强度 |
| `lora_2` – `lora_5` | LORAS 列表 | ❌ | Empty | 额外的 LoRA 文件 |
| `strength_2` – `strength_5` | FLOAT | ❌ | 1.0 | 额外 LoRA 的强度 |

**输出：** `KSANA_LORA` — 组合 LoRA 配置。

---

### kDiT LoraCombine — LoRA 组合

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaLoraCombineNode` |
| **显示名称** | kDiT LoraCombine |
| **分类** | kdit |
| **源码** | [`lora.py`](../kdit/adapter/comfyui/nodes/lora.py:141) |

将多个 LoraSelect 节点的 LoRA 配置组合为单个配置。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `lora_1` | KSANA_LORA | ✅ | — | 第一个 LoRA 配置 |
| `lora_2` | KSANA_LORA | ❌ | None | 第二个 LoRA 配置 |
| `lora_3` | KSANA_LORA | ❌ | None | 第三个 LoRA 配置 |

**输出：** `KSANA_LORA` — 组合后的 LoRA 配置。

---

## 6. 视频控制节点

### kDiT WanVace To Video — VACE 视频编码

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaWanVaceToVideoNode` |
| **显示名称** | kDiT WanVace To Video |
| **分类** | kdit/vace |
| **源码** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:39) |

为 Wan VACE（视频-音频条件编码）视频生成编码控制视频和参考图像。支持带遮罩的控制视频、参考图像和多个 VACE 输入的链式连接。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `width` | INT | ✅ | 512 | 输出视频宽度（16–8192） |
| `height` | INT | ✅ | 512 | 输出视频高度（16–8192） |
| `strength` | FLOAT | ✅ | 1.0 | VACE 条件强度（0–100） |
| `num_frames` | INT | ✅ | 25 | 生成帧数（1–8192） |
| `batch_size` | INT | ✅ | 1 | 批次大小（1–4096） |
| `vace_start_percent` | FLOAT | ✅ | 0.0 | VACE 应用的开始百分比（0–1） |
| `vace_end_percent` | FLOAT | ✅ | 1.0 | VACE 应用的结束百分比（0–1） |
| `vae` | KDIT_VAE_MODEL | ❌ | None | 用于编码的 VAE 模型（操作必需） |
| `control_video` | IMAGE | ❌ | None | 控制视频帧 |
| `control_masks` | MASK | ❌ | None | 控制视频的遮罩 |
| `reference_image` | IMAGE | ❌ | None | 风格引导的参考图像 |
| `prev_vace_embeds` | KSANA_VACE_EMBEDS | ❌ | None | 前一个 VACE 嵌入（链式连接用） |

**输出：**
- `KSANA_VACE_EMBEDS` — VACE 嵌入（vace_context、vace_scale、元数据）
- `KSANA_VAE_ENCODE_OUTPUT` — 视频的潜空间张量

---

### kDiT Skip Layer Guidance — 跳层引导

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaSLGNode` |
| **显示名称** | kDiT Skip Layer Guidance |
| **分类** | kdit/vace |
| **源码** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:309) |

跳层引导（SLG）：在指定的 Transformer 块上跳过无条件推理，加速基于 CFG 的采样，且不会显著降低质量。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `blocks` | STRING | ✅ | `"10"` | 逗号分隔的块索引（如 `"9,10,11"`） |
| `start_percent` | FLOAT | ✅ | 0.1 | 步骤开始百分比（0–1） |
| `end_percent` | FLOAT | ✅ | 1.0 | 步骤结束百分比（0–1） |

**输出：** `KSANA_SLG_ARGS` — SLG 配置。

---

### kDiT Enhance-A-Video — 视频增强

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaEnhanceAVideoNode` |
| **显示名称** | kDiT Enhance-A-Video |
| **分类** | kdit/vace |
| **源码** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:370) |

Enhance-A-Video（FETA）：通过跨帧分数调制注意力来改善时间一致性。减少闪烁并改善运动平滑度。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `weight` | FLOAT | ✅ | 2.0 | 增强权重（0–100），典型范围：1.0–5.0 |
| `start_percent` | FLOAT | ✅ | 0.0 | 步骤开始百分比（0–1） |
| `end_percent` | FLOAT | ✅ | 1.0 | 步骤结束百分比（0–1） |

**输出：** `KSANA_FETA_ARGS` — FETA 配置。

**参考：** [Enhance-A-Video](https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video)

---

### kDiT Experimental Args — 实验性参数

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaExperimentalArgsNode` |
| **显示名称** | kDiT Experimental Args |
| **分类** | kdit/vace |
| **源码** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:432) |

实验性采样优化集合。每种技术可以独立启用/禁用。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| **CFG-Zero-Star** | | | | |
| `cfg_zero_star` | BOOLEAN | ✅ | False | 启用 CFG-Zero-Star：减少高 CFG 导致的过饱和 |
| `use_zero_init` | BOOLEAN | ✅ | False | 初始步骤返回零噪声以稳定采样 |
| `zero_star_steps` | INT | ✅ | 0 | 零初始化步数（0–100） |
| **FreSca** | | | | |
| `use_fresca` | BOOLEAN | ✅ | False | 启用频域滤波以减少 CFG 伪影 |
| `fresca_scale_low` | FLOAT | ✅ | 1.0 | 低频缩放因子（0–10） |
| `fresca_scale_high` | FLOAT | ✅ | 1.25 | 高频缩放因子（0–10） |
| `fresca_freq_cutoff` | INT | ✅ | 20 | 频率截止阈值（0–10000） |
| **TCFG** | | | | |
| `use_tcfg` | BOOLEAN | ✅ | False | 启用切平面 CFG 投影以减少色偏 |
| **RAAG** | | | | |
| `raag_alpha` | FLOAT | ✅ | 0.0 | 自适应 CFG 调整 alpha（0 = 禁用，0–10） |
| **双向采样** | | | | |
| `bidirectional_sampling` | BOOLEAN | ✅ | False | 前向 + 后向时间采样（计算量翻倍） |
| **TSR** | | | | |
| `temporal_score_rescaling` | BOOLEAN | ✅ | False | 基于时间统计重缩放噪声预测 |
| `tsr_k` | FLOAT | ✅ | 0.95 | TSR 温度（0–100），越低重缩放越强 |
| `tsr_sigma` | FLOAT | ✅ | 1.0 | TSR sigma：TSR 影响采样的时机（0–1） |
| **其他** | | | | |
| `video_attention_split_steps` | STRING | ✅ | `""` | 逗号分隔的步骤索引，用于多提示词注意力分割 |

**输出：** `KSANA_EXPERIMENTAL_ARGS` — 实验性配置。

**参考文献：**
- [CFG-Zero-Star](https://github.com/WeichenFan/CFG-Zero-star)
- [FreSca](https://github.com/WikiChao/FreSca)
- [TCFG](https://arxiv.org/abs/2503.18137)
- [双向采样 (WanFM)](https://github.com/ff2416/WanFM)
- [TSR](https://github.com/temporalscorerescaling/TSR)

---

### kDiT VideoControlConfig — 视频控制配置

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaVideoControlConfigNode` |
| **显示名称** | kDiT VideoControlConfig |
| **分类** | kdit |
| **源码** | [`vace.py`](../kdit/adapter/comfyui/nodes/vace.py:617) |

将视频控制参数（SLG、FETA、实验性）组合为单个配置，供生成器节点使用。同时接受 WanVideoWrapper 兼容输入，实现跨插件兼容。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `slg_args` | KSANA_SLG_ARGS | ❌ | None | kDiT 跳层引导配置 |
| `feta_args` | KSANA_FETA_ARGS | ❌ | None | kDiT Enhance-A-Video 配置 |
| `experimental_args` | KSANA_EXPERIMENTAL_ARGS | ❌ | None | kDiT 实验性参数配置 |
| `wanvideo_slg_args` | WANVIDEO_SLG_ARGS | ❌ | None | WanVideoWrapper SLG（兼容输入） |
| `wanvideo_feta_args` | WANVIDEO_FETA_ARGS | ❌ | None | WanVideoWrapper FETA（兼容输入） |
| `wanvideo_exp_args` | WANVIDEO_EXPERIMENTAL_ARGS | ❌ | None | WanVideoWrapper 实验性（兼容输入） |

**输出：** `KSANA_VIDEO_CONTROL_CONFIG` — 组合后的视频控制配置。

---

## 7. 工具节点

### kDiT TorchCompile — 编译优化

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaTorchCompileNode` |
| **显示名称** | kDiT TorchCompile |
| **分类** | kdit |
| **源码** | [`torch_compile.py`](../kdit/adapter/comfyui/nodes/torch_compile.py:20) |

配置扩散模型的 `torch.compile()` 优化。初始编译预热后可显著加速推理。NPU 上不支持。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `backend` | 枚举 | ❌ | `inductor` | 编译后端 |
| `mode` | 枚举 | ❌ | `max-autotune` | 优化模式 |
| `fullgraph` | BOOLEAN | ❌ | True | 作为完整图编译 |
| `dynamic` | BOOLEAN | ❌ | False | 启用动态形状 |

**输出：** `KSANA_TORCH_COMPILE` — Torch 编译配置。

---

### kDiT TextEmbConverter — 文本嵌入转换

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaTextEmbConverterNode` |
| **显示名称** | kDiT TextEmbConverter |
| **分类** | kdit |
| **源码** | [`converter.py`](../kdit/adapter/comfyui/nodes/converter.py:23) |

将其他 ComfyUI 文本编码器节点（如标准 CLIP/T5 节点）的文本嵌入转换为 kDiT 兼容格式。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text_emb` | CONDITIONING | ✅ | — | ComfyUI 文本编码器的文本嵌入 |

**输出：** `KSANA_TEXT_ENCODE_OUTPUT` — 转换后的文本嵌入。

---

### kDiT DebugNode — 调试节点

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaDebugNode` |
| **显示名称** | kDiT DebugNode |
| **分类** | kdit |
| **源码** | [`debug.py`](../kdit/adapter/comfyui/nodes/debug.py:22) |

透传调试节点，记录任何输入张量或数据结构的形状、数据类型等属性。用于工作流调试。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `source` | *（任意） | ✅ | — | 要检查的任意数据 |
| `name` | STRING | ❌ | `""` | 调试输出的标签 |

**输出：** 与输入相同类型（透传）。

---

### kDiT Empty Torch Cache — 清空显存缓存

| 属性 | 值 |
|------|-----|
| **类名** | `KsanaEmptyTorchCacheNode` |
| **显示名称** | kDiT Empty Torch Cache |
| **分类** | kdit |
| **源码** | [`empty_torch_cache.py`](../kdit/adapter/comfyui/nodes/empty_torch_cache.py:19) |

透传节点，通过清空 PyTorch CUDA/NPU 缓存来释放 GPU/NPU 显存。在重型操作之间连接以回收显存。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `source` | *（任意） | ✅ | — | 任意数据（透传） |

**输出：** 与输入相同类型（清空缓存后透传）。

---

## 节点汇总表

| # | 显示名称 | 类名 | 分类 | 用途 |
|---|---------|------|------|------|
| 1 | kDiT Model Loader | `KsanaModelLoaderNode` | kdit | 加载扩散模型 |
| 2 | kDiT Vace Model Select | `KsanaVaceModelSelectNode` | kdit | 选择 VACE 模型文件 |
| 3 | kDiT Generator | `KsanaGeneratorNode` | kdit | 主采样/去噪 |
| 4 | kDiT EmptyLatent | `EmptyLatentNode` | kdit | 创建空潜空间 |
| 5 | kDiT VAE Loader | `KsanaVAELoaderNode` | kdit | 加载 VAE 模型 |
| 6 | kDiT VAE Encoder | `KsanaVAEEncodeNode` | kdit | 编码视频到潜空间 |
| 7 | kDiT VAE Image Encoder | `KsanaVAEImageEncodeNode` | kdit | 编码图像到潜空间 |
| 8 | kDiT VAE Decoder | `KsanaVAEDecodeNode` | kdit | 解码潜空间到像素 |
| 9 | kDiT HybridCache | `KsanaHybridCacheNode` | kdit/cache | 组合步级+块级缓存 |
| 10 | kDiT CacheCombine | `KsanaCacheCombineNode` | kdit/cache | 组合双模型缓存 |
| 11 | kDiT DCache | `KsanaDCacheNode` | kdit/cache | 块级角度缓存 |
| 12 | kDiT DBCache | `KsanaDBCacheNode` | kdit/cache | 动态块缓存 + TaylorSeer |
| 13 | kDiT TeaCache | `KsanaTeaCacheNode` | kdit/cache | 步级阈值缓存 |
| 14 | kDiT EasyCache | `KsanaEasyCacheNode` | kdit/cache | 步级百分比缓存 |
| 15 | kDiT MagCache | `KsanaMagCacheNode` | kdit/cache | 步级幅度缓存 |
| 16 | kDiT CustomStepCache | `KsanaCustomStepCacheNode` | kdit/cache | 手动步骤缓存 |
| 17 | kDiT AttentionConfig | `KsanaAttentionConfigNode` | kdit/configs | 注意力后端配置 |
| 18 | kDiT RadialSageAttentionConfig | `KsanaRadialSageAttentionConfigNode` | kdit/configs | 径向稀疏注意力 |
| 19 | kDiT SageSLAttentionConfig | `KsanaSageSLAConfigNode` | kdit/configs | Top-k 稀疏注意力 |
| 20 | kDiT LoraSelect | `KsanaLoraSelectNode` | kdit | 单 LoRA 选择 |
| 21 | kDiT LoraSelectMulti | `KsanaLoraSelectMultiNode` | kdit | 多 LoRA 选择（最多 5 个） |
| 22 | kDiT LoraCombine | `KsanaLoraCombineNode` | kdit | 组合 LoRA 配置 |
| 23 | kDiT WanVace To Video | `KsanaWanVaceToVideoNode` | kdit/vace | VACE 视频条件 |
| 24 | kDiT Skip Layer Guidance | `KsanaSLGNode` | kdit/vace | SLG 加速优化 |
| 25 | kDiT Enhance-A-Video | `KsanaEnhanceAVideoNode` | kdit/vace | 时间一致性（FETA） |
| 26 | kDiT Experimental Args | `KsanaExperimentalArgsNode` | kdit/vace | 实验性采样优化 |
| 27 | kDiT VideoControlConfig | `KsanaVideoControlConfigNode` | kdit | 组合视频控制参数 |
| 28 | kDiT TorchCompile | `KsanaTorchCompileNode` | kdit | torch.compile 配置 |
| 29 | kDiT TextEmbConverter | `KsanaTextEmbConverterNode` | kdit | 转换文本嵌入 |
| 30 | kDiT DebugNode | `KsanaDebugNode` | kdit | 调试检查 |
| 31 | kDiT Empty Torch Cache | `KsanaEmptyTorchCacheNode` | kdit | 释放 GPU 显存 |
