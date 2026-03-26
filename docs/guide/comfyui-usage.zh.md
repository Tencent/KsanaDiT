# ComfyUI 使用指南

[English Version](comfyui-usage.md)

## 概述

kDiT 以自定义节点插件的形式集成到 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)，提供可视化工作流设计来进行视频和图像生成。所有 kDiT 节点在 ComfyUI 节点菜单中以 **"kDiT"** 为前缀。

## 安装

### 作为 ComfyUI 自定义节点安装

```bash
# 1. 进入 ComfyUI 的 custom_nodes 目录
cd /path/to/ComfyUI/custom_nodes

# 2. 克隆 kDiT 仓库
git clone https://github.com/Tencent/kDiT.git

# 3. 进入 kDiT 目录并通过脚本安装
cd kDiT
./scripts/install_public.sh
```

安装脚本会自动检测平台（GPU/NPU/XPU），安装所有依赖，并自动配置 ComfyUI 自定义节点。

### 模型文件

将模型文件放置在 ComfyUI 标准目录中：

- **Diffusion 模型**: `ComfyUI/models/diffusion_models/`
- **VAE 模型**: `ComfyUI/models/vae/`
- **LoRA 模型**: `ComfyUI/models/loras/`

## 基本工作流

kDiT 在 ComfyUI 中的典型工作流遵循以下模式：

```
[模型加载器] → [生成器] → [VAE 解码器] → [预览/保存]
     ↑            ↑
[注意力配置]    [文本编码器]
[LoRA 选择]    [VAE 编码器]
[Torch 编译]   [缓存配置]
```

### 核心流程

1. **kDiT Model Loader** — 加载 diffusion 模型及可选配置
2. **文本编码器** — 编码文本提示词（使用 ComfyUI 内置或 WanVideoWrapper 文本编码器）
3. **kDiT VAE Encoder** — 将输入图像/视频编码到潜空间（用于 I2V/编辑任务）
4. **kDiT Generator** — 执行去噪/采样过程
5. **kDiT VAE Decoder** — 将潜空间解码回图像/视频
6. **预览/保存** — 使用 ComfyUI 内置的预览或保存节点

## 各任务工作流

### 文生视频 (T2V)

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [保存视频]
                              ↑
[文本编码器] ──→ positive/negative
[kDiT VAE Encoder] ──→ image_embeds（空潜空间）
```

关键设置：
- **模型**：选择 Wan2.2 T2V 模型（如 `wan2.2_t2v_high_noise_14B_fp16.safetensors`）
- **VAE 编码器**：设置 `num_frames`、`width`、`height` 控制输出尺寸
- **生成器**：配置 `steps`、`seed`、`solver_name`、`sample_guide_scale`、`sample_shift`

### 图生视频 (I2V)

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [保存视频]
                              ↑
[文本编码器] ──→ positive/negative
[加载图像] ──→ [kDiT VAE Encoder] ──→ image_embeds
```

关键设置：
- **模型**：选择 Wan2.2 I2V 模型（如 `wan2.2_i2v_high_noise_14B_fp16.safetensors`）
- **VAE 编码器**：连接 `start_image`（可选连接 `end_image`）
- 可选使用双模型（高噪声 + 低噪声），通过 `low_noise_model_name` 设置

### VACE（视频可控编辑）

```
[kDiT Vace Model Select] ──→ [kDiT Model Loader]
                                      ↓
[kDiT WanVace To Video] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder]
         ↑                        ↑
[加载视频/图像]              [文本编码器]
[kDiT VAE Loader]
```

关键设置：
- **Vace Model Select**：选择 VACE diffusion 模型
- **WanVace To Video**：配置控制视频、遮罩、参考图像、强度
- 支持通过 `prev_vace_embeds` 链式连接多个 VACE 输入

### 文生图 (T2I) — Qwen-Image

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [保存图像]
                              ↑
[文本编码器] ──→ positive/negative
[kDiT VAE Encoder] ──→ image_embeds（空潜空间）
```

关键设置：
- **模型**：选择 Qwen-Image 模型（如 `qwen_image_bf16.safetensors`）
- **sample_shift**：使用 `-1` 自动计算（让 pipeline 自动决定）
- **solver_name**：Qwen 模型使用 `flowmatch_euler`

### 图像编辑 — Qwen-Image Edit

```
[kDiT Model Loader] ──→ [kDiT Generator] ──→ [kDiT VAE Decoder] ──→ [保存图像]
                              ↑
[文本编码器] ──→ positive/negative
[加载图像] ──→ [kDiT VAE Image Encoder] ──→ image_embeds
```

关键设置：
- **模型**：选择 Qwen-Image-Edit 模型
- **VAE Image Encoder**：连接参考图像用于编辑

## 双模型（高/低噪声）

对于具有独立高噪声和低噪声检查点的模型（如 Wan2.2）：

1. 在 **kDiT Model Loader** 中，将 `model_name` 设为高噪声模型
2. 将 `low_noise_model_name` 设为低噪声模型
3. 调整 `model_boundary`（默认：0.875）控制切换点

双模型的缓存和 LoRA 配置：
- 使用 **kDiT CacheCombine** 为每个模型设置独立缓存
- 使用 **kDiT LoraCombine** 为每个模型设置独立 LoRA

## 视频控制功能

### 跳层引导 (SLG)

通过跳过指定 transformer 块的无条件推理来加速采样。

1. 添加 **kDiT Skip Layer Guidance** 节点
2. 配置 `blocks`（如 "10"）、`start_percent`、`end_percent`
3. 连接到 **kDiT VideoControlConfig** → **kDiT Generator**

### Enhance-A-Video (FETA)

通过跨帧注意力分数调制来改善时序一致性。

1. 添加 **kDiT Enhance-A-Video** 节点
2. 配置 `weight`、`start_percent`、`end_percent`
3. 连接到 **kDiT VideoControlConfig** → **kDiT Generator**

### 实验性采样参数

高级采样优化，包括：
- **CFG-Zero-Star**：减少高 CFG 导致的过饱和
- **FreSca**：频域滤波减少 CFG 伪影
- **TCFG**：切平面 CFG，减少色偏
- **RAAG**：自适应 CFG 调整
- **双向采样**：前向 + 后向时序采样
- **TSR**：时序分数重缩放

1. 添加 **kDiT Experimental Args** 节点
2. 启用所需功能
3. 连接到 **kDiT VideoControlConfig** → **kDiT Generator**

## WanVideoWrapper 兼容性

kDiT 支持来自 WanVideoWrapper 节点的输入：

- **kDiT TextEmbConverter**：将 `WANVIDEOTEXTEMBEDS` 转换为 kDiT 格式（positive + negative）
- **kDiT VideoControlConfig**：同时接受 kDiT 和 WanVideoWrapper 的 SLG/FETA/Experimental 参数

## 性能优化

### 注意力后端

使用 **kDiT AttentionConfig** 选择：
- `flash_attn` — Flash Attention（大多数情况推荐）
- `sage_attn` — Sage Attention
- `torch_sdpa` — PyTorch SDPA

使用 **kDiT RadialSageAttentionConfig** 配置稀疏注意力模式。

使用 **kDiT SageSLAttentionConfig** 配置 top-k 稀疏注意力。

### 缓存

将缓存节点连接到 Generator 的 `cache_config` 输入：
- **kDiT DCache** — 快速步级缓存
- **kDiT DBCache** — 块级缓存（含 TaylorSeer）
- **kDiT TeaCache** / **kDiT EasyCache** / **kDiT MagCache** — 各种基于阈值的缓存
- **kDiT HybridCache** — 组合步级 + 块级缓存

### torch.compile

使用 **kDiT TorchCompile** 节点连接到 Model Loader，启用 JIT 编译加速。

### 显存管理

在 kDiT 节点和高显存消耗的后处理节点之间插入 **kDiT Empty Torch Cache** 节点，释放 GPU 显存。

## 节点参考

完整的 32 个 ComfyUI 节点详细说明，请参阅：

📖 [ComfyUI 节点参考 (中文)](comfyui-nodes_cn.md) | [ComfyUI Node Reference (EN)](comfyui-nodes.md)
