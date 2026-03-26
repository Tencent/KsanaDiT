# Generator — Diffusion 去噪引擎

Generator 是 Node 内部的实现细节（被 [`GeneratorNode`](../../kdit/nodes/infers/generator_node.py:42) 封装），负责 Diffusion 去噪流程。Generator 内部的 tensor 流转（noise、denoise step 等）不受 DAG 改造影响。

---

## 声明式架构 (Plan E)

- **`GeneratorFactory` 已废弃**，新架构使用 [`GeneratorDef`](../../kdit/generators/generator_def.py:30) (frozen dataclass) + [`GeneratorRunner`](../../kdit/generators/generator_runner.py:38) (final 类，无子类)
- **三个 Handler 注入模型差异**: [`TextHandler`](../../kdit/generators/handlers/text_handler.py) / [`LatentHandler`](../../kdit/generators/handlers/latent_handler.py) / [`DenoiseHandler`](../../kdit/generators/handlers/denoise_handler.py)
- **注册表模式**: [`register_generator_def()`](../../kdit/generators/generator_def.py:48) 注册到 `_GENERATOR_DEF_REGISTRY`，定义文件在 [`kdit/generators/defs/`](../../kdit/generators/defs/) 下
- **`GeneratorRunner.run(ctx)`** 接收 [`GeneratorInferContext`](../../kdit/generators/generator_context.py:64) 结构化上下文

---

## BaseLatent 与 AuxLatent 语义规范

### 概述

Generator 的输入 latent 分为两类：**BaseLatent**（主 latent）和 **AuxLatent**（辅助 latent），定义在 [`kdit/generators/generator_context.py`](../../kdit/generators/generator_context.py)。

### BaseLatent — 主 latent，决定输出尺寸

`BaseLatent` 是 Generator 的**必需**输入，其核心职责是**决定 `noise_shape`**，即 Generator 输入输出的主尺寸：

- **视频场景**：`noise_shape` 决定视频的分辨率（H × W）和时长（F 帧数）
- **图片场景**：`noise_shape` 决定图片的分辨率（H × W）

`noise_shape` 从 `base_latent.latent.shape[1:]` 推导（`GeneratorRunner.run()` 中 `noise_shape = list(base_latent_obj.latent.shape[1:])`）。

```python
# kdit/generators/generator_context.py
@dataclass
class BaseLatent:
    latent: torch.Tensor          # 主 tensor，noise_shape 从此推导
    mask: torch.Tensor | None = None  # 仅 WAN I2V / VACE 场景非 None
```

#### BaseLatent.mask 的使用场景

| 模型 | `base_latent.latent` | `base_latent.mask` | 说明 |
|------|---------------------|-------------------|------|
| WAN T2V | 空 latent（`torch.zeros`） | `None` | 通过 empty latent 方法创建，仅决定 noise_shape |
| WAN I2V | VAE encode 的首帧 latent | 非 None（mask tensor） | `preprocess_base_latent()` 将 `[latent, mask]` concat |
| WAN VACE | VAE encode 的首帧 latent | 非 None（mask tensor） | 继承 WAN I2V 的行为 |
| Qwen T2I | 空 latent（`torch.zeros`） | `None` | 通过 empty latent 方法创建 |
| Qwen Edit | 空 latent（`torch.zeros`） | `None` | 通过 empty latent 方法创建 |

**关键规则**：目前只有 WAN I2V 及其衍生的 VACE 在传入 `BaseLatent` 时 `mask` 非 None（`base_latent_list` 为 `[latent, mask]`）。其他所有模型都通过 empty latent 方法给一个空 tensor 来决定 `noise_shape` 和 `batch_size`。

### AuxLatent — 辅助 latent，模型自定义用途

`AuxLatent` 是**可选**输入，理论上可以是任意 tensor 或 list，传入后由模型子类自行决定如何与 base 一起作用：

```python
# kdit/generators/generator_context.py
@dataclass
class AuxLatent:
    latent: ImageEmbeds | MultiPromptImageEmbeds | torch.Tensor
```

类型别名：
- `ImageEmbeds = torch.Tensor` — 单个 Tensor，`shape[0]` 是 batch 维度
- `MultiPromptImageEmbeds = list[torch.Tensor]` — list 长度 = prompt 数量，每个 Tensor 对应一个 prompt 的参考图

#### 各模型的 AuxLatent 用途

| 模型 | AuxLatent 内容 | 使用方式 |
|------|---------------|---------|
| WAN T2V | `None` | 不使用 |
| WAN I2V | 用于噪声混合的 latent | `_apply_aux_latent()` 中与 noise 混合 |
| WAN VACE | 用于噪声混合的 latent | 继承 WAN I2V 的 `_apply_aux_latent()` |
| Qwen T2I | `None` | 不使用 |
| Qwen Edit | 参考图片的 VAE encode 结果 | `prepare_model_forward_kargs()` 中作为 `ref_latents` 传入模型 |

### 子类覆写点

| 方法 | 基类行为 | 覆写场景 |
|------|---------|---------|
| `preprocess_base_latent(base_latent_list)` | 取 `list[0]`（仅 latent） | WAN I2V: 将 `[latent, mask]` concat 为单个 tensor |
| `_apply_aux_latent(noise, aux, ...)` | `raise NotImplementedError` | 每个子类必须实现：WAN 做噪声混合，Qwen 直接返回 noise（不混合） |
| `pack_aux_latent(ref_latent, patch_size)` | 直接返回 | Qwen: `pack_ref_latents()` 做 patchify |

### 禁止事项

- ❌ **禁止**在 `BaseLatent` 中放非 latent 数据（如文本 embedding、配置参数）
- ❌ **禁止**绕过 `BaseLatent` 直接在 `context.metadata` 中传递 noise_shape — noise_shape 必须从 `base_latent.latent.shape[1:]` 推导
- ❌ **禁止**在新模型中假设 `BaseLatent.mask` 一定为 None — 应通过 `preprocess_base_latent()` 处理
- ❌ **禁止**在 `AuxLatent` 中放不可序列化对象 — AuxLatent 通过 tensor_pool 流转，必须是 tensor 或 list[tensor]
