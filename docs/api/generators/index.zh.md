# 生成器

不同模型架构的去噪循环实现。

## 概览

- [`BaseGenerator`](base_generator.md) — 所有生成器的抽象基类
- [`WanGenerator`](wan_generator.md) — Wan 模型去噪循环
- [`QwenGenerator`](qwen_generator.md) — Qwen 模型去噪循环
- [`VaceGenerator`](vace_generator.md) — VACE 模型去噪循环
- [`GeneratorFactory`](generator_factory.md) — 通过 `ModelKey` 创建生成器的工厂

## Latent 输入语义

Generator 的输入 latent 分为 **BaseLatent** 和 **AuxLatent**（定义在 `kdit/generators/generator_context.py`）：

### BaseLatent

主 latent，决定输出尺寸（`noise_shape`）。`noise_shape` 从 `base_latent.latent.shape[1:]` 推导。

- **视频生成**：控制分辨率（H × W）和时长（F 帧数）
- **图片生成**：控制分辨率（H × W）

大多数模型通过空 latent（`torch.zeros`）来定义 shape 和 batch size。只有 **WAN I2V** 及其衍生的 **VACE** 会在 latent 旁传入非 None 的 `mask`（用于图像条件生成）。

### AuxLatent

可选的辅助输入，可以是任意 tensor 或 `list[tensor]`，由模型子类自行决定如何使用：

| 模型 | AuxLatent 内容 | 使用方式 |
|------|---------------|---------|
| WAN T2V | `None` | 不使用 |
| WAN I2V / VACE | 噪声混合 latent | 通过 `_apply_aux_latent()` 与噪声混合 |
| Qwen T2I | `None` | 不使用 |
| Qwen Edit | 参考图片的 VAE embedding | 作为 `ref_latents` 传入模型 |
