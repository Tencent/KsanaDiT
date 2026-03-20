# Generators

Denoising loop implementations for different model architectures.

## Overview

- [`BaseGenerator`](base_generator.md) — Abstract base class for all generators
- [`WanGenerator`](wan_generator.md) — Wan model denoising loop
- [`QwenGenerator`](qwen_generator.md) — Qwen model denoising loop
- [`VaceGenerator`](vace_generator.md) — VACE model denoising loop
- [`GeneratorFactory`](generator_factory.md) — Factory for creating generators by `ModelKey`

## Latent Input Semantics

Generator inputs are split into **BaseLatent** and **AuxLatent** (defined in `kdit/generators/generator_context.py`):

### BaseLatent

The primary latent that determines the output dimensions (`noise_shape`). `noise_shape` is derived from `base_latent.latent.shape[1:]`.

- **Video generation**: controls resolution (H × W) and duration (F frames)
- **Image generation**: controls resolution (H × W)

Most models create an empty latent (`torch.zeros`) to define the shape and batch size. Only **WAN I2V** and its derivative **VACE** pass a non-None `mask` alongside the latent (used for image-conditioned generation).

### AuxLatent

An optional auxiliary input. Can be any tensor or `list[tensor]`, with each model subclass deciding how to use it:

| Model | AuxLatent Content | Usage |
|-------|------------------|-------|
| WAN T2V | `None` | Not used |
| WAN I2V / VACE | Noise-blending latent | Mixed with noise via `_apply_aux_latent()` |
| Qwen T2I | `None` | Not used |
| Qwen Edit | Reference image VAE embeddings | Passed as `ref_latents` to the model |
