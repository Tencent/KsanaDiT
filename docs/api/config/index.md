# Config

Configuration dataclasses for all aspects of the kDiT framework.

## Overview

| Config | Description |
|--------|-------------|
| [`ModelConfig`](model_config.md) | Model paths and loading options |
| [`SampleConfig`](sample_config.md) | Sampling parameters (steps, CFG, solver) |
| [`RuntimeConfig`](runtime_config.md) | Runtime behavior (offload, compile) |
| [`AttentionConfig`](attention_config.md) | Attention backend selection |
| [`LinearConfig`](linear_config.md) | Linear backend (standard / FP8) |
| [`DistributedConfig`](distributed_config.md) | Multi-GPU / distributed settings |
| [`CacheConfig`](cache_config.md) | Step-level caching strategies |
| [`TorchCompileConfig`](torch_compile_config.md) | `torch.compile` options |
