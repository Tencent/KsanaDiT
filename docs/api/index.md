# API Reference

This section provides auto-generated API documentation for all public modules in the `kdit` package.

## Package Structure

```
kdit/
├── engine/          # Engine — thread-safe singleton, orchestrates inference
├── pipelines/       # Pipeline — declarative pipeline definitions
├── config/          # Configuration dataclasses
├── nodes/           # InferNode / LoaderNode — pipeline building blocks
│   ├── core/        # Base classes and factories
│   ├── infers/      # Inference nodes (text encode, generate, VAE decode)
│   └── loaders/     # Model loader nodes
├── models/          # Model wrappers (Diffusion, TextEncoder, VAE)
├── generators/      # Denoising loop implementations
├── executor/        # Local and Ray executors
├── operations/      # Low-level operators (attention, linear, QKV fusion)
├── cache/           # Step-level caching strategies
├── tensor/          # TensorPool and TensorKey
├── accelerator/     # Platform detection and dtype utilities
├── sample_solvers/  # ODE/SDE solvers (Euler, UniPC, DPM)
├── scheduler/       # Batch scheduling
├── memory/          # Memory management
└── utils/           # Shared utilities
```

## Quick Navigation

| Module | Description |
|--------|-------------|
| [Engine](engine.md) | Thread-safe singleton, `auto_dispatch` decorator |
| [Pipeline](pipeline/index.md) | `Pipeline.from_models()`, `PipelineDef`, `ContextBuilder` |
| [Config](config/index.md) | All configuration dataclasses |
| [Nodes](nodes/index.md) | `InferNode`, `LoaderNode`, factories |
| [Models](models/index.md) | Model wrappers and `ModelKey` |
| [Generators](generators/index.md) | Denoising loops (Wan, Qwen, VACE) |
| [Executor](executor/index.md) | Local and Ray executors |
| [Operations](operations/index.md) | Attention, Linear, QKV fusion |
| [Cache](cache/index.md) | TeaCache, DBCache, DCache, etc. |
| [Tensor](tensor/index.md) | `TensorPool`, `TensorKey` |
| [Accelerator](accelerator.md) | Platform detection, dtype |
| [Sample Solvers](sample_solvers.md) | Euler, UniPC, DPM solvers |
| [Scheduler](scheduler.md) | Batch scheduling |
| [Memory](memory.md) | Pinned memory manager |
| [Utilities](utils.md) | Logging, loading, profiling, etc. |
