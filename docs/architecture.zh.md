# 架构概览

kDiT 采用 **6 层模块化架构**，专为高性能 DiT 推理设计，兼顾最大灵活性和可扩展性。

## 层次图

```mermaid
graph TB
    subgraph "Layer 1: Interaction"
        A1[Pipeline API]
        A2[ComfyUI Adapter]
    end

    subgraph "Layer 2: Execution"
        B1[Engine]
        B2[Executor]
        B3[Ray Executor]
    end

    subgraph "Layer 3: Unit"
        C1[InferNode]
        C2[LoaderNode]
        C3[Generator]
    end

    subgraph "Layer 4: Cache"
        D1[TeaCache]
        D2[DBCache]
        D3[DCache]
        D4[EasyCache / MagCache]
    end

    subgraph "Layer 5: Operator"
        E1[Attention Backends]
        E2[Linear / FP8]
        E3[QKV Fusion]
    end

    subgraph "Layer 6: Hardware"
        F1[CUDA / GPU]
        F2[NPU / Ascend]
        F3[Platform Detection]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2
    B1 --> B3
    B2 --> C1
    B2 --> C2
    B3 --> C1
    C1 --> C3
    C3 --> D1
    C3 --> D2
    C3 --> D3
    C3 --> D4
    C1 --> E1
    C1 --> E2
    C1 --> E3
    E1 --> F1
    E1 --> F2
    E2 --> F1
    F1 --> F3
    F2 --> F3
```

## 各层详解

### 第 1 层：交互层 (Interaction)

用户入口，支持两种使用模式：

| 组件 | 说明 |
|------|------|
| [`Pipeline`](api/pipeline/pipeline.md) | Python API — `Pipeline.from_models()` 自动检测模型类型并构建推理流水线 |
| ComfyUI Adapter | 32 个自定义节点，支持 ComfyUI 可视化工作流设计 |

### 第 2 层：执行层 (Execution)

管理推理生命周期，包括单 GPU 和多 GPU（Ray）执行。

| 组件 | 说明 |
|------|------|
| [`Engine`](api/engine.md) | 线程安全单例，通过 `get_default()` 获取。编排 加载 → 推理 → 输出 |
| [`Executor`](api/executor/executor.md) | 本地单进程执行器。管理模型加载、张量同步和节点调度 |
| [`RayExecutor`](api/executor/ray_executor.md) | 基于 Ray 的分布式执行器，支持多 GPU 推理 |

### 第 3 层：单元层 (Unit)

推理流水线的构建模块。

| 组件 | 说明 |
|------|------|
| [`InferNode`](api/nodes/core.md) | 推理节点基类。固定 `run()` 签名；数据通过 `TensorPool` 流转 |
| [`LoaderNode`](api/nodes/loaders.md) | 负责加载模型（扩散模型、文本编码器、VAE）的节点 |
| [`Generator`](api/generators/index.md) | 去噪循环实现（Wan、Qwen、VACE） |

### 第 4 层：缓存层 (Cache)

步级缓存策略，跳过去噪过程中的冗余计算。

| 组件 | 说明 |
|------|------|
| [`TeaCache`](api/cache/teacache.md) | 时序误差感知缓存 |
| [`DBCache`](api/cache/dbcache.md) | 基于增量的缓存 |
| [`DCache`](api/cache/dcache.md) | D-Cache 实现 |
| [`EasyCache`](api/cache/easycache.md) | 轻量级步缓存 |
| [`MagCache`](api/cache/magcache.md) | 基于幅度的缓存 |

### 第 5 层：算子层 (Operator)

底层计算原语，支持多后端。

| 组件 | 说明 |
|------|------|
| [注意力](api/operations/attention.md) | FlashAttention、SageAttention、SDPA、Radial-SageAttention 后端 |
| [线性层](api/operations/linear.md) | 标准和 FP8 线性后端 |
| [QKV 融合](api/operations/fuse_qkv.md) | 融合 QKV 投影（FP8 激活时自动禁用） |

### 第 6 层：硬件层 (Hardware)

GPU 和 NPU 的平台抽象。

| 组件 | 说明 |
|------|------|
| [加速器](api/accelerator.md) | 平台检测（`is_npu()`、`is_gpu()`）、dtype 归一化 |
| CUDA/GPU | NVIDIA GPU，通过 NCCL 通信 |
| NPU/Ascend | 华为昇腾，通过 HCCL + `torch_npu` |

## 数据流

```mermaid
sequenceDiagram
    participant User
    participant Pipeline
    participant Engine
    participant Executor
    participant LoaderNode
    participant InferNode
    participant TensorPool

    User->>Pipeline: from_models() / generate()
    Pipeline->>Engine: dispatch (auto_dispatch)
    Engine->>Executor: execute phases

    Note over Executor: Load Phase
    Executor->>LoaderNode: load models to CPU
    LoaderNode->>Executor: models ready

    Note over Executor: Infer Phase
    Executor->>InferNode: run(context)
    InferNode->>TensorPool: get() input tensors
    InferNode->>TensorPool: put() output tensors
    InferNode->>Executor: done

    Executor->>Engine: results
    Engine->>Pipeline: output tensors
    Pipeline->>User: generated images/videos
```

## 核心设计模式

### 声明式 Pipeline 定义

Pipeline 通过 `PipelineDefBuilder` 定义为不可变的 `PipelineDef` 数据结构：

```python
PipelineDefBuilder("wan_t2v") \
    .load(LoadPhase.TEXT_ENCODER, ...) \
    .load(LoadPhase.DIFFUSION_MODEL, ...) \
    .load(LoadPhase.VAE, ...) \
    .infer(InferPhase.TEXT_ENCODE, ...) \
    .infer(InferPhase.GENERATE, ...) \
    .infer(InferPhase.VAE_DECODE, ...) \
    .build()
```

### 节点调度策略

节点通过 `NodeDispatchPolicy` 声明其多 GPU 行为：

| 策略 | 输入 | 执行 | 输出 | 使用场景 |
|------|------|------|------|----------|
| `ALL_ALL_ALL` | 所有卡 | 所有卡 | 所有卡 | 各 GPU 独立工作 |
| `R0_R0_BCAST` | Rank 0 | Rank 0 | 广播 | 文本编码（执行一次，共享结果） |
| `ALL_R0_R0` | 所有卡 | Rank 0 | 仅 Rank 0 | 在 Rank 0 上保存/输出 |

### TensorPool

所有张量数据通过 `TensorPool` 使用 `TensorKey` 枚举流转 — 禁止通过函数参数或 metadata 传递张量。
