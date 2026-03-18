# API 参考

本节提供 `kdit` 包所有公开模块的自动生成 API 文档。

## 包结构

```
kdit/
├── engine/          # 引擎 — 线程安全单例，编排推理流程
├── pipelines/       # 流水线 — 声明式流水线定义
├── config/          # 配置数据类
├── nodes/           # 推理节点 / 加载节点 — 流水线构建模块
│   ├── core/        # 基类和工厂
│   ├── infers/      # 推理节点（文本编码、生成、VAE 解码）
│   └── loaders/     # 模型加载节点
├── models/          # 模型封装（扩散模型、文本编码器、VAE）
├── generators/      # 去噪循环实现
├── executor/        # 本地和 Ray 执行器
├── operations/      # 底层算子（注意力、线性层、QKV 融合）
├── cache/           # 步级缓存策略
├── tensor/          # TensorPool 和 TensorKey
├── accelerator/     # 平台检测和 dtype 工具
├── sample_solvers/  # ODE/SDE 求解器（Euler、UniPC、DPM）
├── scheduler/       # 批调度
├── memory/          # 内存管理
└── utils/           # 共享工具函数
```

## 快速导航

| 模块 | 说明 |
|------|------|
| [引擎](engine.md) | 线程安全单例，`auto_dispatch` 装饰器 |
| [流水线](pipeline/index.md) | `Pipeline.from_models()`、`PipelineDef`、`ContextBuilder` |
| [配置](config/index.md) | 所有配置数据类 |
| [节点](nodes/index.md) | `InferNode`、`LoaderNode`、工厂 |
| [模型](models/index.md) | 模型封装和 `ModelKey` |
| [生成器](generators/index.md) | 去噪循环（Wan、Qwen、VACE） |
| [执行器](executor/index.md) | 本地和 Ray 执行器 |
| [算子](operations/index.md) | 注意力、线性层、QKV 融合 |
| [缓存](cache/index.md) | TeaCache、DBCache、DCache 等 |
| [张量](tensor/index.md) | `TensorPool`、`TensorKey` |
| [加速器](accelerator.md) | 平台检测、dtype |
| [采样求解器](sample_solvers.md) | Euler、UniPC、DPM 求解器 |
| [调度器](scheduler.md) | 批调度 |
| [内存](memory.md) | 固定内存管理器 |
| [工具函数](utils.md) | 日志、加载、性能分析等 |
