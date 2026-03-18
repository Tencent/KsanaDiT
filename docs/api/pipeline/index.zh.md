# 流水线

流水线模块提供声明式流水线系统，用于定义和执行推理工作流。

## 概览

- [`Pipeline`](pipeline.md) — 主入口，通过 `Pipeline.from_models()` 使用
- [`PipelineDef`](pipeline_def.md) — 不可变的流水线定义数据结构
- [`ContextBuilder`](context_builder.md) — 构建 `NodeContext` 的策略类
- [`PipelineKey`](pipeline_key.md) — 流水线类型标识
