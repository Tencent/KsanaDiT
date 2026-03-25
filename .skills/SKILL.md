---
name: kdit-knowledge-base
description: >
  kDiT 项目知识库：包含架构概览、编码规范、Node/Tensor API、Pipeline/Generator 声明式架构、
  Import 风格、类型注解、Key 体系、工作流规范等。适用于理解项目结构、编写符合规范的代码、
  评审架构决策、以及新增模块时的参考。
---

## When to Use

当你需要：
- 理解 kDiT 的整体架构（Node、Pipeline、Generator、Adapter 等模块关系）
- 编写或修改 kDiT 代码时查阅编码规范
- 开发新的 InferNode、Pipeline 定义、Generator Handler
- 了解 Tensor/Model Pool 的数据流转机制
- 确认 Import 风格、类型注解、命名规范等约定

**不适用于**：ComfyUI 插件节点的开发（参考 `kdit/adapter/comfyui/` 下的代码）、性能调优（参考 benchmark 目录）。

## 知识文件索引

### 架构

| 文件 | 主题 |
|------|------|
| [`architecture.md`](architecture.md) | 核心架构概览：模块职责、Node 体系、Pipeline/Generator 声明式架构、Adapter 依赖规则、类命名规范、Settings YAML 规范 |

### 编码规范

| 文件 | 主题 |
|------|------|
| [`coding.md`](coding.md) | 编码规范总索引 |
| [`coding/spec-coding.md`](coding/spec-coding.md) | Spec Coding：先设计后编码的 5 步工作流（需求澄清→可行性分析→结构设计→测试设计→编码实现） |
| [`coding/imports-and-types.md`](coding/imports-and-types.md) | Import 风格（方案 B）、类型注解规范、Lint 抑制注释 |
| [`coding/key-system.md`](coding/key-system.md) | Key 类型体系（ModelKey / PipelineKey / InferNodeType） |
| [`coding/node-and-tensor.md`](coding/node-and-tensor.md) | Node/Tensor API、Ownership 关系、InferNode 开发规范 |
| [`coding/generator.md`](coding/generator.md) | BaseLatent 与 AuxLatent 语义规范 |
| [`coding/workflow.md`](coding/workflow.md) | 单元测试、代码格式检查、文档同步规范 |
