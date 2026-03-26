---
name: kdit-skill
description: >
  kDiT 项目知识库：包含架构概览、编码规范、Node/Tensor API、Pipeline/Generator 声明式架构、
  Import 风格、类型注解、Key 体系、工作流规范等。适用于理解项目结构、编写符合规范的代码、
  评审架构决策、以及新增模块时的参考。
---

## When to Use

当你需要：
- 理解 kDiT 的整体架构（Node、Pipeline、Generator、Adapter 等模块关系）
- 编写或修改 kDiT 代码时查阅编码规范
- 重构或者新增定义，Node，Tensor等关键定义，调整任何架构设计
- 开发新的 InferNode、Pipeline 定义、Node定义、 Generator Handler
- 了解 Tensor/Model Pool 的数据流转机制
- 确认 Import 风格、类型注解、命名规范、格式检查工具等约定
- 修复bug，需要确认是否已经达成目的



## 知识文件索引

### 00 安装

| 文件 | 主题 |
|------|------|
| [`00_install.md`](00_install.md) | Roo Code / Claude Code 的 Skill 安装方式（符号链接） |

### 01 Development Spec

| 文件 | 主题 |
|------|------|
| [`01_development_spec/development_spec.md`](01_development_spec/development_spec.md) | Development Spec：先设计后编码的 7 步工作流（需求澄清→可行性分析→结构设计→测试设计→编码实现→校验→文档同步），适用于编码实现和架构讨论/设计评审 |

### 02 架构

| 文件 | 主题 |
|------|------|
| [`02_architecture/overview.md`](02_architecture/overview.md) | 全局架构图（Mermaid）、数据流、关键设计决策 |
| [`02_architecture/node.md`](02_architecture/node.md) | Node 计算单元：IONode / InferNode、Pin 声明、run() 签名 |
| [`02_architecture/pin-hub.md`](02_architecture/pin-hub.md) | PinHub 沙箱化数据访问器：核心机制、沙箱约束、构建位置 |
| [`02_architecture/pool-key.md`](02_architecture/pool-key.md) | ModelPoolKey / TensorPoolKey：PoolKey 间接寻址机制 |
| [`02_architecture/device-info.md`](02_architecture/device-info.md) | DeviceInfo 设备信息：frozen dataclass、Executor 注入 |
| [`02_architecture/node-context.md`](02_architecture/node-context.md) | NodeContext：Node 间传递的可序列化上下文 |
| [`02_architecture/pipeline.md`](02_architecture/pipeline.md) | Pipeline 编排层：PipelineDef、ContextBuilder、ExtraInputs、DAG 校验 |
| [`02_architecture/generator.md`](02_architecture/generator.md) | Generator 去噪引擎：GeneratorDef、Handler 注入、BaseLatent/AuxLatent 语义 |
| [`02_architecture/adapter.md`](02_architecture/adapter.md) | Adapter / ComfyUI：适配层、依赖方向、包结构 |

### 03 编码规范

| 文件 | 主题 |
|------|------|
| [`03_standards/imports-and-types.md`](03_standards/imports-and-types.md) | Import 风格（方案 B）、类型注解规范、未使用的变量import注释 |
| [`03_standards/key-system.md`](03_standards/key-system.md) | Key 类型体系（ModelKey / PipelineKey / ModelPoolKey / TensorPoolKey / InferNodeType / IONodeType） |
| [`03_standards/node-and-tensor.md`](03_standards/node-and-tensor.md) | Node/Tensor API、Ownership 关系、InferNode 开发规范 |
| [`03_standards/exception-handling.md`](03_standards/exception-handling.md) | 异常处理规范：禁止裸 except Exception、常见异常类型速查 |
| [`03_standards/naming.md`](03_standards/naming.md) | 类命名规范：Ksana 前缀规则 |

### 04 质量保障

| 文件 | 主题 |
|------|------|
| [`04_quality/verification.md`](04_quality/verification.md) | 交付验收：单元测试、代码格式检查、文档同步规范 |
