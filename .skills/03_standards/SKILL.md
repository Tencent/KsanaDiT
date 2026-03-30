---
name: kdit-standards
description: >
  kDiT 编码规范：Import 风格（方案 B）、Python 3.10+ 类型注解、Key 类型体系、
  Node/Tensor API 开发约束、异常处理规则、类命名规范。
  关键词：standards、规范、import、类型注解、Key、异常处理、命名。
---

# kDiT 编码规范

## 适用场景

- 编写代码时确认 Import/类型注解/Key 体系/异常处理/命名
- Code review 时检查规范合规性

## 文档索引

| 文件 | 说明 |
|------|------|
| [`imports-and-types.md`](imports-and-types.md) | Import 风格、类型注解、Lint 抑制 |
| [`key-system.md`](key-system.md) | ModelKey/PipelineKey/InferNodeType/IONodeType 体系 |
| [`node-and-tensor.md`](node-and-tensor.md) | InferNode 编码实操规范、TensorKey 使用 |
| [`exception-handling.md`](exception-handling.md) | 异常处理规范 |
| [`naming.md`](naming.md) | 类命名规范（kdit/ 内不加 Ksana 前缀） |

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-architecture` | 理解模块关系后再确认规范 |
| `/kdit-quality` | 测试设计、格式检查 |
