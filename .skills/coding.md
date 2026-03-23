# kDiT Coding Skills

本文件是 kDiT 编码规范的索引。详细规范已按主题拆分到 `coding/` 子目录下的模块文件中。

---

## 模块索引

| 模块文件 | 包含章节 | 主题 |
|---------|---------|------|
| [`coding/imports-and-types.md`](coding/imports-and-types.md) | §1, §2, §11 | Import 风格、类型注解、Lint 抑制注释 |
| [`coding/key-system.md`](coding/key-system.md) | §3 | Key 类型体系（ModelKey / PipelineKey / InferNodeType） |
| [`coding/node-and-tensor.md`](coding/node-and-tensor.md) | §4, §5, §6 | Node / Tensor API、Ownership 关系、InferNode 开发规范 |
| [`architecture.md`](architecture.md) | §5–§8 | Adapter 依赖方向、类命名规范、Metadata 重构、Pipeline 声明式架构、Settings YAML |
| [`coding/generator.md`](coding/generator.md) | §12 | BaseLatent 与 AuxLatent 语义规范 |
| [`coding/workflow.md`](coding/workflow.md) | §13, §14, §15 | 单元测试同步、代码格式检查、文档同步 |

---

## 章节速查

| § | 标题 | 模块文件 |
|---|------|---------|
| 1 | Import 风格规范（方案 B） | [`imports-and-types.md`](coding/imports-and-types.md) |
| 2 | 类型注解与导入规范 | [`imports-and-types.md`](coding/imports-and-types.md) |
| 3 | Key 类型体系设计规范 | [`key-system.md`](coding/key-system.md) |
| 4 | V5 Node / Tensor API 规范 | [`node-and-tensor.md`](coding/node-and-tensor.md) |
| 5 | Ownership 与状态关系图 | [`node-and-tensor.md`](coding/node-and-tensor.md) |
| 6 | InferNode 开发规范 | [`node-and-tensor.md`](coding/node-and-tensor.md) |
| 7 | Adapter 依赖方向规则 | [`architecture.md`](architecture.md) §5 |
| 8 | 类命名规范：去除 `Ksana` 前缀 | [`architecture.md`](architecture.md) §6 |
| 9 | NodeContext metadata 重构（TODO） | [`architecture.md`](architecture.md) §7 |
| 10 | Pipeline 声明式架构 | [`architecture.md`](architecture.md) §3 |
| 11 | Lint 抑制注释规范 | [`imports-and-types.md`](coding/imports-and-types.md) |
| 12 | BaseLatent 与 AuxLatent 语义规范 | [`generator.md`](coding/generator.md) |
| 13 | 新增功能必须同步单元测试 | [`workflow.md`](coding/workflow.md) |
| 14 | 代码格式检查规范 | [`workflow.md`](coding/workflow.md) |
| 15 | 代码修改必须同步更新文档 | [`workflow.md`](coding/workflow.md) |
