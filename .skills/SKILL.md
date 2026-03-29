---
name: kdit-skill
description: >
  kDiT 项目知识库总索引：包含所有 skill 入口和知识文件的完整索引。
  适用于快速定位需要查阅的文档，或了解项目 skill 体系全貌。
  关键词：kdit、索引、知识库、skill 列表。
---

# kDiT 知识库总索引

> 不确定查哪个文档时，先看这里。

## 独立 Skill 速查

| Skill | 命令 | 入口 | 说明 |
|-------|------|------|------|
| kdit-install | `/kdit-install` | [`00_install/SKILL.md`](00_install/SKILL.md) | 安装配置指南 |
| kdit-development-spec | `/kdit-development-spec` | [`01_development_spec/SKILL.md`](01_development_spec/SKILL.md) | 先设计后编码 7 步工作流 |
| kdit-architecture | `/kdit-architecture` | [`02_architecture/SKILL.md`](02_architecture/SKILL.md) | 架构知识库（9 篇） |
| kdit-standards | `/kdit-standards` | [`03_standards/SKILL.md`](03_standards/SKILL.md) | 编码规范（5 篇） |
| kdit-quality | `/kdit-quality` | [`04_quality/SKILL.md`](04_quality/SKILL.md) | 测试与质量保障 |
| kdit-bugfix | `/kdit-bugfix` | [`05_bugfix/SKILL.md`](05_bugfix/SKILL.md) | Bug 修复 5 步工作流 |
| kdit-code-review | `/kdit-code-review` | [`06_code_review/SKILL.md`](06_code_review/SKILL.md) | 代码评审（P0-P3 优先级） |

---

## 知识文件索引

### 00_install — 安装配置

| 文件 | 说明 |
|------|------|
| [`install.md`](00_install/install.md) | 环境安装、灰度控制 |

### 01_development_spec — 开发流程

| 文件 | 说明 |
|------|------|
| [`development_spec.md`](01_development_spec/development_spec.md) | 需求澄清 → 可行性 → 设计 → 测试 → 编码 → 校验 → 文档同步 |

### 02_architecture — 架构设计

| 文件 | 说明 |
|------|------|
| [`overview.md`](02_architecture/overview.md) | 架构总览、组件关系、数据流、Ownership |
| [`pipeline.md`](02_architecture/pipeline.md) | Pipeline 编排层、DAG、ExtraInputs、ContextBuilder |
| [`generator.md`](02_architecture/generator.md) | Generator 去噪引擎、BaseLatent/AuxLatent、Handler |
| [`node.md`](02_architecture/node.md) | Node 计算单元、Def/Pin、dispatch_policy |
| [`node-context.md`](02_architecture/node-context.md) | NodeContext 上下文、metadata 禁止含 tensor |
| [`pin-hub.md`](02_architecture/pin-hub.md) | PinHub 沙箱化数据访问器 |
| [`pool-key.md`](02_architecture/pool-key.md) | TensorPool/ModelPool、PoolKey、引用计数 |
| [`adapter.md`](02_architecture/adapter.md) | ComfyUI 适配层、依赖方向 |
| [`device-info.md`](02_architecture/device-info.md) | DeviceInfo 设备信息 |

### 03_standards — 编码规范

| 文件 | 说明 |
|------|------|
| [`imports-and-types.md`](03_standards/imports-and-types.md) | Import 风格、类型注解、Lint 抑制 |
| [`key-system.md`](03_standards/key-system.md) | ModelKey/PipelineKey/InferNodeType/IONodeType 体系 |
| [`node-and-tensor.md`](03_standards/node-and-tensor.md) | InferNode 编码实操规范、TensorKey 使用 |
| [`exception-handling.md`](03_standards/exception-handling.md) | 异常处理规范 |
| [`naming.md`](03_standards/naming.md) | 类命名规范（kdit/ 内不加 Ksana 前缀） |

### 04_quality — 质量保障

| 文件 | 说明 |
|------|------|
| [`verification.md`](04_quality/verification.md) | 测试命名、pre-commit、文档同步 |

### 05_bugfix — Bug 修复

| 文件 | 说明 |
|------|------|
| [`SKILL.md`](05_bugfix/SKILL.md) | 完整 5 步 bug 修复工作流 |

### 06_code_review — 代码评审

| 文件 | 说明 |
|------|------|
| [`SKILL.md`](06_code_review/SKILL.md) | 完整评审规范（P0 架构合规 → P3 代码质量） |
