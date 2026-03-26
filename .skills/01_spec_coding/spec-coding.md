---
name: spec-coding
description: >
  在编码前强制执行"需求澄清 → 可行性分析 → 结构设计 → 测试设计 → 编码实现 → 校验 → 文档同步"的七步流程。
  适用于功能开发、重构、新模块创建等需要先设计后编码的场景。
  关键词：spec coding、先设计后编码、需求确认、结构体设计、单元测试先行、vibe testing。
---


# Spec Coding — 先设计，后编码

> **路径基准**：本文件位于 `.skills/01_spec_coding/spec-coding.md`。
> 下文所有模块引用均使用 `.skills/` 内的相对路径（`../02_architecture/` 等）。
> 若通过符号链接访问本文件，请确保整个 `.skills/` 目录被链接（而非单文件链接），以保证相对路径有效。

## 适用场景

- 新功能开发、模块重构、数据结构新增等需要设计决策的编码任务
- 用户希望在编码前充分沟通需求、确认设计方案、规划测试策略

## 不适用场景

- 纯文档编写或非代码类工作

## 核心原则

> **先 Vibe Testing，再 Spec Coding。**
> 不要急于写代码——先与用户对齐需求、确认设计、约定测试，再动手实现。

> **结论不落地，等于没讨论。**
> 每次与用户交互后，必须将所有已达成的结论汇总记录在一处，不得遗漏。当待确认事项较多时，须逐条与用户核对；实现完成后，逐项回顾并打钩确认。实现前，先梳理各确认点之间的依赖关系，按依赖顺序推进。

## 关联模块文档

以下文档在工作流各步骤中按需引用，**不要预先全部阅读**，仅在对应步骤需要时查阅：

| 模块 | `.skills/` 内路径 | 何时查阅 |
|------|-------------------|---------|
| 架构总览 | [`02_architecture/overview.md`](../02_architecture/overview.md) | 第 2 步：理解现有组件关系、评估改动范围 |
| Pipeline 编排层 | [`02_architecture/pipeline.md`](../02_architecture/pipeline.md) | 第 2–3 步：涉及 Pipeline/PipelineDef 改动时 |
| Generator 去噪引擎 | [`02_architecture/generator.md`](../02_architecture/generator.md) | 第 2–3 步：涉及 Generator/Handler 改动时 |
| Node 计算单元 | [`02_architecture/node.md`](../02_architecture/node.md) | 第 2–3 步：涉及 Node 新增或修改时 |
| Node Context | [`02_architecture/node-context.md`](../02_architecture/node-context.md) | 第 3 步：设计 NodeContext/metadata 时 |
| Pin Hub | [`02_architecture/pin-hub.md`](../02_architecture/pin-hub.md) | 第 3 步：涉及 tensor 流转设计时 |
| Pool Key 体系 | [`02_architecture/pool-key.md`](../02_architecture/pool-key.md) | 第 3 步：涉及 PoolKey 设计时 |
| Key 类型体系 | [`03_standards/key-system.md`](../03_standards/key-system.md) | 第 3 步：新增 Key 枚举成员或使用 Key 时 |
| 命名规范 | [`03_standards/naming.md`](../03_standards/naming.md) | 第 3 步：类/函数命名决策时 |
| Import 与类型注解 | [`03_standards/imports-and-types.md`](../03_standards/imports-and-types.md) | 第 5 步：编码时确认导入风格和类型注解 |
| Node/Tensor API | [`03_standards/node-and-tensor.md`](../03_standards/node-and-tensor.md) | 第 5 步：编写 Node 代码时 |
| 异常处理 | [`03_standards/exception-handling.md`](../03_standards/exception-handling.md) | 第 5 步：编写异常处理逻辑时 |
| Adapter 规范 | [`02_architecture/adapter.md`](../02_architecture/adapter.md) | 第 5 步：涉及 ComfyUI adapter 改动时 |
| 设备信息 | [`02_architecture/device-info.md`](../02_architecture/device-info.md) | 第 5 步：涉及多卡/设备相关逻辑时 |
| 测试与质量 | [`04_quality/checklist.md`](../04_quality/checklist.md) | 第 4–6 步：测试设计、格式检查、文档同步 |

---

## 工作流程

### 第 1 步：需求澄清

1. 请用户描述本次开发的**目标**和**预期行为**
2. 询问用户是否已有初步的结构设计或实现思路
3. 确认输入/输出边界、依赖关系和约束条件
4. 建立一个本项目的`checklist.md`，记录所有与需求相关的确认点、设计决策、测试策略等
5. **产出**：一段简洁的需求摘要，与用户确认无误后进入下一步

### 第 2 步：可行性分析

> 📖 **必读**：先查阅 [`02_architecture/overview.md`](../02_architecture/overview.md) 理解现有组件关系。
> 若涉及特定子系统，按需查阅， 如[`02_architecture/pipeline.md`](../02_architecture/pipeline.md)、[`02_architecture/generator.md`](../02_architecture/generator.md)、[`02_architecture/node.md`](../02_architecture/node.md)。

1. 分析现有代码库的数据结构和模块，评估能否直接满足需求
2. 如果不能，列出需要**新增**或**修改**的部分（结构体、函数、模块等）
3. 给出改动范围的概要（涉及哪些文件、哪些模块）
4. **兼容性**：评估是否需要考虑兼容性，要与用户确认是否需求考虑兼容，如果不需要考虑，则不要保留历史代码，修改彻底。这个点很非常重要，要写入`checklist.md`确认
5. **产出**：可行性评估结论 + 改动范围清单，与用户确认后进入下一步

### 第 3 步：核心结构设计与命名确认

> 📖 **按需查阅**：
> - 命名决策 → [`03_standards/naming.md`](../03_standards/naming.md)（`kdit/` 内不加 `Ksana` 前缀等规则）
> - 新增 Key 枚举 → [`03_standards/key-system.md`](../03_standards/key-system.md)（ModelKey / PipelineKey / InferNodeType / IONodeType 的职责边界）
> - Node 设计 → [`02_architecture/node.md`](../02_architecture/node.md) + [`02_architecture/node-context.md`](../02_architecture/node-context.md)
> - Tensor 流转 → [`02_architecture/pin-hub.md`](../02_architecture/pin-hub.md) + [`02_architecture/pool-key.md`](../02_architecture/pool-key.md)
> - Pipeline 改动 → [`02_architecture/pipeline.md`](../02_architecture/pipeline.md)
> - Generator 改动 → [`02_architecture/generator.md`](../02_architecture/generator.md)

1. 对每个新建结构体/类/接口，说明：
   - **功能定位**：它解决什么问题
   - **字段/参数**：每个字段的含义和类型
   - **关系**：与其他结构体的关联（组合、继承、依赖等）
2. 给出**伪代码**或接口草稿，展示核心逻辑流程
3. 对关键命名提供 **2–3 个备选方案**，说明各自优劣，让用户选择
4. 分析每个新结构体是否**必要**——是否有现有结构体或功能可以复用
5. 确认新增的核心结构体、类、函数等的**目录结构与文件位置**，与用户达成一致后再动手
6. 分析梳理所有项目的实现依赖顺序, 用户已明确确认当前步骤的产出, 没有跳过任何步骤直接开始编码
7. **产出**：确认后的结构设计文档（命名、字段、关系、伪代码），用户确认后进入下一步， 并建立本项目的`checklist.md`，列入"设计模块"，包含所有讨论项的结论，每一个讨论项的结论都要记录下来，实现后挨个确认
8. **确认**：所有新概念的必要性，命名，目录结构，设计改动，执行顺序，所有都经过用户选择确认

> **重要**：当设计中引入新概念或新术语时，必须从概念本质出发给出解释，并提供命名备选方案供用户选择。

### 第 4 步：测试设计

> 📖 **必读**：查阅 [`04_quality/checklist.md`](../04_quality/checklist.md) 了解测试文件命名（`*_test.py` 后缀）、格式检查流程等要求。

1. 询问用户对测试的关注点：
   - 希望覆盖哪些场景？
   - 有哪些边界条件需要重点关注？
   - 是否有特定的测试框架偏好？
2. 基于用户反馈，设计单元测试方案，覆盖：
   - **正常路径**：核心功能的完整流程
   - **边界情况**：空值、极值、类型错误等
   - **参数有效性**：非法输入的处理
   - **用户描述的所有功能点**：逐一对照检查是否遗漏
3. **产出**：测试用例清单（含输入、预期输出、测试目的），列入`checklist.md`的"测试模块"， 与用户确认后进入下一步
4. **确认**：测试用例覆盖了用户描述的全部功能点


### 第 5 步：编码实现

> 📖 **编码前必读**（按需选择）：
> - Import 风格与类型注解 → [`03_standards/imports-and-types.md`](../03_standards/imports-and-types.md)
> - Node/Tensor API 与开发约束 → [`03_standards/node-and-tensor.md`](../03_standards/node-and-tensor.md)（编写 InferNode 时）
> - 异常处理 → [`03_standards/exception-handling.md`](../03_standards/exception-handling.md)（禁止裸 `except Exception` 等规则）
> - ComfyUI Adapter → [`02_architecture/adapter.md`](../02_architecture/adapter.md)（涉及 adapter 层改动时）
> - 设备/多卡 → [`02_architecture/device-info.md`](../02_architecture/device-info.md)（涉及设备相关逻辑时）

1. 按照已确认的设计和测试方案开始实现核心代码
2. 实现过程中如需引入新概念或术语，暂停并：
   - 解释该概念的含义
   - 提供命名备选方案
   - 等待用户确认后继续
3. 实现过程中不要检查代码合适，放在最后验证。
4. 实现完成后，使用 `git commit` 检查代码格式和变更范围，运行全部单元测试，确保通过
5. **产出**：通过所有测试的完整实现代码

### 第 6 步：校验checklist.md

> 📖 参考 [`04_quality/checklist.md`](../04_quality/checklist.md) 中的格式检查和文档同步要求。

1. 确保`checklist.md`里面的所有项目，包括设计模块，和测试模块，都已经被用户确认过，没有遗漏


### 第 7 步：更新设计到文档和skill

1. 回顾本次开发中新增或变更的概念、结构体、设计决策
2. 将这些变更同步更新到 `.skills/` 目录内对应的模块文档中，具体包括：
   - 架构变更 → 更新 [`02_architecture/`](../02_architecture/) 下对应文件
   - 编码规范变更 → 更新 [`03_standards/`](../03_standards/) 下对应文件
   - 测试/质量规范变更 → 更新 [`04_quality/checklist.md`](../04_quality/checklist.md)
3. 确保文档描述与最终实现保持一致，避免文档与代码脱节，更新所有相关的模块文档
4. **产出**：更新后的`.skills/` 和模块文档，与用户确认变更内容

---

在每一步结束时，确认以下事项：

- [ ] 每步模块的待确认事项已逐条与用户核对，实现完成后已逐项回顾打钩


## 最终检查清单

- [ ] 检查`checklist.md`，确保所有项目都已经被用户确认过，没有遗漏
- [ ] 新接受的概念或设计变化已同步更新到 `.skills/` 目录内相应模块，已经更新doc功能文档
