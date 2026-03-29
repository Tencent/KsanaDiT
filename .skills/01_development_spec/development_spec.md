---
name: kdit-development-spec
description: >
  在编码或架构设计前强制执行"需求澄清 → 可行性分析 → 结构设计 → 测试设计 → 编码实现 → 校验 → 文档同步"的七步流程。
  适用于功能开发、重构、新模块创建、架构讨论与设计评审等需要先设计后实现的场景。
  关键词：development spec、先设计后编码、架构讨论、设计评审、需求确认、结构体设计、单元测试先行、vibe testing。
---


# Development Spec — 先设计，后编码

> **路径基准**：本文件位于 `.skills/01_development_spec/development_spec.md`。
> 下文所有模块引用均使用 `.skills/` 内的相对路径（`../02_architecture/` 等）。

## 适用场景

- 新功能开发、模块重构、数据结构新增等需要设计决策的编码任务
- 架构讨论与设计评审：系统架构方案讨论、模块设计评审、技术选型决策
- 用户希望在编码或设计前充分沟通需求、确认设计方案、规划测试策略

## 不适用场景

- 纯非技术类文档编写

## 快速通道

> 如果改动范围明确（≤ 3 个文件）、无新概念引入、无架构决策，可走快速通道：
> 合并步骤 1-3 为一个"需求+设计确认"，跳过 step 0 的模板创建和独立的 step 6 校验。
> 仍需创建 checklist.md 记录结论，仍需跑测试和 pre-commit。

## 核心原则

> **先 Vibe Testing，再 Spec Coding。**
> 不要急于写代码——先与用户对齐需求、确认设计、约定测试，再动手实现。

> **结论不落地，等于没讨论。**
> 每次与用户交互后，必须将所有已达成的结论汇总记录在一处，不得遗漏。当待确认事项较多时，须逐条与用户核对；实现完成后，逐项回顾并打钩确认。实现前，先梳理各确认点之间的依赖关系，按依赖顺序推进。

> **下一步验证上一步，自己不打自己的钩。**
> 每一步的 checklist 由**下一步开始时**回顾并打钩确认，而非当前步骤自行打钩。当前步骤只负责**记录结论和待确认项**，打钩权交给下一步。这确保了每一步的产出都经过独立回顾验证。

---

## design_drafts 目录与文件规范

每次任务启动时，在 `design_drafts/` 下创建一个以 `<feature>` 命名的独立文件夹，所有过程文档和 checklist 都存放在该文件夹内。

### 目录结构

```
design_drafts/<feature>/
├── checklist.md              # 唯一真相源：贯穿七步的累积式 checklist
└── NN_xxx.md                 # 按需创建：走到哪步创建哪步的环节文件
```

### 命名规则

- **`<feature>`**：本次任务的简短标识，例如 `nodedef_refactor`、`vace_pipeline`
- **checklist.md**：固定名称，是最终的唯一真相源和交付记录
- **环节 md 文件**：按 `序号_英文环节名.md` 格式命名（如 `01_requirements.md`、`03_design.md`），**走到该步骤时再创建**，不要预先创建全部模板

### checklist.md 的角色

- **唯一真相源**：记录所有步骤的确认项及其最终状态
- **累积式 TODO list**：不仅记录当前步骤的结论，更承载所有历史步骤的确认项
- **最终交付记录**：流程结束后，checklist.md 的最终状态即为项目的完整交付记录

### 打钩规则（核心机制）

| 操作 | 说明 |
|------|------|
| **当前步骤** | 只负责**记录结论和待确认项**到 checklist.md，所有条目保持 `- [ ]` 未打钩状态 |
| **下一步开始时** | 先读取整个 checklist.md，回顾**上一步**的所有条目，逐条验证后打钩 `- [x]`，同时回顾所有历史步骤的状态 |
| **最后一步（第 7 步）** | 由最终检查环节负责打钩第 7 步自身的条目 |

> **为什么这样设计？** 自己不打自己的钩，确保每一步的产出都经过下一步的独立回顾验证，避免遗漏或自我确认偏差。

---

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
| 测试与质量 | [`04_quality/verification.md`](../04_quality/verification.md) | 第 4–6 步：测试设计、格式检查、文档同步 |

---

## 工作流程

### 第 0 步：创建 design_drafts 目录与 Checklist

1. 根据用户描述的任务，确定 `<feature>` 标识名
2. 创建 `design_drafts/<feature>/` 目录
3. 创建 `design_drafts/<feature>/checklist.md`，初始内容包含：
   - 标题和打钩规则说明
   - 空的"第 0 步"到"第 7 步"各章节框架
   - 每个章节包含"设计模块"/"实现顺序"/"测试模块"等子章节（按需）

### 第 1 步：需求澄清

> 📋 **checklist 操作**：
> - 开始前：读取整个 checklist.md，**回顾并打钩第 0 步**的所有条目
> - 过程中：将讨论内容写入 `01_requirements.md`（此时创建该文件）
> - 结束时：将需求确认项写入 checklist.md 第 1 步章节（保持 `- [ ]` 未打钩）

1. **先分析用户已提供的信息**，提取出能确认的需求、目标、约束，列出已知项
2. 仅针对**缺失的信息**向用户提问（目标、预期行为、输入/输出边界、依赖关系等）
3. 确认用户是否已有初步的结构设计或实现思路
4. 将讨论的完整内容记录到 `01_requirements.md`
5. 将所有与需求相关的确认点、设计决策、测试策略等写入 checklist.md
6. **产出**：一段简洁的需求摘要（写入 `01_requirements.md`），与用户确认无误后进入下一步

### 第 2 步：可行性分析

> 📖 **必读**：先查阅 [`02_architecture/overview.md`](../02_architecture/overview.md) 理解现有组件关系。
> 若涉及特定子系统，按需查阅，如 [`02_architecture/pipeline.md`](../02_architecture/pipeline.md)、[`02_architecture/generator.md`](../02_architecture/generator.md)、[`02_architecture/node.md`](../02_architecture/node.md)。
>
> 📋 **checklist 操作**：
> - 开始前：读取整个 checklist.md，**回顾并打钩第 1 步**的所有条目
> - 过程中：将讨论内容写入 `02_feasibility.md`（此时创建该文件）
> - 结束时：将可行性结论写入 checklist.md 第 2 步章节（保持 `- [ ]` 未打钩）

1. 分析现有代码库的数据结构和模块，评估能否直接满足需求
2. 如果不能，列出需要**新增**或**修改**的部分（结构体、函数、模块等）
3. 给出改动范围的概要（涉及哪些文件、哪些模块）
4. **兼容性**：评估是否需要考虑兼容性，要与用户确认是否需求考虑兼容，如果不需要考虑，则不要保留历史代码，修改彻底。这个点很非常重要，要写入 checklist 确认
5. 将讨论的完整内容记录到 `02_feasibility.md`
6. **产出**：可行性评估结论 + 改动范围清单（写入 `02_feasibility.md`），与用户确认后进入下一步

### 第 3 步：核心结构设计与命名确认

> 📖 **按需查阅**：
> - 命名决策 → [`03_standards/naming.md`](../03_standards/naming.md)（`kdit/` 内不加 `Ksana` 前缀等规则）
> - 新增 Key 枚举 → [`03_standards/key-system.md`](../03_standards/key-system.md)（ModelKey / PipelineKey / InferNodeType / IONodeType 的职责边界）
> - Node 设计 → [`02_architecture/node.md`](../02_architecture/node.md) + [`02_architecture/node-context.md`](../02_architecture/node-context.md)
> - Tensor 流转 → [`02_architecture/pin-hub.md`](../02_architecture/pin-hub.md) + [`02_architecture/pool-key.md`](../02_architecture/pool-key.md)
> - Pipeline 改动 → [`02_architecture/pipeline.md`](../02_architecture/pipeline.md)
> - Generator 改动 → [`02_architecture/generator.md`](../02_architecture/generator.md)
>
> 📋 **checklist 操作**：
> - 开始前：读取整个 checklist.md，**回顾并打钩第 2 步**的所有条目
> - 过程中：将讨论内容写入 `03_design.md`（此时创建该文件）
> - 结束时：将设计结论、命名决策、实现顺序写入 checklist.md 第 3 步"设计模块"章节（保持 `- [ ]` 未打钩）

1. 对每个新建结构体/类/接口，说明：
   - **功能定位**：它解决什么问题
   - **字段/参数**：每个字段的含义和类型
   - **关系**：与其他结构体的关联（组合、继承、依赖等）
2. 给出**伪代码**或接口草稿，展示核心逻辑流程
3. 对关键命名提供 **2–3 个备选方案**，说明各自优劣，让用户选择
4. 分析每个新结构体是否**必要**——是否有现有结构体或功能可以复用
5. 确认新增的核心结构体、类、函数等的**目录结构与文件位置**，与用户达成一致后再动手
6. 分析梳理所有项目的实现依赖顺序，用户已明确确认当前步骤的产出，没有跳过任何步骤直接开始编码
7. 将讨论的完整内容记录到 `03_design.md`
8. **产出**：确认后的结构设计文档（命名、字段、关系、伪代码），用户确认后进入下一步，并在 checklist.md 中列入"设计模块"，包含所有讨论项的结论，每一个讨论项的结论都要记录下来，实现后挨个确认
9. **确认**：所有新概念的必要性，命名，目录结构，设计改动，执行顺序，所有都经过用户选择确认

> **重要**：当设计中引入新概念或新术语时，必须从概念本质出发给出解释，并提供命名备选方案供用户选择。

### 第 4 步：测试设计

> 📖 **必读**：查阅 [`04_quality/verification.md`](../04_quality/verification.md) 了解测试文件命名（`*_test.py` 后缀）、格式检查流程等要求。
>
> 📋 **checklist 操作**：
> - 开始前：读取整个 checklist.md，**回顾并打钩第 3 步**的所有条目
> - 过程中：将讨论内容写入 `04_test_design.md`（此时创建该文件）
> - 结束时：将测试用例清单写入 checklist.md 第 4 步"测试模块"章节（保持 `- [ ]` 未打钩）

1. 询问用户对测试的关注点：
   - 希望覆盖哪些场景？
   - 有哪些边界条件需要重点关注？
   - 是否有特定的测试框架偏好？
2. 基于用户反馈，设计单元测试方案，覆盖：
   - **正常路径**：核心功能的完整流程
   - **边界情况**：空值、极值、类型错误等
   - **参数有效性**：非法输入的处理
   - **用户描述的所有功能点**：逐一对照检查是否遗漏
3. 将讨论的完整内容记录到 `04_test_design.md`
4. **产出**：测试用例清单（含输入、预期输出、测试目的），列入 checklist.md 的"测试模块"，与用户确认后进入下一步
5. **确认**：测试用例覆盖了用户描述的全部功能点


### 第 5 步：编码实现

> 📖 **编码前必读**（按需选择）：
> - Import 风格与类型注解 → [`03_standards/imports-and-types.md`](../03_standards/imports-and-types.md)
> - Node/Tensor API 与开发约束 → [`03_standards/node-and-tensor.md`](../03_standards/node-and-tensor.md)（编写 InferNode 时）
> - 异常处理 → [`03_standards/exception-handling.md`](../03_standards/exception-handling.md)（禁止裸 `except Exception` 等规则）
> - ComfyUI Adapter → [`02_architecture/adapter.md`](../02_architecture/adapter.md)（涉及 adapter 层改动时）
> - 设备/多卡 → [`02_architecture/device-info.md`](../02_architecture/device-info.md)（涉及设备相关逻辑时）
>
> 📋 **checklist 操作**：
> - 开始前：读取整个 checklist.md，**回顾并打钩第 4 步**的所有条目
> - 过程中：将实现记录写入 `05_implementation.md`（此时创建该文件），按 checklist "实现顺序"逐项推进
> - 结束时：将实现完成项写入 checklist.md 第 5 步章节（保持 `- [ ]` 未打钩）

1. 按照已确认的设计和测试方案开始实现核心代码
2. 实现过程中如需引入新概念或术语，暂停并：
   - 解释该概念的含义
   - 提供命名备选方案
   - 等待用户确认后继续
3. 实现过程中不要检查代码格式，放在最后验证。
4. 实现完成后，先做静态检查，再跑测试。两者检查维度不同，缺一不可：
   - **静态检查**：`git add <本次改动的文件>` 然后 `pre-commit run`（仅检查 staged 文件）。捕获 F821（注解引用未定义名称）、格式问题、unused import 等。**注意**：`from __future__ import annotations` 使注解运行时不求值，pytest 无法发现注解中的未定义名称，只有 ruff 能发现
   - **运行时测试**：`pytest -s -v tests/kdit`。捕获逻辑错误、功能缺陷、运行时异常
5. 两项都通过后再 `git commit`
6. 将实现过程的关键决策和变更记录到 `05_implementation.md`
7. **产出**：通过 pre-commit + 全部测试的完整实现代码

### 第 6 步：校验与验收

> 📖 参考 [`04_quality/verification.md`](../04_quality/verification.md) 中的格式检查和文档同步要求。
>
> 📋 **checklist 操作**：
> - 开始前：读取整个 checklist.md，**回顾并打钩第 5 步**的所有条目
> - 过程中：逐项审查整个 checklist.md，将校验结果写入 `06_verification.md`（此时创建该文件）
> - 结束时：将校验确认项写入 checklist.md 第 6 步章节（保持 `- [ ]` 未打钩）

1. **checklist 逐项审查**：读取 checklist.md，逐项检查"设计模块"/"实现顺序"/"测试模块"中的所有条目，确保每一项都已与用户确认过，没有遗漏
2. **完整测试套件**：运行 `pytest -s -v tests/kdit`，确认全量测试通过（不只是本次新增的测试）
3. **改动范围一致性**：对比 `git diff` 的实际改动文件与 step 2 确定的改动范围清单，检查是否引入了计划外的变更
4. **设计一致性**：核对实际实现与 step 3 的设计文档是否一致（类名、字段、接口签名、目录结构），如有偏差需记录原因
5. 如有未确认项或偏差，回到对应步骤与用户补齐确认
6. 将校验结果记录到 `06_verification.md`


### 第 7 步：更新设计到文档和 skill

> 📋 **checklist 操作**：
> - 开始前：读取整个 checklist.md，**回顾并打钩第 6 步**的所有条目
> - 过程中：将文档更新记录写入 `07_doc_sync.md`（此时创建该文件）
> - 结束时：将文档更新项写入 checklist.md 第 7 步章节，并由**最终检查**环节打钩

1. 回顾本次开发中新增或变更的概念、结构体、设计决策
2. 将这些变更同步更新到 `.skills/` 目录内对应的模块文档中，具体包括：
   - 架构变更 → 更新 [`02_architecture/`](../02_architecture/) 下对应文件
   - 编码规范变更 → 更新 [`03_standards/`](../03_standards/) 下对应文件
   - 测试/质量规范变更 → 更新 [`04_quality/verification.md`](../04_quality/verification.md)
3. 确保文档描述与最终实现保持一致，避免文档与代码脱节，更新所有相关的模块文档
4. 将更新记录写入 `07_doc_sync.md`
5. **产出**：更新后的 `.skills/` 和模块文档，与用户确认变更内容

---

## 步骤间检查（累积式 + 交叉验证）

> **核心理念**：每一步的 check 不是只检查当前步骤——而是回顾 **checklist.md 中所有历史累积项目**。
> 上一步的全部产出 = 下一步的 TODO list。确保不会因为步骤推进而遗忘早期讨论的结论。
> **打钩权归下一步**：当前步骤只记录，下一步负责验证并打钩。

在每一步**开始时**，执行以下操作：

1. 读取整个 checklist.md（不仅是上一步的章节）
2. 回顾**上一步**的所有条目，逐条验证后打钩 `- [x]`
3. 回顾所有历史步骤的状态，确认无遗漏
4. 如发现上一步有未完成或有疑问的条目，先与用户确认后再继续

在每一步**结束时**，执行以下操作：

1. 将当前步骤的讨论内容写入对应的环节 md 文件
2. 将确认结论提炼写入 checklist.md 当前步骤章节（保持 `- [ ]` 未打钩）
3. 确认本步产出已完整记录，作为下一步的 TODO 输入

---

## 最终检查清单

在第 7 步完成后，执行最终检查：

- [ ] 回顾并打钩第 7 步的所有条目
- [ ] 逐项审查整个 checklist.md（从第 0 步到第 7 步的所有累积项），确保每一项都已被打钩确认且已实现，无遗漏
- [ ] 所有环节 md 文件内容完整，可供用户阅读回顾
- [ ] 新接受的概念或设计变化已同步更新到 `.skills/` 目录内相应模块，已更新 doc 功能文档
- [ ] checklist.md 的最终状态即为项目的完整交付记录
