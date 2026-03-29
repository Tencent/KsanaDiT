---
name: kdit-development-spec
description: >
  在编码或架构设计前强制执行"需求澄清 → 可行性分析 → 结构设计 → 测试设计 → 编码实现 → 校验 → 文档同步"的七步流程。
  适用于功能开发、重构、新模块创建、架构讨论与设计评审等需要先设计后实现的场景。
  关键词：development spec、先设计后编码、架构讨论、设计评审、需求确认、结构体设计、单元测试先行、vibe testing。
---

# Development Spec 入口

> 本文件是 development_spec skill 的入口。完整 7 步工作流请阅读 [`development_spec.md`](development_spec.md)。

## 适用场景

- 新功能开发、模块重构、数据结构新增等需要设计决策的编码任务
- 架构讨论与设计评审：系统架构方案讨论、模块设计评审、技术选型决策
- 用户希望在编码或设计前充分沟通需求、确认设计方案、规划测试策略

## 不适用场景

- 纯非技术类文档编写

## 核心原则

1. **先 Vibe Testing，再 Spec Coding** — 先对齐需求、确认设计、约定测试，再动手
2. **结论不落地，等于没讨论** — 每次交互后必须汇总记录结论
3. **下一步验证上一步，自己不打自己的钩** — 交叉验证机制

## 7 步流程概览

| 步骤 | 内容 |
|------|------|
| 0 | 创建 `design_drafts/<feature>/` 目录与 checklist |
| 1 | 需求澄清 |
| 2 | 可行性分析 |
| 3 | 核心结构设计与命名确认 |
| 4 | 测试设计 |
| 5 | 编码实现 |
| 6 | 校验 checklist |
| 7 | 更新文档和 skill |

详见 → [`development_spec.md`](development_spec.md)

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-architecture` | 第 2 步：理解现有组件关系、评估改动范围 |
| `/kdit-standards` | 第 3/5 步：命名、Import、类型注解、异常处理等规范 |
| `/kdit-quality` | 第 4/6 步：测试设计、格式检查、文档同步 |
