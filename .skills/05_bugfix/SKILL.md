---
name: kdit-bugfix
description: >
  Bug 修复工作流：修复 bug 时强制执行"根因分析 → 反思测试缺口 → 修复代码 → 补充单元测试 → 验证"流程。
  确保每次 bug 修复都附带回归测试，并反思为什么现有测试没能覆盖到。
  关键词：bug fix、修复、regression、回归测试、测试缺口。
---

# Bug 修复工作流

修 bug 不只是改代码——必须反思为什么测试没拦住，并补回归测试。

## 适用场景

- 修复已发现的 bug（无论来源：用户报告、CI 失败、自测发现）
- 修复 pre-commit / ruff 报出的逻辑问题（非纯格式问题）

## 不适用场景

- 纯格式修复（走 pre-commit 即可）
- 新功能开发（走 `/kdit-development-spec`）

## 流程概览（5 步）

1. **复现与定位** — 明确根因
2. **反思测试缺口** — 为什么测试没拦住（核心步骤）
3. **编写回归测试** — 先写测试，后改代码
4. **修复代码** — 最小改动修复
5. **验证与总结** — 全量测试 + 4 项总结输出

详见 → [`bugfix.md`](bugfix.md)

## 关联规范

| 规范 | 路径 |
|------|------|
| 测试规范 | [`04_quality/verification.md`](../04_quality/verification.md) |
| 异常处理 | [`03_standards/exception-handling.md`](../03_standards/exception-handling.md) |
| Node/Tensor API | [`03_standards/node-and-tensor.md`](../03_standards/node-and-tensor.md) |
| 架构总览 | [`02_architecture/overview.md`](../02_architecture/overview.md) |

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-quality` | 编写回归测试时查阅测试命名和目录规范 |
| `/kdit-architecture` | 定位根因需理解模块关系时 |
| `/kdit-standards` | bug 涉及异常处理或 Node API 时 |
