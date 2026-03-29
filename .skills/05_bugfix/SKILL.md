---
name: kdit-bugfix
description: >
  Bug 修复工作流：修复 bug 时强制执行"根因分析 → 反思测试缺口 → 修复代码 → 补充单元测试 → 验证"流程。
  确保每次 bug 修复都附带回归测试，并反思为什么现有测试没能覆盖到。
  关键词：bug fix、修复、regression、回归测试、测试缺口。
---

# Bug 修复工作流

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

## 详细流程

### 第 1 步：复现与定位

1. 确认 bug 的复现步骤和预期行为
2. 定位根因（具体到函数/行），分析是逻辑错误、边界遗漏还是接口误用
3. 记录根因分析结论

### 第 2 步：反思测试缺口

1. 检查现有测试为什么没有覆盖到这个 bug
2. 分析是测试场景遗漏、mock 过度、还是测试数据不够
3. 记录测试缺口原因，作为补测试的指导

### 第 3 步：编写回归测试

1. 先写测试，确认测试能复现 bug（测试应该失败）
2. 测试文件遵循 `*_test.py` 命名规范
3. 覆盖 bug 场景 + 相关边界条件

### 第 4 步：修复代码

1. 最小改动修复，不做额外重构
2. 确认回归测试通过

### 第 5 步：验证与总结

1. 运行全量测试 `pytest -s -v tests/kdit`
2. 运行 `pre-commit run` 检查格式
3. 输出 4 项总结：根因、修复方案、测试缺口原因、补充的测试

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
