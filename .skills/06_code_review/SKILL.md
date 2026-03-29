---
name: kdit-code-review
description: >
  kDiT 代码评审技能：除常规 CR 项（安全漏洞、格式规范、逻辑正确性）外，重点基于项目 .skills/ 中的架构设计
  和编码规范进行评审，确保代码不违背设计原则和约束。
  关键词：code review、CR、评审、review、架构合规。
---

# kDiT Code Review

## 适用场景

- Review PR / MR
- Review 自己或他人的代码改动
- 用户要求 `/kdit-code-review` 或 "帮我 review 一下"

## 评审维度（按优先级）

| 优先级 | 维度 | 说明 |
|--------|------|------|
| **P0** | 架构合规 | 基于 `.skills/02_architecture/` 和 `.skills/03_standards/` 检查 |
| **P1** | 逻辑正确性 | 边界条件、资源泄漏、并发安全、数据流完整 |
| **P2** | 安全漏洞 | OWASP Top 10 |
| **P3** | 代码质量 | 格式规范、测试覆盖、复杂度 |

详见 → [`code_review.md`](code_review.md)

## 评审流程

1. **获取改动范围** — `git diff` 或 PR diff，确认涉及的模块
2. **按优先级逐项检查** — P0 架构合规 → P1 逻辑正确性 → P2 安全漏洞 → P3 代码质量
3. **查阅关联规范** — 根据改动模块查阅对应的架构和编码规范文档
4. **输出评审结果** — 按优先级列出发现的问题，附带具体文件/行号和修复建议

## 关联规范

| 改动模块 | 必查文档 |
|---------|---------|
| Node 相关 | [`node.md`](../02_architecture/node.md)、[`pin-hub.md`](../02_architecture/pin-hub.md)、[`node-and-tensor.md`](../03_standards/node-and-tensor.md) |
| Pipeline 相关 | [`pipeline.md`](../02_architecture/pipeline.md) |
| Generator 相关 | [`generator.md`](../02_architecture/generator.md) |
| Engine/Executor | [`overview.md`](../02_architecture/overview.md) |
| Adapter 相关 | [`adapter.md`](../02_architecture/adapter.md) |
| Key/Pool 相关 | [`key-system.md`](../03_standards/key-system.md)、[`pool-key.md`](../02_architecture/pool-key.md) |

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-architecture` | P0 架构合规检查：查阅依赖方向、Node 签名、PinHub 机制等设计约束 |
| `/kdit-standards` | P0 编码规范检查：查阅 Import/类型注解/异常处理/命名规则 |
| `/kdit-quality` | P3 质量检查：查阅测试覆盖要求和格式规范 |
