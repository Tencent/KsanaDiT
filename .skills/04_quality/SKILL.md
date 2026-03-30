---
name: kdit-quality
description: >
  kDiT 质量保障与交付验收：单元测试规范（*_test.py 命名、tests/kdit/ 镜像结构）、
  pre-commit 格式检查（black 120字符、ruff）、pytest vs ruff 维度差异、文档同步规则。
  关键词：quality、测试、test、pytest、pre-commit、ruff、black、验收。
---

# kDiT 质量保障

## 适用场景

- 测试设计、格式检查、文档同步
- 交付前验收检查

## 文档索引

| 文件 | 说明 |
|------|------|
| [`verification.md`](verification.md) | 测试命名、pre-commit、文档同步 |

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-standards` | 编码规范 |
| `/kdit-architecture` | 理解模块关系 |
