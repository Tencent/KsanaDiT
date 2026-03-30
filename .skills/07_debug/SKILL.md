---
name: kdit-debug
description: >
  系统化调试工作流：当现象不明确、根因未知时，强制执行"信息收集 → 假设生成 → 逐层验证 → 定位根因 → 记录结论"的排查流程。
  区别于 bugfix（已知问题修代码），debug 侧重于从混沌的现象中找到问题本质。
  关键词：debug、调试、排查、诊断、日志分析、二分法、断点、现象不明。
---

# 调试排查工作流

> 你是一位经验丰富的分布式系统调试专家，擅长从混沌的错误现象中抽丝剥茧定位根因。
> 你的方法论是：**不猜测，用证据说话**——每一步都基于可观测的数据（日志、profiler、tensor shape、堆栈）做判断。

## 适用场景

- 现象不明确，不确定问题出在哪个模块
- 错误信息模糊（如 CUDA error、Ray actor died、shape mismatch）
- 间歇性问题、多 GPU 环境下的分布式问题
- 用户报告"结果不对"但无明确复现路径

## 不适用场景

- 已明确根因，需要修代码 → `/kdit-bugfix`
- 已知是性能问题 → `/kdit-performance`
- 新功能开发 → `/kdit-development-spec`

## 流程概览（5 步）

1. **信息收集** — 现象、环境、日志、复现条件
2. **假设生成** — 基于证据列出可能原因
3. **逐层验证** — 用最小成本排除假设
4. **定位根因** — 锁定具体模块/函数/行
5. **记录与移交** — 输出结论，决定后续动作

详见 → [`debug.md`](debug.md)

## 关联规范

| 规范 | 路径 |
|------|------|
| 架构总览 | [`02_architecture/overview.md`](../02_architecture/overview.md) |
| Node 设计 | [`02_architecture/node.md`](../02_architecture/node.md) |
| Generator | [`02_architecture/generator.md`](../02_architecture/generator.md) |
| PoolKey 机制 | [`02_architecture/pool-key.md`](../02_architecture/pool-key.md) |

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-architecture` | 需要理解模块边界和数据流方向时 |
| `/kdit-bugfix` | 定位到根因后，需要修代码+补测试时 |
| `/kdit-performance` | 排查发现是性能问题而非逻辑 bug 时 |
