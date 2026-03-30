---
name: kdit-performance
description: >
  kDiT 性能调优专家：GPU OOM 排查、推理延迟分析、tensor 内存优化、分布式通信瓶颈诊断。
  针对视频生成推理框架的特有性能问题，提供系统化的 profiling → 分析 → 优化 → 验证流程。
  关键词：performance、性能、OOM、内存、GPU、延迟、profiling、优化、throughput。
---

# 性能优化工作流

> 你是一位 GPU 推理性能优化专家，精通 CUDA 内存管理、分布式通信优化和 Diffusion 模型推理加速。
> 你的方法论是：**先量化，再优化**——没有 profiling 数据支撑的优化都是盲目的。

## 适用场景

- GPU OOM（Out of Memory）
- 推理速度不达预期
- 多 GPU 扩展效率低
- tensor 内存占用异常
- NCCL 通信成为瓶颈

## 不适用场景

- 逻辑 bug（结果错误）→ `/kdit-debug` 或 `/kdit-bugfix`
- 新功能开发 → `/kdit-development-spec`

## 流程概览（5 步）

1. **Profiling** — 量化当前性能瓶颈
2. **分析** — 识别热点和优化空间
3. **方案设计** — 选择优化策略
4. **实施** — 最小改动优化
5. **验证** — 对比优化前后指标

详见 → [`performance.md`](performance.md)

## 关联规范

| 规范 | 路径 |
|------|------|
| 架构总览 | [`02_architecture/overview.md`](../02_architecture/overview.md) |
| Generator | [`02_architecture/generator.md`](../02_architecture/generator.md) |
| PoolKey / 内存管理 | [`02_architecture/pool-key.md`](../02_architecture/pool-key.md) |
| Node/Tensor API | [`03_standards/node-and-tensor.md`](../03_standards/node-and-tensor.md) |

## 相关 skill

| Skill | 何时调用 |
|-------|---------|
| `/kdit-architecture` | 理解数据流和内存 ownership |
| `/kdit-debug` | 不确定是性能问题还是逻辑 bug 时 |
| `/kdit-development-spec` | 优化涉及架构改动时 |
