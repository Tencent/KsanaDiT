# 执行器

本地与分布式执行器，负责管理模型加载与推理调度。

## 概览

- [`Executor`](executor.md) — 本地单进程执行器
- [`RayExecutor`](ray_executor.md) — 基于 Ray 的分布式多卡执行器
- [`DistributedGroup`](distributed_group.md) — 分布式组管理
