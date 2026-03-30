# 性能优化详细流程

> 核心原则：先量化指标，再优化。没有 profiling 数据支撑的优化都是盲目的。

---

## 第 1 步：Profiling（量化瓶颈）

在优化之前，必须先明确**瓶颈在哪里**。禁止凭直觉优化。

### kDiT 内置 Profiling 工具

| 工具 | 用途 | 使用方式 |
|------|------|---------|
| `TimeProfiler` | 层级化耗时分析 | `KSANA_PROFILE=1` 环境变量启用 |
| `MemoryProfiler` | 内存快照记录 | `MemoryProfiler.record_memory("checkpoint_name")` |
| `nvtx_range` | CUDA timeline 标记 | `with nvtx_range("label"):` 配合 Nsight Systems |

### 外部工具

| 工具 | 用途 | 命令 |
|------|------|------|
| `nvidia-smi` | GPU 实时利用率和内存 | `nvidia-smi -l 1` 或 `watch -n 1 nvidia-smi` |
| `torch.cuda.memory_summary` | PyTorch 内存分配详情 | `print(torch.cuda.memory_summary())` |
| `torch.profiler` | 算子级别耗时 | `with torch.profiler.profile() as prof:` |
| Nsight Systems | GPU timeline 可视化 | `nsys profile python ...` |

### 按场景选择 Profiling 方案

| 问题 | 首选工具 | 关注指标 |
|------|---------|---------|
| **GPU OOM** | `nvidia-smi` + `MemoryProfiler` + `torch.cuda.memory_summary` | 峰值内存、分配模式 |
| **推理慢** | `TimeProfiler` + `torch.profiler` | 各阶段耗时占比 |
| **多 GPU 低效** | `TimeProfiler` + NCCL 日志 | 通信 vs 计算时间比 |
| **tensor 内存异常** | `MemoryProfiler` + tensor shape/dtype 检查 | 不必要的大 tensor、dtype 不一致 |

**输出**：
- Profiling 数据（耗时分布 / 内存曲线 / GPU 利用率）
- 瓶颈定位（具体到模块/函数）

---

## 第 2 步：分析（识别优化空间）

### 常见瓶颈模式

#### GPU OOM

| 原因 | 特征 | 检查方法 |
|------|------|---------|
| 中间 tensor 未释放 | 内存只增不减 | `MemoryProfiler` 对比各 checkpoint |
| tensor 副本过多 | `.clone()` / `.contiguous()` 滥用 | grep 代码 |
| batch_size 过大 | 单步就 OOM | 减小 batch 测试 |
| 模型权重 + KV cache + 激活值叠加 | 加载模型后剩余内存不足 | `MODEL_MEMORY_CONFIG` 对比实际 |
| ComfyUI tensor pool 未 clear | pool 中累积历史 tensor | 检查 `clear_all_tensors()` 调用点 |

#### 推理延迟

| 原因 | 特征 | 检查方法 |
|------|------|---------|
| attention 实现低效 | attention 阶段耗时占比 > 60% | `TimeProfiler` |
| CPU-GPU 同步点 | `.item()` / `.cpu()` / `print(tensor)` | grep 代码 |
| 不必要的 dtype 转换 | `to(dtype)` 频繁调用 | profiler + grep |
| 分布式通信阻塞 | all_gather / all_to_all 耗时高 | NCCL 日志 |
| torch.compile 未生效 | 每次都 recompile | 检查 `torch._dynamo` 日志 |

**输出**：
- 瓶颈分析结论（具体原因 + 影响量化）
- 优化空间估算（理论可优化多少）

---

## 第 3 步：方案设计

### 优化策略优先级

| 优先级 | 策略 | 风险 | 收益 |
|--------|------|------|------|
| **P0** | 消除 bug 级浪费（未释放 tensor、多余副本） | 低 | 高 |
| **P1** | 算法替换（更快的 attention backend、cache 策略） | 中 | 高 |
| **P2** | 内存优化（dtype 降精度、activation checkpointing） | 中 | 中 |
| **P3** | 系统级优化（torch.compile、通信优化） | 高 | 因场景而异 |

### 方案模板

```
优化方案：[简述]
- 目标：[具体指标，如"峰值内存从 24GB 降到 18GB"]
- 方法：[具体改动]
- 风险：[可能的副作用]
- 验证标准：[如何确认优化有效]
```

**输出**：
- 1-3 个优化方案，按优先级排序

---

## 第 4 步：实施

### 原则

1. **一次只优化一个点** — 多个优化同时做无法归因效果
2. **保持功能不变** — 优化不能改变推理结果（除非精度降级且用户同意）
3. **最小改动** — 不借优化之名重构

### kDiT 常见优化手段

| 手段 | 适用场景 | 注意事项 |
|------|---------|---------|
| 切换 attention backend | sage_attn / flash_attn | 检查精度影响 |
| 启用 cache（TeaCache / DCache 等） | 重复帧加速 | 检查质量损失 |
| FP8 量化 | 显存不足 | 需要 H100 / 4090 |
| torch.compile | 计算密集 | 首次编译耗时长，NPU 暂不支持 |
| 及时 `del tensor` + `torch.cuda.empty_cache()` | 内存碎片 | 不要过度调用 |

---

## 第 5 步：验证

### 对比指标

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| 峰值 GPU 内存 | ? GB | ? GB | -?% |
| 单步推理耗时 | ? ms | ? ms | -?% |
| 端到端生成耗时 | ? s | ? s | -?% |
| 输出质量 | baseline | 对比 | 无损 / 可接受损失 |

### 验证检查清单

- [ ] 优化前后使用相同输入、相同 seed
- [ ] 性能指标有统计意义（跑 3 次以上取平均）
- [ ] 输出结果质量无退化（或退化在可接受范围）
- [ ] 全量测试通过：`pytest -s -v tests/kdit`
- [ ] 不同 batch_size 下都有效

### 总结输出（5 项）

| # | 项目 | 内容 |
|---|------|------|
| 1 | **瓶颈** | profiling 发现的具体瓶颈 |
| 2 | **优化方案** | 采用了什么策略 |
| 3 | **性能对比** | 优化前后的关键指标对比 |
| 4 | **质量影响** | 输出质量是否有变化 |
| 5 | **适用范围** | 在什么条件下有效（GPU 型号、模型大小等） |
