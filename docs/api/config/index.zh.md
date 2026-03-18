# 配置

kDiT 框架各方面的配置数据类。

## 概览

| 配置 | 说明 |
|------|------|
| [`ModelConfig`](model_config.md) | 模型路径和加载选项 |
| [`SampleConfig`](sample_config.md) | 采样参数（步数、CFG、求解器） |
| [`RuntimeConfig`](runtime_config.md) | 运行时行为（卸载、编译） |
| [`AttentionConfig`](attention_config.md) | 注意力后端选择 |
| [`LinearConfig`](linear_config.md) | 线性层后端（标准 / FP8） |
| [`DistributedConfig`](distributed_config.md) | 多 GPU / 分布式设置 |
| [`CacheConfig`](cache_config.md) | 步级缓存策略 |
| [`TorchCompileConfig`](torch_compile_config.md) | `torch.compile` 选项 |
