# 生成器

不同模型架构的去噪循环实现。

## 概览

- [`BaseGenerator`](base_generator.md) — 所有生成器的抽象基类
- [`WanGenerator`](wan_generator.md) — Wan 模型去噪循环
- [`QwenGenerator`](qwen_generator.md) — Qwen 模型去噪循环
- [`VaceGenerator`](vace_generator.md) — VACE 模型去噪循环
- [`GeneratorFactory`](generator_factory.md) — 通过 `ModelKey` 创建生成器的工厂
