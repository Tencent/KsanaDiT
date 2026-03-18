# 算子

底层计算算子，支持多种后端实现。

## 概览

- [注意力](attention.md) — 多后端注意力算子（FlashAttention、SageAttention、SDPA）
- [线性层](linear.md) — 标准与 FP8 线性层后端
- [QKV 融合](fuse_qkv.md) — 融合 QKV 投影（FP8 模式下自动禁用）
