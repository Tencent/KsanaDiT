# 缓存

步级缓存策略，用于在去噪过程中跳过冗余计算。

## 概览

- [`BaseCache`](base_cache.md) — 所有缓存的抽象基类
- [`TeaCache`](teacache.md) — 时序误差感知缓存
- [`DBCache`](dbcache.md) — 基于增量的缓存
- [`DCache`](dcache.md) — D-Cache 实现
- [`EasyCache`](easycache.md) — 轻量级步缓存
- [`MagCache`](magcache.md) — 基于幅值的缓存
