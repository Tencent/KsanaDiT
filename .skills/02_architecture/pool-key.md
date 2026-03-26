# PoolKey — 实例化的存储 key

---

## ModelPoolKey / TensorPoolKey

- **`ModelPoolKey`** 定义在 `kdit/models/model_pool_key.py`
- **`TensorPoolKey`** 定义在 `kdit/tensor/tensor_pool_key.py`

```python
# kdit/models/model_pool_key.py
@dataclass(frozen=True)
class ModelPoolKey:
    node_id: int       # Builder 自动分配的 Node 唯一 ID
    pin: ModelKey       # 枚举直接存储

# kdit/tensor/tensor_pool_key.py
@dataclass(frozen=True)
class TensorPoolKey:
    node_id: int       # Builder 自动分配的 Node 唯一 ID
    pin: TensorKey      # 枚举直接存储
```

- `frozen=True` 自动生成 `__hash__` 和 `__eq__`，可直接用作 dict key
- `pin` 直接用枚举（不用 `.value`），类型安全且可读
- 分离 ModelPoolKey / TensorPoolKey 是因为 Model 和 Tensor 生命周期不同
- Pool 中直接传 `ModelKey`/`TensorKey` 作为 key 已标记为 deprecated，新代码应使用 `ModelPoolKey`/`TensorPoolKey`

## 为什么需要 PoolKey

解决"同一个 Pipeline 中不能有两个相同类型的 Node 实例"的问题。例如两个 `VAE_ENCODE_SPATIAL` 节点，各自写入 `TensorPoolKey(node_id=4, TensorKey.BASE_LATENT)` 和 `TensorPoolKey(node_id=5, TensorKey.BASE_LATENT)`，互不冲突。

---

## TensorPool ([`kdit/tensor/tensor_pool.py`](../../kdit/tensor/tensor_pool.py))

- **Owner**: Executor
- **生命周期**: Pipeline 推理结束时由 `engine.clear_all_tensors()` 清理；DAG 模式下中间 tensor 通过引用计数自动释放
- **内容**: `dict[TensorPoolKey, TensorValue]`，每个 TensorValue 持有 `Tensor | list[Tensor]`
- **用途**: Node 间通过 `TensorKey` 引用 tensor，避免 tensor 跨 Ray 边界序列化

### 关键方法

| 方法 | 说明 |
|------|------|
| `put(key, tensor)` | 写入 tensor，自动包装为 TensorValue |
| `get(key)` | 读取 TensorValue（消费引用计数） |
| `peek(key)` | 读取 TensorValue（不消费引用计数） |
| `has(key)` | 检查 key 是否存在 |
| `clear(exclude)` | 释放除 exclude 列表外的所有 tensor，重置引用计数 |
| `register(pool_key, ref_count)` | 注册 tensor 的下游消费者数 |
| `consume(pool_key)` | 消费一次引用计数，降为 0 时自动 release |
| `remove(pool_key)` | 强制移除 tensor |
| `rename(old, new)` | 重命名 key |

### TensorValue 类

`TensorValue` 是 tensor_pool 中值的包装类，持有单个 `torch.Tensor` 或 `list[torch.Tensor]`，负责释放：

```python
class TensorValue:
    __slots__ = ['data']

    def __init__(self, data: torch.Tensor | list[torch.Tensor]):
        self.data = data

    def release(self):
        """释放持有的 tensor 引用。"""
        if isinstance(self.data, list):
            for i in range(len(self.data)):
                self.data[i] = None
            self.data.clear()
        self.data = None
```

- Pool 的 `get(key)` 返回 `TensorValue`，Node 内部用 `.data` 取裸 tensor
- Pool 的 `clear()` 遍历调用 `TensorValue.release()`
- 只有最终边界（如 vae_decode 输出给用户）才允许 `.data` 取裸 tensor

### 引用计数机制

TensorPool 内置 `register()` / `consume()` / `remove()` 引用计数机制，替代了旧的 `tensor_scope`：

```python
# Pipeline DAG 模式 — Executor 自动管理
# 1. Pipeline 构建 DAG 时，Engine 调用 register_tensor() 注册每个 tensor 的下游消费者数
# 2. Executor.run_node() 执行后自动 consume 输入 tensor
# 3. consume 时 ref_count 降为 0 → 自动 release TensorValue

# ComfyUI 模式 — try/finally 手动清理
try:
    engine.put_tensors({TensorKey.IMAGE: image})
    engine.run_node(node_def, input_pins, context)
    result = engine.get_tensor(TensorKey.AUX_LATENT)
finally:
    engine.clear_all_tensors()
```

### clear(exclude=[...])

`TensorPool.clear(exclude)` 释放除 exclude 列表外的所有 tensor，同时重置引用计数：

```python
def clear(self, exclude: list[TensorKey | TensorPoolKey] | None = None) -> None:
    exclude_set = set(_normalize_key(k) for k in exclude) if exclude else set()
    keys_to_remove = [k for k in self._store if k not in exclude_set]
    for key in keys_to_remove:
        self._store[key].release()
        del self._store[key]
        self._ref_counts.pop(key, None)
```

---

## ModelPool ([`kdit/models/model_pool.py`](../../kdit/models/model_pool.py))

- **Owner**: Executor
- **生命周期**: 与 Executor 同生命周期，`clear_models()` 可手动清理
- **内容**: `dict[ModelKey, ModelBase]`
- **用途**: IONode 写入模型，InferNode 读取模型

---

## DistributedGroupManager ([`kdit/executor/distributed_group.py`](../../kdit/executor/distributed_group.py))

- **Owner**: Executor
- **状态**: `rank_id`, `world_size`, `_initialized`
- **用途**: 提供 `broadcast_tensors()` 能力，配合 tensor_pool 实现跨 rank 数据同步

---

## 相关文档

- Node 设计与 Pin 声明 → [`node.md`](node.md)
- PinHub 沙箱机制 → [`pin-hub.md`](pin-hub.md)
- Key 类型体系 → [`../03_standards/key-system.md`](../03_standards/key-system.md)
- 架构总览（Ownership 层级） → [`overview.md`](overview.md)
