# PinHub — 沙箱化数据访问器

**职责**：Node 运行时的数据读写路由层。每个 Node 实例拥有独立的 PinHub，被严格约束在 DAG 声明的范围内。

---

## 核心机制

- **读操作**：根据 `input_pins`（DAG 连线计算的映射），将上游 Node 的输出 PoolKey 映射到当前 Node 的 input pin
- **写操作**：自动用 `当前 node_id + pin` 生成 PoolKey 写入 Pool

```python
pins.get_tensor(TensorKey.POSITIVE)   # 从 input_pins 查找上游的 TensorPoolKey，读取
pins.put_tensor(TensorKey.LATENTS, data)  # 写入 TensorPoolKey(self.node_id, TensorKey.LATENTS)
pins.get_model(ModelKey.T5TextEncoder)    # 从 input_pins 查找上游的 ModelPoolKey，读取
pins.put_model(ModelKey.VAE_WAN2_1, model)  # 写入 ModelPoolKey(self.node_id, ModelKey.VAE_WAN2_1)
```

---

## API 详细说明

### Tensor 操作

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_tensor(key)` | `(TensorKey) → TensorValue` | 从 input_pins 查找上游 TensorPoolKey，读取 TensorValue。未连线返回 `None` |
| `peek_tensor(key)` | `(TensorKey) → TensorValue \| None` | 同 `get_tensor`，但不消费引用计数（只读不减） |
| `put_tensor(key, tensor)` | `(TensorKey, Tensor) → None` | 写入 `TensorPoolKey(self.node_id, key)` 到 tensor_pool |

### Model 操作

| 方法 | 签名 | 说明 |
|------|------|------|
| `get_model()` | `() → ModelBase` | **无参**，自动从 `node_def.model_key` 获取对应的 ModelPoolKey，读取模型 |
| `get_model(key)` | `(ModelKey) → ModelBase` | 指定 ModelKey 读取（多模型 Node 场景） |
| `put_model(key, model)` | `(ModelKey, ModelBase) → None` | 写入 `ModelPoolKey(self.node_id, key)` 到 model_pool |

---

## 沙箱约束

- **读**：只能读 `input_pins` 中存在的 key（即 DAG 连线声明的上游输出），读不到其他 Node 的数据
- **写**：只能写 `PoolKey(self.node_id, pin)`，即自己 node_id 命名空间下的 key，写不到别人的命名空间
- 未连线的 optional tensor pin 返回 `None`，未连线的 required model pin 抛出 `KeyError`

---

## 构建位置

PinHub 在 **Executor 内部构建**（不在 Pipeline 层），因为：
- `tensor_pool` 和 `model_pool` 活在 Executor 上（每卡一份）
- 多卡下每个 Executor 各自构建自己的 PinHub，天然正确
- PinHub 不需要跨进程序列化

---

## 注意事项

- `input_pins` 是纯数据（`dict[TensorKey, TensorPoolKey]` + `dict[ModelKey, ModelPoolKey]`），由 Pipeline 层从 DAG edges 计算，通过 Engine 分发到 Executor
- PinHub 不持有 `DeviceInfo`，设备信息通过 `context.device` 获取

---

## 相关文档

- Node 设计原则与 Pin 声明 → [`node.md`](node.md)
- PoolKey 体系 → [`pool-key.md`](pool-key.md)
- NodeContext → [`node-context.md`](node-context.md)
