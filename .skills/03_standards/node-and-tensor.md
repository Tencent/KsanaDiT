# InferNode 编码实操规范

> 本文档聚焦**编码时的实操规则和代码示例**。架构设计、Ownership、数据流等内容已迁移至 [`02_architecture/`](../02_architecture/) 目录：
> - Node 架构设计（Def/Pin 术语、dispatch_policy、Node 状态等） → [`02_architecture/node.md`](../02_architecture/node.md)
> - PinHub 沙箱机制 → [`02_architecture/pin-hub.md`](../02_architecture/pin-hub.md)
> - Ownership 与 Engine/Executor API → [`02_architecture/overview.md`](../02_architecture/overview.md)
> - TensorPool / ModelPool / 引用计数 → [`02_architecture/pool-key.md`](../02_architecture/pool-key.md)

---

## run() 签名固定，禁止扩展

`InferNode.run()` 签名是固定的：

```python
def run(self, pins: PinHub, *, context: NodeContext) -> None:
```

- `pins` 是 `PinHub` 实例，提供 `get_model()` / `get_tensor()` / `put_tensor()` 等方法
- `context` 是 `NodeContext` 实例，包含 metadata 和 device_info
- **禁止** 添加 `**kwargs` 或任何额外参数
- **禁止** 返回值（必须返回 `None`）
- 如需传递额外配置，使用 `context.metadata` 字典

---

## Tensor 只能通过 PinHub 流转

- **输入 tensor**: 只能通过 `pins.get_tensor(key)` 或 `pins.peek_tensor(key)` 获取
- **输出 tensor**: 只能通过 `pins.put_tensor(key, tensor)` 写入
- **输入 model**: 只能通过 `pins.get_model(key)` 获取
- **禁止** 在 `run()` 参数中传递 tensor
- **禁止** 在 `context.metadata` 中放 tensor（`NodeContext.__post_init__` 会校验）

---

## 声明 tensor 契约

每个 Node 必须声明 `input_defs` 和 `output_defs`（`list[PinDef]`）：

```python
class MyNode(InferNode):
    input_defs = [TensorKey.POSITIVE, TensorKey.NEGATIVE]
    output_defs = [TensorKey.LATENTS]
```

- `input_defs` 可以为空列表 `[]`（如 TextEncodeNode 不依赖其他 tensor）
- `output_defs` 用于 Executor 构建 `output_pins` 和 `R0_R0_BCAST` 时指定需要 broadcast 的 key
- model 端口由 `NodeDef.model_key` 隐含，**不在** `input_defs`/`output_defs` 中声明

---

## TensorKey 枚举使用规范

tensor_pool 的 key **必须**使用 `TensorKey(str, Enum)` 枚举，全局统一，不使用裸字符串：

```python
from kdit.tensor import TensorKey

# ✅ 使用枚举
pins.put_tensor(TensorKey.LATENTS, latents)
video = engine.get_tensor(TensorKey.VIDEO)
engine.put_tensors(**{TensorKey.POSITIVE: positive, TensorKey.NEGATIVE: negative})

# ❌ 裸字符串
tensor_pool.put("latents", latents)
```

所有 tensor key 枚举值：

| 枚举 | 值 | 生产者 | 消费者 |
|------|-----|--------|--------|
| `TensorKey.POSITIVE` | `"positive"` | TextEncodeNode | GeneratorNode |
| `TensorKey.NEGATIVE` | `"negative"` | TextEncodeNode | GeneratorNode |
| `TensorKey.LATENTS` | `"latents"` | GeneratorNode | VAEDecodeNode |
| `TensorKey.VIDEO` | `"video"` | VAEDecodeNode | SaveVideoNode |
| `TensorKey.IMAGE` | `"image"` | ReadImageNode | VAEEncodeSpatialNode / VAEEncodeImagesNode（通过 DAG edge connect） |
| `TensorKey.START_IMG` | `"start_img"` | ReadImageNode（DAG edge） | VAEEncodeSpatialNode |
| `TensorKey.END_IMG` | `"end_img"` | ReadImageNode（DAG edge） | VAEEncodeSpatialNode |
| `TensorKey.BASE_LATENT` | `"base_latent"` | VAEEncodeSpatialNode / VAEComputeShapeNode | GeneratorNode |
| `TensorKey.AUX_LATENT` | `"aux_latent"` | VAEEncodeImagesNode / VACEPreprocessNode | GeneratorNode |
| `TensorKey.VACE_CONTEXT` | `"vace_context"` | VACEPreprocessNode | GeneratorNode |

---

## NodeContext 禁止 tensor

`NodeContext.metadata` 中**禁止**包含 `torch.Tensor`。`__post_init__` 递归检查 metadata dict 的 values：

```python
def __post_init__(self):
    for field_name, value in self.__dict__.items():
        if isinstance(value, torch.Tensor):
            raise TypeError(...)
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"NodeContext.{field_name}[{k!r}] is a Tensor! "
                        f"Use engine.put_tensors() + TensorKey instead."
                    )
```

**原因**：
- 单卡模式下 metadata 中的 tensor 能跑，但违反设计约束
- 多卡 Ray 模式下 context 会被 pickle 序列化，metadata 中的大 tensor 导致性能严重下降
- 所有 tensor 输入必须走 `engine.put_tensors()` → `tensor_pool`，Node 从 `tensor_pool` 读取

---

## Node 写入规则

Node 必须**无条件写入** tensor，不得判断 `rank_id`：

```python
# ✅ 正确
def run(self, pins: PinHub, *, context: NodeContext) -> None:
    result = unit.run(...)
    pins.put_tensor(TensorKey.LATENTS, result)

# ❌ 错误 — Node 不应关心下游的 dispatch_policy
def run(self, pins: PinHub, *, context: NodeContext) -> None:
    result = unit.run(...)
    if context.device.rank_id == 0:
        pins.put_tensor(TensorKey.LATENTS, result)
```

---

## 完整 InferNode 示例

```python
from kdit.tensor import TensorKey
from kdit.nodes.core.base_node import InferNode
from kdit.nodes.core.node_types import NodeDispatchPolicy

class GeneratorNode(InferNode):
    input_defs = [TensorKey.POSITIVE, TensorKey.NEGATIVE, TensorKey.BASE_LATENT,
                  TensorKey.AUX_LATENT, TensorKey.VACE_CONTEXT]
    output_defs = [TensorKey.LATENTS]
    dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

    def run(self, pins: PinHub, *, context: NodeContext) -> None:
        # 1. 读取模型（无参，自动从 node_def.model_key 获取）
        model = pins.get_model()

        # 2. 读取输入 tensor（可能为 None，需自行处理）
        positive = pins.get_tensor(TensorKey.POSITIVE)
        negative = pins.get_tensor(TensorKey.NEGATIVE)
        base_latent = pins.get_tensor(TensorKey.BASE_LATENT)
        aux_latent = pins.get_tensor(TensorKey.AUX_LATENT)      # 可能为 None
        vace_ctx = pins.get_tensor(TensorKey.VACE_CONTEXT)       # 可能为 None

        # 3. 执行推理
        latents = generator.run(model, positive, negative, base_latent, ...)

        # 4. 写入输出 tensor（无条件写入，不判断 rank_id）
        pins.put_tensor(TensorKey.LATENTS, latents)
```

---

## 相关文档

- Node 架构设计（Def/Pin、dispatch_policy、状态分析） → [`02_architecture/node.md`](../02_architecture/node.md)
- PinHub 沙箱机制与 API → [`02_architecture/pin-hub.md`](../02_architecture/pin-hub.md)
- TensorPool / 引用计数 → [`02_architecture/pool-key.md`](../02_architecture/pool-key.md)
- Key 类型体系 → [`key-system.md`](key-system.md)
- 异常处理规范 → [`exception-handling.md`](exception-handling.md)
- Import 与类型注解 → [`imports-and-types.md`](imports-and-types.md)
