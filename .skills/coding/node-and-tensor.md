# Node / Tensor API、Ownership 与 InferNode 开发规范

> 本文件从 [`.skills/coding.md`](../coding.md) 拆分，包含 §4、§5、§6。

---

## Def/Pin 术语规范

### 核心概念

- **Def（声明）**：编译时/类定义时的端口声明。Node 类上的 `input_defs`/`output_defs`。
- **Pin（绑定）**：运行时的端口地址。Executor 构建的 `input_pins`/`output_pins`，值为 PoolKey。

### 命名规则

- 变量名含 `def` → 声明层（TensorKey / ModelKey）
- 变量名含 `pin` → 运行时层（TensorPoolKey / ModelPoolKey）
- Node 类属性用 `input_defs` / `output_defs`
- Executor/Engine 参数用 `input_pins` / `output_pins`
- PinHub 是 Pin 层的访问器

### 类型

| 名称 | 类型 | 层级 |
|------|------|------|
| `PinDef` | `TensorKey \| ModelKey` | Def |
| `PinPoolKey` | `TensorPoolKey \| ModelPoolKey` | Pin |
| `Pins` | `dict[PinDef, PinPoolKey]` | Pin |

---

## 4. V5 Node / Tensor API 规范

### Node.run() 返回值

| Node 类型 | 返回值 | 说明 |
|-----------|--------|------|
| `IONode.run()` | `None` | 模型写入 model_pool，不返回 |
| `InferNode.run()` | `None` | 结果写入 tensor_pool，不返回 tensor |

**禁止** `InferNode.run()` 返回 dict 或 tensor。所有中间结果通过 `tensor_pool.put(key, tensor)` 写入。

### output_defs 静态声明

`output_defs` 保留为**类属性**（`list[PinDef]`），Executor 用它做 `R0_R0_BCAST` broadcast 和 `_build_output_pins()`。
对于条件性输出（如 VAEEncodeNode 可能不写入 AUX_LATENT），broadcast 逻辑对 `tensor_pool.peek(key) is None` 的 key 做 skip 容错。

```python
class GeneratorNode(InferNode):
    input_defs = [TensorKey.POSITIVE, TensorKey.NEGATIVE, TensorKey.BASE_LATENT,
                  TensorKey.AUX_LATENT, TensorKey.VACE_CONTEXT]
    output_defs = [TensorKey.LATENTS]

    def run(self, pins: PinHub, *, context: NodeContext) -> None:
        model = pins.get_model()  # 无参 — 自动从 node_def.model_key 获取
        latents = generator.run(...)
        pins.put_tensor(TensorKey.LATENTS, latents)
```

### Node.run() 写入规则

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

### TensorKey 枚举

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

### NodeContext 禁止 tensor

`NodeContext.metadata` 中**禁止**包含 `torch.Tensor`。`__post_init__` 必须递归检查 metadata dict 的 values：

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

### Engine 公开 API

| 方法 | 用途 | 说明 |
|------|------|------|
| `engine.get_tensor(key)` | 取回 TensorValue | 自动从 rank 0 取，返回 `TensorValue`（需 `.data` 取裸 tensor） |
| `engine.put_tensors(tensors)` | 写入 tensor | 写入所有 Executor 的 tensor_pool，自动包装为 `TensorValue` |
| `engine.has_tensor(key)` | 检查 key 存在性 | 检查 rank 0 的 tensor_pool 中是否存在指定 key |
| `engine.register_tensor(pool_key, ref_count)` | 注册引用计数 | 透传到所有 Executor 的 tensor_pool.register() |
| `engine.clear_all_tensors()` | 清理所有 tensor | 清理所有 Executor 的 tensor_pool — 用于 try/finally 异常恢复 |
| `engine.rename_tensor(old, new)` | 重命名 key | 透传到所有 Executor 的 tensor_pool.rename() |

> **`tensor_scope` 已删除**。异常安全通过 `try/finally + engine.clear_all_tensors()` 实现。
> TensorPool 内置引用计数（register/consume）自动管理中间 tensor 的释放。

### Tensor 生命周期与 TensorPool 引用计数

#### TensorValue 类

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

#### TensorPool 内置引用计数

TensorPool 内置 `register()` / `consume()` / `remove()` 引用计数机制，替代了旧的 `tensor_scope`：

```python
# Pipeline DAG 模式 — Executor 自动管理
# 1. Pipeline 构建 DAG 时，Engine 调用 register_tensor() 注册每个 tensor 的下游消费者数
# 2. Executor.run_node() 执行后自动 consume 输入 tensor
# 3. consume 时 ref_count 降为 0 → 自动 release TensorValue

# ComfyUI 模式 — try/finally 手动清理
try:
    engine.put_tensors({TensorKey.IMAGE: image})
    engine.run_node(node_def, pins_mapping, context)
    result = engine.get_tensor(TensorKey.AUX_LATENT)
finally:
    engine.clear_all_tensors()
```

#### ComfyUI adapter 之间传递 key

ComfyUI adapter 之间传递的是 `TensorKey`（而非裸 tensor），真正的 `TensorValue` 存在 Executor 的 pool 中：

```python
# vae_encode 返回 key
def vae_encode_image(...):
    try:
        engine.put_tensors(...)
        engine.run_node(...)
        return KsanaNodeVAEEncodeOutput(samples=TensorKey.BASE_LATENT, ...)
    except Exception:
        engine.clear_all_tensors()
        raise

# generate 收到 key，先检查存在性
def generate(...):
    if not engine.has_tensor(base_latent_key):
        raise RuntimeError(f"Tensor {base_latent_key} not found in pool")
    try:
        engine.run_node(...)
        return KsanaNodeGeneratorOutput(samples=TensorKey.LATENTS, ...)
    except Exception:
        engine.clear_all_tensors()
        raise
```

#### Pool 的 clear(exclude=[...])

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

### Node 间数据传递

- Node 间通过 **同一个 Executor 内的 tensor_pool** 传递数据，不经过 Engine
- 跨 rank 数据传递由 `DispatchPolicy.RANK_0_BROADCAST` 的 broadcast 机制自动处理
- `engine.get_tensor()` 只用于 Pipeline/ComfyUI 取回最终结果，不用于 Node 间传递
- Pipeline/ComfyUI 向 Node 传递 tensor 输入时，必须通过 `engine.put_tensors()` 写入 tensor_pool
- **禁止**在 ComfyUI adapter 之间传递裸 `torch.Tensor`，必须传递 `TensorKey`

---

## 5. Ownership 与状态关系图

### 整体 Ownership 层级

```
Engine (singleton via get_default / 或多实例)
 ├── owns: executors
 │    ├── 单卡模式: 1 个 Executor 实例
 │    └── 多卡模式: N 个 RayExecutor (Ray Actor)
 ├── owns: num_gpus, _is_ray, _cleaned_up (引擎级元数据)
 ├── NOT own: model_pool, tensor_pool, device 信息 (这些属于 Executor)
 └── NOT own: 任何 Node 实例 (Node 由 AdvancedFactory 按需创建，用完即弃)

Executor (每卡一个实例)
 ├── owns: model_pool        — ModelPool (存储已加载的模型)
 ├── owns: tensor_pool       — TensorPool (存储推理中间 tensor)
 ├── owns: dist_group        — DistributedGroupManager (管理 torch.distributed)
 ├── owns: device_ctx        — NodeDeviceContext (frozen dataclass, 只读)
 ├── owns: device / offload_device / device_id (设备信息)
 ├── owns: rank_id / world_size (分布式信息)
 ├── owns: dist_config / shard_fn (分布式配置)
 ├── owns: local_pipeline (遗留 V4 接口，将废弃)
 └── NOT own: Node 实例 (Node 在 run_node 中临时创建)
```

### 各组件详细 Ownership

#### Engine (`kdit/engine/engine.py`)

| 属性 | 类型 | 说明 |
|------|------|------|
| `executors` | `Executor` 或 `list[RayExecutor]` | **唯一核心持有物**。单卡时是一个实例，多卡时是 Ray Actor 列表 |
| `num_gpus` | `int` | GPU 数量，从 dist_config 复制 |
| `_is_ray` | `bool` | 是否使用 Ray 分布式 |
| `_cleaned_up` | `bool` | 清理标记，防止重复清理 |

**Engine 不持有**：model_pool、tensor_pool、device 信息、Node 实例。Engine 是纯粹的**分发层**，所有实际资源都在 Executor 上。

**Engine 的桥接方法**（透传到 Executor）：
- `run_node()` → 分发到所有 Executor，**返回 `output_pins`**（`{TensorKey | ModelKey: TensorPoolKey | ModelPoolKey}`）。Ray 模式取 rank 0 结果（output_pins 是纯元数据，所有 rank 相同）
- `put_tensors()` → 写入所有 Executor 的 tensor_pool
- `get_tensor()` → 从 rank 0 Executor 的 tensor_pool 读取
- `register_tensor()` → 注册引用计数到所有 Executor 的 tensor_pool
- `clear_all_tensors()` → 清理所有 Executor 的 tensor_pool（异常恢复）

#### Executor (`kdit/executor/executor.py`)

| 属性 | 类型 | 生命周期 | 说明 |
|------|------|---------|------|
| `model_pool` | `ModelPool` | 与 Executor 同生命周期 | 存储所有已加载模型，按 `ModelKey` 索引 |
| `tensor_pool` | `TensorPool` | 每次推理结束时 clear（Pipeline 用 try/finally） | 存储推理中间 tensor，内置引用计数 |
| `dist_group` | `DistributedGroupManager` | 与 Executor 同生命周期 | 管理 broadcast 等分布式操作 |
| `device_ctx` | `NodeDeviceContext` | 初始化后不变（frozen） | 只读设备上下文，传入 Node.run() |
| `device` | `torch.device` | 不变 | 计算设备 (如 `cuda:0`) |
| `offload_device` | `torch.device` | 不变 | 卸载设备 (如 `cpu`) |
| `dist_config` | `DistributedConfig` | `init_torch_dist_group()` 后更新 | 分布式配置 |
| `shard_fn` | `partial` 或 `None` | `init_torch_dist_group()` 后设置 | FSDP 分片函数 |

#### NodeDeviceContext (`kdit/nodes/core/device_context.py`)

```python
@dataclass(frozen=True)  # ← frozen! Node 无法篡改
class NodeDeviceContext:
    device: torch.device        # 计算设备
    offload_device: torch.device # 卸载设备
    rank_id: int                # 当前 rank
    world_size: int             # 总 rank 数
```

- **由 Executor 创建**，在 `__init__` 和 `init_torch_dist_group()` 中构建
- **传入 Node.run()** 作为只读参数
- **frozen=True** 保证 Node 无法修改设备配置

#### TensorPool (`kdit/tensor/tensor_pool.py`)

- **Owner**: Executor
- **生命周期**: Pipeline 推理结束时由 `engine.clear_all_tensors()` 清理；DAG 模式下中间 tensor 通过引用计数自动释放
- **内容**: `dict[TensorPoolKey, TensorValue]`，每个 TensorValue 持有 `Tensor | list[Tensor]`
- **引用计数**: `register(pool_key, ref_count)` / `consume(pool_key)` / `remove(pool_key)` — Executor 自动管理
- **用途**: Node 间通过 `TensorKey` 引用 tensor，避免 tensor 跨 Ray 边界序列化
- **关键方法**: `put` / `get` / `peek` / `has` / `clear(exclude)` / `register` / `consume` / `remove` / `rename`

#### ModelPool (`kdit/models/model_pool.py`)

- **Owner**: Executor
- **生命周期**: 与 Executor 同生命周期，`clear_models()` 可手动清理
- **内容**: `dict[ModelKey, KsanaModel]`
- **用途**: IONode 写入模型，InferNode 读取模型

#### DistributedGroupManager (`kdit/executor/distributed_group.py`)

- **Owner**: Executor
- **状态**: `rank_id`, `world_size`, `_initialized`
- **用途**: 提供 `broadcast_tensors()` 能力，配合 tensor_pool 实现跨 rank 数据同步

### Node 状态分析

#### IONode (`IONode` 子类)

| Node | 类变量状态 | 实例变量状态 | 说明 |
|------|-----------|-------------|------|
| `DiffusionLoaderNode` | `_pinned_memory_manager: PinnedMemoryManager` (类变量) | 无 | **有类级状态**：`_pinned_memory_manager` 是跨调用共享的单例，首次 `run()` 时惰性初始化 |
| `TextEncoderLoaderNode` | 无 | 无 | **无状态** |
| `VAELoaderNode` | 无 | 无 | **无状态** |

**关键问题**：`DiffusionLoaderNode._pinned_memory_manager` 是类变量，在 `NodeFactory.create()` 每次创建新实例时不会重置。这意味着：
- 同一进程内所有 `DiffusionLoaderNode` 实例共享同一个 `PinnedMemoryManager`
- 这是有意设计（共享 pinned memory 池），但违反了"Node 无状态"的理想模型

#### InferNode (`InferNode` 子类)

| Node | 类变量状态 | 实例变量状态 | 说明 |
|------|-----------|-------------|------|
| `TextEncodeNode` | 无 | 无 | **完全无状态** |
| `GeneratorNode` | 无 | 无 | **完全无状态** |
| `VAEDecodeNode` | 无 | 无 | **完全无状态** |
| `VAEEncodeSpatialNode` | 无 | 无 | **完全无状态** |
| `VAEEncodeImagesNode` | 无 | 无 | **完全无状态** |

所有 InferNode **完全无状态**：
- 每次 `run()` 从 PinHub 读输入 tensor 和 model
- 计算结果通过 `pins.put_tensor()` 写入
- 不持有任何跨调用的状态

### Node 创建方式

Node 由 Factory **按 node_def 创建**，Executor 内部按 `node_id` 缓存：

```python
# executor.py
def _get_or_create_node(self, node_def):
    if node_def.node_id in self._node_cache:
        return self._node_cache[node_def.node_id]
    if node_def.is_io:
        node = LoaderNodeFactory.create(node_def.model_key)
    else:
        node = InferNodeFactory.create(node_def.node_type, node_def.model_key)
    self._node_cache[node_def.node_id] = node
    return node

def run_node(self, node_def, pins_mapping, context):
    node = self._get_or_create_node(node_def)
    pin_hub = self._build_pin_hub(node_def, pins_mapping)
    self._pre_sync_tensors(node, policy)
    if is_active_rank:
        node.run(pin_hub, context=context)
    self._post_sync_tensors(node, node_def, policy)
    self._consume_input_tensors(pins_mapping)  # 自动消费输入 tensor
```

Node 实例按 `node_id` 缓存在 Executor 中，同一 node_id 复用同一实例。

### 设计约束总结

| 约束 | 说明 |
|------|------|
| Engine 不持有资源 | Engine 只是分发层，所有实际资源（model_pool, tensor_pool, device）在 Executor 上 |
| Executor 持有所有资源 | model_pool + tensor_pool + dist_group + device_ctx + node_cache |
| Node 无状态（理想） | InferNode 完全无状态；IONode 中 `DiffusionLoaderNode` 有类级 `_pinned_memory_manager` 例外 |
| NodeDeviceContext 只读 | `frozen=True` dataclass，Node 无法篡改 |
| NodeContext 无 tensor | `__post_init__` 强制校验不含 `torch.Tensor`，保证可跨 Ray 序列化 |
| tensor_pool 生命周期 | Pipeline 用 `try/finally + clear_all_tensors()`；DAG 模式下引用计数自动释放 |
| model_pool 生命周期 | 与 Executor 同生命周期，需手动 `clear_models()` 释放 |
| InferNode.run() 签名固定 | `(self, pins: PinHub, *, context: NodeContext) -> None`，禁止扩展 |
| Tensor 只能通过 PinHub 流转 | 禁止在 run() 参数或 context.metadata 中传递 tensor |

---

## 6. InferNode 开发规范

### run() 签名固定，禁止扩展

`InferNode.run()` 签名是固定的：

```python
def run(self, pins: PinHub, *, context: NodeContext) -> None:
```

- `pins` 是 `PinHub` 实例，提供 `get_model()` / `get_tensor()` / `put_tensor()` 等方法
- `context` 是 `NodeContext` 实例，包含 metadata 和 device_info
- **禁止** 添加 `**kwargs` 或任何额外参数
- **禁止** 返回值（必须返回 `None`）
- 如需传递额外配置，使用 `context.metadata` 字典

### Tensor 只能通过 PinHub 流转

- **输入 tensor**: 只能通过 `pins.get_tensor(key)` 或 `pins.peek_tensor(key)` 获取
- **输出 tensor**: 只能通过 `pins.put_tensor(key, tensor)` 写入
- **输入 model**: 只能通过 `pins.get_model(key)` 获取
- **禁止** 在 `run()` 参数中传递 tensor
- **禁止** 在 `context.metadata` 中放 tensor（`NodeContext.__post_init__` 会校验）

### 声明 tensor 契约

每个 Node 必须声明 `input_defs` 和 `output_defs`（`list[PinDef]`）：

```python
class MyNode(InferNode):
    input_defs = [TensorKey.POSITIVE, TensorKey.NEGATIVE]
    output_defs = [TensorKey.LATENTS]
```

- `input_defs` 可以为空列表 `[]`（如 TextEncodeNode 不依赖其他 tensor）
- `output_defs` 用于 Executor 构建 `output_pins` 和 `R0_R0_BCAST` 时指定需要 broadcast 的 key
- model 端口由 `NodeDef.model_key` 隐含，**不在** `input_defs`/`output_defs` 中声明

### dispatch_policy 三维度命名

`NodeDispatchPolicy` 使用 `input_exec_output` 三维度拼接命名：

| Policy | 输入要求 | 执行范围 | 输出行为 | 典型场景 |
|--------|---------|---------|---------|---------|
| `ALL_ALL_ALL` | 所有卡都有 | 所有卡 | 各卡独立持有 | TextEncode, Generator |
| `R0_R0_BCAST` | rank0 有即可 | 仅 rank0 | broadcast 到所有卡 | VAEEncode |
| `ALL_R0_R0` | 所有卡都有 | 仅 rank0 | 仅 rank0 持有 | VAEDecode |

### Node 注册

- 使用 `@InferNodeFactory.register()` 装饰器注册
- 注册键为 `(InferNodeType, [ModelKey, ...])`
- `InferNodeType` 枚举值：`TEXT_ENCODE`, `VAE_ENCODE_SPATIAL`, `VAE_ENCODE_IMAGES`, `VAE_COMPUTE_SHAPE`, `VAE_DECODE`, `GENERATE`, `SAVE_VIDEO`, `SAVE_IMAGE`, `READ_IMAGE`, `VACE_PREPROCESS`

### 现有 Node 参考

| Node | dispatch_policy | input_defs | output_defs |
|------|----------------|------------|-------------|
| `T5TextEncodeNode` | `ALL_ALL_ALL` | `[]` | `[POSITIVE, NEGATIVE]` |
| `QwenTextEncodeNode` | `ALL_ALL_ALL` | `[]` | `[POSITIVE, NEGATIVE]` |
| `VAEEncodeSpatialNode` | `R0_R0_BCAST` | `[START_IMG, END_IMG]` | `[BASE_LATENT]` |
| `VAEEncodeImagesNode` | `R0_R0_BCAST` | `[IMAGE]` | `[AUX_LATENT]` |
| `VAEComputeShapeNode` | `R0_R0_BCAST` | `[]` | `[BASE_LATENT]` |
| `VAEDecodeNode` | `ALL_R0_R0` | `[LATENTS]` | `[VIDEO]` |
| `GeneratorNode` | `ALL_ALL_ALL` | `[POSITIVE, NEGATIVE, BASE_LATENT, AUX_LATENT, VACE_CONTEXT]` | `[LATENTS]` |
| `SaveVideoNode` | `ALL_R0_R0` | `[VIDEO]` | `[]` |
| `SaveImageNode` | `ALL_R0_R0` | `[VIDEO]` | `[]` |
| `ReadImageNode` | `R0_R0_BCAST` | `[]` | `[IMAGE]` |

### Executor 同步机制

`Executor.run_node()` 负责：

1. **`_pre_sync_tensors()`**: 执行前的 tensor 同步（预留接口，未来可自动 broadcast 输入）
2. **`is_active_rank`**: 根据 policy 判断当前卡是否执行 `run()`
3. **`_post_sync_tensors()`**: 执行后的 tensor 同步（`R0_R0_BCAST` 时 broadcast `output_defs` 中的 key）
4. **`_consume_input_tensors()`**: 自动消费输入 tensor 引用计数

Node 内部不需要感知多卡逻辑，Executor 负责所有 tensor 的 pre/post 同步和引用计数管理。
