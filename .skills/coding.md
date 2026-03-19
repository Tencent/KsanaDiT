# kDiT Coding Skills

## 1. Import 风格规范（方案 B）

### 规则

| 导入层级 | 写法 | 示例 |
|---------|------|------|
| 同目录（`.`） | **相对导入** | `from .base import Foo` |
| 同子包内（`..`） | **相对导入** | `from ..core.base_node import LoaderNode` |
| 跨子包（`...` 及以上） | **绝对导入** | `from kdit.utils.factory import AdvancedFactory` |

### 判定标准

- **"同子包"** 定义：共享 `kdit/` 下同一个一级子目录。例如 `kdit/nodes/loaders/` 和 `kdit/nodes/core/` 同属 `nodes` 子包。
- **三级及以上相对导入（`from ...xxx`）一律禁止**，必须改为绝对导入 `from kdit.xxx`。
- `kdit/operations/` 内部的深层嵌套（如 `backends/radial_sage_attn/`）跨子目录时也使用绝对导入。

### 示例

```python
# kdit/nodes/loaders/diffusion_model_loader.py

# ✅ 跨子包 → 绝对导入
from kdit.config import LoraConfig, ModelConfig
from kdit.memory import PinnedMemoryManager
from kdit.models import KsanaWanModel
from kdit.utils import is_file_or_dir, log

# ✅ 同子包 (nodes) → 相对导入
from ..core.base_node import LoaderNode
from ..core.node_factory import LoaderNodeFactory
from ..core.node_types import NodeDispatchPolicy
```

```python
# kdit/operations/attention/backends/radial_sage_attn/radial_sage_attn.py

# ✅ 跨子目录 → 绝对导入
from kdit.operations.attention.attention_op import KsanaAttentionBackendImpl

# ❌ 禁止
# from ...attention_op import KsanaAttentionBackendImpl
```

### 适用范围

本规则适用于 `kdit/` 包下所有 Python 模块，包括但不限于：
- `kdit/nodes/` — loader / encoder / decoder / generator 节点
- `kdit/operations/` — attention / linear / fuse_qkv 算子
- `kdit/adapter/comfyui/` — ComfyUI 适配层
- `kdit/models/` — 模型实现（wan / qwen）

### 自动检查

可通过以下命令验证是否存在违规的三级相对导入：

```bash
grep -rn "from \.\.\." kdit/
```

预期输出为空。

---

## 2. 类型注解与导入规范

### 规则

项目要求 Python ≥3.10，PEP 604（`X | Y`）和 PEP 585（`list[str]`）均已原生支持，**一般不需要 `from __future__ import annotations`**。

**唯一例外**：当类方法的返回类型引用自身类（前向引用）时，仍需 `from __future__ import annotations` 使注解延迟求值。例如 `-> Pipeline`（在 `Pipeline` 类内部）、`-> PipelineDefBuilder`（在 `PipelineDefBuilder` 类内部）。

### 禁止事项

- ❌ **禁止使用 `from __future__ import annotations`** — 除非文件中存在前向引用（类方法返回自身类型），否则不需要
- ❌ **禁止使用 `typing.TYPE_CHECKING`** — 所有导入必须是普通导入，不使用 `if TYPE_CHECKING:` 保护
- ❌ **尽量避免 `from typing import`** — 优先使用 `collections.abc`（如 `Callable`, `Sequence`, `Mapping`）和内置泛型（如 `list[str]`, `dict[str, int]`）。只有 `Any` 等无替代品的类型才从 `typing` 导入
- ❌ **避免重复导入** — 同一个模块中不要出现多条导入同一来源的语句，应合并为一条

### 示例

```python
# ❌ 禁止 — 无前向引用时不需要 future annotations
from __future__ import annotations  # 删除

# ✅ 例外 — 类方法返回自身类型（前向引用）时需要保留
from __future__ import annotations  # 保留

class PipelineDefBuilder:
    def load(self, ...) -> PipelineDefBuilder:  # 引用自身类
        ...

# ❌ 禁止 — 不使用 TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..foo import Bar  # 改为普通导入
```

```python
# ✅ Python 3.10+ 原生支持联合类型和内置泛型
@dataclass(frozen=True)
class SampleConfig:
    steps: int | None = None
    cfg_scale: float | list[float, float] | None = None
```

```python
# ✅ 优先使用 collections.abc 而非 typing
from collections.abc import Callable, Sequence  # ✅
# from typing import Callable, Sequence  # ❌ 避免

from typing import Any  # ✅ Any 无替代品，可以从 typing 导入
```

---

## 3. Key 类型体系设计规范

### 三种 Key 的职责

| Key 类型 | 定义位置 | 语义 | 使用场景 |
|----------|---------|------|---------|
| `ModelKey` | `kdit/models/model_key.py` | 标识一个具体的模型类别 | `ModelPool` 存取、`KsanaModel.__init__`、Loader/Infer Node 注册与分发、`GeneratorFactory` 注册、`settings` 配置映射 |
| `PipelineKey` | `kdit/pipelines/pipeline_key.py` | 标识一条完整的推理流水线 | `KsanaBasePipeline.__init__`、pipeline 创建与路由、`base_pipeline` 中 pipeline→model 映射表的 key 侧 |
| `InferNodeType` | `kdit/nodes/core/node_types.py` | 标识推理节点类型 | `InferNodeFactory` 注册与分发、`executor.run_infer_node` |

### 核心约束

1. **`ModelPool` 只接受 `ModelKey`** — 不允许传入 `PipelineKey` 或其他类型。
2. **`ModelKey` 和 `PipelineKey` 是独立枚举** — 不存在别名关系（如 `ModelKey = PipelineKey`），即使部分成员同名。
3. **DiffusionModel 的 `ModelKey` 成员与 `PipelineKey` 同名** — 因为不同 pipeline 的 diffusion model 权重不同，需要独立的 key。
4. **`get_model_key_from_path()` 统一返回 `ModelKey`** — 调用方如需 `PipelineKey`，必须自行通过 `PipelineKey[model_key.name]` 转换。

### 成员分类

```python
class ModelKey(Enum):
    # Text Encoders
    T5TextEncoder = auto()
    Qwen2VLTextEncoder = auto()
    Qwen2VLTextEncoderMultimodal = auto()

    # VAE
    QwenImageVAE = auto()
    VAE_WAN2_1 = auto()
    VAE_WAN2_2 = auto()

    # Diffusion Models — 与 PipelineKey 同名
    Wan2_2_T2V_14B = auto()
    Wan2_2_I2V_14B = auto()
    Wan2_2_TI2V_5B = auto()
    Wan2_1_VACE_14B = auto()
    QwenImage_T2I = auto()
    QwenImage_Edit = auto()
```

### Pipeline → Model 映射方向

`base_pipeline.py` 中的映射表遵循 **pipeline key → model key** 方向：

```python
# key 侧: PipelineKey（输入）
# value 侧: ModelKey（输出）
_TEXT_ENCODER_MAP = {
    PipelineKey.Wan2_2_T2V_14B: ModelKey.T5TextEncoder,
    ...
}
```

### 禁止事项

- ❌ 不要创建 `ModelKey = PipelineKey` 这样的别名
- ❌ 不要让 `ModelPool` 接受 `PipelineKey`
- ❌ 不要在 `ModelKey` 中添加 pipeline 级别的概念
- ❌ 不要创建未被任何代码使用的 Key 成员（如曾经的 `WanDiT_14B`）

## 4. V5 Node / Tensor API 规范

### Node.run() 返回值

| Node 类型 | 返回值 | 说明 |
|-----------|--------|------|
| `LoaderNode.run()` | `None` | 模型写入 model_pool，不返回 |
| `InferNode.run()` | `None` | 结果写入 tensor_pool，不返回 tensor |

**禁止** `InferNode.run()` 返回 dict 或 tensor。所有中间结果通过 `tensor_pool.put(key, tensor)` 写入。

### output_tensor_keys 静态声明

`output_tensor_keys` 保留为**类属性**，Executor 用它做 `RANK_0_BROADCAST`。
对于条件性输出（如 VAEEncodeNode 可能不写入 img_latents），broadcast 逻辑对 `tensor_pool.peek(key) is None` 的 key 做 skip 容错。

```python
class GeneratorNode(InferNode):
    output_tensor_keys = [TensorKey.LATENTS]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx) -> None:
        latents = generator.run(...)
        tensor_pool.put(TensorKey.LATENTS, latents)
```

### Node.run() 写入规则

Node 必须**无条件写入** tensor_pool，不得判断 `rank_id`：

```python
# ✅ 正确
def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx) -> None:
    result = unit.run(...)
    tensor_pool.put(TensorKey.LATENTS, result)

# ❌ 错误 — Node 不应关心下游的 dispatch_policy
def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx) -> None:
    result = unit.run(...)
    if device_ctx.rank_id == 0:
        tensor_pool.put(TensorKey.LATENTS, result)
```

### TensorKey 枚举

tensor_pool 的 key **必须**使用 `TensorKey(str, Enum)` 枚举，全局统一，不使用裸字符串：

```python
from kdit.nodes.core.tensor_keys import TensorKey

# ✅ 使用枚举
tensor_pool.put(TensorKey.LATENTS, latents)
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
| `TensorKey.IMG_LATENTS` | `"img_latents"` | VAEEncodeNode | GeneratorNode |
| `TensorKey.LATENTS` | `"latents"` | GeneratorNode | VAEDecodeNode |
| `TensorKey.VIDEO` | `"video"` | VAEDecodeNode | Pipeline/ComfyUI |
| `TensorKey.IMAGE` | `"image"` | Pipeline put_tensors | VAEEncodeNode |
| `TensorKey.START_IMG` | `"start_img"` | Pipeline put_tensors | VAEEncodeNode |
| `TensorKey.END_IMG` | `"end_img"` | Pipeline put_tensors | VAEEncodeNode |
| `TensorKey.BASE_LATENT` | `"base_latent"` | VAEEncodeSpatialNode / Pipeline put_tensors | GeneratorNode |
| `TensorKey.AUX_LATENT` | `"aux_latent"` | Pipeline put_tensors | GeneratorNode |

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
| `engine.put_tensors(**kw)` | 写入 tensor | 写入所有 Executor 的 tensor_pool，自动包装为 `TensorValue` |
| `engine.tensor_scope(keep=[...])` | 生命周期管理 | context manager，scope 结束自动 clear（keep 列表中的 key 保留） |
| `engine.has_tensor(key)` | 检查 key 存在性 | 检查 rank 0 的 tensor_pool 中是否存在指定 key |

**保持 `with` 模式的理由**：
1. **异常安全**：即使 Node 抛异常，tensor_pool 也会被清理
2. **语义清晰**：标记一次推理的 tensor 边界
3. **Node 不应自管理清理**：Node 不知道 pipeline 的全局编排，无法判断哪些 tensor 还被下游需要
4. **tensor 数量少**：pool 中通常只有 5-6 个 tensor，全量清理开销可忽略

### Tensor 生命周期与 TensorValue

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

#### tensor_scope(keep=[...]) 声明式保留

`tensor_scope` 支持 `keep` 参数，声明哪些 key 在 scope 退出时保留：

```python
# 分段调用：vae_encode 保留 BASE_LATENT
with engine.tensor_scope(keep=[TensorKey.BASE_LATENT]):
    engine.put_tensors(**{TensorKey.IMAGE: image})
    engine.run_infer_node(InferNodeType.VAE_ENCODE_SPATIAL, vae, context)
# scope 退出：IMAGE 被 release，BASE_LATENT 保留在 pool 中

# 最终步骤：vae_decode 不 keep，全部清理
with engine.tensor_scope():
    engine.run_infer_node(InferNodeType.VAE_DECODE, vae, context)
    video = engine.get_tensor(TensorKey.VIDEO).data  # 取裸 tensor
# scope 退出：全部 release
```

#### ComfyUI adapter 之间传递 key

ComfyUI adapter 之间传递的是 `TensorKey`（而非裸 tensor），真正的 `TensorValue` 存在 Executor 的 pool 中：

```python
# vae_encode 返回 key
def vae_encode_image(...):
    with engine.tensor_scope(keep=[TensorKey.BASE_LATENT]):
        engine.put_tensors(...)
        engine.run_infer_node(...)
    return KsanaNodeVAEEncodeOutput(samples=TensorKey.BASE_LATENT, ...)

# generate 收到 key，先检查存在性
def generate(...):
    if not engine.has_tensor(base_latent_key):
        raise RuntimeError(f"Tensor {base_latent_key} not found in pool")
    with engine.tensor_scope(keep=[TensorKey.LATENTS]):
        engine.run_infer_node(...)
    return KsanaNodeGeneratorOutput(samples=TensorKey.LATENTS, ...)
```

#### Pool 的 clear(exclude=[...])

`TensorPool.clear(exclude)` 释放除 exclude 列表外的所有 tensor：

```python
def clear(self, exclude: list[TensorKey] | None = None) -> None:
    exclude_set = set(exclude) if exclude else set()
    keys_to_remove = [k for k in self._tensors if k not in exclude_set]
    for key in keys_to_remove:
        self._tensors[key].release()
        del self._tensors[key]
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
 │    ├── 单卡模式: 1 个 KsanaExecutor 实例
 │    └── 多卡模式: N 个 RayKsanaExecutor (Ray Actor)
 ├── owns: num_gpus, _is_ray, _cleaned_up (引擎级元数据)
 ├── NOT own: model_pool, tensor_pool, device 信息 (这些属于 Executor)
 └── NOT own: 任何 Node 实例 (Node 由 AdvancedFactory 按需创建，用完即弃)

KsanaExecutor (每卡一个实例)
 ├── owns: model_pool        — ModelPool (存储已加载的模型)
 ├── owns: tensor_pool       — TensorPool (存储推理中间 tensor)
 ├── owns: dist_group        — DistributedGroupManager (管理 torch.distributed)
 ├── owns: device_ctx        — NodeDeviceContext (frozen dataclass, 只读)
 ├── owns: device / offload_device / device_id (设备信息)
 ├── owns: rank_id / world_size (分布式信息)
 ├── owns: dist_config / shard_fn (分布式配置)
 ├── owns: local_pipeline (遗留 V4 接口，将废弃)
 └── NOT own: Node 实例 (Node 在 run_loader_node / run_infer_node 中临时创建)
```

### 各组件详细 Ownership

#### Engine (`kdit/engine/engine.py`)

| 属性 | 类型 | 说明 |
|------|------|------|
| `executors` | `KsanaExecutor` 或 `list[RayKsanaExecutor]` | **唯一核心持有物**。单卡时是一个实例，多卡时是 Ray Actor 列表 |
| `num_gpus` | `int` | GPU 数量，从 dist_config 复制 |
| `_is_ray` | `bool` | 是否使用 Ray 分布式 |
| `_cleaned_up` | `bool` | 清理标记，防止重复清理 |

**Engine 不持有**：model_pool、tensor_pool、device 信息、Node 实例。Engine 是纯粹的**分发层**，所有实际资源都在 Executor 上。

**Engine 的桥接方法**（透传到 Executor）：
- `run_loader_node()` → 分发到所有 Executor
- `run_infer_node()` → 分发到所有 Executor
- `put_tensors()` → 写入所有 Executor 的 tensor_pool
- `get_tensor()` → 从 rank 0 Executor 的 tensor_pool 读取
- `inference_session()` → 管理所有 Executor 的 tensor_pool 生命周期

#### KsanaExecutor (`kdit/executor/executor.py`)

| 属性 | 类型 | 生命周期 | 说明 |
|------|------|---------|------|
| `model_pool` | `ModelPool` | 与 Executor 同生命周期 | 存储所有已加载模型，按 `ModelKey` 索引 |
| `tensor_pool` | `TensorPool` | 每次 `inference_session()` 结束时 clear | 存储推理中间 tensor，按 string key 索引 |
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
- **生命周期**: 每次 `engine.tensor_scope()` 退出（depth→0）时 `clear(exclude=keep)`
- **内容**: `dict[TensorKey, TensorValue]`，每个 TensorValue 持有 `Tensor | list[Tensor]`
- **用途**: Node 间通过 `TensorKey` 引用 tensor，避免 tensor 跨 Ray 边界序列化
- **关键方法**: `put` / `get`（返回 TensorValue）/ `clear(exclude)` / `has` / `keys` / `__len__`

#### ModelPool (`kdit/models/model_pool.py`)

- **Owner**: Executor
- **生命周期**: 与 Executor 同生命周期，`clear_models()` 可手动清理
- **内容**: `dict[ModelKey, KsanaModel]`
- **用途**: LoaderNode 写入模型，InferNode 读取模型

#### DistributedGroupManager (`kdit/executor/distributed_group.py`)

- **Owner**: Executor
- **状态**: `rank_id`, `world_size`, `_initialized`
- **用途**: 提供 `broadcast_tensors()` 能力，配合 tensor_pool 实现跨 rank 数据同步

### Node 状态分析

#### LoaderNode (`LoaderNode` 子类)

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
- 每次 `run()` 从 `tensor_pool` 读输入、从 `model_pool` 读模型
- 计算结果写入 `tensor_pool`
- 不持有任何跨调用的状态

### Node 创建方式

Node 由 AdvancedFactory **按需创建**，每次 `run_loader_node()` / `run_infer_node()` 调用时创建新实例：

```python
# executor.py
def run_loader_node(self, model_key, **kwargs):
    node = LoaderNodeFactory.create(model_key)  # ← 每次新建
    node.run(model_key, model_pool=self.model_pool, device_ctx=self.device_ctx, **kwargs)

def run_infer_node(self, infer_node_type, model_key, context):
    node = InferNodeFactory.create(infer_node_type, model_key)  # ← 每次新建
    self._pre_sync_tensors(node, policy)
    is_active_rank = policy == NodeDispatchPolicy.ALL_ALL_ALL or self.device_ctx.rank_id == 0
    if is_active_rank:
        node.run(model_key, context, tensor_pool=self.tensor_pool, model_pool=self.model_pool, device_ctx=self.device_ctx)
    self._post_sync_tensors(node, policy)
```

Node 实例是**临时对象**，用完即弃。Executor 不持有 Node 引用。

### 设计约束总结

| 约束 | 说明 |
|------|------|
| Engine 不持有资源 | Engine 只是分发层，所有实际资源（model_pool, tensor_pool, device）在 Executor 上 |
| Executor 持有所有资源 | model_pool + tensor_pool + dist_group + device_ctx |
| Node 无状态（理想） | InferNode 完全无状态；LoaderNode 中 `DiffusionLoaderNode` 有类级 `_pinned_memory_manager` 例外 |
| NodeDeviceContext 只读 | `frozen=True` dataclass，Node 无法篡改 |
| NodeContext 无 tensor | `__post_init__` 强制校验不含 `torch.Tensor`，保证可跨 Ray 序列化 |
| tensor_pool 生命周期 | 由 `engine.inference_session()` 管理，session 结束自动 clear |
| model_pool 生命周期 | 与 Executor 同生命周期，需手动 `clear_models()` 释放 |
| InferNode.run() 签名固定 | 禁止添加 `**kwargs` 或额外参数，额外配置通过 `context.metadata` 传递 |
| Tensor 只能通过 tensor_pool 流转 | 禁止在 run() 参数或 context.metadata 中传递 tensor |

---

## 6. InferNode 开发规范

### run() 签名固定，禁止扩展

`InferNode.run()` 签名是固定的：

```python
def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx) -> None:
```

- **禁止** 添加 `**kwargs` 或任何额外参数
- **禁止** 返回值（必须返回 `None`）
- 如需传递额外配置，使用 `context.metadata` 字典

### Tensor 只能通过 tensor_pool 流转

- **输入 tensor**: 只能通过 `tensor_pool.get(key)` 或 `tensor_pool.peek(key)` 获取
- **输出 tensor**: 只能通过 `tensor_pool.put(key, tensor)` 写入
- **禁止** 在 `run()` 参数中传递 tensor
- **禁止** 在 `context.metadata` 中放 tensor（`NodeContext.__post_init__` 会校验）

### 声明 tensor 契约

每个 Node 必须声明 `input_tensor_keys` 和 `output_tensor_keys`：

```python
class MyNode(InferNode):
    input_tensor_keys = [TensorKey.POSITIVE, TensorKey.NEGATIVE]
    output_tensor_keys = [TensorKey.LATENTS]
```

- `input_tensor_keys` 可以为空列表 `[]`（如 TextEncodeNode 不依赖其他 tensor）
- `output_tensor_keys` 用于 `R0_R0_BCAST` 策略时指定需要 broadcast 的 key

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
- `InferNodeType` 枚举值：`TEXT_ENCODE`, `VAE_ENCODE_SPATIAL`, `VAE_ENCODE_IMAGES`, `VAE_DECODE`, `GENERATE`

### 现有 Node 参考

| Node | dispatch_policy | input_tensor_keys | output_tensor_keys |
|------|----------------|-------------------|-------------------|
| `TextEncodeNode` | `ALL_ALL_ALL` | `[]` | `[POSITIVE, NEGATIVE]` |
| `VAEEncodeSpatialNode` | `R0_R0_BCAST` | `[START_IMG, END_IMG]` | `[BASE_LATENT]` |
| `VAEEncodeImagesNode` | `R0_R0_BCAST` | `[IMAGE]` | `[IMAGE_EMBEDS]` |
| `VAEDecodeNode` | `ALL_R0_R0` | `[LATENTS]` | `[VIDEO]` |
| `GeneratorNode` | `ALL_ALL_ALL` | `[POSITIVE, NEGATIVE, BASE_LATENT, AUX_LATENT]` | `[LATENTS]` |

### Executor 同步机制

`KsanaExecutor.run_infer_node()` 负责：

1. **`_pre_sync_tensors()`**: 执行前的 tensor 同步（预留接口，未来可自动 broadcast 输入）
2. **`is_active_rank`**: 根据 policy 判断当前卡是否执行 `run()`
3. **`_post_sync_tensors()`**: 执行后的 tensor 同步（`R0_R0_BCAST` 时 broadcast output_tensor_keys）

Node 内部不需要感知多卡逻辑，Executor 负责所有 tensor 的 pre/post 同步。

---

## 7. Adapter 依赖方向规则

### 规则

第三方框架的适配代码**只能**放在 `kdit/adapter/` 目录下。依赖方向是**单向**的：

```
kdit/adapter/comfyui/  →  kdit/  (✅ adapter 可以 import kdit 核心代码)
kdit/                  →  kdit/adapter/  (❌ 核心代码禁止 import adapter)
```

### 约束

| 规则 | 说明 |
|------|------|
| adapter → kdit 核心 | ✅ 允许。adapter 代码可以自由使用 kdit 核心模块 |
| kdit 核心 → adapter | ❌ **禁止**。`kdit/` 下除 `adapter/` 外的任何模块不得 import `kdit.adapter.*` |
| adapter 间互相引用 | ⚠️ 谨慎。不同 adapter 之间尽量不互相依赖 |

### 原因

- **防止循环引用**：adapter 依赖核心，核心不依赖 adapter，保证依赖图是 DAG
- **可选安装**：adapter 可以作为可选组件，核心包不因缺少第三方框架而报错
- **解耦**：新增/删除 adapter 不影响核心代码

### 自动检查

```bash
# 检查 kdit/ 核心代码（排除 adapter/）是否引用了 adapter
grep -rn "from kdit.adapter" kdit/ --include="*.py" | grep -v __pycache__ | grep -v "kdit/adapter/"
grep -rn "import kdit.adapter" kdit/ --include="*.py" | grep -v __pycache__ | grep -v "kdit/adapter/"
```

预期输出为空。

---

## 8. 类命名规范：去除 `Ksana` 前缀

### 规则

`kdit/` 包内的自定义类名**不加** `Ksana` 前缀。因为已经在 `kdit` 命名空间下，前缀是冗余的。

| 场景 | 规则 | 示例 |
|------|------|------|
| `kdit/` 包内类定义 | **不加** `Ksana` 前缀 | `Engine`、`Executor`、`Pipeline`、`ModelKey` |
| `comfyui/` 及 `kdit/adapter/comfyui/` 中的类 | **可以保留** `Ksana` 前缀 | `KsanaNodeModelLoader`、`KsanaNodeGeneratorOutput` |
| `KSANA_` 开头的常量 | **保留** | `KSANA_LOGGER_LEVEL`、`KSANA_PREFETCH_WEIGHTS` |

### 重命名映射表（待逐个确认执行）

以下类需要去除 `Ksana` 前缀，重命名时需同步修改所有引用（包括 `kdit/`、`tests/`、`examples/`、`.roo/rules-code/`、`.skills/` 中的引用）：

| 当前名称 | 目标名称 | 定义文件 |
|----------|---------|---------|
| `KsanaCache` | `Cache` | `kdit/cache/base_cache.py` |
| `KsanaStepCache` | `StepCache` | `kdit/cache/base_cache.py` |
| `KsanaBlockCache` | `BlockCache` | `kdit/cache/base_cache.py` |
| `KsanaHybridCache` | `HybridCache` | `kdit/cache/base_cache.py` |
| `KsanaLinearBackend` | `LinearBackend` | `kdit/config/linear_config.py` |
| `KsanaBlockCacheConfig` | `BlockCacheConfig` | `kdit/config/cache_config/base.py` |
| `KsanaVideoControlConfig` | `VideoControlConfig` | `kdit/config/video_control_config.py` |
| `KsanaAttentionBackend` | `AttentionBackend` | `kdit/config/attention_config.py` |
| `KsanaAttentionConfig` | `AttentionConfig` | `kdit/config/attention_config.py` |
| `KsanaRadialSageAttentionConfig` | `RadialSageAttentionConfig` | `kdit/config/attention_config.py` |
| `KsanaSageSLAConfig` | `SageSLAConfig` | `kdit/config/attention_config.py` |
| `KsanaTorchCompileConfig` | `TorchCompileConfig` | `kdit/config/torch_compile_config.py` |
| `KsanaSLGConfig` | `SLGConfig` | `kdit/config/wan_experimental_config.py` |
| `KsanaFETAConfig` | `FETAConfig` | `kdit/config/wan_experimental_config.py` |
| `KsanaExperimentalConfig` | `ExperimentalConfig` | `kdit/config/wan_experimental_config.py` |
| `KsanaBatchScheduler` | `BatchScheduler` | `kdit/scheduler/scheduler.py` |
| `KsanaProfiler` | `Profiler` | `kdit/utils/profile.py` |
| `KsanaQwenImageVAE` | `QwenImageVAE` | `kdit/models/qwen/vae.py` |
| `KsanaDiffusionModel` | `DiffusionModel` | `kdit/models/diffusion_model.py` |
| `KsanaWanModel` | `WanModel` | `kdit/models/diffusion_model.py` |
| `KsanaWanVaceModel` | `WanVaceModel` | `kdit/models/diffusion_model.py` |
| `KsanaQwenImageModel` | `QwenImageModel` | `kdit/models/diffusion_model.py` |
| `KsanaVAEModel` | `VAEModel` | `kdit/models/vae_model.py` |
| `KsanaWanVAEModel` | `WanVAEModel` | `kdit/models/vae_model.py` |
| `KsanaQwenVAEModel` | `QwenVAEModel` | `kdit/models/vae_model.py` |
| `KsanaTextEncoderModel` | `TextEncoderModel` | `kdit/models/text_encoder_model.py` |
| `KsanaAttentionOp` | `AttentionOp` | `kdit/operations/attention/attention_op.py` |
| `KsanaAttentionBackendImpl` | `AttentionBackendImpl` | `kdit/operations/attention/backends/base.py` |
| `KsanaWanGenerator` | `WanGenerator` | `kdit/generators/wan_generator.py` |
| `KsanaQwenGenerator` | `QwenGenerator` | `kdit/generators/qwen_generator.py` |
| `KsanaBaseGenerator` | `BaseGenerator` | `kdit/generators/base_generator.py` |
| `KsanaVaceGenerator` | `VaceGenerator` | `kdit/generators/vace_generator.py` |
| `KsanaExecutor` | `Executor` | `kdit/executor/executor.py` |

### 保留 `Ksana` 前缀的类（comfyui 适配层）

以下类位于 `kdit/adapter/comfyui/`，保留 `Ksana` 前缀：

- `KsanaNodeTeaCache`、`KsanaNodeEasyCache`、`KsanaNodeMagCache`、`KsanaNodeDBCache`
- `KsanaNodeModelLoaderOutput`、`KsanaNodeGeneratorOutput`、`KsanaNodeVAEEncodeOutput`
- `KsanaNodeModelLoader`、`KsanaNodeVAELoader`

### 执行步骤

每次重命名一个类时：
1. 修改类定义
2. `grep -rn "旧名称" kdit/ tests/ examples/ .roo/ .skills/` 找到所有引用
3. 同步修改所有引用
4. 运行 `python -c "import ast; ast.parse(open('文件').read())"` 验证语法
5. 运行相关单测验证功能

### 自动检查

```bash
# 检查 kdit/ 下（排除 adapter/comfy/）是否还有 Ksana 前缀的类
grep -rn "class Ksana" kdit/ --include="*.py" | grep -v __pycache__ | grep -v "adapter/comfyui/"
```

预期输出为空（重构完成后）。

---

## 9. NodeContext metadata 重构（TODO）

> **状态**: 待实施（Pipeline 重构完成后）
> **关联设计**: [`plans/pipeline_refactor_design.md` §1.2](../plans/pipeline_refactor_design.md:62)

### 现状问题

[`NodeContext.metadata`](../kdit/nodes/core/node_context.py:37) 是一个无类型的 `dict`，当前承载了多种混合关注点：

| metadata key | 使用方 | 类型 |
|---|---|---|
| `text_run_device` | [`T5TextEncodeNode`](../kdit/nodes/infers/text_encoder_node.py:80) | `str` |
| `offload_model` | [`T5TextEncodeNode`](../kdit/nodes/infers/text_encoder_node.py:82), [`VAEDecodeNode`](../kdit/nodes/infers/vae_decoder_node.py:43) | `bool` |
| `noise_shape` | [`GeneratorNode`](../kdit/nodes/infers/generator_node.py:55) | `list[int]` |
| `control_video_config` | [`GeneratorNode`](../kdit/nodes/infers/generator_node.py:57) | `VideoControlConfig` |
| `video_control` | [`GeneratorNode`](../kdit/nodes/infers/generator_node.py:56) | `dict` |
| `with_end_image` | [`VAEDecodeNode`](../kdit/nodes/infers/vae_decoder_node.py:44) | `bool` |
| `condition_image_path` | [`QwenTextEncodeNode`](../kdit/nodes/infers/text_encoder_node.py:126) | `str` |
| `comfy_bar_callback` | [`GeneratorNode`](../kdit/nodes/infers/generator_node.py:58) | `callable` |

### 重构方向

1. **提升高频 key 为 `NodeContext` 的显式字段**：
   - `offload_model: bool = False`
   - `text_run_device: str | None = None`
   - `noise_shape: list[int] | None = None`
   - `target_size: tuple[int, int, int] | None = None` （合并 target_f/h/w）

2. **保留 `metadata: dict` 用于扩展**：
   - ComfyUI adapter 动态参数（如 `comfy_bar_callback`）
   - 实验性功能参数

3. **同步更新所有 InferNode 的 `run()` 方法**：
   - 从 `context.metadata["key"]` 改为 `context.key`
   - 保持向后兼容：优先读显式字段，fallback 到 metadata

4. **同步更新 ComfyUI adapter**：
   - [`generate.py`](../kdit/adapter/comfyui/generate.py) 中构建 context 的代码

### 约束

- `__post_init__` 中的 tensor 校验逻辑保留
- 不改变 `NodeContext` 的可序列化约束（Ray 多卡广播）
- `comfy_bar_callback` 等不可序列化对象**仍然放 metadata**，不提升为字段

### 自动检查

```bash
# 检查 metadata 中还有哪些 key 在使用
grep -rn 'metadata\[' kdit/nodes/ --include="*.py" | grep -v __pycache__
grep -rn 'metadata\.get' kdit/nodes/ --include="*.py" | grep -v __pycache__
```

---

## 10. Pipeline 声明式架构（V4 设计规范）

> **设计文档**: [`plans/pipeline_refactor_design.md`](../plans/pipeline_refactor_design.md)
> **核心理念**: Pipeline 是声明式的 Node 流程定义，不是命令式的代码流程

### 架构总览

```
PipelineDefBuilder  ──build()──▶  PipelineDef (不可变)
                                    ├── load_phases: list[LoadPhase]
                                    ├── infer_phases: list[InferPhase]
                                    └── context_builder_cls: type[ContextBuilder]

Pipeline.from_models(pipeline_key)  ──▶  Pipeline 实例
Pipeline.generate(prompt, ...)      ──▶  输出 tensor / 保存文件
```

### 核心数据类

```python
# kdit/pipelines/pipeline_def.py

@dataclass(frozen=True)
class LoadPhase:
    model_role: str           # "text_encoder" / "diffusion" / "vae"
    model_key: ModelKey       # 具体模型 key

@dataclass(frozen=True)
class InferPhase:
    node_type: InferNodeType  # TEXT_ENCODE / GENERATE / VAE_DECODE / SAVE_VIDEO ...
    model_role: str | None    # 关联的 model_role，SaveNode 为 None
    condition: str | None     # ContextBuilder 上的条件方法名

@dataclass(frozen=True)
class PipelineDef:
    pipeline_key: PipelineKey
    load_phases: tuple[LoadPhase, ...]
    infer_phases: tuple[InferPhase, ...]
    context_builder_cls: type[ContextBuilder]
    keep_tensors: tuple[TensorKey, ...] = ()  # tensor_scope keep 列表
```

### PipelineDefBuilder — 链式构建

```python
# 使用示例
WAN_T2V_DEF = (
    PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
    .load("text_encoder", ModelKey.T5TextEncoder)
    .load("diffusion", ModelKey.Wan2_2_T2V_14B)
    .load("vae", ModelKey.Wan2_2_VAE)
    .add_infer(NT.TEXT_ENCODE, model_role="text_encoder")
    .add_infer(NT.GENERATE, model_role="diffusion")
    .add_infer(NT.VAE_DECODE, model_role="vae")
    .add_infer(NT.SAVE_VIDEO)                          # model_role=None
    .keep_tensors(TensorKey.VIDEO)
    .context_builder(WanT2VContextBuilder)
    .build()
)
```

**规则**:
- `load()` 的 `model_role` 是自由字符串，在 `add_infer()` 中通过同名引用
- `add_infer()` 不指定 `model_role` 时默认为 `None`（如 SaveNode）
- `.when("condition_name")` 链在 `add_infer()` 后，设置条件执行
- `build()` 返回 frozen `PipelineDef`，之后不可修改

### ContextBuilder — 为每个 Phase 构建上下文

```python
# kdit/pipelines/context_builder.py

class ContextBuilder(ABC):
    """生命周期：
    1. prepare_generate_inputs(base_inputs, **kwargs) — 一次性：提取 Pipeline 特有输入
    2. 对每个 InferPhase:
       a. check_condition(name, inputs) — 是否跳过
       b. prepare_tensors(phase, inputs) — 准备 tensor → put 到 pool
       c. build_context(phase, inputs) — 构建 NodeContext
    3. post_process(output, inputs) — 输出后处理
    """

    def prepare_generate_inputs(self, base_inputs: PipelineGenerateInputs, **kwargs) -> None:
        """从 kwargs 提取 Pipeline 特有输入，存入 self._extra。"""
        pass

    @abstractmethod
    def build_context(self, phase: InferPhase, inputs: PipelineGenerateInputs) -> NodeContext:
        """按 phase.node_type 分支，构建该 Node 的 context。"""
        ...

    def prepare_tensors(self, phase: InferPhase, inputs: PipelineGenerateInputs) -> dict | None:
        """返回需要 put 到 tensor_pool 的 tensor dict。默认 None。"""
        return None

    def check_condition(self, condition_name: str, inputs: PipelineGenerateInputs) -> bool:
        """查找 self 上的同名方法并调用。"""
        checker = getattr(self, condition_name, None)
        if checker is None:
            raise ValueError(f"Condition '{condition_name}' not found")
        return checker(inputs)

    def post_process(self, output_tensor, inputs: PipelineGenerateInputs) -> any:
        """输出后处理。默认直接返回。"""
        return output_tensor
```

**方法命名约定**:
- `prepare_generate_inputs` — 全局一次，"准备 generate 阶段的输入"
- `build_context` — 每个 phase，"构建 NodeContext"
- `prepare_tensors` — 每个 phase，"准备 tensor 到 pool"
- 条件方法 — 在子类上定义，如 `has_start_image(self, inputs) -> bool`

### PipelineGenerateInputs — 最小公共集

```python
# kdit/pipelines/generate_inputs.py

@dataclass
class PipelineGenerateInputs:
    """所有 Pipeline 共有的输入。"""
    prompt: str | list[str]
    prompt_negative: str | list[str] | None
    num_prompts: int
    sample_config: SampleConfig
    runtime_config: RuntimeConfig
    cache_config: list | None
    has_lora: bool
```

**规则**:
- 只包含**所有** Pipeline 都需要的字段
- Pipeline 特有字段由 `ContextBuilder.prepare_generate_inputs()` 从 `**kwargs` 提取，存入 `self._extra`
- `self._extra` 类型由子类自定义（推荐用内部 `@dataclass class ExtraPipelineGenerateInputs`）

### SaveNode — 输出保存作为 InferNode

```python
# kdit/nodes/infers/save_node.py

@InferNodeFactory.register(NT.SAVE_VIDEO, None)
class SaveVideoNode(InferNode):
    input_tensor_keys = [TensorKey.VIDEO]
    output_tensor_keys = []
    dispatch_policy = NodeDispatchPolicy.ALL_R0_R0  # 只在 rank 0 保存

@InferNodeFactory.register(NT.SAVE_IMAGE, None)
class SaveImageNode(InferNode):
    input_tensor_keys = [TensorKey.VIDEO]  # 复用 VIDEO key
    output_tensor_keys = []
    dispatch_policy = NodeDispatchPolicy.ALL_R0_R0
```

**规则**:
- SaveNode 注册时 `model_key=None`（不需要模型）
- SaveNode 在 `kdit/nodes/infers/` 中，**不在** `kdit/adapter/comfyui/` 中
- ComfyUI 模式下不使用 SaveNode（ComfyUI 有自己的输出机制）
- `InferPhase` 中 `model_role=None` 表示 SaveNode

### Pipeline.generate() 核心循环

```python
# Pipeline.generate() 伪代码
def generate(self, prompt, *, sample_config, runtime_config, **kwargs):
    inputs = PipelineGenerateInputs(prompt=prompt, ...)
    self._ctx_builder.prepare_generate_inputs(inputs, **kwargs)

    with self._engine.tensor_scope(keep=list(self._def.keep_tensors)):
        for phase in self._def.infer_phases:
            # 1. 条件检查
            if phase.condition and not self._ctx_builder.check_condition(phase.condition, inputs):
                continue
            # 2. 准备 tensor
            tensors = self._ctx_builder.prepare_tensors(phase, inputs)
            if tensors:
                self._engine.put_tensors(**tensors)
            # 3. 构建 context
            node_ctx = self._ctx_builder.build_context(phase, inputs)
            # 4. 执行
            model_key = self._model_keys.get(phase.model_role) if phase.model_role else None
            self._engine.run_infer_node(phase.node_type, model_key, node_ctx)

    return self._ctx_builder.post_process(
        self._engine.get_tensor(TensorKey.VIDEO), inputs
    )
```

### 条件执行 — `.when()` 机制

```python
# PipelineDefBuilder 中
.add_infer(NT.VAE_ENCODE_SPATIAL, model_role="vae").when("has_start_image")

# ContextBuilder 子类中
def has_start_image(self, inputs) -> bool:
    return self._extra.start_img_path is not None
```

**规则**:
- `.when("name")` 的 `name` 必须是 ContextBuilder 子类上的方法名
- 方法签名固定：`(self, inputs: PipelineGenerateInputs) -> bool`
- `check_condition()` 通过 `getattr` 查找，找不到则 raise

### 文件结构

```
kdit/pipelines/
├── __init__.py
├── pipeline.py              # 统一的 Pipeline 类
├── pipeline_def.py          # PipelineDef, LoadPhase, InferPhase, PipelineDefBuilder
├── pipeline_key.py          # PipelineKey 枚举（已有）
├── context_builder.py       # ContextBuilder 基类
├── generate_inputs.py       # PipelineGenerateInputs 数据类
├── context_builders/        # 各 Pipeline 的 ContextBuilder
│   ├── __init__.py
│   ├── wan.py               # WanContextBuilder, WanT2VContextBuilder, WanI2VContextBuilder
│   └── qwen.py              # QwenContextBuilder, QwenT2IContextBuilder, QwenEditContextBuilder
└── defs/                    # 各 Pipeline 的 PipelineDef 定义
    ├── __init__.py
    ├── wan_t2v.py
    ├── wan_i2v.py
    ├── qwen_t2i.py
    └── qwen_edit.py
```

### 禁止事项

- **禁止**在 `PipelineDef` 中放命令式逻辑（if/else/循环）
- **禁止**在 `ContextBuilder.build_context()` 中直接操作 tensor_pool（通过 `prepare_tensors` 返回）
- **禁止**在 `PipelineGenerateInputs` 中放 Pipeline 特有字段（用 `ContextBuilder._extra`）
- **禁止** `kdit/` 核心代码依赖 `kdit/adapter/`（方向：adapter → kdit）
- SaveNode 的 `model_key` 参数**必须为 None**
- **禁止**在 `Pipeline` 或公共函数中根据 `PipelineKey` / `ModelKey` 做 if/else 分支处理不同 Pipeline 的特例逻辑 — 所有 Pipeline 特有的输入处理、模型路径解析、LoRA 配置等**必须**放到对应的 `ContextBuilder` 子类中（通过覆盖 `resolve_model_paths()`、`resolve_lora_config()`、`build_loader_kwargs()` 等方法实现多态）

### ContextBuilder Load 阶段方法

`ContextBuilder` 基类提供三个 Load 阶段的默认实现，子类可按需覆盖：

| 方法 | 职责 | 覆盖场景 |
|------|------|---------|
| `resolve_model_paths()` | 解析 `model_path`（目录扫描、列表展开） | Wan 14B 高低噪声模型拆分 |
| `resolve_lora_config()` | 校验并包装 `LoraConfig` | Wan 14B 高低噪声 LoRA 拆分 |
| `build_loader_kwargs()` | 按 `ModelKey` 类别构建 loader 参数 | 默认实现已覆盖大多数场景 |

**设计原则**：Pipeline 特例逻辑通过 settings YAML 属性驱动（如 `diffusion.high_noise_checkpoint`），而非硬编码 PipelineKey 判断。ContextBuilder 子类读取 settings 属性决定行为，这样新增 Pipeline 变体时只需修改 YAML 配置，无需改动代码。

### Settings YAML 模块分类规范

`kdit/settings/{model_family}/modules/` 下的 YAML 文件按以下四个子目录分类：

| 子目录 | 内容 | 示例 |
|--------|------|------|
| `text_encoder/` | 文本编码器配置（`text_encoder:` 段） | `t5_encoder.yaml`、`text_encoder.yaml` |
| `diffusion/` | 扩散模型配置（`diffusion:` 段） | `14b.yaml`、`5b.yaml`、`vace.yaml` |
| `vae/` | VAE 配置（`vae:` 段） | `vae_2_1.yaml`、`vae.yaml` |
| `config/` | 运行时/采样配置（`runtime_config:`、`sample_config:` 段） | `t2v.yaml`、`i2v.yaml`、`lora.yaml` |

**规则**：
- `common.yaml` 保持在 `modules/` 根目录，不归入子目录
- 一个 YAML 文件只包含**一个类别**的配置；如果原始文件跨类别（如同时含 `diffusion:` 和 `sample_config:`），必须拆分为多个文件分别放入对应子目录
- 新增 YAML 模块时必须放入正确的子目录
- Pipeline 级 YAML（如 `wan/t2v_14b.yaml`）的 `_base_modules` 引用路径必须包含子目录前缀

**目录结构示例**（Wan）：

```
kdit/settings/wan/
├── t2v_14b.yaml              # Pipeline 级配置
├── i2v_14b.yaml
├── vace_14b.yaml
└── modules/
    ├── common.yaml           # 公共配置（不归入子目录）
    ├── text_encoder/
    │   └── t5_encoder.yaml
    ├── diffusion/
    │   ├── 5b.yaml
    │   ├── 14b.yaml
    │   └── vace.yaml         # VACE diffusion 覆盖
    ├── vae/
    │   ├── vae_2_1.yaml
    │   └── vae_2_2.yaml
    └── config/
        ├── t2v.yaml          # T2V runtime/sample 配置
        ├── i2v.yaml
        ├── lora.yaml
        └── vace.yaml         # VACE runtime/sample 配置
```

---

## 11. Lint 抑制注释规范

### 规则

对于**必要但未直接使用**的 import 和变量，必须添加对应的 lint 抑制注释，避免 CI 或 IDE 误报。

| 场景 | 抑制注释 | 说明 |
|------|---------|------|
| 必要但未使用的 import | `# noqa: F401  # pylint: disable=unused-import` | 如 `__init__.py` 中的 re-export、side-effect import、`TYPE_CHECKING` 块外的前向引用等 |
| 必要但未使用的变量 | `# noqa: F841  # pylint: disable=unused-variable` | 如从环境变量读取但仅用于触发副作用的变量、解构赋值中的占位变量等 |

对于如果需要因为方便都直接使用`import *`的时候，需要加上 `# noqa: F403`

### 示例

```python
# ✅ __init__.py 中 re-export（import 了但本文件未直接使用）
from .pipeline import Pipeline  # noqa: F401  # pylint: disable=unused-import
from .pipeline_def import PipelineDef  # noqa: F401  # pylint: disable=unused-import

# ✅ side-effect import（导入时触发注册逻辑，本文件不直接引用）
import kdit.generators.wan_generator  # noqa: F401  # pylint: disable=unused-import

# ✅ 必要但未使用的环境变量
SOME_FLAG = os.environ.get("SOME_FLAG", "0")  # noqa: F841  # pylint: disable=unused-variable

# ✅ 解构赋值中的占位变量
c1, c2, t, h, w = conv_weight.size()  # noqa: F841  # pylint: disable=unused-variable

# ✅ 导入所有内容， 通常不推荐，非必要情况还是显示import具体内容
from .defs import * # noqa: F403
```

### 判定标准

- **"必要的 import"** 指：删除后会导致功能异常的 import（如 re-export、side-effect 注册、`TYPE_CHECKING` 相关）
- **"必要的变量"** 指：删除后会导致功能异常的变量赋值（如环境变量读取触发副作用、模块级配置）
- 如果 import 或变量**确实不需要**，应该直接删除，而不是加抑制注释
- 抑制注释同时覆盖 flake8（`noqa: F401` / `noqa: F841`）和 pylint（`pylint: disable=unused-import` / `pylint: disable=unused-variable`），确保两种 linter 都不报警
- **注释顺序**：`# noqa` 在前，`# pylint` 在后，中间用两个空格分隔
