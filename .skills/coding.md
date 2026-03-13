# kDiT Coding Skills

## 1. Import 风格规范（方案 B）

### 规则

| 导入层级 | 写法 | 示例 |
|---------|------|------|
| 同目录（`.`） | **相对导入** | `from .base import Foo` |
| 同子包内（`..`） | **相对导入** | `from ..core.base_node import KsanaLoadNode` |
| 跨子包（`...` 及以上） | **绝对导入** | `from kdit.utils.factory import Factory` |

### 判定标准

- **"同子包"** 定义：共享 `kdit/` 下同一个一级子目录。例如 `kdit/nodes/loaders/` 和 `kdit/nodes/core/` 同属 `nodes` 子包。
- **三级及以上相对导入（`from ...xxx`）一律禁止**，必须改为绝对导入 `from kdit.xxx`。
- `kdit/operations/` 内部的深层嵌套（如 `backends/radial_sage_attn/`）跨子目录时也使用绝对导入。

### 示例

```python
# kdit/nodes/loaders/diffusion_model_loader.py

# ✅ 跨子包 → 绝对导入
from kdit.config import KsanaLoraConfig, KsanaModelConfig
from kdit.memory import PinnedMemoryManager
from kdit.models import KsanaWanModel
from kdit.utils import is_file_or_dir, log

# ✅ 同子包 (nodes) → 相对导入
from ..core.base_node import KsanaLoadNode
from ..core.node_factory import KsanaLoaderNodeFactory
from ..core.node_types import KsanaDispatchPolicy
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

## 2. `from __future__ import annotations` 使用规范

### 规则

**按需添加，不要预防性添加。** 只在以下场景使用 `from __future__ import annotations`：

| 场景 | 示例 |
|------|------|
| `TYPE_CHECKING` 前向引用 | `if TYPE_CHECKING: from ..foo import Bar` 然后在注解中使用 `Bar` |
| PEP 604 联合类型 `X \| Y` 用于运行时注解 | `def f(x: int \| None)` |
| PEP 585 内置泛型用于运行时注解 | `def f() -> list[str]`、`dict[str, int]` |
| 同文件内前向引用 | 类方法返回自身类型 `-> "MyClass"` |

### 不需要的场景

- 文件中**没有任何类型注解**（纯运行时代码）
- 只有基础类型注解（`-> bool`、`-> str`、`-> int`）
- 纯枚举/常量定义文件

### 示例

```python
# ❌ 不需要 — 无类型注解
from __future__ import annotations  # 多余，删除

class TextEncodeNode(KsanaInferNode):
    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        ...
```

```python
# ✅ 需要 — TYPE_CHECKING 前向引用
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..nodes.core.device_context import KsanaDeviceContext

class KsanaExecutor(ABC):
    def build_ctx(self) -> KsanaDeviceContext:  # 需要延迟求值
        ...
```

```python
# ✅ 需要 — PEP 604 联合类型
from __future__ import annotations

@dataclass(frozen=True)
class KsanaSampleConfig:
    steps: int | None = None          # 需要延迟求值
    cfg_scale: float | list[float, float] | None = None
```

---

## 3. Key 类型体系设计规范

### 三种 Key 的职责

| Key 类型 | 定义位置 | 语义 | 使用场景 |
|----------|---------|------|---------|
| `KsanaModelKey` | `kdit/models/model_key.py` | 标识一个具体的模型类别 | `KsanaModelPool` 存取、`KsanaModel.__init__`、Loader/Infer Node 注册与分发、`GeneratorFactory` 注册、`settings` 配置映射 |
| `KsanaPipelineKey` | `kdit/pipelines/pipeline_key.py` | 标识一条完整的推理流水线 | `KsanaBasePipeline.__init__`、pipeline 创建与路由、`base_pipeline` 中 pipeline→model 映射表的 key 侧 |
| `KsanaInferNodeType` | `kdit/nodes/core/node_types.py` | 标识推理节点类型 | `KsanaInferNodeFactory` 注册与分发、`executor.run_infer_node` |

### 核心约束

1. **`KsanaModelPool` 只接受 `KsanaModelKey`** — 不允许传入 `KsanaPipelineKey` 或其他类型。
2. **`KsanaModelKey` 和 `KsanaPipelineKey` 是独立枚举** — 不存在别名关系（如 `KsanaModelKey = KsanaPipelineKey`），即使部分成员同名。
3. **DiffusionModel 的 `KsanaModelKey` 成员与 `KsanaPipelineKey` 同名** — 因为不同 pipeline 的 diffusion model 权重不同，需要独立的 key。
4. **`get_model_key_from_path()` 统一返回 `KsanaModelKey`** — 调用方如需 `KsanaPipelineKey`，必须自行通过 `KsanaPipelineKey[model_key.name]` 转换。

### 成员分类

```python
class KsanaModelKey(Enum):
    # Text Encoders
    T5TextEncoder = auto()
    Qwen2VLTextEncoder = auto()
    Qwen2VLTextEncoderMultimodal = auto()

    # VAE
    QwenImageVAE = auto()
    VAE_WAN2_1 = auto()
    VAE_WAN2_2 = auto()

    # Diffusion Models — 与 KsanaPipelineKey 同名
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
# key 侧: KsanaPipelineKey（输入）
# value 侧: KsanaModelKey（输出）
_TEXT_ENCODER_MAP = {
    KsanaPipelineKey.Wan2_2_T2V_14B: KsanaModelKey.T5TextEncoder,
    ...
}
```

### 禁止事项

- ❌ 不要创建 `KsanaModelKey = KsanaPipelineKey` 这样的别名
- ❌ 不要让 `KsanaModelPool` 接受 `KsanaPipelineKey`
- ❌ 不要在 `KsanaModelKey` 中添加 pipeline 级别的概念
- ❌ 不要创建未被任何代码使用的 Key 成员（如曾经的 `WanDiT_14B`）

## 4. V5 Node / Tensor API 规范

### Node.run() 返回值

| Node 类型 | 返回值 | 说明 |
|-----------|--------|------|
| `KsanaLoadNode.run()` | `None` | 模型写入 model_pool，不返回 |
| `KsanaInferNode.run()` | `None` | 结果写入 tensor_pool，不返回 tensor |

**禁止** `KsanaInferNode.run()` 返回 dict 或 tensor。所有中间结果通过 `tensor_pool.put(key, tensor)` 写入。

### output_tensor_keys 静态声明

`output_tensor_keys` 保留为**类属性**，Executor 用它做 `RANK_0_BROADCAST`。
对于条件性输出（如 VAEEncodeNode 可能不写入 img_latents），broadcast 逻辑对 `tensor_pool.peek(key) is None` 的 key 做 skip 容错。

```python
class GeneratorNode(KsanaInferNode):
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
| `TensorKey.INPUT_LATENT` | `"input_latent"` | Pipeline put_tensors | GeneratorNode |

### KsanaNodeContext 禁止 tensor

`KsanaNodeContext.metadata` 中**禁止**包含 `torch.Tensor`。`__post_init__` 必须递归检查 metadata dict 的 values：

```python
def __post_init__(self):
    for field_name, value in self.__dict__.items():
        if isinstance(value, torch.Tensor):
            raise TypeError(...)
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"KsanaNodeContext.{field_name}[{k!r}] is a Tensor! "
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
# 分段调用：vae_encode 保留 IMAGE_EMBEDS
with engine.tensor_scope(keep=[KsanaTensorKey.IMAGE_EMBEDS]):
    engine.put_tensors(**{KsanaTensorKey.IMAGE: image})
    engine.run_infer_node(KsanaInferNodeType.VAE_ENCODE_IMAGES, vae, context)
# scope 退出：IMAGE 被 release，IMAGE_EMBEDS 保留在 pool 中

# 最终步骤：vae_decode 不 keep，全部清理
with engine.tensor_scope():
    engine.run_infer_node(KsanaInferNodeType.VAE_DECODE, vae, context)
    video = engine.get_tensor(KsanaTensorKey.VIDEO).data  # 取裸 tensor
# scope 退出：全部 release
```

#### ComfyUI adapter 之间传递 key

ComfyUI adapter 之间传递的是 `KsanaTensorKey`（而非裸 tensor），真正的 `TensorValue` 存在 Executor 的 pool 中：

```python
# vae_encode 返回 key
def vae_encode_image(...):
    with engine.tensor_scope(keep=[KsanaTensorKey.IMAGE_EMBEDS]):
        engine.put_tensors(...)
        engine.run_infer_node(...)
    return KsanaNodeVAEEncodeOutput(samples=KsanaTensorKey.IMAGE_EMBEDS, ...)

# generate 收到 key，先检查存在性
def generate(...):
    if not engine.has_tensor(image_embeds_key):
        raise RuntimeError(f"Tensor {image_embeds_key} not found in pool")
    with engine.tensor_scope(keep=[KsanaTensorKey.LATENTS]):
        engine.run_infer_node(...)
    return KsanaNodeGeneratorOutput(samples=KsanaTensorKey.LATENTS, ...)
```

#### Pool 的 clear(exclude=[...])

`KsanaTensorStorePool.clear(exclude)` 释放除 exclude 列表外的所有 tensor：

```python
def clear(self, exclude: list[KsanaTensorKey] | None = None) -> None:
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
- **禁止**在 ComfyUI adapter 之间传递裸 `torch.Tensor`，必须传递 `KsanaTensorKey`

---

## 5. Ownership 与状态关系图

### 整体 Ownership 层级

```
KsanaEngine (singleton via get_default / 或多实例)
 ├── owns: executors
 │    ├── 单卡模式: 1 个 KsanaExecutor 实例
 │    └── 多卡模式: N 个 RayKsanaExecutor (Ray Actor)
 ├── owns: num_gpus, _is_ray, _cleaned_up (引擎级元数据)
 ├── NOT own: model_pool, tensor_pool, device 信息 (这些属于 Executor)
 └── NOT own: 任何 Node 实例 (Node 由 Factory 按需创建，用完即弃)

KsanaExecutor (每卡一个实例)
 ├── owns: model_pool        — KsanaModelPool (存储已加载的模型)
 ├── owns: tensor_pool       — KsanaTensorStorePool (存储推理中间 tensor)
 ├── owns: dist_group        — DistributedGroupManager (管理 torch.distributed)
 ├── owns: device_ctx        — KsanaDeviceContext (frozen dataclass, 只读)
 ├── owns: device / offload_device / device_id (设备信息)
 ├── owns: rank_id / world_size (分布式信息)
 ├── owns: dist_config / shard_fn (分布式配置)
 ├── owns: local_pipeline (遗留 V4 接口，将废弃)
 └── NOT own: Node 实例 (Node 在 run_loader_node / run_infer_node 中临时创建)
```

### 各组件详细 Ownership

#### KsanaEngine (`kdit/engine/engine.py`)

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
| `model_pool` | `KsanaModelPool` | 与 Executor 同生命周期 | 存储所有已加载模型，按 `KsanaModelKey` 索引 |
| `tensor_pool` | `KsanaTensorStorePool` | 每次 `inference_session()` 结束时 clear | 存储推理中间 tensor，按 string key 索引 |
| `dist_group` | `DistributedGroupManager` | 与 Executor 同生命周期 | 管理 broadcast 等分布式操作 |
| `device_ctx` | `KsanaDeviceContext` | 初始化后不变（frozen） | 只读设备上下文，传入 Node.run() |
| `device` | `torch.device` | 不变 | 计算设备 (如 `cuda:0`) |
| `offload_device` | `torch.device` | 不变 | 卸载设备 (如 `cpu`) |
| `dist_config` | `KsanaDistributedConfig` | `init_torch_dist_group()` 后更新 | 分布式配置 |
| `shard_fn` | `partial` 或 `None` | `init_torch_dist_group()` 后设置 | FSDP 分片函数 |

#### KsanaDeviceContext (`kdit/nodes/core/device_context.py`)

```python
@dataclass(frozen=True)  # ← frozen! Node 无法篡改
class KsanaDeviceContext:
    device: torch.device        # 计算设备
    offload_device: torch.device # 卸载设备
    rank_id: int                # 当前 rank
    world_size: int             # 总 rank 数
```

- **由 Executor 创建**，在 `__init__` 和 `init_torch_dist_group()` 中构建
- **传入 Node.run()** 作为只读参数
- **frozen=True** 保证 Node 无法修改设备配置

#### KsanaTensorStorePool (`kdit/tensor/tensor_store_pool.py`)

- **Owner**: Executor
- **生命周期**: 每次 `engine.tensor_scope()` 退出（depth→0）时 `clear(exclude=keep)`
- **内容**: `dict[KsanaTensorKey, TensorValue]`，每个 TensorValue 持有 `Tensor | list[Tensor]`
- **用途**: Node 间通过 `KsanaTensorKey` 引用 tensor，避免 tensor 跨 Ray 边界序列化
- **关键方法**: `put` / `get`（返回 TensorValue）/ `clear(exclude)` / `has` / `keys` / `__len__`

#### KsanaModelPool (`kdit/models/model_pool.py`)

- **Owner**: Executor
- **生命周期**: 与 Executor 同生命周期，`clear_models()` 可手动清理
- **内容**: `dict[KsanaModelKey, KsanaModel]`
- **用途**: LoaderNode 写入模型，InferNode 读取模型

#### DistributedGroupManager (`kdit/executor/distributed_group.py`)

- **Owner**: Executor
- **状态**: `rank_id`, `world_size`, `_initialized`
- **用途**: 提供 `broadcast_tensors()` 能力，配合 tensor_pool 实现跨 rank 数据同步

### Node 状态分析

#### LoaderNode (`KsanaLoadNode` 子类)

| Node | 类变量状态 | 实例变量状态 | 说明 |
|------|-----------|-------------|------|
| `DiffusionLoaderNode` | `_pinned_memory_manager: PinnedMemoryManager` (类变量) | 无 | **有类级状态**：`_pinned_memory_manager` 是跨调用共享的单例，首次 `run()` 时惰性初始化 |
| `TextEncoderLoaderNode` | 无 | 无 | **无状态** |
| `VAELoaderNode` | 无 | 无 | **无状态** |

**关键问题**：`DiffusionLoaderNode._pinned_memory_manager` 是类变量，在 `NodeFactory.create()` 每次创建新实例时不会重置。这意味着：
- 同一进程内所有 `DiffusionLoaderNode` 实例共享同一个 `PinnedMemoryManager`
- 这是有意设计（共享 pinned memory 池），但违反了"Node 无状态"的理想模型

#### InferNode (`KsanaInferNode` 子类)

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

Node 由 Factory **按需创建**，每次 `run_loader_node()` / `run_infer_node()` 调用时创建新实例：

```python
# executor.py
def run_loader_node(self, model_key, **kwargs):
    node = KsanaLoaderNodeFactory.create(model_key)  # ← 每次新建
    node.run(model_key, model_pool=self.model_pool, device_ctx=self.device_ctx, **kwargs)

def run_infer_node(self, infer_node_type, model_key, context):
    node = KsanaInferNodeFactory.create(infer_node_type, model_key)  # ← 每次新建
    self._pre_sync_tensors(node, policy)
    is_active_rank = policy == KsanaDispatchPolicy.ALL_ALL_ALL or self.device_ctx.rank_id == 0
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
| DeviceContext 只读 | `frozen=True` dataclass，Node 无法篡改 |
| NodeContext 无 tensor | `__post_init__` 强制校验不含 `torch.Tensor`，保证可跨 Ray 序列化 |
| tensor_pool 生命周期 | 由 `engine.inference_session()` 管理，session 结束自动 clear |
| model_pool 生命周期 | 与 Executor 同生命周期，需手动 `clear_models()` 释放 |
| InferNode.run() 签名固定 | 禁止添加 `**kwargs` 或额外参数，额外配置通过 `context.metadata` 传递 |
| Tensor 只能通过 tensor_pool 流转 | 禁止在 run() 参数或 context.metadata 中传递 tensor |

---

## 6. KsanaInferNode 开发规范

### run() 签名固定，禁止扩展

`KsanaInferNode.run()` 签名是固定的：

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
- **禁止** 在 `context.metadata` 中放 tensor（`KsanaNodeContext.__post_init__` 会校验）

### 声明 tensor 契约

每个 Node 必须声明 `input_tensor_keys` 和 `output_tensor_keys`：

```python
class MyNode(KsanaInferNode):
    input_tensor_keys = [TensorKey.POSITIVE, TensorKey.NEGATIVE]
    output_tensor_keys = [TensorKey.LATENTS]
```

- `input_tensor_keys` 可以为空列表 `[]`（如 TextEncodeNode 不依赖其他 tensor）
- `output_tensor_keys` 用于 `R0_R0_BCAST` 策略时指定需要 broadcast 的 key

### dispatch_policy 三维度命名

`KsanaDispatchPolicy` 使用 `input_exec_output` 三维度拼接命名：

| Policy | 输入要求 | 执行范围 | 输出行为 | 典型场景 |
|--------|---------|---------|---------|---------|
| `ALL_ALL_ALL` | 所有卡都有 | 所有卡 | 各卡独立持有 | TextEncode, Generator |
| `R0_R0_BCAST` | rank0 有即可 | 仅 rank0 | broadcast 到所有卡 | VAEEncode |
| `ALL_R0_R0` | 所有卡都有 | 仅 rank0 | 仅 rank0 持有 | VAEDecode |

### Node 注册

- 使用 `@KsanaInferNodeFactory.register()` 装饰器注册
- 注册键为 `(KsanaInferNodeType, [KsanaModelKey, ...])`
- `KsanaInferNodeType` 枚举值：`TEXT_ENCODE`, `VAE_ENCODE_SPATIAL`, `VAE_ENCODE_IMAGES`, `VAE_DECODE`, `GENERATE`

### 现有 Node 参考

| Node | dispatch_policy | input_tensor_keys | output_tensor_keys |
|------|----------------|-------------------|-------------------|
| `TextEncodeNode` | `ALL_ALL_ALL` | `[]` | `[POSITIVE, NEGATIVE]` |
| `VAEEncodeSpatialNode` | `R0_R0_BCAST` | `[START_IMG, END_IMG]` | `[IMG_LATENTS]` |
| `VAEEncodeImagesNode` | `R0_R0_BCAST` | `[IMAGE]` | `[IMG_LATENTS]` |
| `VAEDecodeNode` | `ALL_R0_R0` | `[LATENTS]` | `[VIDEO]` |
| `GeneratorNode` | `ALL_ALL_ALL` | `[POSITIVE, NEGATIVE, IMG_LATENTS, INPUT_LATENT]` | `[LATENTS]` |

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
| `KsanaRuntimeConfig` | `RuntimeConfig` | `kdit/config/runtime_config.py` |
| `KsanaSolverType` | `SolverType` | `kdit/config/sample_config.py` |
| `KsanaSampleConfig` | `SampleConfig` | `kdit/config/sample_config.py` |
| `KsanaModelConfig` | `ModelConfig` | `kdit/config/model_config.py` |
| `KsanaCacheConfig` | `CacheConfig` | `kdit/config/cache_config/base.py` |
| `KsanaBlockCacheConfig` | `BlockCacheConfig` | `kdit/config/cache_config/base.py` |
| `KsanaStepCacheConfig` | `StepCacheConfig` | `kdit/config/cache_config/base.py` |
| `KsanaHybridCacheConfig` | `HybridCacheConfig` | `kdit/config/cache_config/base.py` |
| `KsanaVideoControlConfig` | `VideoControlConfig` | `kdit/config/video_control_config.py` |
| `KsanaLoraConfig` | `LoraConfig` | `kdit/config/lora_config.py` |
| `KsanaAttentionBackend` | `AttentionBackend` | `kdit/config/attention_config.py` |
| `KsanaAttentionConfig` | `AttentionConfig` | `kdit/config/attention_config.py` |
| `KsanaRadialSageAttentionConfig` | `RadialSageAttentionConfig` | `kdit/config/attention_config.py` |
| `KsanaSageSLAConfig` | `SageSLAConfig` | `kdit/config/attention_config.py` |
| `KsanaTorchCompileConfig` | `TorchCompileConfig` | `kdit/config/torch_compile_config.py` |
| `KsanaSLGConfig` | `SLGConfig` | `kdit/config/wan_experimental_config.py` |
| `KsanaFETAConfig` | `FETAConfig` | `kdit/config/wan_experimental_config.py` |
| `KsanaExperimentalConfig` | `ExperimentalConfig` | `kdit/config/wan_experimental_config.py` |
| `KsanaDistributedConfig` | `DistributedConfig` | `kdit/config/distributed_config.py` |
| `KsanaLoaderNodeFactory` | `LoaderNodeFactory` | `kdit/nodes/core/node_factory.py` |
| `KsanaInferNodeFactory` | `InferNodeFactory` | `kdit/nodes/core/node_factory.py` |
| `KsanaNodeContext` | `NodeContext` | `kdit/nodes/core/node_context.py` |
| `KsanaLoadNode` | `LoadNode` | `kdit/nodes/core/base_node.py` |
| `KsanaInferNode` | `InferNode` | `kdit/nodes/core/base_node.py` |
| `KsanaDispatchPolicy` | `DispatchPolicy` | `kdit/nodes/core/node_types.py` |
| `KsanaInferNodeType` | `InferNodeType` | `kdit/nodes/core/node_types.py` |
| `KsanaDeviceContext` | `DeviceContext` | `kdit/nodes/core/device_context.py` |
| `KsanaBatchScheduler` | `BatchScheduler` | `kdit/scheduler/scheduler.py` |
| `KsanaPipeline` | `Pipeline` | `kdit/pipelines/x2x_pipeline.py` |
| `KsanaBasePipeline` | `BasePipeline` | `kdit/pipelines/base_pipeline.py` |
| `KsanaPipelineKey` | `PipelineKey` | `kdit/pipelines/pipeline_key.py` |
| `KsanaProfiler` | `Profiler` | `kdit/utils/profile.py` |
| `KsanaVaceContext` | `VaceContext` | `kdit/utils/vace.py` |
| `KsanaModelKey` | `ModelKey` | `kdit/models/model_key.py` |
| `KsanaQwenImageVAE` | `QwenImageVAE` | `kdit/models/qwen/vae.py` |
| `KsanaDiffusionModel` | `DiffusionModel` | `kdit/models/diffusion_model.py` |
| `KsanaWanModel` | `WanModel` | `kdit/models/diffusion_model.py` |
| `KsanaWanVaceModel` | `WanVaceModel` | `kdit/models/diffusion_model.py` |
| `KsanaQwenImageModel` | `QwenImageModel` | `kdit/models/diffusion_model.py` |
| `KsanaVAEModel` | `VAEModel` | `kdit/models/vae_model.py` |
| `KsanaWanVAEModel` | `WanVAEModel` | `kdit/models/vae_model.py` |
| `KsanaQwenVAEModel` | `QwenVAEModel` | `kdit/models/vae_model.py` |
| `KsanaModelPool` | `ModelPool` | `kdit/models/model_pool.py` |
| `KsanaTextEncoderModel` | `TextEncoderModel` | `kdit/models/text_encoder_model.py` |
| `KsanaAttentionOp` | `AttentionOp` | `kdit/operations/attention/attention_op.py` |
| `KsanaAttentionBackendImpl` | `AttentionBackendImpl` | `kdit/operations/attention/backends/base.py` |
| ~~`KsanaRunnerUnit`~~ | ~~`RunnerUnit`~~ | ~~`kdit/units/runner_unit.py`~~ | ✅ 已删除（冗余） |
| ~~`KsanaBaseTextEncoder`~~ | ~~`BaseTextEncoder`~~ | ~~`kdit/units/text_encoder.py`~~ | ✅ 已删除（融入 `text_encoder_node.py`） |
| ~~`KsanaTextEncoder`~~ | ~~`T5TextEncoder`~~ | ~~`kdit/units/text_encoder.py`~~ | ✅ 已删除（融入 `T5TextEncodeNode`） |
| ~~`KsanaQwenVLTextEncoderUnit`~~ | ~~`QwenVLTextEncoder`~~ | ~~`kdit/units/text_encoder.py`~~ | ✅ 已删除（融入 `QwenTextEncodeNode`） |
| `KsanaWanGenerator` | `WanGenerator` | `kdit/generators/wan_generator.py` |
| `KsanaQwenGenerator` | `QwenGenerator` | `kdit/generators/qwen_generator.py` |
| `KsanaBaseGenerator` | `BaseGenerator` | `kdit/generators/base_generator.py` |
| `KsanaVaceGenerator` | `VaceGenerator` | `kdit/generators/vace_generator.py` |
| ~~`KsanaVaeEncoder`~~ | ~~`VaeEncoder`~~ | ~~`kdit/units/vae_encoder_unit.py`~~ | ✅ 已删除（VAE Unit 移除） |
| ~~`KsanaUnit`~~ | ~~`Unit`~~ | ~~`kdit/units/base_unit.py`~~ | ✅ 已删除（units 包移除） |
| ~~`KsanaUnitType`~~ | ~~`UnitType`~~ | ~~`kdit/units/base_unit.py`~~ | ✅ 已删除（units 包移除） |
| ~~`KsanaUnitFactory`~~ | ~~`UnitFactory`~~ | ~~`kdit/units/base_unit.py`~~ | ✅ 已删除（替换为 `GeneratorFactory`） |
| ~~`KsanaVaeDecoder`~~ | ~~`VaeDecoder`~~ | ~~`kdit/units/decoder_unit.py`~~ | ✅ 已删除（VAE Unit 移除） |
| `KsanaExecutor` | `Executor` | `kdit/executor/executor.py` |
| ~~`KsanaTensorStore`~~ | `TensorValue` | `kdit/tensor/tensor_value.py` | ✅ 已完成 |
| `KsanaTensorKey` | `TensorKey` | `kdit/tensor/tensor_key.py` |
| `KsanaTensorStorePool` | `TensorStorePool` | `kdit/tensor/tensor_store_pool.py` |
| `KsanaEngine` | `Engine` | `kdit/engine/engine.py` |

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
