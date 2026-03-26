# Node — 计算单元

分两类：**IONode**（加载模型）和 **InferNode**（前向推理）。

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

## 设计原则

- Node 的入参和出参**不直接传输 tensor 和 model**，这两者都通过 Pool 存储，通过 PoolKey 引用。避免多卡 Ray 场景下耗时的序列化操作。
- 其他入参只能包含简单的 config 内容（通过 `NodeContext`），不允许存在 tensor 或 model 直接传输。
- 每个 Node 实例有唯一的 `node_id`（int），由模块级全局计数器（`itertools.count(1)`）自动分配，用户不感知。
- DAG 中未连接的输入 pin 代表"不输入"，Node 收到 `None`。Node **必须**自行处理 `None` 输入。

---

## Pin 声明

每个 Node 类通过类属性声明自己的输入输出端口（pin）：

```python
class SomeInferNode(InferNode):
    input_defs = [TensorKey.POSITIVE, TensorKey.NEGATIVE]  # 从 TensorPool 读的 tensor
    output_defs = [TensorKey.LATENTS]                       # 写入 TensorPool 的 tensor
```

- Pin 用 `TensorKey` / `ModelKey` 枚举声明
- `IONode` 的 model 输出由 Factory 注册时自动填充，不需要手动声明
- Pin 声明用于 build 时校验（悬空检测）和运行时 PinHub 沙箱约束

### output_defs 静态声明

`output_defs` 保留为**类属性**（`list[PinDef]`）

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

---

## run() 签名

```python
# InferNode
def run(self, pins: PinHub, *, context: NodeContext) -> None:

# IONode — 签名与 InferNode 完全一致
def run(self, pins: PinHub, *, context: NodeContext) -> None:
```

- `pins` 是位置参数（必须），是 Node 读写数据的唯一通道
- `context` 是 keyword-only，包含配置、元数据和设备信息
- Node 内部**禁止**直接访问 TensorPool / ModelPool / DeviceInfo，全部通过 `pins` 和 `context` 获取
- IONode 的加载参数（model_path、model_config、lora_config 等）统一放入 `context.metadata`

### run() 返回值

| Node 类型 | 返回值 | 说明 |
|-----------|--------|------|
| `IONode.run()` | `None` | 模型写入 model_pool，不返回 |
| `InferNode.run()` | `None` | 结果写入 tensor_pool，不返回 tensor |

**禁止** `InferNode.run()` 返回 dict 或 tensor。所有中间结果通过 `tensor_pool.put(key, tensor)` 写入。
> **非常重要** : 待重构决定是否需要返回`output_pins`, 待更新

### run() 写入规则

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

## 注意事项

- `run()` 签名固定，**禁止**添加额外参数或 `**kwargs`
- 额外配置（包括 IONode 的加载参数）通过 `context.metadata` 传递
- tensor 只能通过 `pins.get_tensor()` / `pins.put_tensor()` 流转，**禁止**在参数或 metadata 中传递 tensor
- `NodeContext.__post_init__` 递归校验 metadata 不含 `torch.Tensor`

---

## dispatch_policy 三维度命名

`NodeDispatchPolicy` 使用 `input_exec_output` 三维度拼接命名：

| Policy | 输入要求 | 执行范围 | 输出行为 | 典型场景 |
|--------|---------|---------|---------|---------|
| `ALL_ALL_ALL` | 所有卡都有 | 所有卡 | 各卡独立持有 | TextEncode, Generator |
| `R0_R0_BCAST` | rank0 有即可 | 仅 rank0 | broadcast 到所有卡 | VAEEncode |
| `ALL_R0_R0` | 所有卡都有 | 仅 rank0 | 仅 rank0 持有 | VAEDecode |

---

## Node 间数据传递

- Node 间通过 **同一个 Executor 内的 tensor_pool** 传递数据，不经过 Engine
- 跨 rank 数据传递由 `DispatchPolicy.RANK_0_BROADCAST` 的 broadcast 机制自动处理
- `engine.get_tensor()` 只用于 Pipeline/ComfyUI 取回最终结果，不用于 Node 间传递
- Pipeline/ComfyUI 向 Node 传递 tensor 输入时，必须通过 `engine.put_tensors()` 写入 tensor_pool
- **禁止**在 ComfyUI adapter 之间传递裸 `torch.Tensor`，必须传递 `TensorKey`

---

## Node 状态分析

### IONode (`IONode` 子类)

| Node | 类变量状态 | 实例变量状态 | 说明 |
|------|-----------|-------------|------|
| `DiffusionLoaderNode` | `_pinned_memory_manager: PinnedMemoryManager` (类变量) | 无 | **有类级状态**：`_pinned_memory_manager` 是跨调用共享的单例，首次 `run()` 时惰性初始化 |
| `TextEncoderLoaderNode` | 无 | 无 | **无状态** |
| `VAELoaderNode` | 无 | 无 | **无状态** |

**关键问题**：`DiffusionLoaderNode._pinned_memory_manager` 是类变量，在 `NodeFactory.create()` 每次创建新实例时不会重置。这意味着：
- 同一进程内所有 `DiffusionLoaderNode` 实例共享同一个 `PinnedMemoryManager`
- 这是有意设计（共享 pinned memory 池），但违反了"Node 无状态"的理想模型

### InferNode (`InferNode` 子类)

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

---

## Node 创建方式

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

def run_node(self, node_def, input_pins, context):
    node = self._get_or_create_node(node_def)
    pin_hub = self._build_pin_hub(node_def, input_pins)
    self._pre_sync_tensors(node, policy)
    if is_active_rank:
        node.run(pin_hub, context=context)
    self._post_sync_tensors(node, node_def, policy)
    self._consume_input_tensors(input_pins)  # 自动消费输入 tensor
```

Node 实例按 `node_id` 缓存在 Executor 中，同一 node_id 复用同一实例。

---

## Node 注册

- 使用 `@InferNodeFactory.register()` 装饰器注册
- 注册键为 `(InferNodeType, [ModelKey, ...])`
- `InferNodeType` 枚举值：`TEXT_ENCODE`, `VAE_ENCODE_SPATIAL`, `VAE_ENCODE_IMAGES`, `VAE_COMPUTE_SHAPE`, `VAE_DECODE`, `GENERATE`, `SAVE_VIDEO`, `SAVE_IMAGE`, `READ_IMAGE`, `VACE_PREPROCESS`

---

## 现有 Node 参考

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

---

## 相关文档

- 编码实操规范（InferNode 开发 checklist） → [`03_standards/node-and-tensor.md`](../03_standards/node-and-tensor.md)
- PinHub 沙箱机制 → [`pin-hub.md`](pin-hub.md)
- NodeContext 详情 → [`node-context.md`](node-context.md)
- TensorPool / ModelPool → [`pool-key.md`](pool-key.md)
- Key 类型体系 → [`03_standards/key-system.md`](../03_standards/key-system.md)
