# kDiT 架构核心概念

## 核心组件总览

```
Pipeline（编排层）→ Engine（分发层）→ Executor（执行层）→ Node（计算单元）
                                                          ↕
                                              PinHub（数据访问沙箱）
                                                          ↕
                                          TensorPool / ModelPool（数据存储）
```

---

## Node — 计算单元

分两类：**IONode**（加载模型）和 **InferNode**（前向推理）。

### 设计原则

- Node 的入参和出参**不直接传输 tensor 和 model**，这两者都通过 Pool 存储，通过 PoolKey 引用。避免多卡 Ray 场景下耗时的序列化操作。
- 其他入参只能包含简单的 config 内容（通过 `NodeContext`），不允许存在 tensor 或 model 直接传输。
- 每个 Node 实例有唯一的 `node_id`（int），由 `NodeDef` 创建时通过模块级全局计数器（`itertools.count(1)`）自动分配，用户不感知。
- DAG 中未连接的输入 pin 代表"不输入"，Node 收到 `None`。Node **必须**自行处理 `None` 输入。

### Pin 声明

每个 Node 类通过类属性声明自己的输入输出端口（pin）：

```python
class SomeInferNode(InferNode):
    input_model_pins = [ModelKey.T5TextEncoder]           # 从 ModelPool 读的 model
    input_tensor_pins = [TensorKey.POSITIVE, TensorKey.NEGATIVE]  # 从 TensorPool 读的 tensor
    output_tensor_pins = [TensorKey.LATENTS]              # 写入 TensorPool 的 tensor
```

- Pin 用 `TensorKey` / `ModelKey` 枚举声明
- `IONode` 的 `output_model_pins` 由 Factory 注册时自动填充，不需要手动声明
- Pin 声明用于 build 时校验（悬空检测）和运行时 PinHub 沙箱约束

### run() 签名

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

### 注意事项

- `run()` 签名固定，**禁止**添加额外参数或 `**kwargs`
- 额外配置（包括 IONode 的加载参数）通过 `context.metadata` 传递
- tensor 只能通过 `pins.get_tensor()` / `pins.put_tensor()` 流转，**禁止**在参数或 metadata 中传递 tensor
- `NodeContext.__post_init__` 递归校验 metadata 不含 `torch.Tensor`

---

## PinHub — 沙箱化数据访问器

**职责**：Node 运行时的数据读写路由层。每个 Node 实例拥有独立的 PinHub，被严格约束在 DAG 声明的范围内。

### 核心机制

- **读操作**：根据 `pins_mapping`（DAG 连线计算的映射），将上游 Node 的输出 PoolKey 映射到当前 Node 的 input pin
- **写操作**：自动用 `当前 node_id + pin` 生成 PoolKey 写入 Pool

```python
pins.get_tensor(TensorKey.POSITIVE)   # 从 pins_mapping 查找上游的 TensorPoolKey，读取
pins.put_tensor(TensorKey.LATENTS, data)  # 写入 TensorPoolKey(self.node_id, TensorKey.LATENTS)
pins.get_model(ModelKey.T5TextEncoder)    # 从 pins_mapping 查找上游的 ModelPoolKey，读取
pins.put_model(ModelKey.VAE_WAN2_1, model)  # 写入 ModelPoolKey(self.node_id, ModelKey.VAE_WAN2_1)
```

### 沙箱约束

- **读**：只能读 `pins_mapping` 中存在的 key（即 DAG 连线声明的上游输出），读不到其他 Node 的数据
- **写**：只能写 `PoolKey(self.node_id, pin)`，即自己 node_id 命名空间下的 key，写不到别人的命名空间
- 未连线的 optional tensor pin 返回 `None`，未连线的 required model pin 抛出 `KeyError`

### 构建位置

PinHub 在 **Executor 内部构建**（不在 Pipeline 层），因为：
- `tensor_pool` 和 `model_pool` 活在 Executor 上（每卡一份）
- 多卡下每个 Executor 各自构建自己的 PinHub，天然正确
- PinHub 不需要跨进程序列化

### 注意事项

- `pins_mapping` 是纯数据（`dict[TensorKey, TensorPoolKey]` + `dict[ModelKey, ModelPoolKey]`），由 Pipeline 层从 DAG edges 计算，通过 Engine 分发到 Executor
- PinHub 不持有 `DeviceInfo`，设备信息通过 `context.device` 获取

---

## PoolKey — 实例化的存储 key

### ModelPoolKey / TensorPoolKey

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

### 为什么需要 PoolKey

解决"同一个 Pipeline 中不能有两个相同类型的 Node 实例"的问题。例如两个 `VAE_ENCODE_SPATIAL` 节点，各自写入 `TensorPoolKey(node_id=4, TensorKey.BASE_LATENT)` 和 `TensorPoolKey(node_id=5, TensorKey.BASE_LATENT)`，互不冲突。

---

## DeviceInfo — 设备信息

```python
@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device        # 计算设备
    offload_device: torch.device  # 卸载设备
    rank_id: int                # 当前卡的 rank
    world_size: int             # 总卡数
```

### 关键规则

- DeviceInfo 嵌入 `NodeContext.device` 字段
- **由 Executor 自动注入，禁止用户手动设置**
- Pipeline 层构建 NodeContext 时 `device` 字段为 `None`
- Executor 在调用 Node 前自动注入：`context.device = self.device_info`
- `NodeContext.__post_init__` 校验：如果 `device` 不为 None 则报错，防止用户误填
- Node 中通过 `context.device.device`、`context.device.offload_device` 等访问

---

## NodeContext — Node 间传递的上下文

**职责**：携带配置、元数据和设备信息，可安全跨 Ray 边界序列化。

```python
@dataclass
class NodeContext:
    prompt: str | list[str] = None
    negative_prompt: str | list[str] = None
    img_path: str | list[str] | list[list[str]] = None
    sample_config: SampleConfig = None
    runtime_config: RuntimeConfig = None
    cache_config: list = None
    metadata: dict = field(default_factory=dict)
    device: DeviceInfo = None  # Executor 自动注入，禁止手动设置
```

### 注意事项

- **不含任何 tensor**（`__post_init__` 强制校验）
- **device 字段禁止手动设置**（`__post_init__` 校验，Executor 自动注入）
- 可安全跨 Ray 边界序列化
- 额外配置通过 `metadata` dict 传递

---

## NodeRef / PinRef — DAG 连线引用

- `NodeRef`：`add_loader()` / `add_infer()` 返回的 Node 引用，支持属性访问生成 PinRef
- `PinRef`：`(node_id, pin)` 二元组，用于 `connect()` 声明连线

```python
vae_a = builder.add_infer(InferNodeType.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_1)
gen   = builder.add_infer(InferNodeType.GENERATE, ModelKey.Wan2_2_I2V_14B)

# vae_a.BASE_LATENT → PinRef(node_id, TensorKey.BASE_LATENT)
builder.connect((vae_a.BASE_LATENT, gen.BASE_LATENT))
```

- `NodeRef.__getattr__` 在 TensorKey 和 ModelKey 枚举中查找属性名，不需要引号
- 只有相同类型（都是 TensorKey 或都是 ModelKey）的 pin 才能 connect，但不要求同名

---

## Pipeline — 编排层

**职责**：DAG 遍历 + 计算 `pins_mapping` + 分发执行。

- `PipelineDef` 是不可变的 DAG 定义（frozen dataclass），包含 `nodes` + `edges`
- `PipelineDefBuilder` 链式构建，通过 `.add_loader()` / `.add_infer()` / `.connect()` 声明
- Pipeline 层负责 DAG 拓扑排序、条件检查、构建 NodeContext
- Pipeline 层计算 `pins_mapping`（纯数据），传给 Engine → Executor
- Executor 只执行单个 Node，不感知 DAG
- `Pipeline.generate()` 接收 `extra_inputs: ExtraInputs | None` 传递模型特有输入，**禁止** `**kwargs`
- `ContextBuilder` 是 Pipeline 和 Node 之间的桥梁，负责 `prepare_generate_inputs()` + `build_context()`

详见 [`.skills/coding/pipeline-and-context.md`](pipeline-and-context.md)

---

## Adapter / ComfyUI

adapter 为了让 kdit 的 Node 可以适配多种其他工具的一个适配层。目前只有 ComfyUI，未来或许还有别的。

本质上，adapter 只需要一点点的适配层，就可以直接调用 kdit 的 Node 达到实现适配的目的。

语义上，ComfyUI 的 workflow 是一个 JSON 文件，包含了 Node 的组合以及 Node 之间的连接关系，这个理论上就应该可以通过 PipelineDef 描述对等后在本地运行。本质 ComfyUI 的 workflow 应该和 Pipeline 在一个层级。

**依赖方向**：`kdit/adapter/comfyui/` → `kdit/` ✅；`kdit/` → `kdit/adapter/` ❌ 禁止

---

## Generator

Generator 是 Node 内部的实现细节（被 GeneratorNode 封装），负责 Diffusion 去噪流程。Generator 内部的 tensor 流转（noise、denoise step 等）不受 DAG 改造影响。

详见 [`.skills/coding/generator.md`](.skills/coding/generator.md)
