# Pipeline 接口、ExtraInputs 与 ContextBuilder 规范

> 本文件定义 Pipeline.generate() 接口规范、模型特有输入管理（ExtraInputs）、
> ContextBuilder 开发约束。

---

## 1. Pipeline.generate() 接口规范

### 签名

```python
def generate(
    self,
    prompt: str | list[str],
    *,
    prompt_negative: str | list[str] | None = None,
    sample_config: SampleConfig = None,
    runtime_config: RuntimeConfig = None,
    cache_config: list[CacheConfig | HybridCacheConfig] | None = None,
    extra_inputs: ExtraInputs | None = None,
):
```

### 规则

- **禁止** `**kwargs` — `generate()` 签名中不允许出现 `**kwargs`
- **禁止**在 `generate()` 签名中添加模型特有的具名参数（如 `start_img_path=`）
- 模型特有输入**必须**通过 `extra_inputs` 参数传入
- 无特有输入的模型（如 T2V、T2I）不需要传 `extra_inputs`

### 公共参数 vs 模型特有参数

| 类别 | 参数 | 说明 |
|------|------|------|
| 公共 | `prompt`, `prompt_negative` | 所有模型都需要 |
| 公共 | `sample_config`, `runtime_config`, `cache_config` | 采样/运行时/缓存配置 |
| 模型特有 | `extra_inputs` | 通过 `ExtraInputs` 子类传入 |

---

## 2. ExtraInputs — 模型特有输入管理

### 基类

```python
# kdit/pipelines/extra_inputs.py
from dataclasses import dataclass

@dataclass
class ExtraInputs:
    """模型特有输入的基类。

    每个 Pipeline 定义自己的子类。
    T2V/T2I 等无特有输入的 Pipeline 不需要传此参数。
    """
    pass
```

### 子类定义位置

每个 `ExtraInputs` 子类定义在对应的 ContextBuilder 文件中：

| 子类 | 文件 | 字段 |
|------|------|------|
| `WanI2VExtraInputs` | `kdit/pipelines/context_builders/wan.py` | `start_img_path`, `end_img_path`, `aux_latent`, `video_control_config` |
| `WanVACEExtraInputs` | `kdit/pipelines/context_builders/wan.py` | `video_control_config` |
| `QwenEditExtraInputs` | `kdit/pipelines/context_builders/qwen.py` | `img_path` |

### 设计原则

1. **所有字段都有默认值**（通常为 `None`），使 `ExtraInputs()` 空构造合法
2. **字段类型明确**，IDE 可自动补全
3. **子类只包含用户传入的原始输入**（如 `start_img_path: str`），不包含内部中间数据（如 `start_img_tensor: Tensor`）
4. **内部中间数据**由 ContextBuilder 的 `_extra` 属性管理，与用户侧 `ExtraInputs` 分离

### 用户侧调用示例

```python
# WanI2V — 有模型特有输入
from kdit.pipelines.context_builders.wan import WanI2VExtraInputs

pipeline.generate(
    prompts,
    extra_inputs=WanI2VExtraInputs(
        start_img_path="path/to/img.jpg",
        end_img_path="path/to/end.jpg",
    ),
    sample_config=SampleConfig(steps=40),
    runtime_config=RuntimeConfig(seed=1234, size=(1280, 720), frame_num=81),
)

# WanT2V — 无特有输入，不传 extra_inputs
pipeline.generate(
    prompts,
    sample_config=SampleConfig(steps=40),
    runtime_config=RuntimeConfig(seed=1234, size=(1280, 720), frame_num=81),
)
```

---

## 3. ContextBuilder 开发规范

### 职责

ContextBuilder 是 Pipeline 和 Node 之间的桥梁：

1. **`prepare_generate_inputs()`** — 从 `ExtraInputs` 提取、校验、预处理模型特有输入
2. **`build_context()`** — 为每个 NodeDef 构建 `NodeContext`
3. **`check_condition()`** — 判断条件节点是否执行

### prepare_generate_inputs() 签名

```python
def prepare_generate_inputs(
    self,
    base_inputs: PipelineGenerateInputs,
    extra_inputs: ExtraInputs | None,
    *,
    default_settings: Any,
    engine: Engine,
    vae_model_key: ModelKey | None,
) -> None:
```

- `extra_inputs` 是用户传入的结构化输入
- `default_settings`、`engine`、`vae_model_key` 是内部注入的显式参数（不再通过 kwargs）
- 子类应在此方法中校验 `extra_inputs` 类型，并将处理后的中间数据存入 `self._extra`

### build_context() 签名

```python
@abstractmethod
def build_context(
    self,
    node_def: NodeDef,
    inputs: PipelineGenerateInputs,
) -> NodeContext:
```

- 参数是 `NodeDef`（不是 `InferTask`）
- 通过 `node_def.node_type` 分支构建不同 Node 的 context
- 通过 `node_def.node_id` 区分同类型 Node 的不同实例

### 多实例 Node 的 context 区分

当 DAG 中有多个同类型 Node 实例（如两个 `ReadImageNode`）时，ContextBuilder 通过 DAG edges 查找 `node_id` 的输出连接到下游的哪个 pin，从而决定传入哪个输入：

```python
def _build_read_image_ctx(self, node_def: NodeDef, inputs) -> NodeContext:
    # 查找此 ReadImage 实例的 IMAGE 输出连接到下游的哪个 pin
    dst_pin = self._find_downstream_pin(node_def.node_id, TensorKey.IMAGE)
    if dst_pin == TensorKey.START_IMG:
        img_paths = self._extra.start_img_path
    elif dst_pin == TensorKey.END_IMG:
        img_paths = self._extra.end_img_path
    return NodeContext(metadata={"img_paths": img_paths})
```

**前提**：ContextBuilder 初始化时需要接收 `pipeline_def` 以访问 edges 信息。

### 禁止事项

- **禁止** `prepare_tensors()` — 所有 tensor 注入通过 DAG Node 完成
- **禁止**在 `prepare_generate_inputs()` 中使用 `**kwargs` — 所有参数显式声明
- **禁止**在 `build_context()` 中直接操作 tensor_pool — tensor 只能通过 Node 的 PinHub 流转

---

## 4. 悬空 Pin 规则

### 规则

DAG 中未连接的输入 pin 代表"不输入"：

- Node 通过 `pins.get_tensor(key)` 读取时返回 `None`
- Node **必须**自行处理 `None` 输入
- 不需要自动补全、不需要 INPUT_NODE 概念

### 示例

```python
# GeneratorNode 已正确处理 AUX_LATENT = None
class GeneratorNode(InferNode):
    input_tensor_pins = [TensorKey.POSITIVE, TensorKey.NEGATIVE,
                         TensorKey.BASE_LATENT, TensorKey.AUX_LATENT,
                         TensorKey.VACE_CONTEXT]

    def run(self, pins, *, context):
        aux_latent_val = pins.get_tensor(TensorKey.AUX_LATENT)
        aux_latent = aux_latent_val.data if aux_latent_val is not None else None
        # aux_latent 可能为 None，后续逻辑正确处理
```

### _validate_dag() 校验

- 悬空的输入 tensor pin → **不报错**，Node 收到 None
- 悬空的输入 model pin → **报错**（model 是必需的）
- 重复的 `(dst_id, dst_pin)` → **报错**（一个输入 pin 只能有一条入边）

---

## 5. Cross-Pin Connect

### 规则

DAG 连线允许不同名的 pin 之间连接，只要它们是同一类型（都是 `TensorKey` 或都是 `ModelKey`）：

```python
# 合法 — IMAGE → START_IMG（都是 TensorKey）
read_s.IMAGE >> venc.START_IMG

# 合法 — IMAGE → END_IMG（都是 TensorKey）
read_e.IMAGE >> venc.END_IMG

# 非法 — TensorKey → ModelKey（类型不同）
read_s.IMAGE >> venc.MODEL  # TypeError
```

### 用途

Cross-pin connect 使单一功能的 Node（如 ReadImageNode 只输出 IMAGE）可以连接到不同语义的下游 pin，通过 DAG 多实例实现复用。
