# Key 类型体系设计规范

---

## 四种 Key 的职责

| Key 类型 | 定义位置 | 语义 | 使用场景 |
|----------|---------|------|---------|
| `ModelKey` | [`kdit/models/model_key.py`](../../kdit/models/model_key.py:49) | 标识一个具体的模型类别 | `ModelPool` 存取、`ModelBase.__init__`、Loader/Infer Node 注册与分发、`GeneratorDef` 注册、`settings` 配置映射 |
| `PipelineKey` | [`kdit/pipelines/pipeline_key.py`](../../kdit/pipelines/pipeline_key.py) | 标识一条完整的推理流水线 | `Pipeline.__init__`、pipeline 创建与路由、pipeline→model 映射表的 key 侧 |
| `InferNodeType` | [`kdit/nodes/core/node_types.py`](../../kdit/nodes/core/node_types.py:55) | 标识推理节点类型 | `InferNodeFactory` 注册与分发、`PipelineDefBuilder.add_infer()` |
| `IONodeType` | [`kdit/nodes/core/node_types.py`](../../kdit/nodes/core/node_types.py:41) | 标识加载节点类型 | `IONodeFactory` 注册与分发、`PipelineDefBuilder.add_loader()` |

## 核心约束

1. **`ModelPool` 只接受 `ModelKey` 创建的 `ModelPoolKey`** — 不允许传入 `PipelineKey`创建的 `ModelPoolKey` 或其他类型。
2. **`ModelKey` 和 `PipelineKey` 是独立枚举** — 不存在别名关系（如 `ModelKey = PipelineKey`），即使部分成员同名。
3. **DiffusionModel 的 `ModelKey` 成员与 `PipelineKey` 同名** — 因为不同 pipeline 的 diffusion model 权重不同，需要独立的 key。
4. **`get_model_key_from_path()` 统一返回 `ModelKey`** — 调用方如需 `PipelineKey`，必须自行通过 `PipelineKey[model_key.name]` 转换。

## 成员分类

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

## Pipeline → Model 映射方向

`ContextBuilder` 中的映射逻辑遵循 **pipeline key → model key** 方向。`model_key.py` 中定义了 `TEXT_ENCODER_KEYS`、`DIFFUSION_KEYS`、`VAE_KEYS` 等分类集合，用于按 `ModelKey` 类别进行分组和查找。

调用方通过 `PipelineKey[model_key.name]` 进行 `ModelKey` ↔ `PipelineKey` 转换。

## 禁止事项

- ❌ 不要创建 `ModelKey = PipelineKey` 这样的别名
- ❌ 不要让 `ModelPool` 接受 `PipelineKey`
- ❌ 不要在 `ModelKey` 中添加 pipeline 级别的概念
- ❌ 不要创建未被任何代码使用的 Key 成员（如曾经的 `WanDiT_14B`）
