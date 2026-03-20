# Adapter 依赖、类命名、Metadata 重构与 Pipeline 架构

> 本文件从 [`.skills/coding.md`](../coding.md) 拆分，包含 §7、§8、§9、§10。

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

> **关联设计**: [`plans/pipeline_refactor_design.md` §1.2](../../plans/pipeline_refactor_design.md:62)

### 现状问题

[`NodeContext.metadata`](../../kdit/nodes/core/node_context.py:37) 是一个无类型的 `dict`，当前承载了多种混合关注点：

| metadata key | 使用方 | 类型 |
|---|---|---|
| `text_run_device` | [`T5TextEncodeNode`](../../kdit/nodes/infers/text_encoder_node.py:80) | `str` |
| `offload_model` | [`T5TextEncodeNode`](../../kdit/nodes/infers/text_encoder_node.py:82), [`VAEDecodeNode`](../../kdit/nodes/infers/vae_decoder_node.py:43) | `bool` |
| `noise_shape` | [`GeneratorNode`](../../kdit/nodes/infers/generator_node.py:55) | `list[int]` |
| `control_video_config` | [`GeneratorNode`](../../kdit/nodes/infers/generator_node.py:57) | `VideoControlConfig` |
| `video_control` | [`GeneratorNode`](../../kdit/nodes/infers/generator_node.py:56) | `dict` |
| `with_end_image` | [`VAEDecodeNode`](../../kdit/nodes/infers/vae_decoder_node.py:44) | `bool` |
| `condition_image_path` | [`QwenTextEncodeNode`](../../kdit/nodes/infers/text_encoder_node.py:126) | `str` |
| `comfy_bar_callback` | [`GeneratorNode`](../../kdit/nodes/infers/generator_node.py:58) | `callable` |

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
   - [`generate.py`](../../kdit/adapter/comfyui/generate.py) 中构建 context 的代码

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

> **设计文档**: [`plans/pipeline_refactor_design.md`](../../plans/pipeline_refactor_design.md)

> **核心理念**: Pipeline 是声明式的 Node 流程定义，不是命令式的代码流程

### 架构总览

```
PipelineDefBuilder  ──build()──▶  PipelineDef (不可变)
                                    ├── load_phases: list[LoadTask]
                                    ├── infer_phases: list[InferTask]
                                    └── context_builder_cls: type[ContextBuilder]

Pipeline.from_models(pipeline_key)  ──▶  Pipeline 实例
Pipeline.generate(prompt, ...)      ──▶  输出 tensor / 保存文件
```

### 核心数据类

```python
# kdit/pipelines/pipeline_def.py

@dataclass(frozen=True)
class LoadTask:
    model_role: str           # "text_encoder" / "diffusion" / "vae"
    model_key: ModelKey       # 具体模型 key

@dataclass(frozen=True)
class InferTask:
    node_type: InferNodeType  # TEXT_ENCODE / GENERATE / VAE_DECODE / SAVE_VIDEO ...
    model_role: str | None    # 关联的 model_role，SaveNode 为 None
    condition: str | None     # ContextBuilder 上的条件方法名

@dataclass(frozen=True)
class PipelineDef:
    pipeline_key: PipelineKey
    load_phases: tuple[LoadTask, ...]
    infer_phases: tuple[InferTask, ...]
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
    2. 对每个 InferTask:
       a. check_condition(name, inputs) — 是否跳过
       b. prepare_tensors(phase, inputs) — 准备 tensor → put 到 pool
       c. build_context(phase, inputs) — 构建 NodeContext
    3. post_process(output, inputs) — 输出后处理
    """

    def prepare_generate_inputs(self, base_inputs: PipelineGenerateInputs, **kwargs) -> None:
        """从 kwargs 提取 Pipeline 特有输入，存入 self._extra。"""
        pass

    @abstractmethod
    def build_context(self, phase: InferTask, inputs: PipelineGenerateInputs) -> NodeContext:
        """按 phase.node_type 分支，构建该 Node 的 context。"""
        ...

    def prepare_tensors(self, phase: InferTask, inputs: PipelineGenerateInputs) -> dict | None:
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
- `InferTask` 中 `model_role=None` 表示 SaveNode

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
├── pipeline_def.py          # PipelineDef, LoadTask, InferTask, PipelineDefBuilder
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
