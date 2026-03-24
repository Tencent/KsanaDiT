# Copyright 2026 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline — 统一的声明式 Pipeline 类。

通过 PipelineDef 驱动 Load 和 Generate 两个阶段，
ContextBuilder 负责为每个 NodeDef 构建 NodeContext。

用法::

    pipeline = Pipeline.from_models(model_path)
    output = pipeline.generate(prompt, sample_config=..., runtime_config=...)
"""


import gc
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from kdit.config import DistributedConfig, ModelConfig, RuntimeConfig, SampleConfig, SolverType
from kdit.config.cache_config import CacheConfig, HybridCacheConfig
from kdit.config.lora_config import LoraConfig
from kdit.engine import get_engine
from kdit.engine.engine import Engine
from kdit.models.model_key import VAE_KEYS
from kdit.settings import load_default_settings
from kdit.tensor import TensorKey
from kdit.utils import log
from kdit.utils.env import KSANA_PROFILE
from kdit.utils.monitor import report
from kdit.utils.profile import TimeProfiler
from kdit.utils.types import evolve_with_recommend

from .context_builder import ContextBuilder
from .extra_inputs import ExtraInputs
from .generate_inputs import PipelineGenerateInputs
from .pipeline_def import NodeDef, PipelineDef, get_pipeline_def
from .pipeline_key import get_pipeline_key_from_path
from .pipeline_phase import LoadTask

# ── 辅助：醒目的 phase 计时日志 ──────────────────────────────────────────

_SEPARATOR = "=" * 60


def _phase_display_name(phase: LoadTask | NodeDef) -> str:
    """根据 phase 类型构建醒目的显示名称。"""
    if isinstance(phase, NodeDef):
        return _node_def_display_name(phase)
    if isinstance(phase, LoadTask):
        return f"LOAD({phase.model_key.name})"
    return str(phase)


def _node_def_display_name(node_def: NodeDef) -> str:
    """为 DAG 模式的 NodeDef 构建醒目的显示名称。"""
    if node_def.is_loader:
        model_name = node_def.model_key.name if node_def.model_key else "UNKNOWN"
        return f"LOAD({model_name})"
    node_name = node_def.node_type.name if node_def.node_type else f"node_{node_def.node_id}"
    if node_def.model_key is not None:
        return f"{node_name}({node_def.model_key.name})"
    return node_name


@contextmanager
def _task_node_timer(phase: LoadTask | NodeDef):
    """为 load / infer phase 打印醒目的 START / FINISH 日志及耗时。"""
    name = _phase_display_name(phase)
    log.info(_SEPARATOR)
    log.info(f"▶ START  {name}")
    log.info(_SEPARATOR)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    log.info(_SEPARATOR)
    log.info(f"✔ FINISH {name}  ⏱ {elapsed:.2f}s")
    log.info(_SEPARATOR)


# ── Pipeline 类 ─────────────────────────────────────────────────────────


class Pipeline:
    """统一的声明式 Pipeline — 由 PipelineDef 驱动。

    Pipeline 有两个阶段：
    1. Load（from_models / load_models）：按 PipelineDef.nodes 加载模型
    2. Generate（generate）：按 PipelineDef.nodes 执行推理
    """

    def __init__(self, pipeline_def: PipelineDef, engine: Engine, offload_device: str = "cpu"):
        self._def = pipeline_def
        self._engine = engine
        self._offload_device = offload_device
        self._ctx_builder: ContextBuilder = pipeline_def.context_builder_cls()
        # 注入 pipeline_def 引用到 ContextBuilder（用于 edges 查询）
        self._ctx_builder._pipeline_def = pipeline_def

        # 从 settings 加载的默认配置
        self._default_settings = None
        self._has_lora = False
        self._pipeline_name = pipeline_def.pipeline_key.name

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def pipeline_key(self):
        return self._def.pipeline_key

    # ── Load 阶段 ──────────────────────────────────────────────────────

    @staticmethod
    def from_models(
        model_path,
        *,
        model_config: ModelConfig = None,
        dist_config: DistributedConfig = None,
        pipeline_key=None,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora_config: LoraConfig | list[LoraConfig] | None = None,
        offload_device="cpu",
    ):
        """从模型路径创建 Pipeline — 100% 兼容旧 API。"""
        log.info(f"Loading models from {model_path}")

        # 推导 pipeline_key
        if pipeline_key is None:
            if model_path is None:
                raise ValueError("model_path must be provided when pipeline_key is None")
            if isinstance(model_path, str) and not Path(model_path).exists():
                raise ValueError(f"model_path {model_path} does not exist")
            path = None
            if isinstance(model_path, (list, tuple)):
                path = text_checkpoint_dir or vae_checkpoint_dir
            pipeline_key = get_pipeline_key_from_path(model_path if path is None else text_checkpoint_dir)

        # 获取 PipelineDef
        pipeline_def = get_pipeline_def(pipeline_key)

        # 创建 Engine
        model_config = model_config or ModelConfig()
        dist_config = dist_config or __import__("kdit.config", fromlist=["DistributedConfig"]).DistributedConfig()
        engine = get_engine(dist_config=dist_config, offload_device=offload_device)

        # 创建 Pipeline 并加载模型
        pipeline = Pipeline(pipeline_def, engine, offload_device)
        pipeline.load_models(
            model_path,
            model_config=model_config,
            text_checkpoint_dir=text_checkpoint_dir,
            vae_checkpoint_dir=vae_checkpoint_dir,
            lora_config=lora_config,
        )
        return pipeline

    def load_models(
        self,
        model_path,
        *,
        model_config: ModelConfig = None,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora_config: LoraConfig | list[LoraConfig] | None = None,
    ):
        """按 PipelineDef 加载所有模型。"""
        # 先清理本 PipelineDef 声明的所有模型，保证全新加载
        self._engine.clear_models()
        self._has_lora = lora_config is not None
        self._default_settings = load_default_settings(self._def.pipeline_key, with_lora=self._has_lora)

        # 委托给 ContextBuilder
        load_model_path, text_checkpoint_dir, vae_checkpoint_dir = self._ctx_builder.resolve_model_paths(
            model_path, text_checkpoint_dir, vae_checkpoint_dir, self._default_settings
        )
        lora_list = self._ctx_builder.resolve_lora_config(lora_config, self._default_settings) if lora_config else None

        # DAG 模式：按拓扑序遍历 Loader 节点
        from kdit.nodes.core.node_context import NodeContext

        from .dag import compute_pins_mapping, topo_sort

        sorted_nodes = topo_sort(self._def.nodes, self._def.edges)
        for node_def in sorted_nodes:
            if not node_def.is_loader:
                continue
            pins_mapping = compute_pins_mapping(node_def, self._def.edges)
            # 构建 loader context — metadata 中放 build_loader_kwargs() 的结果
            loader_kwargs = self._ctx_builder.build_loader_kwargs(
                node_def.model_key,
                load_model_path,
                text_checkpoint_dir,
                vae_checkpoint_dir,
                model_config=model_config,
                lora_list=lora_list,
                pipeline_settings=self._default_settings,
            )
            context = NodeContext(metadata=loader_kwargs)
            with _task_node_timer(node_def):
                self._engine.run_loader_node(node_def, pins_mapping, context)

    def clear(self):
        """清理所有已加载的模型。"""
        load_keys = [n.model_key for n in self._def.nodes if n.is_loader and n.model_key is not None]
        if load_keys:
            self._engine.cleanup_distributed()

    # ── Generate 阶段 ──────────────────────────────────────────────────

    @report("pipeline_generate")
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
        """按 PipelineDef 执行推理。

        公共参数在此校验，Pipeline 特有参数通过 extra_inputs 传给 ContextBuilder。
        """
        # 校验公共输入
        num_prompts = _get_num_prompts(prompt)
        if num_prompts == 0:
            raise ValueError("prompt must be str or list of str")

        sample_config, runtime_config, cache_config = _prepare_configs(
            sample_config, runtime_config, cache_config, self._default_settings, num_prompts
        )

        # 启动层级 profiler session（仅在 KSANA_PROFILE=1 时生效）
        _profiler = TimeProfiler.start_session("pipeline_generate") if KSANA_PROFILE else None

        log.info(f"generate prompt: {prompt}")
        log.info(f"sample_config : {sample_config}")
        log.info(f"runtime_config : {runtime_config}")
        log.info(f"cache_config : {cache_config}")

        # 构建公共输入
        inputs = PipelineGenerateInputs(
            prompt=prompt,
            prompt_negative=prompt_negative,
            num_prompts=num_prompts,
            sample_config=sample_config,
            runtime_config=runtime_config,
            cache_config=cache_config,
            has_lora=self._has_lora,
        )

        # ContextBuilder 提取特有输入
        vae_model_key = self._find_vae_model_key()
        self._ctx_builder.prepare_generate_inputs(
            inputs,
            extra_inputs,
            _default_settings=self._default_settings,
            _engine=self._engine,
            _vae_model_key=vae_model_key,
        )

        # 执行 infer phases
        keep = list(self._def.keep_tensors)
        with self._engine.tensor_scope(keep=keep):
            self._generate_dag(inputs)

            # 获取输出
            output_tv = self._engine.get_tensor(TensorKey.VIDEO)
            output = output_tv.data if output_tv is not None else None

        # offload 后清理
        if runtime_config.offload_model:
            gc.collect()
            torch.cuda.synchronize()

        # 后处理
        output = self._ctx_builder.post_process(output, inputs)

        # 结束 profiler session 并打印摘要
        if _profiler is not None:
            _profiler.finish()
            _profiler.print_summary()

        return output if runtime_config.return_frames else None

    def _find_vae_model_key(self):
        """从 DAG nodes 中查找 VAE ModelKey。"""
        return next(
            (n.model_key for n in self._def.nodes if n.is_loader and n.model_key in VAE_KEYS),
            None,
        )

    def _generate_dag(self, inputs: PipelineGenerateInputs):
        """按拓扑序遍历 Infer 节点，通过 engine.run_infer_node() 执行。"""
        from .dag import compute_pins_mapping, topo_sort

        sorted_nodes = topo_sort(self._def.nodes, self._def.edges)
        for node_def in sorted_nodes:
            if node_def.is_loader:
                continue

            # 1. 条件检查
            if node_def.condition and not self._ctx_builder.check_condition(node_def.condition, inputs):
                continue

            # 2. 构建 context — 直接传 NodeDef
            node_ctx = self._ctx_builder.build_context(node_def, inputs)

            # 3. 计算 pins_mapping 并执行
            pins_mapping = compute_pins_mapping(node_def, self._def.edges)
            with _task_node_timer(node_def):
                self._engine.run_infer_node(node_def, pins_mapping, node_ctx)


# ── 辅助函数 ─────────────────────────────────────────────────────────────


def _get_num_prompts(prompt: str | list[str]) -> int:
    """计算 prompt 数量。"""
    if isinstance(prompt, str):
        return 1
    if isinstance(prompt, list):
        return len(prompt)
    return 0


def _prepare_configs(
    sample_config: SampleConfig | None,
    runtime_config: RuntimeConfig | None,
    cache_config: list[CacheConfig | HybridCacheConfig] | None,
    default_settings,
    num_prompts: int,
) -> tuple[SampleConfig, RuntimeConfig, list[CacheConfig | HybridCacheConfig] | None]:
    """统一校验并合并所有配置。"""
    sample_config = _merge_sample_config(sample_config, default_settings.sample_config)
    runtime_config = _merge_runtime_config(runtime_config, default_settings.runtime_config, num_prompts)
    cache_config = _ensure_cache_config_list(cache_config, getattr(default_settings, "cache", None))
    return sample_config, runtime_config, cache_config


def _merge_sample_config(sample_config: SampleConfig | None, default_configs) -> SampleConfig:
    """合并 sample_config 与默认配置。"""
    from omegaconf import OmegaConf

    sample_config = sample_config if sample_config else SampleConfig()
    cfg_scale = getattr(default_configs, "cfg_scale", None)
    cfg_scale = OmegaConf.to_container(cfg_scale, resolve=True) if OmegaConf.is_list(cfg_scale) else cfg_scale
    solver = getattr(default_configs, "solver", None)
    solver = SolverType(solver) if isinstance(solver, str) else solver
    recommend_configs = {
        "steps": getattr(default_configs, "steps", None),
        "shift": getattr(default_configs, "shift", None),
        "denoise": getattr(default_configs, "denoise", None),
        "cfg_scale": cfg_scale,
        "solver": solver,
    }
    return evolve_with_recommend(sample_config, recommend_configs)


def _merge_runtime_config(runtime_config: RuntimeConfig | None, default_configs, num_prompts: int) -> RuntimeConfig:
    """合并 runtime_config 与默认配置。"""
    runtime_config = runtime_config or RuntimeConfig()
    batch_size_per_prompts = runtime_config.batch_size_per_prompts
    if batch_size_per_prompts is None:
        batch_size_per_prompts = [1] * num_prompts
    elif isinstance(batch_size_per_prompts, int):
        batch_size_per_prompts = [batch_size_per_prompts] * num_prompts
    elif isinstance(batch_size_per_prompts, (list, tuple)):
        if len(batch_size_per_prompts) != num_prompts:
            raise ValueError(
                f"batch_size_per_prompts({batch_size_per_prompts}) len must match num_prompts ({num_prompts})"
            )
    else:
        raise TypeError(f"batch_size_per_prompts must be int/list[int]/None, but got {type(batch_size_per_prompts)}")
    runtime_config = evolve_with_recommend(
        runtime_config,
        {"batch_size_per_prompts": batch_size_per_prompts},
        force_update=True,
    )
    recommend_configs = {
        "size": getattr(default_configs, "target_size", None),
        "frame_num": getattr(default_configs, "frame_num", None),
    }
    return evolve_with_recommend(runtime_config, recommend_configs, force_update=False)


def _ensure_cache_config_list(
    cache_config: list[CacheConfig | HybridCacheConfig] | CacheConfig | HybridCacheConfig | None,
    default_configs,  # noqa: ARG001
) -> list[CacheConfig | HybridCacheConfig] | None:
    """确保 cache_config 为列表形式。"""
    if cache_config is None:
        return None
    if isinstance(cache_config, (tuple, list)):
        return list(cache_config)
    if isinstance(cache_config, (HybridCacheConfig, CacheConfig)):
        return [cache_config]
    raise ValueError(f"cache_config must be HybridCacheConfig or CacheConfig, but got {type(cache_config)}")
