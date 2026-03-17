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
ContextBuilder 负责为每个 InferPhase 构建 NodeContext。

用法::

    pipeline = Pipeline.from_models(model_path)
    output = pipeline.generate(prompt, sample_config=..., runtime_config=...)
"""


import gc
import os
from pathlib import Path

import torch

from kdit.config import DistributedConfig, KsanaSampleConfig, ModelConfig, RuntimeConfig
from kdit.config.cache_config import HybridCacheConfig, KsanaCacheConfig
from kdit.config.lora_config import KsanaLoraConfig
from kdit.engine import get_engine
from kdit.engine.engine import Engine
from kdit.models.model_key import DIFFUSION_KEYS, TEXT_ENCODER_KEYS, VAE_KEYS
from kdit.settings import load_default_settings
from kdit.tensor import TensorKey
from kdit.utils import log
from kdit.utils.env import KSANA_PROFILE
from kdit.utils.monitor import report
from kdit.utils.profile import TimeProfiler

from .context_builder import ContextBuilder
from .generate_inputs import GenerateInputs
from .pipeline_def import PipelineDef, get_pipeline_def
from .pipeline_key import PipelineKey, get_pipeline_key_from_path

# ── Pipeline 类 ─────────────────────────────────────────────────────────


class Pipeline:
    """统一的声明式 Pipeline — 由 PipelineDef 驱动。

    Pipeline 有两个阶段：
    1. Load（from_models / load_models）：按 PipelineDef.load_phases 加载模型
    2. Generate（generate）：按 PipelineDef.infer_phases 执行推理

    100% 向后兼容 BasePipeline.from_models() 和 pipeline.generate() 的调用方式。
    """

    def __init__(self, pipeline_def: PipelineDef, engine: Engine, offload_device: str = "cpu"):
        self._def = pipeline_def
        self._engine = engine
        self._offload_device = offload_device
        self._ctx_builder: ContextBuilder = pipeline_def.context_builder_cls()

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
        lora_config: KsanaLoraConfig | list[KsanaLoraConfig] | None = None,
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
        lora_config: KsanaLoraConfig | list[KsanaLoraConfig] | None = None,
    ):
        """按 PipelineDef.load_phases 加载所有模型。"""
        self._has_lora = lora_config is not None
        self._default_settings = load_default_settings(self._def.pipeline_key, with_lora=self._has_lora)

        # 解析模型路径
        load_model_path, text_checkpoint_dir, vae_checkpoint_dir = _resolve_model_paths(
            model_path, text_checkpoint_dir, vae_checkpoint_dir, self._def.pipeline_key, self._default_settings
        )

        # 解析 LoRA
        lora_list = _resolve_lora_config(lora_config, self._default_settings.diffusion) if lora_config else None

        # 按 load_phases 顺序加载
        for phase in self._def.load_phases:
            model_key = phase.model_key
            kwargs = self._build_loader_kwargs(
                model_key,
                load_model_path,
                text_checkpoint_dir,
                vae_checkpoint_dir,
                model_config=model_config,
                lora_list=lora_list,
            )
            self._engine.run_loader_node(model_key, **kwargs)

    def _build_loader_kwargs(self, model_key, model_path, text_dir, vae_dir, *, model_config, lora_list) -> dict:
        """根据 ModelKey 类别构建 loader node 的 kwargs。"""
        if model_key in TEXT_ENCODER_KEYS:
            return {"model_path": text_dir}
        if model_key in DIFFUSION_KEYS:
            kwargs = {"model_path": model_path, "model_config": model_config}
            if lora_list:
                kwargs["lora_config"] = lora_list
            return kwargs
        if model_key in VAE_KEYS:
            vae_ckpt = getattr(self._default_settings, "vae", None)
            vae_checkpoint = vae_ckpt.checkpoint if vae_ckpt else ""
            return {"model_path": os.path.join(vae_dir, vae_checkpoint)}
        # 未知类别 — 默认传 model_path
        return {"model_path": model_path}

    def clear(self):
        """清理所有已加载的模型。"""
        load_keys = [lp.model_key for lp in self._def.load_phases]
        if load_keys:
            self._engine.cleanup_distributed()

    # ── Generate 阶段 ──────────────────────────────────────────────────

    @report("pipeline_generate")
    def generate(
        self,
        prompt: str | list[str],
        *,
        prompt_negative: str | list[str] | None = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: RuntimeConfig = None,
        cache_config: list[KsanaCacheConfig | HybridCacheConfig] | None = None,
        **kwargs,
    ):
        """按 PipelineDef.infer_phases 执行推理 — 100% 兼容旧 API。

        公共参数在此校验，Pipeline 特有参数通过 **kwargs 传给 ContextBuilder。
        """
        # 校验公共输入
        num_prompts = _get_num_prompts(prompt)
        if num_prompts == 0:
            raise ValueError("prompt must be str or list of str")

        sample_config = _valid_sample_config(sample_config, self._default_settings.sample_config)
        runtime_config = _valid_runtime_config(runtime_config, self._default_settings.runtime_config, num_prompts)
        cache_config = _valid_cache_config(cache_config, getattr(self._default_settings, "cache", None))

        # 启动层级 profiler session（仅在 KSANA_PROFILE=1 时生效）
        _profiler = TimeProfiler.start_session("pipeline_generate") if KSANA_PROFILE else None

        log.info(f"generate prompt: {prompt}")
        log.info(f"sample_config : {sample_config}")
        log.info(f"runtime_config : {runtime_config}")
        log.info(f"cache_config : {cache_config}")

        # 构建公共输入
        inputs = GenerateInputs(
            prompt=prompt,
            prompt_negative=prompt_negative,
            num_prompts=num_prompts,
            sample_config=sample_config,
            runtime_config=runtime_config,
            cache_config=cache_config,
            has_lora=self._has_lora,
        )

        # ContextBuilder 提取特有输入（注入内部引用供 noise_shape / VACE 等计算）
        vae_model_key = next((lp.model_key for lp in self._def.load_phases if lp.model_key in VAE_KEYS), None)
        self._ctx_builder.prepare_generate_inputs(
            inputs,
            _default_settings=self._default_settings,
            _engine=self._engine,
            _vae_model_key=vae_model_key,
            **kwargs,
        )

        # 执行 infer phases
        keep = list(self._def.keep_tensors)
        with self._engine.tensor_scope(keep=keep):
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
                self._engine.run_infer_node(phase.node_type, phase.model_key, node_ctx)

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


# ── 辅助函数（从 BasePipeline 迁移） ────────────────────────────────────


def _get_num_prompts(prompt: str | list[str]) -> int:
    """计算 prompt 数量。"""
    if isinstance(prompt, str):
        return 1
    if isinstance(prompt, list):
        return len(prompt)
    return 0


def _valid_sample_config(sample_config, default_config):
    """校验并合并 sample_config。"""
    if sample_config is None:
        return default_config
    # 合并默认值
    if sample_config.steps is None:
        sample_config.steps = default_config.steps
    if sample_config.cfg_scale is None:
        sample_config.cfg_scale = default_config.cfg_scale
    if sample_config.shift is None:
        sample_config.shift = default_config.shift
    if sample_config.solver_name is None:
        sample_config.solver_name = default_config.solver_name
    if sample_config.fps is None:
        sample_config.fps = default_config.fps
    return sample_config


def _valid_runtime_config(runtime_config, default_config, num_prompts: int):
    """校验并合并 runtime_config。"""
    if runtime_config is None:
        runtime_config = RuntimeConfig()
    if runtime_config.size is None:
        runtime_config.size = default_config.size
    if runtime_config.frame_num is None:
        runtime_config.frame_num = default_config.frame_num
    if runtime_config.batch_size_per_prompts is None:
        runtime_config.batch_size_per_prompts = [1] * num_prompts
    if runtime_config.output_folder is None:
        runtime_config.output_folder = default_config.output_folder
    return runtime_config


def _valid_cache_config(cache_config, default_config):
    """校验 cache_config。"""
    if cache_config is None:
        return default_config
    return cache_config


def _resolve_model_paths(model_path, text_checkpoint_dir, vae_checkpoint_dir, pipeline_key, pipeline_settings):
    """解析模型路径 — 从 BasePipeline._valid_input_models_path 迁移。"""
    if isinstance(model_path, (list, tuple)):
        if not Path(text_checkpoint_dir).is_dir():
            raise ValueError(
                f"text_checkpoint_dir must be provided when loading from local checkpoint "
                f"with diffusion model {model_path}"
            )
        if not Path(vae_checkpoint_dir).is_dir():
            raise ValueError(
                f"vae_checkpoint_dir must be provided when loading from local checkpoint "
                f"with diffusion model {model_path}"
            )
        load_model_path = list(model_path)
    elif Path(model_path).is_dir():
        load_model_path = model_path
        text_checkpoint_dir = text_checkpoint_dir or model_path
        vae_checkpoint_dir = vae_checkpoint_dir or model_path
        diffusion_settings = pipeline_settings.diffusion
        if pipeline_key in [PipelineKey.Wan2_2_I2V_14B, PipelineKey.Wan2_2_T2V_14B]:
            load_model_path = [
                os.path.join(model_path, diffusion_settings.high_noise_checkpoint),
                os.path.join(model_path, diffusion_settings.low_noise_checkpoint),
            ]
    else:
        raise ValueError(f"model_path {model_path} should be a directory or list of diffusion model files")
    return load_model_path, text_checkpoint_dir, vae_checkpoint_dir


def _resolve_lora_config(lora_config, diffusion_settings):
    """解析 LoRA 配置 — 从 BasePipeline._valid_input_lora 迁移。"""
    if lora_config is None:
        return None
    if isinstance(lora_config, KsanaLoraConfig):
        lora_config = [lora_config]
    # 返回 list of list（每个 diffusion model shard 一份）
    return [lora_config]
