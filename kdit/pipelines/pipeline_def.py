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

"""Pipeline 声明式定义 — PipelineDef, LoadPhase, InferPhase, PipelineDefBuilder.

PipelineDef 是不可变的数据结构，描述一条 Pipeline 的完整流程：
- load_phases: 模型加载阶段列表
- infer_phases: 推理阶段列表（含条件执行）
- context_builder_cls: 构建 NodeContext 的策略类

通过 PipelineDefBuilder 链式构建。
"""

from __future__ import annotations

from dataclasses import dataclass

from kdit.models.model_key import KsanaModelKey
from kdit.nodes.core.node_types import KsanaInferNodeType
from kdit.tensor import TensorKey

from .context_builder import ContextBuilder
from .pipeline_key import PipelineKey

# ── 数据类 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoadPhase:
    """模型加载阶段 — 声明一个需要加载的模型。

    Attributes:
        model_role: 角色名（如 "text_encoder", "diffusion", "vae"），
                    在 InferPhase 中通过同名引用。
        model_key: 具体的模型 key。
    """

    model_role: str
    model_key: KsanaModelKey


@dataclass(frozen=True)
class InferPhase:
    """推理阶段 — 声明一个 InferNode 的执行。

    Attributes:
        node_type: InferNode 类型枚举。
        model_role: 关联的 model_role（与 LoadPhase 对应），
                    SaveNode 等无模型 Node 为 None。
        condition: ContextBuilder 上的条件方法名，
                   为 None 时无条件执行。
    """

    node_type: KsanaInferNodeType
    model_role: str | None = None
    condition: str | None = None


@dataclass(frozen=True)
class PipelineDef:
    """Pipeline 的不可变定义 — 由 PipelineDefBuilder.build() 生成。

    Attributes:
        pipeline_key: Pipeline 标识。
        load_phases: 模型加载阶段列表（有序）。
        infer_phases: 推理阶段列表（有序）。
        context_builder_cls: ContextBuilder 子类，用于构建 NodeContext。
        keep_tensors: tensor_scope 中需要保留的 TensorKey 列表。
    """

    pipeline_key: PipelineKey
    load_phases: tuple[LoadPhase, ...]
    infer_phases: tuple[InferPhase, ...]
    context_builder_cls: type[ContextBuilder]
    keep_tensors: tuple[TensorKey, ...] = ()


# ── Builder ─────────────────────────────────────────────────────────────


class PipelineDefBuilder:
    """链式构建 PipelineDef 的 Builder。

    用法::

        WAN_T2V_DEF = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load("text_encoder", KsanaModelKey.T5TextEncoder)
            .load("diffusion", KsanaModelKey.Wan2_2_T2V_14B)
            .load("vae", KsanaModelKey.VAE_WAN2_2)
            .add_infer(KsanaInferNodeType.TEXT_ENCODE, model_role="text_encoder")
            .add_infer(KsanaInferNodeType.GENERATE, model_role="diffusion")
            .add_infer(KsanaInferNodeType.VAE_DECODE, model_role="vae")
            .add_infer(KsanaInferNodeType.SAVE_VIDEO)
            .keep_tensors(TensorKey.VIDEO)
            .context_builder(WanT2VContextBuilder)
            .build()
        )
    """

    def __init__(self, pipeline_key: PipelineKey):
        self._pipeline_key = pipeline_key
        self._load_phases: list[LoadPhase] = []
        self._infer_phases: list[InferPhase] = []
        self._context_builder_cls: type[ContextBuilder] | None = None
        self._keep_tensors: list[TensorKey] = []

    def load(self, model_role: str, model_key: KsanaModelKey) -> PipelineDefBuilder:
        """添加一个模型加载阶段。"""
        self._load_phases.append(LoadPhase(model_role=model_role, model_key=model_key))
        return self

    def add_infer(
        self,
        node_type: KsanaInferNodeType,
        model_role: str | None = None,
    ) -> _InferPhaseChain:
        """添加一个推理阶段，返回链式对象以支持 .when() 条件。"""
        phase = InferPhase(node_type=node_type, model_role=model_role)
        self._infer_phases.append(phase)
        return _InferPhaseChain(self, len(self._infer_phases) - 1)

    def keep_tensors(self, *keys: TensorKey) -> PipelineDefBuilder:
        """声明 tensor_scope 中需要保留的 TensorKey。"""
        self._keep_tensors.extend(keys)
        return self

    def context_builder(self, cls: type[ContextBuilder]) -> PipelineDefBuilder:
        """设置 ContextBuilder 子类。"""
        self._context_builder_cls = cls
        return self

    def build(self) -> PipelineDef:
        """构建不可变的 PipelineDef。"""
        if self._context_builder_cls is None:
            raise ValueError("context_builder_cls is required. Call .context_builder(cls) before .build().")
        if not self._load_phases:
            raise ValueError("At least one load phase is required. Call .load() before .build().")
        if not self._infer_phases:
            raise ValueError("At least one infer phase is required. Call .add_infer() before .build().")

        # 校验 infer_phases 中的 model_role 都在 load_phases 中声明过
        load_roles = {lp.model_role for lp in self._load_phases}
        for ip in self._infer_phases:
            if ip.model_role is not None and ip.model_role not in load_roles:
                raise ValueError(
                    f"InferPhase references model_role='{ip.model_role}' "
                    f"which is not declared in any LoadPhase. "
                    f"Available roles: {load_roles}"
                )

        return PipelineDef(
            pipeline_key=self._pipeline_key,
            load_phases=tuple(self._load_phases),
            infer_phases=tuple(self._infer_phases),
            context_builder_cls=self._context_builder_cls,
            keep_tensors=tuple(self._keep_tensors),
        )


class _InferPhaseChain:
    """add_infer() 返回的链式对象，支持 .when() 条件设置。

    调用 .when() 后返回 PipelineDefBuilder 继续链式构建。
    不调用 .when() 时，后续的 .load() / .add_infer() / .build() 等
    方法通过 __getattr__ 代理到 PipelineDefBuilder。
    """

    def __init__(self, builder: PipelineDefBuilder, phase_index: int):
        self._builder = builder
        self._phase_index = phase_index

    def when(self, condition_name: str) -> PipelineDefBuilder:
        """设置条件执行 — condition_name 必须是 ContextBuilder 上的方法名。"""
        old = self._builder._infer_phases[self._phase_index]
        self._builder._infer_phases[self._phase_index] = InferPhase(
            node_type=old.node_type,
            model_role=old.model_role,
            condition=condition_name,
        )
        return self._builder

    def __getattr__(self, name):
        """代理到 PipelineDefBuilder — 允许不调用 .when() 直接继续链式构建。"""
        return getattr(self._builder, name)
