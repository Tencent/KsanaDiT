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

"""Pipeline 声明式定义 — PipelineDef, NodeDef, Edge, PipelineDefBuilder.

PipelineDef 是不可变的数据结构，描述一条 Pipeline 的完整流程：
- DAG 模式（新）：nodes + edges 描述有向无环图
- 旧线性模式（向后兼容）：load_phases + infer_phases
- context_builder_cls: 构建 NodeContext 的策略类

通过 PipelineDefBuilder 链式构建。
"""

from __future__ import annotations

from dataclasses import dataclass

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import InferNodeType
from kdit.tensor import TensorKey
from kdit.utils import log

from .context_builder import ContextBuilder
from .pin_ref import NodeRef
from .pipeline_key import PipelineKey
from .pipeline_phase import InferTask, LoadTask

# ── DAG 数据结构 ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NodeDef:
    """DAG 中一个 Node 实例的定义。

    Attributes:
        node_id: Builder 自动分配的唯一 ID。
        is_loader: True=LoaderNode, False=InferNode。
        node_type: InferNode 的类型（Loader 为 None）。
        model_key: 关联的模型（用于 Factory 查找 Node 类）。
        condition: 条件执行（ContextBuilder 上的方法名）。
    """

    node_id: int
    is_loader: bool
    node_type: InferNodeType | None = None
    model_key: ModelKey | None = None
    condition: str | None = None


@dataclass(frozen=True)
class Edge:
    """DAG 中的一条边 — 从上游 pin 到下游 pin。

    Attributes:
        src_node_id: 源 Node 的 ID。
        src_pin: 源 pin 枚举（ModelKey 或 TensorKey）。
        dst_node_id: 目标 Node 的 ID。
        dst_pin: 目标 pin 枚举（ModelKey 或 TensorKey）。
        edge_type: "model" 或 "tensor"。
    """

    src_node_id: int
    src_pin: ModelKey | TensorKey
    dst_node_id: int
    dst_pin: ModelKey | TensorKey
    edge_type: str


# ── PipelineDef ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineDef:
    """Pipeline 的不可变定义 — 支持 DAG 和旧线性模式。

    Attributes:
        pipeline_key: Pipeline 标识。
        nodes: DAG 模式 — 所有 Node 定义。
        edges: DAG 模式 — 所有连线。
        load_phases: 旧线性模式 — 模型加载阶段列表（有序）。
        infer_phases: 旧线性模式 — 推理阶段列表（有序）。
        keep_tensors: tensor_scope 中需要保留的 TensorKey 列表。
        context_builder_cls: ContextBuilder 子类，用于构建 NodeContext。
    """

    pipeline_key: PipelineKey

    # ── DAG 模式（新）──
    nodes: tuple[NodeDef, ...] = ()
    edges: tuple[Edge, ...] = ()

    # ── 旧线性模式（向后兼容，Phase 5 移除）──
    load_phases: tuple[LoadTask, ...] = ()
    infer_phases: tuple[InferTask, ...] = ()

    # ── 公共 ──
    keep_tensors: tuple[TensorKey, ...] = ()
    context_builder_cls: type[ContextBuilder] | None = None


# ── Builder ─────────────────────────────────────────────────────────────


class PipelineDefBuilder:
    """链式构建 PipelineDef 的 Builder — 支持旧线性模式和新 DAG 模式。

    旧模式用法（向后兼容）::

        WAN_T2V_DEF = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.T5TextEncoder)
            .load(ModelKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(InferNodeType.TEXT_ENCODE, ModelKey.T5TextEncoder)
            .add_infer(InferNodeType.GENERATE, ModelKey.Wan2_2_T2V_14B)
            .add_infer(InferNodeType.VAE_DECODE, ModelKey.VAE_WAN2_2)
            .add_infer(InferNodeType.SAVE_VIDEO)
            .keep_tensors(TensorKey.VIDEO)
            .context_builder(WanT2VContextBuilder)
            .build()
        )

    新 DAG 模式用法::

        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        t5 = builder.add_loader(ModelKey.T5TextEncoder)
        dit = builder.add_loader(ModelKey.Wan2_2_T2V_14B)
        vae = builder.add_loader(ModelKey.VAE_WAN2_2)
        enc = builder.add_infer(InferNodeType.TEXT_ENCODE, ModelKey.T5TextEncoder)
        gen = builder.add_infer(InferNodeType.GENERATE, ModelKey.Wan2_2_T2V_14B)
        dec = builder.add_infer(InferNodeType.VAE_DECODE, ModelKey.VAE_WAN2_2)
        save = builder.add_infer(InferNodeType.SAVE_VIDEO)
        builder.connect(
            (t5.T5TextEncoder, enc.T5TextEncoder),
            (enc.POSITIVE, gen.POSITIVE),
            (enc.NEGATIVE, gen.NEGATIVE),
            (dit.Wan2_2_T2V_14B, gen.Wan2_2_T2V_14B),
            (gen.LATENTS, dec.LATENTS),
            (vae.VAE_WAN2_2, dec.VAE_WAN2_2),
            (dec.VIDEO, save.VIDEO),
        )
        pipeline_def = builder.keep_tensors(TensorKey.VIDEO).context_builder(ctx_cls).build()
    """

    def __init__(self, pipeline_key: PipelineKey):
        self._pipeline_key = pipeline_key
        self._id_counter = 0
        self._node_defs: list[NodeDef] = []
        self._edges: list[Edge] = []
        # 旧模式兼容
        self._load_phases: list[LoadTask] = []
        self._infer_phases: list[InferTask] = []
        self._is_dag_mode = False
        # 公共
        self._context_builder_cls: type[ContextBuilder] | None = None
        self._keep_tensors: list[TensorKey] = []

    def _alloc_node_id(self) -> int:
        """分配唯一的 node_id，从 0 开始递增。"""
        nid = self._id_counter
        self._id_counter += 1
        return nid

    # ── 旧线性模式 API ──

    def load(self, model_key: ModelKey) -> PipelineDefBuilder:
        """添加一个模型加载阶段（旧线性模式）。"""
        self._load_phases.append(LoadTask(model_key=model_key))
        return self

    # ── 新 DAG API ──

    def add_loader(self, model_key: ModelKey) -> NodeRef:
        """添加一个 Loader Node，返回 NodeRef（切换到 DAG 模式）。"""
        self._is_dag_mode = True
        node_id = self._alloc_node_id()
        self._node_defs.append(NodeDef(node_id=node_id, is_loader=True, model_key=model_key))
        return NodeRef(node_id)

    def add_infer(
        self,
        node_type: InferNodeType,
        model_key: ModelKey | None = None,
    ) -> NodeRef | _InferTaskChain:
        """添加一个 Infer Node。

        DAG 模式返回 NodeRef，旧模式返回 _InferTaskChain（向后兼容）。
        """
        if self._is_dag_mode:
            node_id = self._alloc_node_id()
            self._node_defs.append(NodeDef(node_id=node_id, is_loader=False, node_type=node_type, model_key=model_key))
            return NodeRef(node_id)
        else:
            # 旧模式兼容 — 保持原有行为
            phase = InferTask(node_type=node_type, model_key=model_key)
            self._infer_phases.append(phase)
            return _InferTaskChain(self, len(self._infer_phases) - 1)

    def connect(self, *edges) -> PipelineDefBuilder:
        """声明连线。支持两种格式 + 一对多。

        格式 1（4-tuple）::

            (InferNodeType_or_ModelKey, src_pin, InferNodeType_or_ModelKey, dst_pin)

        格式 2（2-tuple of PinRef）::

            (src_node_ref.PIN, dst_node_ref.PIN)

        一对多::

            (src_node_ref.PIN, [dst1_ref.PIN, dst2_ref.PIN])
        """
        self._is_dag_mode = True
        for edge in edges:
            if len(edge) == 4:
                src_ref_key, src_pin, dst_ref_key, dst_pin = edge
                src_id = self._find_node_by_ref_key(src_ref_key)
                dst_id = self._find_node_by_ref_key(dst_ref_key)
                self._add_edge(src_id, src_pin, dst_id, dst_pin)
            elif len(edge) == 2:
                src_ref, dst_ref = edge
                if isinstance(dst_ref, list):
                    for d in dst_ref:
                        self._add_edge(src_ref.node_id, src_ref.pin, d.node_id, d.pin)
                else:
                    self._add_edge(src_ref.node_id, src_ref.pin, dst_ref.node_id, dst_ref.pin)
            else:
                raise ValueError(f"connect() expects 2-tuple or 4-tuple, got {len(edge)}-tuple")
        return self

    def _add_edge(self, src_id: int, src_pin: ModelKey | TensorKey, dst_id: int, dst_pin: ModelKey | TensorKey):
        """添加一条边，校验 pin 类型匹配。"""
        if type(src_pin) is not type(dst_pin):
            raise TypeError(
                f"Cannot connect {type(src_pin).__name__}.{src_pin.name} "
                f"to {type(dst_pin).__name__}.{dst_pin.name}: type mismatch"
            )
        edge_type = "model" if isinstance(src_pin, ModelKey) else "tensor"
        self._edges.append(Edge(src_id, src_pin, dst_id, dst_pin, edge_type))

    def _find_node_by_ref_key(self, ref_key: InferNodeType | ModelKey) -> int:
        """格式 1 查找：InferNodeType 或 ModelKey（loader）。"""
        if isinstance(ref_key, InferNodeType):
            matches = [n for n in self._node_defs if n.node_type == ref_key]
            if len(matches) != 1:
                raise ValueError(f"InferNodeType {ref_key} must be unique in pipeline, found {len(matches)}")
            return matches[0].node_id
        if isinstance(ref_key, ModelKey):
            matches = [n for n in self._node_defs if n.is_loader and n.model_key == ref_key]
            if len(matches) != 1:
                raise ValueError(f"Loader ModelKey {ref_key} must be unique in pipeline, found {len(matches)}")
            return matches[0].node_id
        raise TypeError(f"Unsupported ref_key type: {type(ref_key)}")

    # ── 公共 API ──

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
        if self._is_dag_mode:
            return self._build_dag()
        else:
            return self._build_legacy()

    def _build_dag(self) -> PipelineDef:
        """DAG 模式构建。"""
        self._validate_dag()
        return PipelineDef(
            pipeline_key=self._pipeline_key,
            nodes=tuple(self._node_defs),
            edges=tuple(self._edges),
            keep_tensors=tuple(self._keep_tensors),
            context_builder_cls=self._context_builder_cls,
        )

    def _build_legacy(self) -> PipelineDef:
        """旧线性模式构建 — 保持原有校验逻辑。"""
        if self._context_builder_cls is None:
            raise ValueError("context_builder_cls is required. Call .context_builder(cls) before .build().")
        if not self._load_phases:
            raise ValueError("At least one load phase is required. Call .load() before .build().")
        if not self._infer_phases:
            raise ValueError("At least one infer phase is required. Call .add_infer() before .build().")

        # 校验 infer_phases 中的 model_key 都在 load_phases 中声明过
        load_keys = {lp.model_key for lp in self._load_phases}
        for ip in self._infer_phases:
            if ip.model_key is not None and ip.model_key not in load_keys:
                raise ValueError(
                    f"InferTask references model_key={ip.model_key!r} "
                    f"which is not declared in any LoadTask. "
                    f"Available keys: {load_keys}"
                )

        return PipelineDef(
            pipeline_key=self._pipeline_key,
            load_phases=tuple(self._load_phases),
            infer_phases=tuple(self._infer_phases),
            context_builder_cls=self._context_builder_cls,
            keep_tensors=tuple(self._keep_tensors),
        )

    def _validate_dag(self):
        """DAG 校验规则。

        1. 类型匹配 — 已在 _add_edge 中校验
        2. 重复输入检测 — 同一个 dst pin 不能有两条入边
        3. Model 重复检测 — 两个 loader 加载同一个 ModelKey
        """
        # 重复输入检测
        seen: set[tuple[int, ModelKey | TensorKey]] = set()
        for edge in self._edges:
            key = (edge.dst_node_id, edge.dst_pin)
            if key in seen:
                raise ValueError(
                    f"Duplicate input: node {edge.dst_node_id} pin {edge.dst_pin} " f"has multiple sources"
                )
            seen.add(key)

        # Model 重复检测
        loader_keys = [n.model_key for n in self._node_defs if n.is_loader and n.model_key is not None]
        if len(loader_keys) != len(set(loader_keys)):
            dupes = {k for k in loader_keys if loader_keys.count(k) > 1}
            raise ValueError(f"Duplicate loader ModelKey: {dupes}")


class _InferTaskChain:
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
        self._builder._infer_phases[self._phase_index] = InferTask(
            node_type=old.node_type,
            model_key=old.model_key,
            condition=condition_name,
        )
        return self._builder

    def __getattr__(self, name):
        """代理到 PipelineDefBuilder — 允许不调用 .when() 直接继续链式构建。"""
        return getattr(self._builder, name)


# ── PipelineDef 注册表 ──────────────────────────────────────────────────

_PIPELINE_DEF_REGISTRY: dict[PipelineKey, PipelineDef] = {}


def register_pipeline_def(pipeline_def: PipelineDef) -> PipelineDef:
    """注册一个 PipelineDef 到全局注册表。"""
    key = pipeline_def.pipeline_key
    if key in _PIPELINE_DEF_REGISTRY:
        log.warning(f"PipelineDef for {key} is being overwritten.")
    _PIPELINE_DEF_REGISTRY[key] = pipeline_def
    return pipeline_def


def get_pipeline_def(pipeline_key) -> PipelineDef:
    """从全局注册表获取 PipelineDef。"""
    if pipeline_key not in _PIPELINE_DEF_REGISTRY:
        raise KeyError(
            f"No PipelineDef registered for {pipeline_key}. " f"Available: {list(_PIPELINE_DEF_REGISTRY.keys())}"
        )
    return _PIPELINE_DEF_REGISTRY[pipeline_key]
