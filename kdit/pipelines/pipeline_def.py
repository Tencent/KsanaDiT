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

PipelineDef 是不可变的数据结构，描述一条 Pipeline 的完整 DAG 流程：
- nodes + edges 描述有向无环图
- context_builder_cls: 构建 NodeContext 的策略类

通过 PipelineDefBuilder 链式构建。
"""

from __future__ import annotations

from dataclasses import dataclass

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_def import NodeDef
from kdit.nodes.core.node_types import InferNodeType, IONodeType
from kdit.tensor import TensorKey
from kdit.utils import log

from .context_builder import ContextBuilder
from .pin_ref import NodeRef
from .pipeline_key import PipelineKey

# ── DAG 数据结构 ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Edge:
    """DAG 中的一条边 — 从上游 pin 到下游 pin。

    Attributes:
        src_node_id: 源 Node 的 ID。
        src_pin: 源 pin 枚举（ModelKey 或 TensorKey）。
        dst_node_id: 目标 Node 的 ID。
        dst_pin: 目标 pin 枚举（ModelKey 或 TensorKey）。
    """

    src_node_id: int
    src_pin: ModelKey | TensorKey
    dst_node_id: int
    dst_pin: ModelKey | TensorKey


# ── PipelineDef ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineDef:
    """Pipeline 的不可变定义 — DAG 模式。

    Attributes:
        pipeline_key: Pipeline 标识。
        nodes: DAG 中所有 Node 定义。
        edges: DAG 中所有连线。
        keep_tensors: 最终输出 TensorKey 列表（不会被自动 consume 释放）。
        context_builder_cls: ContextBuilder 子类，用于构建 NodeContext。
    """

    pipeline_key: PipelineKey

    # ── DAG ──
    nodes: tuple[NodeDef, ...] = ()
    edges: tuple[Edge, ...] = ()

    # ── 公共 ──
    keep_tensors: tuple[TensorKey, ...] = ()
    context_builder_cls: type[ContextBuilder] | None = None


# ── Builder ─────────────────────────────────────────────────────────────


class PipelineDefBuilder:
    """链式构建 PipelineDef 的 Builder — 纯 DAG 模式。

    用法::

        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        t5 = builder.add_loader(ModelKey.T5TextEncoder)
        dit = builder.add_loader(ModelKey.Wan2_2_T2V_14B)
        vae = builder.add_loader(ModelKey.VAE_WAN2_2)
        enc = builder.add_infer(InferNodeType.TEXT_ENCODE, ModelKey.T5TextEncoder)
        gen = builder.add_infer(InferNodeType.GENERATE, ModelKey.Wan2_2_T2V_14B)
        dec = builder.add_infer(InferNodeType.VAE_DECODE, ModelKey.VAE_WAN2_2)
        save = builder.add_infer(InferNodeType.SAVE_VIDEO)
        builder.connect(
            t5.T5TextEncoder >> enc.T5TextEncoder,
            enc.POSITIVE >> gen.POSITIVE,
            enc.NEGATIVE >> gen.NEGATIVE,
            dit.Wan2_2_T2V_14B >> gen.Wan2_2_T2V_14B,
            gen.LATENTS >> dec.LATENTS,
            vae.VAE_WAN2_2 >> dec.VAE_WAN2_2,
            dec.VIDEO >> save.VIDEO,
        )
        pipeline_def = builder.keep_tensors(TensorKey.VIDEO).context_builder(ctx_cls).build()

    条件节点::

        vae_enc = builder.add_infer(NT.VAE_ENCODE_IMAGES, ModelKey.QwenImageVAE).when("has_ref_images")
    """

    def __init__(self, pipeline_key: PipelineKey):
        self._pipeline_key = pipeline_key
        self._id_counter = 0
        self._node_defs: list[NodeDef] = []
        self._edges: list[Edge] = []
        # 公共
        self._context_builder_cls: type[ContextBuilder] | None = None
        self._keep_tensors: list[TensorKey] = []

    def _alloc_node_id(self) -> int:
        """分配唯一的 node_id，从 0 开始递增。"""
        nid = self._id_counter
        self._id_counter += 1
        return nid

    # ── DAG API ──

    def add_loader(self, model_key: ModelKey) -> NodeRef:
        """添加一个 Loader Node，返回 NodeRef。"""
        node_id = self._alloc_node_id()
        self._node_defs.append(NodeDef(node_id=node_id, node_type=IONodeType.LOAD_MODEL, model_key=model_key))
        return NodeRef(node_id)

    def add_infer(
        self,
        node_type: InferNodeType,
        model_key: ModelKey | None = None,
    ) -> _NodeRefWithWhen:
        """添加一个 Infer Node，返回 _NodeRefWithWhen（支持 .when() 和 pin 访问）。"""
        node_id = self._alloc_node_id()
        self._node_defs.append(NodeDef(node_id=node_id, node_type=node_type, model_key=model_key))
        return _NodeRefWithWhen(self, node_id)

    def connect(self, *edges) -> PipelineDefBuilder:
        """声明连线。支持两种格式 + 一对多。

        格式 1（2-tuple of PinRef，推荐用 ``>>`` 操作符）::

            src_node_ref.PIN >> dst_node_ref.PIN

        格式 2（一对多）::

            (src_node_ref.PIN, [dst1_ref.PIN, dst2_ref.PIN])
        """
        for edge in edges:
            if len(edge) == 2:
                src_ref, dst_ref = edge
                if isinstance(dst_ref, list):
                    for d in dst_ref:
                        self._add_edge(src_ref.node_id, src_ref.pin, d.node_id, d.pin)
                else:
                    self._add_edge(src_ref.node_id, src_ref.pin, dst_ref.node_id, dst_ref.pin)
            else:
                raise ValueError(f"connect() expects 2-tuple (use >> operator), got {len(edge)}-tuple")
        return self

    def _add_edge(self, src_id: int, src_pin: ModelKey | TensorKey, dst_id: int, dst_pin: ModelKey | TensorKey):
        """添加一条边，校验 pin 类型匹配。"""
        if type(src_pin) is not type(dst_pin):
            raise TypeError(
                f"Cannot connect {type(src_pin).__name__}.{src_pin.name} "
                f"to {type(dst_pin).__name__}.{dst_pin.name}: type mismatch"
            )
        self._edges.append(Edge(src_id, src_pin, dst_id, dst_pin))

    # ── 公共 API ──

    def keep_tensors(self, *keys: TensorKey) -> PipelineDefBuilder:
        """声明最终输出 TensorKey — 不会被自动 consume 释放。"""
        self._keep_tensors.extend(keys)
        return self

    def context_builder(self, cls: type[ContextBuilder]) -> PipelineDefBuilder:
        """设置 ContextBuilder 子类。"""
        self._context_builder_cls = cls
        return self

    def build(self) -> PipelineDef:
        """构建不可变的 PipelineDef。"""
        self._validate_dag()
        return PipelineDef(
            pipeline_key=self._pipeline_key,
            nodes=tuple(self._node_defs),
            edges=tuple(self._edges),
            keep_tensors=tuple(self._keep_tensors),
            context_builder_cls=self._context_builder_cls,
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


class _NodeRefWithWhen(NodeRef):
    """add_infer() 返回的 Node 引用，支持 .when() 条件设置 + pin 访问。

    调用 .when() 后返回普通 NodeRef（不再支持链式 .when()）。
    不调用 .when() 时，行为与 NodeRef 完全一致。
    """

    def __init__(self, builder: PipelineDefBuilder, node_id: int):
        super().__init__(node_id)
        self._builder = builder

    def when(self, condition_name: str) -> NodeRef:
        """设置条件执行 — condition_name 必须是 ContextBuilder 上的方法名。"""
        # 找到对应的 NodeDef 并替换
        for i, nd in enumerate(self._builder._node_defs):
            if nd.node_id == self._node_id:
                self._builder._node_defs[i] = NodeDef(
                    node_id=nd.node_id,
                    node_type=nd.node_type,
                    model_key=nd.model_key,
                    condition=condition_name,
                )
                break
        return NodeRef(self._node_id)


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
