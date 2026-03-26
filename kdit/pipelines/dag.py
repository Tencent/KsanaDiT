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

"""DAG 拓扑排序与 pin 映射工具。

提供 topo_sort() 和 compute_input_pins()，供 Pipeline 执行器使用。
"""

from collections import deque

from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.node_def import NodeDef
from kdit.nodes.core.pin_def import Pins
from kdit.tensor.tensor_pool_key import TensorPoolKey

from .pipeline_def import Edge


def topo_sort(nodes: tuple[NodeDef, ...], edges: tuple[Edge, ...]) -> list[NodeDef]:
    """对 DAG 进行拓扑排序，返回执行顺序。

    Loader 节点排在前面，Infer 节点按依赖顺序排列。
    同一入度的节点按 node_id 排序以保证确定性。

    Raises:
        ValueError: DAG 中存在环。
    """
    node_by_id: dict[int, NodeDef] = {n.node_id: n for n in nodes}

    # 构建邻接表（按 (src, dst) 对去重）
    adj: dict[int, list[int]] = {n.node_id: [] for n in nodes}
    unique_deps: set[tuple[int, int]] = set()
    in_degree: dict[int, int] = {n.node_id: 0 for n in nodes}

    for edge in edges:
        pair = (edge.src_node_id, edge.dst_node_id)
        if pair not in unique_deps:
            unique_deps.add(pair)
            adj[edge.src_node_id].append(edge.dst_node_id)
            in_degree[edge.dst_node_id] += 1

    # BFS — 入度为 0 的节点入队，按 node_id 排序保证确定性
    queue: deque[NodeDef] = deque()
    zero_in = sorted(
        [n for n in nodes if in_degree[n.node_id] == 0],
        key=lambda n: n.node_id,
    )
    queue.extend(zero_in)

    result: list[NodeDef] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        # 按 neighbor_id 排序保证确定性
        for neighbor_id in sorted(adj[node.node_id]):
            in_degree[neighbor_id] -= 1
            if in_degree[neighbor_id] == 0:
                queue.append(node_by_id[neighbor_id])

    if len(result) != len(nodes):
        raise ValueError("DAG contains a cycle")

    return result


def compute_input_pins(
    node_def: NodeDef,
    edges: tuple[Edge, ...],
    all_outputs: dict[int, Pins] | None = None,
) -> Pins:
    """从 DAG edges 计算当前 Node 的 input_pins（扁平映射）。

    Args:
        node_def: 目标节点定义。
        edges: DAG 边集合。
        all_outputs: 上游 Node 的 output_pins 集合（``{node_id: Pins}``）。
            提供时从中查找实际 PoolKey（动态模式）；
            为 ``None`` 时从 edge 静态构建 PoolKey（向后兼容）。

    Returns:
        ``{PinDef: PinPoolKey}`` — 即 ``{TensorKey | ModelKey: TensorPoolKey | ModelPoolKey}``。
    """
    pins: Pins = {}
    for edge in edges:
        if edge.dst_node_id != node_def.node_id:
            continue
        if all_outputs is not None:
            src_outputs = all_outputs.get(edge.src_node_id, {})
            pool_key = src_outputs.get(edge.src_pin)
            if pool_key is not None:
                pins[edge.dst_pin] = pool_key
        else:
            if isinstance(edge.src_pin, ModelKey):
                pins[edge.dst_pin] = ModelPoolKey(edge.src_node_id, edge.src_pin)
            else:
                pins[edge.dst_pin] = TensorPoolKey(edge.src_node_id, edge.src_pin)
    return pins
