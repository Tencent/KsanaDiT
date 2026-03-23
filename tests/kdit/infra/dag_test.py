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

"""DAG 拓扑排序与 pin 映射 单元测试。"""

import unittest

from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.pipelines.dag import compute_pins_mapping, topo_sort
from kdit.pipelines.pipeline_def import Edge, NodeDef
from kdit.tensor import TensorKey
from kdit.tensor.tensor_pool_key import TensorPoolKey


class TestTopoSort(unittest.TestCase):
    """topo_sort() 拓扑排序测试。"""

    def test_linear_dag(self):
        """线性 DAG: A → B → C 拓扑排序结果正确。"""
        a = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)
        b = NodeDef(node_id=1, is_loader=False, node_type=NT.TEXT_ENCODE)
        c = NodeDef(node_id=2, is_loader=False, node_type=NT.SAVE_VIDEO)

        nodes = (a, b, c)
        edges = (
            Edge(0, ModelKey.T5TextEncoder, 1, ModelKey.T5TextEncoder, "model"),
            Edge(1, TensorKey.POSITIVE, 2, TensorKey.POSITIVE, "tensor"),
        )

        result = topo_sort(nodes, edges)
        self.assertEqual([n.node_id for n in result], [0, 1, 2])

    def test_diamond_dag(self):
        """菱形 DAG: A→B, A→C, B→D, C→D 拓扑排序正确。"""
        a = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)
        b = NodeDef(node_id=1, is_loader=False, node_type=NT.TEXT_ENCODE)
        c = NodeDef(node_id=2, is_loader=False, node_type=NT.VAE_DECODE)
        d = NodeDef(node_id=3, is_loader=False, node_type=NT.SAVE_VIDEO)

        nodes = (a, b, c, d)
        edges = (
            Edge(0, TensorKey.POSITIVE, 1, TensorKey.POSITIVE, "tensor"),
            Edge(0, TensorKey.NEGATIVE, 2, TensorKey.NEGATIVE, "tensor"),
            Edge(1, TensorKey.LATENTS, 3, TensorKey.LATENTS, "tensor"),
            Edge(2, TensorKey.VIDEO, 3, TensorKey.VIDEO, "tensor"),
        )

        result = topo_sort(nodes, edges)
        ids = [n.node_id for n in result]

        # A 必须在 B 和 C 之前
        self.assertLess(ids.index(0), ids.index(1))
        self.assertLess(ids.index(0), ids.index(2))
        # B 和 C 必须在 D 之前
        self.assertLess(ids.index(1), ids.index(3))
        self.assertLess(ids.index(2), ids.index(3))

    def test_multiple_edges_same_pair(self):
        """同一对 src→dst 有多条边时，依赖只算一次。"""
        a = NodeDef(node_id=0, is_loader=False, node_type=NT.TEXT_ENCODE)
        b = NodeDef(node_id=1, is_loader=False, node_type=NT.GENERATE)

        nodes = (a, b)
        edges = (
            Edge(0, TensorKey.POSITIVE, 1, TensorKey.POSITIVE, "tensor"),
            Edge(0, TensorKey.NEGATIVE, 1, TensorKey.NEGATIVE, "tensor"),
        )

        result = topo_sort(nodes, edges)
        self.assertEqual([n.node_id for n in result], [0, 1])

    def test_isolated_nodes(self):
        """无边的独立节点也能排序。"""
        a = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)
        b = NodeDef(node_id=1, is_loader=True, model_key=ModelKey.VAE_WAN2_2)

        nodes = (a, b)
        edges = ()

        result = topo_sort(nodes, edges)
        self.assertEqual(len(result), 2)
        # 按 node_id 排序
        self.assertEqual([n.node_id for n in result], [0, 1])

    def test_cycle_detection(self):
        """有环时抛出 ValueError。"""
        a = NodeDef(node_id=0, is_loader=False, node_type=NT.TEXT_ENCODE)
        b = NodeDef(node_id=1, is_loader=False, node_type=NT.GENERATE)

        nodes = (a, b)
        edges = (
            Edge(0, TensorKey.POSITIVE, 1, TensorKey.POSITIVE, "tensor"),
            Edge(1, TensorKey.NEGATIVE, 0, TensorKey.NEGATIVE, "tensor"),
        )

        with self.assertRaises(ValueError, msg="cycle"):
            topo_sort(nodes, edges)

    def test_loaders_first(self):
        """Loader 节点（入度 0）排在 Infer 节点前面。"""
        loader = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)
        infer = NodeDef(node_id=1, is_loader=False, node_type=NT.TEXT_ENCODE)

        nodes = (infer, loader)  # 故意反序
        edges = (Edge(0, ModelKey.T5TextEncoder, 1, ModelKey.T5TextEncoder, "model"),)

        result = topo_sort(nodes, edges)
        self.assertEqual(result[0].node_id, 0)  # loader first
        self.assertEqual(result[1].node_id, 1)


class TestComputePinsMapping(unittest.TestCase):
    """compute_pins_mapping() 测试。"""

    def test_tensor_mapping(self):
        """正确计算 tensor 映射。"""
        node = NodeDef(node_id=2, is_loader=False, node_type=NT.GENERATE)
        edges = (
            Edge(1, TensorKey.POSITIVE, 2, TensorKey.POSITIVE, "tensor"),
            Edge(1, TensorKey.NEGATIVE, 2, TensorKey.NEGATIVE, "tensor"),
        )

        mapping = compute_pins_mapping(node, edges)

        self.assertEqual(len(mapping["tensor"]), 2)
        self.assertEqual(len(mapping["model"]), 0)
        self.assertEqual(mapping["tensor"][TensorKey.POSITIVE], TensorPoolKey(1, TensorKey.POSITIVE))
        self.assertEqual(mapping["tensor"][TensorKey.NEGATIVE], TensorPoolKey(1, TensorKey.NEGATIVE))

    def test_model_mapping(self):
        """正确计算 model 映射。"""
        node = NodeDef(node_id=1, is_loader=False, node_type=NT.TEXT_ENCODE)
        edges = (Edge(0, ModelKey.T5TextEncoder, 1, ModelKey.T5TextEncoder, "model"),)

        mapping = compute_pins_mapping(node, edges)

        self.assertEqual(len(mapping["tensor"]), 0)
        self.assertEqual(len(mapping["model"]), 1)
        self.assertEqual(mapping["model"][ModelKey.T5TextEncoder], ModelPoolKey(0, ModelKey.T5TextEncoder))

    def test_mixed_mapping(self):
        """同时有 tensor 和 model 映射。"""
        node = NodeDef(node_id=2, is_loader=False, node_type=NT.GENERATE)
        edges = (
            Edge(0, ModelKey.Wan2_2_T2V_14B, 2, ModelKey.Wan2_2_T2V_14B, "model"),
            Edge(1, TensorKey.POSITIVE, 2, TensorKey.POSITIVE, "tensor"),
            Edge(1, TensorKey.NEGATIVE, 2, TensorKey.NEGATIVE, "tensor"),
            # 不相关的边（目标不是 node 2）
            Edge(0, ModelKey.T5TextEncoder, 1, ModelKey.T5TextEncoder, "model"),
        )

        mapping = compute_pins_mapping(node, edges)

        self.assertEqual(len(mapping["tensor"]), 2)
        self.assertEqual(len(mapping["model"]), 1)
        self.assertEqual(
            mapping["model"][ModelKey.Wan2_2_T2V_14B],
            ModelPoolKey(0, ModelKey.Wan2_2_T2V_14B),
        )

    def test_no_edges_for_node(self):
        """节点没有入边时返回空映射。"""
        node = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)
        edges = (Edge(0, ModelKey.T5TextEncoder, 1, ModelKey.T5TextEncoder, "model"),)

        mapping = compute_pins_mapping(node, edges)

        self.assertEqual(mapping["tensor"], {})
        self.assertEqual(mapping["model"], {})


if __name__ == "__main__":
    unittest.main()
