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

"""FetchTensorNode — 从 DAG 命名空间取回 tensor 到 staging 区。

Pipeline/ComfyUI 通过 engine.fetch_tensor() 按需创建 FetchTensorNode，
将指定 tensor 从上游 node 的命名空间写回 staging(node_id=0)，
使外部代码可以通过 engine.get_tensor() 读取最终结果。

context.metadata 必须包含:
    fetch_keys: list[TensorKey] — 要取回的 tensor key 列表
"""

from kdit.tensor import TensorKey
from kdit.tensor.tensor_pool import _FEED_STAGING_ID
from kdit.tensor.tensor_pool_key import TensorPoolKey

from ..core.base_node import IONode
from ..core.node_factory import IONodeFactory
from ..core.node_types import IONodeType, NodeDispatchPolicy


@IONodeFactory.register(IONodeType.FETCH_TENSOR, [None])
class FetchTensorNode(IONode):
    """从 DAG 命名空间取回 tensor 到 staging 区。

    input_pins 由 engine 构建，指向上游 node 的 TensorPoolKey。
    run() 读取后直接写入 staging 区 TensorPoolKey(0, key)。
    """

    dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
    input_defs: list = []
    output_defs: list = []

    def run(self, pins, *, context):
        fetch_keys: list[TensorKey] = context.metadata.get("fetch_keys", [])
        for key in fetch_keys:
            data = pins.get_tensor(key)
            if data is not None:
                staging_key = TensorPoolKey(_FEED_STAGING_ID, key)
                pins._tensor_pool.put(staging_key, data)
