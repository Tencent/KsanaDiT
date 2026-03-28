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

"""FeedTensorNode — 将 staging 区 tensor 注入 DAG 命名空间。

外部代码通过 engine.feed_tensors() 将 tensor 写入 staging(node_id=0)，
FeedTensorNode 将其复制到自身 node_id 的命名空间，使下游 Node 通过 DAG 连线获取。

context.metadata 必须包含:
    feed_keys: list[TensorKey] — 要注入的 tensor key 列表
"""

from kdit.tensor import TensorKey

from ..core.base_node import IONode
from ..core.node_factory import IONodeFactory
from ..core.node_types import IONodeType, NodeDispatchPolicy


@IONodeFactory.register(IONodeType.FEED_TENSOR, [None])
class FeedTensorNode(IONode):
    """将 staging 区 tensor 注入 DAG 命名空间。

    input_pins 由 engine 构建，指向 staging 区的 TensorPoolKey(0, key)。
    run() 读取后写入自身命名空间，下游通过 output_pins 连接。
    """

    dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
    input_defs: list = []
    output_defs: list = []

    def run(self, pins, *, context):
        feed_keys: list[TensorKey] = context.metadata.get("feed_keys", [])
        for key in feed_keys:
            data = pins.get_tensor(key)
            if data is not None:
                pins.put_tensor(key, data)
