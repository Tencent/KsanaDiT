# Copyright 2025 Tencent
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

"""DistributedGroupManager 单元测试。"""

import torch

from kdit.executor.distributed_group import DistributedGroupManager
from kdit.tensor import TensorKey, TensorPool
from kdit.tensor.tensor_pool_key import TensorPoolKey


class TestDistributedGroupManager:
    def test_default_state(self):
        mgr = DistributedGroupManager()
        assert mgr.rank_id == 0
        assert mgr.world_size == 1
        assert not mgr.is_initialized

    def test_init_single_gpu(self):
        mgr = DistributedGroupManager()
        mgr.init(0, 1)
        # world_size=1 → 不算 initialized
        assert not mgr.is_initialized

    def test_broadcast_noop_when_not_initialized(self):
        mgr = DistributedGroupManager()
        pool = TensorPool()
        key = TensorPoolKey(0, TensorKey.POSITIVE)
        pool.put(key, torch.zeros(2))
        # 不应抛异常，直接跳过
        mgr.broadcast_tensors(tensor_pool=pool, keys=[key], src_rank=0)

    def test_broadcast_list_tensor_noop_when_not_initialized(self):
        mgr = DistributedGroupManager()
        pool = TensorPool()
        key = TensorPoolKey(0, TensorKey.BASE_LATENT)
        pool.put(key, [torch.zeros(2, 3), torch.ones(4, 5)])
        # list[Tensor] 也不应抛异常，直接跳过
        mgr.broadcast_tensors(tensor_pool=pool, keys=[key], src_rank=0)
        # 验证 pool 中的值未被修改
        tensor_value = pool.get(key)
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 2
