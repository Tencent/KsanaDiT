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

import os

import ray

from ..accelerator import platform
from ..config import KsanaDistributedConfig
from .executor import KsanaExecutor

_ray_options = {"num_gpus": 1}
if platform.is_npu():
    _ray_options = {"num_gpus": 1, "resources": {"NPU": 1}}


@ray.remote(**_ray_options)
class RayKsanaExecutor(KsanaExecutor):

    def init_torch_dist_group(self, rank_id, dist_config: KsanaDistributedConfig):
        """
        初始化 PyTorch DDP。
        """
        master_addr = "127.0.0.1"  # ray.util.get_node_ip_address()
        master_port = dist_config.port
        os.environ["RANK"] = str(rank_id)
        os.environ["WORLD_SIZE"] = str(dist_config.num_gpus)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        super().init_torch_dist_group(rank_id, dist_config)
        return True
