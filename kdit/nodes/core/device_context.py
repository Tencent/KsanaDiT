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

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KsanaDeviceContext:
    """运行时设备信息，由 Executor 创建并传入 Node.run()。

    frozen=True 保证 Node 无法篡改设备配置。
    """

    device: torch.device
    offload_device: torch.device
    rank_id: int
    world_size: int
