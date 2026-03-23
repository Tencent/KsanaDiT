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
class DeviceInfo:
    """设备信息 — 由 Executor 创建，注入到 NodeContext.device。

    替代原 NodeDeviceContext，名字更短。frozen=True 保证 Node 无法篡改。
    """

    device: torch.device
    offload_device: torch.device
    rank_id: int
    world_size: int


# 向后兼容别名 — Phase 5 移除
NodeDeviceContext = DeviceInfo
