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

"""Global pytest configuration: enforce CPU memory limits in CI (Docker) environments.

Uses cgroup memory limit to restrict **physical** (RSS) memory only,
without affecting GPU virtual address space mappings.
"""

import os
import platform
from pathlib import Path

# Default CPU memory limit: 160 GB (in bytes)
_DEFAULT_CPU_MEM_LIMIT_GB = 160
_BYTES_PER_GB = 1024 * 1024 * 1024


def _set_cgroup_memory_limit(limit_bytes: int) -> bool:
    """Try to set cgroup memory limit. Returns True on success.

    Supports both cgroup v2 (``memory.max``) and cgroup v1
    (``memory.limit_in_bytes``).  Inside Docker the current process
    typically belongs to the root cgroup (``/sys/fs/cgroup/``).
    """
    # cgroup v2
    cg2 = Path("/sys/fs/cgroup/memory.max")
    if cg2.exists():
        try:
            cg2.write_text(str(limit_bytes))
            return True
        except OSError:
            pass

    # cgroup v1
    cg1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cg1.exists():
        try:
            cg1.write_text(str(limit_bytes))
            return True
        except OSError:
            pass

    return False


def _enforce_cpu_memory_limit():
    """Set cgroup memory limit to cap CPU (physical) memory usage.

    The limit can be overridden via the ``KSANA_CPU_MEM_LIMIT_GB`` environment
    variable.  Set it to ``0`` to disable the limit entirely.

    Only applied on Linux (the typical Docker CI environment).
    """
    if platform.system() != "Linux":
        return

    limit_gb_str = os.environ.get("KSANA_CPU_MEM_LIMIT_GB", str(_DEFAULT_CPU_MEM_LIMIT_GB))
    try:
        limit_gb = int(limit_gb_str)
    except ValueError:
        return

    if limit_gb <= 0:
        return

    limit_bytes = limit_gb * _BYTES_PER_GB
    ok = _set_cgroup_memory_limit(limit_bytes)
    if ok:
        print(f"[conftest] CPU memory limit set to {limit_gb} GB via cgroup")
    else:
        print("[conftest] WARNING: failed to set cgroup memory limit (not in Docker or no permission)")


_enforce_cpu_memory_limit()
