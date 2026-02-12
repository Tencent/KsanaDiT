#!/bin/bash
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

# Common CI test environment setup
# Usage: source test_env.sh [GPU_CARDS]

# CPU memory limit (160 GB), override via KSANA_CPU_MEM_LIMIT_GB env var.
# Uses cgroup to limit physical (RSS) memory only, so GPU virtual address
# space mappings are not affected.
KSANA_CPU_MEM_LIMIT_GB=${KSANA_CPU_MEM_LIMIT_GB:-160}
export KSANA_CPU_MEM_LIMIT_GB
if [ "${KSANA_CPU_MEM_LIMIT_GB}" -gt 0 ] 2>/dev/null; then
    MEM_LIMIT_BYTES=$((KSANA_CPU_MEM_LIMIT_GB * 1024 * 1024 * 1024))
    if [ -f /sys/fs/cgroup/memory.max ]; then
        # cgroup v2
        echo "${MEM_LIMIT_BYTES}" > /sys/fs/cgroup/memory.max 2>/dev/null && \
            echo "CPU memory limit set to ${KSANA_CPU_MEM_LIMIT_GB} GB via cgroup v2" || \
            echo "WARNING: failed to set cgroup v2 memory limit (no permission?)"
    elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        # cgroup v1
        echo "${MEM_LIMIT_BYTES}" > /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null && \
            echo "CPU memory limit set to ${KSANA_CPU_MEM_LIMIT_GB} GB via cgroup v1" || \
            echo "WARNING: failed to set cgroup v1 memory limit (no permission?)"
    else
        echo "WARNING: cgroup memory interface not found, CPU memory limit not applied"
    fi
fi

GPU_CARDS_ARG=$1

# Conda setup
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

# Unset proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# NPU/Ascend environment setup (if available)
if command -v npu-smi > /dev/null 2>&1; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export ASCEND_CUSTOM_OPP_PATH=/data/MindIE-SD/build/pkg/vendors/customize:/data/MindIE-SD/build/pkg/vendors/aie_ascendc:
    export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:${ASCEND_TOOLKIT_HOME}/lib64:$LD_LIBRARY_PATH
    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
    export XFORMERS_FORCE_DISABLE_TRITON=1
    [ -n "${GPU_CARDS_ARG}" ] && export ASCEND_RT_VISIBLE_DEVICES=${GPU_CARDS_ARG}
fi
