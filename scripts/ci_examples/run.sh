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

set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1
GPU_CARDS=$2
EXAMPLES=${3:-"wan/wan2_2_t2v"}

if [ -z "${BK_CI_GIT_REPO_HEAD_COMMIT_ID}" ]; then
    echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID is empty"
    exit 1
fi
if [ -z "${GPU_CARDS}" ]; then
    GPU_CARDS=1
fi
echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "
echo "USE GPUS ${GPU_CARDS}"
echo "TEST EXAMPLES ${EXAMPLES}"

gpu_count=$(echo "$GPU_CARDS" | tr ',' '\n' | wc -l)
echo "GPU COUNT ${gpu_count}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU_CARDS}

# Only set Ascend RT visible devices for NPU
if which npu-smi > /dev/null 2>&1; then
    export ASCEND_RT_VISIBLE_DEVICES=${GPU_CARDS}
fi

cd /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/
python examples/${EXAMPLES}.py
