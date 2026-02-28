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
#
# Supports both CI (Docker) and local development environments.
# Override defaults via environment variables:
#   CONDA_SH_PATH    - path to conda.sh (default: /data/miniconda3/etc/profile.d/conda.sh)
#   CONDA_ENV_NAME   - conda environment name (default: env-novelai)
#   KSANA_CPU_MEM_LIMIT_GB - CPU memory limit in GB (default: 160, set 0 to disable)

# --- CPU memory limit (Linux/Docker only) ---
KSANA_CPU_MEM_LIMIT_GB=${KSANA_CPU_MEM_LIMIT_GB:-160}
export KSANA_CPU_MEM_LIMIT_GB

if [ "$(uname -s)" = "Linux" ] && [ "${KSANA_CPU_MEM_LIMIT_GB}" -gt 0 ] 2>/dev/null; then
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
else
    if [ "$(uname -s)" != "Linux" ]; then
        echo "INFO: non-Linux platform ($(uname -s)), skipping cgroup memory limit"
    fi
fi

GPU_CARDS_ARG=$1

# --- Conda setup (skip if already in the target environment) ---
CONDA_SH_PATH="${CONDA_SH_PATH:-/data/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env-novelai}"

if [ -n "${CONDA_DEFAULT_ENV}" ] && [ "${CONDA_DEFAULT_ENV}" = "${CONDA_ENV_NAME}" ]; then
    echo "INFO: already in conda environment '${CONDA_ENV_NAME}', skipping activation"
elif [ -f "${CONDA_SH_PATH}" ]; then
    source "${CONDA_SH_PATH}"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "WARNING: conda.sh not found at '${CONDA_SH_PATH}', skipping conda activation"
    echo "  Set CONDA_SH_PATH to your conda.sh path, or activate your environment manually before running"
fi

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
