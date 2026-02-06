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

GPU_CARDS_ARG=$1

# Conda setup
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

# Unset proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# NPU/Ascend environment setup (if available)
if command -v npu-smi > /dev/null 2>&1; then
    export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:${ASCEND_TOOLKIT_HOME}/lib64:$LD_LIBRARY_PATH
    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
    export XFORMERS_FORCE_DISABLE_TRITON=1
    [ -n "${GPU_CARDS_ARG}" ] && export ASCEND_RT_VISIBLE_DEVICES=${GPU_CARDS_ARG}
fi
