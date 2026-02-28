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

# Run ksana single-GPU tests.
#
# CI usage (unchanged):
#   bash ci_ksana_single.sh <commit_id> [gpu_cards]
#
# Local usage:
#   bash ci_ksana_single.sh              # auto-detect project root, GPU=1
#   bash ci_ksana_single.sh "" 0         # auto-detect project root, GPU=0
#   KSANA_CI_MODELS_ROOT=/my/models bash ci_ksana_single.sh

set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1
GPU_CARDS=$2

# --- Determine working directory ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${BK_CI_GIT_REPO_HEAD_COMMIT_ID}" ]; then
    # CI mode: use /ci_workspace/<commit_id>
    WORK_DIR="/ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}"
    echo "CI mode: commit=${BK_CI_GIT_REPO_HEAD_COMMIT_ID}"
else
    # Local mode: project root is two levels up from scripts/ci_tests/
    WORK_DIR="${KSANA_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
    echo "Local mode: project root=${WORK_DIR}"
fi

if [ -z "${GPU_CARDS}" ]; then
    GPU_CARDS=1
fi
echo "USE GPUS ${GPU_CARDS}"

source "${SCRIPT_DIR}/test_env.sh" ${GPU_CARDS}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU_CARDS}
export KSANA_CI_MODELS_ROOT=${KSANA_CI_MODELS_ROOT:-/dockerdata/ci-models}

cd "${WORK_DIR}"
pytest -s -v tests/ksana
