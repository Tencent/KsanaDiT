#!/bin/bash
set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1
GPU_CARDS=$2
if [ -z "${BK_CI_GIT_REPO_HEAD_COMMIT_ID}" ]; then
    echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID is empty"
    exit 1
fi
if [ -z "${GPU_CARDS}" ]; then
    GPU_CARDS=1
fi
echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "
echo "USE GPUS ${GPU_CARDS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/test_env.sh" ${GPU_CARDS}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU_CARDS}
export KSANA_CI_MODELS_ROOT=/dockerdata/ci-models
cd /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/
pytest -s -v tests/ksana
