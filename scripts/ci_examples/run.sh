#!/bin/bash

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU_CARDS}

# Only set Ascend RT visible devices for NPU
if which npu-smi > /dev/null 2>&1; then
    export ASCEND_RT_VISIBLE_DEVICES=${GPU_CARDS}
fi

cd /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/
python examples/${EXAMPLES}.py
