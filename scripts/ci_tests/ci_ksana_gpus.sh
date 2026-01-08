#!/bin/bash

set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1
GPU_CARDS=$2
KSANA_TEST_PORT=$3
if [ -z "${BK_CI_GIT_REPO_HEAD_COMMIT_ID}" ]; then
    echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID is empty"
    exit 1
fi
if [ -z "${GPU_CARDS}" ]; then
    GPU_CARDS=0,1
fi
if [ -z "${KSANA_TEST_PORT}" ]; then
    KSANA_TEST_PORT=29500
fi
echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "
echo "USE GPUS ${GPU_CARDS}"
echo "USE KSANA_TEST_PORT ${KSANA_TEST_PORT}"

# sudo su - mqq
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

export KSANA_TEST_PORT=${KSANA_TEST_PORT}
export CUDA_VISIBLE_DEVICES=${GPU_CARDS}
cd /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/
pytest -s -v tests/ksana/gpus/

# TODO: use torchrun way to run test
# torchrun --nproc_per_node=2 tests/ksana/gpus/nodes_test.py
