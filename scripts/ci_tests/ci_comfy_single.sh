#!/bin/bash
set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1
GIT_REPO_HEAD_COMMIT_ID=$1
GPU_CARDS=$2
COMFYUI_PORT=$3
if [ -z "${BK_CI_GIT_REPO_HEAD_COMMIT_ID}" ]; then
    echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID is empty"
    exit 1
fi
if [ -z "${GPU_CARDS}" ]; then
    GPU_CARDS=0
fi
if [ -z "${COMFYUI_PORT}" ]; then
    COMFYUI_PORT=8188
fi
echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "
echo "USE GPUS ${GPU_CARDS}"
echo "USE COMFYUI_PORT ${COMFYUI_PORT}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd /data/ComfyUI/custom_nodes/KsanaDiT/tests/comfy
python workflow_test.py --workflows-file ./test_configs.json --gpus ${GPU_CARDS} --seed 321 --port ${COMFYUI_PORT}
