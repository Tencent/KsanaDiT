#!/bin/bash
set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1

echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID}"

source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

sudo pkill -f comfyui
cd /data/ComfyUI/custom_nodes/KsanaDiT/comfyui/tests
python workflow_test.py --workflows-file ./test_configs.json --gpus 0,1 --seed 321
