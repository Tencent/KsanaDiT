#!/bin/bash

set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1
echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "

# sudo su - mqq
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

export CUDA_VISIBLE_DEVICES=0,1
cd /data/ComfyUI/custom_nodes/KsanaDiT
pytest -s -v ksana/tests/models/gpus/
