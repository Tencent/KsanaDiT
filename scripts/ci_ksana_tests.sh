#!/bin/bash

set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1

sudo su - mqq
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "
sudo chown mqq:mqq -R /ci_workspace/
cd /ci_workspace

rm -rf /data/ComfyUI/custom_nodes/KsanaDiT
mkdir -p /data/ComfyUI/custom_nodes/
cd /data/ComfyUI/custom_nodes/

tar xf /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}.tgz -C /data/ComfyUI/custom_nodes/
mv ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} KsanaDiT
sudo chown mqq:mqq -R KsanaDiT/

cd /data/ComfyUI/custom_nodes/KsanaDiT
ln -sf /dockerdata/ci-models/comfy_models /data/ComfyUI/custom_nodes/KsanaDiT/
ln -sf /dockerdata/ci-models/Wan2.2-Lightning /data/ComfyUI/custom_nodes/KsanaDiT/
ln -sf /dockerdata/ci-models/Wan2.2-T2V-A14B /data/ComfyUI/custom_nodes/KsanaDiT/

./scripts/install_dev.sh
pkill -f pytest
pkill -f workflow_test
pkill -f raylet
export CUDA_VISIBLE_DEVICES=0,1
pytest -s -v ksana/tests/
