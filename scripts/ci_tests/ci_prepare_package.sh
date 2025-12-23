#!/bin/bash
set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1

# sudo su - mqq
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "
sudo chown mqq:mqq -R /ci_workspace/
sudo chown mqq:mqq -R /data/ComfyUI/
# sudo chown mqq:mqq -R /root/.cache/

rm -rf /data/ComfyUI/custom_nodes/KsanaDiT
mkdir -p /data/ComfyUI/custom_nodes/

cd /data/ComfyUI/custom_nodes/
tar xf /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}.tgz -C /data/ComfyUI/custom_nodes/
mv ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} KsanaDiT
sudo chown mqq:mqq -R KsanaDiT/

cd /data/ComfyUI/custom_nodes/KsanaDiT
./scripts/install_dev.sh

ln -sf /dockerdata/ci-models/single/comfy_models .
ln -sf /dockerdata/ci-models/single/Wan2.2-Lightning .
ln -sf /dockerdata/ci-models/single/Wan2.2-T2V-A14B .
ln -sf /dockerdata/ci-models/single/Wan2.2-I2V-A14B .


mkdir -p /data/stable-diffusion-webui
ln -sf /dockerdata/ci-models/single/comfy_models /data/stable-diffusion-webui/models

sudo pkill -f comfy
sudo pkill -f ksana
