#!/bin/bash
set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1

# sudo su - mqq
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "
sudo chown mqq:mqq -R /ci_workspace/
sudo chown mqq:mqq -R /data/ComfyUI/

rm -rf /data/ComfyUI/custom_nodes/ComfyUI_KsanaDiT
rm -rf /data/ComfyUI/custom_nodes/KsanaDiT

cd /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/
python -m build --wheel
pip uninstall -y ksana
pip install dist/ksana*

mkdir -p /data/ComfyUI/custom_nodes/ComfyUI_KsanaDiT
cp -r /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/comfyui/* /data/ComfyUI/custom_nodes/ComfyUI_KsanaDiT/

ln -sf /dockerdata/ci-models/single/comfy_models .
ln -sf /dockerdata/ci-models/single/Wan2.2-Lightning .
ln -sf /dockerdata/ci-models/single/Wan2.2-T2V-A14B .
ln -sf /dockerdata/ci-models/single/Wan2.2-I2V-A14B .

mkdir -p /data/stable-diffusion-webui
ln -sf /dockerdata/ci-models/single/comfy_models /data/stable-diffusion-webui/models

sudo pkill -f comfy
sudo pkill -f ksana
