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

cd /ci_workspace
rm -rf /data/ComfyUI/custom_nodes/KsanaDiT
mkdir -p /data/ComfyUI/custom_nodes/
cd /data/ComfyUI/custom_nodes/

tar xf /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}.tgz -C /data/ComfyUI/custom_nodes/
mv ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} KsanaDiT
sudo chown mqq:mqq -R KsanaDiT/

cd /data/ComfyUI/custom_nodes/KsanaDiT

ln -sf /dockerdata/models/comfy_models /data/ComfyUI/custom_nodes/KsanaDiT/
ln -sf /dockerdata/models/Wan2.2-Lightning /data/ComfyUI/custom_nodes/KsanaDiT/
ln -sf /dockerdata/models/Wan2.2-T2V-A14B /data/ComfyUI/custom_nodes/KsanaDiT/

# mkdir -p /data/stable-diffusion-webui
# ln -s /dockerdata/models/comfy_models /data/stable-diffusion-webui/models

# pip install numpy==1.26.4
# pip install playwright && playwright install chromium

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

./scripts/install_dev.sh
cd comfyui/tests
export CUDA_VISIBLE_DEVICES=0
python workflow_test.py --workflows-file ./test_configs.json

# export CUDA_VISIBLE_DEVICES=0,1
# python workflow_test.py --workflows-file ./test_configs.json --num_gpus 2

#  python workflow_test.py --workflows-file ./test_configs.json --no-server
