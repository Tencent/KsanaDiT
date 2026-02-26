#!/bin/bash
# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xe

BK_CI_GIT_REPO_HEAD_COMMIT_ID=$1

# sudo su - mqq
source /data/miniconda3/etc/profile.d/conda.sh
conda activate env-novelai

echo "BK_CI_GIT_REPO_HEAD_COMMIT_ID: ${BK_CI_GIT_REPO_HEAD_COMMIT_ID} "

sudo pkill -9 -f comfy 2>/dev/null || true
sudo pkill -9 -f ksana 2>/dev/null || true
sudo pkill -9 -f pytest 2>/dev/null || true
sudo pkill -9 -f workflow_test 2>/dev/null || true
ray stop --force 2>/dev/null || true
sudo pkill -9 -u mqq -f "ray::" 2>/dev/null || true
sudo pkill -9 -u mqq -f raylet 2>/dev/null || true
sudo pkill -9 -u mqq -f gcs_server 2>/dev/null || true
sleep 5
rm -rf /tmp/ray /tmp/ray_ksana /tmp/ray_comfy 2>/dev/null || true
if command -v npu-smi > /dev/null 2>&1; then
    rm -rf /dev/shm/hccl_* 2>/dev/null || true
fi

sudo chown mqq:mqq -R /ci_workspace/
sudo chown mqq:mqq -R /data/ComfyUI/

rm -rf /data/ComfyUI/custom_nodes/ComfyUI_KsanaDiT
rm -rf /data/ComfyUI/custom_nodes/KsanaDiT

cd /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/
python -m build --wheel
pip uninstall -y ksana-dit
pip install dist/ksana*

mkdir -p /data/ComfyUI/custom_nodes/ComfyUI_KsanaDiT
cp -r /ci_workspace/${BK_CI_GIT_REPO_HEAD_COMMIT_ID}/comfyui/* /data/ComfyUI/custom_nodes/ComfyUI_KsanaDiT/

ln -sf /dockerdata/ci-models/single/comfy_models .
ln -sf /dockerdata/ci-models/single/Wan2.2-Lightning .
ln -sf /dockerdata/ci-models/single/Wan2.2-T2V-A14B .
ln -sf /dockerdata/ci-models/single/Wan2.2-I2V-A14B .
ln -sf /dockerdata/ci-models/single/Qwen-Image .
ln -sf /dockerdata/ci-models/single/Wan2.1-VACE-14B .
ln -sf /dockerdata/ci-models/single/TurboWan2.2-I2V-A14B-720P .


mkdir -p /data/stable-diffusion-webui
ln -sf /dockerdata/ci-models/single/comfy_models /data/stable-diffusion-webui/models
