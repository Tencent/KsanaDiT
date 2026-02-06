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

sudo pkill -f comfy
sudo pkill -f ksana
