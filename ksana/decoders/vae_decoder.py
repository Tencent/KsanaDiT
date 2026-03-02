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

from ..models.model_key import KsanaModelKey
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import log, time_range


@KsanaUnitFactory.register(
    KsanaUnitType.DECODER, [KsanaModelKey.VAE_WAN2_2, KsanaModelKey.VAE_WAN2_1, KsanaModelKey.QwenImageVAE]
)
class KsanaVaeDecoder(KsanaRunnerUnit):
    @time_range
    def run(
        self,
        vae_model,
        *,
        latents,
        local_rank,
        device=None,
        offload_model=False,  # Note: add offload choice to encoder too?
        offload_device=None,
        with_end_image: bool = False,
    ):
        if vae_model.device != device:
            vae_model.to(device)

        # TODO(qiannan): 位置: vae_encode_image() 中 image.sub(0.5).div(0.5)
        # 和 _comfy_process_output() 中 (image + 1.0) / 2.0
        # ●原因: 归一化（[0,1] → [-1,1]）放在节点层而非模型层，需要与 encode 端保持同步。如果改动 encode 的归一化位置，
        # decode 的反归一化也必须同步修改。

        # TODO(qiannan): save_image() 的 [-1,1] → [0,1] 归一化
        # ●原因: 与 save_video() 保持一致的语义——都假设输入在 [-1,1] 范围，内部做 (x + 1) / 2 转换到 [0,1]。
        # 建议： image.sub(0.5).div(0.5) 这种处理 似乎不应该放在vae.py 节点层或者save image里面，
        # 后续统一重构时，将归一化逻辑下沉到 VAE 模型内部。
        outputs = vae_model.forward_decode(
            latents=latents, local_rank=local_rank, device=device, with_end_image=with_end_image
        )

        if offload_model and offload_device is not None:
            vae_model.to(offload_device)

        log.info(f"decoder output shape: {outputs.shape}")
        return outputs
