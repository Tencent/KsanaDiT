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

"""DenoiseHandler — 去噪循环钩子的默认实现。

去噪相关方法，子类可覆写以定制行为。
"""

import torch

from kdit.models.diffusion_model import KsanaDiffusionModel


class DenoiseHandler:
    """去噪循环中各阶段的钩子方法。"""

    def get_running_model(
        self,
        diffusion_model: list[KsanaDiffusionModel],
        *,
        timestep_id: int,
        device: torch.device,
        offload_device: torch.device,
    ):
        """从 diffusion_model 列表中选取当前 timestep 使用的模型。"""
        return diffusion_model[0] if isinstance(diffusion_model, (list, tuple)) else diffusion_model

    def get_running_cache(self, dit_cache: list, *, timestep_id: int):
        """从 dit_cache 列表中选取当前 timestep 使用的 cache。"""
        return dit_cache[0] if isinstance(dit_cache, (list, tuple)) else dit_cache

    def get_running_cfg_scale(self, cfg_scale: list[float], *, timestep_id: int):
        """从 cfg_scale 列表中选取当前 timestep 使用的 cfg_scale。"""
        return cfg_scale[0] if isinstance(cfg_scale, (list, tuple)) else cfg_scale

    def prepare_model_forward_kargs(self, cfg_scale, **kwargs) -> dict | tuple[dict, dict]:
        """构建模型 forward 参数，子类必须实现。"""
        raise NotImplementedError("prepare_model_forward_kargs must be implemented in subclass")

    def apply_cfg(self, cfg_scale, cond, uncond, **kwargs):
        """应用 classifier-free guidance。"""
        return uncond + float(cfg_scale) * (cond - uncond)

    def init_denoising_loop(self, video_control_kwargs, diffusion_model, sample_scheduler):
        """去噪循环开始前的初始化，返回额外状态 dict。"""
        return {}

    def get_step_kwargs(self, denoise_video_control_args, current_step_percent, iter_id, total_steps):
        """获取当前去噪步的额外参数。"""
        return {}

    def post_noise_prediction(self, noise_pred, noise_latent, t, denoise_video_control_args):
        """噪声预测后的后处理。"""
        return noise_pred

    def finalize_step(self, noise_latent, noise_latent_forward, denoise_video_control_args, **step_state):
        """单步去噪完成后的收尾处理。"""
        return noise_latent_forward

    def build_video_control_kwargs(self, **kwargs):
        """构建 video control 参数 dict。"""
        return {}

    def post_run(self, noise_latents, video_control_kwargs):
        """整个去噪循环结束后的后处理。"""
        return noise_latents
