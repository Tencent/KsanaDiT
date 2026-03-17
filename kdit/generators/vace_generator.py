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

import torch

from kdit.config import SolverType
from kdit.config.wan_experimental_config import KsanaExperimentalConfig, KsanaFETAConfig, KsanaSLGConfig
from kdit.models import ModelKey
from kdit.utils.vace import (
    apply_bidirectional_sampling,
    apply_experimental_cfg,
    apply_temporal_score_rescaling,
    apply_vace_trim,
    build_vace_kwargs,
    get_step_video_control,
    parse_video_control_kwargs,
)

from .generator_factory import GeneratorFactory
from .wan_generator import WanGenerator


@GeneratorFactory.register(ModelKey.Wan2_1_VACE_14B)
class VaceGenerator(WanGenerator):
    def init_denoising_loop(self, video_control_kwargs, diffusion_model, sample_scheduler):
        return parse_video_control_kwargs(
            video_control_kwargs,
            diffusion_model,
            sample_scheduler,
            slg_config_cls=KsanaSLGConfig,
            feta_config_cls=KsanaFETAConfig,
            experimental_config_cls=KsanaExperimentalConfig,
        )

    def get_step_kwargs(self, denoise_video_control_args, current_step_percent, iter_id, total_steps):
        return get_step_video_control(
            denoise_video_control_args,
            current_step_percent,
            iter_id,
            total_steps,
            slg_config_cls=KsanaSLGConfig,
            feta_config_cls=KsanaFETAConfig,
        )

    def apply_cfg(
        self,
        cfg_scale,
        cond,
        uncond,
        *,
        denoise_video_control_args=None,
        experimental_config=None,
        step_index=0,
        total_steps=1,
        **kwargs,
    ):
        exp_config = experimental_config or (
            denoise_video_control_args.get("exp_config") if denoise_video_control_args else None
        )
        if exp_config is None:
            return super().apply_cfg(cfg_scale, cond, uncond)
        return apply_experimental_cfg(cfg_scale, cond, uncond, exp_config, step_index)

    def post_noise_prediction(self, noise_pred, noise_latent, t, denoise_video_control_args):
        return apply_temporal_score_rescaling(noise_pred, noise_latent, t, denoise_video_control_args.get("exp_config"))

    def finalize_step(self, noise_latent, noise_latent_forward, denoise_video_control_args, **step_state):
        if not (denoise_video_control_args.get("bidirectional_sampling") and noise_latent.ndim == 5):
            return noise_latent_forward
        return apply_bidirectional_sampling(
            noise_latent=noise_latent,
            noise_latent_forward=noise_latent_forward,
            running_model=step_state["running_model"],
            running_cfg_scale=step_state["running_cfg_scale"],
            timestep=step_state["timestep"],
            t=step_state["t"],
            iter_id=step_state["iter_id"],
            total_steps=step_state["total_steps"],
            current_step_percent=step_state["current_step_percent"],
            combine_cond_uncond=step_state["combine_cond_uncond"],
            positive=step_state["positive"],
            negative=step_state["negative"],
            image_embeds=step_state["image_embeds"],
            step_vc=step_state["step_kwargs"],
            exp_config=denoise_video_control_args["exp_config"],
            sample_scheduler_flipped=denoise_video_control_args["sample_scheduler_flipped"],
            sample_config=step_state["sample_config"],
            seed_g=step_state["seed_g"],
            prepare_model_forward_kargs_fn=self.prepare_model_forward_kargs,
            use_cfg_fn=self._use_cfg,
            apply_cfg_fn=self.apply_cfg,
            solver_type_euler=SolverType.EULER,
        )

    def build_video_control_kwargs(self, **kwargs):
        return build_vace_kwargs(**kwargs)

    def post_run(self, noise_latents, video_control_kwargs):
        return apply_vace_trim(noise_latents, video_control_kwargs.get("trim_latent", 0))

    def prepare_model_forward_kargs(
        self,
        cfg_scale: float,
        *,
        noise_latent,
        timestep,
        combine_cond_uncond,
        step_iter,
        cache,
        positive,
        negative,
        image_embeds,
        vace_context=None,
        vace_context_scale=1.0,
        slg_config=None,
        feta_config=None,
        current_step_percent=0.0,
        **_,
    ) -> dict:
        base = {"cache": cache, "step_iter": step_iter}

        if slg_config is not None:
            base["slg_config"] = slg_config
        if feta_config is not None:
            base["feta_config"] = feta_config
        base["current_step_percent"] = current_step_percent

        if vace_context is not None:
            base["vace_context"] = vace_context
            base["vace_context_scale"] = vace_context_scale

        use_cfg = self._use_cfg(cfg_scale)
        if use_cfg and combine_cond_uncond:
            combine_x = torch.cat([noise_latent, noise_latent], dim=0)
            combine_t = torch.cat([timestep, timestep], dim=0)
            combine_context = torch.cat([positive, negative], dim=0)
            combine_kargs = {
                "phase": "combine",
                "x": combine_x,
                "t": combine_t,
                "context": combine_context,
            }
            if vace_context is not None:
                combine_kargs["vace_context"] = vace_context + vace_context
            return base | combine_kargs

        base.update({"x": noise_latent, "t": timestep})
        arg_cond = {"phase": "cond", "context": positive}
        arg_uncond = {"phase": "uncond", "context": negative}
        if use_cfg:
            return base | arg_cond, base | arg_uncond
        else:
            return base | arg_cond
