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

import os
import unittest
from dataclasses import dataclass

from comfy_nodes.nodes_test_helper import (
    COMFY_MODEL_DIFFUSION_ROOT,
    IMG_SHAPE_I2V,
    IMG_SHAPE_T2I,
    IMG_SHAPE_T2V,
    QWEN_TEXT_SHAPE,
    TEST_GPUS_EPS_PLACE,
    TEST_ONE_GPU_EPS_PLACE,
    WAN_TEXT_SHAPE,
    run_load_and_generate,
)
from platform_test_helper import get_platform_expected_or_skip

from kdit import get_engine
from kdit.config import KsanaAttentionBackend, KsanaLinearBackend
from kdit.models.model_key import ModelKey
from kdit.utils.distribute import get_gpu_count, get_rank_id

TEST_STEPS = 1


@dataclass
class KsanaNodesTestCase:
    model_names: list[str]
    image_latent_shape: list[int]
    attention_backends: KsanaAttentionBackend | None
    linear_backends: KsanaLinearBackend
    rope_function: str
    expect_model_key: ModelKey
    mean_config: dict


test_cases = [
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        image_latent_shape=IMG_SHAPE_T2V,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.DEFAULT,
        rope_function="comfy",
        expect_model_key=ModelKey.Wan2_2_T2V_14B,
        mean_config={"GPU": {"single_mean": 0.76318359375, "multi_mean": 0.76318359375}},
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        image_latent_shape=IMG_SHAPE_I2V,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.DEFAULT,
        rope_function="default",
        expect_model_key=ModelKey.Wan2_2_I2V_14B,
        mean_config={"GPU": {"single_mean": 0.7939453125, "multi_mean": 0.7939453125}},
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", None],
        expect_model_key=ModelKey.Wan2_2_T2V_14B,
        image_latent_shape=IMG_SHAPE_T2V,
        attention_backends=None,
        linear_backends=KsanaLinearBackend.FP8_GEMM,
        rope_function="comfy",
        mean_config={"GPU": {"single_mean": 0.77978515625, "multi_mean": 0.77978515625}},
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_t2v_high_noise_14B_fp16.safetensors",
            "wan2.2_t2v_low_noise_14B_fp16.safetensors",
        ],
        expect_model_key=ModelKey.Wan2_2_T2V_14B,
        image_latent_shape=IMG_SHAPE_T2V,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM_DYNAMIC,
        rope_function="default",
        mean_config={"GPU": {"single_mean": 0.759765625, "multi_mean": 0.759765625}},
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_i2v_high_noise_14B_fp16.safetensors",
            "wan2.2_i2v_low_noise_14B_fp16.safetensors",
        ],
        expect_model_key=ModelKey.Wan2_2_I2V_14B,
        image_latent_shape=IMG_SHAPE_I2V,
        attention_backends=None,
        linear_backends=KsanaLinearBackend.FP16_GEMM,
        rope_function="default",
        mean_config={
            "GPU": {"single_mean": 0.79052734375, "multi_mean": 0.79052734375},
            "NPU": {"single_mean": 0.79052734375, "multi_mean": 0.79052734375},
        },
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_i2v_high_noise_14B_fp16.safetensors", None],
        expect_model_key=ModelKey.Wan2_2_I2V_14B,
        image_latent_shape=IMG_SHAPE_I2V,
        attention_backends=None,
        linear_backends=KsanaLinearBackend.FP8_GEMM_DYNAMIC,
        rope_function="comfy",
        mean_config={
            "GPU": {"single_mean": 0.79052734375, "multi_mean": 0.79052734375},
            "NPU": {"single_mean": 0.79052734375, "multi_mean": 0.79052734375},
        },
    ),
    KsanaNodesTestCase(
        model_names="qwen_image_2512_fp8_e4m3fn.safetensors",
        expect_model_key=ModelKey.QwenImage_T2I,
        image_latent_shape=IMG_SHAPE_T2I,
        attention_backends=None,
        linear_backends=KsanaLinearBackend.FP8_GEMM,
        rope_function="comfy",
        mean_config={"GPU": {"single_mean": 0.283203125, "multi_mean": 0.283203125}},
    ),
]


class TestModelSwitchAndGenerate(unittest.TestCase):
    def test_base_and_swith_models(self):
        print("-----------------test_swith_models_and_generate-----------------")

        for test_case in test_cases:
            case_config = get_platform_expected_or_skip(test_case.mean_config)
            print(f"----------- test model_name: {test_case.model_names} -------------")
            if test_case.expect_model_key in [ModelKey.Wan2_2_I2V_14B, ModelKey.Wan2_2_T2V_14B]:
                high_noise_model_path = os.path.join(COMFY_MODEL_DIFFUSION_ROOT, test_case.model_names[0])
                low_noise_model_path = (
                    os.path.join(COMFY_MODEL_DIFFUSION_ROOT, test_case.model_names[1])
                    if test_case.model_names[1]
                    else None
                )

            else:
                high_noise_model_path = os.path.join(COMFY_MODEL_DIFFUSION_ROOT, test_case.model_names)
                low_noise_model_path = None
            if test_case.expect_model_key in [ModelKey.QwenImage_T2I]:
                text_shape = QWEN_TEXT_SHAPE
            else:
                text_shape = WAN_TEXT_SHAPE

            load_output, generate_output = run_load_and_generate(
                high_noise_model_path,
                test_case.image_latent_shape,
                text_shape,
                TEST_STEPS,
                model_boundary=0.5,
                attn_backend=test_case.attention_backends,
                linear_backend=test_case.linear_backends,
                low_noise_model_path=low_noise_model_path,
                rope_function=test_case.rope_function,
                low_sample_guide_scale=3.0,
            )
            self.assertEqual(load_output.model, test_case.expect_model_key)
            latent_key = generate_output.samples
            tensor_value = get_engine().get_tensor(latent_key)
            latent_tensor = tensor_value.data if tensor_value is not None else None
            if get_rank_id() == 0:
                # only return tensor on rank 0
                self.assertIsNotNone(latent_tensor)
            else:
                self.assertIsNone(latent_tensor)
                continue

            target_latent_shape = test_case.image_latent_shape.copy()
            target_latent_shape[1] = 16  # always 16
            self.assertEqual(list(latent_tensor.shape), target_latent_shape)
            mean = latent_tensor.cpu().abs().mean().item()
            if get_gpu_count() == 1:
                self.assertAlmostEqual(mean, case_config["single_mean"], places=TEST_ONE_GPU_EPS_PLACE)
            else:
                expected_multi = case_config.get("multi_mean", case_config["single_mean"])
                self.assertAlmostEqual(mean, expected_multi, places=TEST_GPUS_EPS_PLACE)

    # TODO: for all models, only one high, load once, and test belows for
    def test_cache(self):
        pass


if __name__ == "__main__":
    unittest.main()
