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

import argparse
import os

import torch

os.environ["KSANA_LOGGER_LEVEL"] = "INFO"

from ksana import KsanaPipeline
from ksana.config import (
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)
from ksana.utils.media import save_image

prompts = [
    "一只可爱的橘猫坐在窗台上，阳光透过窗户洒在它的毛发上，背景是模糊的花园景色",
]

SEED = 42


def run_simple(args):
    generator = KsanaPipeline.from_models(
        args.model_dir,
        model_config=KsanaModelConfig(run_dtype=torch.bfloat16),
        offload_device="cpu",
    )

    image = generator.generate(
        prompts[0] + ", Ultra HD, 4K, cinematic composition.",
        prompt_negative=" ",
        sample_config=KsanaSampleConfig(
            steps=20,
            cfg_scale=4.0,
            solver=KsanaSolverType.FLOWMATCH_EULER,
        ),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(1024, 1024),
        ),
    )

    os.makedirs("outputs", exist_ok=True)
    save_image(image, "outputs/qwen_image.png")
    print("图片已保存到: outputs/qwen_image.png")


if __name__ == "__main__":
    """使用示例:
    python examples/qwen/qwen_image_t2i.py --model_dir /path/to/qwen/model
    """
    parser = argparse.ArgumentParser(description="Qwen 图像生成示例")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./Qwen-Image",
        help="Qwen 模型目录路径",
    )

    args = parser.parse_args()
    run_simple(args)
