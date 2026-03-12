"""
Qwen-Image-Edit 图像编辑示例

使用示例:
    # 不使用 LoRA
    python examples/qwen/qwen_image_edit.py --model_dir /path/to/Qwen-Image-Edit

    # 使用 LoRA
    python examples/qwen/qwen_image_edit.py --model_dir /path/to/Qwen-Image-Edit --lora_dir /path/to/lora.safetensors
"""

import argparse
import os

import torch

os.environ["KSANA_LOGGER_LEVEL"] = "INFO"

from ksana import KsanaPipeline
from ksana.config import (
    KsanaLoraConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)
from ksana.models.model_key import KsanaModelKey
from ksana.utils.media import save_image

# 生成配置
config = {
    "img_path": ["examples/images/woman.png", "examples/images/man.png"],
    "prompt": "the woman and man are hugging together",
    "negative_prompt": "blur, bad anatomy, deformed face",
    "seed": 321,
    "size": (1024, 1024),
    # 普通模式
    "steps": 40,
    "cfg_scale": 4.0,
    # LoRA 模式
    "lora_steps": 4,
    "lora_cfg_scale": 1.0,
}


def run_edit(model_dir: str, lora_dir: str = None):
    """运行 Qwen-Image-Edit 图像编辑"""

    # LoRA 配置
    lora_config = KsanaLoraConfig(path=lora_dir, strength=1.0) if lora_dir else None

    # 加载模型
    pipeline = KsanaPipeline.from_models(
        model_dir,
        model_config=KsanaModelConfig(run_dtype=torch.bfloat16),
        pipeline_key=KsanaModelKey.QwenImage_Edit,
        lora_config=lora_config,
        offload_device="cpu",
    )

    # 采样参数
    steps = config["lora_steps"] if lora_dir else config["steps"]
    cfg_scale = config["lora_cfg_scale"] if lora_dir else config["cfg_scale"]

    print(f"LoRA: {lora_dir or '无'}")
    print(f"参考图像: {config['img_path']}")
    print(f"提示词: {config['prompt']}")
    print(f"采样步数: {steps}, CFG: {cfg_scale}")

    # 生成
    image = pipeline.generate(
        config["prompt"],
        prompt_negative=config["negative_prompt"],
        img_path=config["img_path"],
        sample_config=KsanaSampleConfig(
            steps=steps,
            cfg_scale=cfg_scale,
            solver=KsanaSolverType.FLOWMATCH_EULER,
        ),
        runtime_config=KsanaRuntimeConfig(
            seed=config["seed"],
            size=config["size"],
            return_frames=True,
        ),
    )

    # 保存
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/qwen_image_edit_seed{config['seed']}.png"
    save_image(image, output_path)
    print(f"输出: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit 图像编辑示例")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录")
    parser.add_argument("--lora_dir", type=str, default=None, help="LoRA 路径")
    args = parser.parse_args()
    run_edit(args.model_dir, args.lora_dir)
