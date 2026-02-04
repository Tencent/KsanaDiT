import os
from unittest import SkipTest

import torch

from ksana.accelerator import platform

PROMPTS = [
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
]

SEED = 123
TEST_DTYPE = torch.float16

TEST_SIZE = (720, 480)
TEST_STEPS = 1
TEST_FRAME_NUM = 9
TEST_EPS_PLACE = 3

RADIAL_ATTN_TEST_SIZE = (1280, 768)  # should be divisible by block_size
RADIAL_ATTN_TEST_FRAME_NUM = 33

TEST_PORT = int(os.environ.get("KSANA_TEST_PORT", 29500))

CURRENT_PLATFORM = "NPU" if platform.is_npu() else "GPU"


def get_platform_config_or_skip(config_map: dict, *, test_name: str = ""):
    normalized = {str(key).upper(): value for key, value in (config_map or {}).items()}
    config = normalized.get(CURRENT_PLATFORM)
    if config is None:
        raise SkipTest(f"{test_name} skipped on {CURRENT_PLATFORM}: no config defined")
    return config
