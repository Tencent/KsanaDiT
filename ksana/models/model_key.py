from enum import Enum, auto, unique

WAN2_2 = ["wan2.2", "wan22", "wan2_2", "wan_2_2", "wan_2.2"]
WAN2_1 = ["wan2.1", "wan21", "wan2_1", "wan_2_1", "wan_2.1"]

# TODO: support "s2v", "ti2v"
X2V_TYPES = ["t2v", "i2v"]

# model type means: wan2.1, wan2.2, qwen
# model key means: combine of "{model_type}_{task_type}_{model_size}"


@unique
class KsanaModelKey(Enum):
    """
    暴露在外部comfy层面的模型key, 与model type的区别在于
    用于在model pool中查找model本身
    """

    T5TextEncoder = auto()
    VAE_WAN2_1 = auto()
    VAE_WAN2_2 = auto()
    Wan2_2_T2V_14B_HIGH = auto()
    Wan2_2_T2V_14B_LOW = auto()
    Wan2_2_I2V_14B_HIGH = auto()
    Wan2_2_I2V_14B_LOW = auto()
