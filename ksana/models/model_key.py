from enum import Enum, auto, unique


@unique
class KsanaModelKey(Enum):
    T5TextEncoder = auto()
    VAE = auto()
    Wan2_2_HIGH = auto()
    Wan2_2_LOW = auto()
