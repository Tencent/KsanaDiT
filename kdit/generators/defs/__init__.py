# Copyright 2025 Tencent

# 导入所有 def 文件以触发 register_generator_def() 调用
from . import (
    qwen_edit,  # noqa: F401
    qwen_t2i,  # noqa: F401
    wan_i2v,  # noqa: F401
    wan_t2v,  # noqa: F401
    wan_vace,  # noqa: F401
)
