from functools import partial

from .selector import AttentionBackendEnum
from .attention_op import LocalAttentionOp


def pick_attn_op(backend: AttentionBackendEnum = AttentionBackendEnum.FLASH_ATTN):
    return partial(LocalAttentionOp, attn_backend=backend)
