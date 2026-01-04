from functools import partial

from .backends import KsanaAttentionBackend
from .attention_op import KsanaAttentionOp


def pick_attn_op(backend: KsanaAttentionBackend):
    return partial(KsanaAttentionOp, attn_backend=backend)
