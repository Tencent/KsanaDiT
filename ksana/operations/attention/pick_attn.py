from functools import partial

from .attention_op import KsanaAttentionOp


def pick_attn_op(attention_config=None):
    return partial(KsanaAttentionOp, attention_config=attention_config)
