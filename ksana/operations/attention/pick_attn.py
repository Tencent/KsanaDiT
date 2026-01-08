from functools import partial

from ksana.config import KsanaAttentionConfig

from .attention_op import KsanaAttentionOp


def pick_attn_op(attention_config: KsanaAttentionConfig):
    return partial(KsanaAttentionOp, attention_config=attention_config)
