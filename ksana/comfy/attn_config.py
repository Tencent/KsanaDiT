from ksana.config import KsanaAttentionConfig
from ksana.operations.attention import KsanaAttentionBackend


def attention_config(backend=None):
    if backend is None:
        backend = KsanaAttentionBackend.FLASH_ATTN.value
    return KsanaAttentionConfig(backend=backend)
