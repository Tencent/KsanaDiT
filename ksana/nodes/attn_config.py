from ksana.config import KsanaAttentionBackend, KsanaAttentionConfig, KsanaRadialSageAttentionConfig


def attention_config(backend=None):
    if backend is None:
        backend = KsanaAttentionBackend.FLASH_ATTN.value
    return KsanaAttentionConfig(backend=backend)


def radial_sage_attention_config(
    dense_blocks_num=1,
    dense_attn_steps=1,
    decay_factor=0.02,
    block_size=128,
    dense_backend=None,
):
    """
    Create a KsanaRadialSageAttentionConfig object with the specified parameters.

    Args:
        dense_blocks_num: Number of dense attention blocks
        dense_attn_steps: Number of steps to use dense attention
        decay_factor: Decay factor for radial pattern
        block_size: Block size for sparse computation
        dense_backend: Backend for dense attention
    Returns:
        configuration object
    """
    if dense_backend is None:
        dense_backend = KsanaAttentionBackend.SAGE_ATTN.value
    return KsanaRadialSageAttentionConfig(
        dense_blocks_num=dense_blocks_num,
        dense_attn_steps=dense_attn_steps,
        decay_factor=decay_factor,
        block_size=block_size,
        dense_attention_config=KsanaAttentionConfig(backend=dense_backend),
    )
