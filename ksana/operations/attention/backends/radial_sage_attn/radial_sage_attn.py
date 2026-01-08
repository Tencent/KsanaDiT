import torch

from ksana.config import KsanaAttentionBackend
from ksana.config.attention_config import KsanaRadialSageAttentionConfig
from ksana.operations.attention.backends.radial_sage_attn.attn_mask import MaskMap

from ..base import KsanaAttentionBackendImpl

try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda

    sparse_attn_func = block_sparse_sage2_attn_cuda
    _RADIAL_SAGE_AVAILABLE = True
except ModuleNotFoundError:
    _RADIAL_SAGE_AVAILABLE = False


class RadialSageAttentionImpl(KsanaAttentionBackendImpl):
    _global_cache = {}
    _max_cache_size = 16  # 限制缓存条目数，防止OOM

    @staticmethod
    def type() -> KsanaAttentionBackend:
        return KsanaAttentionBackend.RADIAL_SAGE_ATTN

    @staticmethod
    def supports(**_) -> bool:
        return _RADIAL_SAGE_AVAILABLE

    def __init__(
        self,
        attention_config: KsanaRadialSageAttentionConfig,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
    ) -> None:
        if not _RADIAL_SAGE_AVAILABLE:
            raise RuntimeError("RaidialSageAttention backend requested but 'spas_sage_attn' package is not installed. ")
        if causal:
            raise ValueError("RaidialSageAttention backend does not support causal attention")
        if not isinstance(attention_config, KsanaRadialSageAttentionConfig):
            raise ValueError("KsanaRadialSageAttentionConfig must be provided")
        self.attention_config = attention_config
        self.block_size = attention_config.block_size
        self.decay_factor = attention_config.decay_factor
        self.dense_blocks_num = attention_config.dense_blocks_num
        self.dense_attn_steps = attention_config.dense_attn_steps
        self.mask_map = None
        from ...attention_op import KsanaAttentionOp  # lazy import

        self.dense_attn = KsanaAttentionOp(
            num_heads,
            head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            attention_config=attention_config.dense_attention_config,
        )
        self.check_config()

    @torch.compiler.disable()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dense_only: bool = False,
        k_lens: torch.Tensor | None = None,
        step_iter: int = -1,
        block_id: int = -1,
        latent_shape: list[int] = {},
        **kwargs,
    ) -> torch.Tensor:
        if dense_only or step_iter < self.dense_attn_steps or block_id < self.dense_blocks_num:
            # dense attention
            return self.dense_attn(
                query,
                key,
                value,
                **kwargs,
            )
        else:
            # radial attention
            if not (k_lens == k_lens[0]).all().item():
                raise ValueError("k_lens must be the same for all tokens")
            return self._radial_sage_attn_forward(
                query,
                key,
                value,
                k_lens=k_lens,
                latent_shape=latent_shape,
            )

    @torch.compiler.disable()
    def _radial_sage_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_lens,
        latent_shape,
    ) -> torch.Tensor:
        if not (k_lens == k_lens[0]).all().item():
            raise ValueError("k_lens must be the same for all tokens")
        video_token_num = k_lens[0].item()
        _, _, num_frame, l_h, l_w = latent_shape
        if l_h % (self.block_size // 8) != 0 or l_w % (self.block_size // 8) != 0:
            raise ValueError(f"output's width and height must be divisible by block size {self.block_size}")
        # init mask map
        if (
            self.mask_map is None
            or MaskMap.create_signature(video_token_num, num_frame) != self.mask_map.get_signature()
        ):
            self.mask_map = MaskMap(
                video_token_num=video_token_num, num_frame=num_frame, block_size=self.block_size, device=query.device
            )

        block_size = self.mask_map.block_size
        cache_key = (
            str(query.shape),
            self.mask_map.block_size,
            self.decay_factor,
            self.mask_map.video_token_num,
            self.mask_map.num_frame,
            str(query.device),
        )
        if cache_key in RadialSageAttentionImpl._global_cache:
            input_mask = RadialSageAttentionImpl._global_cache[cache_key]
        else:
            bs = query.shape[0]
            video_mask = self.mask_map.queryLogMask(
                query.shape[1], "radial", block_size=block_size, decay_factor=self.decay_factor
            )

            # based on https://github.com/mit-han-lab/radial-attention/blob/3ec33ce9633adadadcbb7692c8a1983d5e82d15a/radial_attn/attn_mask.py#L7
            if block_size == 128:
                mask = torch.repeat_interleave(video_mask, 2, dim=1)
            elif block_size == 64:
                reshaped_mask = video_mask.view(video_mask.shape[0] // 2, 2, video_mask.shape[1])
                mask = torch.max(reshaped_mask, dim=1).values
            input_mask = mask.unsqueeze(0).unsqueeze(1).expand(bs, query.shape[-2], mask.shape[0], mask.shape[1])

            # 限制缓存大小，防止内存泄漏
            if len(RadialSageAttentionImpl._global_cache) >= RadialSageAttentionImpl._max_cache_size:
                # 删除最旧的条目（FIFO策略）
                oldest_key = next(iter(RadialSageAttentionImpl._global_cache))
                del RadialSageAttentionImpl._global_cache[oldest_key]

            RadialSageAttentionImpl._global_cache[cache_key] = input_mask

        return sparse_attn_func(
            query[:, :, : self.mask_map.video_token_num, :],
            key[:, :, : self.mask_map.video_token_num, :],
            value[:, :, : self.mask_map.video_token_num, :],
            mask_id=input_mask.to(torch.int8),
            tensor_layout="NHD",
        ).contiguous()
