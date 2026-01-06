import torch
import torch.nn as nn
from ...utils.logger import log

from .backends import KsanaAttentionBackend, KsanaAttentionBackendImpl, FlashAttentionImpl, SageAttentionImpl, SDPAImpl


_ATTN_BACKEND_TO_IMPL = {
    FlashAttentionImpl.type(): FlashAttentionImpl,
    SageAttentionImpl.type(): SageAttentionImpl,
    SDPAImpl.type(): SDPAImpl,
}


def _get_attention_backend_impl(attn_backend: KsanaAttentionBackend, **kwargs) -> KsanaAttentionBackendImpl:
    if not KsanaAttentionBackend.support(attn_backend):
        raise ValueError(
            f"attn_backend:{attn_backend} is not in supported_list:{ KsanaAttentionBackend.get_supported_list()}"
        )
    # input attn backend at first
    for backend_type in [attn_backend] + KsanaAttentionBackend.get_supported_list():
        backend_type = KsanaAttentionBackend(backend_type)
        backend_impl = _ATTN_BACKEND_TO_IMPL.get(backend_type, None)
        if backend_impl is None:
            raise ValueError(f"{backend_type} not in {_ATTN_BACKEND_TO_IMPL.keys()}")
        if backend_impl.supports(**kwargs):
            log.debug(f"Using {backend_impl.type()} backend for {kwargs}")
            return backend_impl
        else:
            log.debug(f"{backend_impl.type()} backend unavailable for {kwargs}")
            continue

    raise RuntimeError(
        f"No compatible attention({KsanaAttentionBackend.get_supported_list}) backend available for {kwargs}. "
    )


class KsanaAttentionOp(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        *,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        attention_config=None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_size**-0.5
        self.causal = causal
        if attention_config is None:
            raise ValueError("attention_config should not be None")
        self.attention_config = attention_config
        log.debug(f"KsanaAttentionOp with config: {self.attention_config}")
        self.extra_impl_args = extra_impl_args

        self._attn_impl: KsanaAttentionBackendImpl | None = None
        self._attn_impl_dtype: torch.dtype | None = None

    @property
    def backend_type(self) -> str | None:
        return self._attn_impl.type() if self._attn_impl else None

    def _ensure_impl(self, dtype: torch.dtype) -> None:
        if self._attn_impl is not None and dtype == self._attn_impl_dtype:
            return

        backend_impl = _get_attention_backend_impl(
            attn_backend=self.attention_config.backend,
            head_size=self.head_size,
            dtype=dtype,
        )
        self._attn_impl = backend_impl(
            num_heads=self.num_heads,
            head_size=self.head_size,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            **self.extra_impl_args,
        )
        self._attn_impl_dtype = dtype

    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **backend_kwargs,
    ) -> torch.Tensor:
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("LocalAttentionOp expects tensors shaped [B, L, H, D].")

        self._ensure_impl(query.dtype)
        assert self._attn_impl is not None

        return self._attn_impl.forward(
            query,
            key,
            value,
            **backend_kwargs,
        )
