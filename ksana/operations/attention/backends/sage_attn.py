import torch

from ksana.config import KsanaAttentionBackend, KsanaAttentionConfig

from .base import KsanaAttentionBackendImpl

try:
    from sageattention import sageattn

    _SAGE_AVAILABLE = True
except ModuleNotFoundError:
    sageattn = None  # pylint: disable=invalid-name
    _SAGE_AVAILABLE = False


class SageAttentionImpl(KsanaAttentionBackendImpl):

    @staticmethod
    def type() -> KsanaAttentionBackend:
        return KsanaAttentionBackend.SAGE_ATTN

    @staticmethod
    def supports(**_) -> bool:
        return _SAGE_AVAILABLE

    def __init__(
        self,
        attention_config: KsanaAttentionConfig,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        if not _SAGE_AVAILABLE:
            raise RuntimeError("SageAttention backend requested but 'sageattention' package is not installed. ")
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = float(extra_impl_args.get("dropout_p", 0.0))
        self.attention_config = attention_config
        self.check_config()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if sageattn is None:
            raise RuntimeError("sageattention module missing at runtime.")
        original_dtype = query.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = (
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            )
            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)
        output = sageattn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=self.causal,
        )
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output
