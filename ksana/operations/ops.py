import torch

from .attention import pick_attn_op, KsanaAttentionBackend
from .linear import pick_linear


class Conv1d(torch.nn.Conv1d):
    def reset_parameters(self):
        return None


class Conv2d(torch.nn.Conv2d):
    def reset_parameters(self):
        return None


class Conv3d(torch.nn.Conv3d):
    def reset_parameters(self):
        return None


class GroupNorm(torch.nn.GroupNorm):
    def reset_parameters(self):
        return None


class LayerNorm(torch.nn.LayerNorm):
    def reset_parameters(self):
        return None

    def fp32_layernorm(self, input):  # keep float32 while doing layernorm
        return super().forward(input.float()).type_as(input)

    def forward(self, *args, **kwargs):
        if self.weight is not None and self.weight.dtype == torch.float32:
            return self.fp32_layernorm(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)


class RMSNorm(torch.nn.RMSNorm):
    def reset_parameters(self):
        self.bias = None
        return None

    def fp32_rmsnorm(self, input):
        def _norm(x):
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return (_norm(input.float()) * self.weight).type_as(input)

    def forward(self, *args, **kwargs):
        if self.weight.dtype == torch.float32:
            return self.fp32_rmsnorm(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def reset_parameters(self):
        return None


class ConvTranspose1d(torch.nn.ConvTranspose1d):
    def reset_parameters(self):
        return None


class Embedding(torch.nn.Embedding):
    def reset_parameters(self):
        self.bias = None
        return None

    def forward(self, *args, **kwargs):
        if "out_dtype" in kwargs:
            kwargs.pop("out_dtype")
        return super().forward(*args, **kwargs)


class Ops:

    def __init__(self):
        self._registry = {}

    def register(self, name, cls):
        self._registry[name] = cls
        setattr(self, name, cls)

    def __getattr__(self, name):
        if name in self._registry:
            return self._registry[name]
        raise AttributeError(f"'{type(self).__name__}' 没有属性 '{name}'")

    def __getitem__(self, name):
        return self._registry[name]

    def keys(self):
        return self._registry.keys()


def build_ops(
    run_dtype,
    state_dict,
    linear_backend: str,
    attn_backend: KsanaAttentionBackend = KsanaAttentionBackend.FLASH_ATTN,
    load_device=None,
):
    ops = Ops()
    ops.register("Conv1d", Conv1d)
    ops.register("Conv2d", Conv2d)
    ops.register("Conv3d", Conv3d)
    ops.register("GroupNorm", GroupNorm)
    ops.register("LayerNorm", LayerNorm)
    ops.register("RMSNorm", RMSNorm)
    ops.register("ConvTranspose2d", ConvTranspose2d)
    ops.register("ConvTranspose1d", ConvTranspose1d)
    ops.register("Embedding", Embedding)
    ops.register("Linear", pick_linear(run_dtype, state_dict, linear_backend, load_device))
    ops.register("Attn", pick_attn_op(attn_backend))
    ops.attn_backend = attn_backend
    return ops
