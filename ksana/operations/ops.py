# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .attention import pick_attn_op
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
    def __init__(self, *args, rms_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rms_dtype = rms_dtype

    def reset_parameters(self):
        self.bias = None
        return None

    def float_rmsnorm(self, input):
        """RMSNorm with float precision - converts input to float32 for computation"""

        def _norm(x):
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return (_norm(input.float()) * self.weight).type_as(input)

    def half_rmsnorm(self, input):
        """RMSNorm with half precision - keeps input in original dtype"""

        def _norm(x):
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return (_norm(input) * self.weight).type_as(input)

    def forward(self, *args, **kwargs):
        if hasattr(self, "rms_dtype") and self.rms_dtype is not None:
            if self.rms_dtype in [torch.float16, torch.bfloat16]:
                return self.half_rmsnorm(*args, **kwargs)
        elif self.weight.dtype == torch.float32:
            return self.float_rmsnorm(*args, **kwargs)

        return super().forward(*args, **kwargs).type_as(args[0])


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
    attention_config=None,
    load_device=None,
    rms_dtype=None,
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
    ops.register("Attn", pick_attn_op(attention_config))
    ops.rms_dtype = rms_dtype
    ops.linear_backend = linear_backend
    ops.state_dict = state_dict
    return ops
