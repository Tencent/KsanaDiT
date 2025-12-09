import torch
import logging

from ksana.utils import (
    cast_bias_weight,
    stochastic_rounding,
)


def fp8_linear(self, input):
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    input_shape = input.shape
    input_dtype = input.dtype
    if len(input.shape) == 3:
        w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)
        w = w.t()

        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)

        # always e4m3fn because e5m2 * e5m2 is not supported
        fixed_input_dtype_after_scale = torch.float8_e4m3fn
        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            input = input.reshape(-1, input_shape[2]).to(fixed_input_dtype_after_scale).contiguous()
        else:
            input = (
                (input * (1.0 / scale_input).to(input_dtype))
                .reshape(-1, input_shape[2])
                .to(fixed_input_dtype_after_scale)
                .contiguous()
            )

        if bias is not None:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(input, w, out_dtype=input_dtype, scale_a=scale_input, scale_b=scale_weight)

        if isinstance(o, tuple):
            o = o[0]

        if tensor_2d:
            return o.reshape(input_shape[0], -1)

        return o.reshape((-1, input_shape[1], self.weight.shape[0]))

    return None


class Fp8Linear(torch.nn.Linear):
    def reset_parameters(self):
        self.scale_weight = None
        self.scale_input = None
        return None

    def forward(self, input):
        if not self.training:
            try:
                out = fp8_linear(self, input)
                if out is not None:
                    return out
            except Exception as e:
                logging.info("Exception during fp8 op: {}".format(e))

        weight, bias = cast_bias_weight(self, input)
        return torch.nn.functional.linear(input, weight, bias)


def scaled_fp8_ops(fp8_matrix_mult=False, scale_input=False, override_dtype=None):
    logging.info("Using scaled fp8: fp8 matrix mult: {}, scale input: {}".format(fp8_matrix_mult, scale_input))

    class ScaledFp8Linear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            if override_dtype is not None:
                kwargs["dtype"] = override_dtype
            super().__init__(*args, **kwargs)

        def reset_parameters(self):
            if not hasattr(self, "scale_weight"):
                self.scale_weight = torch.nn.parameter.Parameter(
                    data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False
                )

            if not scale_input:
                self.scale_input = None

            if not hasattr(self, "scale_input"):
                self.scale_input = torch.nn.parameter.Parameter(
                    data=torch.ones((), device=self.weight.device, dtype=torch.float32), requires_grad=False
                )
            return None

        def forward(self, input):
            if fp8_matrix_mult:
                out = fp8_linear(self, input)
                if out is not None:
                    return out
                else:
                    logging.info("failed to run fp8_gemm, falling back to normal linear")

            weight, bias = cast_bias_weight(self, input)

            if weight.numel() < input.numel():  # TODO: optimize
                return torch.nn.functional.linear(
                    input, weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype), bias
                )
            else:
                return torch.nn.functional.linear(
                    input * self.scale_weight.to(device=weight.device, dtype=weight.dtype), weight, bias
                )

        def convert_weight(self, weight, inplace=False, **kwargs):
            if inplace:
                weight *= self.scale_weight.to(device=weight.device, dtype=weight.dtype)
                return weight
            else:
                return weight * self.scale_weight.to(device=weight.device, dtype=weight.dtype)

        def set_weight(self, weight, inplace_update=False, seed=None, **kwargs):
            weight = stochastic_rounding(
                weight / self.scale_weight.to(device=weight.device, dtype=weight.dtype),
                self.weight.dtype,
                seed=seed,
            )
            if inplace_update:
                self.weight.data.copy_(weight)
            else:
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

    return ScaledFp8Linear
