# Adapted from https://git.woa.com/longwind/venus-aigc/ComfyUI/blob/venus-develop/comfy/ops.py#L19
# TODO(rockcao): 需要统一Linear和Attention backed的调用方式
import torch
import logging

from ksana.utils import (
    cast_bias_weight,
    supports_fp8_compute,
    stochastic_rounding,
)


#  TODO support in future
# if torch.cuda.is_available() and torch.backends.cudnn.is_available() and PerformanceFeature.AutoTune in args.fast:
#     torch.backends.cudnn.benchmark = True


class BaseOps:  # return None in reset_parameters to speed up loading
    class Linear(torch.nn.Linear):
        def reset_parameters(self):
            return None

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


def fp8_linear(self, input):
    # TODO(rockcao): support fp8e5m2
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
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

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            input = torch.clamp(input, min=-448, max=448, out=input)
            input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
        else:
            input = (input * (1.0 / scale_input).to(input_dtype)).reshape(-1, input_shape[2]).to(dtype).contiguous()

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


class fp8_ops(BaseOps):
    class Linear(BaseOps.Linear):
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

    class scaled_fp8_op(BaseOps):
        class Linear(BaseOps.Linear):
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

    return scaled_fp8_op


CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear

    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:

    class cublas_ops(BaseOps):  # use cublas linear when dtype is float16
        class Linear(CublasLinear, BaseOps.Linear):
            def reset_parameters(self):
                return None

            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)


# TODO(rockcao): 优化pick逻辑
def pick_operations(weight_dtype, load_device=None, fp8_gemm=False, scaled_fp8=None):
    """
    根据硬件能力和数据类型选择最优的神经网络操作实现。

    该函数实现了一个选择策略，按优先级返回不同的操作类：
    1. scaled_fp8_ops - 带缩放因子的FP8操作（最高优先级，如果指定）
    2. fp8_ops - 标准FP8操作（需要硬件支持）
    3. cublas_ops - CUBLAS优化的FP16操作
    4. BaseOps - 基础PyTorch操作（默认回退）

    Args:
        weight_dtype (torch.dtype): 模型权重的数据类型（如 torch.float16, torch.float8_e4m3fn）
        load_device (torch.device, optional): 模型加载的设备，用于检测硬件FP8支持能力
        fp8_gemm (bool, optional): 是否启用FP8优化。默认False
        scaled_fp8 (torch.dtype, optional): 指定使用带缩放的FP8类型（如torch.float8_e4m3fn）
            如果设置，将强制使用scaled_fp8_ops

    Returns:
        class: 返回操作类（BaseOps的子类），包含优化的Linear、Conv等层实现


    Note:
        - CUBLAS优化仅在FP16且可用时启用
    """
    support_fp8_compute = supports_fp8_compute(load_device)

    if scaled_fp8 is not None:
        return scaled_fp8_ops(
            fp8_matrix_mult=support_fp8_compute and fp8_gemm, scale_input=fp8_gemm, override_dtype=scaled_fp8
        )

    if support_fp8_compute and fp8_gemm:
        return fp8_ops

    if CUBLAS_IS_AVAILABLE and weight_dtype == torch.float16:
        logging.info("Using cublas ops")
        return cublas_ops

    return BaseOps
