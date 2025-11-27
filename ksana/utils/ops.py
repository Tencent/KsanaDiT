import torch

torch_version = torch.version.__version__
temp = torch_version.split(".")
torch_version_numeric = (int(temp[0]), int(temp[1]))


def device_supports_non_blocking(device):
    return True


def supports_fp8_compute(device=None):
    props = torch.cuda.get_device_properties(device)
    if props.major >= 9:
        return True
    if props.major < 8:
        return False
    if props.minor < 9:
        return False

    if torch_version_numeric < (2, 3):
        return False
    return True


def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        if stream is not None:
            with stream:
                return weight.to(dtype=dtype, copy=copy)
        return weight.to(dtype=dtype, copy=copy)

    if stream is not None:
        with stream:
            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight, non_blocking=non_blocking)
    else:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
    return r


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias = None
    non_blocking = device_supports_non_blocking(device)
    if s.bias is not None:
        has_function = len(s.bias_function) > 0
        bias = cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function)

        if has_function:
            for f in s.bias_function:
                bias = f(bias)

    has_function = len(s.weight_function) > 0
    weight = cast_to(s.weight, dtype, device, non_blocking=non_blocking, copy=has_function)
    if has_function:
        for f in s.weight_function:
            weight = f(weight)

    return weight, bias


def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS))),
    )

    mantissa_scaled += torch.rand(
        mantissa_scaled.size(),
        dtype=mantissa_scaled.dtype,
        layout=mantissa_scaled.layout,
        device=mantissa_scaled.device,
        generator=generator,
    )
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)


# Not 100% sure about this
def manual_stochastic_round_to_float8(x, dtype, generator=None):
    if dtype == torch.float8_e4m3fn:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 4, 3, 7
    elif dtype == torch.float8_e5m2:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 5, 2, 15
    else:
        raise ValueError("Unsupported dtype")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS, 0, 2**EXPONENT_BITS - 1)

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask, (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x), (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    inf = torch.finfo(dtype)
    torch.clamp(sign, min=inf.min, max=inf.max, out=sign)
    return sign


def stochastic_rounding(value, dtype, seed=0):
    if dtype == torch.float32:
        return value.to(dtype=torch.float32)
    if dtype == torch.float16:
        return value.to(dtype=torch.float16)
    if dtype == torch.bfloat16:
        return value.to(dtype=torch.bfloat16)
    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        output = torch.empty_like(value, dtype=dtype)
        num_slices = max(1, (value.numel() / (4096 * 4096)))
        slice_size = max(1, round(value.shape[0] / num_slices))
        for i in range(0, value.shape[0], slice_size):
            output[i : i + slice_size].copy_(
                manual_stochastic_round_to_float8(value[i : i + slice_size], dtype, generator=generator)
            )
        return output

    return value.to(dtype=dtype)
