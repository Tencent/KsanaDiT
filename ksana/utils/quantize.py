import torch

from ..config import KsanaLinearBackend
from .logger import log

_FP8_TENSOR_PATCHED = False
_LOG_SAMPLE_LIMIT = 5

# Default layers to exclude from FP8 quantization for better generation quality.
# These layers are typically more sensitive to precision loss:
# - embed/embedder: Embedding layers affect semantic understanding
# - proj_out/head: Output projection layers directly impact generation quality
#   - Qwen uses `proj_out` as final output layer
#   - WAN uses `head.head` as final output layer
DEFAULT_FP8_EXCLUDE_LAYERS: list[str] = [
    "embed",  # Embedding layers (text, position, time, etc.)
    "proj_out",  # Output projection layers (Qwen model)
    "head",  # Final output head (WAN model: head.head)
]


def _patch_float8_tensor_dispatch():
    """
    Monkey-patch torchao's Float8Tensor to support device transfers.

    torchao's Float8Tensor doesn't implement aten.to with dtype_layout overload,
    which causes errors when ComfyUI tries to move models between CPU and GPU.

    This patch adds support for the missing .to() operations by registering
    handlers for aten.to.dtype_layout and similar operations.
    """
    global _FP8_TENSOR_PATCHED

    if _FP8_TENSOR_PATCHED:
        return True

    Float8Tensor = None  # pylint: disable=invalid-name
    try:
        from torchao.quantization import Float8Tensor
    except ImportError:
        try:
            from torchao.float8 import Float8Tensor
        except ImportError:
            return False

    if Float8Tensor is None:
        return False

    # Check if already patched
    if hasattr(Float8Tensor, "_ksana_patched"):
        _FP8_TENSOR_PATCHED = True
        return True

    aten = torch.ops.aten

    def _move_float8_tensor_to_device(tensor, device):
        if hasattr(tensor, "qdata") and hasattr(tensor, "scale"):
            new_qdata = tensor.qdata.to(device=device)
            new_scale = tensor.scale.to(device=device)

            # Get all the optional attributes
            kwargs = {}
            for attr in getattr(Float8Tensor, "optional_tensor_attribute_names", []):
                if hasattr(tensor, attr):
                    kwargs[attr] = getattr(tensor, attr)

            # Create new Float8Tensor with moved data
            return Float8Tensor(
                new_qdata,
                new_scale,
                **kwargs,
            )

        elif hasattr(tensor, "_data") and hasattr(tensor, "_scale"):
            new_data = tensor._data.to(device=device)
            new_scale = tensor._scale.to(device=device)

            # Try to reconstruct using available attributes
            try:
                return Float8Tensor(
                    new_data,
                    new_scale,
                    tensor._orig_dtype if hasattr(tensor, "_orig_dtype") else tensor.dtype,
                    tensor._linear_mm_config if hasattr(tensor, "_linear_mm_config") else None,
                    tensor._gemm_input_role if hasattr(tensor, "_gemm_input_role") else None,
                )
            except TypeError:
                # If constructor signature doesn't match, try manual attribute copy
                result = object.__new__(Float8Tensor)
                result._data = new_data
                result._scale = new_scale
                for attr in ["_orig_dtype", "_linear_mm_config", "_gemm_input_role", "_mm_config"]:
                    if hasattr(tensor, attr):
                        setattr(result, attr, getattr(tensor, attr))
                return result
        else:
            raise ValueError(f"Unknown Float8Tensor structure: {dir(tensor)}")

    def handle_to_dtype_layout(func, types, args, kwargs):
        tensor = args[0]
        device = kwargs.get("device", None)

        if device is not None:
            try:
                return _move_float8_tensor_to_device(tensor, device)
            except Exception as e:  # pylint: disable=broad-except
                log.warning(f"Failed to handle aten.to.dtype_layout: {e}")
                raise

        # If no device change, just return the tensor
        return tensor

    def handle_has_compatible_shallow_copy_type(func, types, args, kwargs):
        return False

    def handle_is_pinned(func, types, args, kwargs):
        # Float8Tensor does not support pinned-memory semantics; treat as not pinned.
        return False

    try:
        if hasattr(aten, "_has_compatible_shallow_copy_type"):
            Float8Tensor._ATEN_OP_OR_TORCH_FN_TABLE[Float8Tensor][
                aten._has_compatible_shallow_copy_type.default
            ] = handle_has_compatible_shallow_copy_type
            log.info("Registered handler for aten._has_compatible_shallow_copy_type")

        if hasattr(aten, "is_pinned"):
            # Some code paths call tensor.is_pinned() which dispatches to aten.is_pinned.
            Float8Tensor._ATEN_OP_OR_TORCH_FN_TABLE[Float8Tensor][aten.is_pinned.default] = handle_is_pinned
            log.info("Registered handler for aten.is_pinned")

        if hasattr(aten.to, "dtype_layout"):
            Float8Tensor._ATEN_OP_OR_TORCH_FN_TABLE[Float8Tensor][aten.to.dtype_layout] = handle_to_dtype_layout
            log.info("Registered handler for aten.to.dtype_layout")

        if hasattr(aten.to, "device"):
            Float8Tensor._ATEN_OP_OR_TORCH_FN_TABLE[Float8Tensor][aten.to.device] = handle_to_dtype_layout
            log.info("Registered handler for aten.to.device")

        if hasattr(aten.to, "dtype"):
            Float8Tensor._ATEN_OP_OR_TORCH_FN_TABLE[Float8Tensor][aten.to.dtype] = handle_to_dtype_layout
            log.info("Registered handler for aten.to.dtype")

    except Exception as e:  # pylint: disable=broad-except
        log.warning(f"Failed to register aten.to handlers: {e}")
        original_dispatch = Float8Tensor.__torch_dispatch__

        @classmethod
        def patched_torch_dispatch(cls, func, types, args, kwargs):
            func_name = str(func)
            if "aten.to" in func_name:
                tensor = args[0]
                device = kwargs.get("device", None)
                if device is not None:
                    try:
                        return _move_float8_tensor_to_device(tensor, device)
                    except Exception as e:  # pylint: disable=broad-except
                        log.warning(f"Fallback dispatch failed: {e}")

            return original_dispatch.__func__(cls, func, types, args, kwargs)

        Float8Tensor.__torch_dispatch__ = patched_torch_dispatch
        log.info("Patched Float8Tensor.__torch_dispatch__ as fallback")

    Float8Tensor._ksana_patched = True
    _FP8_TENSOR_PATCHED = True
    log.info("Successfully patched Float8Tensor for ComfyUI device transfer support")
    return True


def adjust_fp8_backend_for_dtype(run_dtype, linear_backend, model_name):
    """
    Normalize fp8_gemm backend against the requested run_dtype.
    - If weight dtype is non-FP8, fall back to a backend that can handle it.
    - Models can use dynamic FP8 backend for dynamic FP8 quantization.
    """
    rd_str = run_dtype if isinstance(run_dtype, str) else str(run_dtype)
    has_fp8 = rd_str is not None and ("float8" in rd_str.lower() or "fp8" in rd_str.lower())
    if run_dtype != "default" and not has_fp8:
        if linear_backend == KsanaLinearBackend.FP8_GEMM_DYNAMIC:
            log.warning(
                f"weight_dtype {run_dtype} is not FP8, using fp8_gemm_dynamic backend for dynamic FP8 quantization."
            )
            linear_backend = KsanaLinearBackend.FP8_GEMM_DYNAMIC
        else:
            log.warning(
                f"weight_dtype {run_dtype} can not use fp8_gemm linear_backend, will use run_dtype back to default"
            )
            run_dtype = "float16"
            linear_backend = KsanaLinearBackend.DEFAULT

    return run_dtype, linear_backend


def _apply_dynamic_fp8_quant(module: torch.nn.Module):
    # https://github.com/pytorch/ao/issues/2919
    try:
        from torchao.quantization import quantize_

        try:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor

            fp8_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
        except ImportError:
            from torchao.quantization import float8_dynamic_activation_float8_weight

            fp8_config = float8_dynamic_activation_float8_weight()
    except Exception as e:  # pylint: disable=broad-except
        log.warning(f"torchao.quantization is not available, skip FP8 dynamic quantization: {e}")
        return

    if not isinstance(module, torch.nn.Module):
        log.warning("apply_dynamic_fp8_quant called but target is not an nn.Module; skipping.")
        return

    exclude_layers = DEFAULT_FP8_EXCLUDE_LAYERS
    log.info(f"FP8 quantization exclude patterns: {exclude_layers}")

    # Check model device (for logging only - torchao can quantize on CPU)
    module_device = next(module.parameters()).device if len(list(module.parameters())) > 0 else None
    log.info(f"model device: {module_device}")

    # Note: torchao quantize_ works on CPU (it modifies module structure/weights)
    # The actual FP8 compute happens at runtime when model is on CUDA
    if module_device is not None and module_device.type != "cuda":
        log.info(
            f"model is on {module_device}. Quantization will be applied, "
            f"FP8 compute will activate when model moves to CUDA."
        )

    linear_count_before = 0
    eligible_count = 0
    torch_nn_linear_count = 0
    excluded_count = 0
    ineligible_dims = []
    non_standard_linear = []
    excluded_layers_names = []

    for name, m in module.named_modules():
        module_type = type(m).__name__
        module_class = type(m)

        # Check if it's a Linear-like module
        if module_type == "Linear" or "Linear" in module_type:
            linear_count_before += 1

            # Check if it's exactly torch.nn.Linear (required by torchao)
            is_torch_nn_linear = isinstance(m, torch.nn.Linear)
            if is_torch_nn_linear:
                torch_nn_linear_count += 1
            else:
                if len(non_standard_linear) < _LOG_SAMPLE_LIMIT:
                    non_standard_linear.append(f"{name}: {module_class.__module__}.{module_class.__name__}")

            # Check dimensions
            if hasattr(m, "in_features") and hasattr(m, "out_features"):
                in_f = m.in_features
                out_f = m.out_features
                # FP8 requires dimensions divisible by 16
                if in_f % 16 == 0 and out_f % 16 == 0:
                    if is_torch_nn_linear:
                        # Check if layer is excluded
                        is_excluded = False
                        for exclude_name in exclude_layers:
                            if exclude_name in name:
                                is_excluded = True
                                excluded_count += 1
                                if len(excluded_layers_names) < _LOG_SAMPLE_LIMIT:
                                    excluded_layers_names.append(f"{name} (pattern: {exclude_name})")
                                break
                        if not is_excluded:
                            eligible_count += 1
                else:
                    if len(ineligible_dims) < _LOG_SAMPLE_LIMIT:
                        ineligible_dims.append(f"{name}({in_f}x{out_f})")

    log.info(
        f"Linear layers: total={linear_count_before}, "
        f"torch.nn.Linear={torch_nn_linear_count}, "
        f"eligible (torch.nn.Linear + dims%16==0 - excluded)={eligible_count}, "
        f"excluded={excluded_count}"
    )

    if excluded_layers_names:
        log.info(f"Excluded layers (kept in FP16 for quality): {excluded_layers_names}")

    if non_standard_linear:
        log.warning(f"Found non-standard Linear modules (won't be quantized by torchao): " f"{non_standard_linear}")

    if ineligible_dims:
        log.warning(f"Ineligible layers (dims not %16): {ineligible_dims}")

    if eligible_count == 0:
        if torch_nn_linear_count == 0:
            log.warning(
                "No torch.nn.Linear modules found! "
                "torchao only quantizes torch.nn.Linear, not custom Linear implementations. "
                "This usually happens when ComfyUI uses custom operations. "
                "Try setting comfy_operations=None to use standard PyTorch Linear."
            )
        else:
            log.warning("No eligible Linear layers found (dims not divisible by 16 or all excluded)")
        return

    # Patch Float8Tensor BEFORE quantization to ensure device transfers work
    # This is needed for ComfyUI's model offloading mechanism
    _patch_float8_tensor_dispatch()

    # Build filter function for selective quantization
    num_quant = 0
    num_skip = 0

    def _default_filter_fn(m: torch.nn.Module, name: str) -> bool:
        """Default filter function that respects exclude_layers."""
        nonlocal num_quant, num_skip

        if not isinstance(m, torch.nn.Linear):
            return False

        # Check dimensions (FP8 requires dims divisible by 16)
        if hasattr(m, "in_features") and hasattr(m, "out_features"):
            if m.in_features % 16 != 0 or m.out_features % 16 != 0:
                num_skip += 1
                return False

        # Check exclude list
        for exclude_name in exclude_layers:
            if exclude_name in name:
                log.debug(f"Skip FP8 quantization: {name} -> pattern<{exclude_name}>")
                num_skip += 1
                return False

        num_quant += 1
        return True

    log.info(f"Applying dynamic float8 quantization via torchao to {eligible_count} eligible modules...")
    try:
        quantize_(module, fp8_config, filter_fn=_default_filter_fn)

        # Verify quantization was applied by checking module types and weight types
        quantized_count = 0
        linear_count_after = 0
        fp8_weight_count = 0
        sample_weight_types = []

        for name, m in module.named_modules():
            module_type = type(m).__name__
            # torchao wraps Linear with Float8DynamicLinear or similar
            if "Float8" in module_type or "Quantized" in module_type:
                quantized_count += 1
            if module_type == "Linear" or isinstance(m, torch.nn.Linear):
                linear_count_after += 1
                # Check weight type
                if hasattr(m, "weight") and m.weight is not None:
                    weight_type = type(m.weight).__name__
                    if "Float8" in weight_type or "Quantized" in weight_type or "Affine" in weight_type:
                        fp8_weight_count += 1
                    if len(sample_weight_types) < _LOG_SAMPLE_LIMIT:
                        sample_weight_types.append(f"{name}: {weight_type}")

        log.info(
            f"Quantization complete: "
            f"Linear before={linear_count_before}, Linear after={linear_count_after}, "
            f"Quantized modules={quantized_count}, FP8 weights={fp8_weight_count}, "
            f"Skipped (kept FP16)={num_skip}"
        )

        if sample_weight_types:
            log.info(f"Sample weight types: {sample_weight_types}")

        if quantized_count > 0 or fp8_weight_count > 0:
            log.info(
                f"Successfully quantized {max(quantized_count, fp8_weight_count)} modules to FP8, "
                f"kept {excluded_count} sensitive layers in original precision"
            )
        else:
            log.warning(
                "No modules were quantized. Possible reasons:\n"
                "  1. Linear modules are not torch.nn.Linear instances\n"
                "  2. Weight tensors are already wrapped/quantized\n"
                "  3. Model is on CPU (try moving to CUDA first)\n"
                "  4. torchao version incompatibility"
            )

    except Exception as e:  # pylint: disable=broad-except
        # Fail soft so that a bad torchao install or unsupported module does not break overall loading.
        log.warning(f"Failed to apply dynamic FP8 quantization via torchao, continue without FP8: {e}")
        import traceback

        log.warning(traceback.format_exc())


def find_fp8_info_from_state_dict(state_dict):
    if state_dict is None:
        return None, False

    fp8_dtype = None
    is_scaled_fp8 = False

    for _, value in state_dict.items():
        if hasattr(value, "dtype") and value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            fp8_dtype = value.dtype
            break

    for key, _ in state_dict.items():
        if key.endswith("scale_weight"):
            is_scaled_fp8 = True
            break

    return fp8_dtype, is_scaled_fp8


def apply_dynamic_fp8_quant(model: torch.nn.Module, load_device=None) -> bool:
    if model is None:
        log.warning("model is None; skip quantization.")
        return False

    log.info("linear_backend=fp8_gemm_dynamic, applying dynamic FP8 quantization")

    try:
        current_device = next(model.parameters()).device
    except StopIteration:
        log.warning("model has no parameters; skip quantization.")
        return False

    # Move model to CUDA BEFORE quantization to avoid device transfer issues
    # torchao's Float8Tensor doesn't support all aten ops needed for CPU->CUDA transfer
    if current_device.type != "cuda" and load_device is not None:
        target_device = load_device if isinstance(load_device, torch.device) else torch.device(load_device)
        if target_device.type == "cuda":
            log.info(f"Moving model from {current_device} to {target_device} before quantization...")
            model.to(target_device)
            log.info(f"Model moved to {target_device}")

    current_device = next(model.parameters()).device
    log.info(f"Model is on {current_device}, applying FP8 quantization...")
    _apply_dynamic_fp8_quant(model)
    log.info("FP8 quantization applied successfully.")
    return True
