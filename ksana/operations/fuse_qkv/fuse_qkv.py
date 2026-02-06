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

"""
Provides functions to remap legacy separate q/k/v weights to fused qkv format.
This enables QKV fusion optimization where 3 separate GEMM operations are
combined into a single GEMM for better GPU utilization.
"""

import torch

from ksana.utils.logger import log

MODEL_QKV_PATTERNS = {
    "wan": [
        (".self_attn.q", ".self_attn.k", ".self_attn.v", ".self_attn.qkv"),
    ],
    "qwen": [
        (".to_q", ".to_k", ".to_v", ".to_qkv"),  # Image stream
        (".add_q_proj", ".add_k_proj", ".add_v_proj", ".add_qkv_proj"),  # Text stream
    ],
}


def should_use_qkv_fusion(operations):
    if operations is None:
        return True
    linear_backend = getattr(operations, "linear_backend", None)
    if linear_backend is not None:
        from ksana.config import KsanaLinearBackend

        if linear_backend in [KsanaLinearBackend.FP8_GEMM, KsanaLinearBackend.FP8_GEMM_DYNAMIC]:
            return False

    state_dict = getattr(operations, "state_dict", None)
    if state_dict is not None:
        for v in state_dict.values():
            if hasattr(v, "dtype") and v.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                return False

    linear_class = getattr(operations, "Linear", None)
    if linear_class is None:
        return True

    if "Fp8" in linear_class.__name__:
        return False

    return True


def remap_qkv_weights(
    state_dict,
    patterns,
):
    remapped = state_dict.copy()

    for q_suffix, k_suffix, v_suffix, qkv_suffix in patterns:
        remapped = _apply_pattern(remapped, q_suffix, k_suffix, v_suffix, qkv_suffix)

    return remapped


def _apply_pattern(
    state_dict,
    q_suffix,
    k_suffix,
    v_suffix,
    qkv_suffix,
):
    remapped = state_dict.copy()
    q_weight_suffix = q_suffix + ".weight"
    q_keys = [k for k in state_dict.keys() if k.endswith(q_weight_suffix)]
    fused_count = 0
    for q_weight_key in q_keys:
        prefix = q_weight_key[: -len(q_weight_suffix)]

        k_weight_key = prefix + k_suffix + ".weight"
        v_weight_key = prefix + v_suffix + ".weight"
        qkv_weight_key = prefix + qkv_suffix + ".weight"

        if qkv_weight_key in state_dict:
            continue

        if k_weight_key not in state_dict or v_weight_key not in state_dict:
            continue

        remapped[qkv_weight_key] = torch.cat(
            [
                state_dict[q_weight_key],
                state_dict[k_weight_key],
                state_dict[v_weight_key],
            ],
            dim=0,
        )

        q_bias_key = prefix + q_suffix + ".bias"
        if q_bias_key in state_dict:
            k_bias_key = prefix + k_suffix + ".bias"
            v_bias_key = prefix + v_suffix + ".bias"
            qkv_bias_key = prefix + qkv_suffix + ".bias"

            remapped[qkv_bias_key] = torch.cat(
                [
                    state_dict[q_bias_key],
                    state_dict[k_bias_key],
                    state_dict[v_bias_key],
                ],
                dim=0,
            )

            for old_key in [q_bias_key, k_bias_key, v_bias_key]:
                remapped.pop(old_key, None)

        for old_key in [q_weight_key, k_weight_key, v_weight_key]:
            remapped.pop(old_key, None)

        fused_count += 1

    if fused_count > 0:
        log.debug(
            f"QKV fusion: fused {fused_count} layer(s) with pattern {q_suffix}/{k_suffix}/{v_suffix} -> {qkv_suffix}"
        )

    return remapped


def model_uses_qkv_fusion(model) -> bool:
    fusion_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "qkv_fused"):
            fusion_modules.append((name, module.qkv_fused))
            if module.qkv_fused:
                log.debug(f"[QKV Remap] Found qkv_fused=True in module: {name}")
                return True
    if fusion_modules:
        log.debug(f"[QKV Remap] Modules with qkv_fused attr: {fusion_modules[:5]}...")
    return False


def remap_state_dict_for_model(
    model,
    state_dict,
    model_name,
):
    uses_fusion = model_uses_qkv_fusion(model)
    log.info(f"[QKV Remap] model_uses_qkv_fusion={uses_fusion}, model_name={model_name}")
    if not uses_fusion:
        log.warning("[QKV Remap] Skipping remap because model does not use QKV fusion")
        return state_dict
    return _remap_qkv_if_needed(state_dict, model_name)


def _remap_qkv_if_needed(state_dict, model_name):
    model_name_lower = model_name.lower()
    patterns = None
    for key, value in MODEL_QKV_PATTERNS.items():
        if key in model_name_lower:
            patterns = value
            log.info(f"[QKV Remap] Found pattern for '{key}' in model_name '{model_name}'")
            break

    if patterns is None:
        log.warning(f"[QKV Remap] No patterns found for model_name '{model_name}'")
        return state_dict

    for q_suffix, _, _, qkv_suffix in patterns:
        q_weight_suffix = q_suffix + ".weight"
        qkv_weight_suffix = qkv_suffix + ".weight"
        has_separate = any(k.endswith(q_weight_suffix) for k in state_dict.keys())
        has_fused = any(k.endswith(qkv_weight_suffix) for k in state_dict.keys())

        log.info(
            f"[QKV Remap] Checking pattern: q_suffix='{q_suffix}', has_separate={has_separate}, has_fused={has_fused}"
        )

        if has_separate and not has_fused:
            log.info(f"[QKV Remap] Remapping separate q/k/v weights to fused qkv for {model_name}")
            return remap_qkv_weights(state_dict, patterns)

    log.info(f"[QKV Remap] No remapping needed - has_separate={has_separate}, has_fused={has_fused}")
    return state_dict


class QKVProjectionMixin:
    def _setup_qkv_projection(
        self,
        dim,
        operations,
        device=None,
        dtype=None,
        bias=True,
        fused_name="qkv",
        separate_names=("q", "k", "v"),
        prefix="",
    ):
        qkv_fused = should_use_qkv_fusion(operations)
        if not hasattr(self, "_qkv_configs"):
            self._qkv_configs = {}
        self._qkv_configs[prefix] = {
            "fused": qkv_fused,
            "fused_name": fused_name,
            "separate_names": separate_names,
        }
        self.qkv_fused = getattr(self, "qkv_fused", False) or qkv_fused
        if qkv_fused:
            setattr(self, fused_name, operations.Linear(dim, dim * 3, bias=bias, device=device, dtype=dtype))
        else:
            q_name, k_name, v_name = separate_names
            setattr(self, q_name, operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype))
            setattr(self, k_name, operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype))
            setattr(self, v_name, operations.Linear(dim, dim, bias=bias, device=device, dtype=dtype))

    def compute_qkv(self, x, prefix=""):
        config = self._qkv_configs[prefix]
        if config["fused"]:
            qkv = getattr(self, config["fused_name"])(x)
            return qkv.chunk(3, dim=-1)

        q_name, k_name, v_name = config["separate_names"]
        return getattr(self, q_name)(x), getattr(self, k_name)(x), getattr(self, v_name)(x)
