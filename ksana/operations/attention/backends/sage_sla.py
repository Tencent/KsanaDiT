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

# pylint: disable=invalid-name
# modified from https://github.com/thu-ml/TurboDiffusion/blob/216ccc258495e8392c52f47aef838ffefa2313ff/turbodiffusion/SLA/core.py#L1 # pylint: disable=line-too-long # noqa: E501
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ksana.config import KsanaAttentionBackend
from ksana.config.attention_config import KsanaSageSLAConfig

from .base import KsanaAttentionBackendImpl

_SAGESLA_ENABLED = True
try:
    import spas_sage_attn._fused as fused
    import spas_sage_attn._qattn as qattn
    from spas_sage_attn.utils import block_map_lut_triton, get_vanilla_qk_quant
except ImportError:
    _SAGESLA_ENABLED = False

_SAGE2PP_ENABLED = True
try:
    from spas_sage_attn._qattn import qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
except ImportError:
    _SAGE2PP_ENABLED = False


@triton.jit
def compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


def get_cuda_arch(device_index):
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm{major}{minor}"


class SageSLAAttentionImpl(KsanaAttentionBackendImpl):
    @staticmethod
    def type() -> KsanaAttentionBackend:
        return KsanaAttentionBackend.SAGE_SLA

    @staticmethod
    def supports(**_) -> bool:
        return _SAGESLA_ENABLED

    def __init__(
        self,
        attention_config: KsanaSageSLAConfig,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
    ) -> None:
        if not _SAGESLA_ENABLED:
            raise RuntimeError("SageSLA backend requested but 'spas_sage_attn' package is not installed. ")
        if not isinstance(attention_config, KsanaSageSLAConfig):
            raise ValueError("KsanaSageSLAConfig must be provided")
        self.attention_config = attention_config
        self.topk = attention_config.topk

        from ..attention_op import KsanaAttentionOp  # lazy import

        self.dense_attn = KsanaAttentionOp(
            num_heads,
            head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            attention_config=attention_config.dense_attention_config,
        )
        self.check_config()

        def softmax_feature_map(x):
            return F.softmax(x, dim=-1)

        self.feature_map_q = softmax_feature_map
        self.feature_map_k = softmax_feature_map

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        dense_only: bool = False,
        proj_l: nn.Linear = None,
        **kwargs,
    ) -> torch.Tensor:
        if dense_only:
            # dense attention
            return self.dense_attn(
                query,
                key,
                value,
                **kwargs,
            )
        else:
            return self.sla_forward(
                query,
                key,
                value,
                proj_l,
            )

    @torch.compiler.disable()
    def sla_forward(self, query, k, v, proj_l, return_sparsity=False):
        R"""
        Args:
            q: queries of shape (B, H, L, D).
            k: keys of shape (B, H, L, D).
            v: values of shape (B, H, L, D).
            return_sparsity: whether to return the actual sparsity.
            timestep: current timestep for diffusion models.
            total_timesteps: total timesteps for diffusion models.
        """

        dtype = query.dtype
        self.dtype = dtype

        q = query.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        arch = get_cuda_arch(q.device.index)
        if arch == "sm90":
            sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=64, BLKK=128)
        else:
            sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=self.topk, BLKQ=128, BLKK=64)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        # SPARGE BEGIN

        km = k.mean(dim=-2, keepdim=True)
        headdim = q.size(-1)

        if arch == "sm90":
            q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 128)
        else:
            q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 128, 64)
        lut, valid_block_num = block_map_lut_triton(sparse_map)
        scale = 1.0 / (headdim**0.5)

        assert headdim in [
            64,
            128,
        ], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

        o_s = torch.empty_like(q)

        if arch in ("sm80", "sm86", "sm87"):
            pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
            v_fp16 = v.to(torch.float16)
            qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
                q_int8, k_int8, v_fp16, o_s, lut, valid_block_num, pvthreshold, q_scale, k_scale, 1, False, 1, scale, 0
            )
        else:
            b, h_kv, kv_len, head_dim = v.shape
            padded_len = (kv_len + 127) // 128 * 128
            v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
            fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
            v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
            v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
            fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

            if arch == "sm90":
                qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
                    q_int8, k_int8, v_fp8, o_s, lut, valid_block_num, q_scale, k_scale, v_scale, 1, False, 1, scale
                )
            else:
                pvthreshold = torch.full((q.shape[-3],), 1e6, dtype=torch.float32, device=q.device)
                if _SAGE2PP_ENABLED:
                    qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8,
                        k_int8,
                        v_fp8,
                        o_s,
                        lut,
                        valid_block_num,
                        pvthreshold,
                        q_scale,
                        k_scale,
                        v_scale,
                        1,
                        False,
                        1,
                        scale,
                        0,
                    )
                else:
                    qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
                        q_int8,
                        k_int8,
                        v_fp8,
                        o_s,
                        lut,
                        valid_block_num,
                        pvthreshold,
                        q_scale,
                        k_scale,
                        v_scale,
                        1,
                        False,
                        1,
                        scale,
                        0,
                    )

        # SPARGE END

        q = self.feature_map_q(q).contiguous().to(self.dtype)  # c_q
        k = self.feature_map_k(k).contiguous().to(self.dtype)  # c_k

        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))

        o_l = calc_linear(q, k, v)

        with torch.amp.autocast("cuda", dtype=self.dtype):
            o_l = proj_l(o_l)
        o = (o_s + o_l).to(dtype).transpose(1, 2)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o
