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

import ksana.nodes as nodes
from ksana.config import KsanaAttentionBackend


class KsanaAttentionConfigNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "optional": {
                "backend": (
                    KsanaAttentionBackend.get_supported_list(exclude=[KsanaAttentionBackend.RADIAL_SAGE_ATTN]),
                    {"default": KsanaAttentionBackend.FLASH_ATTN.value},
                    {"tooltip": "attention backend"},
                ),
            }
        }

    RETURN_TYPES = (nodes.KSANA_ATTENTION_CONFIG,)
    RETURN_NAMES = ("attention_config",)
    FUNCTION = "func"
    CATEGORY = nodes.KSANA_CATEGORY_CONFIGS
    DESCRIPTION = "attention config settings"

    def func(self, *args, **kwargs):
        return (nodes.attention_config(*args, **kwargs),)


class KsanaRadialSageAttentionConfigNode:
    """
    ComfyUI node for configuring Radial Sage Attention parameters.

    This node creates a configuration object for the Radial Sage Attention backend,
    which implements sparse attention patterns for efficient video generation.
    """

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "optional": {
                "dense_blocks_num": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of dense attention blocks at the beginning",
                    },
                ),
                "dense_attn_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of steps to use dense attention",
                    },
                ),
                "decay_factor": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.01,
                        "max": 0.99,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Decay factor for radial attention pattern (must be in range (0, 1))",
                    },
                ),
                "block_size": (
                    [64, 128],
                    {
                        "default": 64,
                        "tooltip": "Block size for sparse attention computation (64 or 128)",
                    },
                ),
                "dense_backend": (
                    KsanaAttentionBackend.get_supported_list(exclude=[KsanaAttentionBackend.RADIAL_SAGE_ATTN]),
                    {"default": KsanaAttentionBackend.SAGE_ATTN.value},
                    {"tooltip": "dense attention backend"},
                ),
            }
        }

    RETURN_TYPES = (nodes.KSANA_ATTENTION_CONFIG,)
    RETURN_NAMES = ("radial_sage_attention_config",)
    FUNCTION = "func"
    CATEGORY = nodes.KSANA_CATEGORY_CONFIGS
    DESCRIPTION = "Configure Radial Sage Attention parameters for sparse attention computation"

    def func(self, *args, **kwargs):
        return (nodes.radial_sage_attention_config(*args, **kwargs),)


class KsanaSageSLAConfigNode:
    """
    ComfyUI node for configuring Sage SL Attention parameters.

    This node creates a configuration object for the Sage SL Attention backend,
    which implements sparse attention with top-k selection for efficient computation.
    """

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "required": {
                "topk": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.99,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Top-k ratio for sparse attention (must be in range (0, 1))",
                    },
                ),
                "dense_backend": (
                    KsanaAttentionBackend.get_supported_list(
                        exclude=[KsanaAttentionBackend.RADIAL_SAGE_ATTN, KsanaAttentionBackend.SAGE_SLA]
                    ),
                    {"default": KsanaAttentionBackend.SAGE_ATTN.value},
                    {"tooltip": "dense attention backend"},
                ),
            }
        }

    RETURN_TYPES = (nodes.KSANA_ATTENTION_CONFIG,)
    RETURN_NAMES = ("sage_sla_config",)
    FUNCTION = "func"
    CATEGORY = nodes.KSANA_CATEGORY_CONFIGS
    DESCRIPTION = "Configure Sage SL Attention parameters for sparse attention with top-k selection"

    def func(self, *args, **kwargs):
        return (nodes.sage_sla_config(*args, **kwargs),)
