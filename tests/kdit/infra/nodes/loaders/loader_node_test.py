# Copyright 2026 Tencent
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

"""Loader 节点集成测试 — 只 mock 真正的模型加载，验证 run() 内部 put_model 正确性。

设计原则：
- 使用真实的 PinHub + ModelPool，不 mock 数据流转层
- 只 mock 模型构造函数、load_default_settings、文件系统检查等重 I/O 操作
- 验证 run() 执行后 ModelPool 中存在正确的 ModelPoolKey → model 映射
- 这样可以捕获 put_model API 迁移类的 bug（如旧 API 调用方式）
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from kdit.models.model_key import ModelKey
from kdit.models.model_pool import ModelPool
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.device_info import DeviceInfo
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_def import NodeDef
from kdit.nodes.core.node_types import IONodeType
from kdit.nodes.core.pin_hub import PinHub
from kdit.nodes.loaders.diffusion_model_loader import DiffusionLoaderNode
from kdit.nodes.loaders.text_encoder_loader import TextEncoderLoaderNode
from kdit.nodes.loaders.vae_loader import VAELoaderNode
from kdit.tensor.tensor_pool import TensorPool


def _make_device_info():
    return DeviceInfo(
        compute_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        rank_id=0,
        world_size=1,
    )


def _make_loader_pins(*, model_key, model_pool=None):
    """为 Loader 节点构建 PinHub — Loader 没有输入 pin，只有输出。"""
    node_def = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=model_key)
    pins = PinHub(
        node_def=node_def,
        input_pins={},
        tensor_pool=TensorPool(),
        model_pool=model_pool or ModelPool(),
    )
    return pins, node_def


class TestTextEncoderLoaderNode(unittest.TestCase):
    """TextEncoderLoaderNode.run() 集成测试 — 验证 put_model 写入 ModelPool。"""

    def setUp(self):
        self.node = TextEncoderLoaderNode()
        self.model_key = ModelKey.T5TextEncoder
        self.node._factory_model_key = self.model_key
        self.model_pool = ModelPool()
        self.device_info = _make_device_info()

    @patch("kdit.nodes.loaders.text_encoder_loader.KsanaTextEncoderModel")
    @patch("kdit.nodes.loaders.text_encoder_loader.load_default_settings")
    @patch("kdit.nodes.loaders.text_encoder_loader.Path")
    @patch("kdit.nodes.loaders.text_encoder_loader.os.path.exists", return_value=True)
    def test_run_puts_model_into_pool(self, _mock_exists, mock_path_cls, mock_load_settings, mock_model_cls):
        # ── arrange ──
        mock_path_cls.return_value.is_dir.return_value = True
        mock_settings = MagicMock()
        mock_load_settings.return_value = mock_settings
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        context = NodeContext(
            prompt=["test"],
            metadata={"model_path": "/fake/text_encoder"},
            device=self.device_info,
        )
        pins, node_def = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

        # ── act ──
        self.node.run(pins, context=context)

        # ── assert: model 被正确写入 ModelPool ──
        expected_key = ModelPoolKey(node_def.node_id, self.model_key)
        stored_model = self.model_pool.get_model(expected_key)
        self.assertIs(stored_model, mock_model)

        # ── assert: 构造函数参数正确 ──
        mock_model_cls.assert_called_once_with(
            self.model_key,
            default_settings=mock_settings.text_encoder,
            checkpoint_dir="/fake/text_encoder",
            device=torch.device("cpu"),
            dtype=None,
        )

    @patch("kdit.nodes.loaders.text_encoder_loader.KsanaTextEncoderModel")
    @patch("kdit.nodes.loaders.text_encoder_loader.load_default_settings")
    @patch("kdit.nodes.loaders.text_encoder_loader.Path")
    @patch("kdit.nodes.loaders.text_encoder_loader.os.path.exists", return_value=True)
    def test_run_with_qwen_model_key(self, _mock_exists, mock_path_cls, mock_load_settings, mock_model_cls):
        """验证不同 model_key 也能正确写入 ModelPool。"""
        self.node._factory_model_key = ModelKey.Qwen2VLTextEncoder
        mock_path_cls.return_value.is_dir.return_value = True
        mock_load_settings.return_value = MagicMock()
        mock_model_cls.return_value = MagicMock()

        context = NodeContext(
            prompt=["test"],
            metadata={"model_path": "/fake/qwen_encoder"},
            device=self.device_info,
        )
        pins, node_def = _make_loader_pins(model_key=ModelKey.Qwen2VLTextEncoder, model_pool=self.model_pool)

        self.node.run(pins, context=context)

        expected_key = ModelPoolKey(node_def.node_id, ModelKey.Qwen2VLTextEncoder)
        stored_model = self.model_pool.get_model(expected_key)
        self.assertIs(stored_model, mock_model_cls.return_value)

    @patch("kdit.nodes.loaders.text_encoder_loader.Path")
    @patch("kdit.nodes.loaders.text_encoder_loader.os.path.exists", return_value=False)
    def test_run_raises_on_invalid_path(self, _mock_exists, _mock_path_cls):
        """checkpoint_dir 不存在时应抛出 ValueError。"""
        context = NodeContext(
            prompt=["test"],
            metadata={"model_path": "/nonexistent"},
            device=self.device_info,
        )
        pins, _ = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

        with self.assertRaises(ValueError):
            self.node.run(pins, context=context)


class TestVAELoaderNode(unittest.TestCase):
    """VAELoaderNode.run() 集成测试 — 验证 put_model 写入 ModelPool。"""

    def setUp(self):
        self.node = VAELoaderNode()
        self.model_key = ModelKey.VAE_WAN2_1
        self.node._factory_model_key = self.model_key
        self.model_pool = ModelPool()
        self.device_info = _make_device_info()

    @patch("kdit.nodes.loaders.vae_loader.load_default_settings")
    @patch("kdit.nodes.loaders.vae_loader.is_file_or_dir", return_value=True)
    @patch("kdit.nodes.loaders.vae_loader.os.path.exists", return_value=True)
    def test_run_puts_model_into_pool(self, _mock_exists, _mock_is_file, mock_load_settings):
        # ── arrange ──
        mock_settings = MagicMock()
        mock_load_settings.return_value = mock_settings
        mock_model = MagicMock()

        with patch.dict(VAELoaderNode._MAP_KEY_TO_MODEL_CLASS, {self.model_key: MagicMock(return_value=mock_model)}):
            context = NodeContext(
                prompt=["test"],
                metadata={"model_path": "/fake/vae.safetensors"},
                device=self.device_info,
            )
            pins, node_def = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

            # ── act ──
            self.node.run(pins, context=context)

        # ── assert: model 被正确写入 ModelPool ──
        expected_key = ModelPoolKey(node_def.node_id, self.model_key)
        stored_model = self.model_pool.get_model(expected_key)
        self.assertIs(stored_model, mock_model)

        # ── assert: model.load() 被调用 ──
        mock_model.load.assert_called_once_with("/fake/vae.safetensors", shard_fn=None)

    @patch("kdit.nodes.loaders.vae_loader.load_default_settings")
    @patch("kdit.nodes.loaders.vae_loader.is_file_or_dir", return_value=True)
    @patch("kdit.nodes.loaders.vae_loader.os.path.exists", return_value=True)
    def test_run_with_qwen_vae(self, _mock_exists, _mock_is_file, mock_load_settings):
        """验证 QwenImageVAE model_key 也能正确写入 ModelPool。"""
        self.node._factory_model_key = ModelKey.QwenImageVAE
        mock_load_settings.return_value = MagicMock()
        mock_model = MagicMock()

        with patch.dict(
            VAELoaderNode._MAP_KEY_TO_MODEL_CLASS, {ModelKey.QwenImageVAE: MagicMock(return_value=mock_model)}
        ):
            context = NodeContext(
                prompt=["test"],
                metadata={"model_path": "/fake/qwen_vae.safetensors"},
                device=self.device_info,
            )
            pins, node_def = _make_loader_pins(model_key=ModelKey.QwenImageVAE, model_pool=self.model_pool)

            self.node.run(pins, context=context)

        expected_key = ModelPoolKey(node_def.node_id, ModelKey.QwenImageVAE)
        stored_model = self.model_pool.get_model(expected_key)
        self.assertIs(stored_model, mock_model)

    @patch("kdit.nodes.loaders.vae_loader.load_default_settings")
    @patch("kdit.nodes.loaders.vae_loader.is_file_or_dir", return_value=True)
    @patch("kdit.nodes.loaders.vae_loader.os.path.exists", return_value=True)
    def test_run_passes_shard_fn(self, _mock_exists, _mock_is_file, mock_load_settings):
        """验证 metadata 中的 shard_fn 被正确传递给 model.load()。"""
        mock_load_settings.return_value = MagicMock()
        mock_model = MagicMock()
        fake_shard_fn = MagicMock()

        with patch.dict(VAELoaderNode._MAP_KEY_TO_MODEL_CLASS, {self.model_key: MagicMock(return_value=mock_model)}):
            context = NodeContext(
                prompt=["test"],
                metadata={"model_path": "/fake/vae.safetensors", "shard_fn": fake_shard_fn},
                device=self.device_info,
            )
            pins, _ = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

            self.node.run(pins, context=context)

        mock_model.load.assert_called_once_with("/fake/vae.safetensors", shard_fn=fake_shard_fn)

    @patch("kdit.nodes.loaders.vae_loader.is_file_or_dir", return_value=False)
    @patch("kdit.nodes.loaders.vae_loader.os.path.exists", return_value=False)
    def test_run_raises_on_invalid_path(self, _mock_exists, _mock_is_file):
        """model_path 不存在时应抛出 ValueError。"""
        context = NodeContext(
            prompt=["test"],
            metadata={"model_path": "/nonexistent"},
            device=self.device_info,
        )
        pins, _ = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

        with self.assertRaises(ValueError):
            self.node.run(pins, context=context)


class TestDiffusionLoaderNode(unittest.TestCase):
    """DiffusionLoaderNode.run() 集成测试 — 验证 put_model 写入 ModelPool。"""

    def setUp(self):
        self.node = DiffusionLoaderNode()
        self.model_key = ModelKey.Wan2_2_T2V_14B
        self.node._factory_model_key = self.model_key
        self.model_pool = ModelPool()
        self.device_info = _make_device_info()
        # 清理类级别的 PinnedMemoryManager 避免测试间干扰
        DiffusionLoaderNode._pinned_memory_manager = None

    def tearDown(self):
        DiffusionLoaderNode._pinned_memory_manager = None

    def _make_model_config(self):
        """创建一个 MagicMock 的 ModelConfig，避免真实 __post_init__ 校验。"""
        config = MagicMock()
        config.run_dtype = torch.bfloat16
        config.attention_config = None
        config.linear_backend = None
        config.rms_dtype = torch.float32
        config.torch_compile_config = None
        return config

    @patch("kdit.nodes.loaders.diffusion_model_loader.PinnedMemoryManager")
    @patch("kdit.nodes.loaders.diffusion_model_loader.build_ops")
    @patch("kdit.nodes.loaders.diffusion_model_loader.load_default_settings")
    @patch("kdit.nodes.loaders.diffusion_model_loader.is_file_or_dir", return_value=True)
    def test_run_single_model_puts_into_pool(self, _mock_is_file, mock_load_settings, mock_build_ops, _mock_pmm_cls):
        """单模型路径 — run() 后 ModelPool 中存在正确的 model。"""
        # ── arrange ──
        mock_load_settings.return_value = MagicMock()
        mock_ops = MagicMock()
        mock_build_ops.return_value = mock_ops
        mock_model = MagicMock()
        mock_model.preprocess_model_state_dict.side_effect = lambda sd: sd
        mock_model.to.return_value = mock_model  # model.to(offload_device) 返回自身

        mock_model_cls = MagicMock(return_value=mock_model)
        mock_state_dict = {"fake.weight": torch.zeros(1)}

        model_config = self._make_model_config()
        context = NodeContext(
            prompt=["test"],
            metadata={
                "model_path": "/fake/diffusion_model.safetensors",
                "model_config": model_config,
            },
            device=self.device_info,
        )
        pins, node_def = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

        with (
            patch.dict(DiffusionLoaderNode._MAP_KEY_TO_MODEL_CLASS, {self.model_key: mock_model_cls}),
            patch.object(DiffusionLoaderNode, "_load_state_dict", return_value=mock_state_dict),
        ):
            # ── act ──
            self.node.run(pins, context=context)

        # ── assert: model 被正确写入 ModelPool ──
        expected_key = ModelPoolKey(node_def.node_id, self.model_key)
        stored_model = self.model_pool.get_model(expected_key)
        self.assertIs(stored_model, mock_model)

        # ── assert: model 生命周期方法被调用 ──
        mock_model.load.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
        mock_model.enable_only_infer.assert_called_once()
        mock_model.prepare_distributed_model.assert_called_once()
        mock_model.apply_dynamic_fp8_quant.assert_called_once()
        mock_model.apply_torch_compile.assert_called_once()
        mock_model.apply_pinned_memory.assert_called_once()

    @patch("kdit.nodes.loaders.diffusion_model_loader.PinnedMemoryManager")
    @patch("kdit.nodes.loaders.diffusion_model_loader.build_ops")
    @patch("kdit.nodes.loaders.diffusion_model_loader.load_default_settings")
    @patch("kdit.nodes.loaders.diffusion_model_loader.is_file_or_dir", return_value=True)
    def test_run_multi_model_puts_list_into_pool(
        self, _mock_is_file, mock_load_settings, mock_build_ops, _mock_pmm_cls
    ):
        """多模型路径（如 high + low noise）— run() 后 ModelPool 中存在 model 列表。"""
        # ── arrange ──
        mock_load_settings.return_value = MagicMock()
        mock_ops = MagicMock()
        mock_build_ops.return_value = mock_ops
        mock_model_a = MagicMock()
        mock_model_a.preprocess_model_state_dict.side_effect = lambda sd: sd
        mock_model_a.to.return_value = mock_model_a
        mock_model_b = MagicMock()
        mock_model_b.preprocess_model_state_dict.side_effect = lambda sd: sd
        mock_model_b.to.return_value = mock_model_b

        call_count = {"n": 0}

        def _make_model(*_args, **_kwargs):
            call_count["n"] += 1
            return mock_model_a if call_count["n"] == 1 else mock_model_b

        mock_model_cls = MagicMock(side_effect=_make_model)
        mock_state_dict = {"fake.weight": torch.zeros(1)}

        model_config = self._make_model_config()
        context = NodeContext(
            prompt=["test"],
            metadata={
                "model_path": ["/fake/high_noise.safetensors", "/fake/low_noise.safetensors"],
                "model_config": model_config,
            },
            device=self.device_info,
        )
        pins, node_def = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

        with (
            patch.dict(DiffusionLoaderNode._MAP_KEY_TO_MODEL_CLASS, {self.model_key: mock_model_cls}),
            patch.object(DiffusionLoaderNode, "_load_state_dict", return_value=mock_state_dict),
        ):
            self.node.run(pins, context=context)

        # ── assert: model 列表被正确写入 ModelPool ──
        expected_key = ModelPoolKey(node_def.node_id, self.model_key)
        stored = self.model_pool.get_model(expected_key)
        self.assertIsInstance(stored, list)
        self.assertEqual(len(stored), 2)
        self.assertIs(stored[0], mock_model_a)
        self.assertIs(stored[1], mock_model_b)

    @patch("kdit.nodes.loaders.diffusion_model_loader.PinnedMemoryManager")
    @patch("kdit.nodes.loaders.diffusion_model_loader.build_ops")
    @patch("kdit.nodes.loaders.diffusion_model_loader.load_default_settings")
    @patch("kdit.nodes.loaders.diffusion_model_loader.is_file_or_dir", return_value=True)
    def test_run_with_i2v_model_key(self, _mock_is_file, mock_load_settings, mock_build_ops, _mock_pmm_cls):
        """验证 I2V model_key 也能正确写入 ModelPool。"""
        self.node._factory_model_key = ModelKey.Wan2_2_I2V_14B
        mock_load_settings.return_value = MagicMock()
        mock_build_ops.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.preprocess_model_state_dict.side_effect = lambda sd: sd
        mock_model.to.return_value = mock_model

        model_config = self._make_model_config()
        context = NodeContext(
            prompt=["test"],
            metadata={
                "model_path": "/fake/i2v_model.safetensors",
                "model_config": model_config,
            },
            device=self.device_info,
        )
        pins, node_def = _make_loader_pins(model_key=ModelKey.Wan2_2_I2V_14B, model_pool=self.model_pool)

        with (
            patch.dict(
                DiffusionLoaderNode._MAP_KEY_TO_MODEL_CLASS,
                {ModelKey.Wan2_2_I2V_14B: MagicMock(return_value=mock_model)},
            ),
            patch.object(DiffusionLoaderNode, "_load_state_dict", return_value={"w": torch.zeros(1)}),
        ):
            self.node.run(pins, context=context)

        expected_key = ModelPoolKey(node_def.node_id, ModelKey.Wan2_2_I2V_14B)
        stored_model = self.model_pool.get_model(expected_key)
        self.assertIs(stored_model, mock_model)

    def test_run_raises_on_invalid_model_path(self):
        """model_path 不存在时应抛出 ValueError。"""
        model_config = self._make_model_config()
        context = NodeContext(
            prompt=["test"],
            metadata={
                "model_path": "/nonexistent",
                "model_config": model_config,
            },
            device=self.device_info,
        )
        pins, _ = _make_loader_pins(model_key=self.model_key, model_pool=self.model_pool)

        with self.assertRaises(ValueError):
            self.node.run(pins, context=context)


if __name__ == "__main__":
    unittest.main()
