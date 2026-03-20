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

"""GeneratorDef 注册表 + frozen dataclass 单元测试。

测试覆盖:
- 注册 / 查找 / 重复注册 / reset
- frozen dataclass 不可变性
- 字段存在性校验
"""

import dataclasses
import unittest

from kdit.generators.generator_def import (
    GeneratorDef,
    get_generator_def,
    register_generator_def,
    reset_generator_def_registry,
)
from kdit.generators.handlers.denoise_handler import DenoiseHandler
from kdit.generators.handlers.latent_handler import LatentHandler
from kdit.generators.handlers.text_handler import TextHandler
from kdit.models.model_key import ModelKey


class TestGeneratorDefRegistry(unittest.TestCase):
    """测试 GeneratorDef 注册表的注册、查找、重复注册和 reset。"""

    def setUp(self):
        reset_generator_def_registry()

    def tearDown(self):
        reset_generator_def_registry()

    def test_register_and_get(self):
        """注册一个 def，然后 get 回来。"""
        gdef = GeneratorDef(model_key=ModelKey.Wan2_2_T2V_14B)
        register_generator_def(gdef)
        result = get_generator_def(ModelKey.Wan2_2_T2V_14B)
        self.assertIs(result, gdef)

    def test_get_unregistered_raises(self):
        """查找未注册的 key 应 raise KeyError。"""
        with self.assertRaises(KeyError):
            get_generator_def(ModelKey.Wan2_2_T2V_14B)

    def test_duplicate_register_raises(self):
        """重复注册同一个 key 应 raise ValueError。"""
        gdef = GeneratorDef(model_key=ModelKey.Wan2_2_T2V_14B)
        register_generator_def(gdef)
        with self.assertRaises(ValueError):
            register_generator_def(gdef)

    def test_reset_registry(self):
        """reset 后 get 应该 raise KeyError。"""
        gdef = GeneratorDef(model_key=ModelKey.Wan2_2_T2V_14B)
        register_generator_def(gdef)
        reset_generator_def_registry()
        with self.assertRaises(KeyError):
            get_generator_def(ModelKey.Wan2_2_T2V_14B)

    def test_register_multiple_keys(self):
        """注册多个不同 key 应各自独立。"""
        gdef_t2v = GeneratorDef(model_key=ModelKey.Wan2_2_T2V_14B)
        gdef_i2v = GeneratorDef(model_key=ModelKey.Wan2_2_I2V_14B)
        register_generator_def(gdef_t2v)
        register_generator_def(gdef_i2v)
        self.assertIs(get_generator_def(ModelKey.Wan2_2_T2V_14B), gdef_t2v)
        self.assertIs(get_generator_def(ModelKey.Wan2_2_I2V_14B), gdef_i2v)

    def test_register_returns_def(self):
        """register_generator_def 应返回注册的 def 本身。"""
        gdef = GeneratorDef(model_key=ModelKey.Wan2_2_T2V_14B)
        result = register_generator_def(gdef)
        self.assertIs(result, gdef)


class TestGeneratorDefFrozen(unittest.TestCase):
    """测试 GeneratorDef frozen dataclass 的不可变性和字段。"""

    def test_immutable(self):
        """尝试修改属性应 raise FrozenInstanceError。"""
        gdef = GeneratorDef(model_key=ModelKey.Wan2_2_T2V_14B)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            gdef.model_key = ModelKey.Wan2_2_I2V_14B

    def test_fields(self):
        """验证 4 个字段存在: model_key, text_handler, latent_handler, denoise_handler。"""
        field_names = {f.name for f in dataclasses.fields(GeneratorDef)}
        expected = {
            "model_key",
            "text_handler",
            "latent_handler",
            "denoise_handler",
        }
        self.assertEqual(field_names, expected)

    def test_default_handlers(self):
        """默认 handler 应为基类实例。"""
        gdef = GeneratorDef(model_key=ModelKey.Wan2_2_T2V_14B)
        self.assertIsInstance(gdef.text_handler, TextHandler)
        self.assertIsInstance(gdef.latent_handler, LatentHandler)
        self.assertIsInstance(gdef.denoise_handler, DenoiseHandler)

    def test_is_dataclass(self):
        """GeneratorDef 应是 dataclass。"""
        self.assertTrue(dataclasses.is_dataclass(GeneratorDef))

    def test_is_frozen(self):
        """GeneratorDef 应是 frozen dataclass。"""
        self.assertTrue(dataclasses.is_dataclass(GeneratorDef))
        # frozen dataclass 的 __dataclass_params__.frozen 为 True
        params = GeneratorDef.__dataclass_params__
        self.assertTrue(params.frozen)


if __name__ == "__main__":
    unittest.main()
