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

import unittest

from kdit.utils.factory import AdvancedFactory, SimpleFactory

# ──────────────────────────────────────────────────────────────────────
# 二级工厂 (AdvancedFactory) 测试 fixtures
# ──────────────────────────────────────────────────────────────────────


class FactoryA(AdvancedFactory):
    pass


class FactoryB(AdvancedFactory):
    pass


@FactoryA.register("group_a", "key_a")
class FactoryAGroupAKeyA:
    def func(self):
        return "a_a_a"


@FactoryA.register("group_a", "key_b")
class FactoryAGroupAKeyB:
    def func(self):
        return "a_a_b"


@FactoryA.register("group_a", ["key_c", "key_d"])
class FactoryAGroupAKeyX:
    def func(self):
        return "a_a_x"


@FactoryA.register("group_b", "key_a")
@FactoryA.register("group_a", "key_y")
class FactoryAGroupBKeyA:
    def func(self):
        return "a_b_a"


@FactoryA.register("group_b", ["key_b", "key_c", "key_d"])
class FactoryAGroupBKeyX:
    def func(self):
        return "a_b_x"


@FactoryB.register("group_a", "key_a")
class FactoryBGroupAKeyA:
    def func(self):
        return "b_a_a"


# ──────────────────────────────────────────────────────────────────────
# 一级工厂 (SimpleFactory) 测试 fixtures
# ──────────────────────────────────────────────────────────────────────


class SimpleA(SimpleFactory):
    _default_level_1 = "SimpleA"


class SimpleB(SimpleFactory):
    _default_level_1 = "SimpleB"


class SimpleNoName(SimpleFactory):
    """不设置 _default_unit_type，应回退到 cls.__name__。"""

    pass


@SimpleA.register("alpha")
class AlphaImpl:
    def value(self):
        return "alpha"


@SimpleA.register(["beta", "gamma"])
class BetaGammaImpl:
    def value(self):
        return "beta_gamma"


@SimpleB.register("alpha")
class SimpleBAlphaImpl:
    def value(self):
        return "b_alpha"


@SimpleNoName.register("x")
class SimpleNoNameX:
    def value(self):
        return "no_name_x"


# ──────────────────────────────────────────────────────────────────────
# 模拟 SimpleFactory 一级注册行为的 fixture
# ──────────────────────────────────────────────────────────────────────


class MockGeneratorFactory(SimpleFactory):
    """模拟一级注册 Factory 的 create 行为（设置 model_key 属性）。"""

    _default_level_1 = "MockGenerator"

    @classmethod
    def create(cls, model_key, *args, **kwargs):
        obj = super().create(model_key, *args, **kwargs)
        obj.model_key = model_key
        return obj


class _FakeGenerator:
    def __init__(self):
        self.model_key = None

    def run(self):
        return f"gen_{self.model_key}"


MockGeneratorFactory.register("model_a")(_FakeGenerator)
MockGeneratorFactory.register(["model_b", "model_c"])(_FakeGenerator)


# ──────────────────────────────────────────────────────────────────────
# 模拟 LoaderNodeFactory 行为的 fixture
# ──────────────────────────────────────────────────────────────────────


class MockLoaderNodeFactory(SimpleFactory):
    _default_level_1 = "MockLoaderNode"


class _FakeLoaderNode:
    def __init__(self):
        self.loaded = True


MockLoaderNodeFactory.register("loader_a")(_FakeLoaderNode)
MockLoaderNodeFactory.register(["loader_b", "loader_c"])(_FakeLoaderNode)


# ──────────────────────────────────────────────────────────────────────
# 模拟 PipelineFactory 行为的 fixture
# ──────────────────────────────────────────────────────────────────────


class MockPipelineFactory(SimpleFactory):
    _default_level_1 = "MockPipeline"


class _FakePipeline:
    def __init__(self, name="default"):
        self.name = name


MockPipelineFactory.register("pipe_a")(_FakePipeline)
MockPipelineFactory.register(["pipe_b", "pipe_c"])(_FakePipeline)


# ======================================================================
# 测试用例
# ======================================================================


class TestFactory(unittest.TestCase):
    """二级工厂 (AdvancedFactory) 测试。"""

    def test_factory_group(self):
        self.assertEqual(FactoryA.create("group_a", "key_a").func(), "a_a_a")
        self.assertEqual(FactoryA.create("group_a", "key_b").func(), "a_a_b")

        self.assertEqual(FactoryA.create("group_b", "key_a").func(), "a_b_a")
        self.assertEqual(FactoryA.create("group_a", "key_y").func(), "a_b_a")

    def test_factory_key_list(self):
        self.assertEqual(FactoryA.create("group_a", "key_c").func(), "a_a_x")
        self.assertEqual(FactoryA.create("group_a", "key_d").func(), "a_a_x")

        self.assertEqual(FactoryA.create("group_b", "key_b").func(), "a_b_x")
        self.assertEqual(FactoryA.create("group_b", "key_c").func(), "a_b_x")
        self.assertEqual(FactoryA.create("group_b", "key_d").func(), "a_b_x")

    def test_diff_factory(self):
        self.assertEqual(FactoryA.create("group_a", "key_a").func(), "a_a_a")
        self.assertEqual(FactoryB.create("group_a", "key_a").func(), "b_a_a")

    def test_not_exsit(self):
        with self.assertRaises(KeyError):
            FactoryA.create("group_a", "x")
        with self.assertRaises(KeyError):
            FactoryA.create("x", "x")


class TestSimpleFactory(unittest.TestCase):
    """一级工厂 (SimpleFactory) 基础测试。"""

    def test_inherits_factory(self):
        """SimpleFactory 是 AdvancedFactory 的子类。"""
        self.assertTrue(issubclass(SimpleFactory, AdvancedFactory))
        self.assertTrue(issubclass(SimpleA, AdvancedFactory))

    def test_register_single_key(self):
        obj = SimpleA.create("alpha")
        self.assertEqual(obj.value(), "alpha")

    def test_register_key_list(self):
        self.assertEqual(SimpleA.create("beta").value(), "beta_gamma")
        self.assertEqual(SimpleA.create("gamma").value(), "beta_gamma")

    def test_isolation_between_subclasses(self):
        """不同 SimpleFactory 子类的 _registry 互相隔离。"""
        self.assertEqual(SimpleA.create("alpha").value(), "alpha")
        self.assertEqual(SimpleB.create("alpha").value(), "b_alpha")

    def test_not_exist(self):
        with self.assertRaises(KeyError):
            SimpleA.create("nonexistent")

    def test_default_unit_type_fallback(self):
        """未设置 _default_level_1 时回退到 cls.__name__。"""
        obj = SimpleNoName.create("x")
        self.assertEqual(obj.value(), "no_name_x")

        with self.assertRaises(KeyError):
            SimpleNoName.create("missing")

    def test_get_registered_keys(self):
        keys = SimpleA.get_registered_keys()
        self.assertIn("alpha", keys)
        self.assertIn("beta", keys)
        self.assertIn("gamma", keys)

    def test_create_with_args(self):
        """SimpleFactory.create 应透传 *args/**kwargs 给构造函数。"""

        class ArgsFactory(SimpleFactory):
            _default_level_1 = "ArgsFactory"

        @ArgsFactory.register("with_args")
        class WithArgs:  # pylint: disable=unused-variable
            def __init__(self, x, y=10):
                self.x = x
                self.y = y

        obj = ArgsFactory.create("with_args", 42, y=99)
        self.assertEqual(obj.x, 42)
        self.assertEqual(obj.y, 99)

    def test_internal_registry_uses_unit_type(self):
        """SimpleFactory 内部通过 _default_level_1 作为 AdvancedFactory 的第一级 key 存储。"""
        self.assertIn("SimpleA", SimpleA._registry)
        self.assertIsInstance(SimpleA._registry["SimpleA"], dict)
        self.assertIn("alpha", SimpleA._registry["SimpleA"])


class TestMockGeneratorFactory(unittest.TestCase):
    """模拟一级注册 Factory 的 create 行为测试。"""

    def test_create_sets_model_key(self):
        gen = MockGeneratorFactory.create("model_a")
        self.assertEqual(gen.model_key, "model_a")
        self.assertEqual(gen.run(), "gen_model_a")

    def test_create_with_key_list(self):
        gen_b = MockGeneratorFactory.create("model_b")
        gen_c = MockGeneratorFactory.create("model_c")
        self.assertEqual(gen_b.model_key, "model_b")
        self.assertEqual(gen_c.model_key, "model_c")

    def test_not_exist(self):
        with self.assertRaises(KeyError):
            MockGeneratorFactory.create("missing")


class TestMockLoaderNodeFactory(unittest.TestCase):
    """模拟 LoaderNodeFactory 测试。"""

    def test_create(self):
        node = MockLoaderNodeFactory.create("loader_a")
        self.assertTrue(node.loaded)

    def test_create_with_key_list(self):
        node_b = MockLoaderNodeFactory.create("loader_b")
        node_c = MockLoaderNodeFactory.create("loader_c")
        self.assertTrue(node_b.loaded)
        self.assertTrue(node_c.loaded)

    def test_not_exist(self):
        with self.assertRaises(KeyError):
            MockLoaderNodeFactory.create("missing")

    def test_isolation_from_generator(self):
        """LoaderNodeFactory 和 MockGeneratorFactory 的 registry 互不干扰。"""
        self.assertNotIn("model_a", MockLoaderNodeFactory.get_registered_keys())
        self.assertNotIn("loader_a", MockGeneratorFactory.get_registered_keys())


class TestMockPipelineFactory(unittest.TestCase):
    """模拟 PipelineFactory 测试。"""

    def test_create_default(self):
        pipe = MockPipelineFactory.create("pipe_a")
        self.assertEqual(pipe.name, "default")

    def test_create_with_kwargs(self):
        pipe = MockPipelineFactory.create("pipe_a", name="custom")
        self.assertEqual(pipe.name, "custom")

    def test_create_with_key_list(self):
        pipe_b = MockPipelineFactory.create("pipe_b")
        pipe_c = MockPipelineFactory.create("pipe_c")
        self.assertIsInstance(pipe_b, _FakePipeline)
        self.assertIsInstance(pipe_c, _FakePipeline)

    def test_not_exist(self):
        with self.assertRaises(KeyError):
            MockPipelineFactory.create("missing")

    def test_get_registered_keys(self):
        keys = MockPipelineFactory.get_registered_keys()
        self.assertIn("pipe_a", keys)
        self.assertIn("pipe_b", keys)
        self.assertIn("pipe_c", keys)

    def test_isolation_from_others(self):
        """PipelineFactory 的 registry 与其他工厂隔离。"""
        self.assertNotIn("alpha", MockPipelineFactory.get_registered_keys())
        self.assertNotIn("model_a", MockPipelineFactory.get_registered_keys())
        self.assertNotIn("loader_a", MockPipelineFactory.get_registered_keys())


class TestDuplicateRegistration(unittest.TestCase):
    """重复注册应打印警告但不抛异常，后注册覆盖前注册。"""

    def test_simple_factory_duplicate(self):
        class DupFactory(SimpleFactory):
            _default_level_1 = "DupFactory"

        @DupFactory.register("dup_key")
        class First:  # pylint: disable=unused-variable
            def val(self):
                return 1

        @DupFactory.register("dup_key")
        class Second:  # pylint: disable=unused-variable
            def val(self):
                return 2

        # 后注册覆盖前注册
        self.assertEqual(DupFactory.create("dup_key").val(), 2)

    def test_factory_duplicate(self):
        class DupFactory2(AdvancedFactory):
            pass

        @DupFactory2.register("grp", "k")
        class First2:  # pylint: disable=unused-variable
            def val(self):
                return 1

        @DupFactory2.register("grp", "k")
        class Second2:  # pylint: disable=unused-variable
            def val(self):
                return 2

        self.assertEqual(DupFactory2.create("grp", "k").val(), 2)


class TestRegistryIsolation(unittest.TestCase):
    """确保 SimpleFactory 子类之间 _registry 完全隔离。"""

    def test_fresh_subclass_has_empty_registry(self):
        class FreshFactory(SimpleFactory):
            _default_level_1 = "FreshFactory"

        self.assertEqual(FreshFactory.get_registered_keys(), [])

    def test_register_does_not_leak(self):
        class IsoA(SimpleFactory):
            _default_level_1 = "IsoA"

        class IsoB(SimpleFactory):
            _default_level_1 = "IsoB"

        @IsoA.register("only_in_a")
        class OnlyA:  # pylint: disable=unused-variable
            pass

        self.assertIn("only_in_a", IsoA.get_registered_keys())
        self.assertNotIn("only_in_a", IsoB.get_registered_keys())


if __name__ == "__main__":
    unittest.main()
