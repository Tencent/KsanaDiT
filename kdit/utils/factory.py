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


class AdvancedFactory:
    """二级工厂 — 按 (level_1, level_2) 注册。

    子类自动获得独立的 ``_registry``（通过 ``__init_subclass__``）。
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, level_1, level_2_list):
        if not isinstance(level_2_list, (list, tuple)):
            level_2_list = [level_2_list]

        def wrapper(wrapped_class):
            if level_1 not in cls._registry:
                cls._registry[level_1] = {}
            for level_2 in level_2_list:
                if level_2 in cls._registry[level_1]:
                    print(f"{level_2} has already been registered in {level_1} for {cls.__name__}, please check!")
                cls._registry[level_1][level_2] = wrapped_class
                # print(f"---- {cls.__name__} registered {level_1}, {level_2} ----")
            return wrapped_class

        return wrapper

    @classmethod
    def create(cls, level_1, level_2, *args, **kwargs):
        if level_1 not in cls._registry:
            raise KeyError(f"{level_1} is not registered in {cls.__name__}")
        if level_2 not in cls._registry[level_1]:
            raise KeyError(f"{level_2} is not registered in {level_1} for {cls.__name__}")
        return cls._registry[level_1][level_2](*args, **kwargs)


class SimpleFactory(AdvancedFactory):
    """一级工厂 — ``AdvancedFactory`` 的单维特例，固定 ``level_1`` 为 ``_default_level_1``。

    继承 ``AdvancedFactory`` 的二级注册机制，但通过 ``_default_level_1`` 类属性自动填充
    第一级 key，对外只暴露单 key 的 ``register`` / ``create`` 接口。

    子类通过 ``_default_level_1`` 自定义第一级 key（默认取 ``cls.__name__``）。

    用法::

        class MyFactory(SimpleFactory):
            _default_level_1 = "my_type"   # 可选，默认为 cls.__name__

        @MyFactory.register("some_key")
        class Foo: ...

        obj = MyFactory.create("some_key")
    """

    _default_level_1: str = ""

    @classmethod
    def _get_level_1_type(cls) -> str:
        return cls._default_level_1 or cls.__name__

    @classmethod
    def register(cls, key_or_keys):
        """注册一个或多个 key 到被装饰的类（自动填充 level_1）。"""
        return super().register(cls._get_level_1_type(), key_or_keys)

    @classmethod
    def create(cls, key, *args, **kwargs):
        """根据 key 创建实例（自动填充 level_1）。"""
        return super().create(cls._get_level_1_type(), key, *args, **kwargs)

    @classmethod
    def get_registered_keys(cls):
        """返回所有已注册的 key。"""
        level_1 = cls._get_level_1_type()
        if level_1 not in cls._registry:
            return []
        return list(cls._registry[level_1].keys())
