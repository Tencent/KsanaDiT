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


class GeneratorFactory:
    """Generator 一级工厂 — 只按 model_key 注册。

    - register(model_key_list): 注册 Generator 类
    - create(model_key): 创建 Generator 实例，自动设置 model_key 属性
    """

    _registry: dict = {}

    @classmethod
    def register(cls, model_key_list):
        if not isinstance(model_key_list, (list, tuple)):
            model_key_list = [model_key_list]

        def wrapper(wrapped_class):
            for model_key in model_key_list:
                if model_key in cls._registry:
                    print(f"{model_key} has already been registered in {cls.__name__}, please check!")
                cls._registry[model_key] = wrapped_class
            return wrapped_class

        return wrapper

    @classmethod
    def create(cls, model_key, *args, **kwargs):
        if model_key not in cls._registry:
            raise KeyError(f"{model_key} is not registered in {cls.__name__}")
        obj = cls._registry[model_key](*args, **kwargs)
        obj.model_key = model_key
        return obj
