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


class Factory:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, unit_type, model_key_list):
        if not isinstance(model_key_list, (list, tuple)):
            model_key_list = [model_key_list]

        def wrapper(wrapped_class):
            if unit_type not in cls._registry:
                cls._registry[unit_type] = {}
            for model_key in model_key_list:
                if model_key in cls._registry[unit_type]:
                    print(f"{model_key} has already been registered in {unit_type} for {cls.__name__}, please check!")
                cls._registry[unit_type][model_key] = wrapped_class
                # print(f"---- {cls.__name__} registered {unit_type}, {model_key} ----")
            return wrapped_class

        return wrapper

    @classmethod
    def create(cls, unit_type, model_key, *args, **kwargs):
        if unit_type not in cls._registry:
            raise KeyError(f"{unit_type} is not registered in {cls.__name__}")
        if model_key not in cls._registry[unit_type]:
            raise KeyError(f"{model_key} is not registered in {unit_type} for {cls.__name__}")
        return cls._registry[unit_type][model_key](*args, **kwargs)
