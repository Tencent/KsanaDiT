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

from kdit.utils.factory import SimpleFactory


class GeneratorFactory(SimpleFactory):
    """Generator 一级工厂 — 只按 model_key 注册。
    - register(model_key_list): 注册 Generator 类
    - create(model_key): 创建 Generator 实例，自动设置 model_key 属性
    """

    @classmethod
    def create(cls, model_key, *args, **kwargs):
        obj = super().create(model_key, *args, **kwargs)
        setattr(obj, "model_key", model_key)
        return obj
