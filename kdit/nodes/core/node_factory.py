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

from kdit.utils.factory import AdvancedFactory, SimpleFactory


class LoaderNodeFactory(SimpleFactory):
    """Loader Node 工厂 — 纯注册表 + _factory_model_key 注入。

    - register(model_key_list): 注册 IONode 类
    - create(model_key): 创建 IONode 实例，自动注入 _factory_model_key
    """

    @classmethod
    def create(cls, model_key, *args, **kwargs):
        node = super().create(model_key, *args, **kwargs)
        node._factory_model_key = model_key
        return node


class InferNodeFactory(AdvancedFactory):
    """Infer Node 工厂 — 纯注册表 + _factory_model_key 注入。

    复用 AdvancedFactory 的 register() / create() 机制：
    - register(infer_node_type, model_key_list): 注册 InferNode 类
    - create(infer_node_type, model_key): 创建 InferNode 实例，自动注入 _factory_model_key
    """

    @classmethod
    def create(cls, infer_node_type, model_key, *args, **kwargs):
        node = super().create(infer_node_type, model_key, *args, **kwargs)
        node._factory_model_key = model_key
        return node
