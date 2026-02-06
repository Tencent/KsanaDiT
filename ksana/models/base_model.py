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

from abc import ABC, abstractmethod

from .model_key import KsanaModelKey


class KsanaModel(ABC):
    def __init__(self, model_key: KsanaModelKey, default_settings):
        self._model_key = model_key
        self._default_settings = default_settings

    @abstractmethod
    def to(self, *args, **kwargs):
        pass

    @property
    def model_key(self) -> KsanaModelKey:
        return self._model_key

    @property
    def default_settings(self):
        return self._default_settings
