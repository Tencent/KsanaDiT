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

import threading


def singleton(cls):
    """线程安全的单例装饰器"""
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    for attr_name in dir(cls):
        if not attr_name.startswith("__"):
            attr_value = getattr(cls, attr_name)
            if callable(attr_value) or isinstance(attr_value, (staticmethod, classmethod)):
                try:
                    setattr(get_instance, attr_name, attr_value)
                except (AttributeError, TypeError):
                    pass
    return get_instance
