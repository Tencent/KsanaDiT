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

import os

from omegaconf import OmegaConf


def _cache_yaml_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "settings", "cache", filename)


def load_cache_yaml_keys(filename, keys, safe=False):
    path = _cache_yaml_path(filename)
    if safe:
        if not os.path.exists(path):
            return [{}] * len(keys)
        try:
            conf = OmegaConf.load(path)
        except (OSError, ValueError):
            return [{}] * len(keys)
    else:
        conf = OmegaConf.load(path)

    out = []
    for k in keys:
        v = conf.get(k)
        if v is None:
            if safe:
                out.append({})
                continue
            raise RuntimeError(f"missing {k} in {path}")
        out.append(OmegaConf.to_container(v, resolve=True))
    return out


def load_cache_yaml_keys_safe(filename, keys):
    return load_cache_yaml_keys(filename, keys, safe=True)


def save_cache_yaml_key(filename, key, value):
    path = _cache_yaml_path(filename)
    if os.path.exists(path):
        conf = OmegaConf.load(path)
    else:
        conf = OmegaConf.create({})
    OmegaConf.update(conf, key, value, merge=True)
    with open(path, "w") as f:
        OmegaConf.save(conf, f)
