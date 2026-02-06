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

import torch


def print_recursive(obj, print_func=print, indent=0):
    s = f"{' ' * indent}(type:{type(obj)})"
    if isinstance(obj, int | float | str | bool):
        s = f"{s}: {obj}"
        print_func(s)
        return
    if hasattr(obj, "shape"):
        s = f"{s}, shape={obj.shape}"
    if hasattr(obj, "dtype"):
        s = f"{s}, dtype={obj.dtype}"
    if hasattr(obj, "device"):
        s = f"{s}, device={obj.device}"
    if isinstance(obj, torch.Tensor):
        on_cpu = obj.cpu()
        max_value = on_cpu.max()
        min_value = on_cpu.min()
        abs_min_value = on_cpu.abs().min()
        s = f"{s}, max={max_value:.6f}, min={min_value:.6f}, abs_min={abs_min_value:.6f}"
        if isinstance(obj, torch.FloatTensor):
            abs_mean = on_cpu.abs().mean()
            s = f"{s}, abs_mean={abs_mean:.6f}"
        print_func(s)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            print_func(f"{' ' * indent}{k}:")
            print_recursive(v, print_func, indent + 2)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i, v in enumerate(obj):
            print_func(f"{' ' * indent}[{i}]:")
            print_recursive(v, print_func, indent + 2)
    elif hasattr(obj, "__dict__"):
        for var_name in vars(obj):
            if var_name.startswith("_"):
                continue
            var_value = getattr(obj, var_name, None)
            print_func(f"{' ' * indent} {var_name}:")
            print_recursive(var_value, print_func, indent + 2)
    else:
        print_func(f"{' ' * indent} {obj}")
