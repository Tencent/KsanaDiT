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

import cProfile
import csv
import functools
import os
import pstats
import time
from collections.abc import Callable

import torch
import torch.cuda.nvtx as nvtx
from pyinstrument import Profiler

from .env import KSANA_MEMORY_PROFILER
from .logger import log

global G_CPROFILER
G_CPROFILER = cProfile.Profile()


class CProfiler:
    def __init__(self, name=None):
        self.name = name if name else "ksanaProfiler"
        self.pr = G_CPROFILER

    def __enter__(self):
        self.start = time.time()
        self.pr.enable()  # 开始分析

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.time()
        log.info(f"PROF[{self.name}] takes {(self.stop - self.start):.2f} seconds")
        self.pr.disable()  # 停止分析

        stats = pstats.Stats(self.pr).sort_stats("cumulative")
        stats.print_stats(10)  # 打印前10个函数，根据需要调整

    def dump(self, filename="profile_stats.prof"):
        self.pr.dump_stats(filename)


global G_KSANA_PROFILER
G_KSANA_PROFILER = Profiler()


class KsanaProfiler:
    def __init__(self, name=None):
        self.name = name if name else "KsanaProfiler"
        self.profiler = G_KSANA_PROFILER

    def __enter__(self):
        self.start = time.time()
        self.profiler.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.time()
        log.info(f"PROF[{self.name}] takes {(self.stop - self.start):.2f} seconds")
        self.profiler.stop()
        self.profiler.print()

    def dump(self, filename="profile_stats.prof"):
        with open("pyinstrument_report.html", "w") as f:
            f.write(self.profiler.output_html())


class Timer:
    default_name = "Task"

    def __init__(self, name: str = None, print_func: Callable[[str], None] = log.info):
        """support with and call"""
        self.name = name if name else self.default_name
        self.print_func = print_func
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        if self.start_time is not None:
            self.timer(end_time - self.start_time)
        return False

    def timer(self, elapsed):
        self.print_func(f"[{self.name}] takes {elapsed:.6f} s")

    def __call__(self, func: Callable):
        """
        被装饰器调用入口
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 调用 __enter__ 和 __exit__ 实现计时
            with self:
                return func(*args, **kwargs)

        return wrapper


def time_range(func_or_name: Callable | str | None = None, print_func: Callable[[str], None] = log.info):
    """
    support both no-args and args decorator, and with statement.

        usage1:
            @time_range
            def func(self, *args, **kwargs):
                pass
        usage2:
            @time_range("new_time_func_name", log.info)
            def func(self, *args, **kwargs):
                pass
        usage3:
            with time_range():
                some function call
        usage4:
            with time_range("new_time_func_name", print):
                some function call
    """
    if func_or_name is None or isinstance(func_or_name, str):
        # 情况 1 & 2: with time_range() / with time_range('name')
        # 情况 3: @time_range('name') (有参装饰器)

        # 返回 Timer 实例本身。如果是装饰器，Python 会调用它的 __call__
        name = func_or_name if isinstance(func_or_name, str) else Timer.default_name
        return Timer(name=name, print_func=print_func)
    elif callable(func_or_name):
        # 情况 4: @time_range (无参装饰器)
        # 此时 time_range(func) 被调用，func_or_name 是函数。
        # 1. 创建一个 Timer 实例 (Timer 的 __init__ 返回 None，安全)
        timer = Timer(name=func_or_name.__name__, print_func=print_func)
        # 2. 调用实例的 __call__ 方法，返回 wrapper 函数来替换原函数
        return timer(func_or_name)
    else:
        raise TypeError("Invalid argument type for time_range")


class nvtx_range:  # pylint: disable=invalid-name
    def __init__(self, name=None, skip=True):
        """
        支持上下文管理器和装饰器的 NVTX 范围工具

        参数:
        name (str): 范围名称
        skip_compile (bool): 当 torch.compile 激活时是否跳过 NVTX
        """
        self.name = name if name else "nvtx_range"
        self.skip = skip
        self._func = None  # 用于装饰器模式

    def _should_skip(self):
        """检查是否需要跳过 NVTX 记录"""
        return self.skip
        # return self.skip and hasattr(torch, 'is_compiling') and torch.is_compiling()

    def __enter__(self):
        if not self._should_skip():
            nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._should_skip():
            nvtx.range_pop()

    # TODO: test me
    def __call__(self, func=None, *, name=None):
        """装饰器实现"""
        if func is None:
            # 带参数装饰器: @nvtx_range(name="...")
            return lambda f: self.__call__(f, name=name)

        # 不带参数装饰器: @nvtx_range
        self._func = func
        self.name = name or self.name or func.__qualname__

        def wrapper(*args, **kwargs):
            if self._should_skip():
                return func(*args, **kwargs)

            nvtx.range_push(self.name)
            result = func(*args, **kwargs)
            nvtx.range_pop()
            return result

        return wrapper


# TODO(qian): MemoryProfiler cloud be memory_range, like time_range, nvtx_range
class MemoryProfiler:
    enabled = KSANA_MEMORY_PROFILER

    @staticmethod
    # TODO(qian): this method is very un-pythonic, change it to a more pythonic way
    def record_memory(tag: str, project_name: str = "KsanaDit"):
        """
        记录内存使用情况到CSV文件
        CSV格式: project_name, tag, allocated_memory, reserved_memory, max_allocated_memory

        Args:
            tag: 内存记录标签
            project_name: 项目名称，用于CSV文件中的第一列和文件名
        """
        # 检查是否启用了内存分析
        if not MemoryProfiler.enabled:
            return

        if not torch.cuda.is_available():
            log.warn(f"CUDA not available, skipping memory record for tag: {tag}")
            return

        # 获取当前内存使用情况
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()

        # 生成CSV文件路径
        csv_file_path = f"{project_name.lower()}_memory_usage.csv"

        # 记录到CSV文件
        file_exists = os.path.exists(csv_file_path)

        with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(
                    ["Project", "Tag", "Allocated_Memory_GB", "Reserved_Memory_GB", "Max_Allocated_Memory_GB"]
                )

            # 写入数据行
            writer.writerow(
                [
                    project_name,
                    tag,
                    allocated / (1024**3),  # 转换为GB
                    reserved / (1024**3),  # 转换为GB
                    max_allocated / (1024**3),  # 转换为GB
                ]
            )

        log.info(
            f"[{project_name}] Memory usage recorded for tag '{tag}': Allocated={allocated/1024**3:.2f}GB, "
            f"Reserved={reserved/1024**3:.2f}GB, Max_Allocated={max_allocated/1024**3:.2f}GB"
        )
