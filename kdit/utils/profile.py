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

from __future__ import annotations

import cProfile
import csv
import functools
import os
import pstats
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.cuda.nvtx as nvtx
from pyinstrument import Profiler

from .env import KSANA_MEMORY_PROFILER, KSANA_PROFILE, KSANA_PROFILE_CUDA_SYNC
from .logger import log

global G_CPROFILER
G_CPROFILER = cProfile.Profile()


class CProfiler:
    def __init__(self, name=None):
        self.name = name if name else "kditProfiler"
        self.pr = G_CPROFILER

    def __enter__(self):
        self.start = time.time()
        self.pr.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.time()
        log.info(f"PROF[{self.name}] takes {(self.stop - self.start):.2f} seconds")
        self.pr.disable()

        stats = pstats.Stats(self.pr).sort_stats("cumulative")
        stats.print_stats(10)

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
    """纯计时器 — 支持 with 语句和装饰器。"""

    default_name = "Task"

    def __init__(self, name: str = None, print_func: Callable[[str], None] = log.info):
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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class TimerProfiler(Timer):
    """带层级 profile 能力的计时器。

    继承 Timer 的计时 + log 打印，额外在 ``KSANA_PROFILE=1`` 时
    将计时节点挂到 :class:`TimeProfiler` 层级树上。

    ``TimeProfiler`` 实例通过 ``threading.local`` 线程局部存储，
    每次 ``__enter__`` 动态查询当前 session，无 session 时退化为纯 Timer。
    """

    def __init__(self, name: str = None, print_func: Callable[[str], None] = log.info, *, note: str | None = None):
        super().__init__(name=name, print_func=print_func)
        self.note = note
        self._profiler: TimeProfiler | None = None

    def __enter__(self):
        if KSANA_PROFILE:
            self._profiler = TimeProfiler.get_current()
        if self._profiler is not None:
            self._profiler.begin(self.name, self.note)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profiler is not None:
            self._profiler.end()
            self._profiler = None
        return super().__exit__(exc_type, exc_val, exc_tb)


def time_profile(
    func_or_name: Callable | str | None = None,
    print_func: Callable[[str], None] = log.info,
    *,
    note: str | None = None,
    profile: bool = True,
):
    """统一计时入口 — 支持装饰器和 with 语句。

    Args:
        func_or_name: 被装饰的函数（无参装饰器）或名称字符串。
        print_func: 计时结果的打印函数。
        note: 层级树节点的附加备注（仅 *profile=True* 且有活跃 session 时生效），
              如 ``note="t=1.0"`` 会在树中显示为 ``step_0  0.5s  (t=1.0)``。
        profile: 是否参与层级 profile 树（默认 True）。
                 设为 False 则使用纯 Timer，不出现在 TimeProfiler 树中。

    用法::

        @time_profile
        def func(): ...

        @time_profile("custom_name")
        def func(): ...

        with time_profile("step_0", note="t=1.0"):
            ...

        @time_profile(profile=False)
        def internal_helper(): ...
    """
    if func_or_name is None or isinstance(func_or_name, str):
        name = func_or_name if isinstance(func_or_name, str) else Timer.default_name
        if profile:
            return TimerProfiler(name=name, print_func=print_func, note=note)
        else:
            return Timer(name=name, print_func=print_func)
    elif callable(func_or_name):
        if profile:
            timer = TimerProfiler(name=func_or_name.__name__, print_func=print_func, note=note)
        else:
            timer = Timer(name=func_or_name.__name__, print_func=print_func)
        return timer(func_or_name)
    else:
        raise TypeError("Invalid argument type for time_profile")


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
        self._func = None

    def _should_skip(self):
        """检查是否需要跳过 NVTX 记录"""
        return self.skip

    def __enter__(self):
        if not self._should_skip():
            nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._should_skip():
            nvtx.range_pop()

    def __call__(self, func=None, *, name=None):
        """装饰器实现"""
        if func is None:
            return lambda f: self.__call__(f, name=name)

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


# TODO(qian): MemoryProfiler could be memory_profile, like time_profile, nvtx_range
class MemoryProfiler:
    enabled = KSANA_MEMORY_PROFILER

    @staticmethod
    # TODO(qian): this method is very un-pythonic, change it to a more pythonic way
    def record_memory(tag: str, project_name: str = "kDiT"):
        """
        记录内存使用情况到CSV文件
        CSV格式: project_name, tag, allocated_memory, reserved_memory, max_allocated_memory

        Args:
            tag: 内存记录标签
            project_name: 项目名称，用于CSV文件中的第一列和文件名
        """
        if not MemoryProfiler.enabled:
            return

        if not torch.cuda.is_available():
            log.warn(f"CUDA not available, skipping memory record for tag: {tag}")
            return

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()

        csv_file_path = f"{project_name.lower()}_memory_usage.csv"

        file_exists = os.path.exists(csv_file_path)

        with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(
                    ["Project", "Tag", "Allocated_Memory_GB", "Reserved_Memory_GB", "Max_Allocated_Memory_GB"]
                )

            writer.writerow(
                [
                    project_name,
                    tag,
                    allocated / (1024**3),
                    reserved / (1024**3),
                    max_allocated / (1024**3),
                ]
            )

        log.info(
            f"[{project_name}] Memory usage recorded for tag '{tag}': Allocated={allocated/1024**3:.2f}GB, "
            f"Reserved={reserved/1024**3:.2f}GB, Max_Allocated={max_allocated/1024**3:.2f}GB"
        )


# -- Hierarchical Profiler ------------------------------------------------


def _cuda_sync_if_needed():
    """在 KSANA_PROFILE_CUDA_SYNC 启用时执行 CUDA 同步。"""
    if KSANA_PROFILE_CUDA_SYNC:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (AssertionError, RuntimeError):
            # macOS / CPU-only torch: is_available() 可能返回 True 但 synchronize() 失败
            pass


@dataclass
class ProfileNode:
    """树形 profiler 中的一个计时节点。"""

    name: str
    elapsed: float = 0.0
    note: str | None = None
    children: list[ProfileNode] = field(default_factory=list)
    parent: ProfileNode | None = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)


class TimeProfiler:
    """层级式 Profiler — 线程局部单例，通过栈维护父子关系。

    用法::

        profiler = TimeProfiler.start_session("pipeline_generate")
        with time_profile("step_0", note="timestep=1.0"):
            with time_profile("model_forward"):
                ...
        profiler.finish()
        profiler.print_summary()
    """

    _local = threading.local()

    def __init__(self, name: str):
        self._root = ProfileNode(name=name)
        self._current = self._root
        self._stack: list[ProfileNode] = [self._root]
        self._root._start_time = time.perf_counter()
        self._finished = False

    @classmethod
    def start_session(cls, name: str) -> TimeProfiler:
        """启动一个新的 profiler session（线程局部）。"""
        profiler = cls(name)
        cls._local.current_profiler = profiler
        return profiler

    @classmethod
    def get_current(cls) -> TimeProfiler | None:
        """获取当前线程的活跃 profiler，无则返回 None。"""
        return getattr(cls._local, "current_profiler", None)

    def begin(self, name: str, note: str | None = None) -> None:
        """压栈：开始一个新的计时区间。"""
        if self._finished:
            return
        node = ProfileNode(name=name, note=note, parent=self._current)
        self._current.children.append(node)
        self._stack.append(node)
        self._current = node
        _cuda_sync_if_needed()
        node._start_time = time.perf_counter()

    def end(self) -> float:
        """弹栈：结束当前计时区间，返回耗时（秒）。"""
        if self._finished or len(self._stack) <= 1:
            return 0.0
        _cuda_sync_if_needed()
        elapsed = time.perf_counter() - self._current._start_time
        self._current.elapsed = elapsed
        self._stack.pop()
        self._current = self._stack[-1]
        return elapsed

    def finish(self) -> ProfileNode:
        """结束整个 session，返回根节点。"""
        if self._finished:
            return self._root
        while len(self._stack) > 1:
            self.end()
        _cuda_sync_if_needed()
        self._root.elapsed = time.perf_counter() - self._root._start_time
        self._finished = True
        if getattr(self._local, "current_profiler", None) is self:
            self._local.current_profiler = None
        return self._root

    def print_summary(self, file=None) -> None:
        """打印树形摘要到 file（默认 sys.stdout）。"""
        out = file or sys.stdout
        lines = [""]
        lines.append("KsanaDiT Profile Summary")
        lines.append("=" * 60)
        self._format_node(self._root, lines, prefix="", is_last=True, is_root=True)
        lines.append("=" * 60)
        lines.append("")
        out.write("\n".join(lines) + "\n")

    def _format_node(
        self,
        node: ProfileNode,
        lines: list[str],
        prefix: str,
        is_last: bool,
        is_root: bool = False,
    ) -> None:
        COL_WIDTH = 55  # pylint: disable=invalid-name

        if is_root:
            name_part = node.name
        else:
            connector = "└── " if is_last else "├── "
            name_part = prefix + connector + node.name

        time_str = f"{node.elapsed:.3f}s"
        note_str = f"  ({node.note})" if node.note else ""

        padding = max(1, COL_WIDTH - len(name_part))
        line = f"{name_part}{' ' * padding}{time_str}{note_str}"
        lines.append(line)

        if is_root:
            child_prefix = "  "
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(node.children):
            child_is_last = i == len(node.children) - 1
            self._format_node(child, lines, child_prefix, child_is_last)
