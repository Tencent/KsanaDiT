import torch.cuda.nvtx as nvtx
from pyinstrument import Profiler
import cProfile
from .logger import log
import time
import pstats
import torch
import os
import csv
from .env import KSANA_MEMORY_PROFILER

global g_cprofiler
g_cprofiler = cProfile.Profile()


class cProfiler:
    def __init__(self, name=None):
        self.name = name if name else "ksanaProfiler"
        self.pr = g_cprofiler

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


global g_ksana_profiler
g_ksana_profiler = Profiler()


class ksanaProfiler:
    def __init__(self, name=None):
        self.name = name if name else "ksanaProfiler"
        self.profiler = g_ksana_profiler

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


# TODO: implement time_range decorator
# class time_range:
#     def __init__(self, func_or_name=None, print_func=log.info):
#         """
#         :param func_or_name: 可以是被装饰的函数（无参装饰器用法），
#                              也可以是任务名称（有参装饰器用法或with用法）
#         :param print_func: 用于输出信息的函数，默认为 print，可以传入 logger.info 等
#         """
#         self.print_func = print_func
#         self.start_time = None
#         self.func = None
#         self.name = "Task"

#         # 逻辑判断：区分是 @Timer 还是 @Timer(...) / with Timer(...)
#         if callable(func_or_name):
#             # 情况 1: @Timer (无括号，func_or_name 是被装饰的函数)
#             self.func = func_or_name
#             self.name = func_or_name.__name__
#             # 让 Timer 实例看起来像原函数（保留元数据）
#             functools.update_wrapper(self, func_or_name)
#         elif func_or_name is not None:
#             # 情况 2: @Timer('name') 或 with Timer('name')
#             self.name = func_or_name

#     def __enter__(self):
#         """上下文管理器入口"""
#         self.start_time = time.perf_counter()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """上下文管理器出口"""
#         end_time = time.perf_counter()
#         elapsed = end_time - self.start_time
#         self.print_func(f"[{self.name}] takes {elapsed:.6f} s")
#         return False

#     def __call__(self, *args, **kwargs):
#         """装饰器入口"""
#         # 场景 A: 之前已经是 @Timer (无参)，self.func 已经保存了函数
#         if self.func:
#             # 直接执行包裹逻辑
#             with self:
#                 return self.func(*args, **kwargs)

#         # 场景 B: 之前是 @Timer(...) (有参)，现在传入的是被装饰的函数
#         # 此时 args[0] 应该是被装饰的函数
#         func = args[0]

#         @functools.wraps(func)
#         def wrapper(*w_args, **w_kwargs):
#             if self.name == "Task":
#                 self.name = func.__name__
#             with self:
#                 return func(*w_args, **w_kwargs)

#         return wrapper


def time_range(func):
    def wrapped_fn(*args, **kwargs):
        start = time.time()
        ret_val = func(*args, **kwargs)
        stop = time.time()
        log.info(f"FUNC[{func.__qualname__}] takes {(stop - start):.2f} seconds")
        return ret_val

    return wrapped_fn


class nvtx_range:
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


class MemoryProfiler:
    enabled = KSANA_MEMORY_PROFILER

    @staticmethod
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
            f"[{project_name}] Memory usage recorded for tag '{tag}': Allocated={allocated/1024**3:.2f}GB, Reserved={reserved/1024**3:.2f}GB, Max_Allocated={max_allocated/1024**3:.2f}GB"
        )
