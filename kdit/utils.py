import numpy as np
import torch
import logging
import time
import sys

import cProfile
import pstats

import torch.cuda.nvtx as nvtx

from pyinstrument import Profiler

log = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s |kDit| %(levelname)s|%(filename)s:%(lineno)d|%(funcName)s| %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)  # 应用格式
log.addHandler(console_handler)

# def _init_logging(rank):
#     # logging
#     if rank == 0:
#         # set format
#         logging.basicConfig(
#             level=logging.INFO,
#             format="[%(asctime)s] %(levelname)s | kDit |: %(message)s",
#             handlers=[logging.StreamHandler(stream=sys.stdout)])
#     else:
#         logging.basicConfig(level=logging.ERROR)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        # else:
        # instances[cls].clean()
        # instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


global g_cprofiler
g_cprofiler = cProfile.Profile()


class cProfiler:
    def __init__(self, name=None):
        self.name = name if name else "vProfiler"
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


global g_vprofiler
g_vprofiler = Profiler()


class vProfiler:
    def __init__(self, name=None):
        self.name = name if name else "vProfiler"
        self.profiler = g_vprofiler

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


def print_recursive(obj, indent=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{' ' * indent}{k}:")
            print_recursive(v, indent + 2)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i, v in enumerate(obj):
            print(f"{' ' * indent}[{i}]:")
            print_recursive(v, indent + 2)
    else:
        if hasattr(obj, "shape"):
            s = f"{' ' * indent}(type:{type(obj)}) shape {obj.shape}"
        else:
            s = f"{' ' * indent}(type:{type(obj)}){obj}"
        if hasattr(obj, "dtype"):
            s = f"{s}, dtype={obj.dtype}"
        if hasattr(obj, "device"):
            s = f"{s}, device={obj.device}"
        if isinstance(obj, torch.Tensor):
            on_cpu = obj.cpu()
            s = f"{s}, abs_mean={on_cpu.abs().mean()}, max={on_cpu.max()}, min={on_cpu.min()}, abs_min={on_cpu.abs().min()}"
        print(s)


def get_gpu_count():
    if not torch.cuda.is_available():
        return 0
    else:
        return torch.cuda.device_count()


if __name__ == "__main__":
    print_recursive({"a": 1, "b": np.array([1, 2, 3])})
