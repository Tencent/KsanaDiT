import numpy as np
import torch
import logging
import time
import sys

import cProfile
import pstats

from pyinstrument import Profiler

log = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s |vDit| %(levelname)s|%(filename)s:%(lineno)d|%(funcName)s| %(message)s",
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
#             format="[%(asctime)s] %(levelname)s | vDit |: %(message)s",
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


def nvtx_range(func):
    def wrapped_fn(*args, **kwargs):
        ret_val = func(*args, **kwargs)
        # print(f"tjinfo: FUNC[{func.__qualname__}] takes {(stop - start):.2f} seconds")
        return ret_val

    return wrapped_fn


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
        print(s)


def get_gpu_count():
    if not torch.cuda.is_available():
        return 0
    else:
        return torch.cuda.device_count()


if __name__ == "__main__":
    print_recursive({"a": 1, "b": np.array([1, 2, 3])})
