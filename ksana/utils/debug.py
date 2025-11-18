import torch


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


# TODO: add unit tests
if __name__ == "__main__":
    import numpy as np

    print_recursive({"a": 1, "b": np.array([1, 2, 3])})
