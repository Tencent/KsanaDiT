import torch


def print_recursive(obj, print_func=print, indent=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            print_func(f"{' ' * indent}{k}:")
            print_recursive(v, print_func, indent + 2)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i, v in enumerate(obj):
            print_func(f"{' ' * indent}[{i}]:")
            print_recursive(v, print_func, indent + 2)
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
            max_value = on_cpu.max()
            min_value = on_cpu.min()
            abs_min_value = on_cpu.abs().min()
            s = f"{s}, max={max_value:.6f}, min={min_value:.6f}, abs_min={abs_min_value:.6f}"
            if isinstance(obj, torch.FloatTensor):
                abs_mean = on_cpu.abs().mean()
                s = f"{s}, abs_mean={abs_mean:.6f}"
        print_func(s)
