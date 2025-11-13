import torch.distributed as dist


def init_distributed_group():
    """r initialize sequence parallel group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()
