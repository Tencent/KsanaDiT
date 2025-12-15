import torch.distributed as dist
import torch
import os

import socket
from contextlib import closing


def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def is_launched_by_torchrun() -> bool:
    required_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR"]
    return all(var in os.environ for var in required_vars)


def get_torchrun_env() -> tuple:
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank_id = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", world_size))
    # num_nodes = int(os.getenv("NUM_NODES", 1))
    # node_rank = int(os.getenv("NODE_RANK", 0))
    # master_addr = os.getenv("MASTER_ADDR", "localhost")
    # master_port = int(os.getenv("MASTER_PORT", 29500))
    return world_size, rank_id, local_rank, local_world_size


def get_rank_id():
    """get rank in total world size"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def get_gpu_count():
    if not torch.cuda.is_available():
        return 0
    else:
        return torch.cuda.device_count()


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def gather_forward(input, dim):
    # skip if world_size == 1
    world_size = dist.get_world_size()
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input)
    return torch.cat(output, dim=dim).contiguous()
