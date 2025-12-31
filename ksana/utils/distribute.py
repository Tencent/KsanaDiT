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


def get_rank_id_result(func_res: list | dict | None, rank_id: int = 0, check_no_none_res: bool = False):
    """
    default get rank 0 result
    func_res :
        - when single card:
            - return func_res[rank_id] if func_res is dict else return full result func_res but not for list
        - when multi-cards:
            - can be list[any]
            1. [dct{rank_id_1: [None, None, ...]}, dict{rank_id_0: [tensor, tensor, ...]}] return [tensor,tensor,...]
            2. [dict{rank_id_1: None}, dict{rank_id_0: tensor}] return tensor
            3. [[tensor, tensor], [tensor, tensor]] return [tensor,tensor] any list not None
            4. [tensor, tensor] return tensor any one not None
    """
    if not isinstance(func_res, list):
        if isinstance(func_res, dict):
            return func_res.get(rank_id) if rank_id in func_res else func_res
    any_rank_id = min(rank_id, len(func_res) - 1)
    return_res = func_res[any_rank_id]
    for one_rank_res in func_res:
        if isinstance(one_rank_res, dict) and rank_id in one_rank_res:
            return_res = one_rank_res.get(rank_id)
            if check_no_none_res:
                if return_res is None:
                    raise ValueError(f"rank {rank_id} res can not be None: full func_res {func_res}")
                elif isinstance(return_res, (list, tuple)):
                    has_none = [x is None for x in return_res]
                    if any(has_none):
                        raise ValueError(f"rank {rank_id} res has None: {return_res}")
            return return_res
        else:
            continue
    return return_res


def get_rank_id():
    """get rank in total world size"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_multi_process() -> bool:
    return dist.is_initialized()


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
