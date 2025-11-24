import torch.distributed as dist
import torch
from ..config import KsanaDistributedConfig
import os


def get_ksana_distributed_config_from_torchrun_env(**kwargs) -> KsanaDistributedConfig:

    world_size = int(os.getenv("WORLD_SIZE", 1))
    num_nodes = int(os.getenv("NUM_NODES", 1))
    node_rank = int(os.getenv("NODE_RANK", 0))
    rank_id = int(os.getenv("RANK", 0))

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", world_size))

    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = int(os.getenv("MASTER_PORT", 29500))

    dist_config = kwargs.get("dist_config", KsanaDistributedConfig())

    return KsanaDistributedConfig(
        world_size=world_size,
        num_nodes=num_nodes,
        node_rank=node_rank,
        rank_id=rank_id,
        local_world_size=local_world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        use_sp=dist_config.use_sp,
        dit_fsdp=dist_config.dit_fsdp,
        ulysses_size=dist_config.ulysses_size,
    )


def init_distributed_group():
    """r initialize sequence parallel group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def get_rank():
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
