import os

import ray

from ..config import KsanaDistributedConfig
from .executor import KsanaExecutor


@ray.remote(num_gpus=1)
class RayKsanaExecutor(KsanaExecutor):

    def init_torch_dist_group(self, rank_id, dist_config: KsanaDistributedConfig):
        """
        初始化 PyTorch DDP。
        """
        master_addr = "127.0.0.1"  # ray.util.get_node_ip_address()
        master_port = dist_config.port
        os.environ["RANK"] = str(rank_id)
        os.environ["WORLD_SIZE"] = str(dist_config.num_gpus)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        super().init_torch_dist_group(rank_id, dist_config)
        return True
