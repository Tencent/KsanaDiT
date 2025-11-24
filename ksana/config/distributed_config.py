from dataclasses import dataclass, field


@dataclass(frozen=True)
class KsanaDistributedConfig:
    # num_gpus: int = field(default=1, metadata={"help": "number of total gpus, only used as input args"})

    world_size: int = field(default=1, metadata={"help": "total number of gpus"})
    num_nodes: int = field(default=1, metadata={"help": "total number of nodes"})
    node_rank: int = field(default=0, metadata={"help": "current node rank id"})
    rank_id: int = field(default=0, metadata={"help": "current gpu rank id: 0~world_size-1"})

    local_world_size: int = field(default=1, metadata={"help": "local world size"})
    local_rank: int = field(default=0, metadata={"help": "current gpu local rank id in current local node"})

    master_addr: str | None = field(default=None, metadata={"help": "master address"})
    master_port: int | None = field(default=None, metadata={"help": "master port"})

    use_sp: bool = field(default=False, metadata={"help": "use sequence parallel"})
    dit_fsdp: bool = field(default=False, metadata={"help": "use fully sharded data parallel"})
    ulysses_size: int = field(default=1, metadata={"help": "ulysses size"})

    def __post_init__(self):
        # assert self.num_gpus == self.world_size, "num_gpus must be equal to world_size"
        assert self.num_nodes is None or self.num_nodes == 1, "only support num_nodes == 1 yet"
        assert self.node_rank is None or self.node_rank == 0, "only support node_rank == 0 yet"

        assert self.rank_id < self.world_size, "rank_id must be smaller than world_size"
        assert self.local_world_size <= self.world_size, "local_world_size must be smaller than world_size"
        assert (
            self.num_nodes * self.local_world_size <= self.world_size
        ), "local_world_size * num_nodes must be smaller than world_size"
        assert (
            self.local_rank < self.local_world_size
        ), f"local_rank({self.local_rank}) must be smaller than local_world_size({self.local_world_size})."
        if self.ulysses_size > 1:
            assert (
                self.ulysses_size == self.world_size
            ), f"The number of ulysses_size({self.ulysses_size}) should be equal to the world size({self.world_size})."

        if self.world_size <= 1:
            assert not (
                self.dit_fsdp or self.use_sp
            ), "dit_fsdp and use_sp are not supported in non-distributed environments."
            assert not (self.ulysses_size > 1), "sequence parallel are not supported in non-distributed environments."
