from dataclasses import dataclass, field

from ..utils.distribute import get_free_port


@dataclass()
class KsanaDistributedConfig:
    num_gpus: int = field(default=1, metadata={"help": "total number of gpus"})
    port: int | None = field(default=29500, metadata={"help": "port for distributed communication"})

    use_sp: bool | None = field(default=None, metadata={"help": "use sequence parallel"})
    dit_fsdp: bool | None = field(default=None, metadata={"help": "use fully sharded data parallel"})
    ulysses_size: int | None = field(default=None, metadata={"help": "ulysses size"})

    def __post_init__(self):
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus({self.num_gpus}) must be greater than 0")
        if self.num_gpus > 1:
            self.use_sp = self.use_sp if self.use_sp is not None else True
            self.dit_fsdp = self.dit_fsdp if self.dit_fsdp is not None else False
            self.ulysses_size = self.ulysses_size if self.ulysses_size is not None else self.num_gpus
        else:
            self.use_sp = self.use_sp if self.use_sp is not None else False
            self.dit_fsdp = self.dit_fsdp if self.dit_fsdp is not None else False
            self.ulysses_size = self.ulysses_size if self.ulysses_size is not None else 1
            assert not (
                self.dit_fsdp or self.use_sp
            ), "dit_fsdp and use_sp are not supported in non-distributed environments."
            assert not (self.ulysses_size > 1), "sequence parallel are not supported in non-distributed environments."

        if self.ulysses_size > 1:
            assert (
                self.ulysses_size == self.num_gpus
            ), f"The number of ulysses_size({self.ulysses_size}) should be equal to the num_gpus({self.num_gpus})."

        if self.port is None or self.port <= 0:
            self.port = get_free_port()
