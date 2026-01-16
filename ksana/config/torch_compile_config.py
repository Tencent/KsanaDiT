from dataclasses import dataclass


@dataclass
class KsanaTorchCompileConfig:
    backend: str = "inductor"
    fullgraph: bool = False
    mode: str = "default"
    dynamic: bool = False
    compile_transformer_blocks_only: bool = True
    dynamo_recompile_limit: int = 128
    dynamo_cache_size_limit: int = 128
    force_parameter_static_shapes: bool = False
    allow_unmerged_lora_compile: bool = False

    def __post_init__(self):
        assert self.backend in [
            "inductor",
            "cudagraphs",
        ], f"backend must be in ['inductor', 'cudagraphs'], but got {self.backend}"
        assert self.mode in [
            "default",
            "max-autotune",
            "max-autotune-no-cudagraphs",
            "reduce-overhead",
        ], (
            f"mode must be in ['default', 'max-autotune', 'max-autotune-no-cudagraphs', 'reduce-overhead'],"
            f" but got {self.mode}"
        )
        assert self.dynamo_recompile_limit > 0 and self.dynamo_recompile_limit <= 1024, (
            f"dynamo_recompile_limit must be greater than 0 and less than or equal to 1024, "
            f"but got {self.dynamo_recompile_limit}"
        )
        assert self.dynamo_cache_size_limit > 0 and self.dynamo_cache_size_limit <= 1024, (
            f"dynamo_cache_size_limit must be greater than 0 and less than or equal to 1024, "
            f"but got {self.dynamo_cache_size_limit}"
        )
