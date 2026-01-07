from ksana.config import KsanaTorchCompileConfig


def torch_compile_config(
    backend,
    fullgraph,
    mode,
    dynamic,
    compile_transformer_blocks_only,
    dynamo_cache_size_limit=128,
    dynamo_recompile_limit=128,
    force_parameter_static_shapes=False,
    allow_unmerged_lora_compile=False,
):
    return KsanaTorchCompileConfig(
        backend=backend,
        fullgraph=fullgraph,
        mode=mode,
        dynamic=dynamic,
        compile_transformer_blocks_only=compile_transformer_blocks_only,
        dynamo_cache_size_limit=dynamo_cache_size_limit,
        dynamo_recompile_limit=dynamo_recompile_limit,
        force_parameter_static_shapes=force_parameter_static_shapes,
        allow_unmerged_lora_compile=allow_unmerged_lora_compile,
    )
