import torch
from . import log

try:
    # Avoid Dynamo compiling cache helpers that use numpy/Python control flow
    from torch._dynamo import disable as disable_dynamo
except Exception:

    def disable_dynamo(fn=None):
        return fn if fn is not None else (lambda f: f)


def apply_torch_compile(model, torch_compile_config=None):
    if torch_compile_config is None:
        return model
    log.info(f"apply torch_compile_config: {torch_compile_config}")
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.cache_size_limit = torch_compile_config.dynamo_cache_size_limit
        torch._dynamo.config.force_parameter_static_shapes = torch_compile_config.force_parameter_static_shapes
        try:
            torch._dynamo.config.recompile_limit = torch_compile_config.dynamo_recompile_limit
        except Exception as e:
            log.warning(f"Could not set recompile_limit: {e}")

    if torch_compile_config.compile_transformer_blocks_only:
        log.info("Compiling only transformer blocks")

        blocks = getattr(model, "blocks", None)
        if blocks is None:
            blocks = getattr(model, "transformer_blocks", None)

        if blocks is None:
            log.warning("No transformer blocks found to compile (checked .blocks and .transformer_blocks)")
            return model

        compiled_cnt = 0
        for i, block in enumerate(blocks):
            try:
                blocks[i] = torch.compile(
                    block,
                    backend=torch_compile_config.backend,
                    mode=torch_compile_config.mode,
                    fullgraph=torch_compile_config.fullgraph,
                    dynamic=torch_compile_config.dynamic,
                )
                compiled_cnt += 1
            except Exception as e:
                log.warning(f"torch.compile block[{i}] failed: {e}")
        log.info(f"Applied torch.compile to {compiled_cnt}/{len(blocks)} transformer blocks.")
    else:
        log.info("Compiling entire model")
        model = torch.compile(
            model,
            fullgraph=torch_compile_config.fullgraph,
            dynamic=torch_compile_config.dynamic,
            backend=torch_compile_config.backend,
            mode=torch_compile_config.mode,
        )
    return model
