class KsanaTorchCompileArgs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (
                    ["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"],
                    {"default": "default"},
                ),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "compile_transformer_blocks_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Compile only the transformer blocks, usually enough and can make compilation faster and less error prone",
                    },
                ),
            },
            "optional": {
                "dynamo_recompile_limit": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "torch._dynamo.config.recompile_limit",
                    },
                ),
                "force_parameter_static_shapes": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "torch._dynamo.config.force_parameter_static_shapes"},
                ),
                "allow_unmerged_lora_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow LoRA application to be compiled with torch.compile to avoid graph breaks, causes issues with some LoRAs, mostly dynamic ones",
                    },
                ),
            },
        }

    RETURN_TYPES = ("KSANACOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "set_args"
    CATEGORY = "ksana"
    DESCRIPTION = (
        "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted."
    )

    def set_args(
        self,
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

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "compile_transformer_blocks_only": compile_transformer_blocks_only,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "dynamo_recompile_limit": dynamo_recompile_limit,
            "force_parameter_static_shapes": force_parameter_static_shapes,
            "allow_unmerged_lora_compile": allow_unmerged_lora_compile,
        }

        return (compile_args,)
