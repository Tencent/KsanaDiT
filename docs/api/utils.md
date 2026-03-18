# Utilities

Shared utility functions and classes used across the kDiT framework.

## Environment Variables

::: kdit.utils.env

## Loading

::: kdit.utils.load
    options:
      members:
        - load_file_to_state_dict
        - load_sharded_safetensors
        - remove_prefix_from_state_dict

## Factory

::: kdit.utils.factory.AdvancedFactory

::: kdit.utils.factory.SimpleFactory

## Distributed

::: kdit.utils.distribute
    options:
      members:
        - get_gpu_count
        - get_rank_id
        - get_world_size
        - all_to_all
        - gather_forward

## Profiling

::: kdit.utils.profile

## LoRA

::: kdit.utils.lora

## Quantization

::: kdit.utils.quantize

## Prefetch

::: kdit.utils.prefetch
