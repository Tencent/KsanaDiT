# 工具函数

kDiT 框架中跨模块使用的共享工具函数和类。

## 环境变量

::: kdit.utils.env

## 加载

::: kdit.utils.load
    options:
      members:
        - load_file_to_state_dict
        - load_sharded_safetensors
        - remove_prefix_from_state_dict

## 工厂

::: kdit.utils.factory.AdvancedFactory

::: kdit.utils.factory.SimpleFactory

## 分布式

::: kdit.utils.distribute
    options:
      members:
        - get_gpu_count
        - get_rank_id
        - get_world_size
        - all_to_all
        - gather_forward

## 性能分析

::: kdit.utils.profile

## LoRA

::: kdit.utils.lora

## 量化

::: kdit.utils.quantize

## 预取

::: kdit.utils.prefetch
