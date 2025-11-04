# import comfy
from comfy.comfy_types.node_typing import IO
from vdit.utils import print_recursive
class vDitDebugNode:
    """测试输入类型的调试节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"source": (IO.ANY, {})},
            "optional": {"name": ("STRING", {"default": ""})},
            "hidden": { "node_id": "UNIQUE_ID" }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True
        
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show_func"
    CATEGORY = "vdit/debug"

    def show_func(self, source, name="", node_id=None):
        name = name if name else "source"
        print(f"node_id {node_id} print {name}=")
        try:
            print_recursive(source)
        except Exception as e:
            print(f"处理提示词时出错: {str(e)}")
        return ()

# class SimpleTestNode:
#     """简单测试节点，用于在ComfyUI中测试简单的功能"""
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
#                 "model": ("MODEL", {"tooltip": "The model to be used."}),
#             },
#             "optional": {
#                 "cache_args": ("STRING", {"default": ""}),
#                 "inputs": ("STRING", {"multiline": True, "default": '"key1":"value1",\n"key2":"value2",\n"key3":"value3"'}),
#                 "selected_value": ("STRING", {"default": ""}),
#                 "selected_key": (["key1", "key2", "key3"],),
#                 "quant_config": ("STRING", {"default": ""}),
#                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#             },
#             "hidden": { "node_id": "UNIQUE_ID" }
#         }

#     @classmethod
#     def VALIDATE_INPUTS(cls, selected_key):
#         return True
        
#     RETURN_TYPES = ("LATENT",)
#     RETURN_NAMES = ("samples",)
#     FUNCTION = "process"
#     CATEGORY = "vdit"
  
#     def process(self, model, samples:dict, inputs: str, cache_args:str,selected_value:str,
#                  selected_key: str,  quant_config: str, seed: int, node_id=None) -> tuple:
#         print(f"node_id {node_id} print:")
#         try:
#             print(model)
#             # print_recursive(samples)
#             print(f"node_id={node_id}, inputs={inputs},"
#                   f" cache_args={cache_args}, selected_value={selected_value}, selected_key={selected_key}, quant_config={quant_config}, seed={seed}")
            
#             return (samples,)
#         except Exception as e:
#             print(f"处理提示词时出错: {str(e)}")
#             return ("",)
        