# import comfy
from comfy.comfy_types.node_typing import IO
from ksana.utils import print_recursive


class KsanaDebugNode:
    """测试输入类型的调试节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"source": (IO.ANY, {})},
            "optional": {"name": ("STRING", {"default": ""})},
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show_func"
    CATEGORY = "ksana/debug"

    def show_func(self, source, name="", node_id=None):
        name = name if name else "source"
        print(f"node_id {node_id} print {name}=")
        try:
            print_recursive(source)
        except Exception as e:
            print(f"处理提示词时出错: {str(e)}")
        return ()
