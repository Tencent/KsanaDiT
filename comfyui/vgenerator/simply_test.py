

class SimplyTestNode:
  """简单测试节点，用于在ComfyUI中测试简单的功能"""
  
  @classmethod
  def INPUT_TYPES(cls):
      return {
        "required": {
            "prompt_pairs": ("STRING", {"multiline": True, "default": '"key1":"value1",\n"key2":"value2",\n"key3":"value3"'}),
            "quant_config": ("STRING", {"default": ""}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "selected_key": (["key1", "key2", "key3"],),
        },
        "optional": {
            "cache_args": ("STRING", {"default": ""}),
            "selected_value": ("STRING", {"default": ""}),
        },
        # 这样可以为 FUNCTION 提供 node_id 参数
        "hidden": { "node_id": "UNIQUE_ID" }
      }

  @classmethod
  def VALIDATE_INPUTS(cls, selected_key):
      return True
      
  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("selected_value",)
  FUNCTION = "process"
  CATEGORY = "vdit"
  
  def process(self, prompt_pairs: str, selected_key: str, node_id) -> tuple:
      """处理选择的提示词"""
      try:
          # 解析提示词对并更新可用的keys
          self.parse_prompt_pairs(prompt_pairs)
          
          # 确保选中的key存在，否则使用第一个可用的key
          if selected_key not in self.prompt_dict:
              selected_key = self.keys_list[0] if self.keys_list else "key1"
              
          return (self.prompt_dict.get(selected_key, ""),)
      except Exception as e:
          print(f"处理提示词时出错: {str(e)}")
          return ("",)