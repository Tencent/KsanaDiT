try:
	from .comfyui.vgenerator.nodes import (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
	__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
except ImportError:
	# ComfyUI environment not available, skip comfyui imports
	NODE_CLASS_MAPPINGS = {}
	NODE_DISPLAY_NAME_MAPPINGS = {}
	__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

