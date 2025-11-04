try:
	from .comfyui import (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
	__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
	print(f"loaded vDit nodes: {NODE_CLASS_MAPPINGS.keys()}")
except ImportError:
	# ComfyUI environment not available, skip comfyui imports
	NODE_CLASS_MAPPINGS = {}
	NODE_DISPLAY_NAME_MAPPINGS = {}
	__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

