# WEB_DIRECTORY = "./web"
try:
    from .debug import vDitDebugNode
    from .cache import (
        vDitDCacheNode,
        vDitTeaCacheNode,
        vDitEasyCacheNode,
        vDitMagCacheNode,
    )
    from .model_loader import vDitModelLoaderNode
    from .generator import vDitGeneratorNode
except Exception as e:
    print(f"import错误: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "vDitDCacheNode": vDitDCacheNode,
    "vDitTeaCacheNode": vDitTeaCacheNode,
    "vDitEasyCacheNode": vDitEasyCacheNode,
    "vDitMagCacheNode": vDitMagCacheNode,
    "vDitDebugNode": vDitDebugNode,
    "vDitModelLoaderNode": vDitModelLoaderNode,
    "vDitGeneratorNode": vDitGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "vDitDCacheNode": "vDit DCache Node",
    "vDitTeaCacheNode": "vDit TeaCache Node",
    "vDitEasyCacheNode": "vDit EasyCache Node",
    "vDitMagCacheNode": "vDit MagCache Node",
    "vDitDebugNode": "vDit Debug Node",
    "vDitModelLoaderNode": "vDit Model Loader",
    "vDitGeneratorNode": "vDit Generator",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
