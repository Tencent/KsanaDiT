from ..models import get_default_model_config

from .wan_x2v import KsanaWanX2VPipeline


def create_ksana_pipeline(model_name, model_type, model_size):
    default_model_config = get_default_model_config(model_name, model_type, model_size)
    if model_name != "wan2.2":
        raise RuntimeError(f"model_name {model_name} is not supported yet")
    if model_type != "t2v":
        raise RuntimeError(f"model_type {model_type} is not supported yet")
    # TODO: support more task type here
    # TODO: support other model types
    if model_name == "wan2.2":
        return KsanaWanX2VPipeline("2.2", model_type, default_model_config)
    elif model_name == "wan2.1":
        return KsanaWanX2VPipeline("2.1", model_type, default_model_config)
    else:
        raise ValueError(f"model_type {model_type} is not supported yet")


__all__ = [
    "create_ksana_pipeline",
]
