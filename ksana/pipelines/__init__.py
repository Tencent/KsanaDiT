from ..config import KsanaPipelineConfig
from ..models.model_key import WAN2_1, WAN2_2, X2V_TYPES
from .wan_x2v import KsanaWanX2VPipeline


def create_ksana_pipeline(pipeline_config: KsanaPipelineConfig):
    if pipeline_config.model_name not in WAN2_2:
        raise RuntimeError(f"model_name {pipeline_config.model_name} is not supported yet")

    if pipeline_config.task_type not in X2V_TYPES:
        raise RuntimeError(f"task_type {pipeline_config.task_type} is not in supported list {X2V_TYPES} yet")
    if pipeline_config.model_name in WAN2_2 + WAN2_1:
        return KsanaWanX2VPipeline(pipeline_config)
    else:
        raise ValueError(f"model_name {pipeline_config.model_name} is not supported yet")


__all__ = ["create_ksana_pipeline"]
