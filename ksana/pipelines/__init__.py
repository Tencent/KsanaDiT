from ..config import KsanaPipelineConfig

from .wan_x2v import KsanaWanX2VPipeline


def create_ksana_pipeline(pipeline_config: KsanaPipelineConfig):
    model_name = pipeline_config.model_name
    task_type = pipeline_config.task_type
    if model_name != "wan2.2":
        raise RuntimeError(f"model_name {model_name} is not supported yet")
    if task_type != "t2v":
        raise RuntimeError(f"task_type {task_type} is not supported yet")
    # TODO: support more task type here
    # TODO: support other model types
    if model_name == "wan2.2":
        return KsanaWanX2VPipeline(pipeline_config)
    elif model_name == "wan2.1":
        return KsanaWanX2VPipeline(pipeline_config)
    else:
        raise ValueError(f"task_type {task_type} is not supported yet")


__all__ = [
    "create_ksana_pipeline",
]
