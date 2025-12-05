from .base_cache import KsanaCache
from .cache_config import MagCacheConfig


class MagCache(KsanaCache):
    def __init__(self, model_name: str, model_type: str, model_size: str, config: MagCacheConfig):
        super().__init__(f"{model_name}_{model_type}_{model_size}_{config.name}")
        self.config = config

    def can_use_cache(self, current_x_input, current_timestep: int) -> bool:
        return

    def try_get_prev_cache(self, current_x_input, current_timestep: int):
        return

    def clone_input_x(self, current_timestep: int, x):
        return x.clone()

    def update_states(self, current_timestep: int, current_x_input, current_x_output):
        return

    def offload_to_cpu(self):
        return
