from .base_cache import kDitCache
from .cache_config import MagCacheConfig


class MagCache(kDitCache):
    def __init__(self, model_kind: str, model_type: str, model_size: str, config: MagCacheConfig):
        super().__init__(f"{model_kind}_{model_type}_{model_size}_{config.name}")
        self.config = config

    def can_use_cache(self, current_x_input, current_timestep: int) -> bool:
        return

    def try_get_prev_cache(self, current_x_input, current_timestep: int):
        return

    def update_states(self, current_timestep: int, current_x_input, current_x_output):
        return
