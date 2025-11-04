from abc import ABC, abstractmethod

class vDitCache(ABC):
    def __init__(self, name):
        self.name = name
        self.need_compile_cache = False
        self.timestep_start = None
        self.timestep_end = None
       
    
    @abstractmethod
    def can_use_cache(self, current_x_input, current_timestep: int) -> bool:
        pass

    @abstractmethod
    def try_get_prev_cache(self, current_x_input, current_timestep: int):
        pass  
    
    @abstractmethod
    def update_states(self, current_timestep: int, current_x_input, current_x_output):
        pass  
    
    # def show_cache_rate(self):