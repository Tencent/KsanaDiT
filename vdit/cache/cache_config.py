from abc import ABC, abstractmethod

class vDitCacheConfig(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

class DCacheConfig(vDitCacheConfig):
    def __init__(self, fast_degree, slow_degree, fast_force_calc_every_n_steps, slow_force_calc_every_n_steps, skip_first_n_iter=2, name:str=None):
        self.name = name if name else "DCache"
        super().__init__(self.name)
        self.fast_degree = fast_degree
        self.slow_degree = slow_degree
        self.fast_force_calc_every_n_steps = fast_force_calc_every_n_steps
        self.slow_force_calc_every_n_steps = slow_force_calc_every_n_steps
        self.skip_first_n_iter = skip_first_n_iter
        
    def __str__(self):
        return f"DCacheConfig: fast_degree: {self.fast_degree}, slow_degree: {self.slow_degree}, " \
               f"fast_force_calc_every_n_steps: {self.fast_force_calc_every_n_steps}, " \
               f"slow_force_calc_every_n_steps: {self.slow_force_calc_every_n_steps}, "\
               f"skip_first_n_iter: {self.skip_first_n_iter}"

class TeaCacheConfig(vDitCacheConfig):
    def __init__(self, rel_l1_thresh, cache_device, start_step, end_step, use_coeffecients, mode, name:str=None):
        self.name = name if name else "TeaCache"
        super().__init__(self.name)
        self.start_step = start_step
        self.end_step = end_step
        self.rel_l1_thresh = rel_l1_thresh
        self.cache_device = cache_device
        self.use_coeffecients = use_coeffecients
        self.mode = mode
        
    def __str__(self):
        return f"TeaCacheConfig: rel_l1_thresh: {self.rel_l1_thresh}, cache_device: {self.cache_device}, " \
               f"start_step: {self.start_step}, end_step: {self.end_step}, " \
               f"use_coeffecients: {self.use_coeffecients}, mode: {self.mode}"
        
class EasyCacheConfig(vDitCacheConfig):
    def __init__(self,  reuse_thresh, start_percent, end_percent, verbose, name:str=None):
        self.name = name if name else "EasyCache"
        super().__init__(self.name)
        self.reuse_thresh = reuse_thresh
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.verbose = verbose

    def __str__(self):
        return f"EasyCacheConfig: reuse_thresh: {self.reuse_thresh}, start_percent: {self.start_percent}, " \
               f"end_percent: {self.end_percent}, verbose: {self.verbose}"


class MagCacheConfig(vDitCacheConfig):
    def __init__(self, threshold, K, cache_device, start_step, end_step, name:str=None):
        self.name = name if name else "MagCache"
        super().__init__(self.name)
        self.threshold = threshold
        self.K = K
        self.cache_device = cache_device
        self.start_step = start_step
        self.end_step = end_step
        
    def __str__(self):
        return f"MagCacheConfig: threshold: {self.threshold}, K: {self.K}, cache_device: {self.cache_device}, " \
               f"start_step: {self.start_step}, end_step: {self.end_step}"
