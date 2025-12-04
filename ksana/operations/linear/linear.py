import torch


class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None


CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear

    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:

    class CublassLinear(CublasLinear):
        def reset_parameters(self):
            return None

        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)
