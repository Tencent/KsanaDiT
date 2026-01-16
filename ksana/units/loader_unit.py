from abc import abstractmethod

from .base_unit import KsanaUnit


class KsanaLoaderUnit(KsanaUnit):
    """
    Loader unit for model loader.
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
