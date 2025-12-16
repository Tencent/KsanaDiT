from abc import ABC, abstractmethod

from .model_key import KsanaModelKey


class KsanaModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def to(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_model_key(self) -> KsanaModelKey:
        pass
