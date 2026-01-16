from abc import ABC, abstractmethod

from .model_key import KsanaModelKey


class KsanaModel(ABC):
    def __init__(self, model_key: KsanaModelKey, default_settings):
        self._model_key = model_key
        self._default_settings = default_settings

    @abstractmethod
    def to(self, *args, **kwargs):
        pass

    @property
    def model_key(self) -> KsanaModelKey:
        return self._model_key

    @property
    def default_settings(self):
        return self._default_settings
