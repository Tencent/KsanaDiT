from abc import ABC, abstractmethod
from enum import Enum

from ..utils.factory import Factory


class KsanaUnit(ABC):
    """
    Base class for all Ksana units.
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class KsanaUnitType(Enum):
    LOADER = "loader"
    ENCODER = "encoder"
    DECODER = "decoder"
    GENERATOR = "generator"


class KsanaUnitFactory(Factory):
    @classmethod
    def create(cls, unit_type, model_key, *args, **kwargs):
        obj = super().create(unit_type, model_key, *args, **kwargs)
        setattr(obj, "model_key", model_key)
        return obj
