from abc import ABC, abstractmethod


class KsanaModel(ABC):

    @abstractmethod
    def to(self, *args, **kwargs):
        pass
