from abc import abstractmethod

from .base_unit import KsanaUnit


class KsanaRunnerUnit(KsanaUnit):
    """
    Base Runner class for all runner-able units.
    encoder, decoder, generator, any unit need model to run forward pass should be RunnerUnit
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
