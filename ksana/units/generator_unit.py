from abc import abstractmethod

from .runner_unit import KsanaRunnerUnit


class KsanaGeneratorUnit(KsanaRunnerUnit):
    """
    Generator unit.
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Forward pass of the unit.
        """
        pass
