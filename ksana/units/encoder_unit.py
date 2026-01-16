from abc import abstractmethod

from .runner_unit import KsanaRunnerUnit


class KsanaEncoderUnit(KsanaRunnerUnit):
    """
    Encoder unit for all models.
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
