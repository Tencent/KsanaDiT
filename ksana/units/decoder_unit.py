from abc import abstractmethod

from .runner_unit import KsanaRunnerUnit


class KsanaDecoderUnit(KsanaRunnerUnit):
    """
    Decoder unit for all models.
    """

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
