from abc import ABCMeta, abstractmethod
from typing import Sequence

from python.notebooks.domain_objects import Parameter


class ParameterUpdater:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        pass


class FlatParameterUpdater(ParameterUpdater):
    def __init__(self, learning_rate: float = .01):
        self.learning_rate = learning_rate

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        for parameter in parameters:
            parameter.values -= self.learning_rate * parameter.gradients
        return parameters
