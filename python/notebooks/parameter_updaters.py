from abc import ABCMeta, abstractmethod
from typing import Sequence

from python.notebooks.domain_objects import Parameter
import numpy as np

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
            parameter.values -= np.multiply(self.learning_rate, parameter.gradients)
        return parameters


class GeneticParameterUpdater(ParameterUpdater):
    def __init__(self, update_rate: float = .2):
        self.update_rate = update_rate
        self.previous_update_map = {}

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        for parameter in parameters:
            self.update_parameter(parameter)
        return parameters

    def update_parameter(self, parameter):
        previous = self.previous_update(parameter)
        flat_gradients = parameter.gradients.flatten()
        max_indices = reversed(np.argsort(flat_gradients))
        adjustments = np.zeros(flat_gradients.shape)

        # TODO: We only want to update the worst X % parameters.
        # Rather than computing the length, see how to only take N
        # from a reversed iterator.
        # Plan is to reshape the adjustments array and apply it to the values.
        updates_to_make = len(flat_gradients) * self.update_rate
        for i in np.arange(updates_to_make):
            pass

    def previous_update(self, parameter):
        if parameter.name not in self.previous_update_map:
            self.previous_update_map[parameter.name] = np.zeros(parameter.values.shape)
        return self.previous_update_map[parameter.name]