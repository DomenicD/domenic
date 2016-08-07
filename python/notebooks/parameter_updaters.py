from abc import ABCMeta, abstractmethod
from typing import Sequence, Callable, Mapping

from python.notebooks.domain_objects import ParameterSet, Parameter
import numpy as np


class ParameterUpdateStep:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        pass


class ParameterUpdater:
    def __init__(self, steps: Sequence[ParameterUpdateStep]):
        self.steps = steps

    def __call__(self, parameter_set_map: Mapping[str, ParameterSet]) -> Mapping[str, ParameterSet]:
        parameters = [p for ps in parameter_set_map.values() for p in ps.parameters]

        for step in self.steps:
            parameters = step(parameters)

        for p in parameters:
            p.value += p.delta

        return parameter_set_map


class ParameterDeltaTransform:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, parameter: Parameter) -> float:
        pass


class DeltaParameterUpdateStep(ParameterUpdateStep):
    def __init__(self, transform: ParameterDeltaTransform):
        self.transform = transform

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        for p in parameters:
            p.delta = self.transform(p)
        return parameters


class FlatParameterDeltaTransform(ParameterDeltaTransform):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, parameter: Parameter) -> float:
        return -self.learning_rate * parameter.gradient


class ErrorRegularizedParameterDeltaTransform(ParameterDeltaTransform):
    def __init__(self, total_error_getter_function: Callable[[], float]):
        self.total_error_getter_function = total_error_getter_function

    def __call__(self, parameter: Parameter) -> float:
        total_error = self.total_error_getter_function()
        if total_error > abs(parameter.gradient):
            return parameter.gradient / total_error
        else:
            return total_error / parameter.gradient


class LogarithmicScaleParameterDeltaTransform(ParameterDeltaTransform):
    def __call__(self, parameter: Parameter) -> float:
        return np.sign(parameter.delta) * np.log(1 + abs(parameter.delta))


class LargestEffectFilteringParameterUpdateStep(ParameterUpdateStep):
    def __init__(self, keep_rate: float):
        self.keep_rate = keep_rate

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        sorted_parameters = sorted(parameters, key=lambda p: -abs(p.gradient))
        return sorted_parameters[:int(np.ceil(len(sorted_parameters) * self.keep_rate))]
