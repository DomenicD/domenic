from abc import ABCMeta, abstractmethod
from typing import Sequence, Callable, Mapping

from python.notebooks.domain_objects import ParameterSet, Parameter
import numpy as np


class ParameterUpdateStep:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        pass

# TODO: Rework this so that the effect of each transform step is recorded.

class ParameterUpdater:
    def __init__(self, steps: Sequence[ParameterUpdateStep]):
        self.steps = steps

    def calculate(self, param_map: Mapping[str, ParameterSet]) -> Mapping[str, ParameterSet]:
        parameters = [p for ps in param_map.values() for p in ps.parameters]

        for step in self.steps:
            parameters = step(parameters)

        for p in parameters:
            p.delta = -p.delta

        return param_map

    def adjust(self,
               param_set_maps: Sequence[Mapping[str, ParameterSet]]) -> Mapping[str, ParameterSet]:
        count = len(param_set_maps)
        if count < 1:
            raise ValueError("param_maps must contain at least one element")

        result = param_set_maps[0]

        for param_set_map in param_set_maps:
            for param_set in param_set_map.values():
                for key in param_set.parameter_map.keys():
                    delta = param_set.parameter_map[key].delta
                    scaled_delta = delta / count
                    if param_set_map is result:
                        param_set.parameter_map[key].delta = scaled_delta
                    else:
                        param_set.parameter_map[key].delta += scaled_delta

        for param_set in result.values():
            for p in param_set.parameters:
                p.value += p.delta

        return result


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


class FlatScaleParameterDeltaTransform(ParameterDeltaTransform):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, parameter: Parameter) -> float:
        return self.learning_rate * parameter.delta


class ScaledGradientParameterDeltaTransform(ParameterDeltaTransform):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, parameter: Parameter) -> float:
        return self.learning_rate * parameter.gradient


class ErrorRegularizedParameterDeltaTransform(ParameterDeltaTransform):
    def __init__(self, total_error_getter: Callable[[], float]):
        self.total_error_getter_function = total_error_getter

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
