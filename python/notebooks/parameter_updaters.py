from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Sequence, Callable, Mapping, List

from python.notebooks.domain_objects import ParameterSet, Parameter, DeltaStep
import numpy as np


class ParameterUpdateStep:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        pass


class ParameterUpdater:
    def __init__(self, steps: List[ParameterUpdateStep]):
        self.steps = steps.copy()
        self.steps.append(DeltaParameterUpdateStep(ToNegative()))

    def adjust(self,
               param_set_maps: Sequence[Mapping[str, ParameterSet]]) -> Mapping[str, ParameterSet]:
        count = len(param_set_maps)
        if count < 1:
            raise ValueError("param_maps must contain at least one element")

        result = param_set_maps[0]

        # Scale gradients in batch.
        for param_set_map in param_set_maps:
            for param_set in param_set_map.values():
                for key in param_set.parameter_map.keys():
                    gradient = param_set.parameter_map[key].gradient
                    scaled_gradient = gradient / count
                    if param_set_map is result:
                        param_set.parameter_map[key].gradient = scaled_gradient
                    else:
                        param_set.parameter_map[key].gradient += scaled_gradient

        # Compute delta update.
        parameters = [p for ps in result.values() for p in ps.parameters]

        for step in self.steps:
            parameters = step(parameters)

        # Update the weights.
        for param_set in result.values():
            for p in param_set.parameters:
                p.value += p.delta.value

        return result


class ParameterDeltaTransform:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, parameter: Parameter) -> float:
        pass

    @abstractproperty
    def name(self) -> str: return 'Unknown delta transform'


class DeltaParameterUpdateStep(ParameterUpdateStep):

    @staticmethod
    def of(transform: ParameterDeltaTransform):
        return DeltaParameterUpdateStep(transform)

    @staticmethod
    def foreach(*transforms: Sequence[ParameterDeltaTransform]):
        return [DeltaParameterUpdateStep.of(transform) for transform in transforms]

    def __init__(self, transform: ParameterDeltaTransform):
        self.transform = transform

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        for p in parameters:
            updated_value = self.transform(p)
            p.delta.add_step(DeltaStep(self.transform.name, p.delta.value, updated_value))
            p.delta.value = updated_value
        return parameters


class ToNegative(ParameterDeltaTransform):
    @property
    def name(self):
        return 'To negative'

    def __call__(self, parameter: Parameter) -> float:
        return -parameter.delta.value


class FlatLearningRate(ParameterDeltaTransform):
    @property
    def name(self):
        return 'Flat learning rate'

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, parameter: Parameter) -> float:
        return self.learning_rate * parameter.delta.value


class ErrorRegularizedGradient(ParameterDeltaTransform):
    @property
    def name(self):
        return 'Error regularized gradient'

    def __init__(self, total_error_getter: Callable[[], float]):
        self.total_error_getter_function = total_error_getter

    def __call__(self, parameter: Parameter) -> float:
        total_error = self.total_error_getter_function()
        return parameter.gradient / total_error


class FlatGradient(ParameterDeltaTransform):
    @property
    def name(self):
        return 'Flat gradient'

    def __call__(self, parameter: Parameter) -> float:
        return parameter.gradient


class LogScaledDelta(ParameterDeltaTransform):
    @property
    def name(self):
        return 'Log scaled delta'

    def __call__(self, parameter: Parameter) -> float:
        return np.sign(parameter.delta.value) * np.log(1 + abs(parameter.delta.value))


class LargestEffectOnly(ParameterUpdateStep):
    @property
    def name(self):
        return 'Largest effect only'

    def __init__(self, keep_rate: float):
        self.keep_rate = keep_rate

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        sorted_parameters = sorted(parameters, key=lambda p: -abs(p.gradient))
        cutoff = int(np.ceil(len(sorted_parameters) * self.keep_rate))
        for parameter in sorted_parameters[cutoff:]:
            parameter.delta.value = 0
            parameter.delta.add_step(DeltaStep(self.name, parameter.delta.value, 0))
        return sorted_parameters
