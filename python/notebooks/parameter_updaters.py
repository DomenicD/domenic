from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque
from typing import Sequence, Callable, Mapping, List

from python.notebooks.domain_objects import ParameterSet, Parameter, DeltaStep
import numpy as np
import re


def to_sentence(class_name: str):
    """
    Transforms PascalCase into a sentence.
    eg. FlatLearningRate -> Flat learning rate
    """
    words = re.findall('[A-Z][^A-Z]*', class_name)
    return words[0] + ' ' + ' '.join([w.lower() for w in words[1:]])


class ParameterUpdateStep:
    __metaclass__ = ABCMeta

    @property
    def name(self) -> str:
        return to_sentence(self.__class__.__name__)

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

    @property
    def name(self) -> str:
        return to_sentence(self.__class__.__name__)

    @abstractmethod
    def __call__(self, parameter: Parameter) -> float:
        pass


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
    def __call__(self, parameter: Parameter) -> float:
        return -parameter.delta.value


class FlatLearningRate(ParameterDeltaTransform):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def __call__(self, parameter: Parameter) -> float:
        return self.learning_rate * parameter.delta.value


class ErrorRegularizedGradient(ParameterDeltaTransform):
    def __init__(self, total_error_getter: Callable[[], float]):
        self.total_error_getter_function = total_error_getter

    def __call__(self, parameter: Parameter) -> float:
        total_error = self.total_error_getter_function()
        return parameter.gradient / total_error


class Momentum(ParameterDeltaTransform):
    def __init__(self, history_weights: Sequence[float]):
        if any(w < 0 for w in history_weights) or abs(1 - sum(history_weights)) > .00001:
            raise ValueError('history_weights must be positive and sum to 1')
        self.history_weights = history_weights
        self.history_count = len(history_weights)
        self.history = {}

    def __call__(self, parameter: Parameter):
        values = self._update_history(parameter)
        return sum(w * d for w, d in zip(self.history_weights, values))

    def _update_history(self, parameter: Parameter) -> deque:
        if self.history.get(parameter.name) is None:
            self.history[parameter.name] = deque(maxlen=self.history_count)
        history = self.history[parameter.name]  # type: deque[float]
        if len(history) > 0 and history[0] * parameter.delta.value < 0:
            history.clear()
        history.appendleft(parameter.delta.value)
        return history


class Derivative:
    def __init__(self):
        self._observations = deque(maxlen=2)
        self._first_derivatives = deque(maxlen=2)
        self._second_derivatives = deque(maxlen=2)

    @property
    def observation_sign_change(self) -> bool:
        if len(self._observations) < 2:
            return False
        return np.sign(self._observations[0]) != np.sign(self._observations[1])

    @property
    def first_positive(self) -> bool:
        return np.sign(self.first) >= 0

    @property
    def second_positive(self) -> bool:
        return np.sign(self.second) >= 0

    @property
    def first_negative(self) -> bool:
        return not self.first_positive

    @property
    def second_negative(self) -> bool:
        return not self.second_positive

    @property
    def observation(self) -> float:
        if len(self._observations) > 0:
            return self._observations[0]
        else:
            return 0

    @property
    def first(self) -> float:
        if len(self._first_derivatives) > 0:
            return self._first_derivatives[0]
        else:
            return 0

    @property
    def second(self) -> float:
        if len(self._second_derivatives) > 0:
            return self._second_derivatives[0]
        else:
            return 0

    def add_observation(self, value: float):
        self._observations.appendleft(value)
        if len(self._observations) > 1:
            self._first_derivatives.appendleft(self._observations[1] - self._observations[0])
        if len(self._first_derivatives) > 1:
            self._second_derivatives.appendleft(
                self._first_derivatives[1] - self._first_derivatives[0])


class AdaptiveGradientDerivative(ParameterUpdateStep):
    def __init__(self,
                 total_error_getter: Callable[[], float],
                 initial_start_step: float = 0.001,
                 initial_grow_rate: float = 0.2,
                 max_error_history: float = 4):
        self.max_error_history = max_error_history
        self.error_history = deque(maxlen=self.max_error_history)
        self.total_error_getter = total_error_getter
        self.initial_start_step = initial_start_step
        self.initial_grow_rate = initial_grow_rate
        self._derivatives = {}
        self._steps = {}
        self._start_steps = {}
        self._grow_rates = {}

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        self._account_for_error(parameters)

        for p in parameters:
            updated_value = self.get_delta(p)
            p.delta.add_step(DeltaStep(self.name, p.delta.value, updated_value))
            p.delta.value = updated_value
        return parameters

    def _account_for_error(self, parameters: Sequence[Parameter]):
        self.error_history.appendleft(self.total_error_getter())

        history_length = len(self.error_history)
        if history_length < self.max_error_history:
            return

        all_bad = all(self.error_history[i] > self.error_history[i + 1]
                      for i in range(history_length - 1))

        error_delta = (self.error_history[0] - self.error_history[1]) / self.error_history[1]
        over_error_threshold = error_delta > .5

        if all_bad or over_error_threshold:
            self._reset(parameters)

    def get_delta(self, parameter: Parameter) -> float:
        derivative = self._get_derivative(parameter)
        derivative.add_observation(parameter.gradient)
        if derivative.observation_sign_change:
            self._decrease_rate(parameter)
            self._initialize_step(parameter)
            return 0
        elif derivative.first_negative:
            self._steps[parameter.name] *= (1 + self.initial_grow_rate)
            if derivative.second_positive:
                self._increase_rate(parameter)

        sign = -1 if np.sign(parameter.gradient) < 0 else 1
        return sign * self._steps[parameter.name]

    def _get_derivative(self, parameter: Parameter) -> Derivative:
        if self._derivatives.get(parameter.name) is None:
            self._derivatives[parameter.name] = Derivative()
            self._initialize_step(parameter)

        return self._derivatives[parameter.name]

    def _initialize_step(self, parameter: Parameter):
        if self._start_steps.get(parameter.name) is None:
            self._start_steps[parameter.name] = self.initial_start_step
            self._grow_rates[parameter.name] = self.initial_grow_rate
        self._steps[parameter.name] = self._start_steps[parameter.name]

    def _decrease_rate(self, parameter: Parameter):
        self._start_steps[parameter.name] *= .9
        self._grow_rates[parameter.name] *= .9

    def _increase_rate(self, parameter: Parameter):
        self._start_steps[parameter.name] *= 1.01
        self._grow_rates[parameter.name] *= 1.01

    def _reset(self, parameters: Sequence[Parameter]):
        for p in parameters:
            self._initialize_step(p)


class FlatGradient(ParameterDeltaTransform):
    def __call__(self, parameter: Parameter) -> float:
        return parameter.gradient


class LogScaledDelta(ParameterDeltaTransform):
    def __call__(self, parameter: Parameter) -> float:
        return np.sign(parameter.delta.value) * np.log(1 + abs(parameter.delta.value))


class LargestDeltasFilter(ParameterUpdateStep):
    def __init__(self, keep_rate: float = 0.5):
        self.keep_rate = keep_rate

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        sorted_parameters = sorted(parameters, key=lambda p: -abs(p.delta.value))
        return sorted_parameters[:int(np.ceil(len(sorted_parameters) * self.keep_rate))]


class LargestGradientsFilter(ParameterUpdateStep):
    def __init__(self, keep_rate: float = 0.5):
        self.keep_rate = keep_rate

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        sorted_parameters = sorted(parameters, key=lambda p: -abs(p.gradient))
        return sorted_parameters[:int(np.ceil(len(sorted_parameters) * self.keep_rate))]


class LargestGradientsOnly(ParameterUpdateStep):
    def __init__(self, keep_rate: float = 0.5):
        self.keep_rate = keep_rate

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        sorted_parameters = sorted(parameters, key=lambda p: -abs(p.gradient))
        cutoff = int(np.ceil(len(sorted_parameters) * self.keep_rate))
        for parameter in sorted_parameters[cutoff:]:
            parameter.delta.value = 0
            parameter.delta.add_step(DeltaStep(self.name, parameter.delta.value, 0))
        return sorted_parameters


class LargestDeltasOnly(ParameterUpdateStep):
    def __init__(self, keep_rate: float = 0.5):
        self.keep_rate = keep_rate

    def __call__(self, parameters: Sequence[Parameter]) -> Sequence[Parameter]:
        sorted_parameters = sorted(parameters, key=lambda p: -abs(p.delta.value))
        cutoff = int(np.ceil(len(sorted_parameters) * self.keep_rate))
        for parameter in sorted_parameters[cutoff:]:
            parameter.delta.value = 0
            parameter.delta.add_step(DeltaStep(self.name, parameter.delta.value, 0))
        return sorted_parameters
