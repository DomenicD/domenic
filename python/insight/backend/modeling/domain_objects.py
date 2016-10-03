from typing import Callable, Iterable, Mapping

import numpy as np


class DeltaStep:
    def __init__(self, name: str, input_value: float, output_value: float):
        self.name = name
        self.input_value = input_value
        self.output_value = output_value


class Delta:
    def __init__(self):
        self.value = 0
        self.steps = []

    def add_step(self, step: DeltaStep):
        self.steps.append(step)


class Parameter:
    def __init__(self, set_name: str, index: int, value: float, gradient: float, delta: Delta):
        self.name = set_name + "_" + str(index)
        self.value = value
        self.gradient = gradient
        self.delta = delta


class ParameterSet:
    def __init__(self, name: str, values: np.ndarray, gradients: np.ndarray):
        values = np.asarray(values)
        gradients = np.asarray(gradients)
        if values.shape != gradients.shape:
            raise ValueError("Parameter values and gradients must be the same shape")
        self.shape = values.shape
        self.name = name

        self.parameters = [Parameter(self.name, idx, value, gradient, Delta())
                           for idx, value, gradient in
                           zip(range(values.size), values.flatten(), gradients.flatten())]

        self.parameter_map = {p.name: p for p in self.parameters}

    @property
    def values(self):
        return self.__param_value(lambda p: p.value)

    @property
    def gradients(self):
        return self.__param_value(lambda p: p.gradient)

    @property
    def deltas(self):
        return self.__param_value(lambda p: p.delta)

    def __param_value(self, select: Callable[[Parameter], float]) -> np.ndarray:
        return np.reshape(list(map(select, self.parameters)), self.shape)


def parameter_set_map(parameter_sets: Iterable[ParameterSet]) -> Mapping[str, ParameterSet]:
    return {ps.name: ps for ps in parameter_sets}
