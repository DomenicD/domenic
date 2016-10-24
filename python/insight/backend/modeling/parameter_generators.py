from abc import ABCMeta, abstractmethod
import numpy as np


class ParameterGenerator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, first_layer: int, second_layer: int) -> np.ndarray:
        pass


class RandomParameterGenerator(ParameterGenerator):
    def __call__(self, first_layer: int, second_layer: int) -> np.ndarray:
        return np.random.rand(first_layer, second_layer) - .5


class SequenceParameterGenerator(ParameterGenerator):
    def __init__(self, start: float = -1, stop: float = 1):
        super().__init__()
        self.start = start
        self.stop = stop

    def __call__(self, first_layer: int, second_layer: int) -> np.ndarray:
        count = first_layer * second_layer
        sequence = np.round(np.linspace(self.start, self.stop, count), 3)
        return np.reshape(sequence, [first_layer, second_layer])


class ConstantParameterGenerator(ParameterGenerator):
    def __init__(self, constant: float = 1):
        super().__init__()
        self.constant = constant

    def __call__(self, first_layer: int, second_layer: int) -> np.ndarray:
        return np.ones([first_layer, second_layer], float) * self.constant
