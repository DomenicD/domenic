from abc import ABCMeta, abstractmethod
import numpy as np

from modeling.common.utils import same_type, same_size


class Func(metaclass=ABCMeta):

    def apply(self, value: np.ndarray) -> np.ndarray:
        if same_type(float, value):
            return self._apply(value)
        elif same_size(value):
            return np.array([self._apply(v) for v in value])
        else:
            raise ValueError("Value must be a float or list of floats.")

    def apply_derivative(self, value: np.ndarray) -> \
            np.ndarray:
        if same_type(float, value):
            return self._apply_derivative(value)
        elif same_size(value):
            return np.array([self._apply_derivative(v) for v in value])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, value: float) -> float:
        pass

    @abstractmethod
    def _apply_derivative(self, value: float) -> float:
        pass


class Func2(metaclass=ABCMeta):
    def apply(self, v_1: np.ndarray, v_2: np.ndarray) -> np.ndarray:
        if same_type(float, v_1, v_2):
            return self._apply(v_1, v_2)
        elif same_size(v_1, v_2):
            return np.array([self._apply(a, e) for a, e in zip(v_1, v_2)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    def apply_derivative(self, actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
        if same_type(float, actual, expected):
            return self._apply_derivative(actual, expected)
        elif same_size(actual, expected):
            return np.array([self._apply_derivative(a, e) for a, e in zip(actual, expected)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, v_1: float, v_2: float) -> float:
        pass

    @abstractmethod
    def _apply_derivative(self, v_1: float, v_2: float) -> float:
        pass