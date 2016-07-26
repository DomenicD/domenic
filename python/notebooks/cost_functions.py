from abc import ABCMeta, abstractmethod
import numpy as np

from python.notebooks.utils import same_type, same_size


class Cost:
    __metaclass__ = ABCMeta

    def apply(self, actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
        if same_type(float, actual, expected):
            return self._apply(actual, expected)
        elif same_size(actual, expected):
            return np.array([self._apply(a, e) for a, e in zip(actual, expected)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, actual: float, expected: float) -> float:
        pass

    def apply_derivative(self, actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
        if same_type(float, actual, expected):
            return self._apply_derivative(actual, expected)
        elif same_size(actual, expected):
            return np.array([self._apply_derivative(a, e) for a, e in zip(actual, expected)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply_derivative(self, actual: float, expected: float) -> float:
        pass


class QuadraticCost(Cost):
    def _apply(self, actual: float, expected: float) -> float:
        return .5 * (actual - expected) ** 2

    def _apply_derivative(self, actual: float, expected: float) -> float:
        return actual - expected
