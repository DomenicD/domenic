from abc import ABCMeta, abstractmethod
import numpy as np

from modeling.utils import same_type, same_size


class Activation:
    __metaclass__ = ABCMeta

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


class RectifiedLinearUnitActivation(Activation):
    def __init__(self, leak: float = 0):
        super().__init__()
        self.leak = leak

    def _apply(self, value: float):
        return max(self.leak * value, value)

    def _apply_derivative(self, value: float):
        return 1 if value > 0 else self.leak


class IdentityActivation(Activation):
    # TODO(domenic): Override the apply and apply_derivative functions to improve performance.
    def _apply(self, value: float): return value

    def _apply_derivative(self, value: float): return 1
