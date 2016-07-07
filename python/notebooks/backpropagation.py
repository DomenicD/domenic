from abc import ABCMeta, abstractmethod
from typing import List, Union, Sequence
import collections


def same_size(*args: List[Sequence]):
    if len(args) == 0:
        return True

    if not isinstance(args[0], collections.Sized):
        return False

    size = len(args[0])
    for l in args:
        if not isinstance(l, collections.Sized):
            return False

        if len(l) != size:
            return False

    return True


def same_type(type_check: type, *args):
    for t in args:
        if not isinstance(t, type_check):
            return False

    return True


class Activation:
    __metaclass__ = ABCMeta

    def apply(self, value: Union[float, Sequence[float]]) -> Union[float, Sequence[float]]:
        if same_type(float, value):
            return self._apply(value)
        elif same_size(value):
            return [self._apply(v) for v in value]
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, value: float) -> float:
        pass

    def apply_derivative(self, value: Union[float, Sequence[float]]) -> Union[float, Sequence[float]]:
        if same_type(float, value):
            return self._apply_derivative(value)
        elif same_size(value):
            return [self._apply_derivative(v) for v in value]
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply_derivative(self, value: float) -> float:
        pass


class Cost:
    __metaclass__ = ABCMeta

    def apply(self, actual: Union[float, Sequence[float]], expected: Union[float, Sequence[float]]) -> Union[
        float, Sequence[float]]:
        if same_type(float, actual, expected):
            return self._apply(actual, expected)
        elif same_size(actual, expected):
            return [self._apply(v) for v in zip(actual, expected)]
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, actual: float, expected: float) -> float:
        pass

    def apply_derivative(self, actual: Union[float, Sequence[float]], expected: Union[float, Sequence[float]]) -> Union[
        float, Sequence[float]]:
        if same_type(float, actual, expected):
            return self._apply_derivative(actual, expected)
        elif same_size(actual, expected):
            return [self._apply_derivative(pair[0], pair[1]) for pair in zip(actual, expected)]
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply_derivative(self, actual: float, expected: float) -> float:
        pass


class RectifiedLinearUnit(Activation):
    def __init__(self, threshold: float = 0, leak: float = 0):
        super().__init__()
        self.threshold = threshold
        self.leak = leak

    def _apply(self, value: float):
        return max(0, value)

    def _apply_derivative(self, value: float):
        return value if value > self.threshold else self.leak


class QuadraticCost(Cost):
    def _apply(self, actual: float, expected: float) -> float:
        return .5 * (actual - expected) ** 2

    def _apply_derivative(self, actual: float, expected: float) -> float:
        return actual - expected