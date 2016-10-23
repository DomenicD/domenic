from modeling.function.base import Func2


class QuadraticCost(Func2):
    def _apply(self, actual: float, expected: float) -> float:
        return .5 * (actual - expected) ** 2

    def _apply_derivative(self, actual: float, expected: float) -> float:
        return actual - expected
