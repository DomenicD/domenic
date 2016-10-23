from modeling.function.base import Func


class RectifiedLinearUnitActivation(Func):
    def __init__(self, leak: float = 0):
        super().__init__()
        self.leak = leak

    def _apply(self, value: float):
        return max(self.leak * value, value)

    def _apply_derivative(self, value: float):
        return 1 if value > 0 else self.leak


class IdentityActivation(Func):
    # TODO(domenic): Override the apply and apply_derivative functions to improve performance.
    def _apply(self, value: float): return value

    def _apply_derivative(self, value: float): return 1
