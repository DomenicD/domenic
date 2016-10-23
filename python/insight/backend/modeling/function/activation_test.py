import unittest

from modeling.function.activation import RectifiedLinearUnitActivation


class RectifiedLinearUnit(unittest.TestCase):
    def test_apply(self):
        relu = RectifiedLinearUnitActivation()
        self.assertEqual(relu.apply(5.), 5.)
        self.assertEqual(relu.apply(0.), 0.)
        self.assertEqual(relu.apply(-.5), 0.)

    def test_apply_derivative(self):
        relu = RectifiedLinearUnitActivation()
        self.assertEqual(relu.apply_derivative(5.), 1.)
        self.assertEqual(relu.apply_derivative(0.), 0.)
        self.assertEqual(relu.apply_derivative(-.5), 0.)

    def test_apply_leak(self):
        relu = RectifiedLinearUnitActivation(leak=.01)
        self.assertEqual(relu.apply(5.), 5.)
        self.assertEqual(relu.apply(0.), 0.)
        self.assertEqual(relu.apply(-.5), -.5 * .01)

    def test_apply_derivative_leak(self):
        relu = RectifiedLinearUnitActivation(leak=.01)
        self.assertEqual(relu.apply_derivative(5.), 1.)
        self.assertEqual(relu.apply_derivative(0.), .01)
        self.assertEqual(relu.apply_derivative(-.5), .01)
