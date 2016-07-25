import unittest

import python.notebooks.backpropagation as bp
import numpy as np


class FeedForwardTest(unittest.TestCase):
    def test_forward_pass(self):
        net = bp.FeedForward([2, 3, 1],
                             bp.RectifiedLinearUnitActivation(),
                             bp.QuadraticCost(),
                             bp.ConstantWeightGenerator())
        net.forward_pass([.5, 1.5])
        np.testing.assert_array_equal(np.array([[[1., 1., 1.], [1., 1., 1.]], [[1.], [1.], [1.]]]),
                                      [[[1., 1., 1.], [1., 1., 1.]], [[1.], [1.], [1.]]])

        np.testing.assert_array_equal(net.weights[0], [[1., 1., 1.], [1., 1., 1.]])
        np.testing.assert_array_equal(net.weights[1], [[1.], [1.], [1.]])

        np.testing.assert_array_equal(net.biases[0], [0, 0, 0])
        np.testing.assert_array_equal(net.biases[1], [0])

        np.testing.assert_array_equal(net.inputs[0], [0.5, 1.5])
        np.testing.assert_array_equal(net.inputs[1], [2., 2., 2.])
        np.testing.assert_array_equal(net.inputs[2], [6.])

        np.testing.assert_array_equal(net.outputs[0], [0.5, 1.5])
        np.testing.assert_array_equal(net.outputs[1], [2., 2., 2.])
        np.testing.assert_array_equal(net.outputs[2], [6.])

    def test_backward_pass(self):
        net = bp.FeedForward([2, 3, 1],
                             bp.RectifiedLinearUnitActivation(),
                             bp.QuadraticCost(),
                             bp.ConstantWeightGenerator())
        net.forward_pass([.5, 1.5])
        net.backward_pass([4])

        np.testing.assert_array_equal(net.node_errors[0], [6., 6.])
        np.testing.assert_array_equal(net.node_errors[1], [2., 2., 2.])
        np.testing.assert_array_equal(net.node_errors[2], [2.])

        np.testing.assert_array_equal(net.weight_gradients[0], [3., 9.])
        np.testing.assert_array_equal(net.weight_gradients[1], [4., 4., 4.])

        np.testing.assert_array_equal(net.bias_gradients[0], [2., 2., 2.])
        np.testing.assert_array_equal(net.bias_gradients[1], [2.])

    def test_adjust_weights(self):
        net = bp.FeedForward([2, 3, 1],
                             bp.RectifiedLinearUnitActivation(),
                             bp.QuadraticCost(),
                             bp.ConstantWeightGenerator())
        net.forward_pass([.5, 1.5])
        net.backward_pass([4])
        net.adjust_weights(.01)

        np.testing.assert_array_equal(net.weights[0], [[0.97, 0.97, 0.97], [0.91, 0.91, 0.91]])
        np.testing.assert_array_equal(net.weights[1], [[.96], [.96], [.96]])

    def test_adjust_biases(self):
        net = bp.FeedForward([2, 3, 1],
                             bp.RectifiedLinearUnitActivation(),
                             bp.QuadraticCost(),
                             bp.ConstantWeightGenerator())
        net.forward_pass([.5, 1.5])
        net.backward_pass([4])
        net.adjust_biases(.01)

        np.testing.assert_array_equal(net.biases[0], [-0.02, -0.02, -0.02])
        np.testing.assert_array_equal(net.biases[1], [-0.02])

    def test_adjust_parameters(self):
        net = bp.FeedForward([2, 3, 1],
                             bp.RectifiedLinearUnitActivation(),
                             bp.QuadraticCost(),
                             bp.ConstantWeightGenerator())
        net.forward_pass([.5, 1.5])
        net.backward_pass([4])
        net.adjust_parameters(.01)

        np.testing.assert_array_equal(net.weights[0], [[0.97, 0.97, 0.97], [0.91, 0.91, 0.91]])
        np.testing.assert_array_equal(net.weights[1], [[.96], [.96], [.96]])

        np.testing.assert_array_equal(net.biases[0], [-0.02, -0.02, -0.02])
        np.testing.assert_array_equal(net.biases[1], [-0.02])


class RectifiedLinearUnit(unittest.TestCase):
    def test_apply(self):
        relu = bp.RectifiedLinearUnitActivation()
        self.assertEqual(relu.apply(5.), 5.)
        self.assertEqual(relu.apply(0.), 0.)
        self.assertEqual(relu.apply(-.5), 0.)

    def test_apply_derivative(self):
        relu = bp.RectifiedLinearUnitActivation()
        self.assertEqual(relu.apply_derivative(5.), 1.)
        self.assertEqual(relu.apply_derivative(0.), 0.)
        self.assertEqual(relu.apply_derivative(-.5), 0.)

    def test_apply_leak(self):
        relu = bp.RectifiedLinearUnitActivation(leak=.01)
        self.assertEqual(relu.apply(5.), 5.)
        self.assertEqual(relu.apply(0.), 0.)
        self.assertEqual(relu.apply(-.5), -.5 * .01)

    def test_apply_derivative_leak(self):
        relu = bp.RectifiedLinearUnitActivation(leak=.01)
        self.assertEqual(relu.apply_derivative(5.), 1.)
        self.assertEqual(relu.apply_derivative(0.), .01)
        self.assertEqual(relu.apply_derivative(-.5), .01)
