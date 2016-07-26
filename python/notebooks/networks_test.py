import unittest

import numpy as np

from python.notebooks.activation_functions import RectifiedLinearUnitActivation
from python.notebooks.cost_functions import QuadraticCost
from python.notebooks.layers import QuadraticLayer
from python.notebooks.networks import SimpleFeedForward, FeedForward
from python.notebooks.parameter_generators import ConstantParameterGenerator
from python.notebooks.utils import pretty_print


class FeedForwardTest(unittest.TestCase):
    def test_forward_pass(self):
        layers = [
            QuadraticLayer(1, 3),
            QuadraticLayer(3, 1)
        ]
        feed_forward = FeedForward(layers)
        feed_forward.forward_pass([2])
        pretty_print("output", layers[0].outputs)
        pretty_print("output", layers[1].outputs)
        # TODO(domenic): There is a bug here.
        np.testing.assert_array_equal(layers[-1].outputs, {784})



class SimpleFeedForwardTest(unittest.TestCase):
    def test_forward_pass(self):
        net = SimpleFeedForward([2, 3, 1],
                                RectifiedLinearUnitActivation(),
                                QuadraticCost(),
                                ConstantParameterGenerator())
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
        net = SimpleFeedForward([2, 3, 1],
                                RectifiedLinearUnitActivation(),
                                QuadraticCost(),
                                ConstantParameterGenerator())
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
        net = SimpleFeedForward([2, 3, 1],
                                RectifiedLinearUnitActivation(),
                                QuadraticCost(),
                                ConstantParameterGenerator())
        net.forward_pass([.5, 1.5])
        net.backward_pass([4])
        net.adjust_weights(.01)

        np.testing.assert_array_equal(net.weights[0], [[0.97, 0.97, 0.97], [0.91, 0.91, 0.91]])
        np.testing.assert_array_equal(net.weights[1], [[.96], [.96], [.96]])

    def test_adjust_biases(self):
        net = SimpleFeedForward([2, 3, 1],
                                RectifiedLinearUnitActivation(),
                                QuadraticCost(),
                                ConstantParameterGenerator())
        net.forward_pass([.5, 1.5])
        net.backward_pass([4])
        net.adjust_biases(.01)

        np.testing.assert_array_equal(net.biases[0], [-0.02, -0.02, -0.02])
        np.testing.assert_array_equal(net.biases[1], [-0.02])

    def test_adjust_parameters(self):
        net = SimpleFeedForward([2, 3, 1],
                                RectifiedLinearUnitActivation(),
                                QuadraticCost(),
                                ConstantParameterGenerator())
        net.forward_pass([.5, 1.5])
        net.backward_pass([4])
        net.adjust_parameters(.01)

        np.testing.assert_array_equal(net.weights[0], [[0.97, 0.97, 0.97], [0.91, 0.91, 0.91]])
        np.testing.assert_array_equal(net.weights[1], [[.96], [.96], [.96]])

        np.testing.assert_array_equal(net.biases[0], [-0.02, -0.02, -0.02])
        np.testing.assert_array_equal(net.biases[1], [-0.02])


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
