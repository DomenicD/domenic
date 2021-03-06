import unittest

import numpy as np

from modeling.layers import QuadraticLayer, LinearLayer
from modeling.networks import FeedForward
from modeling.parameter_generators import SequenceParameterGenerator
from modeling.parameter_updaters import ParameterUpdater


class LinearFeedForwardTest(unittest.TestCase):
    def test_forward_pass(self):
        layers = [
            LinearLayer(2, 3, level=1, parameter_updater=ParameterUpdater([]),
                        parameter_generator=SequenceParameterGenerator()),
            LinearLayer(3, 1, level=2, parameter_updater=ParameterUpdater([]),
                        parameter_generator=SequenceParameterGenerator())
        ]
        feed_forward = FeedForward(layers)

        feed_forward.forward_pass([-3, 3])
        np.testing.assert_allclose(layers[-1].outputs, [1])

        feed_forward.forward_pass([.3, .7])
        np.testing.assert_allclose(layers[-1].outputs, [.64])

    def test_backward_pass_basic(self):
        layers = [
            LinearLayer(2, 3, level=1, parameter_updater=ParameterUpdater([]),
                        parameter_generator=SequenceParameterGenerator()),
            LinearLayer(3, 1, level=2, parameter_updater=ParameterUpdater([]),
                        parameter_generator=SequenceParameterGenerator())
        ]
        feed_forward = FeedForward(layers)
        feed_forward.forward_pass([.3, .7])
        feed_forward.backward_pass([1.5])
        self.assertAlmostEqual(feed_forward.total_error, .3698)

        np.testing.assert_allclose(layers[0].fx_weight_gradients, [[0, 0, -.258], [0, 0, -.602]])
        np.testing.assert_allclose(layers[1].fx_weight_gradients, [[0], [-.2064], [-1.4104]])

        np.testing.assert_allclose(layers[0].fx_bias_gradients, [0, 0, -.86])
        np.testing.assert_allclose(layers[1].fx_bias_gradients, [-.86])

    def test_backward_pass_large_network(self):
        layers = [
            LinearLayer(3, 5, level=1, parameter_updater=ParameterUpdater([]),
                        parameter_generator=SequenceParameterGenerator()),
            LinearLayer(5, 4, level=2, parameter_updater=ParameterUpdater([]),
                        parameter_generator=SequenceParameterGenerator()),
            LinearLayer(4, 5, level=3, parameter_updater=ParameterUpdater([]),
                        parameter_generator=SequenceParameterGenerator())
        ]

        feed_forward = FeedForward(layers)
        feed_forward.forward_pass([-5.5, 0, .2])
        feed_forward.backward_pass([-1.8, 4, 0, 3, -.5])


class QuadraticFeedForwardTest(unittest.TestCase):
    def test_forward_pass(self):
        layers = [
            QuadraticLayer(1, 3, level=1, parameter_updater=ParameterUpdater([])),
            QuadraticLayer(3, 1, level=2, parameter_updater=ParameterUpdater([]))
        ]
        feed_forward = FeedForward(layers)
        feed_forward.forward_pass([2])
        np.testing.assert_allclose(layers[-1].outputs, [784])

    def test_backward_pass_basic(self):
        layers = [
            QuadraticLayer(1, 3, level=1, parameter_updater=ParameterUpdater([])),
            QuadraticLayer(3, 1, level=2, parameter_updater=ParameterUpdater([]))
        ]
        feed_forward = FeedForward(layers)
        feed_forward.forward_pass([2])
        feed_forward.backward_pass([500])

        np.testing.assert_allclose(layers[0].fx_weight_gradients, [[95424, 95424, 95424]])
        np.testing.assert_allclose(layers[1].fx_weight_gradients, [[71568], [71568], [71568]])

        np.testing.assert_allclose(layers[0].gx_weight_gradients, [[95424, 95424, 95424]])
        np.testing.assert_allclose(layers[1].gx_weight_gradients, [[71568], [71568], [71568]])

        np.testing.assert_allclose(layers[0].fx_bias_gradients, [47712, 47712, 47712])
        np.testing.assert_allclose(layers[1].fx_bias_gradients, [7952])

        np.testing.assert_allclose(layers[0].gx_bias_gradients, [47712, 47712, 47712])
        np.testing.assert_allclose(layers[1].gx_bias_gradients, [7952])

    def test_backward_pass_multi_input_output(self):
        layers = [
            QuadraticLayer(2, 3, level=1, parameter_updater=ParameterUpdater([]),
                           parameter_generator=SequenceParameterGenerator()),
            QuadraticLayer(3, 2, level=2, parameter_updater=ParameterUpdater([]),
                           parameter_generator=SequenceParameterGenerator())
        ]
        feed_forward = FeedForward(layers)
        feed_forward.forward_pass([-3, 3])
        np.testing.assert_allclose(layers[1].outputs, [5.49434, 428.324], rtol=1e-3)

        feed_forward.backward_pass([18, -18])

        # Test W values.
        np.testing.assert_allclose(layers[0].fx_weight_gradients,
                                   [[86002.3, -40031, -254459],
                                    [-86002.3, 40031, 254459]], rtol=1e-1)
        np.testing.assert_allclose(layers[1].fx_weight_gradients,
                                   [[-198.158, 62443],
                                    [-379.9, 119713],
                                    [-620.269, 195458]], rtol=1e-1)
        # Test R values
        np.testing.assert_allclose(layers[0].gx_weight_gradients,
                                   [[86002.3, -40031, -254459],
                                    [-86002.3, 40031, 254459]], rtol=1e-1)
        np.testing.assert_allclose(layers[1].gx_weight_gradients,
                                   [[-198.158, 62443],
                                    [-379.9, 119713],
                                    [-620.269, 195458]], rtol=1e-1)

        # Test c values
        np.testing.assert_allclose(layers[0].fx_bias_gradients,
                                   [-28667.4, 13343.7, 84819.8], rtol=1e-1)
        np.testing.assert_allclose(layers[1].fx_bias_gradients,
                                   [-29.3133, 9237.13], rtol=1e-1)

        # Test d values
        np.testing.assert_allclose(layers[0].gx_bias_gradients,
                                   [-28667.4, 13343.7, 84819.8], rtol=1e-1)
        np.testing.assert_allclose(layers[1].gx_bias_gradients,
                                   [-29.3133, 9237.13], rtol=1e-1)

    def test_backward_pass_large_network(self):
        layers = [
            QuadraticLayer(3, 5, level=1, parameter_updater=ParameterUpdater([]),
                           parameter_generator=SequenceParameterGenerator()),
            QuadraticLayer(5, 4, level=2, parameter_updater=ParameterUpdater([]),
                           parameter_generator=SequenceParameterGenerator()),
            QuadraticLayer(4, 5, level=3, parameter_updater=ParameterUpdater([]),
                           parameter_generator=SequenceParameterGenerator())
        ]

        feed_forward = FeedForward(layers)
        feed_forward.forward_pass([-5.5, 0, .2])
        feed_forward.backward_pass([-1.8, 4, 0, 3, -.5])

        # Sample and test W values.
        np.testing.assert_allclose(
            layers[0].fx_weight_gradients,
            [[-1.37733703e+12, -7.21006061e+11, -1.33721172e+11, 3.84577706e+11, 8.35181063e+11],
             [0., 0., 0., - 0., - 0.],
             [5.00849829e+10, 2.62184022e+10, 4.86258807e+09, - 1.39846439e+10, - 3.03702205e+10]])

        np.testing.assert_allclose(
            layers[2].fx_weight_gradients,
            [[-1.14103797e+11, -7.41129352e+10, -4.46603547e+10, -2.43732125e+10, -1.14054976e+10],
             [-3.96861055e+10, -2.57770016e+10, -1.55331864e+10, -8.47717521e+09, -3.96691250e+09],
             [-3.53110135e+09, -2.29352828e+09, -1.38207704e+09, -7.54263096e+08, -3.52959050e+08],
             [-6.26563225e+09, -4.06966647e+09, -2.45237552e+09, -1.33837427e+09, -6.26295139e+08]])

        # Sample and test R values.
        np.testing.assert_allclose(
            layers[1].gx_weight_gradients,
            [[-4.36899117e+11, -1.05201074e+11, 1.41135784e+10, -7.93579261e+10],
             [-3.89103594e+11, -9.36923748e+10, 1.25695930e+10, -7.06763943e+10],
             [-3.44109832e+11, -8.28583130e+10, 1.11161156e+10, -6.25037717e+10],
             [-3.01879725e+11, -7.26897123e+10, 9.75191523e+09, -5.48331364e+10],
             [-2.63226095e+11, -6.33822930e+10, 8.50324934e+09, -4.78121292e+10]])

        np.testing.assert_allclose(
            layers[2].gx_weight_gradients,
            [[-1.14103797e+11, -7.41129352e+10, -4.46603547e+10, -2.43732125e+10, -1.14054976e+10],
             [-3.96861055e+10, -2.57770016e+10, -1.55331864e+10, -8.47717521e+09, -3.96691250e+09],
             [-3.53110135e+09, -2.29352828e+09, -1.38207704e+09, -7.54263096e+08, -3.52959050e+08],
             [-6.26563225e+09, -4.06966647e+09, -2.45237552e+09, -1.33837427e+09, -6.26295139e+08]])

        # Sample and test c values.
        np.testing.assert_allclose(layers[0].fx_bias_gradients,
                                   [2.50424914e+11, 1.31092011e+11, 2.43129403e+10, -6.99232193e+10,
                                    -1.51851102e+11])

        np.testing.assert_allclose(layers[2].fx_bias_gradients,
                                   [-2.15644806e+08, -1.40066062e+08, -8.44036196e+07,
                                    -4.60629426e+07, -2.15552538e+07])

        # Sample and test d values.
        np.testing.assert_allclose(layers[1].gx_bias_gradients,
                                   [-2.07754743e+10, -5.00253288e+09, 6.71130412e+08,
                                    -3.77363673e+09])

        np.testing.assert_allclose(layers[2].gx_bias_gradients,
                                   [-2.15644806e+08, -1.40066062e+08, -8.44036196e+07,
                                    -4.60629426e+07, -2.15552538e+07])
