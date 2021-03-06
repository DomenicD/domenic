import unittest

import numpy as np

from modeling.layers import QuadraticLayer
from modeling.parameter_updaters import ParameterUpdater


class QuadraticLayerTest(unittest.TestCase):
    def test_forward_pass(self):
        layer_1 = QuadraticLayer(1, 3, level=1, parameter_updater=ParameterUpdater([]))
        layer_2 = QuadraticLayer(3, 1, level=2, parameter_updater=ParameterUpdater([]))

        layer_1.forward_pass([2])
        np.testing.assert_array_equal(layer_1.inputs, [2])
        np.testing.assert_array_equal(layer_1.pre_activation, [9, 9, 9])
        np.testing.assert_array_equal(layer_1.outputs, [9, 9, 9])

        layer_2.forward_pass(layer_1.outputs)
        np.testing.assert_array_equal(layer_2.inputs, [9, 9, 9])
        np.testing.assert_array_equal(layer_2.pre_activation, [784])
        np.testing.assert_array_equal(layer_2.outputs, [784])
