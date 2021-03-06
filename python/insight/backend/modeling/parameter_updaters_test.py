import unittest

import numpy as np

from modeling.domain_objects import ParameterSet, parameter_set_map, Parameter, Delta
from modeling.parameter_updaters import ParameterUpdater, \
    DeltaParameterUpdateStep, ScaledGradientParameterDeltaTransform, \
    ErrorRegularizedGradient, \
    LogScaledDelta, LargestGradientsOnly


class ParameterUpdaterTest(unittest.TestCase):
    def test_e2e_with_flat_parameter_transform(self):
        updater = ParameterUpdater(
            [DeltaParameterUpdateStep(ScaledGradientParameterDeltaTransform(learning_rate=.01))])

        param_map = parameter_set_map([
            ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]], [[5, 10, -5], [0, 100, -50]]),
            ParameterSet("param_2", [1, 1, 1], [0, 100, -50])
        ])
        result_map = updater.adjust([param_map])
        np.testing.assert_allclose(result_map["param_1"].values, [[0.95, 0.9, 1.05], [1., 0., 1.5]])
        np.testing.assert_allclose(result_map["param_2"].values, [1., 0., 1.5])


class DeltaTransformTest(unittest.TestCase):
    def test_error_regularized(self):
        total_error = 5
        transformer = ErrorRegularizedGradient(
            total_error_getter=lambda: total_error)

        parameter = Parameter("set_a", 1, 4, -10, Delta())
        result = transformer(parameter)
        self.assertEqual(result, -.5)

        total_error = 50
        result = transformer(parameter)
        self.assertEqual(result, -.2)

    def test_logarithmic_scale(self):
        transformer = LogScaledDelta()
        parameter = Parameter("set_a", 1, 4, -10, Delta())

        result = transformer(parameter)
        self.assertEqual(result, 0)

        parameter.delta.value = -10
        result = transformer(parameter)
        self.assertAlmostEqual(result, -2.3979, places=4)

        parameter.delta.value = 10
        result = transformer(parameter)
        self.assertAlmostEqual(result, 2.3979, places=4)


class LargestEffectFilteringParameterUpdateStepTest(unittest.TestCase):
    def test_filter_none(self):
        filter_step = LargestGradientsOnly(keep_rate=1)
        parameter_set = ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]],
                                     [[5, 10, -5], [0, 100, -50]])
        parameters = filter_step(parameter_set.parameters)
        for p in parameter_set.parameters:
            self.assertIn(p, parameters)

    def test_filter_all(self):
        filter_step = LargestGradientsOnly(keep_rate=0)
        parameter_set = ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]],
                                     [[5, 10, -5], [0, 100, -50]])
        parameters = filter_step(parameter_set.parameters)
        self.assertEqual(len(parameters), 0)

    def test_keep_top_third(self):
        filter_step = LargestGradientsOnly(keep_rate=.33)
        parameter_set = ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]],
                                     [[5, 10, -5], [0, 100, -50]])
        parameters = filter_step(parameter_set.parameters)
        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0].gradient, 100)
        self.assertEqual(parameters[1].gradient, -50)
