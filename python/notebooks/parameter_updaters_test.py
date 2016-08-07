import unittest

import numpy as np

from python.notebooks.domain_objects import ParameterSet, parameter_set_map, Parameter
from python.notebooks.parameter_updaters import ParameterUpdater, \
    DeltaParameterUpdateStep, FlatParameterDeltaTransform, ErrorRegularizedParameterDeltaTransform, \
    LogarithmicScaleParameterDeltaTransform, LargestEffectFilteringParameterUpdateStep


class ParameterUpdaterTest(unittest.TestCase):
    def test_e2e_with_flat_parameter_transform(self):
        updater = ParameterUpdater(
            [DeltaParameterUpdateStep(FlatParameterDeltaTransform(learning_rate=.01))])

        param_map = parameter_set_map([
            ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]], [[5, 10, -5], [0, 100, -50]]),
            ParameterSet("param_2", [1, 1, 1], [0, 100, -50])
        ])
        result_map = updater(param_map)
        self.assertEqual(result_map, param_map)
        np.testing.assert_allclose(param_map["param_1"].values, [[0.95, 0.9, 1.05], [1., 0., 1.5]])
        np.testing.assert_allclose(param_map["param_2"].values, [1., 0., 1.5])


class DeltaTransformTest(unittest.TestCase):
    def test_error_regularized(self):
        total_error = 5
        transformer = ErrorRegularizedParameterDeltaTransform(
            total_error_getter_function=lambda: total_error)

        parameter = Parameter("set_a", 1, 4, -10, 0)
        result = transformer(parameter)
        self.assertEqual(result, -.5)

        total_error = 50
        result = transformer(parameter)
        self.assertEqual(result, -.2)

    def test_logarithmic_scale(self):
        transformer = LogarithmicScaleParameterDeltaTransform()
        parameter = Parameter("set_a", 1, 4, -10, 0)

        result = transformer(parameter)
        self.assertEqual(result, 0)

        parameter.delta = -10
        result = transformer(parameter)
        self.assertAlmostEqual(result, -2.3979, places=4)

        parameter.delta = 10
        result = transformer(parameter)
        self.assertAlmostEqual(result, 2.3979, places=4)


class LargestEffectFilteringParameterUpdateStepTest(unittest.TestCase):
    def test_filter_none(self):
        filter_step = LargestEffectFilteringParameterUpdateStep(keep_rate=1)
        parameter_set = ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]],
                                     [[5, 10, -5], [0, 100, -50]])
        parameters = filter_step(parameter_set.parameters)
        for p in parameter_set.parameters:
            self.assertIn(p, parameters)

    def test_filter_all(self):
        filter_step = LargestEffectFilteringParameterUpdateStep(keep_rate=0)
        parameter_set = ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]],
                                     [[5, 10, -5], [0, 100, -50]])
        parameters = filter_step(parameter_set.parameters)
        self.assertEqual(len(parameters), 0)

    def test_keep_top_third(self):
        filter_step = LargestEffectFilteringParameterUpdateStep(keep_rate=.33)
        parameter_set = ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]],
                                     [[5, 10, -5], [0, 100, -50]])
        parameters = filter_step(parameter_set.parameters)
        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0].gradient, 100)
        self.assertEqual(parameters[1].gradient, -50)
