import unittest

import numpy as np

from python.notebooks.domain_objects import ParameterSet, parameter_set_map
from python.notebooks.parameter_updaters import ParameterUpdater, \
    DeltaParameterUpdateStep, FlatParameterDeltaTransform, ErrorRegularizedParameterDeltaTransform


class FlatParameterDeltaTransformTest(unittest.TestCase):
    def test_call(self):
        updater = ParameterUpdater(
            [DeltaParameterUpdateStep(FlatParameterDeltaTransform(learning_rate=.01))])

        param_map = parameter_set_map([
            ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]], [[5, 10, -5], [0, 100, -50]]),
            ParameterSet("param_2", [1, 1, 1], [0, 100, -50])
        ])
        updater(param_map)
        np.testing.assert_allclose(param_map["param_1"].values, [[0.95, 0.9, 1.05], [1., 0., 1.5]])
        np.testing.assert_allclose(param_map["param_2"].values, [1., 0., 1.5])


class ErrorRegularizedParameterDeltaTransformTest(unittest.TestCase):
    def test_call(self):
        # TODO: Test out the new Parameter Update code.
        total_error = 5
        updater = ParameterUpdater(
            [DeltaParameterUpdateStep(ErrorRegularizedParameterDeltaTransform(
                total_error_getter_function=lambda a: total_error))])

        param_map = parameter_set_map([
            ParameterSet("param_1", [[1, 1, 1], [1, 1, 1]], [[5, 10, -5], [0, 100, -50]]),
            ParameterSet("param_2", [1, 1, 1], [0, 100, -50])
        ])
        updater(param_map)
        np.testing.assert_allclose(param_map["param_1"].values, [[0.95, 0.9, 1.05], [1., 0., 1.5]])
        np.testing.assert_allclose(param_map["param_2"].values, [1., 0., 1.5])
