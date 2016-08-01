import unittest

import numpy as np

from python.notebooks.domain_objects import Parameter
from python.notebooks.parameter_updaters import FlatParameterUpdater


class FlatParameterUpdaterTest(unittest.TestCase):
    def test_call(self):
        updater = FlatParameterUpdater(learning_rate=.01)
        params = [
            Parameter("param_1", [[1, 1, 1], [1, 1, 1]], [[5, 10, -5], [0, 100, -50]]),
            Parameter("param_2", [1, 1, 1], [0, 100, -50])
        ]
        updated_params = updater(params)
        np.testing.assert_allclose(updated_params[0].values, [[0.95, 0.9, 1.05], [1., 0., 1.5]])
        np.testing.assert_allclose(updated_params[1].values, [1., 0., 1.5])

