import unittest
from typing import Sequence

import numpy as np

from python.notebooks.assembled_models import quadratic_feed_forward_network
from python.notebooks.trainers import ClosedFormFunctionTrainer


def simple_polynomial(x_list: Sequence[float]) -> Sequence[float]:
    x = x_list[0]
    return [(x - 10) * (x + 2) * (x + 7)]


class QuadraticFeedForwardNetworkTest(unittest.TestCase):
    def test_train_model(self):
        network = quadratic_feed_forward_network([1, 3, 1], param_update_rate=.5)
        trainer = ClosedFormFunctionTrainer(network, simple_polynomial, (-15, 15))
        for i in range(1000):
            trainer.batch_train()
            print(network.total_error)