import unittest
from typing import Sequence

import numpy as np

from modeling.assembled_models import quadratic_feed_forward_network
from modeling.trainers import ClosedFormFunctionTrainer, TrainingPlan


def simple_polynomial(x_list: Sequence[float]) -> Sequence[float]:
    x = x_list[0]
    return [(x - 10) * (x + 2) * (x + 7)]


class QuadraticFeedForwardNetworkTest(unittest.TestCase):
    def test_train_model(self):
        network = quadratic_feed_forward_network([1, 3, 1], param_update_rate=.75)
        trainer = ClosedFormFunctionTrainer(network, simple_polynomial, (-15, 15),
                                            training_plan=TrainingPlan(batch_size=100))
        last_error = trainer.batch_train().avg_error
        for i in range(10):
            current_error = trainer.batch_train().avg_error
            print("c:", current_error, "d:", last_error - current_error)
            last_error = current_error
        print(trainer.batch_results[-1])
