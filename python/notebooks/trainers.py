from typing import Callable, Sequence, Tuple

from python.notebooks.networks import NeuralNetwork
import numpy as np


class TrainingPlan:
    def __init__(self,
                 acceptable_error: float = 0,
                 batch_size: int = 100,
                 parameter_recording_interval: int = 10):
        self.acceptable_error = acceptable_error
        self.batch_size = batch_size
        self.parameter_recording_interval = parameter_recording_interval


# TODO: Create training classes that make training, validation, and testing easy. And that make
# automate recording the transformation of network parameters. Keep in mind that eventually you
# will want to make an abstract Trainer with implementations that can be used to train on images,
# time series, and other data types.
class ClosedFormFunctionTrainer:
    def __init__(self, network: NeuralNetwork,
                 function: Callable[[Sequence[float]], Sequence[float]],
                 domain: Tuple[float, float],
                 training_plan: TrainingPlan = TrainingPlan()):
        self.network = network
        self.training_plan = training_plan
        self.function = function
        self.domain = domain
        self.total_training_steps = 0
        self.parameter_record = []

    def single_training_step(self):
        inputs = np.random.uniform(self.domain[0], self.domain[1], self.network.input_count)
        self.network.forward_pass(inputs)
        self.network.backward_pass(self.function(inputs))
        # TODO: Parameter updating is now two steps.
        # 1) Calculate the deltas
        # 2) Batch adjust the parameters
        # I need to update the Trainer design to leverage these two steps
        # This means that single_training_step is actually a special case of batch_train.
        # Currently, it is setup as the opposite of this.
        parameters = self.network.adjust_parameters()
        if self.total_training_steps % self.training_plan.parameter_recording_interval == 0:
            self.parameter_record.append(parameters)
        self.total_training_steps += 1

    def batch_train(self):
        for i in range(self.training_plan.batch_size):
            self.single_training_step()
