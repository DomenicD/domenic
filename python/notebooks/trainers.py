from typing import Callable, Sequence, Tuple

import numpy as np

from python.notebooks.networks import NeuralNetwork


class TrainingPlan:
    def __init__(self,
                 acceptable_error: float = 0,
                 batch_size: int = 100,
                 parameter_recording_interval: int = 10):
        self.acceptable_error = acceptable_error
        self.batch_size = batch_size
        self.parameter_recording_interval = parameter_recording_interval


class BatchStepResult:
    def __init__(self, inputs: Sequence[float], network: NeuralNetwork):
        self.inputs = inputs
        self.outputs = network.outputs
        self.error = network.total_error
        self.deltas = network.calculate_deltas()


class BatchResult:
    def __init__(self, batch_number: int, network: NeuralNetwork, steps: Sequence[BatchStepResult]):
        self.batch_number = batch_number
        self.batch_size = len(steps)
        self.total_error = sum(map(lambda step_result: step_result.error, steps))
        self.avg_error = self.total_error / self.batch_size
        deltas = np.transpose([step.deltas for step in steps])
        self.parameters = network.adjust_parameters(deltas)


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
        self.batch_step_tally = 0
        self.batch_results = []

    @property
    def batch_tally(self) -> int:
        return len(self.batch_results)

    def single_train(self) -> BatchResult:
        return self.batch_train(1)

    def batch_train(self, batch_size: int = -1) -> BatchResult:
        if batch_size < 1:
            batch_size = self.training_plan.batch_size
        step_results = [self.batch_step() for _ in range(batch_size)]
        batch_result = BatchResult(self.batch_tally + 1, self.network, step_results)
        self.batch_results.append(batch_result)
        return batch_result

    def batch_step(self) -> BatchStepResult:
        inputs = np.random.uniform(self.domain[0], self.domain[1], self.network.input_count)
        self.network.forward_pass(inputs)
        self.network.backward_pass(self.function(inputs))
        self.batch_step_tally += 1
        return BatchStepResult(inputs, self.network)
