import uuid
from abc import ABCMeta, abstractmethod
from typing import Callable, Sequence, Tuple

import numpy as np

from python.notebooks.networks import NeuralNetwork


class BatchStepResult:
    def __init__(self, inputs: Sequence[float], expected: Sequence[float], network: NeuralNetwork):
        self.inputs = inputs
        self.expected = expected
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
        self.inputs = [step.inputs for step in steps]
        self.expected = [step.expected for step in steps]
        self.actual = [step.outputs for step in steps]


class Trainer:
    __metaclass__ = ABCMeta

    def __init__(self, network: NeuralNetwork, batch_size: int):
        self.id = str(uuid.uuid4())
        self.network = network
        self.batch_size = batch_size
        self.step_tally = 0
        self.batch_results = []

    @property
    def batch_tally(self) -> int:
        return len(self.batch_results)

    def single_train(self) -> BatchResult:
        return self.batch_train(1)

    def batch_train(self, batch_size: int = -1) -> BatchResult:
        if batch_size < 1:
            batch_size = self.batch_size
        step_results = [self._batch_step() for _ in range(batch_size)]
        self.step_tally += batch_size
        batch_result = BatchResult(self.batch_tally + 1, self.network, step_results)
        self.batch_results.append(batch_result)
        return batch_result

    @abstractmethod
    def _batch_step(self) -> BatchStepResult: pass


class ClosedFormFunctionTrainer(Trainer):
    def __init__(self, network: NeuralNetwork,
                 function: Callable[[Sequence[float]], Sequence[float]],
                 domain: Tuple[float, float], batch_size: int):
        super().__init__(network, batch_size)
        self.function = function
        self.domain = domain

    def _batch_step(self) -> BatchStepResult:
        inputs = np.random.uniform(self.domain[0], self.domain[1], self.network.input_count)
        self.network.forward_pass(inputs)
        expected = self.function(inputs)
        self.network.backward_pass(expected)
        return BatchStepResult(inputs, expected, self.network)
