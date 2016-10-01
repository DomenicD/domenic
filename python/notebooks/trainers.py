import uuid
import itertools
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
        self.parameters = network.get_parameters()


class BatchResult:
    def __init__(self, batch_number: int, network: NeuralNetwork, steps: Sequence[BatchStepResult]):
        self.batch_number = batch_number
        self.batch_size = len(steps)
        self.total_error = sum(map(lambda step_result: step_result.error, steps))
        self.avg_error = self.total_error / self.batch_size
        self.parameters = network.adjust_parameters(
            np.transpose([step.parameters for step in steps]))
        self.inputs = [step.inputs for step in steps]
        self.expected = [step.expected for step in steps]
        self.actual = [step.outputs for step in steps]


class ValidationResult:
    def __init__(self, steps: Sequence[BatchStepResult]):
        self.inputs = [step.inputs for step in steps]
        self.expected = [step.expected for step in steps]
        self.actual = [step.outputs for step in steps]
        self.error = sum(map(lambda step_result: step_result.error, steps))


class Trainer:
    __metaclass__ = ABCMeta

    def __init__(self, network: NeuralNetwork, batch_size: int):
        self.id = str(uuid.uuid4())
        self.network = network
        self.batch_size = batch_size
        self.step_tally = 0
        self.batch_tally = 0

    def single_train(self) -> BatchResult:
        return self.batch_train(1)

    def batch_train(self, batch_size: int = -1) -> BatchResult:
        if batch_size < 1:
            batch_size = self.batch_size
        step_results = [self._batch_step() for _ in range(batch_size)]
        self.step_tally += batch_size
        self.batch_tally += 1
        batch_result = BatchResult(self.batch_tally, self.network, step_results)
        self.network.reset()
        return batch_result

    def validate(self) -> ValidationResult:
        steps = [self._batch_step(np.array(inputs)) for inputs in self._get_validation_set()]
        return ValidationResult(steps)

    @abstractmethod
    def _batch_step(self, inputs=None) -> BatchStepResult: pass

    @abstractmethod
    def _get_validation_set(self) -> Sequence[Sequence[float]]: pass


class ClosedFormFunctionTrainer(Trainer):
    def __init__(self, network: NeuralNetwork,
                 function: Callable[[Sequence[float]], Sequence[float]],
                 domain: Tuple[float, float], batch_size: int):
        super().__init__(network, batch_size)
        self.function = function
        self.domain = domain

    def _batch_step(self, inputs=None) -> BatchStepResult:
        if inputs is None:
            inputs = np.random.uniform(self.domain[0], self.domain[1], self.network.input_count)
        self.network.forward_pass(inputs)
        expected = self.function(inputs)
        self.network.backward_pass(expected)
        return BatchStepResult(inputs, expected, self.network)

    def _get_validation_set(self) -> Sequence[Sequence[float]]:
        return itertools.product(range(self.domain[0], self.domain[1], 1),
                                 repeat=self.network.input_count)
