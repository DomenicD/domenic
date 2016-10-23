import uuid
from abc import ABCMeta, abstractmethod
from typing import Sequence, Mapping
import numpy as np

from modeling.function.base import Func2
from modeling.function.cost import QuadraticCost
from modeling.domain_objects import ParameterSet
from modeling.layers import Layer


class NeuralNetwork:
    __metaclass__ = ABCMeta

    def __init__(self, layers: Sequence[Layer]):
        self.id = str(uuid.uuid4())
        self.layers = layers
        self.total_error = 0.0
        self.forward_pass_tally = 0
        self.backward_pass_tally = 0

    @property
    def input_count(self):
        return self.layers[0].input_count

    @property
    def output_count(self):
        return self.layers[-1].output_count

    @property
    def layer_count(self):
        return len(self.layers)

    @property
    def outputs(self):
        return self.layers[-1].outputs

    def reset(self):
        self.total_error = 0.0
        self.forward_pass_tally = 0
        self.backward_pass_tally = 0

    def adjust_parameters(self, parameter_batch: Sequence[Sequence[Mapping[str, ParameterSet]]]) -> \
            Sequence[Mapping[str, ParameterSet]]:
        batch_count = len(parameter_batch)
        if self.layer_count != batch_count:
            raise ValueError(
                "Number of delta sequences ({0}) must equal number of layers ({1})".format(
                    batch_count, self.layer_count))
        return [layer.adjust_parameters(param_set_maps) for layer, param_set_maps in
                zip(self.layers, parameter_batch)]

    def get_parameters(self):
        return [layer.get_parameters() for layer in self.layers]

    def forward_pass(self, inputs: Sequence[float]) -> Sequence[float]:
        self.forward_pass_tally += 1
        return self.do_forward_pass(inputs)

    def backward_pass(self, expected: Sequence[float]) -> float:
        self.backward_pass_tally += 1
        error = self.do_backward_pass(expected)
        self.total_error += error
        return error

    @abstractmethod
    def do_forward_pass(self, inputs: Sequence[float]) -> Sequence[float]:
        pass

    @abstractmethod
    def do_backward_pass(self, expected: Sequence[float]) -> float:
        pass


class FeedForward(NeuralNetwork):
    def __init__(self, layers: Sequence[Layer], cost: Func2 = QuadraticCost()):
        super().__init__(layers)
        self.cost = cost

    def do_forward_pass(self, inputs: Sequence[float]) -> Sequence[float]:
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

    def do_backward_pass(self, expected: Sequence[float]) -> float:
        error = sum(self.cost.apply(self.outputs, expected))
        upstream_derivative = np.matrix(self.cost.apply_derivative(self.outputs, expected))
        for layer in reversed(self.layers):
            upstream_derivative = layer.backward_pass(upstream_derivative)
        return error
