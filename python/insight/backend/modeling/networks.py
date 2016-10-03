import uuid
from abc import ABCMeta, abstractmethod
from typing import Sequence, Mapping
import numpy as np

from modeling.activation_functions import Activation
from modeling.cost_functions import Cost, QuadraticCost
from modeling.domain_objects import ParameterSet
from modeling.layers import Layer
from modeling.parameter_generators import ParameterGenerator, RandomParameterGenerator
from modeling.utils import pretty_print


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
    def __init__(self, layers: Sequence[Layer], cost: Cost = QuadraticCost()):
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


# TODO: Legacy code. Need to update to use new pattern.
class SimpleFeedForward:
    def __init__(self, layers: Sequence[int], activation: Activation, cost: Cost,
                 param_generator: ParameterGenerator = RandomParameterGenerator()):
        if isinstance(layers, str) or not isinstance(layers, (list, tuple)):
            raise ValueError("layers must be a list or tuple")
        if not issubclass(activation.__class__, Activation):
            raise ValueError("activation must be a subclass of Activation")
        if not issubclass(cost.__class__, Cost):
            raise ValueError("cost must be a subclass of Cost")
        if not issubclass(param_generator.__class__, ParameterGenerator):
            raise ValueError("param_generator must be a subclass of ParamGenerator")

        self.id = str(uuid.uuid4())
        self.layers = layers
        self.activation = activation
        self.cost = cost
        self.param_generator = param_generator
        self.weights = self._generate_weights()
        self.biases = [np.zeros(n) for n in self.layers[1:]]
        self.inputs = [np.zeros(n) for n in self.layers]
        self.outputs = [np.zeros(n) for n in self.layers]
        self.node_errors = [np.zeros(n) for n in self.layers]
        self.weight_gradients = [np.zeros(n) for n in self.layers[:-1]]
        self.bias_gradients = [np.zeros(n) for n in self.layers[1:]]
        self.total_error = 0

    def _generate_weights(self) -> Sequence[Sequence[Sequence[float]]]:
        return np.array([self.param_generator(self.layers[i], self.layers[i + 1]) for i in
                         range(len(self.layers) - 1)])

    def forward_pass(self, inputs: Sequence[float]):
        self.inputs[0] = inputs
        self.outputs[0] = inputs
        for i in range(1, len(self.layers)):
            self.inputs[i] = np.matmul(self.outputs[i - 1], self.weights[i - 1]) + self.biases[
                i - 1]
            self.outputs[i] = self.activation.apply(self.inputs[i])

    def backward_pass(self, expected: Sequence[float]):
        final_output = self.outputs[-1]
        self.total_error = sum(self.cost.apply(final_output, expected))
        self.node_errors[-1] = self.cost.apply_derivative(
            final_output, expected) * self.activation.apply_derivative(self.inputs[-1])

        for i in reversed(range(len(self.weights))):
            # This is not generalizable. It is a shortcut that is tied to the linear equation
            # used as input to activation functions: w*x + b.
            # For this to generalize to all types of input equations, the three lines below
            # need to be one step.
            weighted_errors = np.matmul(self.node_errors[i + 1], np.transpose(self.weights[i]))
            rate_of_change = self.activation.apply_derivative(self.inputs[i])
            # This is purely for computational efficiency. Since this part of the partial derivative
            # will be required to compute the next layers partial derivative. We can cache this
            # result and reuse it. This makes the backpropagation algorithm a dynamic program.
            self.node_errors[i] = weighted_errors * rate_of_change
            # Full gradient; partial derivative with respect to weight parameter
            self.weight_gradients[i] = self.outputs[i] * self.node_errors[i]
            self.bias_gradients[i] = self.node_errors[i + 1]

    def adjust_weights(self, learn_rate: float = .01):
        for i in range(len(self.weights)):
            self.weights[i] -= learn_rate * np.transpose(np.matrix(self.weight_gradients[i]))

    def adjust_biases(self, learn_rate: float = .01):
        for i in range(len(self.biases)):
            self.biases[i] -= learn_rate * self.bias_gradients[i]

    def adjust_parameters(self, learn_rate: float = .01):
        self.adjust_weights(learn_rate)
        self.adjust_biases(learn_rate)

    def print_state(self):
        self.print_weights()
        self.print_biases()
        self.print_inputs()
        self.print_outputs()
        self.print_weight_errors()
        self.print_weight_gradients()
        self.print_bias_gradients()

    def print_weights(self):
        pretty_print("weights", self.weights)

    def print_biases(self):
        pretty_print("biases", self.biases)

    def print_inputs(self):
        pretty_print("inputs", self.inputs)

    def print_outputs(self):
        pretty_print("outputs", self.outputs)

    def print_weight_errors(self):
        pretty_print("weight_errors", self.node_errors)

    def print_weight_gradients(self):
        pretty_print("weight_gradients", self.weight_gradients)

    def print_bias_gradients(self):
        pretty_print("bias_gradients", self.bias_gradients)
