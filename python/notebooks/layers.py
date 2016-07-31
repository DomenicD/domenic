from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np

from python.notebooks.activation_functions import Activation, IdentityActivation
from python.notebooks.domain_objects import Parameter
from python.notebooks.parameter_generators import ParameterGenerator, ConstantParameterGenerator
from python.notebooks.parameter_updaters import ParameterUpdater, FlatParameterUpdater


class Layer:
    __metaclass__ = ABCMeta

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 parameter_updater: ParameterUpdater = FlatParameterUpdater(),
                 activation: Activation = IdentityActivation()):
        self.inputs = np.zeros(input_size)
        self.pre_activation = np.zeros(input_size)
        self.outputs = np.zeros(output_size)
        self.cached_derivative = np.ones(output_size)
        self.parameter_updater = parameter_updater
        self.activation = activation

    def forward_pass(self, raw_inputs: np.ndarray) -> np.ndarray:
        self.inputs = raw_inputs
        self.pre_activation = self.transform_inputs(raw_inputs)
        self.outputs = self.activation.apply(self.pre_activation)
        return self.outputs

    def backward_pass(self, upstream_derivative: np.ndarray) -> np.ndarray:
        self.cached_derivative = upstream_derivative
        self.calculate_gradients()
        self.cached_derivative = np.multiply(self.activation.apply_derivative(self.inputs),
                                             np.matmul(self.cached_derivative,
                                                       self.cached_gradient_derivative()))
        return self.cached_derivative

    def adjust_parameters(self):
        self.set_parameters(self.parameter_updater(self.get_parameters()))

    @abstractmethod
    def transform_inputs(self, raw_inputs: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def calculate_gradients(self): pass

    @abstractmethod
    def cached_gradient_derivative(self) -> np.ndarray: pass

    @abstractmethod
    def get_parameters(self) -> Sequence[Parameter]: pass

    @abstractmethod
    def set_parameters(self, parameters: Sequence[Parameter]): pass


class QuadraticLayer(Layer):
    @property
    def fx_weights_name(self):
        return "fx_weights"

    @property
    def fx_biases_name(self):
        return "fx_biases"

    @property
    def gx_weights_name(self):
        return "gx_weights"

    @property
    def gx_biases_name(self):
        return "gx_biases"

    @property
    def fx_prime(self):
        return np.transpose(np.matrix(self.inputs))

    @property
    def gx_prime(self):
        return np.transpose(np.matrix(self.inputs))

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 # TODO(domenic): The gradients of this are HUGE. Need to come up with an adaptive
                 #                ParameterUpdater that takes the current weight value into
                 #                consideration.
                 parameter_updater: ParameterUpdater = FlatParameterUpdater(),
                 parameter_generator: ParameterGenerator = ConstantParameterGenerator()):
        super().__init__(input_size, output_size, parameter_updater)
        # Forward pass parameters
        self.fx_weights = parameter_generator(input_size, output_size)
        self.fx_biases = parameter_generator(1, output_size)[0]  # Get 1-d array.
        self.fx = np.zeros(output_size)
        self.gx_weights = parameter_generator(input_size, output_size)
        self.gx_biases = parameter_generator(1, output_size)[0]  # Get 1-d array.
        self.gx = np.zeros(output_size)

        # Backward pass parameters
        self.fx_weight_gradients = np.zeros(input_size)
        self.fx_bias_gradients = np.zeros(input_size)
        self.gx_weight_gradients = np.zeros(input_size)
        self.gx_bias_gradients = np.zeros(input_size)

    def transform_inputs(self, raw_inputs: np.ndarray) -> np.ndarray:
        self.fx = np.matmul(raw_inputs, self.fx_weights) + self.fx_biases
        self.gx = np.matmul(raw_inputs, self.gx_weights) + self.gx_biases
        return self.fx * self.gx

    def calculate_gradients(self):
        fx_error = np.multiply(self.gx, self.cached_derivative)
        self.fx_bias_gradients = fx_error.A1  # A1 converts matrix to 1-d array.
        self.fx_weight_gradients = np.matmul(self.fx_prime, fx_error)

        gx_error = np.multiply(self.fx, self.cached_derivative)
        self.gx_bias_gradients = gx_error.A1  # A1 converts matrix to 1-d array.
        self.gx_weight_gradients = np.matmul(self.gx_prime, gx_error)

    def cached_gradient_derivative(self) -> np.ndarray:
        # TODO(domenic): See how this pattern extends to the linear input transform (x*W + b).
        return np.transpose(
            np.matrix([[f * r + g * w
                        for f, g, w, r in zip(self.fx, self.gx, ws, rs)]
                       for ws, rs in zip(self.fx_weights, self.gx_weights)]))

    def get_parameters(self) -> Sequence[Parameter]:
        return [
            Parameter(self.fx_weights_name, self.fx_weights, self.fx_weight_gradients),
            Parameter(self.fx_biases_name, self.fx_biases, self.fx_bias_gradients),
            Parameter(self.gx_weights_name, self.gx_weights, self.gx_weight_gradients),
            Parameter(self.gx_biases_name, self.gx_biases, self.gx_bias_gradients)
        ]

    def set_parameters(self, parameters: Sequence[Parameter]):
        for parameter in parameters:
            if parameter.name is self.fx_weights_name:
                self.fx_weights = parameter.values
            elif parameter.name is self.fx_biases_name:
                self.fx_biases = parameter.values
            elif parameter.name is self.gx_weights_name:
                self.gx_weights = parameter.values
            elif parameter.name is self.gx_biases_name:
                self.gx_biases = parameter.values
            else:
                raise ValueError(parameter.name + " is not a valid QuadraticLayer parameter")
