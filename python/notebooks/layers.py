from abc import ABCMeta, abstractmethod
from typing import Mapping

import numpy as np

from python.notebooks.activation_functions import Activation, IdentityActivation
from python.notebooks.domain_objects import ParameterSet, parameter_set_map
from python.notebooks.parameter_generators import ParameterGenerator, ConstantParameterGenerator
from python.notebooks.parameter_updaters import ParameterUpdater


class Layer:
    __metaclass__ = ABCMeta

    @property
    def parameter_prefix(self) -> str:
        return "level_" + str(self.level) + "_"

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 level: int,
                 parameter_updater: ParameterUpdater,
                 activation: Activation = IdentityActivation()):
        self.level = level
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

    def adjust_parameters(self) -> Mapping[str, ParameterSet]:
        parameter_update = self.parameter_updater(self.get_parameters())
        self.set_parameters(parameter_update)
        return parameter_update

    @abstractmethod
    def transform_inputs(self, raw_inputs: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def calculate_gradients(self): pass

    @abstractmethod
    def cached_gradient_derivative(self) -> np.ndarray: pass

    @abstractmethod
    def get_parameters(self) -> Mapping[str, ParameterSet]: pass

    @abstractmethod
    def set_parameters(self, parameters: Mapping[str, ParameterSet]): pass


class QuadraticLayer(Layer):
    @property
    def fx_weights_name(self) -> str:
        return self.parameter_prefix + "fx_weights"

    @property
    def fx_biases_name(self) -> str:
        return self.parameter_prefix + "fx_biases"

    @property
    def gx_weights_name(self) -> str:
        return self.parameter_prefix + "gx_weights"

    @property
    def gx_biases_name(self) -> str:
        return self.parameter_prefix + "gx_biases"

    @property
    def fx_prime(self):
        return np.transpose(np.matrix(self.inputs))

    @property
    def gx_prime(self):
        return np.transpose(np.matrix(self.inputs))

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 level: int,
                 parameter_updater: ParameterUpdater,
                 parameter_generator: ParameterGenerator = ConstantParameterGenerator()):
        super().__init__(input_size, output_size, level, parameter_updater)
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

    def get_parameters(self) -> Mapping[str, ParameterSet]:
        return parameter_set_map([
            ParameterSet(self.fx_weights_name, self.fx_weights, self.fx_weight_gradients),
            ParameterSet(self.fx_biases_name, self.fx_biases, self.fx_bias_gradients),
            ParameterSet(self.gx_weights_name, self.gx_weights, self.gx_weight_gradients),
            ParameterSet(self.gx_biases_name, self.gx_biases, self.gx_bias_gradients)
        ])

    def set_parameters(self, parameters: Mapping[str, ParameterSet]):
        if self.fx_weights_name in parameters:
            self.fx_weights = parameters.get(self.fx_weights_name).values

        if self.fx_biases_name in parameters:
            self.fx_biases = parameters.get(self.fx_biases_name).values

        if self.gx_weights_name in parameters:
            self.gx_weights = parameters.get(self.gx_weights_name).values

        if self.gx_biases_name in parameters:
            self.gx_biases = parameters.get(self.gx_biases_name).values
