import uuid
from abc import ABCMeta, abstractmethod
from typing import List, Union, Sequence, Callable, Any
import numpy as np
import collections


def tolist(target: Any):
    if isinstance(target, collections.Iterable) and not isinstance(target, str):
        return [tolist(item) for item in target]
    return target


def same_size(*args: List[Sequence]):
    if len(args) == 0:
        return True

    if not isinstance(args[0], collections.Sized):
        return False

    size = len(args[0])
    for l in args:
        if not isinstance(l, collections.Sized):
            return False

        if len(l) != size:
            return False

    return True


def same_type(type_check: type, *args):
    for t in args:
        if not isinstance(t, type_check):
            return False

    return True


class Activation:
    __metaclass__ = ABCMeta

    def apply(self, value: np.ndarray) -> np.ndarray:
        if same_type(float, value):
            return self._apply(value)
        elif same_size(value):
            return np.array([self._apply(v) for v in value])
        else:
            raise ValueError("Value must be a float or list of floats.")

    def apply_derivative(self, value: np.ndarray) -> \
            np.ndarray:
        if same_type(float, value):
            return self._apply_derivative(value)
        elif same_size(value):
            return np.array([self._apply_derivative(v) for v in value])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, value: float) -> float:
        pass

    @abstractmethod
    def _apply_derivative(self, value: float) -> float:
        pass


class RectifiedLinearUnitActivation(Activation):
    def __init__(self, leak: float = 0):
        super().__init__()
        self.leak = leak

    def _apply(self, value: float):
        return max(self.leak * value, value)

    def _apply_derivative(self, value: float):
        return 1 if value > 0 else self.leak


class IdentityActivation(Activation):
    # TODO(domenic): Override the apply and apply_derivative functions to improve performance.
    def _apply(self, value: float): return value

    def _apply_derivative(self, value: float): return 1


class Cost:
    __metaclass__ = ABCMeta

    def apply(self, actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
        if same_type(float, actual, expected):
            return self._apply(actual, expected)
        elif same_size(actual, expected):
            return np.array([self._apply(a, e) for a, e in zip(actual, expected)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, actual: float, expected: float) -> float:
        pass

    def apply_derivative(self, actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
        if same_type(float, actual, expected):
            return self._apply_derivative(actual, expected)
        elif same_size(actual, expected):
            return np.array([self._apply_derivative(a, e) for a, e in zip(actual, expected)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply_derivative(self, actual: float, expected: float) -> float:
        pass


class ParamGenerator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, first_layer: int, second_layer: int) -> Sequence[Sequence[float]]:
        pass


class QuadraticCost(Cost):
    def _apply(self, actual: float, expected: float) -> float:
        return .5 * (actual - expected) ** 2

    def _apply_derivative(self, actual: float, expected: float) -> float:
        return actual - expected


class RandomParamGenerator(ParamGenerator):
    def __call__(self, first_layer: int, second_layer: int) -> Sequence[Sequence[float]]:
        return np.random.rand(first_layer, second_layer) * 2 - .5


class SequenceParamGenerator(ParamGenerator):
    def __init__(self, start: float = -1, stop: float = 1):
        super().__init__()
        self.start = start
        self.stop = stop

    def __call__(self, first_layer: int, second_layer: int) -> Sequence[Sequence[float]]:
        count = first_layer * second_layer
        sequence = np.round(np.linspace(self.start, self.stop, count), 3)
        return np.reshape(sequence, [first_layer, second_layer])


class ConstantParamGenerator(ParamGenerator):
    def __init__(self, constant: float = 1):
        super().__init__()
        self.constant = constant

    def __call__(self, first_layer: int, second_layer: int) -> Sequence[Sequence[float]]:
        return np.ones([first_layer, second_layer], float) * self.constant


def new_line():
    print("\n")


def pretty_print(title: str, seq: Union[Sequence[Any], Sequence[Sequence[Any]]]):
    print(title)
    if len(seq) > 0 and same_size(seq[0]):
        for l in seq:
            print(np.matrix(l))
            print("~~~~~~~~~~~~")
    else:
        print(np.matrix(seq))
        print("~~~~~~~~~~~~")
    new_line()


class Layer:
    __metaclass__ = ABCMeta

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation = IdentityActivation()):
        self.inputs = np.zeros(input_size)
        self.outputs = np.zeros(output_size)
        self.cached_derivative = np.ones(output_size)
        self.activation = activation

    def forward_pass(self, raw_inputs: np.ndarray) -> np.ndarray:
        self.inputs = self.transform_inputs(raw_inputs)
        self.outputs = self.activation.apply(self.inputs)
        return self.outputs

    def backward_pass(self, upstream_derivative: np.ndarray) -> np.ndarray:
        self.cached_derivative = upstream_derivative
        self.calculate_gradients()
        self.cached_derivative *= self.cached_gradient_derivative * self.activation.apply_derivative(
            self.inputs)
        return self.cached_derivative

    @abstractmethod
    def transform_inputs(self, raw_inputs: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def calculate_gradients(self): pass

    @abstractmethod
    def cached_gradient_derivative(self) -> np.ndarray: pass

    @abstractmethod
    def adjust_parameters(self): pass


class PolynomialLayer(Layer):
    def __init__(self, input_size: int, output_size: int,
                 param_generator: ParamGenerator = RandomParamGenerator()):
        super().__init__(input_size, output_size)
        # Forward pass parameters
        self.fx_weights = param_generator(input_size, output_size)
        self.fx_biases = param_generator(input_size, 1)
        self.fx = np.zeros(output_size)
        self.gx_weights = param_generator(input_size, output_size)
        self.gx_biases = param_generator(input_size, 1)
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
        self.fx_weight_gradients = self.gx * self.fx_prime * self.cached_derivative
        self.fx_bias_gradients = self.gx * self.cached_derivative
        self.gx_weight_gradients = self.fx * self.gx_prime * self.cached_derivative
        self.gx_bias_gradients = self.fx * self.cached_derivative

    def cached_gradient_derivative(self) -> np.ndarray:
        # TODO(domenic): See how this pattern extends to the linear input transform (x*W + b).
        return np.matmul(self.fx, self.gx_weights) + np.matmul(self.gx, self.fx_weights)

    def adjust_parameters(self):
        # TODO(domenic): Need to come up with a design that will let me play with the way that
        # parameters are updated. Should support standard things such as adjustable learning rates
        # and momentum. In addition, should be flexible enough so I can try out new things such as
        # only updating specific parameters on each pass.
        raise NotImplemented("adjust_parameters not implemented")

    @property
    def fx_prime(self): return self.inputs

    @property
    def gx_prime(self): return self.inputs


class FeedForward:
    def __init__(self, layers: Sequence[int], activation: Activation, cost: Cost,
                 param_generator: ParamGenerator = RandomParamGenerator()):
        if isinstance(layers, str) or not isinstance(layers, (list, tuple)):
            raise ValueError("layers must be a list or tuple")
        if not issubclass(activation.__class__, Activation):
            raise ValueError("activation must be a subclass of Activation")
        if not issubclass(cost.__class__, Cost):
            raise ValueError("cost must be a subclass of Cost")
        if not issubclass(param_generator.__class__, ParamGenerator):
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

    def to_web_safe_object(self):
        return {
            "id": self.id,
            "biases": tolist(self.biases),
            "weights": tolist(self.weights),
            "inputs": tolist(self.inputs),
            "outputs": tolist(self.outputs),
            "node_errors": tolist(self.node_errors),
            "weight_gradients": tolist(self.weight_gradients),
            "bias_gradients": tolist(self.bias_gradients),
            "total_error": self.total_error
        }
