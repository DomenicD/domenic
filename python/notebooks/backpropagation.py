import uuid
from abc import ABCMeta, abstractmethod
from typing import List, Union, Sequence, Callable, Any
import numpy as np
import collections

ListableFloat = Union[float, Sequence[float]]


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

    def apply(self, value: ListableFloat) -> ListableFloat:
        if same_type(float, value):
            return self._apply(value)
        elif same_size(value):
            return np.array([self._apply(v) for v in value])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, value: float) -> float:
        pass

    def apply_derivative(self, value: ListableFloat) -> \
            ListableFloat:
        if same_type(float, value):
            return self._apply_derivative(value)
        elif same_size(value):
            return np.array([self._apply_derivative(v) for v in value])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply_derivative(self, value: float) -> float:
        pass


class Cost:
    __metaclass__ = ABCMeta

    def apply(self, actual: ListableFloat, expected: ListableFloat) -> ListableFloat:
        if same_type(float, actual, expected):
            return self._apply(actual, expected)
        elif same_size(actual, expected):
            return np.array([self._apply(a, e) for a, e in zip(actual, expected)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply(self, actual: float, expected: float) -> float:
        pass

    def apply_derivative(self, actual: ListableFloat, expected: ListableFloat) -> ListableFloat:
        if same_type(float, actual, expected):
            return self._apply_derivative(actual, expected)
        elif same_size(actual, expected):
            return np.array([self._apply_derivative(a, e) for a, e in zip(actual, expected)])
        else:
            raise ValueError("Value must be a float or list of floats.")

    @abstractmethod
    def _apply_derivative(self, actual: float, expected: float) -> float:
        pass


class WeightGenerator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, first_layer: int, second_layer: int) -> Sequence[Sequence[float]]:
        pass


class RectifiedLinearUnit(Activation):
    def __init__(self, leak: float = 0):
        super().__init__()
        self.leak = leak

    def _apply(self, value: float):
        return max(self.leak * value, value)

    def _apply_derivative(self, value: float):
        return 1 if value > 0 else self.leak


class QuadraticCost(Cost):
    def _apply(self, actual: float, expected: float) -> float:
        return .5 * (actual - expected) ** 2

    def _apply_derivative(self, actual: float, expected: float) -> float:
        return actual - expected


class RandomWeightGenerator(WeightGenerator):
    def __call__(self, first_layer: int, second_layer: int) -> Sequence[Sequence[float]]:
        return np.random.rand(first_layer, second_layer) * 2 - .5


class SequenceWeightGenerator(WeightGenerator):
    def __init__(self, start: float = -1, stop: float = 1):
        super().__init__()
        self.start = start
        self.stop = stop

    def __call__(self, first_layer: int, second_layer: int) -> Sequence[Sequence[float]]:
        count = first_layer * second_layer
        sequence = np.round(np.linspace(self.start, self.stop, count), 3)
        return np.reshape(sequence, [first_layer, second_layer])


class ConstantWeightGenerator(WeightGenerator):
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


class FeedForward:
    def __init__(self, layers: Sequence[int], activation: Activation, cost: Cost,
                 weight_generator: WeightGenerator = RandomWeightGenerator()):
        if isinstance(layers, str) or not isinstance(layers, (list, tuple)):
            raise ValueError("layers must be a list or tuple")
        if not issubclass(activation.__class__, Activation):
            raise ValueError("activation must be a subclass of Activation")
        if not issubclass(cost.__class__, Cost):
            raise ValueError("cost must be a subclass of Cost")
        if not issubclass(weight_generator.__class__, WeightGenerator):
            raise ValueError("weight_generator must be a subclass of WeightGenerator")

        self.id = uuid.uuid4()
        self.layers = layers
        self.activation = activation
        self.cost = cost
        self.weight_generator = weight_generator
        self.weights = self._generate_weights()
        self.biases = [np.zeros(n) for n in self.layers[1:]]
        self.inputs = [np.zeros(n) for n in self.layers]
        self.outputs = [np.zeros(n) for n in self.layers]
        self.node_errors = [np.zeros(n) for n in self.layers]
        self.weight_gradients = [np.zeros(n) for n in self.layers[:-1]]
        self.bias_gradients = [np.zeros(n) for n in self.layers[1:]]
        self.total_error = 0

    def _generate_weights(self) -> Sequence[Sequence[float]]:
        return np.array([self.weight_generator(self.layers[i], self.layers[i + 1]) for i in
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
            weighted_errors = np.matmul(self.node_errors[i + 1], np.transpose(self.weights[i]))
            rate_of_change = self.activation.apply_derivative(self.inputs[i])
            self.node_errors[i] = weighted_errors * rate_of_change
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
        web_safe_object = {}
        web_safe_object.id = self.id
        web_safe_object.biases = tolist(self.biases)
        web_safe_object.weights = tolist(self.weights)
        web_safe_object.inputs = tolist(self.inputs)
        web_safe_object.outputs = tolist(self.outputs)
        web_safe_object.node_errors = tolist(self.node_errors)
        web_safe_object.weight_gradients = tolist(self.weight_gradients)
        web_safe_object.bias_gradients = tolist(self.bias_gradients)
        web_safe_object.total_error = self.total_error

        return web_safe_object

