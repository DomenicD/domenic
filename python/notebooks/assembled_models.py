from abc import abstractproperty, abstractmethod, ABCMeta
from typing import Sequence, Callable

from python.notebooks.layers import QuadraticLayer
from python.notebooks.networks import FeedForward
from python.notebooks.parameter_generators import RandomParameterGenerator
from python.notebooks.parameter_updaters import ParameterUpdater, \
    LargestEffectOnly, DeltaParameterUpdateStep, \
    ErrorRegularizedGradient, LogScaledDelta, FlatGradient, FlatLearningRate

updaters = {}


class FeedForwardUpdater:
    __metaclass__ = ABCMeta

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def create(self, network: FeedForward) -> ParameterUpdater:
        pass


class SimpleUpdater(FeedForwardUpdater):
    def create(self, network: FeedForward) -> ParameterUpdater:
        keep_rate = .5
        learning_rate = .001

        steps = DeltaParameterUpdateStep.foreach(
            FlatGradient(),
            LogScaledDelta(),
            FlatLearningRate(learning_rate))

        # Only update the parameters that contributed most to the error.
        steps.append(LargestEffectOnly(keep_rate=keep_rate))
        return ParameterUpdater(steps)


simple_updater = SimpleUpdater()
updaters[simple_updater.name] = simple_updater


class ErrorRegularizedUpdater(FeedForwardUpdater):
    def create(self, network: FeedForward) -> ParameterUpdater:
        keep_rate = .5

        def total_error_getter(): return network.total_error / network.forward_pass_tally

        steps = DeltaParameterUpdateStep.foreach(
            ErrorRegularizedGradient(total_error_getter=total_error_getter),
            LogScaledDelta())

        # Only update the parameters that contributed most to the error.
        steps.append(LargestEffectOnly(keep_rate=keep_rate))
        return ParameterUpdater(steps)


error_regularized_updater = ErrorRegularizedUpdater()
updaters[error_regularized_updater.name] = error_regularized_updater


def quadratic_feed_forward_network(nodes: Sequence[int], updater_key: str) -> FeedForward:
    updater = updaters[updater_key]
    layers = []
    network = FeedForward(layers)

    for i in range(len(nodes) - 1):
        layers.append(
            QuadraticLayer(nodes[i], nodes[i + 1],
                           level=i,
                           parameter_updater=updater.create(network),
                           parameter_generator=RandomParameterGenerator()))
    return network
