from typing import Sequence, Callable

from python.notebooks.layers import QuadraticLayer
from python.notebooks.networks import FeedForward
from python.notebooks.parameter_generators import RandomParameterGenerator
from python.notebooks.parameter_updaters import ParameterUpdater, \
    LargestEffectFilteringParameterUpdateStep, DeltaParameterUpdateStep, \
    ErrorRegularizedParameterDeltaTransform, LogarithmicScaleParameterDeltaTransform, \
    FlatScaleParameterDeltaTransform


def selective_error_log_updater(total_error_getter: Callable[[], float],
                                keep_rate: float) -> ParameterUpdater:
    return ParameterUpdater([
        # Only update the parameters that contributed most to the error.
        LargestEffectFilteringParameterUpdateStep(keep_rate=keep_rate),
        # Make the gradients relative to the total cost.
        DeltaParameterUpdateStep(
            ErrorRegularizedParameterDeltaTransform(total_error_getter=total_error_getter)),
        # Log scale the delta by the
        DeltaParameterUpdateStep(LogarithmicScaleParameterDeltaTransform())
    ])


def quadratic_feed_forward_network(nodes: Sequence[int], param_update_rate: float) -> FeedForward:
    layers = []
    network = FeedForward(layers)

    def total_error_getter(): return network.total_error

    for i in range(len(nodes) - 1):
        parameter_updater = selective_error_log_updater(
            total_error_getter=total_error_getter, keep_rate=param_update_rate)
        layers.append(
            QuadraticLayer(nodes[i], nodes[i + 1], level=i, parameter_updater=parameter_updater,
                           parameter_generator=RandomParameterGenerator()))

    return network
