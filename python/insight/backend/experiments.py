import random
from typing import Sequence, Mapping, Tuple, Callable

import multiprocessing
import numpy as np
import h5py
import time

from modeling.domain_objects import ParameterSet
from modeling.function.activation import RectifiedLinearUnitActivation, IdentityActivation
from modeling.layers import QuadraticLayer, Layer, LinearLayer
from modeling.networks import FeedForward
from modeling.parameter_generators import RandomParameterGenerator, SequenceParameterGenerator, \
    ConstantParameterGenerator
from modeling.parameter_updaters import ParameterUpdater, LargestGradientsOnly, \
    DeltaParameterUpdateStep, FlatGradient, LogScaledDelta, DecreasingLearningRate, Momentum, \
    FlatLearningRate, ClampedDelta
from modeling.trainers import ClosedFormFunctionTrainer, BatchResult


class Group:
    CONFIGURATION = 'configuration'
    TOTAL_ERROR = 'total_error'
    AVERAGE_ERROR = 'average_error'
    PARAMETERS = 'parameters'
    INPUTS = 'inputs'
    EXPECTED = 'expected'
    ACTUAL = 'actual'
    VALIDATION = 'validation'


class Dataset:
    EPOCHS = 'epochs'
    LAYERS = 'layers'
    BATCH_SIZE = 'batch_size'


def close_file(file: h5py.File):
    file.flush()
    file.close()


def get_dataset(file: h5py.Group, name: str, rows: int, data_shape: Tuple[int] = (),
                data_type: np.dtype = np.dtype(float)):
    return file.require_dataset(name, (rows,) + data_shape, data_type)


def write_parameters(file: h5py.File, rows: int, epoch: int,
                     parameters: Sequence[Mapping[str, ParameterSet]]):
    param_group = file.require_group(Group.PARAMETERS)
    for layer in range(len(parameters)):
        for params in parameters[layer].values():
            get_dataset(param_group.require_group(params.name), "values", rows,
                        np.shape(params.values))[epoch] = params.values

            get_dataset(param_group.require_group(params.name), "gradients", rows,
                        np.shape(params.gradients))[epoch] = params.gradients

            delta_values = np.reshape([delta.value for delta in params.deltas.flatten()],
                                      params.deltas.shape)
            get_dataset(param_group.require_group(params.name), "delta_values", rows,
                        np.shape(delta_values))[epoch] = delta_values

            # TODO: Record the delta steps.


def write_batch_result(file: h5py.File, rows: int, result: BatchResult):
    epoch = result.batch_number - 1
    get_dataset(file, Group.TOTAL_ERROR, rows)[epoch] = result.total_error
    get_dataset(file, Group.AVERAGE_ERROR, rows)[epoch] = result.avg_error
    write_parameters(file, rows, epoch, result.parameters)
    get_dataset(file, Group.INPUTS, rows, np.shape(result.inputs))[epoch] = result.inputs
    get_dataset(file, Group.EXPECTED, rows, np.shape(result.expected))[epoch] = result.expected
    get_dataset(file, Group.ACTUAL, rows, np.shape(result.actual))[epoch] = result.actual


def simple_updater(epochs: int, learning_rate: float, epoch_getter: Callable[[], int]):
    keep_rate = 1

    steps = DeltaParameterUpdateStep.foreach(
        FlatGradient(),
        # LogScaledDelta(),
        ClampedDelta(),
        # DecreasingLearningRate(learning_rate, epochs, epoch_getter, degree=2),
        # Momentum([.9, .1])
        FlatLearningRate(learning_rate)
    )

    # Only update the parameters that contributed most to the error.
    steps.append(LargestGradientsOnly(keep_rate=keep_rate))
    return ParameterUpdater(steps)


def create_network(layer: Callable[..., Layer],
                   nodes: Sequence[int],
                   updater: Callable[[FeedForward], ParameterUpdater]):
    layers = []
    network = FeedForward(layers)

    for i in range(len(nodes) - 1):
        activation = RectifiedLinearUnitActivation(leak=.01)
        if i == len(nodes) - 2:
            activation = IdentityActivation()

        layers.append(
            layer(nodes[i],
                  nodes[i + 1],
                  level=i,
                  activation=activation,
                  parameter_updater=updater(network),
                  # parameter_generator=ConstantParameterGenerator())
                  parameter_generator=RandomParameterGenerator())
                  # parameter_generator=SequenceParameterGenerator())
        )
    return network


def quad(run: int):
    start_time = time.time()
    # epochs = 10000 * 2**(run % 5)
    epochs = 100000
    epoch = 0
    nodes = [1, 5, 5, 1]
    learning_rate = .001
    network = create_network(LinearLayer,
                             nodes,
                             lambda net: simple_updater(epochs, learning_rate, lambda: epoch))
    batch_size = 2
    trainer = ClosedFormFunctionTrainer(network, lambda x: x * np.math.sin(x), (-5, 5),
                                        batch_size)
    file = h5py.File('quad_' + str(run) + '.h5', 'w')
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.EPOCHS, data=epochs)
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.BATCH_SIZE, data=batch_size)
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.LAYERS, data=nodes)
    stop_time = time.time()
    print("setup:", stop_time - start_time)

    start_time = time.time()
    # Train the model.
    for i in range(epochs):
        epoch = i
        result = trainer.batch_train(batch_size)
        write_batch_result(file, epochs, result)
        if epoch % 100 == 0:
            print("run_" + str(run) + ":", epoch)

    # Validate the final model.
    validation = trainer.validate()
    file.require_group(Group.VALIDATION).create_dataset('error', data=validation.error)
    file.require_group(Group.VALIDATION).create_dataset('inputs', data=validation.inputs)
    file.require_group(Group.VALIDATION).create_dataset('expected', data=validation.expected)
    file.require_group(Group.VALIDATION).create_dataset('actual', data=validation.actual)

    stop_time = time.time()
    print("experiment", stop_time - start_time)

    start_time = time.time()
    close_file(file)
    stop_time = time.time()
    print("output:", stop_time - start_time)


if __name__ == "__main__":
    pool = multiprocessing.Pool(7)
    pool.map(quad, range(7))
