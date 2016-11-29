from typing import Sequence, Mapping, Tuple

import numpy as np
import h5py
import time

import modeling.assembled_models as am
from modeling.domain_objects import ParameterSet
from modeling.layers import QuadraticLayer
from modeling.trainers import ClosedFormFunctionTrainer, BatchResult


class Group:
    CONFIGURATION = 'configuration'
    TOTAL_ERROR = 'total_error'
    AVERAGE_ERROR = 'average_error'
    PARAMETERS = 'parameters'
    INPUTS = 'inputs'
    EXPECTED = 'expected'
    ACTUAL = 'actual'


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


def quad():
    start_time = time.time()
    epochs = 1000
    layers = [1, 2, 1]
    network = am.feed_forward_network(QuadraticLayer, layers, 'SimpleUpdater')
    batch_size = 100
    trainer = ClosedFormFunctionTrainer(network, lambda x: x ** 2, (-10, 10), batch_size)
    file = h5py.File('test.h5', 'w')
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.EPOCHS, data=epochs)
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.BATCH_SIZE, data=batch_size)
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.LAYERS, data=layers)
    stop_time = time.time()
    print("setup:", stop_time - start_time)

    start_time = time.time()
    for i in range(epochs):
        result = trainer.batch_train()
        write_batch_result(file, epochs, result)
    stop_time = time.time()
    print("experiment", stop_time - start_time)

    start_time = time.time()
    close_file(file)
    stop_time = time.time()
    print("output:", stop_time - start_time)


if __name__ == "__main__":
    quad()
