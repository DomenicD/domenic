from typing import Sequence, Mapping

import numpy as np
import h5py
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


def write_deltas(group: h5py.Group, epoch: str, params: ParameterSet):
    group.require_group("deltas").require_group("values").require_dataset(
        epoch, data=np.reshape([delta.value for delta in params.deltas.flatten()],
                               params.deltas.shape))
    # group.create_group("deltas").create_group("steps").require_dataset(epoch,)


def write_parameters(file: h5py.File, epoch: str, parameters: Sequence[Mapping[str, ParameterSet]]):
    for layer in range(len(parameters)):
        layer_group = file[Group.PARAMETERS].require_group("layer").require_group(str(layer))
        for params in parameters[layer].values():
            param_group = layer_group.require_group("parameter_set").require_group(params.name)
            param_group.require_group("values").require_dataset(epoch, data=params.values)
            param_group.require_group("gradients").require_dataset(epoch, data=params.gradients)
            write_deltas(param_group, epoch, params)


def write_batch_result(file: h5py.File, total_epochs: int, result: BatchResult):
    epoch = result.batch_number - 1
    # TODO: clean up the dataset creation/require.
    file.require_dataset(Group.TOTAL_ERROR, (total_epochs,), np.dtype(float))[epoch] = result.total_error
    file.require_dataset(Group.AVERAGE_ERROR, (total_epochs,), np.dtype(float))[epoch] = result.avg_error
    # TODO: Update write_parameters to append to dataset rather than creating a new one for each epoch.
    # write_parameters(file, epoch, result.parameters)
    file.require_dataset(Group.INPUTS, (total_epochs,) + np.shape(result.inputs), np.dtype(float))[epoch] = result.inputs
    file.require_dataset(Group.EXPECTED, (total_epochs,) + np.shape(result.expected), np.dtype(float))[epoch] = result.expected
    file.require_dataset(Group.ACTUAL, (total_epochs,) + np.shape(result.actual), np.dtype(float))[epoch] = result.actual


def quad():
    epochs = 100
    layers = [1, 2, 1]
    network = am.feed_forward_network(QuadraticLayer, layers, 'SimpleUpdater')
    batch_size = 50
    trainer = ClosedFormFunctionTrainer(network, lambda x: x ** 2, (-10, 10), batch_size)
    file = file = h5py.File('test.h5', 'w')
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.EPOCHS, data=epochs)
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.BATCH_SIZE, data=batch_size)
    file.require_group(Group.CONFIGURATION).create_dataset(Dataset.LAYERS, data=layers)

    for i in range(epochs):
        result = trainer.batch_train()
        write_batch_result(file, epochs, result)

    close_file(file)


if __name__ == "__main__":
    quad()
