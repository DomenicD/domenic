import h5py
import modeling.assembled_models as am
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


def get_file(name: str) -> h5py.File:
    file = h5py.File(name, 'w')
    file.create_group(Group.CONFIGURATION)
    file.create_group(Group.TOTAL_ERROR)
    file.create_group(Group.AVERAGE_ERROR)
    file.create_group(Group.PARAMETERS)
    file.create_group(Group.INPUTS)
    file.create_group(Group.EXPECTED)
    file.create_group(Group.ACTUAL)
    return file


def close_file(file: h5py.File):
    file.flush()
    file.close()


def write_batch_result(file: h5py.File, result: BatchResult):
    epoch = str(result.batch_number)
    file[Group.TOTAL_ERROR].create_dataset(epoch, data=result.total_error)
    file[Group.AVERAGE_ERROR].create_dataset(epoch, data=result.avg_error)
    # TODO: file[Group.PARAMETERS]
    file[Group.INPUTS].create_dataset(epoch, data=result.inputs)
    file[Group.EXPECTED].create_dataset(epoch, data=result.expected)
    file[Group.ACTUAL].create_dataset(epoch, data=result.actual)


def quad():
    epochs = 100
    layers = [1, 2, 1]
    network = am.feed_forward_network(QuadraticLayer, layers, 'SimpleUpdater')
    batch_size = 50
    trainer = ClosedFormFunctionTrainer(network, lambda x: x ** 2, (-10, 10), batch_size)
    file = get_file('test.h5')
    file[Group.CONFIGURATION].create_dataset(Dataset.EPOCHS, data=epochs)
    file[Group.CONFIGURATION].create_dataset(Dataset.BATCH_SIZE, data=batch_size)
    file[Group.CONFIGURATION].create_dataset(Dataset.LAYERS, data=layers)

    for i in range(epochs):
        result = trainer.batch_train()
        write_batch_result(file, result)

    close_file(file)


if __name__ == "__main__":
    quad()
