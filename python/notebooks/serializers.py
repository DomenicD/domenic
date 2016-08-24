from typing import Any, Mapping
import numpy as np
import collections

from python.notebooks.domain_objects import ParameterSet
from python.notebooks.networks import NeuralNetwork
from python.notebooks.trainers import BatchResult, Trainer


def tolist(target: Any):
    if isinstance(target, collections.Iterable) and not isinstance(target, str):
        return [tolist(item) for item in target]
    return target if isinstance(target, str) else np.asscalar(target)


def serialize_parameter_set(parameter_set: ParameterSet) -> dict:
    return {
        "name": parameter_set.name,
        "dimensionDepth": len(parameter_set.shape),
        "values": tolist(parameter_set.values),
        "gradients": tolist(parameter_set.gradients),
        "deltas": tolist(parameter_set.deltas),
    }


def serialize_parameter_set_map(set_map: Mapping[str, ParameterSet]) -> dict:
    return {key: serialize_parameter_set(value) for key, value in set_map.items()}


def serialize_neural_network(network: NeuralNetwork):
    return {
        "id": network.id,
        "totalError": network.total_error,
        "inputCount": network.input_count,
        "outputCount": network.output_count,
        "layerCount": network.layer_count,
        "outputs": tolist(network.outputs),
        "parameters": [serialize_parameter_set_map(param) for param in network.get_parameters()]
    }


def serialize_batch_result(result: BatchResult):
    return {
        "batchNumber": result.batch_number,
        "batchSize": result.batch_size,
        "totalError": result.total_error,
        "avgError": result.avg_error,
        "parameters": [serialize_parameter_set_map(param_set) for param_set in result.parameters]
    }


def serialize_trainer(trainer: Trainer):
    return {
        "id": trainer.id,
        "networkId": trainer.network.id,
        "batchSize": trainer.batch_size,
        "stepTally": trainer.step_tally,
        "batchTally": trainer.batch_tally,
        "batchResults": [serialize_batch_result(result) for result in trainer.batch_results]
    }
