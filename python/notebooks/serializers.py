from typing import Any, Mapping

import collections

from python.notebooks.domain_objects import ParameterSet
from python.notebooks.trainers import BatchResult


def tolist(target: Any):
    if isinstance(target, collections.Iterable) and not isinstance(target, str):
        return [tolist(item) for item in target]
    return target


def serialize_parameter_set(parameter_set: ParameterSet) -> dict:
    return {
        "name": parameter_set.name,
        "values": tolist(parameter_set.values),
        "gradients": tolist(parameter_set.gradients),
        "deltas": tolist(parameter_set.deltas),
    }


def serialize_parameter_set_map(set_map: Mapping[str, ParameterSet]) -> dict:
    return {key: serialize_parameter_set(value) for key, value in set_map.items()}


def serialize_batch_result(result: BatchResult):
    return {
        "batch_number": result.batch_number,
        "batch_size": result.batch_size,
        "total_error": result.total_error,
        "avg_error": result.avg_error,
        "parameters": [serialize_parameter_set_map(param_set) for param_set in result.parameters]
    }
