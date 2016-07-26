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
