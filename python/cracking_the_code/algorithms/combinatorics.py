from collections import Sequence
from typing import Any


def combinations(seq: str, memory: set = set()):
    memory.add(seq)
    for i in range(len(seq)):
        sub_seq = seq[:i] + seq[(1 + i):]
        if sub_seq not in memory:
            combinations(sub_seq, memory)
    return memory


# Runtime: O(2^n)
# Actual runtime: 2^n - 1
def combinations_iter(seq: Sequence):
    memory = set()
    memory.add(seq[0:0])
    for el in seq:
        for item in list(memory):
            memory.add(item + el)
    return memory


def permutations(seq: Sequence):
    memory = set()
    memory.add(seq[0:0])
    for el in seq:
        _permutations(el, memory)
    return memory


# Runtime: O(n*n!)
# Actual runtime: Sum(n!/(n - i)!, {i, 0, n})
def _permutations(el: Any, memory: set):
    for item in list(memory):
        for i in range(len(item) + 1):
            next_str = item[:i] + el + item[i:]
            memory.add(next_str)
