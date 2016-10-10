from collections import Sequence


def combinations(seq: Sequence, memory: set = set()):
    memory.add(seq)
    for i in range(len(seq)):
        sub_seq = seq[:i] + seq[(1 + i):]
        if sub_seq not in memory:
            combinations(sub_seq, memory)
    return memory


def naive_combinations(seq: Sequence, memory: set = set()):
    memory.add(seq)
    for i in range(len(seq)):
        sub_seq = seq[:i] + seq[(1 + i):]
        naive_combinations(sub_seq, memory=memory)
    return memory
