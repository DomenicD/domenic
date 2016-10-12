from typing import List

import math


def merge_sort(arr: List):
    length = len(arr)
    buffer = [None] * length  # Initialize buffer array of same size
    window = 2
    is_sorted = False
    while not is_sorted:
        workers = math.ceil(length / window)
        for i in range(workers):
            _merge_sort(arr, buffer, i * window, window, length)

        if length < window:
            is_sorted = True

        window *= 2


def _merge_sort(arr: List, buffer: List, start: int, window: int, length: int):
    middle = min(start + math.floor(window / 2), length)
    end = min(start + window, length)
    left = start
    right = middle
    buffer_index = start
    while left < middle or right < end:
        if left < middle and right < end:
            if arr[left] > arr[right]:
                value = arr[right]
                right += 1
            else:
                value = arr[left]
                left += 1
        elif left < middle:
            value = arr[left]
            left += 1
        else:
            value = arr[right]
            right += 1

        buffer[buffer_index] = value
        buffer_index += 1

    for i in range(start, end):
        arr[i] = buffer[i]
