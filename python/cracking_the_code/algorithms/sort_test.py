import unittest

from algorithms.dynamic_programming import running_child
from algorithms.sort import merge_sort


class SortTest(unittest.TestCase):
    def test_merge_sort(self):
        arr = [1, 2, 1, 4, 1, 5]
        merge_sort(arr)
        self.assertListEqual(arr, [1, 1, 1, 2, 4, 5])
