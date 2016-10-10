import unittest

from algorithms.permutation import combinations


class PermutationTest(unittest.TestCase):
    def test_combinations(self):
        self.assertSetEqual({'', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc'}, combinations('abc'))
