import unittest

from algorithms.combinatorics import combinations, permutations, combinations_iter


class CombinatoricsTest(unittest.TestCase):
    def test_combinations(self):
        self.assertSetEqual({'', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc'}, combinations('abc'))

    def test_combinations_iterative(self):
        self.assertSetEqual({'', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc'}, combinations_iter('abc'))

    def test_permutations(self):
        self.assertSetEqual({'', 'a', 'b', 'c', 'ab', 'ba', 'ac', 'ca', 'bc', 'cb', 'abc', 'acb',
                             'bac', 'bca', 'cab', 'cba'},
                            permutations('abc'))
