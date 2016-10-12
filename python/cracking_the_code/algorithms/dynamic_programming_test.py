import unittest

from algorithms.dynamic_programming import running_child


class DynamicProgrammingTest(unittest.TestCase):
    def test_running_child(self):
        self.assertEqual(running_child(0), 0)
        self.assertEqual(running_child(1), 1)
        self.assertEqual(running_child(2), 2)
        self.assertEqual(running_child(3), 4)
        self.assertEqual(running_child(4), 7)
        self.assertEqual(running_child(20), 121415)

