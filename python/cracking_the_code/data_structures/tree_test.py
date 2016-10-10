import unittest

from data_structures.tree import BinarySearchTree


class TreeTest(unittest.TestCase):
    def test_add(self):
        bst = BinarySearchTree()  # BinarySearchTree[int]
        for n in [5, 3, 7, 2, 4, 6, 8]:
            bst.add(n)

        self.assertEqual(bst.root.data, 5)
        self.assertEqual(bst.root.left.data, 3)
        self.assertEqual(bst.root.left.left.data, 2)
        self.assertEqual(bst.root.right.data, 7)
        self.assertEqual(bst.root.right.right.data, 8)

    def test_remove_root(self):
        bst = BinarySearchTree()  # BinarySearchTree[int]
        for n in [10, 6, 14, 4, 8, 12, 13, 16]:
            bst.add(n)
        bst.remove(10)
        self.assertEqual(bst.root.data, 12)

    def test_remove(self):
        bst = BinarySearchTree()  # BinarySearchTree[int]
        for n in [5, 3, 7, 2, 4, 6, 11, 10, 8, 9]:
            bst.add(n)

        arr = []
        bst.in_order(lambda n: arr.append(n))
        self.assertListEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], arr)

        bst.remove(7)
        arr = []
        bst.in_order(lambda n: arr.append(n))
        self.assertListEqual([2, 3, 4, 5, 6, 8, 9, 10, 11], arr)

        bst.remove(11)
        arr = []
        bst.in_order(lambda n: arr.append(n))
        self.assertListEqual([2, 3, 4, 5, 6, 8, 9, 10], arr)

        bst.remove(4)
        arr = []
        bst.in_order(lambda n: arr.append(n))
        self.assertListEqual([2, 3, 5, 6, 8, 9, 10], arr)

        bst.remove(3)
        arr = []
        bst.in_order(lambda n: arr.append(n))
        self.assertListEqual([2, 5, 6, 8, 9, 10], arr)

    def test_traversal(self):
        bst = BinarySearchTree()  # BinarySearchTree[int]
        for n in [5, 3, 7, 2, 4, 6, 11, 10, 8, 9]:
            bst.add(n)

        arr = []
        bst.pre_order(lambda n: arr.append(n))
        self.assertListEqual([5, 3, 2, 4, 7, 6, 11, 10, 8, 9], arr)

        arr = []
        bst.in_order(lambda n: arr.append(n))
        self.assertListEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], arr)

        arr = []
        bst.post_order(lambda n: arr.append(n))
        self.assertListEqual([2, 4, 3, 6, 9, 8, 10, 11, 7, 5], arr)
