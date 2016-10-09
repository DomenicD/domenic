import unittest

from data_structures.linked_list import SingleLinkList, kth_from_end


class LinkListTest(unittest.TestCase):
    def test_kth_from_end(self):
        link_list = SingleLinkList()  # SingleLinkList[int]
        for i in range(7):
            link_list.add(i)

        self.assertEqual(kth_from_end(link_list, -1), None)
        self.assertEqual(kth_from_end(link_list, 0), 6)
        self.assertEqual(kth_from_end(link_list, 2), 4)
        self.assertEqual(kth_from_end(link_list, 5), 1)
        self.assertEqual(kth_from_end(link_list, 6), 0)
        self.assertEqual(kth_from_end(link_list, 7), None)
        self.assertEqual(kth_from_end(link_list, 8), None)
        self.assertEqual(kth_from_end(link_list, 10), None)
