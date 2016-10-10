from collections import deque
from enum import Enum
from typing import TypeVar, Generic, Any, Callable
from abc import ABCMeta, abstractmethod


class Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: pass


T = TypeVar('T', bound=Comparable)


class BinaryTreeNode(Generic[T]):
    def __init__(self, data: T):
        self.parent = None  # type: BinaryTreeNode[T]
        self.left = None  # type: BinaryTreeNode[T]
        self.right = None  # type: BinaryTreeNode[T]
        self.data = data


class TraversOrder(Enum):
    pre_order = 1
    in_order = 2
    post_order = 3


class BinaryTree(Generic[T], metaclass=ABCMeta):
    def __init__(self):
        self.root = None  # type: BinaryTreeNode[T]

    @abstractmethod
    def add(self, data: T):
        pass

    @abstractmethod
    def remove(self, data: T) -> bool:
        pass

    def pre_order(self, action: Callable[[T], None]):
        self._depth_traversal(self.root, TraversOrder.pre_order, action)

    def in_order(self, action: Callable[[T], None]):
        self._depth_traversal(self.root, TraversOrder.in_order, action)

    def post_order(self, action: Callable[[T], None]):
        self._depth_traversal(self.root, TraversOrder.post_order, action)

    def breadth_order(self, action: Callable[[T], None]):
        if self.root is None:
            return
        visited = set()
        queue = deque()
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.popleft()
            if node in visited:
                continue
            action(node.data)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
            visited.add(node)

    def _depth_traversal(self, node: BinaryTreeNode[T], order: TraversOrder,
                         action: Callable[[T], None]):
        if node is None:
            return

        if order is TraversOrder.pre_order:
            action(node.data)
        self._depth_traversal(node.left, order, action)

        if order is TraversOrder.in_order:
            action(node.data)

        self._depth_traversal(node.right, order, action)

        if order is TraversOrder.post_order:
            action(node.data)


class BinarySearchTree(BinaryTree[T]):
    def add(self, data: T):
        node = BinaryTreeNode(data)
        if self.root is None:
            self.root = node
            return  # Exit early

        parent = self.root
        cur_node = self.root
        while cur_node is not None:
            parent = cur_node
            if data < parent.data:
                cur_node = parent.left
            else:
                cur_node = parent.right

        if data < parent.data:
            parent.left = node
        else:
            parent.right = node

        node.parent = parent

    def remove(self, data) -> bool:
        if self.root is None:
            return False

        cur_node = self.root
        while cur_node is not None and cur_node.data != data:
            if data < cur_node.data:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        if cur_node is None:
            return False

        replacement = None  # type: BinaryTreeNode[T]

        if cur_node.right is None:
            replacement = cur_node.left
        elif cur_node.right.left is None:
            replacement = cur_node.right
            replacement.left = cur_node.left
        else:
            replacement = cur_node.right.left
            while replacement.left is not None:
                replacement = replacement.left
            replacement.parent.left = replacement.right
            replacement.left = cur_node.left
            replacement.right = cur_node.right

        # Did not remove a leaf node.
        if replacement is not None:
            replacement.parent = cur_node.parent
            if replacement.left is not None:
                replacement.left.parent = replacement

            if replacement.right is not None:
                replacement.right.parent = replacement

        if cur_node is self.root:
            self.root = replacement
        else:
            if cur_node.parent.right is cur_node:
                cur_node.parent.right = replacement
            else:
                cur_node.parent.left = replacement

        return True
