from typing import TypeVar, Generic

T = TypeVar('T')


class SingleLinkNode(Generic[T]):
    def __init__(self, data: T):
        self.next = None  # type: SingleLinkNode[T]
        self.data = data


class SingleLinkList(Generic[T]):
    def __init__(self):
        self.head = None  # type: SingleLinkNode[T]

    def add(self, data: T):
        node = SingleLinkNode(data)
        if self.head is None:
            self.head = node
            return  # Exit early

        cur_node = self.head
        while cur_node.next is not None:
            cur_node = cur_node.next

        cur_node.next = node

    def remove(self, data: T) -> bool:
        # Head is None
        if self.head is None:
            return False

        # Need to remove head
        if self.head.data == data:
            next_head = self.head.next
            self.head.next = None
            self.head = next_head
            return True

        # Node may or may not be in the list after head
        cur_node = self.head
        while cur_node.next is not None and cur_node.next.data != data:
            cur_node = cur_node.next

        if cur_node.next is None:
            return False

        to_remove = cur_node.next
        cur_node.next = to_remove.next
        to_remove.next = None
        return True


class DoubleLinkNode(Generic[T]):
    def __init__(self, data: T):
        self.next = None  # type: DoubleLinkNode[T]
        self.previous = None  # type: DoubleLinkNode[T]
        self.data = data


class DoubleLinkList(Generic[T]):
    def __init__(self):
        self.head = None  # type: DoubleLinkNode[T]
        self.tail = None  # type: DoubleLinkNode[T]

    def add(self, data: T):
        node = DoubleLinkNode(data)
        if self.head is None:
            self.head = node
            self.tail = node
            return  # Exit early

        cur_node = self.head
        while cur_node.next is not None:
            cur_node = cur_node.next

        cur_node.next = node
        node.previous = cur_node

        if cur_node is self.tail:
            self.tail = node

    def remove(self, data: T) -> bool:
        if self.head is None:
            return False

        cur_node = self.head
        while cur_node is not None and cur_node.data != data:
            cur_node = cur_node.next

        if cur_node is None:
            return False

        if cur_node is not self.head and cur_node is not self.tail:
            cur_node.next.previous = cur_node.previous
            cur_node.previous.next = cur_node.next

        if cur_node is self.head:
            self.head = cur_node.next
            if self.head is not None:
                self.head.previous = None

        if cur_node is self.tail:
            self.tail = cur_node.previous
            if self.tail is not None:
                self.tail.next = None

        cur_node.next = None
        cur_node.previous = None

        return True


def kth_from_end(link_list: SingleLinkList[T], k: int) -> T:
    if k < 0:
        return None

    head = link_list.head
    if head is None:
        return None

    count = 0

    fast_node = head
    k_back_node = head

    while fast_node.next is not None:
        fast_node = fast_node.next
        if count >= k:
            k_back_node = k_back_node.next
        count += 1

    if count < k:
        return None

    return k_back_node.data


