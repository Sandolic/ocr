class UnionTree:
    def __init__(self):
        """
        UnionTree object constructor
        :return UnionTree
        """

        self.parents = {}

    def new_parent(self, elem: int):
        """
        Initialize new parent
        :param int elem: given element (label)
        """

        self.parents[elem] = elem

    def find_parent(self, elem: int) -> int:
        """
        Find parent of given element
        :param int elem: given element (label)
        :return int
        """

        # If an element is a parent, then the parent of the element is the element
        while self.parents[elem] != elem:
            elem = self.parents[elem]
        return elem

    def set_parent(self, elem: int, parent: int):
        """
        Set parent of given element
        :param int elem: given element (label)
        :param int parent: given parent
        """

        # Path compression : all parents of the element are given the same parent new until the original parent
        while self.parents[elem] != elem:
            elem = self.parents[elem]
            self.parents[elem] = parent
        self.parents[elem] = parent

    def union(self, elem1: int, elem2: int) -> int:
        """
        Process union of two given elements
        :param int elem1: given first element (label)
        :param int elem2: given second element (label)
        :return int
        """

        parent_elem1 = self.find_parent(elem1)
        # If elem2 isn't parent of elem1, we must set same parent for elem1 and elem2
        if parent_elem1 != elem2:
            parent_elem2 = self.find_parent(elem2)
            # We want the lowest parent possible
            if parent_elem1 > parent_elem1:
                parent_elem1 = parent_elem2
            self.set_parent(elem2, parent_elem1)
        self.set_parent(elem1, parent_elem1)
        return parent_elem1

    def flatten(self):
        """
        Flatten the parents dictionary : all elements have the lowest parent possible
        """

        for key, value in self.parents.items():
            self.parents[key] = self.parents[self.parents[key]]
