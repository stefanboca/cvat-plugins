from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._values = OrderedDict()

    def get(self, key):
        value = self._values.pop(key, None)
        if value is not None:
            self._values[key] = value
        return value

    def put(self, key, value):
        self._values.pop(key, None)
        self._values[key] = value
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
