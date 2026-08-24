from collections import OrderedDict
from typing import Generic, Hashable, Optional, TypeVar


Key = TypeVar("Key", bound=Hashable)
Value = TypeVar("Value")


class LRUCache(Generic[Key, Value]):
    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("The cache size must be at least 1")
        self.max_size = max_size
        self._values: OrderedDict[Key, Value] = OrderedDict()

    def get(self, key: Key) -> Optional[Value]:
        value = self._values.get(key)
        if value is None:
            return None
        self._values.move_to_end(key)
        return value

    def put(self, key: Key, value: Value) -> None:
        self._values.pop(key, None)
        self._values[key] = value
        if len(self._values) > self.max_size:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)
