import sys
import unittest
from pathlib import Path


NUCLIO_DIRECTORY = Path(__file__).resolve().parents[1] / "nuclio"
sys.path.insert(0, str(NUCLIO_DIRECTORY))

from lru_cache import LRUCache  # noqa: E402


class LRUCacheTest(unittest.TestCase):
    def test_get_marks_entry_as_recent(self):
        cache = LRUCache(2)
        cache.put("first", 1)
        cache.put("second", 2)

        self.assertEqual(cache.get("first"), 1)
        cache.put("third", 3)

        self.assertEqual(cache.get("first"), 1)
        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("third"), 3)

    def test_put_replaces_entry_without_growing_cache(self):
        cache = LRUCache(1)
        cache.put("key", 1)
        cache.put("key", 2)

        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.get("key"), 2)

    def test_cache_rejects_zero_size(self):
        with self.assertRaises(ValueError):
            LRUCache(0)


if __name__ == "__main__":
    unittest.main()
