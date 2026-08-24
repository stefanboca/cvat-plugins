import unittest

from lru_cache import LRUCache


class LRUCacheTest(unittest.TestCase):
    def test_evicts_the_least_recent_entry(self):
        cache = LRUCache(2)
        cache.put("first", 1)
        cache.put("second", 2)
        self.assertEqual(cache.get("first"), 1)
        cache.put("third", 3)
        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("first"), 1)
        self.assertEqual(cache.get("third"), 3)

    def test_replaces_an_entry(self):
        cache = LRUCache(1)
        cache.put("key", 1)
        cache.put("key", 2)
        self.assertEqual(cache.get("key"), 2)


if __name__ == "__main__":
    unittest.main()
