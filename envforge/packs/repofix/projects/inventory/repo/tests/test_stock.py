import unittest

from inventory.stock import Inventory, OutOfStock


class TestInventory(unittest.TestCase):
    def setUp(self):
        self.inv = Inventory()
        self.inv.add("bolt", 10)

    def test_count_is_case_insensitive(self):
        self.assertEqual(self.inv.count("bolt"), 10)
        self.assertEqual(self.inv.count("BOLT"), 10)

    def test_can_ship_everything_on_hand(self):
        self.inv.remove("bolt", 10)
        self.assertEqual(self.inv.count("bolt"), 0)

    def test_cannot_ship_more_than_on_hand(self):
        with self.assertRaises(OutOfStock):
            self.inv.remove("bolt", 11)

    def test_low_stock_includes_threshold(self):
        self.inv.add("nut", 5)
        self.assertEqual(self.inv.low_stock(threshold=5), ["NUT"])

    def test_low_stock_default_threshold(self):
        self.inv.add("washer", 3)
        self.assertEqual(self.inv.low_stock(), ["WASHER"])


if __name__ == "__main__":
    unittest.main()
