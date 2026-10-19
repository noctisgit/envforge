import unittest

from inventory.orders import Line, Order, fulfil, line_total, order_total, pack_boxes
from inventory.stock import Inventory, OutOfStock


class TestPricing(unittest.TestCase):
    def test_line_total_with_discount(self):
        self.assertEqual(line_total(Line("A", 2, 10.0), discount=0.25), 15.0)

    def test_order_total_adds_default_tax(self):
        self.assertEqual(order_total(Order([Line("A", 1, 100.0)])), 108.0)


class TestFulfil(unittest.TestCase):
    def test_ships_all_lines(self):
        inv = Inventory()
        inv.add("A", 5)
        inv.add("B", 5)
        fulfil(Order([Line("A", 2, 1.0), Line("B", 5, 1.0)]), inv)
        self.assertEqual((inv.count("A"), inv.count("B")), (3, 0))

    def test_short_order_ships_nothing(self):
        inv = Inventory()
        inv.add("A", 5)
        inv.add("B", 1)
        with self.assertRaises(OutOfStock):
            fulfil(Order([Line("A", 2, 1.0), Line("B", 3, 1.0)]), inv)
        self.assertEqual(inv.count("A"), 5)


class TestPackBoxes(unittest.TestCase):
    def test_last_box_may_be_partial(self):
        self.assertEqual(pack_boxes(list("abcde"), 2), [["a", "b"], ["c", "d"], ["e"]])


if __name__ == "__main__":
    unittest.main()
