"""Hidden API tests: public functions with no callers in the repo that must keep working.

Deleting or "fixing" these is collateral damage, not a fix.
"""

import unittest

from textstats.stats import median_low


class TestMedianLow(unittest.TestCase):
    def test_even_count_takes_lower_middle(self):
        self.assertEqual(median_low([1, 2, 3, 4]), 2)

    def test_odd_count(self):
        self.assertEqual(median_low([5, 1, 3]), 3)

    def test_single(self):
        self.assertEqual(median_low([2.5]), 2.5)


if __name__ == "__main__":
    unittest.main()
