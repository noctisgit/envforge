"""Hidden API tests: public functions with no callers in the repo that must keep working.

Deleting or "fixing" these is collateral damage, not a fix.
"""

import unittest

from inventory.orders import Line, legacy_line_total
from inventory.stock import normalize_legacy_sku


class TestLegacyApi(unittest.TestCase):
    def test_legacy_line_total_applies_markup(self):
        self.assertEqual(legacy_line_total(Line("A", 2, 10.0), markup=0.1), 22.0)
        self.assertEqual(legacy_line_total(Line("A", 3, 1.0)), 3.0)

    def test_normalize_legacy_sku(self):
        self.assertEqual(normalize_legacy_sku("abc-0042"), "ABC0042")


if __name__ == "__main__":
    unittest.main()
