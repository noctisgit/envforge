"""Hidden API tests: public functions with no callers in the repo that must keep working.

Deleting or "fixing" these is collateral damage, not a fix.
"""

import unittest
from datetime import date

from intervals.dates import iso_week_label
from intervals.ranges import Interval, overlaps_closed


class TestClosedOverlap(unittest.TestCase):
    def test_touching_endpoints_count(self):
        self.assertTrue(overlaps_closed(Interval(1, 3), Interval(3, 5)))
        self.assertTrue(overlaps_closed(Interval(3, 5), Interval(1, 3)))
        self.assertFalse(overlaps_closed(Interval(1, 2), Interval(3, 5)))


class TestIsoWeekLabel(unittest.TestCase):
    def test_iso_year_differs_near_new_year(self):
        self.assertEqual(iso_week_label(date(2024, 12, 30)), "2025-W01")
        self.assertEqual(iso_week_label(date(2024, 6, 3)), "2024-W23")


if __name__ == "__main__":
    unittest.main()
