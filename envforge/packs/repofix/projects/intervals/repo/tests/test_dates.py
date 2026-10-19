import unittest
from datetime import date

from intervals.dates import add_business_days, days_between, days_in_month, is_weekend, quarter


class TestDates(unittest.TestCase):
    def test_days_between(self):
        self.assertEqual(days_between(date(2024, 1, 1), date(2024, 1, 31)), 30)

    def test_weekend(self):
        self.assertTrue(is_weekend(date(2024, 6, 1)))  # a Saturday
        self.assertFalse(is_weekend(date(2024, 6, 3)))  # a Monday

    def test_add_business_days_skips_weekend(self):
        self.assertEqual(add_business_days(date(2024, 6, 7), 1), date(2024, 6, 10))

    def test_days_in_leap_february(self):
        self.assertEqual(days_in_month(2024, 2), 29)

    def test_quarter(self):
        self.assertEqual(quarter(date(2024, 3, 15)), 1)


if __name__ == "__main__":
    unittest.main()
