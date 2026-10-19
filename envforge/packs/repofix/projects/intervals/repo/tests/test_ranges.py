import unittest

from intervals.ranges import Interval, merge, overlaps, total_covered


class TestInterval(unittest.TestCase):
    def test_end_is_excluded(self):
        iv = Interval(1, 3)
        self.assertTrue(iv.contains(1))
        self.assertFalse(iv.contains(3))

    def test_rejects_backwards(self):
        with self.assertRaises(ValueError):
            Interval(5, 1)


class TestOverlaps(unittest.TestCase):
    def test_overlapping(self):
        self.assertTrue(overlaps(Interval(1, 4), Interval(3, 6)))

    def test_touching_is_not_overlapping(self):
        a, b = Interval(1, 3), Interval(3, 5)
        self.assertFalse(overlaps(a, b))
        self.assertFalse(overlaps(b, a))


class TestMerge(unittest.TestCase):
    def test_touching_intervals_merge(self):
        self.assertEqual(merge([Interval(3, 5), Interval(1, 3)]), [Interval(1, 5)])

    def test_total_covered_counts_overlap_once(self):
        self.assertEqual(total_covered([Interval(0, 2), Interval(1, 3)]), 3)


if __name__ == "__main__":
    unittest.main()
