import unittest

from miniparse.csvlite import column, read_records, split_row


class TestSplitRow(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(split_row("a,b,c"), ["a", "b", "c"])

    def test_quoted_separator(self):
        self.assertEqual(split_row('x,"1,5",y'), ["x", "1,5", "y"])


class TestReadRecords(unittest.TestCase):
    def test_header_and_rows(self):
        recs = read_records("name,qty\napple,3\npear,5\n")
        self.assertEqual(recs, [{"name": "apple", "qty": "3"}, {"name": "pear", "qty": "5"}])

    def test_ragged_row_is_an_error(self):
        with self.assertRaises(ValueError):
            read_records("a,b\n1\n")

    def test_column(self):
        self.assertEqual(column(read_records("k,v\na,1\nb,2"), "v"), ["1", "2"])


if __name__ == "__main__":
    unittest.main()
