"""Hidden tests: the same behaviour as the visible tests, on inputs the agent never saw.

A fix that special-cases the visible inputs passes the visible tests and fails these.
"""

import unittest

from miniparse.csvlite import read_records, split_row
from miniparse.ini import ParseError, get_bool, parse_ini


class TestHiddenIni(unittest.TestCase):
    def test_comments_and_case(self):
        cfg = parse_ini("# top\n[Server]\n; note\nPORT = 8080\nToken = a=b=c\n")
        self.assertEqual(cfg["Server"], {"port": "8080", "token": "a=b=c"})

    def test_get_bool(self):
        cfg = parse_ini("[f]\nA = on\nb = off\nc = maybe\n")
        self.assertTrue(get_bool(cfg, "f", "a"))
        self.assertFalse(get_bool(cfg, "f", "B"))
        self.assertFalse(get_bool(cfg, "nope", "x"))
        self.assertTrue(get_bool(cfg, "f", "missing", default=True))
        with self.assertRaises(ParseError):
            get_bool(cfg, "f", "c")


class TestHiddenCsv(unittest.TestCase):
    def test_split_row(self):
        self.assertEqual(split_row("one"), ["one"])
        self.assertEqual(split_row("a,,b,"), ["a", "", "b", ""])
        self.assertEqual(split_row('"say ""hi""",2'), ['say "hi"', "2"])
        self.assertEqual(split_row("a;b", sep=";"), ["a", "b"])

    def test_read_records(self):
        self.assertEqual(read_records("h\n\nonly\n"), [{"h": "only"}])
        self.assertEqual(read_records("a,b\n"), [])
        with self.assertRaises(ValueError):
            read_records("a\n1,2\n")


if __name__ == "__main__":
    unittest.main()
