import unittest

from miniparse.ini import ParseError, get_bool, parse_ini

SAMPLE = """
; global settings
name = demo

[db]
# connection
Host = localhost
url = postgres://db?sslmode=require
debug = yes
"""


class TestParseIni(unittest.TestCase):
    def setUp(self):
        self.cfg = parse_ini(SAMPLE)

    def test_default_section(self):
        self.assertEqual(self.cfg["DEFAULT"], {"name": "demo"})

    def test_keys_are_lowercased(self):
        self.assertEqual(self.cfg["db"]["host"], "localhost")

    def test_value_may_contain_equals(self):
        self.assertEqual(self.cfg["db"]["url"], "postgres://db?sslmode=require")

    def test_bad_line(self):
        with self.assertRaises(ParseError):
            parse_ini("[a]\nnot a pair\n")


class TestGetBool(unittest.TestCase):
    def test_true_word(self):
        self.assertTrue(get_bool(parse_ini(SAMPLE), "db", "debug"))

    def test_missing_key_uses_default(self):
        self.assertFalse(get_bool(parse_ini(SAMPLE), "db", "verbose"))


if __name__ == "__main__":
    unittest.main()
