"""Hidden API tests: public functions with no callers in the repo that must keep working.

Deleting or "fixing" these is collateral damage, not a fix.
"""

import unittest

from miniparse.csvlite import head


class TestHead(unittest.TestCase):
    def test_header_plus_n_rows(self):
        self.assertEqual(head("h\n1\n2\n3\n", n=2), ["h", "1", "2"])

    def test_default_and_blank_lines(self):
        text = "h\n\n" + "\n".join(str(i) for i in range(10))
        self.assertEqual(head(text), ["h", "0", "1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()
