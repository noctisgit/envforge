import unittest

from textstats.tokenize import ngrams, sentences, words


class TestWords(unittest.TestCase):
    def test_lowercases_by_default(self):
        self.assertEqual(words("The Cat sat"), ["the", "cat", "sat"])

    def test_keeps_apostrophes(self):
        self.assertEqual(words("don't stop"), ["don't", "stop"])


class TestSentences(unittest.TestCase):
    def test_splits_on_terminators(self):
        self.assertEqual(sentences("Hi there. How are you?"), ["Hi there", "How are you"])


class TestNgrams(unittest.TestCase):
    def test_bigrams(self):
        self.assertEqual(ngrams(["a", "b", "c"], 2), [("a", "b"), ("b", "c")])

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            ngrams(["a"], 0)


if __name__ == "__main__":
    unittest.main()
