import unittest

from textstats.stats import lexical_diversity, mean_word_length, median, top_words, word_counts


class TestCounts(unittest.TestCase):
    def test_word_counts(self):
        self.assertEqual(word_counts("a b a")["a"], 2)

    def test_top_words_most_frequent_first(self):
        self.assertEqual(top_words("b a b c b a", k=2), [("b", 3), ("a", 2)])


class TestMeanWordLength(unittest.TestCase):
    def test_average(self):
        self.assertEqual(mean_word_length("ab abcd"), 3.0)

    def test_empty_text(self):
        self.assertEqual(mean_word_length(""), 0.0)


class TestLexicalDiversity(unittest.TestCase):
    def test_ratio(self):
        self.assertEqual(lexical_diversity("a a b b"), 0.5)


class TestMedian(unittest.TestCase):
    def test_odd_count(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_even_count(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            median([])


if __name__ == "__main__":
    unittest.main()
