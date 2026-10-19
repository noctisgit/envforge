"""textstats: small text-statistics helpers."""

from textstats.stats import lexical_diversity, mean_word_length, median, top_words, word_counts
from textstats.tokenize import ngrams, sentences, words

__all__ = [
    "lexical_diversity",
    "mean_word_length",
    "median",
    "ngrams",
    "sentences",
    "top_words",
    "word_counts",
    "words",
]
