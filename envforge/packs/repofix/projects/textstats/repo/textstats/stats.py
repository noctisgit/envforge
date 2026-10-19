"""Word-frequency statistics built on `textstats.tokenize`."""

from collections import Counter

from textstats.tokenize import words


def word_counts(text: str) -> Counter[str]:
    """How often each (lowercased) word occurs."""
    return Counter(words(text))


def top_words(text: str, k: int = 3) -> list[tuple[str, int]]:
    """The k most frequent words, most frequent first; ties broken alphabetically."""
    counts = word_counts(text)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]


def mean_word_length(text: str) -> float:
    """Average word length in characters; 0.0 for text with no words."""
    ws = words(text)
    if not ws:
        return 0.0
    return sum(len(w) for w in ws) / len(ws)


def lexical_diversity(text: str) -> float:
    """Distinct words / total words (the type-token ratio); 0.0 for text with no words."""
    ws = words(text)
    if not ws:
        return 0.0
    return len(set(ws)) / len(ws)


def median(values: list[float]) -> float:
    """Median of `values`: the mean of the two middle values when the count is even."""
    if not values:
        raise ValueError("median() of an empty list")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def median_low(values: list[float]) -> float:
    """The low median: always one of `values` (the smaller middle one when even).

    No callers left in this package; the v1 reporting API still imports it.
    """
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]
