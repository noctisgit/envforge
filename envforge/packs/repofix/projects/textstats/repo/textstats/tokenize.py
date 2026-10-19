"""Split text into words, sentences and n-grams."""

import re

_WORD = re.compile(r"[A-Za-z']+")
_SENTENCE_END = re.compile(r"[.!?]+")


def words(text: str, lower: bool = True) -> list[str]:
    """Return the words in `text`, lowercased unless lower=False.

    Apostrophes stay inside words ("don't"); everything else separates words.
    """
    found = _WORD.findall(text)
    if lower:
        return [w.lower() for w in found]
    return found


def sentences(text: str) -> list[str]:
    """Split `text` into sentences on runs of . ! or ? and drop empty pieces."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Consecutive n-token windows: ngrams(["a", "b", "c"], 2) -> [("a", "b"), ("b", "c")]."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
