"""Measured privacy review: a person checks a random sample of redacted conversations.

Automatic redaction is never perfect, so before an environment is built from real chat
logs a person reviews a sample and marks every conversation in which they can still
see personal data. From that we estimate the *leak rate* (the share of conversations
with something left) together with a 95% confidence interval, and the review passes
only if the interval's upper end is at or below the project's threshold:

    20 conversations, 0 flagged  ->  leak rate 0%, 95% CI 0%-16.1%   (not enough to pass 5%)
    80 conversations, 0 flagged  ->  leak rate 0%, 95% CI 0%-4.6%    (passes 5%)
    80 conversations, 2 flagged  ->  leak rate 2.5%, 95% CI 0.7%-8.7% (fails 5%)

Using the upper bound means a small, lucky sample can't pass: the fewer conversations
reviewed, the wider the interval. `sample_size_for` says how many to review so that a
clean sample passes.

The Wilson score interval is used instead of the textbook "p ± 1.96·sqrt(p(1-p)/n)",
which collapses to 0%-0% when nothing is flagged — exactly the case we care about.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

Z95 = 1.959964  # two-sided 95%
MIN_SAMPLE = 20
MAX_SAMPLE = 400


def wilson_interval(flagged: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion `flagged / n`, as (low, high)."""
    if n <= 0:
        return 0.0, 1.0
    p = flagged / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sample_size_for(threshold: float, z: float = Z95) -> int:
    """How many conversations to review so that a sample with nothing flagged passes.

    With 0 flagged the Wilson upper bound is z²/(n+z²); solving z²/(n+z²) <= t gives
    n >= z²(1-t)/t. A 10% margin leaves room for the bound to be met comfortably.
    """
    t = min(max(threshold, 0.001), 0.5)
    n = math.ceil(z * z * (1 - t) / t * 1.1)
    return max(MIN_SAMPLE, min(MAX_SAMPLE, n))


@dataclass
class ReviewResult:
    reviewed: int
    flagged: int
    population: int  # unique conversations in the corpus when the sample was drawn
    leak_rate: float
    ci_low: float
    ci_high: float
    threshold: float
    passed: bool
    exact: bool  # the whole corpus was reviewed: the rate is a count, not an estimate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(flagged: int, reviewed: int, *, threshold: float, population: int) -> ReviewResult:
    """Leak rate, interval and pass/fail for one completed review."""
    if reviewed <= 0:
        raise ValueError("nothing was reviewed")
    exact = population > 0 and reviewed >= population
    rate = flagged / reviewed
    lo, hi = (rate, rate) if exact else wilson_interval(flagged, reviewed)
    return ReviewResult(
        reviewed=reviewed,
        flagged=flagged,
        population=population,
        leak_rate=round(rate, 4),
        ci_low=round(lo, 4),
        ci_high=round(hi, 4),
        threshold=threshold,
        passed=hi <= threshold,
        exact=exact,
    )
