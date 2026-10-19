"""Half-open numeric intervals [start, end): the start is included, the end is not."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Interval:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end {self.end} is before start {self.start}")

    @property
    def length(self) -> float:
        return self.end - self.start

    def contains(self, x: float) -> bool:
        """True if x lies in [start, end); the end point itself is outside."""
        return self.start <= x < self.end


def overlaps(a: Interval, b: Interval) -> bool:
    """True if two half-open intervals share at least one point. [1, 3) and [3, 5) do not."""
    return a.start < b.end and b.start < a.end


def merge(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping *or touching* intervals; the result is sorted by start."""
    merged: list[Interval] = []
    for iv in sorted(intervals):
        if merged and iv.start <= merged[-1].end:
            last = merged.pop()
            merged.append(Interval(last.start, max(last.end, iv.end)))
        else:
            merged.append(iv)
    return merged


def total_covered(intervals: list[Interval]) -> float:
    """Length of the union of `intervals`: overlapping stretches are counted once."""
    return sum(iv.length for iv in merge(intervals))


def overlaps_closed(a: Interval, b: Interval) -> bool:
    """Overlap test for *closed* intervals [start, end], where touching endpoints count.

    Unused in this package; the calendar UI calls it for all-day events.
    """
    return a.start <= b.end and b.start <= a.end
