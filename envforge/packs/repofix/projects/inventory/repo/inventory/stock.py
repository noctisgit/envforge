"""Units on hand per SKU."""

from __future__ import annotations


class OutOfStock(Exception):
    """Raised when shipping more units of a SKU than are on hand."""


class Inventory:
    """Units on hand per SKU. SKUs are case-insensitive and stored upper-case."""

    def __init__(self) -> None:
        self._units: dict[str, int] = {}

    def add(self, sku: str, qty: int) -> None:
        """Receive `qty` units of `sku`."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        key = sku.upper()
        self._units[key] = self._units.get(key, 0) + qty

    def remove(self, sku: str, qty: int) -> None:
        """Ship `qty` units of `sku`. Raises OutOfStock if there are not enough."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        key = sku.upper()
        on_hand = self._units.get(key, 0)
        if on_hand < qty:
            raise OutOfStock(f"{key}: have {on_hand}, need {qty}")
        self._units[key] = on_hand - qty

    def count(self, sku: str) -> int:
        """Units on hand for `sku` (0 if it has never been stocked)."""
        return self._units.get(sku.upper(), 0)

    def low_stock(self, threshold: int = 5) -> list[str]:
        """SKUs with `threshold` units or fewer, sorted."""
        return sorted(sku for sku, units in self._units.items() if units <= threshold)


def normalize_legacy_sku(code: str) -> str:
    """Convert an old-style SKU like "abc-0042" to the current form "ABC0042".

    Nothing in this repo calls it any more; the nightly import job still does.
    """
    return code.replace("-", "").upper()
