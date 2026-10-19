"""Pricing and fulfilment for customer orders."""

from __future__ import annotations

from dataclasses import dataclass, field

from inventory.stock import Inventory, OutOfStock


@dataclass
class Line:
    """One order line: `qty` units of `sku` at `unit_price` each."""

    sku: str
    qty: int
    unit_price: float


@dataclass
class Order:
    lines: list[Line] = field(default_factory=list)


def line_total(line: Line, discount: float = 0.0) -> float:
    """Price of one line after a fractional discount (0.25 = 25% off), rounded to cents."""
    return round(line.unit_price * line.qty * (1 - discount), 2)


def order_total(order: Order, tax_rate: float = 0.08, discount: float = 0.0) -> float:
    """Sum of the line totals plus sales tax, rounded to cents."""
    subtotal = sum(line_total(line, discount) for line in order.lines)
    return round(subtotal * (1 + tax_rate), 2)


def fulfil(order: Order, inventory: Inventory) -> None:
    """Ship every line of `order`, or nothing at all if any line is short of stock."""
    for line in order.lines:
        if inventory.count(line.sku) < line.qty:
            raise OutOfStock(f"{line.sku}: cannot ship {line.qty}")
    for line in order.lines:
        inventory.remove(line.sku, line.qty)


def pack_boxes(skus: list[str], per_box: int) -> list[list[str]]:
    """Split a pick list into boxes of at most `per_box` items, keeping the order."""
    if per_box < 1:
        raise ValueError("per_box must be >= 1")
    return [skus[i : i + per_box] for i in range(0, len(skus), per_box)]


def legacy_line_total(line: Line, markup: float = 0.0) -> float:
    """Old wholesale pricing: unit price plus a fractional *markup*, rounded to cents.

    Deprecated and unused here, but the wholesale portal still imports it.
    """
    return round(line.unit_price * line.qty * (1 + markup), 2)
