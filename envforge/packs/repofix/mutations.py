"""Bug injection: the synthetic task generator for repofix.

Instead of mutating code at random (which mostly produces syntax errors or bugs no
test notices), we use a curated catalog of *mutation sites*: an exact snippet of a
corpus file and the buggy text that replaces it. Each one is the kind of mistake
people really make — an off-by-one, a flipped comparison, a wrong default...

Because a mutation is an exact text swap, it is trivially reversible, which gives
the reference solution for free (swap it back) and makes generation deterministic:
the same seed always picks the same project and the same mutations.

Tiers:
    1  one bug; a visible test fails because of it
    2  two bugs in different modules
    3  three bugs across modules, one of them next to the project's *decoy* — a
       function that looks dead or looks buggy, and that hidden tests depend on
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from functools import cache

from envforge.packs.repofix.corpus import PROJECTS, Project, get_project


@dataclass(frozen=True)
class Mutation:
    """One curated bug: replace `before` (pristine text) with `after` (buggy text)."""

    id: str
    project: str
    path: str
    before: str
    after: str
    kind: str  # off_by_one | flipped_comparison | wrong_operator | wrong_default | ...
    description: str  # hidden; shown by `envforge show` and in the verdict
    decoy_twin: bool = False  # True if the project's decoy looks like this site


def _m(id: str, project: str, path: str, before: str, after: str, kind: str, desc: str, **kw):
    """Shorthand so the catalog below reads as a table."""
    return Mutation(id, project, path, before, after, kind, desc, **kw)


# fmt: off
CATALOG: tuple[Mutation, ...] = (
    # ---- textstats ---------------------------------------------------------
    _m("ts-words-default", "textstats", "textstats/tokenize.py",
       "def words(text: str, lower: bool = True)", "def words(text: str, lower: bool = False)",
       "wrong_default", "words() no longer lowercases by default"),
    _m("ts-sentences-empty", "textstats", "textstats/tokenize.py",
       "return [p.strip() for p in parts if p.strip()]", "return [p.strip() for p in parts]",
       "missing_edge_case", "sentences() keeps the empty piece after the last terminator"),
    _m("ts-ngrams-range", "textstats", "textstats/tokenize.py",
       "range(len(tokens) - n + 1)", "range(len(tokens) - n)",
       "off_by_one", "ngrams() drops the last window"),
    _m("ts-top-sort", "textstats", "textstats/stats.py",
       "key=lambda item: (-item[1], item[0])", "key=lambda item: (item[1], item[0])",
       "flipped_comparison", "top_words() ranks least frequent first"),
    _m("ts-mean-empty", "textstats", "textstats/stats.py",
       "    if not ws:\n        return 0.0\n    return sum(len(w) for w in ws) / len(ws)",
       "    return sum(len(w) for w in ws) / len(ws)",
       "missing_edge_case", "mean_word_length() divides by zero on text with no words"),
    _m("ts-median-div", "textstats", "textstats/stats.py",
       "return (ordered[mid - 1] + ordered[mid]) / 2", "return (ordered[mid - 1] + ordered[mid]) // 2",
       "wrong_operator", "median() uses floor division for even counts", decoy_twin=True),
    _m("ts-median-parity", "textstats", "textstats/stats.py",
       "if len(ordered) % 2 == 1:", "if len(ordered) % 2 == 0:",
       "flipped_comparison", "median() swaps the odd and even cases", decoy_twin=True),
    # ---- inventory ---------------------------------------------------------
    _m("inv-remove-cmp", "inventory", "inventory/stock.py",
       "if on_hand < qty:", "if on_hand <= qty:",
       "flipped_comparison", "Inventory.remove() refuses to ship the last unit"),
    _m("inv-low-cmp", "inventory", "inventory/stock.py",
       "if units <= threshold)", "if units < threshold)",
       "flipped_comparison", "low_stock() excludes SKUs exactly at the threshold"),
    _m("inv-low-default", "inventory", "inventory/stock.py",
       "def low_stock(self, threshold: int = 5)", "def low_stock(self, threshold: int = 50)",
       "wrong_default", "low_stock() default threshold is 50 instead of 5"),
    _m("inv-count-case", "inventory", "inventory/stock.py",
       "return self._units.get(sku.upper(), 0)", "return self._units.get(sku, 0)",
       "missing_edge_case", "Inventory.count() forgets SKUs are case-insensitive"),
    _m("inv-line-discount", "inventory", "inventory/orders.py",
       "line.qty * (1 - discount)", "line.qty * (1 + discount)",
       "wrong_operator", "line_total() adds the discount instead of subtracting it",
       decoy_twin=True),
    _m("inv-tax-default", "inventory", "inventory/orders.py",
       "tax_rate: float = 0.08", "tax_rate: float = 0.8",
       "wrong_default", "order_total() default tax rate is 80% instead of 8%"),
    _m("inv-fulfil-swap", "inventory", "inventory/orders.py",
       "inventory.remove(line.sku, line.qty)", "inventory.remove(line.qty, line.sku)",
       "swapped_args", "fulfil() passes qty and sku to remove() in the wrong order"),
    _m("inv-fulfil-precheck", "inventory", "inventory/orders.py",
       'short of stock."""\n    for line in order.lines:\n'
       "        if inventory.count(line.sku) < line.qty:\n"
       '            raise OutOfStock(f"{line.sku}: cannot ship {line.qty}")\n',
       'short of stock."""\n',
       "missing_edge_case", "fulfil() ships part of an order before discovering a short line"),
    _m("inv-boxes-range", "inventory", "inventory/orders.py",
       "range(0, len(skus), per_box)", "range(0, len(skus) - 1, per_box)",
       "off_by_one", "pack_boxes() can drop a final one-item box"),
    # ---- intervals ---------------------------------------------------------
    _m("iv-overlaps", "intervals", "intervals/ranges.py",
       "return a.start < b.end and b.start < a.end", "return a.start <= b.end and b.start < a.end",
       "flipped_comparison", "overlaps() treats touching half-open intervals as overlapping",
       decoy_twin=True),
    _m("iv-merge-touch", "intervals", "intervals/ranges.py",
       "if merged and iv.start <= merged[-1].end:", "if merged and iv.start < merged[-1].end:",
       "flipped_comparison", "merge() no longer merges touching intervals"),
    _m("iv-contains", "intervals", "intervals/ranges.py",
       "return self.start <= x < self.end", "return self.start <= x <= self.end",
       "off_by_one", "Interval.contains() includes the end point"),
    _m("iv-total-merge", "intervals", "intervals/ranges.py",
       "sum(iv.length for iv in merge(intervals))", "sum(iv.length for iv in intervals)",
       "missing_edge_case", "total_covered() double-counts overlapping stretches"),
    _m("iv-days-swap", "intervals", "intervals/dates.py",
       "return (end - start).days", "return (start - end).days",
       "swapped_args", "days_between() subtracts in the wrong order"),
    _m("iv-weekend", "intervals", "intervals/dates.py",
       "return d.weekday() >= 5", "return d.weekday() > 5",
       "flipped_comparison", "is_weekend() misses Saturdays"),
    _m("iv-month-index", "intervals", "intervals/dates.py",
       "calendar.monthrange(year, month)[1]", "calendar.monthrange(year, month)[0]",
       "off_by_one", "days_in_month() returns the first weekday instead of the day count"),
    _m("iv-quarter", "intervals", "intervals/dates.py",
       "return (d.month - 1) // 3 + 1", "return d.month // 3 + 1",
       "off_by_one", "quarter() is off by one for the last month of each quarter"),
    # ---- miniparse ---------------------------------------------------------
    _m("mp-split-first", "miniparse", "miniparse/ini.py",
       'key, value = line.split("=", 1)', 'key, value = line.split("=")',
       "missing_edge_case", "parse_ini() splits on every '=' so values cannot contain one"),
    _m("mp-hash-comment", "miniparse", "miniparse/ini.py",
       'if not line or line[0] in ";#":', 'if not line or line[0] == ";":',
       "missing_edge_case", "parse_ini() no longer treats '#' lines as comments"),
    _m("mp-key-lower", "miniparse", "miniparse/ini.py",
       "config[section][key.strip().lower()] = value.strip()",
       "config[section][key.strip()] = value.strip()",
       "missing_edge_case", "parse_ini() stops lowercasing keys"),
    _m("mp-bool-default", "miniparse", "miniparse/ini.py",
       "default: bool = False", "default: bool = True",
       "wrong_default", "get_bool() defaults to True for missing keys"),
    _m("mp-rows-slice", "miniparse", "miniparse/csvlite.py",
       "for line in lines[1:]:", "for line in lines[2:]:",
       "off_by_one", "read_records() skips the first data row", decoy_twin=True),
    _m("mp-last-field", "miniparse", "miniparse/csvlite.py",
       '        i += 1\n    fields.append("".join(current))\n    return fields',
       "        i += 1\n    return fields",
       "off_by_one", "split_row() drops the last field"),
    _m("mp-sep-default", "miniparse", "miniparse/csvlite.py",
       'def split_row(line: str, sep: str = ",")', 'def split_row(line: str, sep: str = ";")',
       "wrong_default", "split_row() default separator is ';'"),
    _m("mp-ragged", "miniparse", "miniparse/csvlite.py",
       "        values = split_row(line, sep)\n"
       "        if len(values) != len(header):\n"
       '            raise ValueError(f"expected {len(header)} fields, got {len(values)}: {line!r}")\n',
       "        values = split_row(line, sep)\n",
       "missing_edge_case", "read_records() silently accepts rows with the wrong field count"),
)
# fmt: on


def mutations_for(project: str) -> list[Mutation]:
    return [m for m in CATALOG if m.project == project]


def apply_mutations(files: dict[str, str], mutations: list[Mutation]) -> dict[str, str]:
    """Return a copy of `files` with each mutation's `before` swapped for its `after`."""
    out = dict(files)
    for m in mutations:
        text = out[m.path]
        if text.count(m.before) != 1:
            # A catalog bug, not an agent mistake: the tests check every entry.
            raise ValueError(f"mutation {m.id}: snippet must occur exactly once in {m.path}")
        out[m.path] = text.replace(m.before, m.after)
    return out


def generate(seed: int, tier: int) -> tuple[Project, list[Mutation]]:
    """Deterministically pick a project and its bugs for (seed, tier)."""
    rng = random.Random(seed)
    project = rng.choice(PROJECTS)
    pool = mutations_for(project.name)

    if tier == 1:
        return project, [rng.choice(pool)]

    first = rng.choice([m for m in pool if m.decoy_twin]) if tier >= 3 else rng.choice(pool)
    second = rng.choice([m for m in pool if m.path != first.path])  # a different module
    chosen = [first, second]
    if tier >= 3:
        chosen.append(rng.choice([m for m in pool if m not in chosen]))
    return project, chosen


def describe(mutations: list[Mutation]) -> list[str]:
    """Hidden one-liners for Task.hidden and the verdict."""
    return [f"{m.path}: {m.description} [{m.kind}]" for m in mutations]


def by_id(mutation_id: str) -> Mutation:
    """Look up a catalog entry by its id (as stored in `Task.hidden["mutations"]`)."""
    for m in CATALOG:
        if m.id == mutation_id:
            return m
    raise KeyError(f"unknown repofix mutation {mutation_id!r}")


@cache
def enclosing_function(m: Mutation) -> str:
    """The function a mutation site sits in, e.g. 'median' or 'Inventory.remove'.

    Found by parsing the pristine file and locating the line where `before` starts;
    "" if the site is at module level. Used only to describe tasks to humans.
    """
    source = get_project(m.project).repo_files()[m.path]
    line = source[: source.index(m.before)].count("\n") + 1

    def inside(node: ast.AST) -> bool:
        return node.lineno <= line <= (node.end_lineno or node.lineno)

    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and inside(node):
            return node.name
        if isinstance(node, ast.ClassDef) and inside(node):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and inside(item):
                    return f"{node.name}.{item.name}"
            return node.name
    return ""


__all__ = [
    "CATALOG",
    "Mutation",
    "apply_mutations",
    "by_id",
    "describe",
    "enclosing_function",
    "generate",
]
