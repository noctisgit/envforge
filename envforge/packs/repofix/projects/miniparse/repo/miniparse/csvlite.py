"""Minimal CSV: separated fields, where double-quoted fields may contain the separator."""

from __future__ import annotations


def split_row(line: str, sep: str = ",") -> list[str]:
    """Split one CSV line into fields. Inside quotes, "" stands for a literal quote."""
    fields: list[str] = []
    current: list[str] = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quotes:
            if ch == '"' and i + 1 < len(line) and line[i + 1] == '"':
                current.append('"')
                i += 1
            elif ch == '"':
                in_quotes = False
            else:
                current.append(ch)
        elif ch == '"':
            in_quotes = True
        elif ch == sep:
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    fields.append("".join(current))
    return fields


def read_records(text: str, sep: str = ",") -> list[dict[str, str]]:
    """Parse CSV text whose first line is a header into one dict per row.

    Blank lines are skipped. A row with the wrong number of fields is an error.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = split_row(lines[0], sep)
    records = []
    for line in lines[1:]:
        values = split_row(line, sep)
        if len(values) != len(header):
            raise ValueError(f"expected {len(header)} fields, got {len(values)}: {line!r}")
        records.append(dict(zip(header, values, strict=False)))
    return records


def column(records: list[dict[str, str]], name: str) -> list[str]:
    """Every value of one column, in row order."""
    return [record[name] for record in records]


def head(text: str, n: int = 5) -> list[str]:
    """The header line plus the first `n` data lines (so n + 1 lines), for previews.

    Not used by read_records; the CLI's preview command imports it.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[: n + 1]
