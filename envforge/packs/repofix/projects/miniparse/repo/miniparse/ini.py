"""A tiny INI reader: [sections], key = value lines, and ; or # comments."""

from __future__ import annotations

TRUE_WORDS = {"1", "true", "yes", "on"}
FALSE_WORDS = {"0", "false", "no", "off"}


class ParseError(ValueError):
    """Raised for a line that is not a section header, a key = value pair or a comment."""


def parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Parse INI text into {section: {key: value}}. Keys before any section go in "DEFAULT".

    Keys are lowercased; values keep their case. Only the first "=" separates key from
    value, so values may themselves contain "=".
    """
    config: dict[str, dict[str, str]] = {"DEFAULT": {}}
    section = "DEFAULT"
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            config.setdefault(section, {})
            continue
        if "=" not in line:
            raise ParseError(f"line {lineno}: expected 'key = value', got {raw!r}")
        key, value = line.split("=", 1)
        config[section][key.strip().lower()] = value.strip()
    return config


def get_bool(
    config: dict[str, dict[str, str]], section: str, key: str, default: bool = False
) -> bool:
    """Read a yes/no style value; returns `default` when the key is missing."""
    value = config.get(section, {}).get(key.lower())
    if value is None:
        return default
    word = value.lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ParseError(f"[{section}] {key}: {value!r} is not a boolean")
