"""Money amounts as a document prints them (F347, night 10b).

F347: the Branded Invoice printed an agent's ``"total": 311.0`` as "311.0", and
the seeded legacy "Invoice" put a "$" in front of every amount, whatever the
business's currency. A document never invents a currency: an amount prints as
the value was given ("€19.50", "1,500.00 EUR"), and a bare number (311, 311.0,
"311") prints with at least two decimals ("311.00"). Digits are never rounded
away: 0.125 stays 0.125.

Which values are amounts is read from the key they are sent under: its last
word (``total``, ``unit_price``, ``amount_due``). ``quantity``, ``tax_rate`` and
``total_hours`` are not amounts.

PRD-255 (FR-7): the kit can hold a currency (``locale_text``). A bare number then
prints with its symbol and two decimals ("£311.00"); an amount given with its own
currency still prints as given, and a kit without a currency prints none.

Pure: no IO.
"""
from __future__ import annotations

import re
from typing import Any, Iterator, List

from .locale_text import currency_prefix

# The last word of a key that holds money: "unit_price", "line_total", "amount_due".
AMOUNT_WORDS = frozenset({
    "amount", "balance", "cost", "deposit", "discount", "due", "fee", "fees",
    "price", "revenue", "shipping", "subtotal", "tax", "total", "vat",
})
KEY_WORD_SEPARATORS = re.compile(r"[\s_.-]+")
MIN_DECIMALS = 2
# A bare number: digits, an optional sign and decimals; no grouping, no currency, no exponent.
_BARE_NUMBER = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")


def is_amount_key(key: Any) -> bool:
    """Whether a value sent under ``key`` is money, by the key's last word."""
    words = [word for word in KEY_WORD_SEPARATORS.split(str(key).strip().lower()) if word]
    return bool(words) and words[-1] in AMOUNT_WORDS


def amount_text(value: Any, currency: str = "") -> str:
    """An amount as printed: a bare number with at least two decimals (after the symbol of
    ``currency``, an ISO 4217 code, when there is one), anything else as given."""
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return str(value)
    text = value.strip() if isinstance(value, str) else str(value)
    if not _BARE_NUMBER.match(text):
        return str(value)
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    whole, _, decimals = digits.partition(".")
    return f"{sign}{currency_prefix(currency)}{whole}.{decimals.ljust(MIN_DECIMALS, '0')}"


def is_bare_amount(value: Any) -> bool:
    """Whether ``value`` is a bare number (311, 311.0, "269.00"): one that would print with no currency
    sign unless the kit has a currency. Pure."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    return bool(_BARE_NUMBER.match(value.strip() if isinstance(value, str) else str(value)))


def _bare_amounts(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if is_amount_key(key) and is_bare_amount(item):
                yield str(key)
            yield from _bare_amounts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _bare_amounts(item)


def bare_amount_keys(data: Any) -> List[str]:
    """The amount keys in ``data`` (rows of a table too) sent as bare numbers, each once, in order. Pure."""
    return list(dict.fromkeys(_bare_amounts(data)))


def field_text(key: Any, value: Any, currency: str = "") -> str:
    """A value as printed under ``key``: an amount key's bare number with two decimals (in
    ``currency`` when the kit has one), else ``str``."""
    if value is None:
        return ""
    return amount_text(value, currency) if is_amount_key(key) else str(value)


__all__ = ["AMOUNT_WORDS", "amount_text", "bare_amount_keys", "field_text", "is_amount_key", "is_bare_amount"]
