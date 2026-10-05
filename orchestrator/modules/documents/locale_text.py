"""A kit's locale as a document prints it: its currency and its date style (PRD-255 FR-7, FR-8).

The kit holds ``currency`` (an ISO 4217 code, or empty for none) and ``date_style``
(``brand_system.DATE_STYLES``). Every renderer reads them through here:

* :func:`currency_of`: the kit's code, read leniently (anything that is not a
  code is no currency, so a render never invents one);
* :func:`currency_prefix`: what goes before an amount: the currency's symbol
  ("£"), or its code and a space ("SEK ") for one without a symbol here;
* :func:`long_date`: a date in the kit's style, "5 October 2026" or "October 5, 2026".

Pure: no IO.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional

from .brand_system import DATE_STYLE_MONTH_FIRST, currency_code

CURRENCY_FIELD = "currency"
DATE_STYLE_FIELD = "date_style"
# The symbols printed before an amount; any other code prints as itself and a space.
CURRENCY_SYMBOLS = {
    "AUD": "A$", "CAD": "C$", "CNY": "¥", "EUR": "€", "GBP": "£", "HKD": "HK$", "INR": "₹", "JPY": "¥",
    "KRW": "₩", "NZD": "NZ$", "SGD": "S$", "USD": "$", "ZAR": "R",
}


def currency_of(kit: Optional[Mapping[str, Any]]) -> str:
    """The kit's ISO 4217 code, or empty when it has none (or holds something that is not a code)."""
    value = (kit or {}).get(CURRENCY_FIELD) if isinstance(kit, Mapping) else None
    if not isinstance(value, str):
        return ""
    try:
        return currency_code(value)
    except ValueError:
        return ""


def currency_prefix(code: str) -> str:
    """What prints before an amount in ``code``: its symbol, or the code and a space; empty for none."""
    if not code:
        return ""
    return CURRENCY_SYMBOLS.get(code, f"{code} ")


def long_date(day: date, style: Any = None) -> str:
    """``day`` in the kit's date style: "5 October 2026", or "October 5, 2026" when the style is month first.

    Built by hand (no ``%-d``, which is not portable)."""
    month = day.strftime("%B")
    if style == DATE_STYLE_MONTH_FIRST:
        return f"{month} {day.day}, {day.year}"
    return f"{day.day} {month} {day.year}"


def date_style_of(kit: Optional[Mapping[str, Any]]) -> Any:
    """The kit's date style as stored (``long_date`` reads anything unknown as the default)."""
    return (kit or {}).get(DATE_STYLE_FIELD) if isinstance(kit, Mapping) else None


__all__ = ["CURRENCY_SYMBOLS", "currency_of", "currency_prefix", "date_style_of", "long_date"]
