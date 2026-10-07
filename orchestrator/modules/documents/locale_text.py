"""A kit's locale as a document prints it: its currency and its date style (PRD-255 FR-7, FR-8).

The kit holds ``currency`` (an ISO 4217 code, or empty for none) and ``date_style``
(``brand_system.DATE_STYLES``, or empty), and a ``country`` that fills either when it
is empty (``country_locale``). Every renderer reads them through here:

* :func:`currency_of`: the kit's code, else its country's, read leniently
  (anything that is not a code is no currency, so a render never invents one);
* :func:`date_style_of`: the kit's date style, else its country's;
* :func:`currency_prefix`: what goes before an amount: the currency's symbol
  ("£"), or its code and a space ("SEK ") for one without a symbol here;
* :func:`long_date`: a date in the kit's style, "5 October 2026" or "October 5, 2026".

Pure: no IO.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional

from .brand_system import DATE_STYLE_MONTH_FIRST, currency_code
from .country_locale import country_currency, country_date_style

CURRENCY_FIELD = "currency"
DATE_STYLE_FIELD = "date_style"
# The symbols printed before an amount; any other code prints as itself and a space.
CURRENCY_SYMBOLS = {
    "AUD": "A$", "CAD": "C$", "CNY": "¥", "EUR": "€", "GBP": "£", "HKD": "HK$", "INR": "₹", "JPY": "¥",
    "KRW": "₩", "NZD": "NZ$", "SGD": "S$", "USD": "$", "ZAR": "R",
}


def currency_of(kit: Optional[Mapping[str, Any]]) -> str:
    """The kit's ISO 4217 code, else its country's; empty when it has neither (a stored non-code is none)."""
    value = (kit or {}).get(CURRENCY_FIELD) if isinstance(kit, Mapping) else None
    try:
        code = currency_code(value) if isinstance(value, str) else ""
    except ValueError:
        code = ""
    return code or country_currency(kit)


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
    """The kit's date style as stored, else its country's (``long_date`` reads anything unknown as the default)."""
    style = kit.get(DATE_STYLE_FIELD) if isinstance(kit, Mapping) else None
    return style or country_date_style(kit)


__all__ = ["CURRENCY_SYMBOLS", "currency_of", "currency_prefix", "date_style_of", "long_date"]
