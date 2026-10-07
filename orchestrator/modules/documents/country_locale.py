"""A country's currency and date style, for a kit that names its country (Gerard, 7 Oct).

The kit's ``country`` (ISO 3166-1 alpha-2, on the Brand kit page's Locale card) fills
what the kit leaves empty: an empty ``currency`` is the country's, and an empty
``date_style`` is the country's (GB: GBP and "5 October 2026"; US: USD and "October 5,
2026"; IE and the other euro countries: EUR). A currency or date style the kit sets
wins. So a kit with a country prints its amounts in that currency, and Auto does not
ask which (F367): ``locale_text.currency_of`` and ``date_style_of`` read through here.

A small table, no dependency: a country it does not hold is refused on save, and its
owner sets the currency and the date style directly. The Brand kit page mirrors it
(``frontend/components/deliverables/brand/country-locale.ts``).

Pure: no IO.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .brand_system import DATE_STYLE_DAY_FIRST, DATE_STYLE_MONTH_FIRST

COUNTRY_FIELD = "country"
# Empty: no country, so nothing is filled from one.
DEFAULT_COUNTRY = ""
COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")
EURO = "EUR"
# The euro area in 2026 (Bulgaria joined on 1 January 2026).
EURO_AREA = (
    "AT", "BE", "BG", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT", "SI", "SK",
)
# Country → (ISO 4217 currency, date style). Month first only where a long date is
# written that way (US, and English Canada).
COUNTRY_LOCALES: Dict[str, Tuple[str, str]] = {
    "GB": ("GBP", DATE_STYLE_DAY_FIRST),
    "US": ("USD", DATE_STYLE_MONTH_FIRST),
    "CA": ("CAD", DATE_STYLE_MONTH_FIRST),
    "AU": ("AUD", DATE_STYLE_DAY_FIRST),
    "NZ": ("NZD", DATE_STYLE_DAY_FIRST),
    **{code: (EURO, DATE_STYLE_DAY_FIRST) for code in EURO_AREA},
    "CZ": ("CZK", DATE_STYLE_DAY_FIRST),
    "DK": ("DKK", DATE_STYLE_DAY_FIRST),
    "HU": ("HUF", DATE_STYLE_DAY_FIRST),
    "PL": ("PLN", DATE_STYLE_DAY_FIRST),
    "RO": ("RON", DATE_STYLE_DAY_FIRST),
    "SE": ("SEK", DATE_STYLE_DAY_FIRST),
    "CH": ("CHF", DATE_STYLE_DAY_FIRST),
    "NO": ("NOK", DATE_STYLE_DAY_FIRST),
    "IS": ("ISK", DATE_STYLE_DAY_FIRST),
    "AE": ("AED", DATE_STYLE_DAY_FIRST),
    "BR": ("BRL", DATE_STYLE_DAY_FIRST),
    "CN": ("CNY", DATE_STYLE_DAY_FIRST),
    "HK": ("HKD", DATE_STYLE_DAY_FIRST),
    "IN": ("INR", DATE_STYLE_DAY_FIRST),
    "JP": ("JPY", DATE_STYLE_DAY_FIRST),
    "KR": ("KRW", DATE_STYLE_DAY_FIRST),
    "MX": ("MXN", DATE_STYLE_DAY_FIRST),
    "SG": ("SGD", DATE_STYLE_DAY_FIRST),
    "ZA": ("ZAR", DATE_STYLE_DAY_FIRST),
}


def country_code(value: str) -> str:
    """``value`` as a country the table holds (two letters, upper case), or empty for none."""
    code = value.strip().upper()
    if code and code not in COUNTRY_LOCALES:
        raise ValueError(
            f"country must be a two-letter ISO 3166-1 code such as GB, IE or US, one of "
            f"{', '.join(sorted(COUNTRY_LOCALES))} (got {value.strip()!r}); for another country, "
            "leave it empty and set the currency and the date style"
        )
    return code


def country_of(kit: Optional[Mapping[str, Any]]) -> str:
    """The kit's country when the table holds it; empty otherwise (read leniently: a render never fails on it)."""
    value = kit.get(COUNTRY_FIELD) if isinstance(kit, Mapping) else None
    code = value.strip().upper() if isinstance(value, str) else ""
    return code if COUNTRY_CODE.match(code) and code in COUNTRY_LOCALES else ""


def country_currency(kit: Optional[Mapping[str, Any]]) -> str:
    """The currency of the kit's country; empty without one."""
    country = country_of(kit)
    return COUNTRY_LOCALES[country][0] if country else ""


def country_date_style(kit: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The date style of the kit's country; ``None`` without one (a date then prints day first)."""
    country = country_of(kit)
    return COUNTRY_LOCALES[country][1] if country else None


__all__ = [
    "COUNTRY_FIELD",
    "COUNTRY_LOCALES",
    "DEFAULT_COUNTRY",
    "EURO_AREA",
    "country_code",
    "country_currency",
    "country_date_style",
    "country_of",
]
