"""The Brand kit page's country table is the server's (Gerard, 7 Oct).

The Locale card shows a country's currency and date style before a save, from
``frontend/components/deliverables/brand/country-locale.ts``; documents print with
``modules/documents/country_locale.py``. The two tables are kept by hand, so this reads
the frontend file as text and pins that every country, its currency and its date style
are the same in both: a change to one without the other fails here.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

from modules.documents.country_locale import COUNTRY_LOCALES

FRONTEND_TABLE = (
    Path(__file__).resolve().parents[2] / "frontend" / "components" / "deliverables" / "brand" / "country-locale.ts"
)
STYLE_CONSTANT = re.compile(r"const (\w+): BrandDateStyle = '([^']*)'")
EURO_AREA = re.compile(r"const EURO_AREA = \[(.*?)\] as const", re.S)
EURO_ENTRY = re.compile(r"EURO_AREA\.map\(\(code\) => \[code, \{ currency: '([A-Z]{3})', dateStyle: (\w+) \}\]\)")
OTHER_BLOCK = re.compile(r"const OTHER_COUNTRIES: Record<string, CountryLocale> = \{(.*?)\n\}", re.S)
OTHER_ENTRY = re.compile(r"^\s*([A-Z]{2}): \{ currency: '([A-Z]{3})', dateStyle: (\w+) \},?\s*$", re.M)
CODE = re.compile(r"'([A-Z]{2})'")


def _one(pattern: re.Pattern, text: str) -> re.Match:
    match = pattern.search(text)
    assert match is not None, f"{FRONTEND_TABLE.name} no longer has {pattern.pattern!r}: update this test with it"
    return match


def _frontend_table() -> Dict[str, Tuple[str, str]]:
    """``country-locale.ts``'s table as ``code -> (currency, date style)``."""
    text = FRONTEND_TABLE.read_text(encoding="utf-8")
    styles = dict(STYLE_CONSTANT.findall(text))
    euro_currency, euro_style = _one(EURO_ENTRY, text).groups()
    table = {code: (euro_currency, styles[euro_style]) for code in CODE.findall(_one(EURO_AREA, text).group(1))}
    others = OTHER_ENTRY.findall(_one(OTHER_BLOCK, text).group(1))
    assert others, "no entries read from OTHER_COUNTRIES"
    for code, currency, style in others:
        assert code not in table, f"{code} is in the frontend table twice"
        table[code] = (currency, styles[style])
    return table


def test_the_frontend_country_table_is_the_servers():
    assert _frontend_table() == COUNTRY_LOCALES
