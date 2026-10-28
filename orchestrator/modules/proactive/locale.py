"""PRD-268 K13 — the language a request's server copy is written in: English or Portuguese (Brazil).

The request's ``Accept-Language`` chooses (RFC 9110 §12.5.4): of the ranges it lists, the
supported one with the highest weight wins, and of equal weights the first. ``pt-BR`` and a
bare ``pt`` are Portuguese (Brazil); ``en`` and any ``en-*`` are English. Another variant of
Portuguese (``pt-PT``) is not Brazil's, so it is passed over like any language without a
table. No header, nothing supported, or a header that cannot be read is English. W1 stores
no preference: the header is the only input.
"""
from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["DEFAULT_LOCALE", "LOCALES", "request_locale"]

ENGLISH = "en"
PORTUGUESE_BRAZIL = "pt-BR"
LOCALES = (ENGLISH, PORTUGUESE_BRAZIL)
DEFAULT_LOCALE = ENGLISH
# A language tag, lower-cased, to the locale whose table it reads.
_EXACT = {"en": ENGLISH, "pt": PORTUGUESE_BRAZIL, "pt-br": PORTUGUESE_BRAZIL}
_ENGLISH_PREFIX = "en-"
# A header is read this far and no further: a client lists a handful of languages.
MAX_RANGES = 20
MAX_HEADER_CHARS = 512
_WEIGHT = "q="
FULL_WEIGHT = 1.0


def _weight(params: str) -> Optional[float]:
    """The ``q`` of a range's parameters: 1 when absent, None when unreadable or out of 0..1."""
    for param in params.split(";"):
        name, _, value = param.strip().partition("=")
        if f"{name.strip().lower()}=" != _WEIGHT:
            continue
        try:
            weight = float(value.strip())
        except ValueError:
            return None
        return weight if 0.0 <= weight <= FULL_WEIGHT else None
    return FULL_WEIGHT


def _range(entry: str) -> Tuple[Optional[str], float]:
    """One ``tag;q=w`` entry: the locale it names (None when unsupported) and its weight."""
    tag, _, params = entry.partition(";")
    tag = tag.strip().lower()
    weight = _weight(params)
    if weight is None:
        return None, 0.0
    locale = _EXACT.get(tag) or (ENGLISH if tag.startswith(_ENGLISH_PREFIX) else None)
    return locale, weight


def request_locale(header: Optional[str]) -> str:
    """The locale a request's ``Accept-Language`` asks for: ``en`` or ``pt-BR``, ``en`` by default."""
    if not isinstance(header, str):
        return DEFAULT_LOCALE
    best, best_weight = DEFAULT_LOCALE, 0.0
    for entry in header[:MAX_HEADER_CHARS].split(",")[:MAX_RANGES]:
        locale, weight = _range(entry)
        if locale is not None and weight > best_weight:
            best, best_weight = locale, weight
    return best
