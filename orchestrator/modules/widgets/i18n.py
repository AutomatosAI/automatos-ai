"""
Widget i18n resolver (PRD-008-A Phase 11).

Tiny translation layer for user-facing strings the orchestrator
synthesises (SLA phrases, default greetings, default canned fallbacks).

v1 ships en-GB. The pattern is in place for future locales —
``packages/i18n/<locale>.json`` files dropped in alongside; resolver
falls back to en-GB on missing keys / unknown locales.

Why an in-memory dict instead of a JSON file
--------------------------------------------
Single-locale v1 doesn't justify the import dance + IO. When we add
the second locale, swap to JSON files keyed by locale and keep the
``t()`` API unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Locale code merchants set on ``site.settings.locale``. Defaults to
# en-GB if unset. Add new locales here as we ship them.
DEFAULT_LOCALE = "en-GB"
SUPPORTED_LOCALES: frozenset[str] = frozenset({"en-GB"})


# ----------------------------------------------------------------------------
# String table — keep keys stable; values are templates with {placeholders}
# ----------------------------------------------------------------------------

_STRINGS: dict[str, dict[str, str]] = {
    "en-GB": {
        # Callback (PRD-008-A Feature B)
        "callback.sla.in_hours.aim_to_call": (
            "We'll aim to call you within {sla_hours} working hours{product_clause}."
        ),
        "callback.sla.in_hours.will_call": (
            "We'll call you within {sla_hours} working hours{product_clause}."
        ),
        "callback.sla.outside_hours.aim_to_call": (
            "We're closed right now — we'll aim to call you "
            "first thing the next working day{product_clause}."
        ),
        "callback.sla.outside_hours.will_call": (
            "We're closed right now — we'll call you "
            "first thing the next working day{product_clause}."
        ),
        "callback.sla.product_clause": " about the {product}",
        "callback.sla.fallback": "Thanks — we'll be in touch.",

        # Cart-idle defaults (PRD-008-A Feature C1)
        "cart_idle.default_greeting": "Any questions before you check out?",

        # Proactive defaults (PRD-007 — re-homed here so all merchant-
        # facing text lives in one place going forward)
        "proactive.default_canned": "Need a hand finding the right product?",
    },
}


def normalise_locale(locale: Optional[str]) -> str:
    """Resolve a Site-supplied locale to a supported one.

    Unknown locales fall back to DEFAULT_LOCALE — no errors, no surprises.
    """
    if not locale:
        return DEFAULT_LOCALE
    if locale in SUPPORTED_LOCALES:
        return locale
    # Try the language portion ('en-US' → 'en' would match 'en-GB' as
    # nearest English neighbour). Soft match keeps the fallback graceful.
    base = locale.split("-")[0].lower()
    for supported in SUPPORTED_LOCALES:
        if supported.lower().startswith(base + "-"):
            return supported
    logger.debug("i18n: unsupported locale %r, falling back to %s", locale, DEFAULT_LOCALE)
    return DEFAULT_LOCALE


def t(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Resolve + format a translation key.

    Missing keys return the key itself (visible-broken fallback so the
    caller and reviewer notice). Missing template placeholders return
    the unformatted template — never raises.
    """
    resolved = normalise_locale(locale)
    table = _STRINGS.get(resolved) or _STRINGS[DEFAULT_LOCALE]
    template = table.get(key)
    if template is None:
        # Try fallback locale before giving up
        if resolved != DEFAULT_LOCALE:
            template = _STRINGS[DEFAULT_LOCALE].get(key)
        if template is None:
            logger.warning("i18n: missing translation key %r in any locale", key)
            return key

    try:
        return template.format(**kwargs)
    except (KeyError, IndexError) as exc:
        logger.warning("i18n: template %r missing placeholder: %s", key, exc)
        return template
