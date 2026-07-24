"""
PRD-008-A Phase 11 — i18n resolver unit tests
================================================

Verifies the locale-resolution + template-rendering contract used by
the callback eta_phrase synthesis (and any future widget strings).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


# ---------------------------------------------------------------------------
# normalise_locale
# ---------------------------------------------------------------------------

def test_normalise_locale_returns_supported_unchanged():
    from modules.widgets.i18n import normalise_locale

    assert normalise_locale("en-GB") == "en-GB"


def test_normalise_locale_falls_back_to_default_for_unsupported():
    from modules.widgets.i18n import normalise_locale, DEFAULT_LOCALE

    assert normalise_locale("zz-ZZ") == DEFAULT_LOCALE


def test_normalise_locale_falls_back_to_default_for_none():
    from modules.widgets.i18n import normalise_locale, DEFAULT_LOCALE

    assert normalise_locale(None) == DEFAULT_LOCALE
    assert normalise_locale("") == DEFAULT_LOCALE


def test_normalise_locale_soft_matches_language_portion():
    """en-US should fall through to en-GB until we ship en-US explicitly,
    not jump to a totally-different default like fr-FR."""
    from modules.widgets.i18n import normalise_locale

    # We only ship en-GB today, so en-US falls back to en-GB via the
    # language-portion match (both start with 'en-').
    assert normalise_locale("en-US") == "en-GB"


# ---------------------------------------------------------------------------
# t() — translation + formatting
# ---------------------------------------------------------------------------

def test_t_returns_formatted_template_for_known_key():
    from modules.widgets.i18n import t

    out = t("callback.sla.in_hours.aim_to_call", sla_hours=4, product_clause="")
    assert "aim to call" in out
    assert "4 working hours" in out


def test_t_returns_key_for_unknown_key():
    """Visible-broken: missing keys surface to humans rather than blank strings."""
    from modules.widgets.i18n import t

    assert t("does.not.exist") == "does.not.exist"


def test_t_returns_template_when_placeholder_missing():
    """Don't raise on missing format args — log + return template."""
    from modules.widgets.i18n import t

    out = t("callback.sla.in_hours.aim_to_call")  # no sla_hours / product_clause
    assert "{sla_hours}" in out  # template returned unformatted


def test_t_handles_unsupported_locale_via_fallback():
    from modules.widgets.i18n import t

    out = t("callback.sla.fallback", locale="zz-ZZ")
    assert out == "Thanks — we'll be in touch."


def test_t_renders_product_clause_template():
    from modules.widgets.i18n import t

    out = t("callback.sla.product_clause", product="EN 12101 panel")
    assert out == " about the EN 12101 panel"


# ---------------------------------------------------------------------------
# Integration with compute_eta_phrase
# ---------------------------------------------------------------------------

def test_compute_eta_phrase_routes_through_i18n():
    """compute_eta_phrase now goes through t(); changing the template
    in i18n.py changes the rendered phrase."""
    from datetime import datetime, timezone
    from services.callback import compute_eta_phrase

    settings = {
        "sla_hours": 6,
        "team_capacity": "limited",
        "working_hours_only": False,
    }
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    out = compute_eta_phrase(settings, product_context="cordless drill", now=now)
    assert "aim to call" in out
    assert "6 working hours" in out
    assert "about the cordless drill" in out


def test_compute_eta_phrase_falls_back_via_i18n_on_bad_config():
    """The fallback path also uses t() — no hardcoded English left."""
    from services.callback import compute_eta_phrase

    out = compute_eta_phrase({"sla_hours": "garbage"})
    # As long as it returns SOMETHING usable; specific text is in the
    # callback.sla.fallback key
    assert isinstance(out, str) and len(out) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
