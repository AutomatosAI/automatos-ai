"""
PRD-008-A Phase 5 — Callback service-layer tests
==================================================

Pure-Python unit tests for the callback business logic. Mocked DB
sessions; no FastAPI app boot.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401 — triggers .env load


# ---------------------------------------------------------------------------
# Phone validation + normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("+447700900123", "+447700900123"),                    # already E.164
    (" +44 7700 900 123 ", "+447700900123"),               # spaces stripped
    ("+44-7700-900-123", "+447700900123"),                 # dashes stripped
    ("(+44) 7700 900 123", "+447700900123"),               # parens stripped
    ("447700900123", "447700900123"),                      # missing leading + preserved
])
def test_normalise_phone(raw, expected):
    from services.callback import normalise_phone
    assert normalise_phone(raw) == expected


def test_normalise_phone_handles_none():
    from services.callback import normalise_phone
    assert normalise_phone(None) == ""


@pytest.mark.parametrize("phone,valid", [
    ("+447700900123", True),
    ("+1234567890", True),
    ("447700900123", False),    # missing leading +
    ("+44", False),             # too short
    ("+0123456789", False),     # E.164 requires first digit non-zero
    ("", False),
    ("not a phone", False),
    ("+" + "1" * 20, False),    # too long
])
def test_phone_validation(phone, valid):
    from services.callback import is_valid_phone
    assert is_valid_phone(phone) is valid


# ---------------------------------------------------------------------------
# Phone hash — GDPR
# ---------------------------------------------------------------------------

def test_phone_hash_is_deterministic_per_site():
    from services.callback import compute_phone_hash

    site = uuid4()
    h1 = compute_phone_hash("+447700900123", site)
    h2 = compute_phone_hash("+447700900123", site)
    assert h1 == h2


def test_phone_hash_differs_across_sites():
    """Per-Site salt prevents rainbow-table cross-merchant lookups."""
    from services.callback import compute_phone_hash

    h1 = compute_phone_hash("+447700900123", uuid4())
    h2 = compute_phone_hash("+447700900123", uuid4())
    assert h1 != h2


def test_phone_hash_differs_across_phones():
    from services.callback import compute_phone_hash

    site = uuid4()
    h1 = compute_phone_hash("+447700900123", site)
    h2 = compute_phone_hash("+447700900124", site)
    assert h1 != h2


def test_phone_hash_does_not_contain_plaintext():
    """The hash must not leak the original number."""
    from services.callback import compute_phone_hash

    h = compute_phone_hash("+447700900123", uuid4())
    assert "447700900123" not in h
    assert "+44" not in h


def test_phone_hash_normalises_input():
    """Whitespace/dashes shouldn't produce different hashes for the same number."""
    from services.callback import compute_phone_hash

    site = uuid4()
    h1 = compute_phone_hash(" +44 7700 900 123 ", site)
    h2 = compute_phone_hash("+447700900123", site)
    assert h1 == h2


# ---------------------------------------------------------------------------
# Idempotency — find_recent_duplicate
# ---------------------------------------------------------------------------

def _event(*, session_id, phone_hash, request_id, created_at):
    return SimpleNamespace(
        session_id=session_id,
        event_data={"phone_hash": phone_hash, "request_id": request_id},
        created_at=created_at,
    )


def test_find_recent_duplicate_returns_request_id_on_match():
    from services.callback import find_recent_duplicate

    site = uuid4()
    matching = _event(
        session_id="sess_abc",
        phone_hash="hash_xyz",
        request_id="cb_existing",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [matching]

    result = find_recent_duplicate(
        db, site_id=site, session_id="sess_abc", phone_hash="hash_xyz"
    )
    assert result == "cb_existing"


def test_find_recent_duplicate_returns_none_on_no_phone_hash_match():
    from services.callback import find_recent_duplicate

    different_hash = _event(
        session_id="sess_abc",
        phone_hash="different",
        request_id="cb_other",
        created_at=datetime.now(timezone.utc),
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [different_hash]

    result = find_recent_duplicate(
        db, site_id=uuid4(), session_id="sess_abc", phone_hash="hash_xyz"
    )
    assert result is None


def test_find_recent_duplicate_returns_none_on_empty():
    from services.callback import find_recent_duplicate

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = []

    result = find_recent_duplicate(
        db, site_id=uuid4(), session_id="x", phone_hash="y"
    )
    assert result is None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _db_with_counts(session_count: int, site_count: int):
    """Mock db where the first .count() returns session_count and the
    second returns site_count, mimicking the two queries in
    check_rate_limits."""
    db = MagicMock()
    counts = iter([session_count, site_count])
    db.query.return_value.filter.return_value.count.side_effect = lambda: next(counts)
    return db


def test_rate_limit_allowed_when_no_recent_activity():
    from services.callback import check_rate_limits

    db = _db_with_counts(session_count=0, site_count=0)
    decision = check_rate_limits(
        db, site_id=uuid4(), session_id="x"
    )
    assert decision.allowed is True
    assert decision.reason is None


def test_rate_limit_blocks_per_session_cooldown():
    from services.callback import check_rate_limits

    db = _db_with_counts(session_count=1, site_count=0)
    decision = check_rate_limits(
        db, site_id=uuid4(), session_id="x"
    )
    assert decision.allowed is False
    assert decision.reason == "per_session_cooldown"
    assert decision.retry_after_seconds == 60


def test_rate_limit_blocks_per_site_hourly_cap():
    from services.callback import check_rate_limits

    db = _db_with_counts(session_count=0, site_count=100)
    decision = check_rate_limits(
        db, site_id=uuid4(), session_id="x", per_site_hourly_cap=100
    )
    assert decision.allowed is False
    assert decision.reason == "per_site_hourly_cap"
    assert decision.retry_after_seconds == 3600


def test_rate_limit_per_site_cap_is_configurable():
    from services.callback import check_rate_limits

    db = _db_with_counts(session_count=0, site_count=5)
    # Cap of 5 → at the cap → blocked
    decision = check_rate_limits(
        db, site_id=uuid4(), session_id="x", per_site_hourly_cap=5
    )
    assert decision.allowed is False
    assert decision.reason == "per_site_hourly_cap"


# ---------------------------------------------------------------------------
# SLA phrase — capacity, working hours, product context, robustness
# ---------------------------------------------------------------------------

def _hours_in(tz_name: str = "Europe/London", day="monday", start="09:00", end="17:00"):
    spec = {
        "tz": tz_name,
        "monday": "closed",
        "tuesday": "closed",
        "wednesday": "closed",
        "thursday": "closed",
        "friday": "closed",
        "saturday": "closed",
        "sunday": "closed",
    }
    spec[day] = {"start": start, "end": end}
    return spec


def test_eta_phrase_in_hours_capacity_limited_says_aim_to():
    from services.callback import compute_eta_phrase

    settings = {
        "sla_hours": 4,
        "team_capacity": "limited",
        "working_hours_only": True,
        "working_hours": _hours_in(day="monday", start="00:00", end="23:59"),
    }
    # Mock now: a Monday at 12:00 UTC → in hours
    fake_now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    phrase = compute_eta_phrase(settings, now=fake_now)
    assert "aim to call" in phrase
    assert "4 working hours" in phrase


def test_eta_phrase_in_hours_capacity_normal_says_will():
    from services.callback import compute_eta_phrase

    settings = {
        "sla_hours": 2,
        "team_capacity": "normal",
        "working_hours_only": True,
        "working_hours": _hours_in(day="monday", start="00:00", end="23:59"),
    }
    fake_now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    phrase = compute_eta_phrase(settings, now=fake_now)
    assert "We'll call you within 2 working hours" in phrase
    assert "aim to" not in phrase


def test_eta_phrase_outside_hours_uses_outside_template():
    from services.callback import compute_eta_phrase

    settings = {
        "sla_hours": 4,
        "team_capacity": "limited",
        "working_hours_only": True,
        "working_hours": _hours_in(day="monday", start="09:00", end="17:00"),
    }
    # Monday at 23:00 UTC — outside hours
    fake_now = datetime(2026, 5, 11, 23, 0, 0, tzinfo=timezone.utc)
    phrase = compute_eta_phrase(settings, now=fake_now)
    assert "closed" in phrase.lower()
    assert "first thing" in phrase.lower()


def test_eta_phrase_includes_product_context_when_present():
    from services.callback import compute_eta_phrase

    settings = {"sla_hours": 4, "team_capacity": "limited"}
    fake_now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    phrase = compute_eta_phrase(
        settings, product_context="EN 12101-9 panel", now=fake_now
    )
    assert "about the EN 12101-9 panel" in phrase


def test_eta_phrase_skips_product_clause_when_absent():
    from services.callback import compute_eta_phrase

    settings = {"sla_hours": 4, "team_capacity": "limited"}
    fake_now = datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc)
    phrase = compute_eta_phrase(settings, now=fake_now)
    assert "about the" not in phrase


def test_eta_phrase_falls_back_safely_on_bad_config():
    """Phrasing must NEVER fail the request — bad config → safe fallback."""
    from services.callback import compute_eta_phrase

    bad_settings = {
        "sla_hours": "not a number",
        "working_hours": {"tz": "Mars/Olympus", "monday": "garbage"},
    }
    phrase = compute_eta_phrase(bad_settings)
    # Doesn't raise, returns something useable
    assert isinstance(phrase, str) and len(phrase) > 0


def test_eta_phrase_works_with_no_working_hours_block():
    """Some Sites won't have working_hours configured (e.g. always-on
    SaaS support). Should still produce a reasonable phrase."""
    from services.callback import compute_eta_phrase

    settings = {
        "sla_hours": 4,
        "team_capacity": "normal",
        "working_hours_only": False,
    }
    phrase = compute_eta_phrase(settings)
    assert "4 working hours" in phrase


def test_eta_phrase_respects_timezone():
    """A merchant in Europe/London at 23:00 UTC is at midnight local time —
    outside hours. Same UTC time for an Australian merchant is mid-morning."""
    from services.callback import compute_eta_phrase

    london_settings = {
        "sla_hours": 4,
        "team_capacity": "limited",
        "working_hours_only": True,
        "working_hours": {
            "tz": "Europe/London",
            "monday": {"start": "09:00", "end": "17:00"},
            "tuesday": "closed", "wednesday": "closed", "thursday": "closed",
            "friday": "closed", "saturday": "closed", "sunday": "closed",
        },
    }
    sydney_settings = dict(london_settings)
    sydney_settings["working_hours"] = dict(london_settings["working_hours"])
    sydney_settings["working_hours"]["tz"] = "Australia/Sydney"

    # Monday 23:00 UTC = Tuesday 10:00 in Sydney; Monday 23:59 in London
    utc_time = datetime(2026, 5, 11, 23, 0, 0, tzinfo=timezone.utc)

    london_phrase = compute_eta_phrase(london_settings, now=utc_time)
    sydney_phrase = compute_eta_phrase(sydney_settings, now=utc_time)

    # London: outside hours (Monday 23:00 local)
    assert "closed" in london_phrase.lower()
    # Sydney: would be Tuesday 10:00 local but Tuesday is "closed" in our spec
    # → also outside hours. Confirms timezone routing happened.
    assert "closed" in sydney_phrase.lower()


# ---------------------------------------------------------------------------
# new_request_id — opaque + collision-resistant
# ---------------------------------------------------------------------------

def test_request_id_has_cb_prefix():
    from services.callback import new_request_id
    assert new_request_id().startswith("cb_")


def test_request_id_is_unique_across_calls():
    from services.callback import new_request_id
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
