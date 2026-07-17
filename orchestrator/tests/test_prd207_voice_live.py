"""PRD-207 — Auto Live: real-time voice + the presence orb.

* S3/S4 — voice_calls lifecycle model + migration chain, the settings plane
  (platform toggle + Retell creds from DB system_settings, never env), the
  workspace voice_live shape, the cap-gate formula with active-call
  reservation, and the fail-closed settings whitelist.
* S1 — the web-call mint: gates in order with honest refusals, dynamic vars,
  the row born at mint.
* S2 — the webhook trust boundary: mint-row cross-validation, binding to the
  on-screen chat + real user, fail-closed fallback, phone lane.
* S3 — lifecycle events idempotency, loud orphans, HMAC refusal, the meter.
* S8 — telemetry parity (voice_turns written by the live path).
* Guard — the live path never imports the 120s-TTS pod client.

Pure logic is tested without a DB; DB seams follow the PRD-205 idiom
(skip cleanly without Postgres); vendor HTTP is always mocked.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))


# ---------------------------------------------------------------------------
# S3/S4 · migration chain — single-parent on the current head
# ---------------------------------------------------------------------------

def test_migration_chains_on_prd206_chat_summary():
    mig = (_orchestrator_root / "alembic" / "versions" / "prd207_voice_live.py").read_text()
    assert 'revision = "prd207_voice_live"' in mig
    # chains directly on the current single head (PRD-206 S2), no second join
    assert 'down_revision = "prd206_chat_summary"' in mig
    assert "voice_calls" in mig and "call_id" in mig


def test_no_other_migration_chains_on_prd206_head():
    """prd207_voice_live must be the ONLY child of prd206_chat_summary —
    a second child would re-fork main's migration history (the #545/#548
    parallel-merge-heads lesson)."""
    versions = _orchestrator_root / "alembic" / "versions"
    children = [
        p.name
        for p in versions.glob("*.py")
        if 'down_revision = "prd206_chat_summary"' in p.read_text()
    ]
    assert children == ["prd207_voice_live.py"]


# ---------------------------------------------------------------------------
# S4 · workspace voice_live shape (pure)
# ---------------------------------------------------------------------------

def test_parse_workspace_voice_live_defaults_fail_closed():
    from modules.voice.live_settings import parse_workspace_voice_live

    for raw in (None, {}, {"voice_live": None}, {"voice_live": "on"}, {"voice_live": []}):
        view = parse_workspace_voice_live(raw)  # type: ignore[arg-type]
        assert view.enabled is False  # never live by accident
        assert view.monthly_cap_minutes > 0  # config default applies
        assert view.retell_voice_id is None


def test_parse_workspace_voice_live_reads_values():
    from modules.voice.live_settings import parse_workspace_voice_live

    view = parse_workspace_voice_live(
        {"voice_live": {"enabled": True, "monthly_cap_minutes": 250, "retell_voice_id": "retell-Cimo"}}
    )
    assert view.enabled is True
    assert view.monthly_cap_minutes == 250
    assert view.retell_voice_id == "retell-Cimo"


def test_parse_workspace_voice_live_bad_cap_falls_back():
    from config import config
    from modules.voice.live_settings import parse_workspace_voice_live

    for bad in ("many", -3, 0, None):
        view = parse_workspace_voice_live({"voice_live": {"enabled": True, "monthly_cap_minutes": bad}})
        assert view.monthly_cap_minutes == int(config.VOICE_LIVE_DEFAULT_MONTHLY_CAP_MINUTES)


def test_validate_voice_live_update_matrix():
    from modules.voice.live_settings import validate_voice_live_update

    # happy path normalizes
    ok = validate_voice_live_update(
        {"enabled": True, "monthly_cap_minutes": 60, "retell_voice_id": " v1 "}
    )
    assert ok == {"enabled": True, "monthly_cap_minutes": 60, "retell_voice_id": "v1"}

    for bad in (
        "on",                                   # not an object
        {"enabled": "yes"},                     # non-bool enabled
        {"monthly_cap_minutes": True},          # bool is not an int here
        {"monthly_cap_minutes": 0},             # zero cap
        {"monthly_cap_minutes": 200_000},       # absurd cap
        {"retell_voice_id": "x" * 65},          # oversized voice id
        {"surprise": 1},                        # unknown key fail-closed
    ):
        with pytest.raises(ValueError):
            validate_voice_live_update(bad)


# ---------------------------------------------------------------------------
# S4 · the cap formula (pure) — reservation bounds the two-tabs race
# ---------------------------------------------------------------------------

def test_cap_formula_boundary_and_reservation():
    from modules.voice.voice_meter import MeterReading, cap_allows_mint

    # under cap, no active calls → allowed
    ok, _ = cap_allows_mint(MeterReading(80, 0, 10), cap_minutes=100)
    assert ok

    # exactly at cap → refused, honest reason
    refused, reason = cap_allows_mint(MeterReading(100, 0, 10), cap_minutes=100)
    assert not refused
    assert "100/100" in reason

    # the second simultaneous mint sees the first call's reservation:
    # 95 ended + 1 active × 10 reserve = 105 ≥ 100 → refused
    refused2, reason2 = cap_allows_mint(MeterReading(95, 1, 10), cap_minutes=100)
    assert not refused2
    assert "reserved" in reason2


def test_month_window_utc_covers_year_rollover():
    from datetime import datetime, timezone

    from modules.voice.voice_meter import month_window_utc

    start, nxt = month_window_utc(datetime(2026, 12, 15, 9, 30, tzinfo=timezone.utc))
    assert (start.year, start.month, start.day) == (2026, 12, 1)
    assert (nxt.year, nxt.month) == (2027, 1)


# ---------------------------------------------------------------------------
# S4 · settings plane reads system_settings, never env
# ---------------------------------------------------------------------------

def test_platform_toggle_reads_system_settings(monkeypatch):
    import modules.voice.live_settings as ls

    calls = []

    def fake_get(category, key, default=None):
        calls.append((category, key))
        return {"live_enabled": "true", "retell_api_key": "k", "retell_webhook_secret": "s",
                "retell_agent_id": "a"}.get(key, default)

    monkeypatch.setattr(ls, "get_system_setting", fake_get)
    assert ls.voice_live_enabled() is True
    creds = ls.retell_credentials()
    assert creds.armed
    assert ("voice", "live_enabled") in calls  # DB settings, not config/env


def test_credentials_not_armed_when_any_missing(monkeypatch):
    import modules.voice.live_settings as ls

    monkeypatch.setattr(
        ls, "get_system_setting",
        lambda c, k, d=None: {"retell_api_key": "k", "retell_webhook_secret": ""}.get(k, d),
    )
    assert ls.retell_credentials().armed is False


# ---------------------------------------------------------------------------
# S4 · workspace-settings whitelist (the PRD-143 S11 fail-closed surface)
# ---------------------------------------------------------------------------

def _mock_ws_db(initial_settings=None):
    ws = SimpleNamespace(settings=dict(initial_settings or {}), id=uuid.uuid4())
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ws
    return db, ws


def test_whitelist_still_refuses_unknown_keys():
    from modules.tools.discovery.handlers_workspace import update_workspace_settings

    db, _ = _mock_ws_db()
    out = asyncio.run(
        update_workspace_settings(db, uuid.uuid4(), {"key": "integrations", "value": {}})
    )
    assert out["success"] is False
    assert "voice_live" in out["error"]  # the whitelist names its members


def test_whitelist_accepts_and_merges_voice_live():
    from modules.tools.discovery.handlers_workspace import update_workspace_settings

    db, ws = _mock_ws_db({"voice_live": {"enabled": False, "retell_voice_id": "keep-me"}})
    out = asyncio.run(
        update_workspace_settings(
            db, uuid.uuid4(), {"key": "voice_live", "value": {"enabled": True}}
        )
    )
    assert out["success"] is True
    assert ws.settings["voice_live"]["enabled"] is True
    # merge, not replace: the untouched key survives
    assert ws.settings["voice_live"]["retell_voice_id"] == "keep-me"


def test_whitelist_refuses_malformed_voice_live():
    from modules.tools.discovery.handlers_workspace import update_workspace_settings

    db, ws = _mock_ws_db({"voice_live": {"enabled": False}})
    out = asyncio.run(
        update_workspace_settings(
            db, uuid.uuid4(), {"key": "voice_live", "value": {"enabled": "yes"}}
        )
    )
    assert out["success"] is False
    assert ws.settings["voice_live"]["enabled"] is False  # nothing written
