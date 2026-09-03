"""PRD-234 S3 — the local edition never skips background work as a 'trial workspace'.

Found 2026-09-03: Auto-led onboarding grants a trial record locally too, so
HeartbeatService._trial_skip answered {"status": "skipped", "reason": "trial_workspace"}
for every agent heartbeat on the operator's own machine. There is no platform-paid
credit to protect there.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import pytest  # noqa: E402

from config import config  # noqa: E402
from services.heartbeat_service import HeartbeatService  # noqa: E402


def test_local_edition_never_trial_skips_a_heartbeat(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local", raising=False)
    # No database is reachable here: a local-edition call must return before touching one.
    assert HeartbeatService._trial_skip(None, "00000000-0000-0000-0000-0000000000c1") is None


def test_saas_edition_still_consults_the_trial_ledger(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas", raising=False)
    calls = []

    class _WS:  # a workspace that IS on trial
        onboarding = {"trial": {"state": "active"}}

    class _Q:
        def get(self, _id):
            calls.append(_id)
            return _WS()

    class _DB:
        def query(self, *_a):
            return _Q()

        def close(self):
            pass

    import core.database.database as dbmod
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _DB())
    out = HeartbeatService._trial_skip(None, "ws-1")
    assert out == {"status": "skipped", "reason": "trial_workspace"} and calls == ["ws-1"]


def test_the_guard_string_pinned_by_prd222_is_still_present():
    src = (_ORCH / "services" / "heartbeat_service.py").read_text()
    assert "is_trial_active_workspace" in src
