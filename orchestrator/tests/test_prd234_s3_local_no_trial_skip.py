"""PRD-234 S3 — the local edition never skips background work for lack of credits.

Found 2026-09-03: Auto-led onboarding granted a trial record locally too, so
``HeartbeatService._trial_skip`` answered a skip for every agent heartbeat on
the operator's own machine. There is no platform-paid credit to protect there.

PRD-265 W1 replaced the trial-record check with
``services.credits.routing.background_allowed`` (the local edition is always
funded — ``workspace_funding`` returns ``FUNDED_BY_LOCAL`` before ever
querying the workspace), and the skip reason is now ``"no_credits"``. The
method keeps its old name (``_trial_skip``) — only what it consults changed.
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

from config import config  # noqa: E402
from services.heartbeat_service import HeartbeatService  # noqa: E402


class _NoQueryDB:
    """A session stand-in that fails loudly if anything queries it — proves
    the local edition's funding check never reaches the workspace row."""

    def query(self, *_args, **_kwargs):
        raise AssertionError("local edition must never query the workspace for credits")

    def close(self):
        pass


def test_local_edition_never_queries_the_database_for_credits(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local", raising=False)
    import core.database.database as dbmod
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _NoQueryDB())
    # No database is reachable here: a local-edition call must return before
    # background_allowed ever queries the workspace (workspace_funding's
    # AUTH_EDITION == "local" short-circuit).
    assert HeartbeatService._trial_skip(None, "00000000-0000-0000-0000-0000000000c1") is None


def test_saas_edition_without_credits_skips_the_tick(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas", raising=False)
    monkeypatch.setattr("services.credits.routing.background_allowed", lambda db, ws: False)

    class _DB:
        def close(self):
            pass

    import core.database.database as dbmod
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _DB())

    out = HeartbeatService._trial_skip(None, "ws-1")
    assert out == {"status": "skipped", "reason": "no_credits"}


def test_saas_edition_with_credits_does_not_skip(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas", raising=False)
    monkeypatch.setattr("services.credits.routing.background_allowed", lambda db, ws: True)

    class _DB:
        def close(self):
            pass

    import core.database.database as dbmod
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _DB())

    assert HeartbeatService._trial_skip(None, "ws-1") is None


def test_the_guard_is_wired_to_background_allowed():
    src = (_ORCH / "services" / "heartbeat_service.py").read_text()
    assert "background_allowed" in src
