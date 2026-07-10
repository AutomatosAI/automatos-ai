"""PRD-142 Wave 3 · WS-M · W3-S1 — heartbeat helper emits primitive-mapped findings.

The Command Centre primitive-health tile (W3-S2/S3) reads
``heartbeat_results`` rows whose finding dict carries ``finding_type =
"primitive_check"`` + ``primitive`` + ``status``. Today the existing
heartbeat writer emits only operational findings (``agent_health`` /
``checklist`` / ``error`` / ``exec_error`` / ``llm_*``) — none of those map
to a product primitive, so the tile has no honest data source.

W3-S1 builds the mechanism: an ``emit_primitive_finding`` helper alongside
``_store_heartbeat_result`` that any primitive's hardening story
(S6 chat … S13 channels) can call when it has a real signal — and wires
Memory (W3-S7's pathfinder) as the first real caller via the durable-store health
probe. Guarantees we PIN here:

1. Each emit writes exactly one ``heartbeat_results`` row whose JSONB
   ``findings`` carries the primitive_check shape (no schema change).
2. The durable-store probe wiring only emits ``memory`` findings — never
   ``rag`` / ``chat`` / etc. Un-hardened primitives stay silent so the
   tile reads ``unknown`` for them (AC#4: never a fake green).
3. A failed write is swallowed — the heartbeat cycle cannot be broken by
   the primitive emit.
4. Only canonical lowercase primitive + status keys are accepted; legacy
   nouns (workflow / recipe) and miscased values are rejected with no
   row written.

All four tests use a recording-session monkeypatch — no real DB.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# heartbeat_service / database modules touch the SQLAlchemy engine on import,
# which refuses to build without POSTGRES_* env. Setdefault keeps real .env wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


class _CapturingSession:
    """Records every ``execute(text, params)`` + lifecycle call.

    Stands in for a real SQLAlchemy session: emit_primitive_finding opens
    SessionLocal(), executes one INSERT, commits, then closes — this records
    each step in order so the test can assert on the persisted shape without
    touching Postgres.
    """

    def __init__(self) -> None:
        self.inserts: list[tuple[str, dict]] = []
        self.commits = 0
        self.closes = 0

    def execute(self, stmt, params=None):  # noqa: D401 — SA Session shape
        self.inserts.append((str(stmt), dict(params or {})))
        return MagicMock()

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closes += 1


def _patch_session_local(monkeypatch: pytest.MonkeyPatch, session) -> None:
    """Swap ``SessionLocal`` so the helper's call hits our recorder.

    The helper does ``from core.database.database import SessionLocal`` at
    call time, so monkeypatching the module attribute is enough — each call
    re-resolves through the module.
    """
    import core.database.database as dbmod

    monkeypatch.setattr(dbmod, "SessionLocal", lambda: session)


# ---------------------------------------------------------------------------
# 1. Shape pin: one row, primitive_check finding, right keys.
# ---------------------------------------------------------------------------


def test_primitive_finding_written_with_status(monkeypatch):
    """emit_primitive_finding writes ONE heartbeat_results row tied to the
    workspace, with a single ``primitive_check`` finding carrying the
    primitive + status + detail."""
    from services.heartbeat_service import emit_primitive_finding

    session = _CapturingSession()
    _patch_session_local(monkeypatch, session)

    ws_id = str(uuid4())
    ok = emit_primitive_finding(ws_id, "memory", "green", "durable store ok")

    assert ok is True
    assert len(session.inserts) == 1, "expected exactly one INSERT"
    assert session.commits == 1
    assert session.closes == 1

    stmt, params = session.inserts[0]
    assert "INSERT INTO heartbeat_results" in stmt
    # Workspace + source bind to the same ws_id (orchestrator-level row).
    assert params["workspace_id"] == ws_id
    assert params["source_id"] == ws_id

    findings = json.loads(params["findings"])
    assert len(findings) == 1, "expected exactly one finding in the payload"
    f = findings[0]
    assert f["finding_type"] == "primitive_check"
    assert f["primitive"] == "memory"
    assert f["status"] == "green"
    assert f["detail"] == "durable store ok"


# ---------------------------------------------------------------------------
# 2. AC#4: un-hardened primitives emit nothing (no fake greens).
# ---------------------------------------------------------------------------


def test_unhardened_primitive_emits_nothing(monkeypatch):
    """Running the W3-S1 wiring (durable-store probe tick) emits ``memory``
    findings ONLY — never rag/chat/nl2sql/graph/missions/playbooks/channels.
    The other 7 primitives have no caller in this story; the tile must read
    ``unknown`` for them (AC#4)."""
    from services.heartbeat_service import HeartbeatService, PRIMITIVE_NAMES

    session = _CapturingSession()
    _patch_session_local(monkeypatch, session)

    # Durable-store stub: healthy probe → wiring will emit a 'memory'/'green'
    # finding. Nothing else should fire.
    async def _health_ok() -> dict:
        return {"healthy": True}

    fake_store = MagicMock()
    fake_store.health = _health_ok

    fake_ums = MagicMock()
    fake_ums._durable = fake_store

    # Stub modules.memory.unified_memory_service in sys.modules BEFORE the
    # heartbeat tick's local ``from ... import get_unified_memory_service``
    # runs, so the real (pgvector/asyncpg-heavy) memory chain never loads.
    # This is the project's standard pattern for testing code that imports
    # heavy infra at call time.
    fake_module = types.ModuleType("modules.memory.unified_memory_service")
    fake_module.get_unified_memory_service = lambda: fake_ums
    monkeypatch.setitem(
        sys.modules, "modules.memory.unified_memory_service", fake_module
    )

    svc = HeartbeatService()
    ws_id = str(uuid4())
    sched = MagicMock()
    job = MagicMock()
    job.id = f"orch_hb_{ws_id}"
    sched.get_jobs.return_value = [job]
    svc._scheduler = sched

    asyncio.run(svc._durable_memory_probe_tick())

    primitives_seen: set[str] = set()
    for _, params in session.inserts:
        for f in json.loads(params["findings"]):
            if f.get("finding_type") == "primitive_check":
                primitives_seen.add(f["primitive"])

    assert primitives_seen == {"memory"}, (
        f"expected only 'memory' primitive findings, got {primitives_seen}"
    )
    unhardened = PRIMITIVE_NAMES - {"memory"}
    assert primitives_seen.isdisjoint(unhardened), (
        "un-hardened primitive emitted a finding (placeholder seed forbidden)"
    )


# ---------------------------------------------------------------------------
# 3. Best-effort: a failed write NEVER raises (cannot break the cycle).
# ---------------------------------------------------------------------------


def test_finding_write_failure_is_swallowed(monkeypatch):
    """Any DB exception during the INSERT is logged + swallowed. The helper
    returns False; it does NOT raise. This is the guarantee that lets every
    primitive's hardening story call emit_primitive_finding freely without
    risking the parent heartbeat cycle (AC#3)."""
    from services.heartbeat_service import emit_primitive_finding

    class _RaisingSession:
        def execute(self, *a, **k):
            raise RuntimeError("simulated DB outage")

        def commit(self) -> None:
            pass

        def close(self) -> None:
            pass

    import core.database.database as dbmod

    monkeypatch.setattr(dbmod, "SessionLocal", lambda: _RaisingSession())

    # MUST NOT raise — even when the underlying write blows up.
    ok = emit_primitive_finding(str(uuid4()), "memory", "green", "anything")
    assert ok is False


# ---------------------------------------------------------------------------
# 4. AC#6: lowercase canonical primitive keys only — legacy nouns rejected.
# ---------------------------------------------------------------------------


def test_primitive_name_must_be_canonical(monkeypatch):
    """The helper rejects non-canonical primitive names (legacy 'workflow' /
    'recipe', miscased, or unknown) AND non-canonical statuses — no row,
    no raise. Belt-and-braces against drift since the analytics endpoint
    relies on the closed set."""
    from services.heartbeat_service import emit_primitive_finding

    session = _CapturingSession()
    _patch_session_local(monkeypatch, session)

    banned_primitives = (
        "workflow",   # legacy: canonical is 'missions'
        "recipe",     # legacy: canonical is 'playbooks'
        "Memory",     # miscased — keys are lowercase
        "documents",  # unknown — not in the 8
        "",           # empty
    )
    for bad in banned_primitives:
        ok = emit_primitive_finding(str(uuid4()), bad, "green", "x")
        assert ok is False, f"expected reject for primitive={bad!r}"

    banned_statuses = ("OK", "healthy", "broken", "yellow", "")
    for bad in banned_statuses:
        ok = emit_primitive_finding(str(uuid4()), "memory", bad, "x")
        assert ok is False, f"expected reject for status={bad!r}"

    # Not one INSERT slipped through for any of the bad inputs.
    assert session.inserts == [], "no row may be written for invalid input"
