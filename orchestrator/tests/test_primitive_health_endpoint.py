"""PRD-142 Wave 3 · WS-M · W3-S2 — GET /api/analytics/primitive-health.

The Command Centre per-primitive health tile reads ONE endpoint that returns
each of the 8 primitives' current status (the Wave 0 US-006 deferred tile).
This endpoint stays honest:

* ALL 8 primitives are ALWAYS returned (chat, memory, rag, nl2sql, graph,
  missions, playbooks, channels — the closed set pinned in W3-S1).
* Each primitive's status is the LATEST ``primitive_check`` finding written
  by the W3-S1 ``emit_primitive_finding`` helper (ORDER BY created_at DESC).
* A primitive with NO finding renders as ``{status: "unknown",
  last_checked: null}`` — never a fake green (AC#3, W3-S1 AC#4).
* The query is workspace-scoped (``WHERE workspace_id = :ws_id``) so
  workspace A cannot read workspace B's primitive findings.

Tests cover the contract: shape (all 8, always), latest-wins, unknown-when-
missing, and workspace isolation. They use a fake DB session that records
``execute(text, params)`` calls — no Postgres needed — mirroring the
US-002 (test_errors_by_subsystem_endpoint.py) and W3-S1
(test_heartbeat_primitive_findings.py) patterns.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# config.py + database engine refuse to build without POSTGRES_* env in test.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import config  # noqa: E402,F401


# The closed set the endpoint MUST always return — duplicated from
# services/heartbeat_service.PRIMITIVE_NAMES so the test fails loudly if the
# canonical set ever drifts (CLAUDE.md §10 — no legacy nouns).
EXPECTED_PRIMITIVES = {
    "chat", "memory", "rag", "nl2sql",
    "graph", "missions", "playbooks", "channels",
}


def _row(primitive: str, status: str, created_at: datetime, detail: str = "") -> SimpleNamespace:
    """Row shape returned by the primitive-health query.

    The endpoint extracts JSONB keys via ``findings->0->>'primitive'`` etc.,
    so the row exposes them as plain attributes — this stand-in keeps the
    test independent of the SQL dialect.
    """
    return SimpleNamespace(
        primitive=primitive,
        status=status,
        detail=detail,
        created_at=created_at,
    )


def _make_client(
    *,
    workspace_id: UUID,
    rows: Optional[List[SimpleNamespace]] = None,
    captured_execs: Optional[List[Tuple]] = None,
):
    """Build a TestClient bound to the analytics_real router.

    Returns ``(client, fake_db, captured_execs)``. ``captured_execs`` records
    every ``db.execute(stmt, params)`` call so tests can assert the
    workspace_id bind + the primitive_check filter actually went into the
    SQL.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.analytics_real import router as analytics_router
    from api.analytics_real import ws_router as analytics_ws_router  # PRD-185 S12: primitive-health moved here
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    fake_db = MagicMock()
    captured_execs = captured_execs if captured_execs is not None else []

    def _capture_execute(stmt, params=None):
        captured_execs.append((str(stmt), dict(params or {})))
        result = MagicMock()
        result.fetchall.return_value = rows or []
        return result

    fake_db.execute.side_effect = _capture_execute

    app = FastAPI()
    app.include_router(analytics_router)
    app.include_router(analytics_ws_router)  # PRD-185 S12: /primitive-health now workspace-admin gated

    def _override_ctx():
        return RequestContext(
            workspace_id=workspace_id,
            # PRD-143 S6: analytics_real is obs-tier — su principal required to reach the handler.
            user=UserContext(id="test-user", email="test@example.com", role="owner", system_role="super_admin"),
            auth_type="clerk",
        )

    def _override_db():
        yield fake_db

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db

    return TestClient(app), fake_db, captured_execs


# ---------------------------------------------------------------------------
# 1. AC#3: ALL 8 primitives ALWAYS present, even when zero rows exist.
# ---------------------------------------------------------------------------


def test_returns_all_eight_primitives():
    """An empty heartbeat table still returns every canonical primitive —
    each with ``status="unknown"`` + ``last_checked=null``. The closed set
    matches services/heartbeat_service.PRIMITIVE_NAMES."""
    ws = uuid4()
    client, _db, _ = _make_client(workspace_id=ws, rows=[])

    resp = client.get("/api/analytics/primitive-health")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert "primitives" in body
    assert "generated_at" in body
    datetime.fromisoformat(body["generated_at"])  # ISO 8601 parseable

    names = {p["name"] for p in body["primitives"]}
    assert names == EXPECTED_PRIMITIVES, (
        f"expected the 8 canonical primitives, got {names}"
    )

    # Every entry is unknown / null when nothing has been emitted.
    for p in body["primitives"]:
        assert p["status"] == "unknown", (
            f"primitive {p['name']} must default to 'unknown'; got {p['status']!r}"
        )
        assert p["last_checked"] is None, (
            f"primitive {p['name']} must have null last_checked; got {p['last_checked']!r}"
        )


# ---------------------------------------------------------------------------
# 2. AC#2: latest finding per primitive wins (ORDER BY created_at DESC).
# ---------------------------------------------------------------------------


def test_latest_finding_wins():
    """When a primitive has multiple findings, the most recent one — by
    created_at — sets the tile's status. Older findings for the SAME
    primitive are silently superseded."""
    ws = uuid4()
    now = datetime.utcnow()
    older = now - timedelta(hours=2)

    # The endpoint queries ORDER BY created_at DESC, so the row list it
    # receives is already newest-first. Memory has TWO findings here: the
    # latest is 'green', the older one is 'down'. The endpoint MUST pick
    # the green one (latest wins).
    rows = [
        _row("memory", "green", now, "probe ok"),
        _row("memory", "down", older, "probe failed earlier"),
        # A different primitive with one finding — proves the dedupe is per-name.
        _row("rag", "degraded", now, "slow ingest"),
    ]
    client, _db, _ = _make_client(workspace_id=ws, rows=rows)

    resp = client.get("/api/analytics/primitive-health")
    assert resp.status_code == 200, resp.text

    by_name = {p["name"]: p for p in resp.json()["primitives"]}

    assert by_name["memory"]["status"] == "green", (
        "latest finding (green) must win over older (down)"
    )
    assert by_name["memory"]["last_checked"] is not None
    # last_checked echoes the latest row's created_at (ISO 8601).
    assert by_name["memory"]["last_checked"].startswith(now.isoformat()[:19])

    assert by_name["rag"]["status"] == "degraded"
    assert by_name["rag"]["last_checked"] is not None


# ---------------------------------------------------------------------------
# 3. AC#3 + W3-S1 AC#4: a primitive with no finding stays 'unknown'.
# ---------------------------------------------------------------------------


def test_missing_primitive_is_unknown():
    """Only Memory has emitted a finding (W3-S7 pathfinder). The other 7
    primitives in the response stay ``unknown`` / ``null``. No fake greens,
    no placeholder seeds (W3-S1 AC#4)."""
    ws = uuid4()
    now = datetime.utcnow()
    rows = [_row("memory", "green", now, "probe ok")]

    client, _db, _ = _make_client(workspace_id=ws, rows=rows)

    resp = client.get("/api/analytics/primitive-health")
    assert resp.status_code == 200, resp.text

    by_name = {p["name"]: p for p in resp.json()["primitives"]}
    assert set(by_name) == EXPECTED_PRIMITIVES

    assert by_name["memory"]["status"] == "green"
    assert by_name["memory"]["last_checked"] is not None

    unhardened = EXPECTED_PRIMITIVES - {"memory"}
    for name in unhardened:
        assert by_name[name]["status"] == "unknown", (
            f"un-hardened primitive {name} must be 'unknown'; got "
            f"{by_name[name]['status']!r}"
        )
        assert by_name[name]["last_checked"] is None


# ---------------------------------------------------------------------------
# 4. Workspace isolation: the SQL binds workspace_id from the caller's ctx.
# ---------------------------------------------------------------------------


def test_workspace_isolation():
    """The query filters by the caller's workspace_id — another tenant's
    primitive findings cannot leak. The bind param MUST match ctx."""
    my_ws = uuid4()
    captured: List[Tuple] = []
    client, _db, _ = _make_client(
        workspace_id=my_ws, rows=[], captured_execs=captured
    )

    resp = client.get("/api/analytics/primitive-health")
    assert resp.status_code == 200, resp.text

    assert captured, "endpoint must call db.execute(...) once"
    stmt, params = captured[0]

    # SQL shape: workspace_id filter + the primitive_check finding_type
    # filter must both be present (index path + the W3-S1 finding shape).
    assert "workspace_id" in stmt, (
        f"query must filter by workspace_id (tenant isolation); got {stmt!r}"
    )
    assert "primitive_check" in stmt, (
        "query must filter to primitive_check findings only — other heartbeat "
        "findings (agent_health, llm_error, checklist) must not leak into the tile"
    )
    assert "created_at" in stmt and "DESC" in stmt.upper(), (
        "query must ORDER BY created_at DESC so latest finding wins"
    )

    # Bind: the workspace_id param is the caller's ctx, not a hardcoded value.
    assert "ws_id" in params, f"expected ws_id bind param; got {params!r}"
    assert str(params["ws_id"]) == str(my_ws), (
        f"workspace_id bind must be caller's workspace; got {params['ws_id']!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
