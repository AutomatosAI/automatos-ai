"""PRD-142 Wave 0 US-002 — GET /api/analytics/errors/by-subsystem.

This endpoint backs the dashboard "Error rate by subsystem" tile. It:

* Aggregates ``error_events`` rows by ``subsystem`` over a configurable
  rolling window (default 24h).
* Filters by the caller's ``ctx.workspace_id`` — system-level rows
  (``workspace_id IS NULL``) are intentionally excluded from the
  workspace-scoped view (US-002 notes).
* Computes ``rate = count / total`` per subsystem; ``rate`` is 0 when
  ``total`` is 0 (no divide-by-zero).
* Uses the ``idx_error_events_subsystem_created`` index path via a filter
  on ``created_at >= window_start`` and a ``GROUP BY subsystem`` — no
  full-table scan.

Tests cover the contract: aggregation shape, window math, empty-window
safety, and workspace isolation.
"""
from __future__ import annotations

import operator
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

import config  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(subsystem: str, count: int) -> SimpleNamespace:
    """Row shape returned by the aggregation query: (subsystem, count)."""
    return SimpleNamespace(subsystem=subsystem, count=count)


def _filter_on(args: Iterable, column_name: str):
    """Return the SQLAlchemy ``BinaryExpression`` in ``args`` whose left
    side is the column named ``column_name``, or ``None`` if absent."""
    for expr in args:
        left = getattr(expr, "left", None)
        if left is not None and getattr(left, "name", None) == column_name:
            return expr
    return None


def _make_client(
    *,
    workspace_id: UUID,
    agg_rows: Optional[List[SimpleNamespace]] = None,
    captured_filters: Optional[List[Tuple]] = None,
):
    """Build a TestClient bound to the analytics_real router with the
    workspace ctx and DB dependencies overridden.

    Returns ``(client, fake_db, captured_filters)``. ``captured_filters``
    accumulates ``(args, kwargs)`` for each ``q.filter(...)`` call so tests
    can introspect the SQL expressions the endpoint applies.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.analytics_real import router as analytics_router
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    fake_db = MagicMock()
    captured_filters = captured_filters if captured_filters is not None else []

    q = MagicMock()

    def _capture_filter(*args, **kwargs):
        captured_filters.append((args, kwargs))
        return q

    q.filter.side_effect = _capture_filter
    q.group_by.return_value = q
    q.all.return_value = agg_rows or []
    fake_db.query.return_value = q

    app = FastAPI()
    app.include_router(analytics_router)

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

    return TestClient(app), fake_db, captured_filters


# ---------------------------------------------------------------------------
# Tests (named exactly as US-002 AC requires)
# ---------------------------------------------------------------------------


def test_groups_by_subsystem():
    """Endpoint groups error_events by subsystem and computes rate = count/total."""
    ws = uuid4()
    rows = [_row("memory", 6), _row("tools", 3), _row("harness", 1)]
    client, _db, _captured = _make_client(workspace_id=ws, agg_rows=rows)

    resp = client.get("/api/analytics/errors/by-subsystem?window=24h")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["window"] == "24h"
    assert body["total"] == 10

    by_sub = {entry["subsystem"]: entry for entry in body["by_subsystem"]}
    assert set(by_sub) == {"memory", "tools", "harness"}

    assert by_sub["memory"]["count"] == 6
    assert by_sub["tools"]["count"] == 3
    assert by_sub["harness"]["count"] == 1

    # rate = count / total (over the window)
    assert by_sub["memory"]["rate"] == pytest.approx(0.6)
    assert by_sub["tools"]["rate"] == pytest.approx(0.3)
    assert by_sub["harness"]["rate"] == pytest.approx(0.1)

    assert "generated_at" in body
    # generated_at is ISO 8601 parseable
    datetime.fromisoformat(body["generated_at"])


def test_window_filtering():
    """The endpoint applies a >= filter on created_at sized to the window arg.

    Verifies (a) the index-eligible filter is present and (b) the cutoff
    is roughly ``now - window`` so we cannot accidentally return all-time
    error counts when the dashboard asks for the last 24h.
    """
    ws = uuid4()
    captured: List[Tuple] = []

    client, _db, _ = _make_client(
        workspace_id=ws, agg_rows=[_row("memory", 1)], captured_filters=captured
    )

    before = datetime.utcnow()
    resp = client.get("/api/analytics/errors/by-subsystem?window=24h")
    after = datetime.utcnow()
    assert resp.status_code == 200, resp.text

    assert captured, "endpoint must call .filter(...) on the query"
    args, _kwargs = captured[0]

    created_expr = _filter_on(args, "created_at")
    assert created_expr is not None, (
        f"created_at window filter must be present in filter args; got {args}"
    )
    # It must be a >= comparator (rolling window, not equality).
    assert created_expr.operator is operator.ge, (
        f"created_at filter must use >= for the window; got {created_expr.operator!r}"
    )

    cutoff = created_expr.right.value
    expected_low = before - timedelta(hours=24, seconds=5)
    expected_high = after - timedelta(hours=24) + timedelta(seconds=5)
    assert expected_low <= cutoff <= expected_high, (
        f"window=24h cutoff must be ~now-24h; got {cutoff}, expected in "
        f"[{expected_low}, {expected_high}]"
    )


def test_empty_window_returns_zero():
    """No rows in the window → total=0, by_subsystem=[], and NO divide-by-zero."""
    ws = uuid4()
    client, _db, _ = _make_client(workspace_id=ws, agg_rows=[])

    resp = client.get("/api/analytics/errors/by-subsystem?window=24h")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["total"] == 0
    assert body["by_subsystem"] == []
    # Window echoed back so the UI can label the tile.
    assert body["window"] == "24h"


def test_workspace_isolation():
    """The query filters by the caller's workspace_id — another tenant cannot leak."""
    my_ws = uuid4()
    captured: List[Tuple] = []
    client, _db, _ = _make_client(
        workspace_id=my_ws, agg_rows=[], captured_filters=captured
    )

    resp = client.get("/api/analytics/errors/by-subsystem?window=24h")
    assert resp.status_code == 200, resp.text

    assert captured, "endpoint must call .filter(...) on the query"
    args, _kwargs = captured[0]

    ws_expr = _filter_on(args, "workspace_id")
    assert ws_expr is not None, (
        f"workspace_id filter must be present (tenant isolation); got {args}"
    )
    # Equality, not IN/IS NULL — we want the caller's workspace exactly.
    assert ws_expr.operator is operator.eq, (
        f"workspace_id filter must use ==; got {ws_expr.operator!r}"
    )
    # The bind value on the right side must be the caller's workspace_id.
    assert ws_expr.right.value == my_ws, (
        f"workspace_id filter must bind to caller's workspace; got {ws_expr.right.value}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
