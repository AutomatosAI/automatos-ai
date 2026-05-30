"""PRD-142 Wave 0 US-004 — GET /api/analytics/widget-engagement.

Read-only aggregation over the existing ``widget_event_log`` sink
(``core/models/widget_event_log.py``; writer in
``modules/widgets/telemetry.py``).

The endpoint backs the dashboard "Widget engagement" tile:

* Resolves the sites belonging to the caller's ``ctx.workspace_id`` and
  aggregates ``widget_event_log`` rows for ONLY those sites — tenant
  isolation, since ``widget_event_log`` itself has no ``workspace_id``.
* Counts grouped by ``event_type`` over a rolling window (default 7d),
  restricted to the ``WIDGET_EVENT_TYPES`` allow-list so the
  ``idx_widget_event_log_type_created`` index path is eligible.
* Returns the number of distinct sessions in the window.
* Is READ-ONLY — no ``WidgetEventLog(...)`` row construction in the
  endpoint module (the writer remains the single source of truth).
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


def _row(event_type: str, count: int) -> SimpleNamespace:
    """Row shape returned by the aggregation query: (event_type, count)."""
    return SimpleNamespace(event_type=event_type, count=count)


def _filter_on(args: Iterable, column_name: str):
    """Return the SQLAlchemy expression in ``args`` whose left side is the
    column named ``column_name``, or ``None`` if absent.
    """
    for expr in args:
        left = getattr(expr, "left", None)
        if left is not None and getattr(left, "name", None) == column_name:
            return expr
    return None


def _make_client(
    *,
    workspace_id: UUID,
    site_ids: Optional[List[UUID]] = None,
    agg_rows: Optional[List[SimpleNamespace]] = None,
    distinct_sessions: int = 0,
    captured_filters: Optional[List[Tuple]] = None,
):
    """Build a TestClient bound to the analytics_real router with the
    workspace ctx and DB dependencies overridden.

    The endpoint issues queries in this order:

    1. ``db.query(Site.id).filter(Site.workspace_id == ws).all()``
    2. ``db.query(WidgetEventLog.event_type, count).filter(...).group_by(...).all()``
    3. ``db.query(func.count(func.distinct(WidgetEventLog.session_id))).filter(...).scalar()``

    When ``site_ids`` is empty the endpoint short-circuits and queries
    2/3 are not invoked.

    ``captured_filters`` collects ``(tag, args, kwargs)`` for every
    ``.filter(...)`` call where ``tag`` identifies which query the filter
    came from ("sites", "agg", or "sessions"). Tests use this to make
    tenant-isolation and window-math assertions.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.analytics_real import router as analytics_router
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    fake_db = MagicMock()
    captured_filters = (
        captured_filters if captured_filters is not None else []
    )

    site_q = MagicMock()
    site_q.filter.side_effect = lambda *a, **k: (
        captured_filters.append(("sites", a, k)) or site_q
    )
    # Site query result rows look like (Site.id,) — emulate that shape.
    site_q.all.return_value = [(sid,) for sid in (site_ids or [])]

    agg_q = MagicMock()
    agg_q.filter.side_effect = lambda *a, **k: (
        captured_filters.append(("agg", a, k)) or agg_q
    )
    agg_q.group_by.return_value = agg_q
    agg_q.all.return_value = agg_rows or []

    sess_q = MagicMock()
    sess_q.filter.side_effect = lambda *a, **k: (
        captured_filters.append(("sessions", a, k)) or sess_q
    )
    sess_q.scalar.return_value = distinct_sessions

    mocks_in_order = [site_q, agg_q, sess_q]
    call_index = {"i": 0}

    def _router(*_args, **_kwargs):
        i = call_index["i"]
        call_index["i"] += 1
        return mocks_in_order[i] if i < len(mocks_in_order) else MagicMock()

    fake_db.query.side_effect = _router

    app = FastAPI()
    app.include_router(analytics_router)

    def _override_ctx():
        return RequestContext(
            workspace_id=workspace_id,
            user=UserContext(id="test-user", email="test@example.com", role="owner"),
            auth_type="clerk",
        )

    def _override_db():
        yield fake_db

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db

    return TestClient(app), fake_db, captured_filters


# ---------------------------------------------------------------------------
# Tests (named exactly as US-004 AC requires)
# ---------------------------------------------------------------------------


def test_groups_by_event_type():
    """Endpoint groups widget_event_log rows by event_type with counts."""
    ws = uuid4()
    site_id = uuid4()
    rows = [
        _row("proactive_fired", 12),
        _row("callback_requested", 3),
        _row("cart_idle_fired", 5),
    ]
    client, _db, _captured = _make_client(
        workspace_id=ws,
        site_ids=[site_id],
        agg_rows=rows,
        distinct_sessions=7,
    )

    resp = client.get("/api/analytics/widget-engagement?window=7d")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["window"] == "7d"

    by_evt = {entry["event_type"]: entry for entry in body["by_event_type"]}
    assert set(by_evt) == {"proactive_fired", "callback_requested", "cart_idle_fired"}
    assert by_evt["proactive_fired"]["count"] == 12
    assert by_evt["callback_requested"]["count"] == 3
    assert by_evt["cart_idle_fired"]["count"] == 5

    assert body["sessions"] == 7

    assert "generated_at" in body
    datetime.fromisoformat(body["generated_at"])


def test_window_filtering():
    """The endpoint applies a >= filter on created_at sized to the window arg.

    Catches a regression where ``widget-engagement?window=7d`` would
    return all-time counts.
    """
    ws = uuid4()
    site_id = uuid4()
    captured: List[Tuple] = []

    client, _db, _ = _make_client(
        workspace_id=ws,
        site_ids=[site_id],
        agg_rows=[_row("proactive_fired", 1)],
        distinct_sessions=1,
        captured_filters=captured,
    )

    before = datetime.utcnow()
    resp = client.get("/api/analytics/widget-engagement?window=7d")
    after = datetime.utcnow()
    assert resp.status_code == 200, resp.text

    agg_filters = [c for c in captured if c[0] == "agg"]
    assert agg_filters, "endpoint must call .filter(...) on the aggregation query"
    _tag, args, _kwargs = agg_filters[0]

    created_expr = _filter_on(args, "created_at")
    assert created_expr is not None, (
        f"created_at window filter must be present on the aggregation query; "
        f"got {args}"
    )
    assert created_expr.operator is operator.ge, (
        f"created_at filter must use >= for the rolling window; "
        f"got {created_expr.operator!r}"
    )

    cutoff = created_expr.right.value
    expected_low = before - timedelta(days=7, seconds=5)
    expected_high = after - timedelta(days=7) + timedelta(seconds=5)
    assert expected_low <= cutoff <= expected_high, (
        f"window=7d cutoff must be ~now-7d; got {cutoff}, expected in "
        f"[{expected_low}, {expected_high}]"
    )


def test_workspace_site_scoping():
    """The endpoint scopes events to the caller's sites — not all sites globally.

    Two checks: (1) the site query filters by ``Site.workspace_id == ws``
    and (2) the aggregation query restricts ``WidgetEventLog.site_id`` to
    the resolved set, so another workspace's events cannot leak in.
    """
    my_ws = uuid4()
    my_site_a = uuid4()
    my_site_b = uuid4()
    captured: List[Tuple] = []

    client, _db, _ = _make_client(
        workspace_id=my_ws,
        site_ids=[my_site_a, my_site_b],
        agg_rows=[_row("proactive_fired", 1)],
        distinct_sessions=1,
        captured_filters=captured,
    )

    resp = client.get("/api/analytics/widget-engagement?window=7d")
    assert resp.status_code == 200, resp.text

    # (1) Sites query filtered by workspace_id == my_ws
    site_filters = [c for c in captured if c[0] == "sites"]
    assert site_filters, "endpoint must filter sites by workspace_id"
    _tag, sargs, _ = site_filters[0]
    ws_expr = _filter_on(sargs, "workspace_id")
    assert ws_expr is not None, (
        f"Site.workspace_id filter is required for tenant isolation; got {sargs}"
    )
    assert ws_expr.operator is operator.eq, (
        f"Site.workspace_id filter must be ==; got {ws_expr.operator!r}"
    )
    assert ws_expr.right.value == my_ws

    # (2) Aggregation query restricts to the caller's sites
    agg_filters = [c for c in captured if c[0] == "agg"]
    assert agg_filters, "endpoint must filter widget_event_log by site_id"
    _tag, aargs, _ = agg_filters[0]
    site_id_expr = _filter_on(aargs, "site_id")
    assert site_id_expr is not None, (
        f"WidgetEventLog.site_id IN (caller's sites) filter must be present; "
        f"got {aargs}"
    )
    # Expect an IN-clause: ``site_id.in_([...])``
    op_name = getattr(site_id_expr.operator, "__name__", "") or repr(
        site_id_expr.operator
    )
    assert "in_op" in op_name or "in" in op_name.lower(), (
        f"WidgetEventLog.site_id filter must be IN(...); got {op_name}"
    )


def test_empty_returns_zero():
    """No matching events in the window → empty list + sessions=0, no crash."""
    ws = uuid4()
    site_id = uuid4()
    client, _db, _ = _make_client(
        workspace_id=ws,
        site_ids=[site_id],
        agg_rows=[],
        distinct_sessions=0,
    )

    resp = client.get("/api/analytics/widget-engagement?window=7d")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["window"] == "7d"
    assert body["by_event_type"] == []
    assert body["sessions"] == 0
    assert "generated_at" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
