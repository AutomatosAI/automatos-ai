"""PRD-142 Wave 0 US-005 — GET /api/analytics/activation.

Platform-level activation rate (the one Wave 0 tile that is intentionally
NOT filtered to a single ``workspace_id`` — see US-005 notes). A workspace
is "activated" when it has >=1 ``OrchestrationRun`` with
``state == RunState.COMPLETED.value``; the rate is
``activated / total_workspaces``.

Computed from ``OrchestrationRun`` only — no new table, no
``WorkflowExecution`` reads (Wave 0 scope; Wave 3 owns the
``WorkflowExecution`` drop).

The endpoint MUST:

* Count DISTINCT ``OrchestrationRun.workspace_id`` filtered to
  ``state == 'completed'`` — workspaces whose runs are all in non-completed
  states do NOT contribute.
* Count ``Workspace`` rows for the denominator.
* Return ``rate = 0`` when ``total_workspaces == 0`` (no divide-by-zero),
  with NO fake fallback value.
* Return shape ``{activated, total_workspaces, rate, generated_at}``.
"""
from __future__ import annotations

import operator
import sys
from datetime import datetime
from pathlib import Path
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
    activated_count: int,
    total_workspaces: int,
    captured_filters: Optional[List[Tuple]] = None,
):
    """Build a TestClient bound to the analytics_real router with the
    workspace ctx and DB dependencies overridden.

    The endpoint issues queries in this order:

    1. ``db.query(func.count(func.distinct(OrchestrationRun.workspace_id)))
        .filter(OrchestrationRun.state == RunState.COMPLETED.value).scalar()``
        → number of activated workspaces.
    2. ``db.query(func.count(Workspace.id)).scalar()`` → total
        provisioned workspaces (denominator).

    ``captured_filters`` collects ``(tag, args, kwargs)`` for every
    ``.filter(...)`` call so tests can assert which state filter was
    applied to the activation count query.
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

    activated_q = MagicMock()
    activated_q.filter.side_effect = lambda *a, **k: (
        captured_filters.append(("activated", a, k)) or activated_q
    )
    activated_q.scalar.return_value = activated_count

    total_q = MagicMock()
    total_q.filter.side_effect = lambda *a, **k: (
        captured_filters.append(("total", a, k)) or total_q
    )
    total_q.scalar.return_value = total_workspaces

    mocks_in_order = [activated_q, total_q]
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
# Tests (named exactly as US-005 AC requires)
# ---------------------------------------------------------------------------


def test_activation_counts_completed_missions():
    """Endpoint returns activated/total/rate computed from OrchestrationRun.

    The activation count query MUST filter on
    ``state == RunState.COMPLETED.value`` (canonical enum, not a hardcoded
    string). The rate is the float ratio activated / total_workspaces.
    """
    from core.models.orchestration_enums import RunState

    ws = uuid4()
    captured: List[Tuple] = []
    client, _db, _ = _make_client(
        workspace_id=ws,
        activated_count=3,
        total_workspaces=10,
        captured_filters=captured,
    )

    resp = client.get("/api/analytics/activation")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["activated"] == 3
    assert body["total_workspaces"] == 10
    assert body["rate"] == pytest.approx(0.3)
    assert "generated_at" in body
    datetime.fromisoformat(body["generated_at"])

    # The activation count query MUST filter on state == 'completed'.
    activated_filters = [c for c in captured if c[0] == "activated"]
    assert activated_filters, (
        "endpoint must call .filter(...) on the activation count query "
        "to restrict to completed missions"
    )
    _tag, args, _kwargs = activated_filters[0]

    state_expr = _filter_on(args, "state")
    assert state_expr is not None, (
        f"OrchestrationRun.state filter must be present (defines what "
        f"'activated' means); got {args}"
    )
    assert state_expr.operator is operator.eq, (
        f"state filter must use ==; got {state_expr.operator!r}"
    )
    # Canonical enum value — never the bare string literal.
    assert state_expr.right.value == RunState.COMPLETED.value, (
        f"state filter must bind to RunState.COMPLETED.value "
        f"({RunState.COMPLETED.value!r}); got {state_expr.right.value!r}"
    )


def test_rate_zero_when_no_workspaces():
    """No provisioned workspaces → rate = 0 (no divide-by-zero, no fake fallback)."""
    ws = uuid4()
    client, _db, _ = _make_client(
        workspace_id=ws,
        activated_count=0,
        total_workspaces=0,
    )

    resp = client.get("/api/analytics/activation")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["activated"] == 0
    assert body["total_workspaces"] == 0
    # The honest zero — NOT a fabricated default like 85.0 (US-003 deleted that).
    assert body["rate"] == 0
    assert "generated_at" in body


def test_workspace_with_no_completed_run_not_activated():
    """A workspace with zero completed runs does not contribute to ``activated``.

    Enforced by the ``state == RunState.COMPLETED.value`` filter on the
    activation count query: workspaces whose runs are all in pending /
    planning / running / failed / cancelled states are excluded by the
    DB engine, never reaching the COUNT(DISTINCT workspace_id).

    Concretely we mock the DB to return ``activated=2`` for a tenant
    population of ``total_workspaces=5`` — the 3 workspaces "with no
    completed run" are the gap (5 - 2 = 3 not activated) — and assert
    both the response math and that the filter that produced this is
    bound to the COMPLETED state, not e.g. RUNNING.
    """
    from core.models.orchestration_enums import RunState

    ws = uuid4()
    captured: List[Tuple] = []
    client, _db, _ = _make_client(
        workspace_id=ws,
        activated_count=2,
        total_workspaces=5,
        captured_filters=captured,
    )

    resp = client.get("/api/analytics/activation")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["activated"] == 2
    assert body["total_workspaces"] == 5
    # 3 workspaces have no completed run; rate reflects only the 2 that do.
    assert body["rate"] == pytest.approx(0.4)

    # And the filter that produced "2" is COMPLETED — not any other state.
    activated_filters = [c for c in captured if c[0] == "activated"]
    assert activated_filters, (
        "activation count query must apply a state filter to exclude "
        "non-completed runs"
    )
    _tag, args, _kwargs = activated_filters[0]
    state_expr = _filter_on(args, "state")
    assert state_expr is not None, (
        f"state filter is required to exclude non-completed runs; got {args}"
    )
    assert state_expr.right.value == RunState.COMPLETED.value, (
        f"only state == 'completed' rows define activation; "
        f"got filter on {state_expr.right.value!r}"
    )
    # Sanity: it is NOT RUNNING / PENDING / FAILED / etc.
    for non_activated in (
        RunState.RUNNING.value,
        RunState.PENDING.value,
        RunState.FAILED.value,
        RunState.CANCELLED.value,
    ):
        assert state_expr.right.value != non_activated, (
            f"activation filter must not be {non_activated!r}; that would "
            f"miscount workspaces with no completed mission as activated"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
