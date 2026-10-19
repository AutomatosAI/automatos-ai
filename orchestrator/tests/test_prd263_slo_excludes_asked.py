"""PRD-263 US-018 (§11-Q8) — the SLO success rate leaves asks out of its denominator.

``services.slo_metrics.tool_call_success_rate`` divided the production successes by every
production row, whatever its status, so a wait for the owner's click (now logged as
``asked``) counted exactly as an error did. At §11-Q8's default an asked row is in neither
count: a wait for a click is not a failure. ``value`` is a percentage, as the SLO's
target (99.0) is.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from core.models.composio_cache import ToolExecutionLog
from services import slo_metrics


def _runs(db, ws, status, n):
    for _ in range(n):
        db.add(ToolExecutionLog(workspace_id=UUID(ws), app_name="PLATFORM", action_name="platform_update_agent",
                                status=status, telemetry_source="production",
                                executed_at=datetime.utcnow() - timedelta(minutes=5)))
    db.flush()


def test_8_successes_and_2_asks_read_as_a_full_success_rate(db_session, seed_workspace):
    ws = seed_workspace(name="slo-asked")
    _runs(db_session, ws, "success", 8)
    _runs(db_session, ws, slo_metrics.ASKED_STATUS, 2)

    sli = slo_metrics.tool_call_success_rate(db_session, workspace_id=ws)

    assert (sli["value"], sli["sample_size"], sli["meets_target"]) == (100.0, 8, True)


def test_errors_still_count_against_the_rate(db_session, seed_workspace):
    ws = seed_workspace(name="slo-asked-errors")
    _runs(db_session, ws, "success", 8)
    _runs(db_session, ws, "error", 2)
    _runs(db_session, ws, slo_metrics.ASKED_STATUS, 5)

    sli = slo_metrics.tool_call_success_rate(db_session, workspace_id=ws)

    assert (sli["value"], sli["sample_size"]) == (80.0, 10)


def test_a_window_of_asks_alone_has_no_rate_to_judge(db_session, seed_workspace):
    ws = seed_workspace(name="slo-only-asks")
    _runs(db_session, ws, slo_metrics.ASKED_STATUS, 3)

    sli = slo_metrics.tool_call_success_rate(db_session, workspace_id=ws)

    assert (sli["value"], sli["sample_size"], sli["meets_target"]) == (None, 0, None)
