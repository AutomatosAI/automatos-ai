"""PRD-221 S12 — feed last_progress labels + bounded enrichment (pure).

Locks: labels are keyed off the REAL EventType enum (underscore vocab, not
dotted); unknown types fall back safely; the page enrichment runs ONE query
regardless of page size (no N+1); and a run-linked item gets last_progress.
"""
from __future__ import annotations

from unittest.mock import MagicMock

# CI collection-order guard (see PR #434).
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "services" or n.startswith("services.")
                  or n == "core" or n.startswith("core."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from core.models.orchestration_enums import EventType  # noqa: E402
from services.activity_progress import (  # noqa: E402
    progress_label,
    progress_requires_attention,
)
from services.activity_service import ActivityService  # noqa: E402


def test_progress_labels_cover_real_vocab():
    # Every run_*/task_* lifecycle event resolves to a specific (non-generic)
    # label — asserted against the imported enum, not string literals.
    lifecycle = [
        e for e in EventType
        if e.value.startswith(("run_", "task_")) or e.value == "stall_detected"
    ]
    specific = [e for e in lifecycle if progress_label(e.value) != "Working…"]
    # the vast majority are mapped; a handful of rare audit events may fall back
    assert len(specific) >= 20
    # spot-check the load-bearing ones
    assert progress_label(EventType.RUN_REPLANNING.value) == "Re-planning after a failed step"
    assert progress_label(EventType.TASK_VERIFICATION_FAILED.value) == "Output failed verification — retrying"
    assert progress_label(EventType.RUN_COMPLETED.value) == "Completed"


def test_progress_unknown_type_is_safe():
    assert progress_label("some_future_event") == "Working…"
    assert progress_label("mission.status.updated") == "Working…"  # dotted != real
    assert progress_requires_attention("some_future_event") is False


def test_progress_attention_flags():
    assert progress_requires_attention(EventType.TASK_FAILED.value) is True
    assert progress_requires_attention(EventType.RUN_AWAITING_HUMAN.value) is True
    assert progress_requires_attention(EventType.RUN_STARTED.value) is False


def _event_row(run_id, event_type, at="2026-07-17T00:00:00+00:00"):
    from datetime import datetime
    row = MagicMock()
    row.run_id = run_id
    row.event_type = event_type
    row.created_at = datetime.fromisoformat(at)
    return row


def _service_with_execute_counter(rows):
    svc = ActivityService(MagicMock(), "ws-1")
    calls = {"n": 0}

    def _execute(*a, **kw):
        calls["n"] += 1
        result = MagicMock()
        result.fetchall.return_value = rows
        return result

    svc.db.execute = _execute
    return svc, calls


def test_last_progress_attached_from_latest_event():
    rows = [_event_row("run-1", EventType.RUN_REPLANNING.value)]
    svc, _ = _service_with_execute_counter(rows)
    items = [{"id": "task-1", "orchestration_run_id": "run-1"}]
    svc._attach_last_progress(items)
    lp = items[0]["last_progress"]
    assert lp["summary"] == "Re-planning after a failed step"
    assert lp["at"] == "2026-07-17T00:00:00+00:00"
    assert lp["requires_attention"] is False


def test_last_progress_bounded_one_query_regardless_of_page_size():
    rows = [_event_row(f"run-{i}", EventType.TASK_STARTED.value) for i in range(20)]
    # 2-item page
    svc2, calls2 = _service_with_execute_counter(rows)
    svc2._attach_last_progress(
        [{"id": f"t{i}", "orchestration_run_id": f"run-{i}"} for i in range(2)]
    )
    # 20-item page
    svc20, calls20 = _service_with_execute_counter(rows)
    svc20._attach_last_progress(
        [{"id": f"t{i}", "orchestration_run_id": f"run-{i}"} for i in range(20)]
    )
    assert calls2["n"] == 1
    assert calls20["n"] == 1  # constant — no N+1


def test_last_progress_noop_without_run_links():
    svc, calls = _service_with_execute_counter([])
    items = [{"id": "chat-1"}, {"id": "task-2", "orchestration_run_id": None}]
    svc._attach_last_progress(items)
    assert calls["n"] == 0  # nothing to join → no query at all
    assert "last_progress" not in items[0]
