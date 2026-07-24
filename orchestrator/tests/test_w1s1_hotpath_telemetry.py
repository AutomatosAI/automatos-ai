"""PRD-142 Wave 1 · WS-A · W1-S1 — record_error adoption at failure hot-paths.

Wave 0 shipped the ``error_events`` sink + ``record_error`` (persistence is
covered by ``test_error_events_sink.py``). The ERRORS-by-subsystem dashboard
tile was empty only because real failure paths never *called* ``record_error``.

These tests prove the *adoption*: each test drives a subsystem's hot-path to
its terminal failure and asserts ``record_error`` fires with the correct
``subsystem``/``operation``. They patch the module-level ``record_error`` with
a stub (so no DB is touched) and patch the one dependency whose failure the
hot-path exists to handle.

Subsystems covered here (Tier 1, clean seams): planner, verification, widget.
board + wizard live in ``test_w1s1_hotpath_telemetry_io.py`` (heavier mocking).
"""
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# These unit tests mock the DB/session entirely (no query ever runs), but
# importing the coordination/widget modules eagerly constructs the SQLAlchemy
# engine, which refuses to build without POSTGRES_* creds. Provide inert
# placeholders so the engine constructs lazily; we never connect. setdefault
# means a real .env (when present) still wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


def _raising_llm():
    """An LLMManager whose generate_response always raises — drives the retry
    loops in planner/verification straight to their terminal failure."""
    llm = MagicMock()
    llm.generate_response = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    return llm


def _active_agent(name: str = "Researcher"):
    """A roster agent shaped for ``_render_agent_roster`` (needs a real str
    name/description and status == 'active'; everything else is guarded)."""
    a = MagicMock()
    a.status = "active"
    a.name = name
    a.description = "Does research and analysis"
    a.model_config = {}
    a.tags = []
    a.skills = []
    return a


# ---------------------------------------------------------------------------
# planner (subsystem="planner")
# ---------------------------------------------------------------------------

def test_decompose_emits_planner_error(monkeypatch):
    from modules.coordination import planner as planner_mod

    rec = MagicMock()
    monkeypatch.setattr(planner_mod, "record_error", rec, raising=False)
    monkeypatch.setattr(planner_mod, "create_llm_manager", lambda **kw: _raising_llm())

    ws = uuid4()
    with pytest.raises(planner_mod.PlanValidationError):
        asyncio.run(
            planner_mod.MissionPlanner.decompose(
                goal="Build a go-to-market plan",
                workspace_id=ws,
                agents=[_active_agent()],
            )
        )

    assert rec.call_count >= 1, "record_error never called from decompose"
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "planner"
    assert kw["operation"] == "decompose"
    assert kw["workspace_id"] == ws


def test_replan_emits_planner_error(monkeypatch):
    from modules.coordination import planner as planner_mod

    rec = MagicMock()
    monkeypatch.setattr(planner_mod, "record_error", rec, raising=False)
    monkeypatch.setattr(planner_mod, "create_llm_manager", lambda **kw: _raising_llm())

    ws = uuid4()
    with pytest.raises(planner_mod.PlanValidationError):
        asyncio.run(
            planner_mod.MissionPlanner.replan(
                goal="Build a go-to-market plan",
                workspace_id=ws,
                agents=[_active_agent()],
                completed_outputs=[],
                failed_task_title="Draft positioning",
                failed_task_reason="ran out of budget",
            )
        )

    assert rec.call_count >= 1, "record_error never called from replan"
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "planner"
    assert kw["operation"] == "replan"
    assert kw["workspace_id"] == ws


# ---------------------------------------------------------------------------
# verification (subsystem="verification") — fails open, so no raise
# ---------------------------------------------------------------------------

def test_verify_task_emits_verification_error(monkeypatch):
    from modules.coordination import verification as verif_mod

    rec = MagicMock()
    monkeypatch.setattr(verif_mod, "record_error", rec, raising=False)
    monkeypatch.setattr(verif_mod, "create_llm_manager", lambda **kw: _raising_llm())

    svc = verif_mod.VerificationService()
    result = asyncio.run(
        svc.verify_task(
            task_title="Write the summary",
            task_description="Summarize the quarterly report",
            output=(
                "This is a sufficiently long task output paragraph so the "
                "deterministic checker does not short-circuit before the LLM "
                "judge runs and exhausts its retries."
            ),
            verification_criteria=[],
        )
    )

    # Verification is advisory: a judge failure degrades to PARTIAL, never raises.
    assert result.verdict == verif_mod.VERDICT_PARTIAL
    assert rec.call_count >= 1, "record_error never called from verify_task"
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "verification"
    assert kw["operation"] == "verify_task"


def test_cross_task_consistency_emits_verification_error(monkeypatch):
    from modules.coordination import verification as verif_mod

    rec = MagicMock()
    monkeypatch.setattr(verif_mod, "record_error", rec, raising=False)
    monkeypatch.setattr(verif_mod, "create_llm_manager", lambda **kw: _raising_llm())

    svc = verif_mod.VerificationService()
    run_id = uuid4()
    result = asyncio.run(
        svc.verify_cross_task_consistency(
            run_id=run_id,
            goal="Build a go-to-market plan",
            task_outputs=[
                {"task_id": "1", "title": "Positioning", "output": "Output A"},
                {"task_id": "2", "title": "Pricing", "output": "Output B"},
            ],
        )
    )

    # Consistency also fails open (defaults to passed) — must still record.
    assert result.passed is True
    assert rec.call_count >= 1, "record_error never called from consistency check"
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "verification"
    assert kw["operation"] == "cross_task_consistency"


# ---------------------------------------------------------------------------
# widget (subsystem="widget") — code-path failure, not an exception
# ---------------------------------------------------------------------------

def test_widget_permanent_failure_emits_widget_error(monkeypatch):
    from services.destinations import dispatcher as disp_mod

    rec = MagicMock()
    monkeypatch.setattr(disp_mod, "record_error", rec, raising=False)
    # log_widget_event is async + writes the DB — stub it out.
    monkeypatch.setattr(disp_mod, "log_widget_event", AsyncMock())

    fail = disp_mod.DispatchResult(
        success=False,
        destination_type="telegram",
        latency_ms=5,
        error="bot token invalid",
        retryable=False,  # permanent → terminal on attempt 1
    )
    monkeypatch.setattr(disp_mod, "dispatch_via_channel", AsyncMock(return_value=fail))

    ws = uuid4()
    result = asyncio.run(
        disp_mod.dispatch_one_destination(
            db=MagicMock(),
            site_id=uuid4(),
            workspace_id=ws,
            session_id="sess-1",
            request_id="req-1",
            destination={"platform": "telegram"},
            payload=MagicMock(),
        )
    )

    assert result.success is False
    assert rec.call_count >= 1, "record_error never called on terminal widget failure"
    kw = rec.call_args.kwargs
    assert kw["subsystem"] == "widget"
    assert kw["operation"] == "deliver_callback"
    assert kw["workspace_id"] == ws


def test_widget_success_does_not_emit(monkeypatch):
    """Guardrail: a successful delivery must NOT record an error."""
    from services.destinations import dispatcher as disp_mod

    rec = MagicMock()
    monkeypatch.setattr(disp_mod, "record_error", rec, raising=False)
    monkeypatch.setattr(disp_mod, "log_widget_event", AsyncMock())

    ok = disp_mod.DispatchResult(
        success=True,
        destination_type="telegram",
        latency_ms=5,
        error=None,
        retryable=False,
    )
    monkeypatch.setattr(disp_mod, "dispatch_via_channel", AsyncMock(return_value=ok))

    result = asyncio.run(
        disp_mod.dispatch_one_destination(
            db=MagicMock(),
            site_id=uuid4(),
            workspace_id=uuid4(),
            session_id="sess-1",
            request_id="req-1",
            destination={"platform": "telegram"},
            payload=MagicMock(),
        )
    )

    assert result.success is True
    rec.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
