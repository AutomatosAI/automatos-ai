"""
PRD-141 US-009: Report iteration/budget limits to the user.
============================================================

Two independent user-facing signals are proven here:

1. Chatbot side — when ``_run_tool_loop`` exhausts ``CHATBOT_MAX_TOOL_ITERATIONS``
   it now emits a ``limit_reached`` SSE frame so the user is told the agent
   stopped on a cap (instead of silently answering). We assert the formatter
   produces the AI SDK data envelope the stream consumer expects.

2. Coordinator side — when a mission crosses 1.5x its token budget,
   ``_record_task_result`` emits a ``BUDGET_WARNING`` carrying the user-facing
   ``limit_type/spent/limit/message`` fields (the mission keeps running; this
   is a purely additive signal, no pause).
"""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from core.models.orchestration_enums import EventType, TaskState
from services.coordinator_service import CoordinatorService


def _load_streaming_handler_class():
    """Load StreamingHandler from the leaf module file directly.

    Importing ``consumers.chatbot.streaming`` the normal way runs
    ``consumers/chatbot/__init__.py``, which eagerly imports the full chatbot
    service (camelot/PDF + DB deps) — none of which the formatter needs. The
    module file itself only uses stdlib, so we load it in isolation.
    """
    streaming_path = _orchestrator_root / "consumers" / "chatbot" / "streaming.py"
    spec = importlib.util.spec_from_file_location(
        "_us009_streaming", streaming_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StreamingHandler


StreamingHandler = _load_streaming_handler_class()


# ---------------------------------------------------------------------------
# 1. Chatbot iteration-limit reporting
# ---------------------------------------------------------------------------

def _parse_aisdk_data_frame(frame: str) -> dict:
    """Strip the ``d:`` prefix + trailing newline and JSON-decode the payload."""
    assert frame.startswith("d:"), f"expected AI SDK data frame, got: {frame!r}"
    assert frame.endswith("\n"), f"AI SDK frame must end with newline, got: {frame!r}"
    return json.loads(frame[2:].strip())


def test_iteration_limit_reports_to_user():
    """The max-iterations cap surfaces as a typed limit_reached SSE frame."""
    handler = StreamingHandler()
    message = (
        "I reached the maximum of 10 tool steps for a single response, so I'm "
        "answering with what I have so far."
    )

    frame = handler.format_aisdk_limit_reached(
        limit="max_tool_iterations",
        value=10,
        message=message,
    )

    payload = _parse_aisdk_data_frame(frame)
    assert payload["type"] == "limit_reached"
    data = payload["data"]
    assert data["limit"] == "max_tool_iterations"
    assert data["value"] == 10
    assert data["message"] == message


def test_iteration_limit_frame_is_stream_safe_string():
    """Consumer feeds a text/plain StreamingResponse — frame must be a str."""
    handler = StreamingHandler()
    frame = handler.format_aisdk_limit_reached(
        limit="max_tool_iterations", value=50, message="hit the cap",
    )
    assert isinstance(frame, str)


# ---------------------------------------------------------------------------
# 2. Coordinator budget-overage reporting
# ---------------------------------------------------------------------------

def _make_run(*, token_budget_estimate, tokens_used=0):
    run = MagicMock()
    run.id = "run-123"
    run.token_budget_estimate = token_budget_estimate
    run.tokens_used = tokens_used  # real int so the += arithmetic works
    return run


def _make_task():
    task = MagicMock()
    task.id = "task-1"
    task.title = "Some task"
    task.state = TaskState.RUNNING.value  # anything that is NOT FAILED
    return task


async def _call_record_task_result(run, result):
    """Invoke the unbound method with a mock ``self`` + targeted patches.

    ``_record_task_result`` does heavy work before the budget block
    (record_task_completion, mission-event dispatch, field injection); we
    patch those out so the test isolates the budget-warning emit.
    """
    mock_self = MagicMock()
    mock_self._inject_task_output_into_field = AsyncMock()

    db = MagicMock()
    task = _make_task()

    with patch("services.coordinator_service.MissionDispatcher"), \
         patch("services.coordinator_service._dispatch_mission_event", new=AsyncMock()), \
         patch("services.coordinator_service.emit_event") as mock_emit:
        await CoordinatorService._record_task_result(
            mock_self, db, run, task, agent_id=1, result=result,
        )
    return mock_emit


def _budget_warnings(mock_emit):
    return [
        c for c in mock_emit.call_args_list
        if c.kwargs.get("event_type") == EventType.BUDGET_WARNING
    ]


@pytest.mark.asyncio
async def test_budget_exceeded_emits_event():
    """Crossing 1.5x budget emits BUDGET_WARNING with user-facing fields."""
    run = _make_run(token_budget_estimate=1000, tokens_used=0)
    result = {"status": "success", "execution": {"tokens_used": 2000}}

    mock_emit = await _call_record_task_result(run, result)

    warnings = _budget_warnings(mock_emit)
    assert len(warnings) == 1, "expected exactly one BUDGET_WARNING"

    payload = warnings[0].kwargs["payload"]
    # User-facing fields (the US-009 increment)
    assert payload["limit_type"] == "mission_token_budget"
    assert payload["spent"] == 2000
    assert payload["limit"] == 1000
    assert "tokens" in payload["message"].lower()
    # Established diagnostic fields retained
    assert payload["tokens_used"] == 2000
    assert payload["token_budget_estimate"] == 1000
    assert payload["ratio"] == 2.0


@pytest.mark.asyncio
async def test_under_budget_does_not_emit_warning():
    """A task that stays under 1.5x budget emits no BUDGET_WARNING."""
    run = _make_run(token_budget_estimate=10000, tokens_used=0)
    result = {"status": "success", "execution": {"tokens_used": 2000}}

    mock_emit = await _call_record_task_result(run, result)

    assert _budget_warnings(mock_emit) == []


@pytest.mark.asyncio
async def test_no_budget_estimate_does_not_emit_warning():
    """A mission with no token_budget_estimate never warns."""
    run = _make_run(token_budget_estimate=None, tokens_used=0)
    result = {"status": "success", "execution": {"tokens_used": 9999}}

    mock_emit = await _call_record_task_result(run, result)

    assert _budget_warnings(mock_emit) == []
