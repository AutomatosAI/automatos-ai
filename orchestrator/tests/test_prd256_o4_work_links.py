"""PRD-256 O4 (#847): background work is a trace of its own, linked to the request that asked for it.

A ticket or Mission inserted while a span is current is stamped with its ``traceparent``
(``planning_data`` / ``config``, ``trace_links``); Run Now and plan approval add theirs. The
board dispatcher and the coordinator run the work carrying those links, and the agent run's
``invoke_agent`` span (every run, whatever the lane) starts with them as span links: a root
in the background, the request's child inside one. Off: nothing is stamped, imported or
linked. Spans go to an in-memory exporter, through the app's own request-rooted sampler.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.observability import genai, otel, work_links  # noqa: E402
from tests.helpers_otel import fresh_global_providers  # noqa: E402

SEEDED = "00-11111111111111111111111111111111-2222222222222222-01"


class _Factory:
    """``AgentFactory``'s shape where the span reads it: active agents, and the decorated run."""

    def __init__(self, result=None):
        self.active_agents = {7: SimpleNamespace(agent_id=7, metadata=SimpleNamespace(name="Auto"),
                                                 workspace_id="ws-1")}
        self.result = result or {"status": "success", "response": "done"}

    @genai.traced_agent_run
    async def execute_with_prompt(self, agent, prompt, system_prompt=None, context=None):
        return self.result


# ── off ──────────────────────────────────────────────────────────────────────

def test_off_nothing_is_stamped_imported_or_linked():
    script = (
        "import asyncio, sys\n"
        "from core.observability import work_links as w, genai\n"
        "data = {'brief': 1}\n"
        "assert w.current_traceparent() is None and w.with_link(data) is data and w.with_link(None) is None\n"
        "class F:\n"
        "    active_agents = {}\n"
        "    @genai.traced_agent_run\n"
        "    async def execute_with_prompt(self, agent, prompt, system_prompt=None, context=None):\n"
        "        return 'ran'\n"
        "async def go():\n"
        "    return await w.run_linked(F().execute_with_prompt(7, 'p'), ['" + SEEDED + "'])\n"
        "assert asyncio.run(go()) == 'ran'\n"
        "loaded = [m for m in sys.modules if m.startswith('opentelemetry')]\n"
        "assert not loaded, loaded\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OTEL_ENABLED"}
    done = subprocess.run([sys.executable, "-c", script], cwd=_ORCH, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


# ── on ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def spans(monkeypatch):
    """Tracing on: a fresh global provider built as the app builds it, into an in-memory exporter."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    exporter = InMemorySpanExporter()
    provider = otel.build_provider(exporter, 1.0)

    def finished():
        provider.force_flush()
        return exporter.get_finished_spans()

    with fresh_global_providers():
        trace.set_tracer_provider(provider)
        yield finished
        provider.shutdown()


def _request():
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind

    return trace.get_tracer("t").start_as_current_span("POST /api/board-tasks", kind=SpanKind.SERVER)


def _traceparent_of(span):
    context = span.get_span_context()
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"


def test_a_ticket_or_mission_asked_for_in_a_request_is_stamped_with_it(spans):
    ticket = SimpleNamespace(planning_data={"brief": "x", "trace_links": [SEEDED]})  # a caller's own: dropped
    mission = SimpleNamespace(config=None)
    with _request() as request:
        work_links._stamp_ticket(None, None, ticket)
        work_links._stamp_mission(None, None, mission)
    assert ticket.planning_data == {"brief": "x", "trace_links": [_traceparent_of(request)]}
    assert mission.config == {"trace_links": [_traceparent_of(request)]}


def test_work_filed_outside_any_span_is_left_as_it_was_or_carries_its_runs_links(spans):
    plain = SimpleNamespace(planning_data=None)
    work_links._stamp_ticket(None, None, plain)
    assert plain.planning_data is None
    filed_by_a_step = SimpleNamespace(planning_data=None)
    with work_links.linked([SEEDED]):     # a Mission's Claude Code step files its ticket
        work_links._stamp_ticket(None, None, filed_by_a_step)
    assert filed_by_a_step.planning_data == {"trace_links": [SEEDED]}


def test_a_rerun_adds_its_request_and_keeps_the_latest_few(spans):
    data = {"trace_links": [f"00-{i:032x}-{i:016x}-01" for i in range(1, 9)]}
    with _request() as rerun:
        linked = work_links.with_link(data)
    assert len(linked["trace_links"]) == work_links.MAX_LINKS
    assert linked["trace_links"][-1] == _traceparent_of(rerun) and data["trace_links"][0] not in linked["trace_links"]
    assert work_links.links_of({"trace_links": ["not-a-traceparent", SEEDED, 3]}) == (SEEDED,)


def test_the_listeners_are_registered_once(spans):
    from sqlalchemy import event

    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun

    work_links.register_work_links()
    work_links.register_work_links()
    assert event.contains(BoardTask, "before_insert", work_links._stamp_ticket)
    assert event.contains(OrchestrationRun, "before_insert", work_links._stamp_mission)


def test_a_background_run_is_a_root_linked_to_the_request_that_asked(spans):
    with _request() as request:
        asked = work_links.current_traceparent()

    async def dispatched():        # the board dispatcher: no span, the ticket's links
        with work_links.linked([asked]):
            return await asyncio.create_task(
                _Factory().execute_with_prompt(7, "brief", context={"source": "board_task", "task_id": 97}))

    assert asyncio.run(dispatched())["status"] == "success"
    [run] = [s for s in spans() if s.name.startswith("invoke_agent")]
    assert run.name == "invoke_agent Auto" and run.parent is None
    assert run.context.trace_id != request.get_span_context().trace_id
    [link] = run.links
    assert link.context.span_id == request.get_span_context().span_id and link.attributes["automatos.link"] == "requested_by"
    assert dict(run.attributes) == {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.id": "7",
                                    "gen_ai.agent.name": "Auto", "automatos.lane": "board_task",
                                    "automatos.execution_id": "board_task:97", "automatos.workspace_id": "ws-1"}


def test_a_run_inside_a_request_is_its_child_and_unlinked(spans):
    with _request() as request:
        asyncio.run(_Factory().execute_with_prompt(7, "hi", context={"source": "chat"}))
    [run] = [s for s in spans() if s.name.startswith("invoke_agent")]
    assert run.parent.span_id == request.get_span_context().span_id and run.links == ()


def test_mission_steps_each_carry_their_own_runs_links_and_the_tick_carries_none(spans):
    first, second = SEEDED, "00-33333333333333333333333333333333-4444444444444444-01"

    async def tick():
        factory = _Factory()
        await asyncio.gather(
            work_links.run_linked(factory.execute_with_prompt(7, "a", context={"run_id": "r1"}), [first]),
            work_links.run_linked(factory.execute_with_prompt(7, "b", context={"run_id": "r2"}), [second]))
        return work_links.pending_links()

    assert asyncio.run(tick()) == ()
    linked = {s.attributes["automatos.execution_id"]: f"{s.links[0].context.trace_id:032x}" for s in spans()}
    assert linked == {"mission:r1": "1" * 32, "mission:r2": "3" * 32}


def test_a_failed_run_is_an_error_span(spans):
    from opentelemetry.trace import StatusCode

    with _request():
        asyncio.run(_Factory(result={"status": "error", "error": "no model"}).execute_with_prompt(7, "p"))
    [run] = [s for s in spans() if s.name.startswith("invoke_agent")]
    assert run.status.status_code == StatusCode.ERROR and run.attributes["error.type"] == "agent_error"


def test_the_dispatcher_launches_a_ticket_carrying_its_links(spans, monkeypatch):
    import api.board_tasks as board_tasks
    from services import board_dispatcher

    launched = []

    def launch(**kwargs):
        launched.append(work_links.pending_links())

    monkeypatch.setattr(board_tasks, "_launch_task_execution", launch)
    board_dispatcher._launch_one({"task_id": 97, "agent_id": 7, "workspace_id": "ws-1", "prompt": "p",
                                  "review_mode": "auto", "attachment_ids": [], "trace_links": (SEEDED,)})
    assert launched == [(SEEDED,)] and work_links.pending_links() == ()


def test_a_creator_cannot_seed_a_missions_links():
    from services.coordinator_service import SERVER_OWNED_MISSION_CONFIG, _creator_config

    assert "trace_links" in SERVER_OWNED_MISSION_CONFIG
    assert "trace_links" not in _creator_config({"trace_links": [SEEDED], "power_mode": "standard"})


def test_the_work_items_and_loops_are_wired():
    from modules.agents.factory.agent_factory import AgentFactory

    assert AgentFactory.execute_with_prompt.__wrapped__.__name__ == "execute_with_prompt"
    source = {name: (_ORCH / name).read_text(encoding="utf-8") for name in (
        "services/board_dispatcher.py", "services/coordinator_service.py", "api/board_tasks.py",
        "core/observability/otel.py")}
    assert '"trace_links": links_of(getattr(t, "planning_data", None))' in source["services/board_dispatcher.py"]
    assert "run_linked(self._task_io(p), links_of(run.config))" in source["services/coordinator_service.py"]
    assert "run.config = with_link(run.config)" in source["services/coordinator_service.py"]
    assert 'task.planning_data = with_link(getattr(task, "planning_data", None))' in source["api/board_tasks.py"]
    assert '("work items", _instrument_work_items)' in source["core/observability/otel.py"]
