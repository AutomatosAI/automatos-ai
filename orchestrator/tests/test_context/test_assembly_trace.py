"""PRD-201 S1 — persist the assembly trace (answerability).

Pure — no live collector, no DB. Asserts the durable trace *shape* the assembler
hands back, and that the tracer seam emits a span only when tracing is ON while
the trace record itself is produced regardless.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_ORCH = Path(__file__).resolve().parent.parent.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.context.budget import TokenBudget, TokenBudgetManager
from modules.context.modes import ContextMode
from modules.context.result import ContextResult
from modules.context.service import ContextService, _Assembly
import core.observability.tracer as tracer_mod


def _result() -> ContextResult:
    return ContextResult(
        system_prompt="identity\n\nmemory",
        mode="chatbot",
        sections_included=["identity", "memory"],
        sections_trimmed=["business_graph"],
        token_estimate=1234,
        token_budget=118000,
        preparation_time_ms=42.36,
        model="claude-opus-4-8",
        budget_total=128000,
        sections=[
            {"name": "identity", "priority": 1, "token_estimate": 200, "rendered_nonempty": True, "trimmed": False},
            {"name": "business_graph", "priority": 5, "token_estimate": 900, "rendered_nonempty": True, "trimmed": True},
        ],
        injected_memory_ids=["mem_1", "mem_2"],
    )


# --- the durable trace shape (written regardless of TRACING_ENABLED) ---


def test_to_assembly_trace_carries_section_token_trim_fields():
    trace = _result().to_assembly_trace()
    assert trace["mode"] == "chatbot"
    assert trace["model"] == "claude-opus-4-8"
    assert trace["budget_total"] == 128000
    assert trace["token_estimate"] == 1234
    assert trace["prep_ms"] == 42.4  # rounded
    assert trace["sections_trimmed"] == ["business_graph"]
    assert trace["injected_memory_ids"] == ["mem_1", "mem_2"]
    # Per-section detail the assembler used to throw away.
    biz = next(s for s in trace["sections"] if s["name"] == "business_graph")
    assert biz["priority"] == 5
    assert biz["trimmed"] is True
    assert biz["rendered_nonempty"] is True
    assert biz["token_estimate"] == 900


def test_trace_is_json_serialisable():
    import json

    # Must survive a JSONB round-trip (it is persisted on messages.context_trace).
    assert json.loads(json.dumps(_result().to_assembly_trace()))["mode"] == "chatbot"


# --- the observability seam: span only when tracing ON, never raises ---


def test_assembly_trace_persisted_is_noop_safe_when_tracing_off():
    tracer_mod.reset_tracer()
    with patch.object(tracer_mod, "get_tracer", return_value=tracer_mod.NoOpTracer()):
        # NoOp path: fires cleanly, emits nothing, never raises (tracing OFF).
        tracer_mod.fire_assembly_trace(trace=_result().to_assembly_trace(), workspace_id="ws")
    assert tracer_mod.NoOpTracer().trace_assembly(trace={}) is None


def test_trace_span_emitted_when_tracing_on():
    mock_tracer = MagicMock()
    with patch.object(tracer_mod, "get_tracer", return_value=mock_tracer):
        tracer_mod.fire_assembly_trace(
            trace=_result().to_assembly_trace(), workspace_id="ws", metadata={"agent_id": 42}
        )
    assert mock_tracer.trace_assembly.call_count == 1
    kwargs = mock_tracer.trace_assembly.call_args.kwargs
    assert kwargs["trace"]["mode"] == "chatbot"
    assert kwargs["workspace_id"] == "ws"


def test_fire_assembly_trace_swallows_tracer_faults():
    boom = MagicMock()
    boom.trace_assembly.side_effect = RuntimeError("collector down")
    with patch.object(tracer_mod, "get_tracer", return_value=boom):
        # Must not raise into the build.
        tracer_mod.fire_assembly_trace(trace={"mode": "chatbot"}, workspace_id="ws")


# --- voice-latency D-5: each section's render time on the chat trace ---

_SLOW_RENDER_S = 0.06  # a little over the 50 ms the assertions ask for
_SLOW_RENDER_FLOOR_MS = 50


class _FakeSection:
    """A section as ``_render_sections`` reads one: name, priority, max_tokens, render()."""

    max_tokens = None

    def __init__(self, name: str, priority: int, content: str = "", wait_s: float = 0.0, fails: bool = False):
        self.name = name
        self.priority = priority
        self._content = content
        self._wait_s = wait_s
        self._fails = fails

    async def render(self, ctx):
        if self._wait_s:
            await asyncio.sleep(self._wait_s)
        if self._fails:
            raise RuntimeError("section source down")
        return self._content


def _is_a_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _chat_trace_sections(rendered, included, trimmed_names, budget) -> dict:
    """The ``sections`` of the trace ``_result`` builds, keyed by section name."""
    assembly = _Assembly(
        rendered=rendered, included=included, trimmed_names=trimmed_names, budget=budget,
        system_prompt="identity", cacheable_prefix=None, prompt_tokens=1, tools=[], tool_choice="auto",
    )
    result = ContextService._result(
        ContextMode.CHATBOT, SimpleNamespace(kwargs={}), assembly, [], 1, 1.0, "claude-opus-4-8", [],
    )
    return {s["name"]: s for s in result.to_assembly_trace()["sections"]}


@pytest.mark.asyncio
async def test_each_section_carries_its_own_render_time():
    sections = [
        _FakeSection("identity", 1, "You are Auto."),
        _FakeSection("memory", 5, "What you know.", wait_s=_SLOW_RENDER_S),
    ]
    rendered = await ContextService._render_sections(sections, SimpleNamespace(kwargs={}))
    by_name = {s.name: s for s in rendered}

    assert _is_a_number(by_name["identity"].render_ms)
    assert _is_a_number(by_name["memory"].render_ms)
    assert by_name["memory"].render_ms >= _SLOW_RENDER_FLOOR_MS
    assert 0 <= by_name["identity"].render_ms < by_name["memory"].render_ms

    budget = TokenBudget(total=100_000, reserved_for_response=0, reserved_for_messages=0)
    trace = _chat_trace_sections(rendered, rendered, [], budget)
    assert trace["memory"]["render_ms"] == by_name["memory"].render_ms
    assert trace["identity"]["render_ms"] == by_name["identity"].render_ms
    assert all(_is_a_number(s["render_ms"]) for s in trace.values())


@pytest.mark.asyncio
async def test_a_section_that_raises_still_has_a_render_time():
    sections = [_FakeSection("business_graph", 5, wait_s=_SLOW_RENDER_S, fails=True)]
    (failed,) = await ContextService._render_sections(sections, SimpleNamespace(kwargs={}))

    assert failed.content == ""
    assert _is_a_number(failed.render_ms)
    assert failed.render_ms >= _SLOW_RENDER_FLOOR_MS


@pytest.mark.asyncio
async def test_a_section_the_budgeter_trimmed_keeps_its_real_render_time():
    sections = [
        _FakeSection("identity", 1, "You are Auto."),
        _FakeSection("business_graph", 5, "word " * 2000, wait_s=_SLOW_RENDER_S),
    ]
    rendered = await ContextService._render_sections(sections, SimpleNamespace(kwargs={}))
    budget = TokenBudget(total=100, reserved_for_response=0, reserved_for_messages=0)
    included, trimmed_names = TokenBudgetManager().allocate(rendered, budget)
    assert trimmed_names == ["business_graph"]  # the budgeter dropped the slow one

    trace = _chat_trace_sections(rendered, included, trimmed_names, budget)
    assert trace["business_graph"]["trimmed"] is True
    assert trace["business_graph"]["render_ms"] >= _SLOW_RENDER_FLOOR_MS


def test_a_trace_with_render_times_survives_json():
    import json

    result = _result()
    sections = [{**s, "render_ms": 12.3} for s in result.sections]
    trace = ContextResult(system_prompt="identity", mode="chatbot", sections=sections).to_assembly_trace()
    assert json.loads(json.dumps(trace))["sections"][0]["render_ms"] == 12.3
