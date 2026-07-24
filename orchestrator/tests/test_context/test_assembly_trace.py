"""PRD-201 S1 — persist the assembly trace (answerability).

Pure — no live Langfuse, no DB. Asserts the durable trace *shape* the assembler
hands back, and that the tracer seam emits a span only when tracing is ON while
the trace record itself is produced regardless.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_ORCH = Path(__file__).resolve().parent.parent.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.context.result import ContextResult
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
    boom.trace_assembly.side_effect = RuntimeError("langfuse down")
    with patch.object(tracer_mod, "get_tracer", return_value=boom):
        # Must not raise into the build.
        tracer_mod.fire_assembly_trace(trace={"mode": "chatbot"}, workspace_id="ws")


def test_langfuse_tracer_builds_one_assembly_span():
    span = MagicMock()
    client = MagicMock()
    client.start_as_current_span.return_value.__enter__.return_value = span
    lf = tracer_mod.LangfuseTracer(client)
    lf.trace_assembly(
        trace={"mode": "chatbot", "budget_total": 100, "token_estimate": 50, "sections": [{}, {}]},
        workspace_id="ws",
    )
    # Named span keyed by mode; metadata folded on; sections not dumped verbatim.
    name = client.start_as_current_span.call_args.kwargs.get("name") or client.start_as_current_span.call_args.args[0]
    assert name == "assembly:chatbot"
    span.update.assert_called()
