"""PRD-256 O5 (#847): trace and span IDs in the logs; the GenAI metrics over OTLP.

On: a log line written inside a sampled span carries its ``trace_id`` and ``span_id``
(the API's ``ContextFilter`` and the worker's ``TraceIdsFilter``; the console line
says `` trace=<id>``), and every ``LLMManager`` call, sampled trace or not, records
``gen_ai.client.token.usage`` and ``gen_ai.client.operation.duration`` with
low-cardinality attributes only. Off: nothing is imported, and log lines are as they
were. The three Prometheus LLM series nothing recorded are gone.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_TESTS = Path(__file__).resolve().parent
_ORCH = _TESTS.parent
for _path in (_TESTS, _ORCH):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from core.observability import metrics, otel  # noqa: E402
from helpers_workspace_worker import WORKER_DIR  # noqa: E402

MODEL = "gpt-test-1"


def _record(message="hello"):
    return logging.LogRecord("t", logging.INFO, __file__, 1, message, (), None)


# ── off ──────────────────────────────────────────────────────────────────────

def test_off_log_lines_are_as_they_were_and_nothing_is_imported():
    script = (
        "import logging, sys\n"
        "from core.utils.logging_adapter import ContextFilter\n"
        "from core.observability import metrics\n"
        "record = logging.LogRecord('t', logging.INFO, 'f', 1, 'hello', (), None)\n"
        "ContextFilter().filter(record)\n"
        "assert (record.trace_id, record.span_id, record._trace_context) == ('', '', '')\n"
        "line = logging.Formatter('[req=%(request_id)s%(_trace_context)s] - %(message)s').format(record)\n"
        "assert line == '[req=] - hello', line\n"
        "assert metrics.start_metrics() is False\n"
        "metrics.record_llm_call({}, 1.0, 5, 5)\n"
        "loaded = [m for m in sys.modules if m.startswith('opentelemetry')]\n"
        "assert not loaded, loaded\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "OTEL_ENABLED"}
    done = subprocess.run([sys.executable, "-c", script], cwd=_ORCH, env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_worker_console_line_is_as_it_was_without_a_trace():
    script = (
        "import logging\n"
        "from automatos_logging import setup_logging\n"
        "root = setup_logging(service='w', enable_relay=False)\n"
        "record = logging.LogRecord('t', logging.INFO, 'f', 1, 'hello', (), None)\n"
        "line = root.handlers[0].format(record)\n"
        "assert line.endswith('[t] INFO: hello'), line\n"
    )
    done = subprocess.run([sys.executable, "-c", script], cwd=WORKER_DIR, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_prometheus_llm_series_nothing_recorded_are_gone():
    from core.monitoring import automatos_metrics

    for name in ("LLM_REQUEST_DURATION", "LLM_TOKEN_USAGE", "AGENT_TOKEN_USAGE"):
        assert not hasattr(automatos_metrics, name), name


# ── on ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def otel_on(monkeypatch):
    """Tracing on: fresh global tracer and meter providers, as the app builds them, into
    an in-memory exporter and reader. Yields ``(ratio -> tracer provider, reader)``."""
    from opentelemetry import trace
    from opentelemetry.metrics import _internal as global_metrics
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.util._once import Once

    monkeypatch.setattr(otel.config, "OTEL_ENABLED", True, raising=False)
    saved = (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE,
             global_metrics._METER_PROVIDER, global_metrics._METER_PROVIDER_SET_ONCE)
    trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE = None, Once()
    global_metrics._METER_PROVIDER, global_metrics._METER_PROVIDER_SET_ONCE = None, Once()
    metrics._State.provider = metrics._State.instruments = None
    reader = InMemoryMetricReader()
    assert metrics.start_metrics(reader=reader) is True
    providers = []

    def tracer_provider(ratio=1.0):
        provider = otel.build_provider(InMemorySpanExporter(), ratio)
        trace.set_tracer_provider(provider)
        providers.append(provider)
        return provider

    yield tracer_provider, reader
    for provider in providers + [metrics._State.provider]:
        provider.shutdown()
    metrics._State.provider = metrics._State.instruments = None
    (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE,
     global_metrics._METER_PROVIDER, global_metrics._METER_PROVIDER_SET_ONCE) = saved


def test_a_log_line_in_a_sampled_span_carries_its_trace_and_span(otel_on):
    from opentelemetry import trace

    from core.utils.logging_adapter import ContextFilter

    otel_on[0]()
    with trace.get_tracer("t").start_as_current_span("GET /api/agents/") as span:
        record = _record()
        ContextFilter().filter(record)
    context = span.get_span_context()
    assert (record.trace_id, record.span_id) == (f"{context.trace_id:032x}", f"{context.span_id:016x}")
    line = logging.Formatter("[req=%(request_id)s%(_trace_context)s] - %(message)s").format(record)
    assert line == f"[req= trace={context.trace_id:032x}] - hello"


def test_a_trace_that_is_not_sampled_is_not_named_in_the_logs(otel_on):
    from opentelemetry import trace

    from core.utils.logging_adapter import ContextFilter

    otel_on[0](0.0)
    with trace.get_tracer("t").start_as_current_span("GET /api/agents/"):
        record = _record()
        ContextFilter().filter(record)
    assert (record.trace_id, record.span_id, record._trace_context) == ("", "", "")


def test_the_worker_names_the_trace_in_its_logs(otel_on, monkeypatch):
    from opentelemetry import trace

    monkeypatch.syspath_prepend(str(WORKER_DIR))
    import worker_otel

    otel_on[0]()
    console = logging.StreamHandler()  # the worker's console format (automatos_logging.setup_logging)
    console.setFormatter(logging.Formatter("%(levelname)s:%(_trace_context)s %(message)s",
                                           defaults={"_trace_context": ""}))
    with trace.get_tracer("t").start_as_current_span("GET /workspaces/{workspace_id}/files") as span:
        record = _record()
        worker_otel.TraceIdsFilter().filter(record)
    trace_id = f"{span.get_span_context().trace_id:032x}"
    assert record.trace_id == trace_id and record.span_id == f"{span.get_span_context().span_id:016x}"
    assert console.format(record) == f"INFO: trace={trace_id} hello"


class _Provider:
    def __init__(self, fail=False):
        self.fail = fail

    async def generate_response(self, messages, tools=None):
        from core.llm.clients.base import LLMResponse

        if self.fail:
            raise RuntimeError("provider said no")
        return LLMResponse(content="reply", model=f"{MODEL}-2026-10-01", finish_reason="stop",
                           usage={"input_tokens": 120, "output_tokens": 30})


def _manager(provider):
    from core.llm.clients.base import LLMConfig, LLMProvider
    from core.llm.manager import LLMManager

    manager = LLMManager(config=LLMConfig(provider=LLMProvider.AZURE, model=MODEL, api_key="test"),
                         workspace_id="ws-1", agent_id=7, request_type="chat")
    manager.provider = provider
    return manager


def _points(reader):
    found = {}
    for resource_metrics in reader.get_metrics_data().resource_metrics:
        for scope in resource_metrics.scope_metrics:
            for metric in scope.metrics:
                found[metric.name] = [(dict(p.attributes), p.count, p.sum) for p in metric.data.data_points]
    return found


def test_every_llm_call_records_its_tokens_and_duration_sampled_or_not(otel_on):
    otel_on[0]()
    manager = _manager(_Provider())
    asyncio.run(manager.generate_response([{"role": "user", "content": "hi"}]))  # no request: no trace
    points = _points(otel_on[1])
    base = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "azure.ai.openai",
            "gen_ai.request.model": MODEL, "gen_ai.response.model": f"{MODEL}-2026-10-01",
            "automatos.llm.request_type": "chat"}
    assert sorted(points["gen_ai.client.token.usage"], key=lambda p: p[0]["gen_ai.token.type"]) == [
        ({**base, "gen_ai.token.type": "input"}, 1, 120), ({**base, "gen_ai.token.type": "output"}, 1, 30)]
    [(attributes, count, seconds)] = points["gen_ai.client.operation.duration"]
    assert attributes == base and count == 1 and seconds >= 0


def test_a_failed_call_records_its_duration_with_its_error_and_no_tokens(otel_on):
    otel_on[0]()
    with pytest.raises(RuntimeError):
        asyncio.run(_manager(_Provider(fail=True)).generate_response([{"role": "user", "content": "hi"}]))
    points = _points(otel_on[1])
    [(attributes, count, _)] = points["gen_ai.client.operation.duration"]
    assert attributes["error.type"] == "RuntimeError" and count == 1
    assert "gen_ai.client.token.usage" not in points


def test_the_meter_provider_is_installed_once_and_never_over_someone_elses(otel_on):
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    first = metrics._State.provider
    assert metrics.start_metrics(reader=InMemoryMetricReader()) is True and metrics._State.provider is first
    metrics.flush_metrics()


def test_the_metrics_reader_targets_the_collectors_metrics_path(monkeypatch):
    from opentelemetry.exporter.otlp.proto.http import metric_exporter
    from opentelemetry.sdk.metrics import export

    made = {}
    monkeypatch.setattr(otel.config, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/", raising=False)
    monkeypatch.setattr(otel.config, "OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic%20abc", raising=False)
    monkeypatch.setattr(metric_exporter, "OTLPMetricExporter", lambda **kwargs: made.update(kwargs) or "exporter")
    monkeypatch.setattr(export, "PeriodicExportingMetricReader", lambda exporter, **kwargs: (exporter, kwargs))
    assert metrics._otlp_reader() == ("exporter", {"export_interval_millis": metrics.EXPORT_INTERVAL_MS})
    assert made == {"endpoint": "http://collector:4318/v1/metrics", "headers": {"Authorization": "Basic abc"}}
