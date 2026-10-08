"""PRD-256 O5 (#847): the GenAI metrics, over OTLP.

Two histograms from the OpenTelemetry GenAI semantic conventions, recorded for
every ``LLMManager`` call (``genai.traced_llm_call``), sampled trace or not:

* ``gen_ai.client.token.usage`` ({token}), once per call for input and once for
  output tokens (``gen_ai.token.type``);
* ``gen_ai.client.operation.duration`` (s), with ``error.type`` on a failed call.

Their attributes are the low-cardinality ones only (operation, provider, model,
lane), never a workspace, agent or anything a person wrote. They go to the same
collector as the traces (``/v1/metrics``), on a meter provider installed once per
process beside the tracer provider. ``prometheus_client``'s ``/metrics`` and the
PRD-73 dashboards are unchanged. Off: nothing is imported or recorded.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.observability.otel import config, otlp_headers, resource, tracing_enabled

logger = logging.getLogger(__name__)

OTLP_METRICS_PATH = "/v1/metrics"
EXPORT_INTERVAL_MS = 60_000
METER_NAME = "automatos.llm"

TOKEN_USAGE = "gen_ai.client.token.usage"
OPERATION_DURATION = "gen_ai.client.operation.duration"
# The bucket boundaries the conventions advise for each.
TOKEN_BUCKETS = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)
DURATION_BUCKETS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92)


class _State:
    """This process's meter provider, and the instruments made on it."""

    provider: Any = None
    instruments: Optional[tuple] = None


def _otlp_reader() -> Any:
    """A reader that exports to the configured collector every minute."""
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    base = str(config.OTEL_EXPORTER_OTLP_ENDPOINT).rstrip("/")
    exporter = OTLPMetricExporter(endpoint=base + OTLP_METRICS_PATH,
                                  headers=otlp_headers(config.OTEL_EXPORTER_OTLP_HEADERS))
    return PeriodicExportingMetricReader(exporter, export_interval_millis=EXPORT_INTERVAL_MS)


def start_metrics(reader: Any = None) -> bool:
    """Install this process's meter provider as the global one, once; True when it is
    in place. ``reader`` replaces the OTLP reader (tests pass an in-memory one)."""
    if not tracing_enabled() or _State.provider is not None:
        return _State.provider is not None
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider

        provider = MeterProvider(resource=resource(), metric_readers=[reader or _otlp_reader()])
        metrics.set_meter_provider(provider)
        if metrics.get_meter_provider() is not provider:
            logger.error("[otel] another meter provider is already installed; serving without GenAI metrics")
            provider.shutdown()
            return False
    except Exception:
        logger.exception("[otel] metrics not started; serving without GenAI metrics")
        return False
    _State.provider, _State.instruments = provider, None
    return True


def flush_metrics() -> None:
    """Export what is recorded when the lifespan ends; the provider stays for a later one."""
    if _State.provider is None:
        return
    try:
        _State.provider.force_flush()
    except Exception:
        logger.exception("[otel] recorded metrics not flushed")


def _instruments() -> tuple:
    if _State.instruments is None:
        meter = _State.provider.get_meter(METER_NAME)
        _State.instruments = (
            meter.create_histogram(TOKEN_USAGE, unit="{token}", description="Tokens used by an LLM call",
                                   explicit_bucket_boundaries_advisory=list(TOKEN_BUCKETS)),
            meter.create_histogram(OPERATION_DURATION, unit="s", description="An LLM call's duration",
                                   explicit_bucket_boundaries_advisory=list(DURATION_BUCKETS)),
        )
    return _State.instruments


def record_llm_call(attributes: Dict[str, Any], seconds: float, input_tokens: int = 0,
                    output_tokens: int = 0) -> None:
    """One LLM call: its duration, and its tokens by type (none for a call that failed).
    ``attributes`` are the call's low-cardinality ones (see the module)."""
    if _State.provider is None:
        return
    tokens, duration = _instruments()
    duration.record(seconds, attributes)
    if "error.type" in attributes:
        return
    for token_type, count in (("input", input_tokens), ("output", output_tokens)):
        tokens.record(count, {**attributes, "gen_ai.token.type": token_type})
