"""The one place tests reset OpenTelemetry's global providers (PRD-256).

The API allows one global tracer provider and one global meter provider per
process, each set once. A test that installs its own needs both cleared first
and restored after, and the only way is through the API's private names
(``_TRACER_PROVIDER``, ``_METER_PROVIDER`` and their ``Once`` guards, as of
opentelemetry 1.45.1). They carry no compatibility promise, so they live here
alone: an upgrade that moves them is fixed in this file (review on #1050).
"""
from __future__ import annotations

import contextlib
from typing import Iterator


@contextlib.contextmanager
def fresh_global_providers() -> Iterator[None]:
    """No global tracer or meter provider inside; the process's own restored after."""
    from opentelemetry import trace
    from opentelemetry.metrics import _internal as global_metrics
    from opentelemetry.util._once import Once

    saved = (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE,
             global_metrics._METER_PROVIDER, global_metrics._METER_PROVIDER_SET_ONCE)
    trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE = None, Once()
    global_metrics._METER_PROVIDER, global_metrics._METER_PROVIDER_SET_ONCE = None, Once()
    try:
        yield
    finally:
        (trace._TRACER_PROVIDER, trace._TRACER_PROVIDER_SET_ONCE,
         global_metrics._METER_PROVIDER, global_metrics._METER_PROVIDER_SET_ONCE) = saved
