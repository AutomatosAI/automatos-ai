"""PRD-256 O3 (#847): one GenAI span per LLM call, at the ``LLMManager`` chokepoint.

:func:`traced_llm_call` wraps ``LLMManager.generate_response`` and
``generate_response_sync``, so every call, whatever the provider or lane, is one
``chat {model}`` CLIENT span under the request that made it, per the OpenTelemetry
GenAI semantic conventions. It records the provider, the requested and answering model,
the tokens (cache reads and writes too), the finish reason and the call's cost (the
provider's figure, or the estimate). The provider's own HTTP call (O2a) is its child.

Private by default (Principle 5): no prompt, completion, reasoning, tool argument or
error message goes on the span, only shape and counts. Off (``OTEL_ENABLED=false``),
the wrapper calls straight through and nothing from ``opentelemetry`` is imported.
A call with no request around it (a Mission step, a heartbeat) starts no trace: the
request-rooted sampler drops a CLIENT span with no parent, and O4 gives that work its own.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from typing import Any, Callable, Dict, Iterator

from core.observability.otel import ATTR_WORKSPACE_ID, tracing_enabled

logger = logging.getLogger(__name__)

TRACER_NAME = "automatos.llm"
OPERATION = "chat"

ATTR_AGENT_ID = "automatos.agent_id"
ATTR_REQUEST_TYPE = "automatos.llm.request_type"
ATTR_BYOK = "automatos.llm.byok"
ATTR_STREAMED = "automatos.llm.streamed"
ATTR_TOOL_CALLS = "automatos.llm.tool_calls"
ATTR_COST_USD = "automatos.llm.cost_usd"
ATTR_COST_SOURCE = "automatos.llm.cost_source"

# Automatos' provider names that differ from gen_ai.provider.name's well-known values;
# any other provider (openai, anthropic, deepseek, openrouter, nvidia…) keeps its own.
_PROVIDER_NAMES = {"google": "gcp.gemini", "azure": "azure.ai.openai", "aws_bedrock": "aws.bedrock",
                   "grok": "x_ai"}


def provider_name(provider: Any) -> str:
    """``gen_ai.provider.name`` for an ``LLMProvider`` (or its value)."""
    value = str(getattr(provider, "value", provider) or "unknown")
    return _PROVIDER_NAMES.get(value, value)


def _attribution(manager: Any) -> Dict[str, Any]:
    """Who the call is for, as ``LLMManager._track_usage`` attributes it: the task's
    usage scope over what the manager was built with."""
    from core.llm.usage_context import current_usage_scope

    scope = current_usage_scope()
    ctx = getattr(manager, "_tracking_ctx", None) or {}
    found = {
        ATTR_WORKSPACE_ID: ctx.get("workspace_id") or scope.get("workspace_id"),
        ATTR_AGENT_ID: ctx.get("agent_id") or scope.get("agent_id"),
        ATTR_REQUEST_TYPE: scope.get("request_type") or ctx.get("request_type") or manager.service_name,
    }
    attrs: Dict[str, Any] = {key: str(value) for key, value in found.items() if value}
    attrs[ATTR_BYOK] = bool(ctx.get("is_byok"))
    return attrs


def request_attributes(manager: Any) -> Dict[str, Any]:
    """The span's attributes before the call: the operation, provider, model and attribution."""
    config = manager.config
    return {
        "gen_ai.operation.name": OPERATION,
        "gen_ai.provider.name": provider_name(config.provider),
        "gen_ai.request.model": config.model or "unknown",
        **_attribution(manager),
    }


def response_attributes(manager: Any, response: Any) -> Dict[str, Any]:
    """The span's attributes after the call: the answering model, finish reason, tokens and cost."""
    from core.llm.usage_counts import usage_counts

    counts = usage_counts(response)
    cost_usd, cost_source = manager.call_cost(counts.input_tokens, counts.output_tokens, counts.reported_cost)
    attrs: Dict[str, Any] = {
        "gen_ai.usage.input_tokens": counts.input_tokens,
        "gen_ai.usage.output_tokens": counts.output_tokens,
        "gen_ai.usage.cache_read.input_tokens": counts.cache_read_tokens,
        "gen_ai.usage.cache_creation.input_tokens": counts.cache_write_tokens,
        ATTR_COST_USD: float(cost_usd),
        ATTR_COST_SOURCE: cost_source,
        ATTR_TOOL_CALLS: len(getattr(response, "tool_calls", None) or []),
    }
    if getattr(response, "model", None):
        attrs["gen_ai.response.model"] = str(response.model)
    if getattr(response, "finish_reason", None):
        attrs["gen_ai.response.finish_reasons"] = [str(response.finish_reason)]
    return attrs


def _guarded(build: Callable[..., Dict[str, Any]], *args: Any) -> Dict[str, Any]:
    """An attribute builder's result, or none: a tracing fault never fails the call (Principle 2)."""
    try:
        return build(*args)
    except Exception:  # noqa: BLE001 — logged; the LLM call goes on without these attributes
        logger.debug("[otel] GenAI span attributes skipped", exc_info=True)
        return {}


@contextlib.contextmanager
def _llm_span(manager: Any, streamed: bool) -> Iterator[Any]:
    """The call's CLIENT span, current while the call runs. A failure sets ``error.type``
    to the exception's class and the status to ERROR; its message is not recorded."""
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    attrs = _guarded(request_attributes, manager)
    attrs[ATTR_STREAMED] = streamed
    name = f"{OPERATION} {attrs.get('gen_ai.request.model', 'unknown')}"
    with trace.get_tracer(TRACER_NAME).start_as_current_span(
            name, kind=SpanKind.CLIENT, attributes=attrs, record_exception=False,
            set_status_on_exception=False) as span:
        try:
            yield span
        except BaseException as exc:
            span.set_attribute("error.type", type(exc).__qualname__)
            span.set_status(Status(StatusCode.ERROR, type(exc).__qualname__))
            raise


def traced_llm_call(call: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an ``LLMManager`` call method (async or sync) in its GenAI span."""
    if asyncio.iscoroutinefunction(call):
        @functools.wraps(call)
        async def traced(manager: Any, *args: Any, **kwargs: Any) -> Any:
            if not tracing_enabled():
                return await call(manager, *args, **kwargs)
            on_delta = kwargs.get("on_delta", args[2] if len(args) > 2 else None)  # (messages, tools, on_delta)
            with _llm_span(manager, streamed=on_delta is not None) as span:
                response = await call(manager, *args, **kwargs)
                span.set_attributes(_guarded(response_attributes, manager, response))
                return response
        return traced

    @functools.wraps(call)
    def traced_sync(manager: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracing_enabled():
            return call(manager, *args, **kwargs)
        with _llm_span(manager, streamed=False) as span:
            response = call(manager, *args, **kwargs)
            span.set_attributes(_guarded(response_attributes, manager, response))
            return response
    return traced_sync
