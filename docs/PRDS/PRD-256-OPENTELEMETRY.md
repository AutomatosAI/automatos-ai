# PRD-256: OpenTelemetry traces over OTLP

**Status:** O1 in review · **Owner:** daarthur (issue #847; the scope and the answers to its questions are Gerard's, 5 Oct 2026) · **Written:** 7 Oct 2026
**Type:** Extension. PRD-73 built metrics, structured logs and correlation IDs and deferred distributed tracing ("OpenTelemetry — future PRD"). This is that PRD, added on top of what's there.

## 1. Introduction

Automatos has metrics (`prometheus_client`, `/metrics`), structured logs with correlation IDs (PRD-73 Phase 2), the Loki pipeline, and an off-by-default Langfuse tracer behind the PRD-185 seam (`core/observability/tracer.py`). It has no distributed traces:

- **You can't follow one request** across API → workspace worker → LLM call, or one Mission from plan → tasks → result, except by matching `request_id` in the logs by hand. #837 (Mission dispatch blocking the event loop for up to 17 s per task) was diagnosed that way. A trace shows where the time goes directly.
- **Platform teams expect OTLP.** Teams running the Helm chart (#797) send traces to their own backend: Grafana Tempo, Jaeger, Azure Monitor, Datadog, or Langfuse, which accepts OTLP.

## 2. Goals

- One request is one trace, from the API through the worker and the LLM call to the result.
- Asynchronous work (Missions, board tickets, heartbeats) links back to the request that started it.
- Any OTLP backend works, chosen by the operator's collector, with no backend SDK in Automatos.
- Off costs nothing: with `OTEL_ENABLED=false` nothing is configured, exported or instrumented.

## 3. Principles

1. **Off by default.** `OTEL_ENABLED=false` means no `opentelemetry` import from Automatos code, as `get_tracer()` never imports `langfuse` when it is off. (redis-py 8 imports the OpenTelemetry *metrics* SDK on its own when the package is installed; it sets no provider and starts nothing. O1's tests pin both facts.)
2. **Never fail the caller.** Every span and export is guarded; a tracing fault is logged.
3. **Vendor-neutral.** OTLP out only; a collector chooses the destinations.
4. **Carry the existing IDs.** `request_id`, `correlation_id`, `workspace_id` and `agent_id` are span attributes; `trace_id`/`span_id` join the structured log context (O5). `X-Request-ID` and the Loki pipeline stay.
5. **Private by default.** No prompt or completion content on spans, ever, on the hosted edition. On a self-hosted install, at most an operator-level env flag, off by default; no per-workspace switch (whoever runs the collector owns that data).
6. **Settings through `config.py`.**

## 4. Phases (one PR each, each with tests)

| | Scope | Status |
|---|---|---|
| **O1** | Foundation: a tracer provider per process, started by the FastAPI lifespan; a server span per request (not the probes), carrying `X-Request-ID` and the workspace; OTLP/HTTP export; a parent-based ratio sampler. Tests use an in-memory exporter. | This PR |
| O2 | Library instrumentation (SQLAlchemy, httpx, Redis, botocore); `traceparent` from the API to the workspace worker and in CORS `allow_headers`. One trace spans API → worker. | |
| O3 | GenAI spans: an OpenTelemetry tracer behind the PRD-185 `Tracer` seam, and LLM spans at the `LLMManager` chokepoint (`generate_response`, and `generate_response_sync` through the same helper or listed as a gap), per the GenAI semantic conventions. **`LangfuseTracer` is deleted in the same PR** (GUARDRAILS B2); Langfuse becomes an OTLP destination behind the collector. O3 keeps what the Langfuse path records today: tool dispatch, RAG retrieval scores and context-assembly metadata. | |
| O4 | Asynchronous boundaries: Missions, board tickets and heartbeats joined by span links, with `traceparent` stored on the work item. | |
| O5 | `trace_id`/`span_id` in the logs. `prometheus_client` and the PRD-73 dashboards stay; new GenAI metrics go over OTLP. | |
| O6 | A reference collector config in `deploy/otel/`, the Helm chart's OTel values, and server-side spans for the web app. | |

## 5. Settings (O1)

| Setting | Default | |
|---|---|---|
| `OTEL_ENABLED` | `false` | Turns tracing on for this process. |
| `OTEL_SERVICE_NAME` | `automatos-api` | `service.name` on every span. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | The collector's OTLP/HTTP base address; `/v1/traces` is appended. |
| `OTEL_EXPORTER_OTLP_HEADERS` | empty | `key=value,key2=value2`, values URL-encoded (the spec's format), e.g. a vendor's `Authorization`. |
| `OTEL_TRACES_SAMPLER_RATIO` | `1.0` | The share of new traces kept; a request whose caller sampled it follows the caller (parent-based). |

The resource carries `service.name`, `deployment.environment.name` (`ENVIRONMENT`) and `automatos.edition` (`AUTH_EDITION`). Standard `OTEL_RESOURCE_ATTRIBUTES` are merged by the SDK.

## 6. Technical considerations

- **Dependencies** (approved on #847): `opentelemetry-api`, `-sdk`, the OTLP exporter, and the FastAPI, SQLAlchemy, httpx, Redis and botocore instrumentations, each added in the phase that uses it. Versions are resolved against the `fastapi`/`starlette`/`uvicorn` pins (F092), which don't move. O1: `opentelemetry-api`/`-sdk`/`-exporter-otlp-proto-http` 1.45.1, `opentelemetry-instrumentation-fastapi` 0.66b1.
- **OTLP over HTTP, not gRPC.** Every collector, Tempo, Jaeger and the hosted vendors accept OTLP/HTTP, and it needs no `grpcio`.
- **Where it hooks in.** The FastAPI instrumentation wraps the middleware stack Starlette builds on the first ASGI call, which is the lifespan's, so `instrument_app(app)` runs right after the app is created in `main.py`. The provider starts in each process (one per uvicorn worker) from `with_tracing(lifespan)`, which wraps the existing lifespan rather than editing it, and is flushed at shutdown.
- **Noise.** Health probes and the metrics scrape are not traced, and neither are the ASGI layer's per-message `send`/`receive` spans: one request is one span until O2 adds its children.

## 7. Non-goals

- Replacing PRD-73's metrics, logs, correlation IDs or Loki pipeline.
- A backend: Automatos exports OTLP; the operator's collector decides where it goes.
- Prompt or completion content on spans (Principle 5).

## 8. Success metrics

- O2: a request that reaches the worker is one trace across both services.
- O3: every LLM call, whatever the provider, is one GenAI span with tokens and cost; the Langfuse path is gone.
- O4: a Mission's tasks are linked to the request that started it.
- Off adds no measurable overhead (O1's tests: no trace SDK, exporter, instrumentation, provider or thread).

## 9. Testing

Each phase ships tests with an in-memory exporter. O1 was also checked against a real collector: the local stack with `grafana/otel-lgtm` (an OpenTelemetry Collector, Tempo, Prometheus, Loki and Grafana in one container) receiving the API's spans over OTLP/HTTP. O6 adds the reference collector config and a check through the Helm chart.

## Related

- Issue #847, with Gerard's answers (5 Oct 2026)
- PRD-73 (`docs/PRDS/74-OBSERVABILITY-MONITORING-STACK.md`): metrics, logs and correlation IDs
- PRD-185 S9: the tracer seam (`core/observability/tracer.py`)
- GUARDRAILS B2 (`docs/architecture/GUARDRAILS.md`): delete what you replace
