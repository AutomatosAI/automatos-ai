# PRD-256: OpenTelemetry traces over OTLP

**Status:** Complete, 9 Oct 2026. O1 (#1040), O2a (#1041), O2b (#1043, #1044), O3 (#1047), O4 (#1049), O5 (#1050), O6a (#1053) and O6b (#1095) merged; follow-ups in §10 · **Owner:** daarthur (issue #847; the scope and the answers to its questions are Gerard's, 5 Oct 2026) · **Written:** 7 Oct 2026
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
| **O1** | Foundation: a tracer provider per process, started by the FastAPI lifespan; a server span per request (not the probes), carrying `X-Request-ID` and the workspace; OTLP/HTTP export; a parent-based ratio sampler. Tests use an in-memory exporter. | Merged (#1040) |
| **O2a** | Library instrumentation in the API (SQLAlchemy, httpx, Redis, botocore): a client span per SQL statement, Redis command, outbound HTTP request and AWS call, under the request that made it. `traceparent` goes out with every HTTP request, the workspace worker included, and is in CORS `allow_headers`. | Merged (#1041) |
| **O2b** | The workspace worker's half: its routes moved out of the 615-line `_health_server` first (#1043; the code-shape rule makes any edit there fail), then a server span per worker request that continues the API's trace. One trace spans API → worker. | Merged (#1043, #1044) |
| O3 | GenAI spans: an OpenTelemetry tracer behind the PRD-185 `Tracer` seam, and LLM spans at the `LLMManager` chokepoint (`generate_response`, and `generate_response_sync` through the same helper or listed as a gap), per the GenAI semantic conventions. **`LangfuseTracer` is deleted in the same PR** (GUARDRAILS B2); Langfuse becomes an OTLP destination behind the collector. O3 keeps what the Langfuse path records today: tool dispatch, RAG retrieval scores and context-assembly metadata. | Merged (#1047) |
| O4 | Asynchronous boundaries: Missions, board tickets and heartbeats joined by span links, with `traceparent` stored on the work item. | Merged (#1049) |
| O5 | `trace_id`/`span_id` in the logs. `prometheus_client` and the PRD-73 dashboards stay; new GenAI metrics go over OTLP. | Merged (#1050) |
| O6 | A reference collector config in `deploy/otel/`, the Helm chart's OTel values, and server-side spans for the web app. | Merged (#1053: the collector config, chart values and kind check; #1095: the web app) |

## 5. Settings (O1)

| Setting | Default | |
|---|---|---|
| `OTEL_ENABLED` | `false` | Turns tracing on for this process. |
| `OTEL_SERVICE_NAME` | `automatos-api` | `service.name` on every span. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | The collector's OTLP/HTTP base address; `/v1/traces` is appended. |
| `OTEL_EXPORTER_OTLP_HEADERS` | empty | `key=value,key2=value2`, values URL-encoded (the spec's format), e.g. a vendor's `Authorization`. |
| `OTEL_TRACES_SAMPLER_RATIO` | `1.0` | The share of new traces kept; a request whose caller sampled it follows the caller (parent-based). |

The resource carries `service.name`, `deployment.environment.name` (`ENVIRONMENT`) and `automatos.edition` (`AUTH_EDITION`). Standard `OTEL_RESOURCE_ATTRIBUTES` are merged by the SDK.

The PRD-185 seam's own settings (`TRACING_ENABLED`, `TRACING_BACKEND`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) are gone with `LangfuseTracer` (O3): the seam follows `OTEL_ENABLED`. Langfuse takes OTLP, either behind a collector or directly: `OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel` and `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<base64 of public_key:secret_key>`.

## 6. Technical considerations

- **Dependencies** (approved on #847): `opentelemetry-api`, `-sdk`, the OTLP exporter, and the FastAPI, SQLAlchemy, httpx, Redis and botocore instrumentations, each added in the phase that uses it. Versions are resolved against the `fastapi`/`starlette`/`uvicorn` pins (F092), which don't move. O1: `opentelemetry-api`/`-sdk`/`-exporter-otlp-proto-http` 1.45.1, `opentelemetry-instrumentation-fastapi` 0.66b1.
- **OTLP over HTTP, not gRPC.** Every collector, Tempo, Jaeger and the hosted vendors accept OTLP/HTTP, and it needs no `grpcio`.
- **Where it hooks in.** The FastAPI instrumentation wraps the middleware stack Starlette builds on the first ASGI call, which is the lifespan's, so `instrument_app(app)` runs right after the app is created in `main.py`. The provider starts in each process (one per uvicorn worker) from `with_tracing(lifespan)`, which wraps the existing lifespan rather than editing it, and is flushed at shutdown.
- **Noise.** Health probes and the metrics scrape are not traced, and neither are the ASGI layer's per-message `send`/`receive` spans: one request is one span until O2 adds its children.
- **No query value leaves.** The ASGI layer puts the full URL on the server span and redacts only a few signature keys, and the HTTP client redacts none (a `?key=` API key would go); Automatos redacts every query value on the URL attributes of both (`code=REDACTED&token=REDACTED`), keeping the keys (Principle 5). Redis commands are recorded as `SET ? ?` (no key or value), SQL as its text without bound values, AWS calls as service and operation.
- **`traceparent` goes to every HTTP callee**, external APIs included: it carries only the trace and span IDs and the sampled flag.
- **A new trace starts at a request, not at a library call.** The background loops (schedulers, the board dispatcher, heartbeats) run SQL and Redis all the time; each statement used to be a root trace (seen live: 84,143 spans in about 7 hours, idle). The sampler drops a CLIENT span with no parent; inside a request it is the request's child. O4's spans for background work start traces of their own kind.
- **LLM calls (O3).** `LLMManager.generate_response` and `generate_response_sync` are wrapped by one decorator (`core/observability/genai.py`), so each call is a `chat {model}` CLIENT span per the GenAI semantic conventions: `gen_ai.provider.name`, the request and response model, `gen_ai.usage.*` (input, output, cache read and cache creation tokens), the finish reason, and `automatos.llm.cost_usd` with its source (the provider's reported figure, or the estimate the cost audit line uses). The provider's HTTP call is its child. A failure is `error.type` (the exception's class) with no message. Calls that bypass `LLMManager` (the memory-stats summary, the plugin security scan, the multimodal ingestion describer) have only their HTTP span.
- **The seam (O3).** `OtelTracer` turns each emit into a span under the current one: `execute_tool {name}` (`gen_ai.tool.name`, success, back-dated by its duration), `rag.retrieve` (document count, top score, status; not the query) and `context.assembly {mode}` (budget, token estimate, section names; not the content). Only inside a trace: with no span around them (Mission steps, heartbeats) the seam records nothing and an LLM call starts no trace; O4 gives that work its own (the agent run's span).
- **Background work (O4).** Every agent run (`AgentFactory.execute_with_prompt`, whatever the lane) is one `invoke_agent {agent}` span (`gen_ai.agent.id` and `.name`, `automatos.lane`, `automatos.execution_id` such as `board_task:97` or `mission:<id>`), so its LLM calls, tools and queries are its children. Inside a request it is the request's child. Run later by a background loop (the board dispatcher, the coordinator's tick, a heartbeat), it is the root of a trace of its own, and carries a **span link** (`automatos.link=requested_by`) to the request that asked for the work. A ticket or Mission inserted while a span is current is stamped with its `traceparent` (`planning_data` / `config`, key `trace_links`, the latest 8; a SQLAlchemy listener, registered only when tracing is on; a caller's own are dropped); Run Now and plan approval add theirs. The dispatcher and the tick run the work carrying those links (a ContextVar its task copies). A heartbeat has no request: its run is a trace of its own, and the tickets it files link to it. Not spanned: the coordinator's own planner and verifier calls on the tick, and scheduled Playbook runs' steps outside an agent run.
- **Logs (O5).** A log line written inside a sampled span carries `trace_id` and `span_id`: the API's `ContextFilter` and the worker's `TraceIdsFilter` (installed when tracing is on) put them on every record, log-relay ships them to Loki with the request ID and the rest, and the console line gains ` trace=<id>` (`[req=… ws=… agent=… trace=…]`). A trace that isn't sampled is never named (it would not reach the collector). Off, or with no span, the lines are as they were.
- **GenAI metrics (O5).** Every `LLMManager` call, sampled trace or not, records `gen_ai.client.token.usage` ({token}, by `gen_ai.token.type`) and `gen_ai.client.operation.duration` (s, `error.type` on a failure), with the conventions' advised buckets and low-cardinality attributes only: operation, provider, request and response model, and the lane (`automatos.llm.request_type`), never a workspace or agent. A meter provider beside the tracer provider (one per process, flushed at the end of each lifespan) exports them every minute to the same collector (`/v1/metrics`). `/metrics` and the PRD-73 dashboards are unchanged; three Prometheus series nothing ever recorded (`automatos_llm_request_duration_seconds`, `automatos_llm_tokens_total`, `automatos_agent_token_usage_total`) are deleted.
- **The collector (O6).** `deploy/otel/collector.yaml` is the reference config: OTLP/HTTP in on 4318, a memory limit, batching, an `attributes/no-content` processor that deletes the GenAI content attributes should any library ever add them (Principle 5, second line), and OTLP/HTTP out to one backend (`OTLP_BACKEND_ENDPOINT`, `OTLP_BACKEND_AUTHORIZATION`). `deploy/otel/langfuse.yaml` merges over it to send the GenAI spans only to Langfuse as well. Both are validated with `otelcol-contrib validate` (0.161.0).
- **The chart (O6).** `otel.enabled` / `otel.endpoint` / `otel.samplerRatio` / `otel.resourceAttributes` set the OTEL_* settings on the API and the worker (not the migration Job), with each one's service name, the pod and namespace names as resource attributes, and the endpoint's headers from the Secret (`OTEL_EXPORTER_OTLP_HEADERS`, optional). The kind e2e runs a collector on the reference config and checks one request's spans from the API and the worker arrive as one trace.
- **The web app (O6b).** `frontend/instrumentation.ts` installs a tracer provider (`frontend/lib/otel/node.ts`) in the Node runtime when `OTEL_ENABLED` is on; off, or in the Edge runtime, nothing from OpenTelemetry is imported (the import sits in the `NEXT_RUNTIME === 'nodejs'` branch, so the Edge bundle doesn't carry the SDK). Next.js then records its own server spans into it (each request, server render and Node-runtime route handler), service `automatos-web`, with the API's settings: OTLP/HTTP to `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_TRACES_SAMPLER_RATIO` (parent-based), `OTEL_RESOURCE_ATTRIBUTES`. Query values are redacted by a wrapper around the exporter, on the URL attributes and on the span name (Next names its request span after the full target, and copies it into `next.span_name`). Node's `fetch` (undici) is instrumented, so a route's request to the API carries `traceparent` and the API continues the web app's trace (Next's own fetch span sends none). Upstream packages only, decided on #847 to stay vendor-neutral: `@opentelemetry/sdk-trace-node` and `resources` ~2.11, `exporter-trace-otlp-http` ~0.222, `instrumentation-undici` ~0.32. Browser-side spans are not part of this. **Not traced: the Edge-runtime routes**, the chat proxy (`app/api/chat/route.ts`, which the chat UI posts to) and the workflow-stream proxy: the Node SDK can't run there, so a chat turn's trace still starts at the API. Raised for a decision as #1094 (move them to the Node runtime, or leave it).
- **The worker** has the same settings, defaults (service `automatos-workspace-worker`) and rules, in `services/workspace-worker/worker_otel.py`; its image can't import the platform's.
- **One provider per process, for its whole life.** The SDK refuses a second global provider, so a later lifespan in the same process reuses the first; the end of a lifespan flushes it, and the SDK shuts it down when the process exits. `OTEL_TRACES_SAMPLER_RATIO` is parsed when tracing starts: a value that isn't a number is logged and keeps every trace, and can't stop a boot.

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

## 10. Follow-ups

Gaps and decisions found while building it, each an issue for triage:

| Issue | |
|---|---|
| #1051 | `httpx2` request lines (the openai 3.x SDK) bypass the httpx log guard. |
| #1094 | The web app's Edge-runtime proxies (chat, workflow stream) aren't traced, so a chat turn's trace starts at the API. |
| #1107 | API traces: query values can leave in exception messages and the span status. |
| #1110 | LLM calls that bypass `LLMManager`, and embeddings, get no GenAI span. |
| #1111 | The coordinator's own model calls on the tick (verifier, joiner, async planner) aren't traced. |
| #1112 | OpenAI-compatible model calls get no HTTP client span (`httpx2`). |
| #1113 | Whether `traceparent` should go to third-party APIs or internal hosts only. |
| #1114 | Two PRDs are numbered 256. |

## Related

- Issue #847, with Gerard's answers (5 Oct 2026)
- PRD-73 (`docs/PRDS/74-OBSERVABILITY-MONITORING-STACK.md`): metrics, logs and correlation IDs
- PRD-185 S9: the tracer seam (`core/observability/tracer.py`)
- GUARDRAILS B2 (`docs/architecture/GUARDRAILS.md`): delete what you replace
