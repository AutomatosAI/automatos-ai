# OpenTelemetry Collector for Automatos

Automatos sends its traces, and the API's GenAI metrics, over **OTLP/HTTP** when
`OTEL_ENABLED=true` ([PRD-256](../../docs/PRDS/PRD-256-OPENTELEMETRY.md)). It ships
no backend SDK: an [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)
between Automatos and your backend decides where the data goes.

| File | |
|---|---|
| [`collector.yaml`](collector.yaml) | The reference config. OTLP/HTTP in on `:4318`; a memory limit, batching, and a processor that strips the GenAI content attributes (in case a library ever adds them); OTLP/HTTP out to one backend. Health check on `:13133`. |
| [`langfuse.yaml`](langfuse.yaml) | An addition: the GenAI spans only (LLM calls, agent runs, tool calls) to Langfuse as well. |

Both are checked with `otelcol-contrib validate` against
`otel/opentelemetry-collector-contrib:0.161.0`, and the kind end-to-end test
([`deploy/kind/otel.sh`](../kind/otel.sh)) runs `collector.yaml` in the cluster.

## What Automatos sends

- **Traces** from the API (`automatos-api`) and the workspace worker
  (`automatos-workspace-worker`). One request is one trace: the server span, SQL,
  Redis, outbound HTTP and AWS calls, every LLM call (`chat {model}`, with tokens and
  cost), every agent run (`invoke_agent`), tool call (`execute_tool`), retrieval and
  context assembly. Background work (board tickets, Missions, heartbeats) is a trace
  of its own, with a span link to the request that asked for it.
- **Metrics** from the API: `gen_ai.client.token.usage` and
  `gen_ai.client.operation.duration`, by provider, model and lane.
- **No content.** No prompt, completion, query, tool argument, request body or query
  string value is put on a span; the collector's `attributes/no-content` processor is a
  second line of defence.

## Settings

The collector reads these from its environment:

| Variable | |
|---|---|
| `OTLP_BACKEND_ENDPOINT` | The backend's OTLP/HTTP base URL; `/v1/traces` and `/v1/metrics` are appended. |
| `OTLP_BACKEND_AUTHORIZATION` | The `Authorization` header the backend wants, if any (e.g. `Basic <base64>`, `Bearer <token>`). |
| `LANGFUSE_OTLP_ENDPOINT`, `LANGFUSE_AUTH` | With `langfuse.yaml` only: `https://cloud.langfuse.com/api/public/otel` (EU), `https://us.cloud.langfuse.com/api/public/otel` (US) or your own host's `/api/public/otel`; and base64 of `<public key>:<secret key>`. |

Some backends' OTLP/HTTP intake:

| Backend | `OTLP_BACKEND_ENDPOINT` | `OTLP_BACKEND_AUTHORIZATION` |
|---|---|---|
| Grafana Tempo + Mimir (self-hosted) | their OTLP/HTTP receiver, e.g. `http://tempo:4318` (traces) | none |
| Grafana Cloud | the stack's OTLP endpoint (`https://otlp-gateway-<zone>.grafana.net/otlp`) | `Basic <base64 of instance-id:token>` |
| Jaeger | `http://jaeger-collector:4318` | none |
| Honeycomb | `https://api.honeycomb.io` | none; it wants an `x-honeycomb-team` header instead: add it under the exporter's `headers` |

Traces and metrics go to the same endpoint. To split them (Tempo for traces, Mimir
or Prometheus for metrics), give the `metrics` pipeline its own exporter.

## Running it

**Docker**, next to `docker compose up`:

```bash
docker run -d --name otel-collector --network automatos_network \
  -v "$PWD/deploy/otel:/conf:ro" -e OTLP_BACKEND_ENDPOINT=http://tempo:4318 \
  otel/opentelemetry-collector-contrib:0.161.0 --config=/conf/collector.yaml
```

Then set `OTEL_ENABLED=true` and `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318`
for the backend and the workspace worker. For a quick look without a backend,
[`grafana/otel-lgtm`](https://github.com/grafana/docker-otel-lgtm) is a collector,
Tempo, Prometheus, Loki and Grafana in one container.

**Kubernetes**, with the
[OpenTelemetry Collector Helm chart](https://github.com/open-telemetry/opentelemetry-helm-charts/tree/main/charts/opentelemetry-collector)
(`mode: deployment`) or any Deployment that mounts `collector.yaml` from a ConfigMap
([`deploy/kind/otel-collector.yaml`](../kind/otel-collector.yaml) is a minimal one).
Then point the Automatos chart at it:

```yaml
otel:
  enabled: true
  endpoint: http://otel-collector.observability:4318
```

If the endpoint needs headers (Automatos straight to a vendor, with no collector), put
them in the chart's Secret as `OTEL_EXPORTER_OTLP_HEADERS`
(`Authorization=Basic%20<base64>`, values URL-encoded); they never go in values.

## Changing it

Check a change before you deploy it:

```bash
docker run --rm -v "$PWD/deploy/otel:/cfg:ro" -e OTLP_BACKEND_ENDPOINT=http://x:4318 \
  otel/opentelemetry-collector-contrib:0.161.0 validate --config=/cfg/collector.yaml
```

Extra config files merge over this one (`--config=collector.yaml --config=yours.yaml`):
maps merge, lists are replaced. That is how `langfuse.yaml` adds a pipeline and the
kind test swaps the exporter.
