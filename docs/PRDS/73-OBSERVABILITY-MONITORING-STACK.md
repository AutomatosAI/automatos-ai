# PRD-73: Observability & Monitoring Stack

**Version:** 1.2
**Status:** Draft
**Priority:** P0
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-03-08
**Updated:** 2026-03-08
**Dependencies:** PRD-06 (Monitoring & Analytics — COMPLETE), PRD-55 (Agent Heartbeats — COMPLETE), PRD-72 (Activity Command Centre — DRAFT)
**Repository:** `automatos-monitoring`
**Branch:** `feat/observability-stack`
**Deployment:** Railway (all services deployed as Railway services within a single project)

---

## Executive Summary

Automatos has internal application-level monitoring (PRD-06) and agent heartbeats (PRD-55), but zero infrastructure observability. No log aggregation, no metrics history, no alerting pipeline, no dashboards showing system health over time. When the backend OOMs at 3am, nobody knows until users complain. When an agent heartbeat silently fails, it's invisible. When Redis memory hits 256MB and starts evicting, there's no alert.

This PRD defines the external observability stack for the Automatos AI Platform — deployed as Railway services within the same Railway project, using Railway's private networking for inter-service communication and Railway's log drain for log forwarding.

### What We're Building

1. **Metrics Pipeline** — Prometheus scraping all Automatos services (backend, postgres, redis, workspace-worker) via Railway private networking, with 15-day retention and Railway volume persistence
2. **Log Aggregation** — Loki receiving logs via Railway's HTTP log drain + direct push from the backend, queryable from Grafana with 7-day retention
3. **Dashboards** — Grafana with purpose-built dashboards: Platform Overview, Agent Performance, Database Health, Redis & Queues, Workspace Worker
4. **Alerting Pipeline** — AlertManager routing alerts by severity to webhooks that Automatos agents can consume for investigation and reporting
5. **Agent-Readable Alert API** — Structured webhook payloads that the orchestrator can ingest, allowing agents to investigate, classify, and recommend actions (automated remediation is out of scope for v1)

### What We're NOT Building

- Application-level metrics instrumentation (that's PRD-06, already done)
- A replacement for the Activity Command Centre (PRD-72 handles operational visibility)
- Custom exporters for Automatos-specific metrics (Phase 2)
- Distributed tracing (OpenTelemetry — future PRD)
- External uptime monitoring / synthetic checks
- Promtail / Docker socket log collection (Railway doesn't expose Docker socket — we use Railway log drains instead)

---

## 1. Architecture Overview

```
┌─────────────── Railway Project: automatos ──────────────────────┐
│                                                                  │
│  ┌─────────── Application Services ──────────────────────────┐  │
│  │                                                            │  │
│  │  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌───────────┐   │  │
│  │  │ Backend  │  │ Frontend │  │ Redis │  │ PostgreSQL│   │  │
│  │  │ :8000    │  │ :3000    │  │ :6379 │  │ :5432     │   │  │
│  │  └────┬─────┘  └──────────┘  └───┬───┘  └─────┬─────┘   │  │
│  │       │                          │             │          │  │
│  │  ┌────┴─────┐                    │             │          │  │
│  │  │Workspace │                    │             │          │  │
│  │  │Worker    │                    │             │          │  │
│  │  │:8081     │                    │             │          │  │
│  │  └──────────┘                    │             │          │  │
│  └───────┬──────────────────────────┼─────────────┼──────────┘  │
│          │ Railway Private Network  │             │              │
│          │ *.railway.internal       │             │              │
│  ┌───────▼──────────────────────────▼─────────────▼──────────┐  │
│  │           Monitoring Services                              │  │
│  │                                                            │  │
│  │  ┌───────────┐                        ┌─────────────────┐ │  │
│  │  │Prometheus │  scrapes via           │ Railway Log     │ │  │
│  │  │  :9090    │  *.railway.internal    │ Drain (HTTP)    │ │  │
│  │  └─────┬─────┘                        └────────┬────────┘ │  │
│  │        │                                        │          │  │
│  │        │ queries    ┌──────────┐      push      │          │  │
│  │        ├───────────►│ Grafana  │◄───────────────┤          │  │
│  │        │            │  :3000   │                │          │  │
│  │        │            └──────────┘       ┌────────▼───────┐ │  │
│  │        │                               │    Loki        │ │  │
│  │  ┌─────▼──────────┐                   │   :3100       │ │  │
│  │  │ AlertManager   │                   └────────────────┘ │  │
│  │  │   :9093        │                                      │  │
│  │  └───────┬────────┘                                      │  │
│  │          │ webhook via private network                    │  │
│  │          ▼                                                │  │
│  │  backend.railway.internal:8000/api/alerts/ingest         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Railway Networking Model

All services live in the **same Railway project**. Railway provides automatic private networking:

- **Private DNS:** Every service is reachable at `<service-name>.railway.internal:<port>`
- **No Docker networks needed** — Railway handles inter-service routing automatically
- **No Docker socket access** — Log collection uses Railway's HTTP log drain → Loki, not Promtail Docker SD
- **Volumes:** Railway persistent volumes for Prometheus data, Grafana data, and Loki storage
- **Public access:** Grafana (dashboard access) and log-relay (Railway log drain target) get public Railway domains. Loki stays private — log-relay forwards to it internally.

**Service DNS Map (private network):**

| Service | Internal DNS | Port |
|---------|-------------|------|
| Backend | `backend.railway.internal` | 8000 |
| PostgreSQL | `postgres.railway.internal` | 5432 |
| Redis | `redis.railway.internal` | 6379 |
| Workspace Worker | `workspace-worker.railway.internal` | 8081 |
| Prometheus | `prometheus.railway.internal` | 9090 |
| Grafana | `grafana.railway.internal` | 3000 |
| Loki | `loki.railway.internal` | 3100 |
| AlertManager | `alertmanager.railway.internal` | 9093 |
| Postgres Exporter | `postgres-exporter.railway.internal` | 9187 |
| Redis Exporter | `redis-exporter.railway.internal` | 9121 |

---

## 2. Services & Configuration

### 2.1 Prometheus (Metrics Collection)

**Image:** `prom/prometheus:v2.51.0`
**Port:** 9090
**Retention:** 15 days
**Scrape Interval:** 15s

**Scrape Targets (via Railway private networking):**

| Job | Target | Metrics | Notes |
|-----|--------|---------|-------|
| `automatos-backend` | `backend.railway.internal:8000/health` | HTTP health, response time | Custom JSON → needs adapter or `/metrics` endpoint |
| `postgres-exporter` | `postgres-exporter.railway.internal:9187` | Connections, query duration, DB size, replication lag | Sidecar service |
| `redis-exporter` | `redis-exporter.railway.internal:9121` | Memory, connected clients, ops/sec, keyspace, evictions | Sidecar service |
| `workspace-worker` | `workspace-worker.railway.internal:8081/health` | Worker health, task queue depth | Custom JSON |
| `prometheus` | `localhost:9090/metrics` | Self-monitoring | Built-in |
| `loki` | `loki.railway.internal:3100/metrics` | Log ingestion rate, storage | Built-in |
| `alertmanager` | `alertmanager.railway.internal:9093/metrics` | Alert pipeline health | Built-in |

> **No node-exporter:** Railway manages the underlying infrastructure. Host-level CPU/memory/disk metrics are available via Railway's built-in metrics dashboard. We monitor application-level resource usage through the backend's `/health` endpoint and process metrics instead.

**Strongly recommended for Phase 1 completeness:** Minimal Prometheus-format `/metrics` endpoints for backend and workspace-worker. Without them, application alerting is limited to health probing and log-derived signals — meaning `HighErrorRate`, `SlowResponses`, `QueueBacklog`, `HeartbeatFailureRate`, and other application alerts in Section 3.2 **cannot fire** until `/metrics` is implemented.

**Minimum `/metrics` surface (Phase 1):**
- `http_requests_total{method, path, status}` — request counter with status codes
- `http_request_duration_seconds{method, path}` — histogram of response times
- `automatos_heartbeat_executions_total` / `automatos_heartbeat_failures_total` — heartbeat counters
- `automatos_workspace_queue_depth{priority}` — current queue sizes

**Phase 2 expansion:** Workflow execution times, agent token usage, per-model cost tracking, embedding throughput.

### 2.2 Grafana (Visualisation)

**Image:** `grafana/grafana:10.4.0`
**Port:** 3000 (Railway assigns public domain, e.g. `grafana-automatos.up.railway.app`)
**Auth:** Admin password from Railway environment variable

**Datasources (auto-provisioned):**

| Name | Type | URL | Default |
|------|------|-----|---------|
| Prometheus | prometheus | `http://prometheus.railway.internal:9090` | Yes |
| Loki | loki | `http://loki.railway.internal:3100` | No |

**Dashboards (auto-provisioned):**

| Dashboard | Purpose | Key Panels |
|-----------|---------|------------|
| **Platform Overview** | Single-pane-of-glass for the whole platform | Service health matrix, CPU/Memory/Disk gauges, alert count, uptime |
| **Agent Performance** | Agent heartbeat and execution monitoring | Heartbeat success rate, execution duration, token spend, agent status grid |
| **Database Health** | PostgreSQL deep-dive | Active connections, query duration p95, DB size growth, dead tuples, cache hit ratio |
| **Redis & Queues** | Redis + workspace task queue health | Memory usage, ops/sec, evicted keys, queue depth per priority, task throughput |
| **Workspace Worker** | Worker process monitoring | Active tasks, task duration histogram, error rate, queue backlog |
| **Logs Explorer** | Pre-configured Loki log views | Error log stream, service-filtered views, log volume over time |

### 2.3 Loki (Log Aggregation)

**Image:** `grafana/loki:2.9.4`
**Port:** 3100
**Retention:** 7 days (168h)
**Storage:** Local filesystem (`/loki/chunks`)

**Configuration highlights:**
- Schema v13 with TSDB index
- Ingestion rate limit: 16MB/s (burst 24MB)
- Max streams per tenant: 10,000
- Compaction: Every 10 minutes
- Single-tenant mode (no auth required internally)

### 2.4 Log Collection (Railway Log Drain → Loki)

Railway doesn't expose Docker sockets, so we **replace Promtail** with two complementary approaches:

#### A. Railway HTTP Log Drain (all container stdout/stderr)

Railway supports HTTP log drains that forward all service logs to a URL. We configure a log drain pointing at Loki's push API:

```
Railway Project Settings → Log Drains → Add HTTP Drain
URL: https://log-relay-automatos.up.railway.app/drain
Header: X-Railway-Secret: <shared-secret>
```

> **Note:** The log drain points at the **log-relay** service (public), NOT Loki directly. Loki remains private. The log-relay transforms Railway's JSON format into Loki's push format and forwards internally. See Section 2.4.1.

#### B. Direct Push from Backend (structured application logs)

The backend pushes structured logs directly to Loki via the `python-logging-loki` library or a simple HTTP handler:

```python
# In backend logging config
handler = LokiHandler(
    url="http://loki.railway.internal:3100/loki/api/v1/push",
    tags={"service": "automatos-backend", "environment": ENVIRONMENT},
)
```

This gives us structured JSON logs with labels: `level`, `module`, `request_id`, `agent_id`, `workspace_id`.

#### 2.4.1 Log Relay Service (Railway Log Drain → Loki)

**Image:** Custom lightweight container or `grafana/promtail:2.9.4` in HTTP receiver mode
**Purpose:** Receives Railway log drain webhooks, transforms to Loki push format, forwards to Loki

Railway log drain sends JSON in this format:
```json
{"message": "log line", "severity": "info", "service": "backend", "timestamp": "..."}
```

The relay:
1. Listens on HTTP port for Railway log drain POSTs
2. Extracts `service`, `severity` as Loki labels
3. Batches and pushes to `http://loki.railway.internal:3100/loki/api/v1/push`
4. Drops health check noise (`GET /health` lines)

**Alternative:** If Railway's log drain format evolves to support Loki natively, this relay can be removed.

### 2.5 AlertManager (Alert Routing)

**Image:** `prom/alertmanager:v0.27.0`
**Port:** 9093

**Alert Routing Tree:**

```
root (default → automatos-webhook, group_wait: 30s, repeat: 4h)
├── severity=critical → critical-webhook (group_wait: 10s, repeat: 1h)
│   Examples: ServiceDown, PostgreSQLDown, RedisDown, DiskSpaceCritical
├── severity=warning → warning-webhook (group_wait: 60s, repeat: 4h)
│   Examples: HighCPU, HighMemory, SlowQueries, QueueBacklog
└── severity=info → info-webhook (group_wait: 5m, repeat: 12h)
    Examples: HeartbeatMissed, DiskSpaceWarning
```

**Webhook Target:** `http://backend.railway.internal:8000/api/alerts/ingest`

**Inhibition Rules:**
- Critical service-down alerts suppress warning-level performance alerts for the same service
- PostgreSQLDown suppresses all database performance alerts

**Webhook Payload Format (AlertManager standard):**
```json
{
  "status": "firing",
  "alerts": [
    {
      "status": "firing",
      "labels": {
        "alertname": "HighMemoryUsage",
        "severity": "warning",
        "service": "automatos-backend",
        "instance": "automatos_backend:8000"
      },
      "annotations": {
        "summary": "Backend memory usage above 85%",
        "description": "automatos_backend memory at 91% for >5m",
        "runbook": "Scale backend or investigate memory leak"
      },
      "startsAt": "2026-03-08T14:30:00Z",
      "generatorURL": "http://prometheus:9090/graph?..."
    }
  ]
}
```

### 2.6 Exporters (Separate Railway Services)

On Railway, exporters run as independent services (not Docker sidecars), connecting to targets via private networking:

| Exporter | Image | Target | Port | Railway Service Name |
|----------|-------|--------|------|---------------------|
| postgres-exporter | `prometheuscommunity/postgres-exporter:0.15.0` | `postgres.railway.internal:5432` | 9187 | `postgres-exporter` |
| redis-exporter | `oliver006/redis_exporter:v1.58.0` | `redis.railway.internal:6379` | 9121 | `redis-exporter` |

> **No node-exporter:** Railway doesn't expose host-level metrics to containers. Use Railway's built-in metrics dashboard for infrastructure-level visibility. Application process metrics (memory RSS, CPU time, open FDs) can be exposed via the backend's `/metrics` endpoint in Phase 2.

---

## 3. Alert Rules

### 3.1 Infrastructure Alerts

```yaml
# Service Health (Railway services)
- ServiceDown:          up == 0 for 1m              [critical]
# Note: Host CPU/Memory/Disk alerts are handled by Railway's built-in
# monitoring and alerting. We focus on application-level health here.

# PostgreSQL
- PostgreSQLDown:       pg_up == 0 for 30s          [critical]
- HighConnections:      connections > 150 for 5m     [warning]  # max 200
- SlowQueries:          p95 query time > 1s for 5m   [warning]
- DeadTuples:           dead tuples > 10000          [warning]
- ReplicationLag:       lag > 30s for 5m             [warning]
- CacheHitLow:          hit ratio < 0.95 for 10m     [warning]
- DBSizeGrowth:         > 1GB growth in 24h          [info]

# Redis
- RedisDown:            redis_up == 0 for 30s        [critical]
- HighMemory:           used > 200MB for 5m          [warning]  # max 256MB
- HighMemoryCritical:   used > 240MB for 2m          [critical]
- EvictedKeys:          evicted > 0 for 5m           [warning]
- HighLatency:          cmd latency > 10ms for 5m    [warning]
- ConnectedClients:     clients > 100 for 5m         [warning]
```

### 3.2 Application Alerts

```yaml
# Backend API
- BackendDown:          health endpoint unreachable for 1m    [critical]
- HighErrorRate:        5xx errors > 5% of requests for 5m   [warning]
- SlowResponses:        p95 response > 5s for 5m             [warning]

# Workspace Worker
- WorkerDown:           health endpoint unreachable for 2m    [critical]
- QueueBacklog:         pending tasks > 20 for 10m            [warning]
- QueueCritical:        pending tasks > 50 for 5m             [critical]
- TaskFailureRate:      failures > 10% for 15m                [warning]

# Agent Heartbeats (scraped from backend /api/heartbeat/status)
- HeartbeatMissed:      scheduled heartbeat not executed       [info]
- HeartbeatFailureRate: failures > 25% in last hour           [warning]
```

### 3.3 Alert Capability Matrix

Which alerts can fire on day one vs. after `/metrics` instrumentation:

| Alert | Data Source | Phase 1 (health + exporters) | Phase 1 (with `/metrics`) | Phase 2 |
|-------|-----------|:---:|:---:|:---:|
| ServiceDown | `up` metric | **Yes** | **Yes** | |
| BackendDown | `up{job=automatos-backend}` | **Yes** | **Yes** | |
| WorkerDown | `up{job=automatos-workspace-worker}` | **Yes** | **Yes** | |
| PostgreSQLDown | `pg_up` | **Yes** | **Yes** | |
| HighConnections | `pg_stat_activity_count` | **Yes** | **Yes** | |
| SlowQueries | `pg_stat_activity_max_tx_duration` | **Yes** | **Yes** | |
| DeadTuples | `pg_stat_user_tables_n_dead_tup` | **Yes** | **Yes** | |
| CacheHitLow | `pg_stat_database_blks_*` | **Yes** | **Yes** | |
| DBSizeGrowth | `pg_database_size_bytes` | **Yes** | **Yes** | |
| RedisDown | `redis_up` | **Yes** | **Yes** | |
| RedisHighMemory | `redis_memory_used_bytes` | **Yes** | **Yes** | |
| RedisEvictedKeys | `redis_evicted_keys_total` | **Yes** | **Yes** | |
| RedisHighLatency | `redis_commands_duration_seconds_total` | **Yes** | **Yes** | |
| RedisHighClients | `redis_connected_clients` | **Yes** | **Yes** | |
| HighErrorRate | `http_requests_total` | No | **Yes** | |
| SlowResponses | `http_request_duration_seconds` | No | **Yes** | |
| QueueBacklog | `automatos_workspace_queue_depth` | No | **Yes** | |
| QueueCritical | `automatos_workspace_queue_depth` | No | **Yes** | |
| HeartbeatFailureRate | `automatos_heartbeat_*` | No | **Yes** | |
| ErrorSpike | Loki log query | No | No | **Yes** |
| OOMKill | Loki log query | No | No | **Yes** |
| DatabaseError | Loki log query | No | No | **Yes** |

> **Takeaway:** Without `/metrics`, only infrastructure alerts (exporters + health probes) fire on day one. With minimal `/metrics` instrumentation (strongly recommended for Phase 1), application alerts also become operational.

### 3.4 Log-Based Alerts (Loki Ruler — Phase 2)

```yaml
# Triggered from log patterns, not metrics
- ErrorSpike:           > 50 ERROR logs in 5m window          [warning]
- OOMKill:              "OOMKilled" in container logs          [critical]
- DatabaseError:        "connection refused" in backend logs   [critical]
- AuthFailure:          > 10 auth failures in 5m              [warning]
```

---

## 4. Agent Alert Integration (SENTINEL — Read-Only in v1)

Alerts feed back into Automatos agents for **investigation and reporting only**. Automated remediation is explicitly out of scope for v1 — agents observe, classify, and recommend; humans approve actions.

### 4.1 Alert Ingestion Endpoint

**New endpoint on the backend:**

```
POST /api/alerts/ingest
Content-Type: application/json
Authorization: Bearer <ALERT_INGEST_TOKEN>
X-Alert-Source: alertmanager
```

**Authentication:** Bearer token validated against `ALERT_INGEST_TOKEN` env var. Requests without a valid token return `401`.

**Deduplication:** Alerts are deduplicated by `(alertname, service, instance, fingerprint)`. If a firing alert with the same key already exists and is still active, the `last_seen_at` timestamp is updated instead of creating a new row. AlertManager's `fingerprint` field (included in webhook payloads) is used as the primary dedupe key.

**Resolved alert handling:** When AlertManager sends `status: "resolved"`, the matching active alert row is updated with `resolved_at` timestamp and `status = "resolved"`. No new row is created.

This endpoint:
1. Validates auth token
2. Deduplicates against existing active alerts
3. Stores new alerts in the `infrastructure_alerts` table
4. Updates resolved alerts when resolution webhook arrives
5. For `critical` severity: triggers an investigation agent heartbeat (read-only)
6. For `warning` severity: logs and surfaces in Activity Command Centre (PRD-72)
7. For `info` severity: logs only

**`infrastructure_alerts` table schema:**

```sql
CREATE TABLE infrastructure_alerts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fingerprint     TEXT NOT NULL,           -- AlertManager fingerprint (dedupe key)
    alertname       TEXT NOT NULL,
    severity        TEXT NOT NULL,           -- critical, warning, info
    status          TEXT NOT NULL DEFAULT 'firing',  -- firing, resolved
    service         TEXT,                    -- e.g. "backend", "postgresql", "redis"
    instance        TEXT,                    -- e.g. "backend.railway.internal:8000"
    summary         TEXT,                    -- annotation.summary
    description     TEXT,                    -- annotation.description
    runbook         TEXT,                    -- annotation.runbook
    labels          JSONB DEFAULT '{}',      -- full AlertManager labels
    annotations     JSONB DEFAULT '{}',      -- full AlertManager annotations
    generator_url   TEXT,                    -- Prometheus graph link
    starts_at       TIMESTAMPTZ NOT NULL,
    ends_at         TIMESTAMPTZ,
    resolved_at     TIMESTAMPTZ,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_response  JSONB,                   -- agent investigation results (v1: read-only)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_active_alert UNIQUE (fingerprint, status)
);

CREATE INDEX idx_alerts_status ON infrastructure_alerts (status);
CREATE INDEX idx_alerts_severity ON infrastructure_alerts (severity);
CREATE INDEX idx_alerts_service ON infrastructure_alerts (service);
CREATE INDEX idx_alerts_created ON infrastructure_alerts (created_at DESC);
```

### 4.2 Agent Investigation Actions (Read-Only)

When a critical alert fires, the platform triggers an agent heartbeat to **investigate and report** — not to take remediation actions.

| Alert | Agent Investigation | Output |
|-------|-------------------|--------|
| `RedisHighMemory` | Query Redis INFO memory, identify large keys | Evidence summary + impact classification + recommended action |
| `QueueBacklog` | Query Redis queue lengths, check for stuck tasks | Stuck task report + queue depth breakdown |
| `HighErrorRate` | Query Loki for recent ERROR logs, group by module | Error summary + top error classes + recommended investigation |
| `PostgreSQLHighConnections` | Query pg_stat_activity for idle/active breakdown | Connection audit + idle connection list + recommendation |
| `ServiceRestart` | Check health endpoints, compare before/after | Recovery status report |

### 4.3 Alert → Agent Flow

```
AlertManager fires webhook
       ↓
POST /api/alerts/ingest (auth validated, deduplicated)
       ↓
Store/update in infrastructure_alerts table
       ↓
If severity == critical:
  → Create ad-hoc agent heartbeat with context:
    "Alert: {alertname} on {service}. {description}.
     Investigate, summarize evidence, classify impact,
     and route recommended actions. Do NOT take
     remediation actions automatically."
  → Agent investigates using read-only tools
  → Agent stores findings in alert.agent_response
  → Agent reports findings to user via configured channel (PRD-55)
       ↓
If severity == warning:
  → Surface in Activity Command Centre feed
  → Include in next scheduled heartbeat context
       ↓
If status == resolved:
  → Update existing alert row with resolved_at
  → Surface resolution in Activity Command Centre
```

---

## 5. Railway Deployment Structure

Each monitoring component deploys as a separate Railway service. The repo contains a `docker-compose.yml` for **local development** and individual Dockerfiles/configs for Railway deployment.

### 5.1 Railway Services (Production)

| Railway Service | Image | Volume | Public | Notes |
|----------------|-------|--------|--------|-------|
| `prometheus` | `prom/prometheus:v2.51.0` | 5GB | No | Internal only |
| `grafana` | `grafana/grafana:10.4.0` | 1GB | **Yes** | Public domain for dashboard access |
| `loki` | `grafana/loki:2.9.4` | 10GB | **No** | Private — log-relay forwards to it internally |
| `log-relay` | Custom (see 2.4.1) | None | **Yes** | Public — receives Railway log drain webhooks |
| `alertmanager` | `prom/alertmanager:v0.27.0` | 500MB | No | Internal only |
| `postgres-exporter` | `prometheuscommunity/postgres-exporter:0.15.0` | None | No | Stateless |
| `redis-exporter` | `oliver006/redis_exporter:v1.58.0` | None | No | Stateless |

**Total services:** 7 (lean, purpose-built)

### 5.2 Docker Compose (Local Development)

For local development and testing, a `docker-compose.yml` is provided that mirrors the Railway setup using Docker networks instead of Railway private networking:

```yaml
services:
  prometheus:        # Metrics collection, 15-day retention
  grafana:           # Dashboards on port 3030
  loki:              # Log storage, 7-day retention
  log-relay:         # Railway log drain transformer (optional locally)
  alertmanager:      # Alert routing to webhooks
  postgres-exporter: # PostgreSQL metrics
  redis-exporter:    # Redis metrics

networks:
  automatos_network:
    external: true   # Shared with automatos-ai local compose

volumes:
  prometheus_data:
  grafana_data:
  loki_data:
```

> **Local vs Railway:** Locally, services use Docker DNS (`prometheus:9090`). On Railway, they use `prometheus.railway.internal:9090`. Config files use environment variable substitution to handle both.

---

## 6. File Structure

```
automatos-monitoring/
├── docker-compose.yml              # Local development
├── .env.example
├── README.md
├── services/
│   └── log-relay/
│       ├── Dockerfile              # Railway log drain → Loki transformer
│       ├── main.py                 # Lightweight HTTP relay
│       └── requirements.txt
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml          # Scrape config (env-var templated)
│   │   └── rules/
│   │       ├── infrastructure.yml  # Service health alerts
│   │       ├── database.yml        # PostgreSQL + Redis alerts
│   │       └── application.yml     # Backend + worker + agent alerts
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   ├── datasources/
│   │   │   │   └── datasources.yml
│   │   │   └── dashboards/
│   │   │       └── dashboards.yml
│   │   └── dashboards/
│   │       ├── platform-overview.json
│   │       ├── agent-performance.json
│   │       ├── database-health.json
│   │       ├── redis-queues.json
│   │       ├── workspace-worker.json
│   │       └── logs-explorer.json
│   ├── loki/
│   │   └── loki-config.yml
│   └── alertmanager/
│       └── alertmanager.yml
├── railway/
│   ├── prometheus.toml             # Railway service config
│   ├── grafana.toml
│   ├── loki.toml
│   ├── alertmanager.toml
│   ├── log-relay.toml
│   ├── postgres-exporter.toml
│   └── redis-exporter.toml
└── scripts/
    ├── setup-local.sh              # Local dev setup (create network, dirs)
    ├── setup-railway.sh            # Railway project/service creation via CLI
    └── health-check.sh             # Verify all monitoring services are up
```

**Files to DELETE from current repo:**
- `config/pgadmin/` — Not needed (adminer is in automatos-ai)
- `monitoring/prometheus/rules/xplaincrypto-alerts.yml` — Crypto-specific
- `monitoring/prometheus/rules/phase1-alerts.yml` — Crypto-specific references
- `monitoring/grafana/dashboards/crypto-overview.json` — Crypto-specific
- `monitoring/grafana/dashboards/xplaincrypto-overview.json` — Crypto-specific
- `monitoring/grafana/dashboards/n8n-*.json` — No n8n in Automatos
- `monitoring/grafana/dashboards/unified-platform.json` — Crypto references
- `monitoring/grafana/dashboards/infrastructure-testing.json` — Crypto-specific
- `monitoring/grafana/dashboards/platform-status-comprehensive.json` — Crypto-specific
- `monitoring/enhanced-n8n-exporter.py` — No n8n
- `monitoring/promtail/` — Replaced by Railway log drain + log-relay service
- `nginx/` — Railway handles routing, no nginx needed
- `tests/test_infrastructure.py` — Crypto-specific tests
- `docs/` — Will be rewritten
- `scripts/` — Will be rewritten for Railway

---

## 7. Configuration Requirements

### 7.1 Environment Variables (Railway Service Variables)

Each Railway service gets its own environment variables. Shared variables use Railway's variable referencing (`${{service.variable}}`).

**Prometheus:**
```bash
PROMETHEUS_RETENTION=15d
```

**Grafana:**
```bash
GF_SECURITY_ADMIN_PASSWORD=<strong-password>
GF_SERVER_ROOT_URL=https://grafana-automatos.up.railway.app
```

**Postgres Exporter:**
```bash
DATA_SOURCE_NAME=postgresql://postgres:${{postgres.POSTGRES_PASSWORD}}@postgres.railway.internal:5432/orchestrator_db?sslmode=disable
```

**Redis Exporter:**
```bash
REDIS_ADDR=redis://default:${{redis.REDIS_PASSWORD}}@redis.railway.internal:6379
```

**Log Relay:**
```bash
LOKI_PUSH_URL=http://loki.railway.internal:3100/loki/api/v1/push
ENVIRONMENT=production
```

**AlertManager:**
```bash
# No special env vars — config file handles routing
```

**Local Development (.env):**
```bash
GRAFANA_ADMIN_PASSWORD=<password>
POSTGRES_CONNECTION_STRING=postgresql://postgres:<pw>@postgres:5432/orchestrator_db
REDIS_PASSWORD=<redis-password>
ALERTMANAGER_WEBHOOK_URL=http://backend:8000/api/alerts/ingest
LOKI_URL=http://loki:3100
```

### 7.2 Prerequisites

1. **Railway project exists** with automatos-ai services already deployed (backend, postgres, redis)
2. **Private networking enabled** on the Railway project (enabled by default)
3. **Railway CLI** installed for deployment scripts (`railway login`, `railway link`)
4. Backend must expose `/api/alerts/ingest` endpoint (new — implemented as part of this PRD)
5. **Railway log drain** configured to point at the log-relay service's public URL

---

## 8. Deploy Runbook

See **[DEPLOY-RUNBOOK.md](../../../automatos-monitoring/DEPLOY-RUNBOOK.md)** for the full day-of execution plan with:
- Exact deploy order (Loki → log-relay → Prometheus → AlertManager → exporters → Grafana → backend ingest → log drain)
- Validation commands and pass/fail criteria per service
- Rollback points after each batch
- Common Railway gotchas
- Smoke tests for log pipeline, metrics pipeline, alert pipeline, and restart tolerance

---

## 9. Implementation Phases

### Phase 1: Core Stack (This PRD)

**Safe Phase 1 — must ship:**

| Task | Priority | Effort |
|------|----------|--------|
| Clean repo — remove all xplaincrypto/crypto references | P0 | S |
| Write docker-compose.yml for local dev (7 services) | P0 | M |
| Create Railway service configs (`.toml` files) | P0 | M |
| Build log-relay service (Railway log drain → Loki) | P0 | M |
| Configure Prometheus scrape targets (Railway private DNS) | P0 | S |
| Configure Loki with 7-day retention | P0 | S |
| Configure AlertManager with webhook routing + auth | P0 | M |
| Create infrastructure + database alert rules (exporter-based) | P0 | M |
| Deploy monitoring services to Railway | P0 | M |
| Configure Railway log drain → log-relay → Loki | P0 | S |
| Build Platform Overview dashboard | P1 | M |
| Build Database Health dashboard | P1 | M |
| Build Redis & Queues dashboard | P1 | S |
| Build Logs Explorer dashboard | P1 | S |
| Implement `/api/alerts/ingest` endpoint (with auth + dedupe) | P1 | M |
| Create `infrastructure_alerts` DB table + migration | P1 | S |
| Minimal `/metrics` endpoint on backend (`prometheus_client`) | P1 | M |
| README with local dev + Railway deployment instructions | P2 | S |

**Can defer without killing v1 (nice-to-haves):**

| Task | Reason to defer |
|------|----------------|
| Agent auto-investigation flows on critical alerts | Requires stable alert pipeline first |
| Application alert rules (HighErrorRate, SlowResponses) | Requires `/metrics` — can slip if `/metrics` isn't ready |
| Queue depth metrics (QueueBacklog, QueueCritical) | Requires worker `/metrics` or custom Redis key queries |
| Agent Performance dashboard | Placeholder until heartbeat metrics are instrumented |
| Workspace Worker dashboard | Placeholder until worker `/metrics` exists |
| Setup scripts (local + Railway) | Nice automation, not blocking |
| Health check script | Nice automation, not blocking |

### Phase 2: Enhanced Observability (Future PRD)

- Loki alerting rules (log-based alerts — ErrorSpike, OOMKill)
- OpenTelemetry distributed tracing
- Grafana alerting (unified with AlertManager)
- Dashboard annotations from deployments
- SLA/uptime tracking dashboard
- Cost monitoring dashboard (LLM token spend trends)
- Automated remediation agent actions (graduated from read-only)

---

## 10. Success Criteria

| Metric | Target |
|--------|--------|
| All 7 Railway monitoring services healthy | 100% uptime when automatos-ai is running |
| Prometheus scrape targets up | All configured targets returning metrics via Railway private network |
| Alert firing → webhook delivery | < 60s for critical, < 120s for warning |
| Log ingestion latency (Railway drain → Loki) | < 10s from log write to Loki queryable |
| Dashboard load time | < 3s for any dashboard |
| Zero crypto/xplaincrypto references | Clean repo audit passes |
| Agent can read alerts | `/api/alerts/ingest` stores and triggers correctly |
| Railway deployment reproducible | Fresh deploy from repo works in < 15 min |

---

## 11. Security Considerations

- **No hardcoded credentials** — All passwords via Railway environment variables (encrypted at rest)
- **Private networking** — Prometheus, AlertManager, exporters communicate only via Railway private network (not publicly accessible)
- **Public services hardened** — Only Grafana and log-relay have public domains. Loki is private (no public access). Grafana requires admin auth. Log-relay validates `X-Railway-Secret` header.
- **Alert ingest auth** — `/api/alerts/ingest` requires `Authorization: Bearer <ALERT_INGEST_TOKEN>` header. Token stored as Railway env var on AlertManager.
- **Grafana auth** — Admin password required, anonymous access disabled, consider SSO in Phase 2
- **Railway variable references** — Database passwords use `${{service.VAR}}` references, never copied as plain text
- **Exporters** — Read-only database access (create a `monitoring` Postgres role with `pg_monitor` grants)
- **Log relay auth** — Validate `X-Railway-Secret` header on incoming log drain requests to prevent spoofing
- **.env in .gitignore** — Never commit credentials (local dev only)

---

## 12. Relationship to Existing PRDs

| PRD | Relationship |
|-----|-------------|
| **PRD-06** (Monitoring & Analytics) | PRD-06 = application-level metrics in the UI. PRD-73 = infrastructure observability. Complementary, not competing. |
| **PRD-55** (Agent Heartbeats) | PRD-73 alerts can trigger heartbeats. Heartbeat results are scraped as metrics. |
| **PRD-72** (Activity Command Centre) | Infrastructure alerts surface in the Activity feed as system events. |
| **PRD-70** (Security Hardening) | PRD-73 implements monitoring best practices from the security audit. |
