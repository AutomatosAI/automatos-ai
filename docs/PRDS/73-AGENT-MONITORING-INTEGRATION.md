# PRD-73: Agent Monitoring Integration Guide

**For:** Auto CTO & Agent Configuration
**Status:** Live — Stack Deployed on Railway
**Date:** 2026-03-09

---

## What's Running

7 monitoring services are live on Railway, all in the same project as the Automatos platform:

| Service | Internal Address | Purpose |
|---------|-----------------|---------|
| Prometheus | `prometheus.railway.internal:9090` | Scrapes metrics every 15s from all services |
| Grafana | `grafana.railway.internal:3000` | Dashboards — also public at `https://grafana-production-5f61.up.railway.app` |
| Loki | `loki.railway.internal:3100` | Log storage (7-day retention) |
| Log-Relay | `log-relay.railway.internal:8080` | Receives logs from services, forwards to Loki |
| AlertManager | `alertmanager.railway.internal:9093` | Routes alerts by severity to backend webhook |
| Postgres Exporter | `postgres-exporter.railway.internal:9187` | Exports PostgreSQL metrics |
| Redis Exporter | `redis-exporter.railway.internal:9121` | Exports Redis metrics |

---

## 1. How Logs Flow (Agent-Readable)

```
Any Service (Python logger)
    ↓ POST /push (JSON)
log-relay.railway.internal:8080
    ↓ batches + transforms
loki.railway.internal:3100
    ↓ queryable via
Grafana Logs Explorer dashboard
```

### Sending Logs from Any Service

Every Python service uses the `automatos_logging` handler installed at `orchestrator/core/monitoring/automatos_logging.py`:

```python
from core.monitoring.automatos_logging import setup_logging
setup_logging(service="your-service-name")

import logging
logger = logging.getLogger(__name__)
logger.info("Task completed", extra={"agent_id": "cto", "duration_ms": 1234})
```

**Environment variables** (set on each Railway service):
```
LOG_RELAY_URL=http://log-relay.railway.internal:8080/push
LOG_RELAY_ENABLED=true
SERVICE_NAME=automatos-backend   # or agent-opt-worker, etc.
```

### Log-Relay Push Format

Direct POST to `http://log-relay.railway.internal:8080/push`:
```json
{
  "service": "automatos-backend",
  "level": "error",
  "message": "Agent heartbeat failed for cto-agent",
  "extra": {
    "agent_id": "cto",
    "module": "heartbeat",
    "request_id": "abc-123"
  }
}
```

Supports arrays for batching:
```json
[
  {"service": "automatos-backend", "level": "info", "message": "..."},
  {"service": "automatos-backend", "level": "error", "message": "..."}
]
```

### Querying Logs (for Agents)

Agents can query Loki directly via its HTTP API:

```
GET http://loki.railway.internal:3100/loki/api/v1/query_range
  ?query={service="automatos-backend",level="error"}
  &start=<unix_nano>
  &end=<unix_nano>
  &limit=100
```

**Label filters available:**
- `service` — service name (automatos-backend, agent-opt-worker, etc.)
- `level` — debug, info, warning, error, critical
- `environment` — production, development
- `source` — direct-push, railway-drain
- `module` — extracted from extra fields

---

## 2. How Metrics Flow

```
Automatos Backend (/metrics)  ←── Prometheus scrapes every 15s
PostgreSQL (via exporter)     ←── Prometheus scrapes every 15s
Redis (via exporter)          ←── Prometheus scrapes every 15s
    ↓
Prometheus evaluates alert rules
    ↓
AlertManager routes by severity
    ↓
POST /api/alerts/ingest → infrastructure_alerts table
```

### Prometheus Scrape Targets

| Target | Address | Metrics Available |
|--------|---------|-------------------|
| Backend API | `automatos-ai.railway.internal:8000/metrics` | HTTP request count, latency, error rate, in-progress requests |
| Agent Worker | `agent-opt-worker.railway.internal:8080/metrics` | Same HTTP metrics (when instrumented) |
| PostgreSQL | `postgres-exporter.railway.internal:9187/metrics` | Connections, dead tuples, cache hit ratio, DB size, replication lag |
| Redis | `redis-exporter.railway.internal:9121/metrics` | Memory usage, connected clients, evicted keys, command latency |
| Prometheus | `prometheus.railway.internal:9090/metrics` | Self-monitoring |
| Loki | `loki.railway.internal:3100/metrics` | Ingestion rate, query performance |
| AlertManager | `alertmanager.railway.internal:9093/metrics` | Alert delivery stats |

### Custom Application Metrics (in code)

Available in `orchestrator/core/monitoring/automatos_metrics.py`:

```python
from core.monitoring.automatos_metrics import (
    AGENT_HEARTBEATS,
    AGENT_HEARTBEAT_DURATION,
    AGENT_TOKEN_USAGE,
    ACTIVE_AGENTS,
    WORKER_ACTIVE_TASKS,
    WORKER_TASK_DURATION,
    WORKER_QUEUE_DEPTH,
    WORKER_TASK_TOTAL,
    WORKER_ERRORS,
    LLM_REQUEST_DURATION,
    LLM_TOKEN_USAGE,
)

# Example: instrument agent heartbeats
AGENT_HEARTBEATS.labels(agent_id="cto", status="success").inc()
AGENT_HEARTBEAT_DURATION.labels(agent_id="cto").observe(1.234)

# Example: track LLM usage
LLM_TOKEN_USAGE.labels(model="claude-sonnet-4-6", provider="anthropic", direction="input").inc(1500)
LLM_REQUEST_DURATION.labels(model="claude-sonnet-4-6", provider="anthropic").observe(2.5)

# Example: worker tasks
WORKER_ACTIVE_TASKS.inc()
WORKER_TASK_TOTAL.labels(task_type="code_generation", status="success").inc()
WORKER_ACTIVE_TASKS.dec()
```

### Querying Metrics (for Agents)

Agents can query Prometheus directly:

```
# Instant query
GET http://prometheus.railway.internal:9090/api/v1/query
  ?query=up{job="automatos-backend"}

# Range query (last hour)
GET http://prometheus.railway.internal:9090/api/v1/query_range
  ?query=rate(automatos_http_requests_total{status_code=~"5.."}[5m])
  &start=<unix>&end=<unix>&step=60

# Get all active alerts
GET http://prometheus.railway.internal:9090/api/v1/alerts
```

**Useful PromQL queries for agents:**
```promql
# Is the backend up?
up{job="automatos-backend"}

# HTTP error rate (5xx) over 5 minutes
rate(automatos_http_requests_total{status_code=~"5.."}[5m])
/ rate(automatos_http_requests_total[5m])

# p95 response time
histogram_quantile(0.95, rate(automatos_http_request_duration_seconds_bucket[5m]))

# PostgreSQL connection count
pg_stat_activity_count

# Redis memory usage in MB
redis_memory_used_bytes / 1024 / 1024

# Redis evicted keys (bad — means OOM pressure)
rate(redis_evicted_keys_total[5m])

# Active agent heartbeat failure rate
rate(automatos_agent_heartbeat_total{status="error"}[1h])
/ rate(automatos_agent_heartbeat_total[1h])
```

---

## 3. How Alerts Flow

```
Prometheus rule triggers
    ↓
AlertManager groups + routes by severity
    ↓
POST http://automatos-ai.railway.internal:8000/api/alerts/ingest
  Authorization: Bearer <ALERT_INGEST_TOKEN>
  X-Alert-Source: alertmanager
  X-Alert-Priority: critical|warning|info
    ↓
infrastructure_alerts table (deduplicated by fingerprint)
```

### Active Alert Rules

#### Infrastructure (fire immediately)
| Alert | Severity | Condition | Meaning |
|-------|----------|-----------|---------|
| ServiceDown | critical | `up == 0` for 1m | Any monitored service unreachable |
| BackendDown | critical | Backend `up == 0` for 30s | API server down |
| WorkerDown | critical | Worker `up == 0` for 2m | Agent worker down |

#### PostgreSQL
| Alert | Severity | Condition |
|-------|----------|-----------|
| PostgreSQLDown | critical | Exporter can't reach PG for 30s |
| HighConnections | warning | >150 connections for 5m |
| SlowQueries | warning | Transactions active >1s for 5m |
| DeadTuples | warning | >10k dead tuples for 10m |
| CacheHitLow | warning | Cache hit ratio <95% for 10m |
| DBSizeGrowth | info | >1GB growth in 24h |

#### Redis
| Alert | Severity | Condition |
|-------|----------|-----------|
| RedisDown | critical | Exporter can't reach Redis for 30s |
| RedisHighMemory | warning | >200MB (of 256MB) for 5m |
| RedisHighMemoryCritical | critical | >240MB for 2m |
| RedisEvictedKeys | warning | Key evictions detected in 5m |
| RedisHighLatency | warning | Avg command latency >10ms |
| RedisHighClients | warning | >100 connected clients for 5m |

#### Application (requires /metrics — now deployed)
| Alert | Severity | Condition |
|-------|----------|-----------|
| HighErrorRate | warning | >5% 5xx responses for 5m |
| SlowResponses | warning | p95 response time >5s |

### Querying Alerts (for Agents)

**From the database (preferred — has full history):**
```sql
-- Active firing alerts
SELECT alertname, severity, service, annotations, created_at
FROM infrastructure_alerts
WHERE status = 'firing'
ORDER BY created_at DESC;

-- Recent critical alerts (last 24h)
SELECT * FROM infrastructure_alerts
WHERE severity = 'critical' AND created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

-- Alert frequency by type (last 7 days)
SELECT alertname, severity, COUNT(*) as occurrences
FROM infrastructure_alerts
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY alertname, severity
ORDER BY occurrences DESC;
```

**Via API:**
```
GET /api/alerts?status=firing
GET /api/alerts?severity=critical&limit=10
```

**From AlertManager directly (current state only):**
```
GET http://alertmanager.railway.internal:9093/api/v2/alerts
```

---

## 4. SENTINEL Pattern — Agent Investigation (Read-Only v1)

When a critical alert is stored, agents should investigate but **NOT auto-remediate**.

### Investigation Playbook per Alert

| Alert | Agent Action | Tools to Use |
|-------|-------------|-------------|
| `RedisHighMemory` | Query Redis INFO, find large keys, report top consumers | Redis CLI via workspace tools |
| `RedisHighMemoryCritical` | Same as above + check eviction policy + flag urgency | Redis CLI |
| `PostgreSQLHighConnections` | Query `pg_stat_activity`, find idle connections, report breakdown | SQL query tool |
| `PostgreSQLDown` | Check if exporter is up, check PG logs in Loki, report status | Loki query + health endpoint |
| `HighErrorRate` | Query Loki for ERROR logs in last 15m, group by module, summarize | Loki HTTP API |
| `SlowResponses` | Query Prometheus for slow endpoints, check for correlation with load | Prometheus HTTP API |
| `ServiceDown` | Check health endpoint, query Loki for crash logs, report last known state | Health endpoint + Loki |
| `BackendDown` | Escalate immediately — check Railway deployment status, recent deploys | Railway API / health check |

### Agent Response Format

Store investigation results in `infrastructure_alerts.agent_response` (JSONB):

```json
{
  "investigated_at": "2026-03-09T10:30:00Z",
  "agent_id": "sentinel-cto",
  "classification": "resource_pressure",
  "impact": "high",
  "evidence": [
    "Redis memory at 242MB/256MB",
    "Top keys: session:* (120MB), queue:agent-tasks (45MB)",
    "Eviction rate: 12 keys/min and rising"
  ],
  "recommended_actions": [
    "Flush expired sessions (> 24h old)",
    "Increase Redis memory limit to 512MB",
    "Review queue:agent-tasks retention policy"
  ],
  "auto_remediation_safe": false,
  "escalation_required": true
}
```

### Triggering Investigation

When the alerts ingest endpoint receives a `critical` alert, trigger an agent heartbeat:

```python
# In the alert ingest handler (after storing the alert):
if alert.severity == "critical":
    # Create investigation task for SENTINEL agent
    investigation_prompt = (
        f"ALERT: {alert.alertname} on {alert.service}\n"
        f"Severity: {alert.severity}\n"
        f"Summary: {alert.annotations.get('summary', 'No summary')}\n"
        f"Description: {alert.annotations.get('description', 'No description')}\n\n"
        f"Investigate this alert. Query the relevant service for evidence. "
        f"Classify the impact, summarize findings, and recommend actions. "
        f"Do NOT take any remediation actions automatically."
    )
    # Route to agent heartbeat system (PRD-55)
```

---

## 5. Grafana Dashboards

**URL:** `https://grafana-production-5f61.up.railway.app`
**Login:** admin / (check GF_SECURITY_ADMIN_PASSWORD env var on grafana service)

### Available Dashboards

| Dashboard | Status | What It Shows |
|-----------|--------|---------------|
| Platform Overview | **Active** | All services up/down, high-level health |
| Database Health | **Active** | PostgreSQL connections, cache hit, dead tuples, DB size |
| Redis & Queues | **Active** | Redis memory, clients, evictions, latency |
| Logs Explorer | **Active** | Full-text log search across all services |
| Agent Performance | Placeholder | Waiting for custom metrics instrumentation |
| Workspace Worker | Placeholder | Waiting for worker metrics instrumentation |

### Embedding Dashboards

Grafana panels can be embedded in the Automatos UI:
```
https://grafana-production-5f61.up.railway.app/d-solo/<dashboard-uid>/<dashboard-slug>?panelId=<id>&orgId=1
```

Set `GF_SECURITY_ALLOW_EMBEDDING=true` and configure CSP if embedding.

---

## 6. Wiring Checklist for New Services

When adding a new Automatos service to monitoring:

### 1. Logging
```bash
# Set env vars on the new Railway service:
railway variable set \
  LOG_RELAY_URL="http://log-relay.railway.internal:8080/push" \
  LOG_RELAY_ENABLED="true" \
  SERVICE_NAME="<new-service-name>"
```

In the service code:
```python
from core.monitoring.automatos_logging import setup_logging
setup_logging(service="<new-service-name>")
```

### 2. Metrics (if Python/FastAPI)
```python
from core.monitoring.automatos_metrics import setup_metrics
setup_metrics(app, service_name="<new-service-name>")
```

### 3. Add to Prometheus Scrape Config

Edit `services/prometheus/prometheus-railway.yml` in `automatos-monitoring`:
```yaml
- job_name: "<new-service-name>"
  static_configs:
    - targets: ["<service>.railway.internal:<port>"]
  metrics_path: "/metrics"
  scrape_interval: 15s
```

### 4. Add Alert Rules (if needed)

Create or update rules in `services/prometheus/rules/` in `automatos-monitoring`.

### 5. Add Grafana Dashboard (if needed)

Create JSON dashboard in `services/grafana/dashboards/` in `automatos-monitoring`.

---

## 7. Environment Variables Reference

### On Backend (automatos-ai-api)
| Variable | Value | Purpose |
|----------|-------|---------|
| `LOG_RELAY_URL` | `http://log-relay.railway.internal:8080/push` | Where to ship logs |
| `LOG_RELAY_ENABLED` | `true` | Enable/disable log shipping |
| `SERVICE_NAME` | `automatos-backend` | Service identifier in logs/metrics |
| `ALERT_INGEST_TOKEN` | `<token>` | Shared secret with AlertManager |

### On Agent Worker
| Variable | Value | Purpose |
|----------|-------|---------|
| `LOG_RELAY_URL` | `http://log-relay.railway.internal:8080/push` | Where to ship logs |
| `LOG_RELAY_ENABLED` | `true` | Enable/disable log shipping |
| `SERVICE_NAME` | `agent-opt-worker` | Service identifier |

### On Monitoring Services
| Service | Key Variables |
|---------|--------------|
| log-relay | `LOKI_PUSH_URL`, `LOG_RELAY_SECRET`, `PORT=8080` |
| grafana | `GF_SECURITY_ADMIN_PASSWORD`, `GF_SERVER_ROOT_URL`, `PORT=3000` |
| alertmanager | `ALERT_INGEST_TOKEN`, `PORT=9093` |
| prometheus | `PORT=9090` |
| loki | `PORT=3100` |
| postgres-exporter | `DATA_SOURCE_NAME=<postgres_dsn>` |
| redis-exporter | `REDIS_ADDR=<redis_url>` |

---

## 8. Repository Structure (automatos-monitoring)

All monitoring infrastructure is defined as code in `AutomatosAI/automatos-monitoring`:

```
automatos-monitoring/
├── services/
│   ├── prometheus/
│   │   ├── Dockerfile
│   │   ├── prometheus-railway.yml    # Scrape targets (Railway DNS)
│   │   ├── prometheus.yml            # Local dev scrape targets
│   │   ├── railway.toml
│   │   └── rules/
│   │       ├── infrastructure.yml    # ServiceDown, BackendDown, WorkerDown
│   │       ├── database.yml          # PostgreSQL + Redis alerts
│   │       └── application.yml       # HTTP error rate, slow responses, agent heartbeats
│   ├── grafana/
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh             # Fixes volume permissions on Railway
│   │   ├── railway.toml
│   │   ├── datasources-railway.yml   # Prometheus + Loki endpoints
│   │   ├── provisioning/             # Auto-provisioned datasources + dashboard config
│   │   └── dashboards/               # 6 JSON dashboards (4 active, 2 placeholder)
│   ├── loki/
│   │   ├── Dockerfile
│   │   ├── loki-config.yaml          # 7-day retention, TSDB v13, filesystem storage
│   │   └── railway.toml
│   ├── log-relay/
│   │   ├── Dockerfile
│   │   ├── main.py                   # aiohttp service: /drain, /push, /health
│   │   ├── requirements.txt
│   │   └── railway.toml
│   └── alertmanager/
│       ├── Dockerfile
│       ├── alertmanager-railway.yml  # Routes to /api/alerts/ingest with Bearer auth
│       ├── alertmanager.yml          # Local dev config
│       ├── entrypoint.sh             # Writes ALERT_INGEST_TOKEN to file at startup
│       └── railway.toml
├── clients/python/                   # Drop-in client libraries
│   ├── automatos_logging.py          # Log handler → log-relay → Loki
│   ├── automatos_metrics.py          # Prometheus /metrics + middleware
│   └── automatos_alerts.py           # Alert ingest endpoint + DB table
├── docker-compose.yml                # Local dev stack (all 7 services)
└── DEPLOY-RUNBOOK.md                 # Production deploy order + validation
```

---

## TL;DR for Auto

1. **Logs** → Query Loki at `loki.railway.internal:3100` or use Grafana Logs Explorer
2. **Metrics** → Query Prometheus at `prometheus.railway.internal:9090` with PromQL
3. **Alerts** → Read from `infrastructure_alerts` table or `GET /api/alerts`
4. **Investigation** → On critical alerts, investigate using read-only tools, store findings in `agent_response` column
5. **New services** → Set 3 env vars + 2 lines of Python + add to prometheus scrape config
6. **Everything is code** → All config lives in `automatos-monitoring` repo, rebuildable anywhere
