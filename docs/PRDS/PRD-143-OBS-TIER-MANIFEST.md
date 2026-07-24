# PRD-143 — Observability-Tier Manifest (`super_admin_only`)

> **Status: PENDING SIGN-OFF — Gerard (PRD-143 Open Q1/Q2).**
> This is the human-readable source of truth for the `super_admin_only` tier.
> `orchestrator/tests/test_prd143_manifest_parity.py` enforces exact
> set-equality between this table and the live registry — adding or removing
> an su action without updating this file fails the suite.

The tier is locked to `system_role == 'super_admin'` (fail-closed, independent
of the autonomy dial). Every action NOT listed here is **operator** tier:
reachable by Auto at `autonomy=full` under gates-and-logs (PRD-140 hierarchy
checks, destructive backstop, Wave 4 audit + rollback, the human kill-switch),
not exclusion.

## super_admin_only actions (7)

| Action | Source | Why super-admin |
|---|---|---|
| `platform_query_loki_logs` | `orchestrator/modules/tools/discovery/actions_monitoring.py` | Obs read: centralized cross-service logs (Loki) expose platform internals across all tenants. |
| `platform_query_prometheus` | `orchestrator/modules/tools/discovery/actions_monitoring.py` | Obs read: raw PromQL over platform-wide metrics (uptime, error rates, DB, Redis). |
| `platform_get_alerts` | `orchestrator/modules/tools/discovery/actions_monitoring.py` | Obs read: the infrastructure alert stream is platform-level, not workspace-level. |
| `platform_get_logs` | `orchestrator/modules/tools/discovery/actions_monitoring.py` | Obs read: Railway deploy logs expose infrastructure internals. |
| `platform_list_services` | `orchestrator/modules/tools/discovery/actions_monitoring.py` | Obs read: the Railway service inventory is deployment topology. |
| `platform_get_system_health` | `orchestrator/modules/tools/discovery/actions_workspace.py` | Obs read: host-level health (CPU/memory/disk, DB, Redis, RAG) is platform internals. |
| `platform_set_autonomy_level` | `orchestrator/modules/tools/discovery/actions_autonomy.py` | Oversight control: the kill-switch dial stays HUMAN — Auto must never raise its own autonomy. |

`platform_get_autonomy_level` (read-only) stays **operator** — Auto may read
its own dial, never set it.

---

## Locked obs/analytics HTTP routers (13)

Router-wide `require_super_admin` (`orchestrator/core/auth/super_admin.py`) —
every endpoint on these routers returns **403 "Super admin only"** for any
principal that is not literally `system_role == 'super_admin'`, including
workspace admins/owners and API keys (`system_role='admin'`). The dashboards
backed by these routers 403 for non-super-admins — the ACCEPTED Rev 2
consequence (PRD-143 Open Q4).

### Batch 1 (S6)

| Router | Prefix |
|---|---|
| `orchestrator/api/heartbeat.py` | `/api/heartbeat` |
| `orchestrator/api/analytics.py` | `/analytics` |
| `orchestrator/api/analytics_api.py` | `/api/analytics` |
| `orchestrator/api/analytics_real.py` | `/api/analytics` |
| `orchestrator/api/analytics_charts.py` | `/api/analytics/charts` |

### Batch 2 (S7)

| Router | Prefix |
|---|---|
| `orchestrator/api/llm_analytics.py` (`router` + `admin_router`) | `/api/analytics/llm`, `/api/admin/analytics` |
| `orchestrator/api/memory_stats.py` | `/api/v1/memory` |
| `orchestrator/api/statistics.py` | `/api/system` |
| `orchestrator/api/composio_analytics.py` | `/api/analytics/composio` |
| `orchestrator/api/database_analytics.py` | `/api/database/analytics` |
| `orchestrator/api/execution_history.py` | `/api/execution-history` |
| `orchestrator/api/kpi_api.py` | `/api/kpi` |
| `orchestrator/api/reports.py` | `/api/reports` |

Prefix-collision notes for sign-off (Open Q1):

- `/api/v1/memory` is shared with `api/memory.py` (NOT locked). `memory_stats`
  is mounted first (`main.py:975`), so its routes — including `GET /health`,
  `POST /consolidate`, `DELETE /{memory_id}`, which shadow identical
  `api/memory.py` routes — are now su-locked. The shadowing itself predates
  this PRD; the lock changes who can reach the winning handler.
- `/api/system` is shared with `api/system.py` (NOT locked, mounted first at
  `main.py:974`). `api/system.py`'s routes (incl. `GET /metrics`) keep serving
  all authenticated users; only `statistics.py`'s route set is locked.
- `llm_analytics.py`'s scattered per-route admin checks (`_is_admin` /
  `_assert_admin` + bootstrap bypass) were deleted — superseded by the
  router-wide lock (they would otherwise 403 the super admin himself, since
  they tested `system_role == 'admin'`).
