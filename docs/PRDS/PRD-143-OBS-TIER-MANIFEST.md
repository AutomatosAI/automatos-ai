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

*Locked obs/analytics HTTP routers (13) are appended by S6/S7.*
