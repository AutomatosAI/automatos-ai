# tool_execution_logs — tool telemetry

Captured 2026-07-04, read-only. **2,341 rows total.**

## Query 1 — status × source distribution

```sql
SELECT status, coalesce(telemetry_source,'-'), coalesce(routing_source,'-'), count(*),
       min(executed_at)::date, max(executed_at)::date
FROM tool_execution_logs GROUP BY 1,2,3 ORDER BY 4 DESC;
```

| status | telemetry_source | routing_source | n | oldest | newest |
|---|---|---|---:|---|---|
| success | synthetic | keyword | 1,911 | 2026-04-05 | 2026-05-05 |
| error | synthetic | keyword | 430 | 2026-04-05 | 2026-05-05 |

**Every row in the table is `telemetry_source='synthetic'`.** There is not a single organically-recorded tool execution.

## Query 2 — recent 25 (all synthetic; email in fixture query redacted)

```sql
SELECT id, executed_at, app_name, action_name, status, execution_time_ms,
       left(workspace_id::text,8), left(user_query,60), left(error_message,40)
FROM tool_execution_logs ORDER BY executed_at DESC LIMIT 25;
```

| at | app | action | status | ms | ws | query |
|---|---|---|---|---:|---|---|
| 05-05 09:25 | PLATFORM | platform_list_documents | error | 1747 | 00000000 | find the contract from Acme in the docs |
| 05-05 09:05 | PLATFORM | platform_submit_report | success | 676 | 00000000 | submit a status report on what I just did |
| 05-05 06:47 | COMPOSIO | composio_execute | success | 816 | 00000000 | send an email to j***@acme.com about the demo |
| 05-05 06:15 | WORKSPACE | workspace_write_file | success | 275 | 00000000 | read main.py from the repo |
| 05-05 05:34 | PLATFORM | platform_create_agent | error | 2415 | 00000000 | wie viele Agenten habe ich? |
| 05-05 04:42 | PLATFORM | platform_create_mission | success | 800 | 00000000 | kick off a mission to write a competitive analysis |
| 05-05 03:26 | PLATFORM | platform_get_memory_stats | success | 1108 | 00000000 | what's in your long-term memory for this workspace? |
| 05-05 02:57 | PLATFORM | platform_list_agents | success | 41 | 00000000 | tell me about the SENTINEL agent |
| ... | | | | | | (all 25 rows: workspace UUID all-zeros, queries verbatim from the seeded eval fixtures) |

The queries match the repo's synthetic seed corpus — `tell me about the SENTINEL agent` is `q004` in `orchestrator/scripts/eval/tool_routing/eval_set.jsonl` (pinned tree), and the seeded date window (2026-04-05 → 2026-05-05) matches `orchestrator/scripts/eval/tool_routing/seed_telemetry.py` generating a synthetic month of history into the all-zeros workspace.

## First look

Tool telemetry is 100% synthetic seed data, frozen on 2026-05-05. Two months of live operation since then (playbooks ran daily, agents heartbeat, chat sessions happened through 06-27) produced **zero** organic rows — the real execution paths do not write to `tool_execution_logs`, so everything downstream of it (learned routing edges, affinities, intent clusters, uplift evals against "production telemetry") has no real signal to learn from. This substantiates and extends the July OS review's Composio-lane observation: it is not just the Composio lane — no lane records telemetry in practice.
