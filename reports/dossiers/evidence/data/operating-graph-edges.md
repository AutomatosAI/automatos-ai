# Operating graph — learned edges, affinities, intent clusters

Captured 2026-07-04, read-only. `tool_routing_edges` = 29 rows, `tool_routing_affinities` = 232, `tool_routing_intent_clusters` = 20.

## Query 1 — all 29 learned edges

```sql
SELECT left(workspace_id::text,8), from_action, to_action, edge_type,
       round(weight::numeric,3), round(confidence::numeric,2), sample_count, last_updated::date
FROM tool_routing_edges ORDER BY weight DESC LIMIT 50;
```

Top rows (all 29 share: `ws=00000000`, `edge_type=used_after`, `last_updated=2026-05-05`):

| from_action | to_action | weight | conf | samples |
|---|---|---:|---:|---:|
| platform_list_agents | platform_get_agent | 34.0 | 0.90 | 34 |
| platform_list_documents | search_knowledge | 29.0 | 0.88 | 29 |
| platform_list_playbooks | platform_execute_playbook | 27.0 | 0.88 | 27 |
| platform_list_agents | platform_create_agent | 26.0 | 0.87 | 26 |
| platform_workspace_stats | platform_get_success_rate | 23.0 | 0.86 | 23 |
| platform_search_memory | platform_store_memory | 22.0 | 0.85 | 22 |
| workspace_read_file | workspace_write_file | 14.0 | 0.78 | 14 |
| platform_get_llm_usage | platform_get_cost_breakdown | 12.0 | 0.76 | 12 |
| platform_get_latest_report | platform_submit_report | 11.0 | 0.74 | 11 |
| platform_list_missions | platform_create_mission | 9.0 | 0.70 | 9 |
| ... (19 more, duplicate from→to pairs appear up to 3× as separate rows) | | | | |

Workspace scope check: **no real workspace has a single learned edge** — every edge belongs to the all-zeros (synthetic seed) workspace. Duplicate (from,to) pairs exist as separate rows (e.g. `platform_list_agents→platform_get_agent` ×3 with weights 34/16/13), so uniqueness is not enforced per (workspace, from, to).

## Query 2 — affinities grouped

```sql
SELECT coalesce(left(workspace_id::text,8),'GLOBAL'), affinity_type, count(*),
       round(avg(weight)::numeric,3), sum(sample_count), max(last_updated)::date
FROM tool_routing_affinities GROUP BY 1,2 ORDER BY 3 DESC;
```

| ws | affinity_type | n | avg weight | samples | newest |
|---|---|---:|---:|---:|---|
| 00000000 | agent_prefers | 103 | 0.029 | 2,331 | 2026-05-05 |
| 00000000 | succeeds_for_intent | 74 | 25.784 | 1,908 | 2026-05-05 |
| 00000000 | fails_for_intent | 55 | 7.455 | 410 | 2026-05-05 |

## Query 3 — all 20 intent clusters

```sql
SELECT id, left(sample_query,60), action_names_hot[1:3], sample_count, embedding_model_key, last_updated::date
FROM tool_routing_intent_clusters ORDER BY sample_count DESC LIMIT 20;
```

| id | sample_query | hot actions (top-3) | samples |
|---|---|---|---:|
| 10 | show me my knowledge base | platform_list_documents, platform_get_memory_stats, platform_create_agent | 505 |
| 6 | what agents do I have? | platform_list_agents, platform_browse_marketplace_agents, platform_get_agent | 330 |
| 19 | read main.py from the repo | workspace_read_file, workspace_exec, platform_create_playbook | 188 |
| 18 | how much have I spent on tokens this week? | platform_get_cost_breakdown, platform_get_llm_usage, platform_get_activity_feed | 154 |
| 4 | how's mission 42 going? | platform_create_mission, platform_get_mission, platform_get_success_rate | 134 |
| 14 | send an email to j***@acme.com about the demo | composio_execute, platform_list_connected_apps | 126 |
| 1 | tell me about the SENTINEL agent | platform_create_agent, platform_list_agents, platform_get_agent | 95 |
| ... 13 more, all `embedding_model_key = openrouter:qwen/qwen3-embedding-8b:2048`, all `last_updated = 2026-05-05` | | | |

## First look

The learned operating graph exists and is well-shaped (weights, confidence, sample counts, per-workspace scoping columns as PRD-177/W7 requires) — but 100% of its content was learned from the synthetic seed batch in the all-zeros workspace and froze on 2026-05-05. Per-tenant it is empty: for the 21 real workspaces the "learn-from-usage" loop has never produced an edge, because its input (`tool_execution_logs`) receives no organic writes (see tool-telemetry.md). Some clusters also look noisy even on synthetic data (cluster 10 "show me my knowledge base" has `platform_create_agent` as a hot action; cluster 19 pairs `read main.py` with `platform_create_playbook`).
