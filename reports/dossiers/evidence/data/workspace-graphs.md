# workspace_graphs — per-workspace graph store

Captured 2026-07-04, read-only. **155 rows** (schema: `workspace_id, path, content(text), updated_at`).

## Query

```sql
SELECT left(workspace_id::text,8), path, length(content), updated_at::date
FROM workspace_graphs ORDER BY updated_at DESC LIMIT 20;
```

| ws | path | bytes | updated |
|---|---|---:|---|
| fa4125e0 | graph/graph.json | 196,580 | 2026-07-01 |
| fa4125e0 | graph/communities.json | 18,196 | 2026-07-01 |
| fa4125e0 | graph/history/2026-07-01_graph.json | 188,539 | 2026-07-01 |
| fa4125e0 | graph/latest_diff.json | 2,618 | 2026-07-01 |
| fa4125e0 | graph/meta.json | 145 | 2026-07-01 |
| fa4125e0 | graph/reports/2026-07-01_build.md | 211 | 2026-07-01 |
| ae8320bc | graph/graph.json | **19,614,690** | 2026-07-01 |
| ae8320bc | graph/history/2026-07-01_graph.json | 19,585,487 | 2026-07-01 |
| ae8320bc | graph/communities.json | 1,986,879 | 2026-07-01 |
| 28a228aa | graph/graph.json | **42,408,361** | 2026-07-01 |
| 28a228aa | graph/history/2026-07-01_graph.json | 41,800,894 | 2026-07-01 |
| 28a228aa | graph/communities.json | 1,152,349 | 2026-07-01 |
| 894519f4 | graph/reports/2026-07-01_build.md + latest_diff.json | 210 / 5,882 | 2026-07-01 |

## First look

This is the freshest learning-adjacent surface in the DB: per-workspace graph bundles (graph.json / communities.json / meta / diff / build report + dated history) rebuilt on 2026-07-01 across multiple workspaces. Sizes are heavy-tailed — one workspace's graph.json is **42 MB stored as a TEXT column**, and history snapshots double the footprint per rebuild; row-per-file-in-Postgres will not age well if rebuilds are frequent. Unlike the `knowledge_nodes/edges` tables (0 rows), this blob store is where the product's graph actually lives.
