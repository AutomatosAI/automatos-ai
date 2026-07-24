# board_tasks — the Board (task claim/dispatch plane)

Captured 2026-07-04, read-only. **1,449 rows.**

## Query 1 — status distribution

```sql
SELECT status, count(*), min(created_at)::date, max(created_at)::date
FROM board_tasks GROUP BY status ORDER BY 2 DESC;
```

| status | n | oldest | newest |
|---|---:|---|---|
| done | 1,349 | 2026-03-27 | 2026-07-03 |
| blocked | 42 | 2026-03-29 | 2026-06-06 |
| inbox | 38 | 2026-03-30 | 2026-06-16 |
| todo | 17 | 2026-03-29 | 2026-05-06 |
| failed | 2 | 2026-07-03 | 2026-07-03 |
| review | 1 | 2026-06-13 | 2026-06-13 |

## Query 2 — recent 30

```sql
SELECT id, left(title,48), status, priority, assigned_agent_id, created_by_type, attempts,
       (error_message IS NOT NULL), (result IS NOT NULL), created_at, completed_at
FROM board_tasks ORDER BY created_at DESC LIMIT 30;
```

| id | title | status | by | has_err | has_result | created→done |
|---|---|---|---|---|---|---|
| 2065 | Review FIXER heartbeat — {'status': 'error', ... | **failed** | agent | t | f | 07-03 |
| 2064 | Review VECTOR heartbeat — {'status': 'error', ... | **failed** | agent | t | f | 07-03 |
| 2063 | Recipe: Daily social analytics report | done | recipe | **t** | **f** | 07-03 |
| 2062 | Recipe: X Carousel - Daily Fact Post | done | recipe | **t** | **f** | 07-03 |
| 2061 | Recipe: Instagram Carousel - Daily Fact Post | done | recipe | **t** | **f** | 07-03 |
| 2060 | Recipe: Nightly Test Pipeline | done | recipe | **t** | **f** | 07-03 |
| 2059 | Review FIXER heartbeat — {'status': 'error', ... | done | agent | f | f | 07-02 |
| 2057 | Recipe: Daily social analytics report | done | recipe | **t** | **f** | 07-02 |
| 2056 | Recipe: X Carousel - Daily Fact Post | done | recipe | **t** | **f** | 07-02 |
| 2051 | Recipe: Daily social analytics report | done | recipe | **t** | **f** | 07-01 |
| 2045 | Recipe: Weekly Growth Strategist Review | done | recipe | **t** | **f** | 06-30 |
| ... same daily pattern back through 06-29 | | | | | | |

(Naming note: `Recipe:` prefix and `created_by_type='recipe'` survive in data even though the canonical product term is Playbook.)

## Query 3 — failed-marked-done probe

```sql
SELECT count(*) FILTER (WHERE status='done' AND error_message IS NOT NULL),
       count(*) FILTER (WHERE status='done' AND (result IS NULL OR length(result)=0)),
       count(*) FILTER (WHERE status='done' AND result ILIKE '%error%'),
       count(*) FILTER (WHERE status='done')
FROM board_tasks;
```

| done with error_message | done with no result | done, result mentions "error" | done total |
|---:|---:|---:|---:|
| **194** | **484** | 264 | 1,349 |

## First look

The Board is genuinely alive (daily playbook-cron tasks through 07-03) — and **failed-marked-done is real**: the daily "Recipe:" playbook tasks whose LLM step has been failing with OpenRouter 402 since at least 07-01 (see memory-short-term.md) are closed as `done` with an `error_message` set and no `result`. Overall 194/1,349 done tasks carry an error message and 484/1,349 have no result at all. Only 2 tasks in the table's whole history are `failed`, both on 2026-07-03 — which suggests a recent wave (PRD-161's failed status / W1 heartbeat lanes) only just started routing real failures to `failed`, while the playbook completion path still ignores step errors. There is also self-referential churn: agents raise daily "Review X heartbeat — {'status':'error'}" tasks about their own failing heartbeats.
