# Missions — orchestration_runs / orchestration_tasks

Captured 2026-07-04, read-only. 17 runs, 122 tasks.

## Query 1 — all 17 runs

```sql
SELECT left(id::text,8), left(workspace_id::text,8), left(goal,55), state,
       coalesce(stop_reason,'-'), tokens_used, replan_count, created_at::date
FROM orchestration_runs ORDER BY created_at DESC LIMIT 20;
```

| id | ws | goal | state | stop | tokens | created |
|---|---|---|---|---|---:|---|
| f10261d3 | ae8320bc | Research and write a blog post about AI Agent Incident | awaiting_approval | - | 0 | 2026-06-13 |
| 5b41a2ea | ae8320bc | Create blog: The Shopify AI Edge | completed | completed | 264,193 | 2026-06-06 |
| 433641e4 | ae8320bc | Create a draft blog post titled "The Shopify AI Edge". | awaiting_approval | - | 0 | 2026-06-06 |
| 7627df27 | ae8320bc | Build a detailed research-led blog post based on the ex | cancelled | human_cancelled | 141,435 | 2026-06-06 |
| 3da461d8 | ae8320bc | Build a detailed research-led blog post based on the ex | failed | dependency_failed | 74,513 | 2026-06-06 |
| 44502708 | ae8320bc | Research and write a high-quality blog post about "Beyo | completed | completed | 430,423 | 2026-06-03 |
| f88c8633 | ae8320bc | ...blog post about "How | awaiting_approval | - | 0 | 2026-05-30 |
| 6db34a0d | ae8320bc | Review VECTOR weekly memo — content engine running blin | awaiting_approval | - | 0 | 2026-05-26 |
| 265f1b54 | ae8320bc | Render and publish Week 2026-W21 Instagram carousel con | awaiting_approval | - | 0 | 2026-05-21 |
| bf0d4257 | ae8320bc | Review VECTOR weekly memo — Content machine running at | awaiting_approval | - | 0 | 2026-05-09 |
| 15340425 | ae8320bc | Remediate the org health findings by enforcing the new | awaiting_approval | - | 0 | 2026-05-08 |
| 8665f828 / 29559e5b | ae8320bc | blog post about 'The ...' | awaiting_approval | - | 0 | 04-29 / 04-25 |
| fcd2dbc8, 55212c46, 9f2777bc, 3f37efee | ae8320bc | BENCHMARK MISSION: multi-domain research | paused ×4 | - | 95k–163k | 2026-03-30 |

States: 8 awaiting_approval / 4 paused / 2 completed / 1 failed / 1 cancelled / 1 (of the above) — all 17 in ONE workspace (ae8320bc).

## Query 2 — recent 20 tasks

```sql
SELECT left(run_id::text,8), left(title,42), state, coalesce(failure_reason_code,'-'),
       attempt_number, tokens_used, created_at::date
FROM orchestration_tasks ORDER BY created_at DESC LIMIT 20;
```

| run | title | state | fail_code | tokens |
|---|---|---|---|---:|
| f10261d3 | Synthesize research into outline for: Rese | pending | - | 0 |
| f10261d3 | Research topic for content: ... (6 tasks, all pending since 06-13) | pending | - | 0 |
| 3da461d8 | Draft Full Blog Post from Synthesized Cont | skipped | dependency_failed | 0 |
| 5b41a2ea | Identify target audience and key messaging | verified | - | 19,003 |
| 5b41a2ea | Analyze the competitive landscape of AI in | verified | - | 21,672 |
| 7627df27 | Generate SEO elements and cover image dire | skipped | dependency_failed | 0 |
| 5b41a2ea | Research Shopify's AI features and integra | verified | - | 21,637 |
| 433641e4 | Edit and polish / Write full draft / Gather source (5 tasks, pending since 06-06) | pending | - | 0 |

## First look

The Mission engine works when driven — the two completed blog missions verified their tasks and burned 264k/430k tokens — but its dominant live state is **stuck at the approval gate**: 8 of 17 runs sit `awaiting_approval` with 0 tokens, some since April, and their pending task trees never start. Only one workspace has ever run a Mission, none created since 2026-06-13, and the single `mission_plan_ready · action_required` notification (see notifications.md) is dated 06-13 — consistent with approval requests that never reach or never re-engage the human. Failure semantics look honest here (`dependency_failed`, `human_cancelled`), unlike the Board's done-with-error pattern.
