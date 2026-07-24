# memory_short_term — durable/distilled memory sample (founder: "quality is LOW")

Captured 2026-07-04, read-only. 2,211 rows total. The long-term table `memory_items` has **0 rows** and `memory_access_log` has **6 rows** (last write 2026-03-11) — so this short-term table is the only populated platform-side memory store. The mem0 fork's own store could not be inspected (see mem0.md).

## Query 1 — recent 30

```sql
SELECT left(id::text,8), left(workspace_id::text,8), agent_id, content_type,
       round(importance::numeric,2), promoted_to_l3, created_at,
       left(regexp_replace(content, E'[\n\r]+', ' ', 'g'), 200)
FROM memory_short_term ORDER BY created_at DESC LIMIT 30;
```

Verbatim examples (content truncated at 200 chars):

| ws | type | imp | created | content |
|---|---|---|---|---|
| ae8320bc | playbook_summary | 0.50 | 07-03 08:00 | `In execution cron-2bda70b9024b, step 1 failed. It took 5.3 seconds. It failed with error: Error code: 402 - {'error': {'message': 'Insufficient credits. Add more using https://openrouter.ai/settings/c` |
| ae8320bc | playbook_summary | 0.60 | 07-03 08:00 | `The playbook execution cron-2bda70b9024b failed. It failed with error: Step 1 failed: Error code: 402 - ...` |
| ae8320bc | playbook_summary | 0.50 | 07-03 06:15 | `In execution cron-f6f6c3a2b6a4, step 1 failed. It took 15.2 seconds. ... Error code: 402 ...` |
| ae8320bc | playbook_summary | 0.60 | 07-03 06:15 | `The playbook execution cron-f6f6c3a2b6a4 failed. ... Error code: 402 ...` |
| ae8320bc | heartbeat_log | 0.40 | 07-03 01:00 | `Heartbeat Daily Summary (2026-07-02 to 2026-07-03): - Total ticks: 13383 - Successful: 13383 - Errors: 0 - Tokens used: 0` |
| ae8320bc | transcript | 0.60 | 07-03 01:00 | `User: Daily heartbeat summary request Assistant: Heartbeat Daily Summary (2026-07-02 to 2026-07-03): - Total ticks: 13383 ...` |
| c71e4753 | heartbeat_log | 0.40 | 07-02 01:00 | `Heartbeat Daily Summary (2026-07-01 to 2026-07-02): - Total ticks: 1 - Successful: 1 - Errors: 0 - Tokens used: 0` |
| 894519f4 / 28a228aa / fa4125e0 | heartbeat_log + transcript | 0.40/0.60 | 07-02 01:00 | identical `Total ticks: 1` summaries, one pair per workspace per day |

All 30 of the most recent rows are one of: (a) the same OpenRouter-402 playbook failure, memorised **twice per run** (step-level + playbook-level) for cron jobs that fail every day at 02:00/06:00/06:15/08:00, or (b) heartbeat daily summaries, memorised **twice** (once as `heartbeat_log`, once wrapped in a synthetic `User:/Assistant:` "transcript"). `promoted_to_l3 = f` on every row.

## Query 2 — distribution by content_type

```sql
SELECT content_type, count(*), count(*) FILTER (WHERE promoted_to_l3),
       count(*) FILTER (WHERE archived_at IS NOT NULL), round(avg(importance)::numeric,2),
       min(created_at)::date, max(created_at)::date
FROM memory_short_term GROUP BY content_type ORDER BY 2 DESC;
```

| content_type | n | promoted_to_l3 | archived | avg imp | oldest | newest |
|---|---:|---:|---:|---:|---|---|
| playbook_summary | 1,255 | **0** | 1,120 | 0.53 | 2026-05-01 | 2026-07-03 |
| heartbeat_log | 346 | 0 | 324 | 0.40 | 2026-04-30 | 2026-07-03 |
| transcript | 336 | 0 | 313 | 0.60 | 2026-04-30 | 2026-07-03 |
| exchange | 253 | 0 | 253 | 0.50 | 2026-04-30 | 2026-06-07 |
| mission_summary | 12 | 0 | 12 | 0.75 | 2026-05-09 | 2026-06-07 |
| task_failure | 7 | 0 | 7 | 0.80 | 2026-05-08 | 2026-06-06 |
| retry_recovery | 2 | 0 | 2 | 0.75 | 2026-05-16 | 2026-05-16 |

## Query 3 — the higher-value types, verbatim (PII redacted: name → initial)

```sql
SELECT content_type, created_at, left(content,190) FROM memory_short_term
WHERE content_type IN ('mission_summary','task_failure','retry_recovery','exchange')
ORDER BY created_at DESC LIMIT 14;
```

| type | date | content |
|---|---|---|
| exchange | 06-07 | `User: auto you are lying again ... Your wishlist is largely already built. Six blog tools exist (actions_blog.py), all promoted=True (so Auto gets them by default ...` |
| exchange | 06-07 | `User: Auto you already know my brand... this post should have gone libe... what is Sholify magic ... :( Assistant: You're right, G. And this time I'm not bluffing. I checked the actual ...` |
| exchange | 06-07 | `User: Auto dont lie to me please you are not here to please me....jokes etc, you are here to help me with making automaos better Assistant: You're right. I lied by implication, and that dama...` |
| exchange | 06-07 | `User: You fixed the post, you can do that ? Assistant: No — not directly from here. That was me overclaiming. I **can't edit the live blog/CMS unless Automatos has a connected publishing too...` |
| mission_summary | 06-07 | `Mission completed: Create blog: The Shopify AI Edge Stats: 7 tasks verified, 25142s duration, 264193 tokens used What worked: - Research Shopify's AI features ...` |
| task_failure | 06-06 | `Task failed: Outline Automatos' Core Capabilities and Differentiators Description: Based on the provided prompt, create a detailed outline ...` |
| exchange | 06-06 | `User: Create a new MISSION to build the blog based on this research please... do not leave anything out Assistant: Done — I created the mission. **Mission:** Shopify/Automatos flagship blog` |

## Memory recall usage — memory_access_log

```sql
SELECT had_results, count(*), min(created_at)::date, max(created_at)::date
FROM memory_access_log GROUP BY had_results;
```

| had_results | n | oldest | newest |
|---|---:|---|---|
| t | 4 | 2026-03-09 | 2026-03-09 |
| f | 2 | 2026-03-10 | 2026-03-11 |

## First look (quality verdict)

The founder's "quality is LOW" is confirmed and quantifiable: ~87% of all stored memories (playbook_summary + heartbeat_log + transcript = 1,937/2,211) are operational chatter, the same failure/heartbeat duplicated 2x per event, with zero client- or preference-level insight. **Nothing has ever been promoted to L3** (0 across all 2,211 rows; `memory_items` empty), so the PRD-178/W8 promotion pipeline has never produced a durable memory in this DB. `memory_access_log` shows recall was exercised 6 times ever, last on 2026-03-11 — memories are written, not read. The only genuinely informative rows (`exchange`) stopped 2026-06-07 and are raw chat clippings — including the user telling Auto it is lying — rather than distilled facts; the typed distill taxonomy (PRD-159) is visible in the schema but the content quality gate clearly is not filtering repetitive failure spam.
