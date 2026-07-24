# deliverables — client-facing outputs

Captured 2026-07-04, read-only. **2,242 rows**, newest **2026-06-16**.

## Query

```sql
SELECT left(id::text,8), artifact_type, left(title,50), status, source_type, file_type, created_at::date
FROM deliverables ORDER BY created_at DESC LIMIT 15;
```

| artifact_type | title | status | source | type | created |
|---|---|---|---|---|---|
| image | Missions Slide4 / Slide3 / Slide2 / Slide1 | ready | chat | png | 2026-06-16 |
| image | Routing Slide1–4 (+ Slide1 Story) | ready | chat | png | 2026-06-12 |
| image | Field Memory Slide1–4 (+ Slide1 Story) | ready | chat | png | 2026-06-10 |
| image | 2026 W22 Tue Missions Slide2 | ready | chat | png | 2026-06-09 |

## First look

2,242 deliverables is a real body of output (dominated by the social-content engine: weekly/daily carousel slide PNGs). But production of deliverables **stopped on 2026-06-16** — the same date `playbook_complete` notifications stopped — and the daily content playbooks have since been failing on OpenRouter 402 while their board tasks close as done. Net effect: the platform has produced zero client-facing artifacts for ~2.5 weeks and nothing surfaced it. For output-quality dossiers: the recent deliverables are all `chat`-sourced PNG slides; no document/blog artifacts appear in the recent window.
