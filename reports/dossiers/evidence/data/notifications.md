# notifications

Captured 2026-07-04, read-only. **5,423 rows.**

## Query

```sql
SELECT event_type, status, count(*), max(created_at)::date
FROM notifications GROUP BY 1,2 ORDER BY 3 DESC LIMIT 20;
```

| event_type | status | n | newest |
|---|---|---:|---|
| report_submitted | ok | 2,099 | 2026-07-03 |
| heartbeat_complete | ok | 1,701 | 2026-07-03 |
| playbook_step_complete | ok | 868 | **2026-06-16** |
| playbook_complete | ok | 314 | **2026-06-16** |
| report_submitted | warning | 177 | 2026-07-03 |
| task_complete | ok | 90 | 2026-07-02 |
| mission_step_complete | ok | 65 | 2026-06-07 |
| task_sla_breach | warn | 64 | 2026-06-14 |
| mission_step_complete | error | 21 | 2026-06-06 |
| mission_complete | ok | 14 | 2026-06-07 |
| report_submitted | critical | 5 | 2026-07-03 |
| task_failed | error | 2 | 2026-07-03 |
| report_submitted | info | 2 | 2026-05-08 |
| mission_plan_ready | action_required | 1 | 2026-06-13 |

## First look

70% of all notifications are machine chatter (`report_submitted` + `heartbeat_complete`), which will bury the handful that matter (1 `mission_plan_ready action_required`, 2 `task_failed`). The smoking gun is `playbook_complete`: it stopped on **2026-06-16**, yet board_tasks shows playbook runs "done" daily through 07-03 — the completion path stopped emitting success notifications exactly when playbook LLM steps started failing, while the board kept closing the tasks as done. There is no `playbook_failed` event type in the data at all.
