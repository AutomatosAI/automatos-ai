# Data and routes

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/core/models/proactive.py](../../../orchestrator/core/models/proactive.py)
- [orchestrator/alembic/versions/prd268_w1_proactive.py](../../../orchestrator/alembic/versions/prd268_w1_proactive.py)
- [orchestrator/alembic/versions/prd268_w2_delivery.py](../../../orchestrator/alembic/versions/prd268_w2_delivery.py)
- [orchestrator/alembic/versions/prd268_w3_follow_ups.py](../../../orchestrator/alembic/versions/prd268_w3_follow_ups.py)
- [orchestrator/api/proactive.py](../../../orchestrator/api/proactive.py)
- [orchestrator/api/proactive_brief.py](../../../orchestrator/api/proactive_brief.py)
- [orchestrator/api/proactive_follow_ups.py](../../../orchestrator/api/proactive_follow_ups.py)
- [orchestrator/api/readiness.py](../../../orchestrator/api/readiness.py)
- [orchestrator/api/approval_views.py](../../../orchestrator/api/approval_views.py)
- [orchestrator/api/auto_calibration.py](../../../orchestrator/api/auto_calibration.py)
- [orchestrator/api/activity.py](../../../orchestrator/api/activity.py)
- [orchestrator/api/heartbeat.py](../../../orchestrator/api/heartbeat.py)
- [orchestrator/services/chat_parts.py](../../../orchestrator/services/chat_parts.py)
- [orchestrator/services/proactive/live.py](../../../orchestrator/services/proactive/live.py)
- [orchestrator/services/mobile/discovery.py](../../../orchestrator/services/mobile/discovery.py)
- [orchestrator/reports/route-manifest.json](../../../orchestrator/reports/route-manifest.json)

</details>

## Tables

All four tables are in `orchestrator/core/models/proactive.py`. `workspace_id` is a foreign key to `workspaces.id`
with `ON DELETE CASCADE` and leads an index. `user_id` and agent ids are plain integers (`users.id`, resolved with
`core.auth.principal.resolve_user_pk`), not foreign keys: the audit outlives the person or agent it names.

### `proactive_suggestions`

| Column | Meaning |
|---|---|
| `id` (UUID), `workspace_id`, `user_id` | The addressee |
| `source_type`, `source_agent_id`, `run_id` | `check_in` (Auto's), `heartbeat` (an agent's), `streak`, `follow_up`; the agent; the run that made it |
| `title` (≤ 120), `detail` (≤ 600) | What Auto says |
| `evidence` (JSON) | 1–5 `{type, id}` refs, each a read of the run |
| `proposed_action` (JSON) | `{kind, payload}`, `kind` one of `board_task`, `chat`, `mission`, `playbook_run`; only the first two start |
| `subject_key` (≤ 200) | `agent:<id>:error:<class>`, `ticket:<number>` or `title:<normalised title>` |
| `severity` | `today` or `can_wait` (default) |
| `status` | `open`, `started`, `hidden`, `expired` (CHECK `ck_proactive_suggestions_status`) |
| `occurrences`, `last_seen_at` | Streak counting |
| `started_ref` (JSON) | `{board_task_id, watch_id}` after a ticket Start |
| `decided_via`, `hide_reason`, `decided_at` | `web`, `go`, `push`, `chat`, `telegram`; `not_now`, `not_useful`, `wrong_agent` |
| `est_credits` | Never written today |
| `created_at`, `delivered_at`, `expires_at` | `delivered_at` is the delivery claim; `expires_at` is 7 days on |

Indexes: `uq_proactive_suggestions_open_subject` (unique on workspace, user, subject key, where `status = 'open'`) and
`ix_proactive_suggestions_workspace_status_created`.

### `proactive_runs`

One row per check-in run or refusal, and per follow-up firing: `id`, `workspace_id`, `agent_id`, `kind`
(`workspace_check_in`, `agent_check_in`, `follow_up`), `window_since`, `window_until`, `outcome`, `signals`,
`suggestions` (count), `model`, `tokens`, `cost_usd`, `error_class`, `created_at`. The outcome CHECK
(`ck_proactive_runs_outcome`) admits `silent`, `suggested`, `error`, `skipped_quiet`, `skipped_cap`,
`skipped_credits`, `skipped_paused`, `skipped_off` and `posted`. Index `ix_proactive_runs_workspace_agent_created`.
No job prunes it.

### `proactive_person_settings`

Primary key `(workspace_id, user_id)`: `paused`, `quiet_start`, `quiet_end` (times), `timezone` (IANA), `updated_at`.

### `proactive_calibrations`

Unique `uq_proactive_calibrations_person (workspace_id, user_id)`, the offered-once claim: `id`, `status` (`declined`,
`open`, `answered`), `calibration` (the `Calibration@1` posted), `rejected_lines`, `accepted_lines`, `chat_id`,
`message_id`, `consented_at`, `decided_at`.

### Existing tables changed, and stored settings

| Where | Change |
|---|---|
| `agent_scheduled_tasks` | `ck_scheduled_task_deliver_as` admits `follow_up`; new nullable `for_user_id`; partial index `ix_scheduled_tasks_follow_ups` |
| `mobile_push_tickets` | `ck_mobile_push_tickets_push_type` admits `suggestion` |
| `mobile_devices` | `push_categories` JSONB NOT NULL DEFAULT `'[]'` |
| `mobile_digest_posts` | Unchanged; it is the brief's claim |
| `digest_feedback` | Unchanged schema; one row per (workspace, person, `state_hash`) kept by an advisory lock |
| `system_settings` | Row `(category 'proactive', key 'mode')`: `off`, `shadow` or `live`. Not seeded. |
| `workspaces.settings` | `proactive.enabled`, `proactive.caps`, `push.suggestion_titles` |
| `agents.configuration` | `heartbeat.mode`: `check_in` or `act` |

## Migrations

One per wave, chained: `prd266_w2_push` → `prd268_w1_proactive` → `prd268_w2_delivery` → `prd268_w3_follow_ups`.
Each is safe on a database `create_all` built (`IF NOT EXISTS`, `DROP CONSTRAINT IF EXISTS` before `ADD`), changes no
existing row, and runs in the one transaction `alembic/env.py` opens.

| Revision | What it does |
|---|---|
| `prd268_w1_proactive` | Creates `proactive_suggestions`, `proactive_runs`, `proactive_person_settings` |
| `prd268_w2_delivery` | Widens the push-type CHECK; adds `mobile_devices.push_categories` (metadata-only on Postgres 11+) |
| `prd268_w3_follow_ups` | Widens `ck_scheduled_task_deliver_as`; adds `for_user_id` and its partial index; widens `ck_proactive_runs_outcome` with `posted`; creates `proactive_calibrations` |

Re-adding a CHECK takes an ACCESS EXCLUSIVE lock while it scans the table; each table here is small.

## Routes

Every route is a plain `def` on the sync session, registered through `RouterSpec` in `orchestrator/router_manifest.py`,
and listed in `orchestrator/reports/route-manifest.json`. A refusal body is `{"code", "detail"}`, the detail in
English or Portuguese (Brazil) by `Accept-Language` (or the body's `locale`). Another workspace's id is 404, never
403.

Roles: `workspace:manage` is owners and admins; `agents:execute` adds editors; `members:read` is every role.
"Answerer" (`require_workspace_answerer`) is an owner, admin, editor or the super admin.

| Method and path | Auth | Returns | Refusals |
|---|---|---|---|
| `GET /api/proactive/suggestions?status=open&limit=` | Any member; owners and admins list everyone's | `{suggestions}` newest first (50 default, 100 max); lapsed rows list as `expired` | 404 `no_workspace`, 403 `no_person`, 503 `unreadable` |
| `POST /api/proactive/suggestions/{suggestion_id}/start` `{via?}` | Answerer | `{suggestion, opens}`; `opens` is `{type: 'board_task', board_task_id}` or `{type: 'chat', suggestion_id}` | 404 `not_found`; 409 `already_started`, `already_hidden`, `expired`, `not_available`, `spend_ceiling`; 503 `unavailable` |
| `POST /api/proactive/suggestions/{suggestion_id}/hide` `{reason?, via?}` | Answerer; the addressee, or an owner or admin | `{suggestion}` | 403 `not_yours`; 404 `not_found`; 409 `already_started`, `expired`; 503 `unavailable` |
| `GET /api/proactive/runs?since=&limit=` | `workspace:manage` | `{runs}` since `since` (7 days default), 200 default, 500 max | 404 `no_workspace`, 503 `unreadable` |
| `GET /api/proactive/settings` | `workspace:manage` | `{enabled, caps, platform_caps, effective_mode}` | 404 `no_workspace`, 409 `caps_stored_wrong`, 503 `settings_unreadable` |
| `PUT /api/proactive/settings` `{enabled?, caps?}` | `workspace:manage` | `{enabled, caps, platform_caps}`; audited | 422 `caps_looser`, `caps_wrong`; 503 `settings_unavailable` |
| `GET /api/proactive/me` | `members:read` | `{paused, quiet_start, quiet_end, timezone, effective_mode}` | 403 `no_person`, 503 `settings_unreadable` |
| `PUT /api/proactive/me` `{paused?, quiet_start?, quiet_end?, timezone?}` | `members:read` | `{paused, quiet_start, quiet_end, timezone}`; audited | 422 on a partial or equal pair of times; 403 `no_person`; 503 `settings_unavailable` |
| `POST /api/proactive/brief/open` `{local_date, timezone, locale?}` | `members:read` | `{chat_id, message_id, posted}`, or `{posted: false, pending: true}` | 404 `brief_off`, `open_no_workspace`; 403 `open_no_person`, `open_not_a_member`; 422 `open_out_of_window`; 503 `open_unavailable` |
| `GET /api/proactive/follow-ups?chat_id=&status=pending\|all` | `members:read`; the caller's own only | `{follow_ups: [{id, at, reason, status, chat_id}]}`, at most 100; `status` `pending`, `fired`, `cancelled` or `missed` | 404 `no_workspace`, 403 `follow_up_no_person`, 503 `follow_ups_unreadable` |
| `POST /api/proactive/follow-ups/{follow_up_id}/cancel` | `agents:execute`; the caller's own only | `{follow_up}`; audited | 404 `follow_up_not_found`; 409 `follow_up_fired`, `follow_up_cancelled`, `follow_up_missed`; 503 `follow_up_unavailable` |
| `GET /api/readiness` | `members:read` | `Readiness@1` (the held check, else a check now) | 404 `no_workspace` |
| `POST /api/readiness/check` | `members:read` | `Readiness@1` (a check now) | 429 `checked_recently` with `Retry-After` |
| `GET /api/v1/approval-grants/{grant_id}/view` | Answerer | `ApprovalView@1` | 403; 404 "Approval not found"; 503 |
| `GET /api/auto/calibration` | `members:read` | `{available, offer, consent_line, status, calibration, rejected_lines, accepted_lines}` | 403, 404 |
| `POST /api/auto/calibration` `{consent, chat_id?, locale?}` | `agents:execute` | `{status, calibration, chat_id, message_id}` | 404 `not_available`, `calibration_no_workspace`; 403 `calibration_no_person`, `calibration_not_a_member`; 409 `already_offered`; 503 `calibration_unavailable` |
| `POST /api/auto/calibration/{calibration_id}/lines/{line_id}/reject` | `agents:execute` | The record, `not_stored`, `note` | 404 `not_available`, `calibration_not_found`; 409 `calibration_answered`; 503 `memory_unavailable` |
| `POST /api/auto/calibration/{calibration_id}/accept` | `agents:execute` | The record, `not_stored`, `note` | 404 `not_available`, `calibration_not_found`; 503 `memory_unavailable` |
| `GET /api/activity/digest/feedback?state_hash=` | `agents:execute` | `{state_hash, rating}`, the caller's own | |

Existing routes that changed:

| Route | Change |
|---|---|
| `POST /api/activity/digest/feedback` | One rating per person per `state_hash` (update, not a new row) |
| `GET`/`PUT /api/heartbeat/agents/{agent_id}/config` | `mode` (`check_in` or `act`). `GET` answers the stated mode or its effective default; a `PUT` that leaves `mode` out keeps the stated one; `null` unsets it. |
| `GET /api/notification-preferences` | `push_types` gains `suggestion` (`suggestion_ready`) |
| `GET /api/activity/needs-you`, the Missions list, board ticket reads | Rows carry `display_state`, `next_step`, `retry` |
| `GET /.well-known/automatos.json` | New capabilities (below) |

## Message parts

Background posts (`services/chat_messenger.py`) and saved turns carry structured parts beside their text part. A
client that doesn't know a part type drops it and shows the text.

| Part | Where it comes from | Carries |
|---|---|---|
| `brief@1` | The brief (origin `daily_digest`) | `{brief}` |
| `suggestion@1` | A delivered suggestion (origin `suggestion`) | `{suggestion_id}`; the client reads live state from the list |
| `calibration@1` | A calibration (origin `calibration`) | The `Calibration@1` |
| `reply_chips@1` | A chat turn that offered chips (`consumers/chatbot/turn_parts.py`) | `{replies}` |
| `follow_up@1` | A chat turn that scheduled a follow-up | `{follow_up_id, at}` |

`services/chat_parts.py` `STRUCTURED_PART_TYPES` is the list a background post may carry: `brief@1`, `suggestion@1`,
`calibration@1`.

## Events

| Event | Lane | When |
|---|---|---|
| `suggestion_changed` | Live, the `board_events` NOTIFY channel | A suggestion posted, started, hidden or expired; carries workspace, id, addressee and status only |
| `suggestion_ready` | Bell (`NotificationDispatcher`), in-app by default; push type `suggestion` | A live run made a `today` suggestion, within the person's daily cap and outside quiet hours |
| `heartbeat_complete` | Bell | In live mode, only when a converted heartbeat's check-in suggested or started a failing streak |
| `auto_message` | Bell and push | Also for a follow-up that posted (origin `follow_up` joined `ANNOUNCED_ORIGINS`); never for a brief posted on open |

## Discovery capabilities

`services/mobile/discovery.py` `CAPABILITY_ROUTES`, each mapped to its live route:

| Capability | Route | Listed |
|---|---|---|
| `brief` | `POST /api/proactive/brief/open` | Always |
| `suggestions` | `GET /api/proactive/suggestions` | Always |
| `readiness` | `GET /api/readiness` | Always |
| `approval_view` | `GET /api/v1/approval-grants/{grant_id}/view` | Always |
| `suggestion_push` | `POST /api/v1/mobile/devices` | Only while `SUGGESTION_PUSH_ENABLED` is on |
| `calibration` | `POST /api/auto/calibration` | Only while `CALIBRATION_ENABLED` is on |

## Audit actions

Written after the decision's commit by `AuditService.log`, which commits its own row and never fails the caller:
`proactive:suggestion_started`, `proactive:suggestion_hidden`, `proactive:settings_changed`, `proactive:paused`,
`proactive:resumed`, `proactive:quiet_hours_changed`, `proactive:follow_up_scheduled`, `proactive:follow_up_cancelled`,
`proactive:follow_up_fired`, `proactive:follow_up_outcome`, and calibration's consent, decline, accept and reject
rows (`services/calibration/audit.py`). None carries a follow-up's reason or a calibration line.
