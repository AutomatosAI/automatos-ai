# Follow-ups and reply chips

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/modules/tools/discovery/actions_follow_ups.py](../../../orchestrator/modules/tools/discovery/actions_follow_ups.py)
- [orchestrator/modules/tools/discovery/handlers_follow_ups.py](../../../orchestrator/modules/tools/discovery/handlers_follow_ups.py)
- [orchestrator/modules/tools/discovery/follow_up_context.py](../../../orchestrator/modules/tools/discovery/follow_up_context.py)
- [orchestrator/modules/proactive/follow_up_caps.py](../../../orchestrator/modules/proactive/follow_up_caps.py)
- [orchestrator/services/proactive/follow_ups.py](../../../orchestrator/services/proactive/follow_ups.py)
- [orchestrator/services/proactive/follow_up_fire.py](../../../orchestrator/services/proactive/follow_up_fire.py)
- [orchestrator/services/proactive/follow_up_prompts.py](../../../orchestrator/services/proactive/follow_up_prompts.py)
- [orchestrator/services/proactive/follow_up_strings.py](../../../orchestrator/services/proactive/follow_up_strings.py)
- [orchestrator/services/scheduled_task_service.py](../../../orchestrator/services/scheduled_task_service.py)
- [orchestrator/services/scheduled_task_steps.py](../../../orchestrator/services/scheduled_task_steps.py)
- [orchestrator/api/proactive_follow_ups.py](../../../orchestrator/api/proactive_follow_ups.py)
- [orchestrator/modules/tools/discovery/actions_replies.py](../../../orchestrator/modules/tools/discovery/actions_replies.py)
- [orchestrator/modules/tools/discovery/handlers_replies.py](../../../orchestrator/modules/tools/discovery/handlers_replies.py)
- [orchestrator/modules/tools/execution/reply_chips_turn.py](../../../orchestrator/modules/tools/execution/reply_chips_turn.py)
- [orchestrator/consumers/chatbot/turn_parts.py](../../../orchestrator/consumers/chatbot/turn_parts.py)
- [orchestrator/consumers/chatbot/receipts.py](../../../orchestrator/consumers/chatbot/receipts.py)
- [frontend/components/proactive/follow-up-line.tsx](../../../frontend/components/proactive/follow-up-line.tsx)
- [frontend/hooks/use-follow-ups.ts](../../../frontend/hooks/use-follow-ups.ts)
- [frontend/components/proactive/thread-reply-chips.tsx](../../../frontend/components/proactive/thread-reply-chips.tsx)

</details>

## Follow-ups

A follow-up is Auto's "I'll check again at 15:10": one scheduled row, fired once in the check-in lane.

### The row

One `agent_scheduled_tasks` row (W3 K7), inserted by `services/proactive/follow_ups.py` and never through
`ScheduledTaskService.create_task`:

| Column | Value |
|---|---|
| `deliver_as` | `'follow_up'` (`scheduled_task_steps.DELIVER_FOLLOW_UP`; not in `DELIVERY_MODES`, so `create_task` and `platform_schedule_task` can't make one) |
| `task_type` | `one_shot` |
| `created_by_agent_id`, `target_agent_id` | The workspace's Auto (the system agent matching `core/llm/auto_model_sync.py` `AUTO_SLUG_PATTERN`) |
| `for_user_id` | The person whose chat turn scheduled it (new column, prd268_w3_follow_ups) |
| `origin_chat_id` | That chat |
| `schedule`, `next_run_at` | The UTC ISO time |
| `description` | The reason, at most `PROACTIVE_FOLLOW_UP_REASON_CHARS` (500) |
| `payload` | `{about}` when the call named one |
| `created_by_user_id` | NULL: it is no operator's row |

The person and the chat come from the server-built caller context only (`driving_user_id`, `conversation_id`),
handed to the tools by `follow_up_context.carries_the_follow_up`; a model argument can't name them.

### Private to their person

The reason is model-written from a private chat, so every generic reader and writer of `agent_scheduled_tasks`
leaves `deliver_as = 'follow_up'` rows out (W3 K8): `ScheduledTaskService._limit_error`'s counts, `list_tasks`
(`platform_list_scheduled_tasks`, `GET /api/v1/scheduled-tasks`), `update_task_status` (so pause, resume, cancel,
delete and `platform_cancel_scheduled_task` answer not found), the Command Center calendar
(`services/activity_scheduled_task_items.py`), the check-in's `scheduled_run_skipped` signal and the board snapshot
(`handlers_analytics.py`). The scheduler's own reads keep them, so they fire. Audit rows never carry the reason.
`orchestrator/tests/test_prd268w3_follow_ups_kept_private.py` covers this.

### The tools

All three are Auto's, in a person's own chat. `platform_schedule_follow_up` (`write`) and `platform_offer_replies` are
`promoted`, so they survive intent filtering on every Auto chat turn; `platform_list_follow_ups` (`read`) and
`platform_cancel_follow_up` (`write`) are reached through the intent router and `platform_execute`. All are in
`check_in_lane.CHAT_ONLY_ACTIONS`, so no check-in and no follow-up's own turn can run them (no chains).

`platform_schedule_follow_up(reason, in_minutes | at, about?)` refuses, with a sentence the model reads
(`services/proactive/follow_up_strings.py`) and an `error_code`:

| Code | When |
|---|---|
| `no_person` | No driving person, or one not in the workspace |
| `not_auto` | The calling agent is not the workspace's Auto |
| `no_conversation` | Not in the person's own chat |
| `not_live` | The workspace's effective mode is not `live` ("Don't promise to check back.") |
| `no_background` | `background_allowed` is false, or can't be read |
| `paused` | The person paused suggestions |
| `use_a_watch` | `about.kind` is `mission`, `playbook_run` or `ticket`, or the reason names a ticket by board number |
| `follow_up_cap` | A cap, named with its value |
| time and reason errors | No reason, a reason too long, no time, both times, an `at` with no offset |

### Caps under a workspace lock

`schedule_follow_up` takes `pg_advisory_xact_lock(namespace, hashtext(workspace_id))`, counts and inserts in one
transaction, commits, registers the one-shot job (any other worker's reconcile tick registers it within
`RECONCILE_INTERVAL_SECONDS`, 60 s), then writes the audit row `proactive:follow_up_scheduled`. The lock is per
workspace, not per person, so it can bound the workspace's daily cap.

| Cap | Config | Default |
|---|---|---|
| Earliest | `PROACTIVE_FOLLOW_UP_MIN_MINUTES` | 10 minutes |
| Latest | `PROACTIVE_FOLLOW_UP_MAX_DAYS` | 7 days |
| Pending per person (from Auto) | `PROACTIVE_FOLLOW_UP_MAX_PENDING` | 5 |
| Scheduled per person per 24 hours, fired or not | `PROACTIVE_FOLLOW_UP_MAX_PER_DAY` | 20 |
| Scheduled per workspace per 24 hours | `PROACTIVE_FOLLOW_UP_MAX_PER_WORKSPACE_PER_DAY` | 100 |

"Per day" is the last 24 hours of `created_at`, with no time zone. The partial index `ix_scheduled_tasks_follow_ups`
(workspace, person, `created_at`, where `deliver_as = 'follow_up'`) serves the counts and the person's list.

### Missed

A row still `active` whose time is further past than `ONE_SHOT_MISFIRE_GRACE_SECONDS` (3 × 60 s) will never run: a
deploy or a lost leader outlasted the grace, or only the trial could pay. It is **read** as missed, never written:
not pending, not counted against the pending cap, not cancellable. The 24-hour and workspace caps still count it.

### The firing

`ScheduledTaskService.execute_task` runs its usual steps (row lock, credit check, `count_the_run` completing the
one-shot, commit), then `_deliver` hands a `follow_up` row to `services/proactive/follow_up_fire.fire_follow_up`.
It writes one `proactive_runs` row, kind `follow_up`, with Auto's agent id.

It refuses, with that row and no model call, when the person is no longer an active member (`skipped_off` /
`no_addressee`), the mode is not live (`skipped_off` / `switched_off`, or `not_live` in shadow), or
`gates.start_refusal(..., follow_up=True)` refuses: the pause, `background_allowed`, the spend ceiling, the model-run
cap. The check-in interval and a zero suggestion cap don't apply to a follow-up.

Otherwise one turn of the check-in loop runs in the `CHECK_IN` lane as Auto, on the `check_in` purpose. Its prompt
(`follow_up_prompts.py`) quotes, as data, the reason, the origin chat's last `PROACTIVE_FOLLOW_UP_CONTEXT_MESSAGES`
(5) messages while that chat is still the person's, and their open and recently hidden titles. Its calls carry
`check_in_user_id`, so a suggestion is addressed to the person with `source_type` `follow_up`.

| Ending | Outcome | Effect |
|---|---|---|
| A suggestion recorded | `suggested` | `deliver_run` delivers it once the run row commits |
| The `FOLLOW_UP_SILENT` sentinel, or no text | `silent` | Nothing posted |
| Any other text | `posted` | One message via `deliver_background_message` in the origin chat while it is the person's, else their Auto thread; source `{origin: 'follow_up', follow_up_id}`; announced as `auto_message`. The text is posted as plain text that renders as itself and is cut short (`posted_text`). |
| A post that doesn't land | `error` / `not_posted` | |
| Any failure | `error` | Named by its class |

Nothing re-registers or retries the row. Fire and outcome each write an audit row after their commit
(`proactive:follow_up_fired`, `proactive:follow_up_outcome`); a cancel writes `proactive:follow_up_cancelled`.

### The routes and the thread line

`GET /api/proactive/follow-ups?chat_id=&status=pending|all` and `POST /api/proactive/follow-ups/{follow_up_id}/cancel`
(see [Data and routes](data-and-routes.md#routes)). `frontend/components/proactive/follow-up-line.tsx` draws, under
the message whose turn scheduled them, "Follow-up at {time} · Cancel", "Cancelled", "Followed up at {time}" or
"Missed follow-up at {time}", from `GET …?chat_id=&status=all` (`frontend/hooks/use-follow-ups.ts`) and the turn's
`follow_up@1` parts. The time is in the viewer's locale.

## Reply chips

### The tool

`platform_offer_replies(replies)` (`actions_replies.py`): `permission_level` `read` (it changes nothing; it shapes
the reply), category `chat`, `promoted`, chat-only. `handlers_replies.py` accepts 2–3 distinct replies (compared
trimmed and case-insensitively), each 1–3 words and at most 24 characters, with no line break, no invisible
character (control, format or private-use), no markdown and no link. A JSON list sent as text is read. Anything else
is refused with a sentence in the turn's language.

### The terminal turn

`ends_on_reply_chips` (`modules/tools/execution/reply_chips_turn.py`) wraps `ToolLoopExecutor.run` and, for that run,
the caller's `on_tool_result` and `on_round_end`. After a round in which the tool succeeded (called directly, or as
`platform_execute`'s action), `on_round_end` answers `force_final` with that round's own text, and the loop returns it
with **no further model call**. The response carries `reply_chips`.

- A round with no text is not ended early: one more model call writes the answer, and the chips ride it.
- A later round of other work drops the chips.
- A turn that called `platform_get_brief` is never ended early and carries no chips.
- A caller's own `force_final` wins; a chat that ends on its fatal-error line gets no chips.

### Storage: message parts

The chips are a message part, `{type: 'reply_chips@1', replies}`, not a `source` entry: `Message.source` is
background provenance and NULL for in-turn messages (W3 K12). `consumers/chatbot/turn_parts.py` builds, once the
loop has ended, the parts the saved reply carries beside its text:

- `{type: 'reply_chips@1', replies}` from the final response;
- `{type: 'follow_up@1', follow_up_id, at}` for each successful `platform_schedule_follow_up` the turn recorded.

PRD-256's receipts decorators (`consumers/chatbot/receipts.py`) carry them: the live stream yields one tool-data
frame under `turn_parts` before the answer, and the saved message appends the parts after its text part. The saved
text appears exactly once. The voice reads text parts only, so a spoken turn never speaks its chips.

### The web

`frontend/components/proactive/thread-reply-chips.tsx` and `reply-chips.tsx` draw the chips under the chat's newest
message only, never in a read-only chat; a tap sends the chip's text as the person's message and disables the chips,
which come back if nothing was sent within 2 s or a turn was still streaming. Each chip is named "Reply: {chip}" and
carries `data-ai-generated`. Themes read `reply_chips@1` through `HostChatMessage.knownParts` (SDK 1.4.0). An older
phone drops the part and shows the text.
