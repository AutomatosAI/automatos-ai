# Check-ins

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/services/proactive/runner.py](../../../orchestrator/services/proactive/runner.py)
- [orchestrator/services/proactive/gates.py](../../../orchestrator/services/proactive/gates.py)
- [orchestrator/services/proactive/loop.py](../../../orchestrator/services/proactive/loop.py)
- [orchestrator/services/proactive/reads.py](../../../orchestrator/services/proactive/reads.py)
- [orchestrator/services/proactive/signals.py](../../../orchestrator/services/proactive/signals.py)
- [orchestrator/services/proactive/runs.py](../../../orchestrator/services/proactive/runs.py)
- [orchestrator/services/proactive/limits.py](../../../orchestrator/services/proactive/limits.py)
- [orchestrator/services/proactive/expiry.py](../../../orchestrator/services/proactive/expiry.py)
- [orchestrator/services/proactive/failing_streaks.py](../../../orchestrator/services/proactive/failing_streaks.py)
- [orchestrator/services/proactive/heartbeat_bridge.py](../../../orchestrator/services/proactive/heartbeat_bridge.py)
- [orchestrator/services/proactive/heartbeat_bell.py](../../../orchestrator/services/proactive/heartbeat_bell.py)
- [orchestrator/services/proactive/delivery.py](../../../orchestrator/services/proactive/delivery.py)
- [orchestrator/modules/proactive/window.py](../../../orchestrator/modules/proactive/window.py)
- [orchestrator/modules/proactive/dedupe.py](../../../orchestrator/modules/proactive/dedupe.py)
- [orchestrator/modules/proactive/provenance.py](../../../orchestrator/modules/proactive/provenance.py)
- [orchestrator/modules/proactive/streaks.py](../../../orchestrator/modules/proactive/streaks.py)
- [orchestrator/modules/proactive/heartbeats.py](../../../orchestrator/modules/proactive/heartbeats.py)
- [orchestrator/modules/tools/discovery/check_in_lane.py](../../../orchestrator/modules/tools/discovery/check_in_lane.py)
- [orchestrator/modules/tools/discovery/handlers_proactive.py](../../../orchestrator/modules/tools/discovery/handlers_proactive.py)
- [orchestrator/modules/tools/discovery/suggestion_fields.py](../../../orchestrator/modules/tools/discovery/suggestion_fields.py)
- [orchestrator/core/read_only_call.py](../../../orchestrator/core/read_only_call.py)
- [orchestrator/modules/context/modes.py](../../../orchestrator/modules/context/modes.py)
- [orchestrator/services/push/suggestion_rules.py](../../../orchestrator/services/push/suggestion_rules.py)

</details>

A check-in is a background turn that reads the workspace and may record suggestions. It is the only thing in
Proactive Auto that runs on a schedule; everything else hangs off what it records.

## Two kinds of check-in

| Kind (`proactive_runs.kind`) | Entry point | What it starts from |
|---|---|---|
| `workspace_check_in` | `run_workspace_check_in` (`services/proactive/runner.py`) | A deterministic **signal diff** over its window (`services/proactive/signals.py`). No signal: the run is `silent` and makes no model call at all. |
| `agent_check_in` | `run_agent_check_in` | The agent's heartbeat prompt and identity, built through `ContextService` in `ContextMode.CHECK_IN`. No pre-check: an agent's work is outside what a diff can see, so its interval is its cost lever. |

A third kind, `follow_up`, is a follow-up's firing; it runs the same loop and lane (see
[Follow-ups and reply chips](follow-ups-and-chips.md)).

**The window** (`modules/proactive/window.py`): from the `window_until` of the last run of the same kind whose outcome
was `silent` or `suggested`, else the last 24 hours (`LOOKBACK`). A failed or skipped run never moves it on.

### Signals

Auto's workspace check-in reads one query per kind, scoped to the workspace, at most `ROWS_PER_KIND` (50) each, one
signal per subject:

| Signal | What it is |
|---|---|
| `ticket_failed`, `ticket_blocked`, `ticket_stuck` | A ticket that failed, was blocked, or is stuck as needs-you reads it, in the window. A Mission's own steps are left to the Mission. |
| `heartbeat_error`, `heartbeat_objective_missed` | A heartbeat result with `status = 'error'` or `objective_met = false`. |
| `watch_missed_run`, `watch_benched`, `watch_expired`, `watch_budget_exhausted` | Watch events (`core/models/watch_enums.py`), scoped through their watch. |
| `report_awaiting_approval` | A report with `requires_approval` that nobody acknowledged. |
| `grant_expiring` | A pending approval within 24 hours of lapsing. |
| `scheduled_run_skipped` | A one-off scheduled task past its time and the scheduler's misfire grace that never ran. Follow-up rows are left out. |
| `credits_low` | The credit balance at the low or run-out alert level, and moving in the window. |

Each signal's summary is one plain line of at most 240 characters, quoted in the prompt as data. Its `ref` is the
`(type, id)` read a suggestion may cite. The prompt lists at most `MAX_SIGNALS` (30).

## The run

```
expire_lapsed ─► start_refusal (gates) ─► [workspace: signals; none → silent]
             ─► run_loop (CHECK_IN lane) ─► run row ─► [agent: note_run streak] ─► deliver_run
```

1. **Lapse.** `services/proactive/expiry.py` moves open suggestions past `expires_at` (7 days after they were made)
   to `expired` first, so no cap, dedupe read or prompt counts them.
2. **Gates** (`services/proactive/gates.py`), in order. The first that refuses names the run's outcome and its
   `error_class`, writes the run row, and no model is called:

   | Check | Outcome | `error_class` |
   |---|---|---|
   | Platform mode and workspace switch (`switches.effective_mode`) | `skipped_off` | `switched_off` |
   | The workspace has no owner to address | `skipped_off` | `no_addressee` |
   | The addressee's pause | `skipped_paused` | `paused` |
   | `background_allowed` (`services/credits/routing.py`): only the trial could pay, or the check failed | `skipped_credits` | `trial_only`, `credits_unreadable` |
   | `refuse_new_work` (`services/daily_spend_guard.py`): the day's spend ceiling | `skipped_cap` | `spend_ceiling` |
   | Caps unreadable or wrong | `skipped_cap` | `caps_unreadable` |
   | `max_model_runs_per_day`, `min_interval_minutes` (less a 5-minute scheduler grace), a suggestion cap at 0 | `skipped_cap` | the cap's name |
   | An agent held back by its failing streak | `skipped_cap` | `failing_streak` |

3. **The loop** (`services/proactive/loop.py`), on `LLMManager(service_name='heartbeat')`: the platform
   `orchestrator_llm` setting (`claude-haiku-5-5` by default), never the agent's own `model_config`. Limits are
   constants in the module, not config:

   | Limit | Value |
   |---|---|
   | Model calls per run (`MAX_ITERATIONS`) | 4 |
   | Output tokens per call | 800 (the `check_in` purpose's budget, `core/llm/output_budget.py`) |
   | Tool calls per model call (`MAX_CALLS_PER_TURN`) | 8 |
   | Context (`ContextMode.CHECK_IN`, `max_tokens`) | 12,000 |
   | Cost (`MAX_COST_USD`) | $0.05, past which the run stops as `error` / `cost_ceiling` |

   A model response with no tool call ends the run; so does a turn whose calls were all recorded suggestions. A
   failure after the first model call ends the run as `error`, named by `services/proactive/failures.py` (a
   provider's refusal is `http_<status>`).
4. **The run row** (`services/proactive/runs.py`) records the window, outcome, signals, suggestion count, model,
   tokens, `cost_usd` and `error_class`. Outcomes: `silent`, `suggested`, `error`, `skipped_off`, `skipped_paused`,
   `skipped_credits`, `skipped_cap`, and `posted` (follow-ups only). The CHECK also admits `skipped_quiet`, which nothing
   writes: check-ins run in quiet hours, and only the bell and push hold.
5. **Spend.** Each run is one ask in the `check_in` lane, which is in the credits' background lanes
   (`services/credits/asks.py`), so the trial never pays for it.
6. **Delivery** runs after the row commits (below). A delivery failure never fails the run.

## The read-only lane

The lane is enforced in the executor, failing closed, by decorators in
`modules/tools/discovery/check_in_lane.py`. A call is in the lane when `caller_context["lane"] == "check_in"`.

| Layer | What it does |
|---|---|
| What is offered | `ContextMode.CHECK_IN` uses `tool_loading="read_only"`: only platform actions whose `permission_level` is `read`, plus `platform_suggest_next_step`. No Composio, plugin, workspace, memory or sub-agent tool and no `platform_execute`. |
| `platform_actions_only_in_a_check_in` on `UnifiedToolExecutor.execute_tool` | Refuses any tool that is not a platform action, before routing. Composio calls never reach the platform executor, so this layer is the one that stops them. |
| `reads_only_in_a_check_in` on `PlatformActionExecutor.clear` | Clears a call only when its registered definition is positively `read`, or it is `platform_suggest_next_step`. Writes, destructive actions, unknown actions and any action whose definition can't be read are refused before any other gate. |
| The web | `platform_web_fetch` and `platform_web_search` are labelled `read` but are neither offered nor cleared in the lane, so nothing read leaves the workspace. |
| Chat-only actions | `CHAT_ONLY_ACTIONS` (the three follow-up tools and `platform_offer_replies`) are never offered or cleared, so no check-in or follow-up turn schedules, lists, cancels or offers chips. |
| Admin actions | A follow-up's person may hold any role, so the lane clears nothing that needs an admin. |
| Side records | `carries_the_check_in` runs every lane call except the suggestion inside `core.read_only_call.reading_only()`. Read actions that keep a record of their own (a code-graph query log, a document's `last_accessed`, the onboarding stamp, a spreadsheet's workspace copy, the NL2SQL audit and training rows, a preview's picture) ask `only_reading()` and skip it. |

The refusal the model reads is "Not available in a check-in, which can only read. Suggest it as a next step
instead." Each refusal is logged as `[check_in_lane] … refused in a check-in`.

`ActionDefinition.permission_level` has no default: every action states its level, checked at registration
(`modules/tools/discovery/action_registry.py`). `orchestrator/tests/test_prd268_read_actions_write_nothing.py` sweeps
every `read` action against a spy session and transport. A mislabelled action is relabelled, never excepted.

The runner's own facts reach the suggestion tool server-side only: `carries_the_check_in` (a decorator on
`PlatformActionExecutor.execute`) strips any caller-supplied `_check_in` and hands the tool `run_id`,
`check_in_reads` (the run's successful reads so far, `services/proactive/reads.py`), `check_in_agent_id` and, for a
follow-up, `check_in_user_id`.

## Recording a suggestion

`platform_suggest_next_step` (`actions_proactive.py`, `permission_level="write"`, the lane's one write) reads every
argument into a structured field (`suggestion_fields.py`) and refuses, with a sentence the model can correct from:

- **No run.** A call without the server's `_check_in`.
- **Not a suggestion.** `title` (≤ 120, one line, control and format characters dropped), `detail` (≤ 600), 1–5
  `evidence` refs `{type, id}`, a `proposed_action` of `board_task` (optional `payload.agent_id`) or `chat`, `severity`
  `today` or `can_wait` (default), an optional `subject_key` (`ticket:<number>` backed by that ticket's read, or
  `title:<…>`). `agent:<id>:error:<class>` is refused: only the runner files a streak's card.
- **Provenance** (`modules/proactive/provenance.py`). Every evidence ref must name a successful read of the same run.
  A suggestion that cites nothing is refused.
- **Caps** (`services/proactive/limits.py`): suggestions per run, new per person per day, open per agent.
- **Dedupe** (`modules/proactive/dedupe.py`): an `open` suggestion with the same `subject_key` or normalised title,
  or a `hidden` one within 14 days, is "Already suggested". The partial unique index
  `uq_proactive_suggestions_open_subject` (workspace, user, subject key, where open) settles a race; the insert runs
  in a savepoint.
- **An unknown agent** for a `board_task`.

Past `max_open_per_person`, the person's oldest open suggestions expire in the insert's savepoint. The addressee is
the workspace owner (W1 K11), or a follow-up's person with `source_type` `follow_up`. Rows are scoped to the
workspace, and evidence and text are quoted as data wherever they are shown.

## Failing streaks

`services/proactive/failing_streaks.py` and `modules/proactive/streaks.py`. A streak is one agent failing with one
error class, counted from `proactive_runs` (skips pass over it); no column holds it.

- **The card.** The first failure records one suggestion, `source_type` `streak`, `subject_key`
  `agent:<id>:error:<class>`, copy from `services/proactive/streak_text.py`, the streak's runs as evidence, Start a
  ticket for Auto. Later failures update `occurrences`, `last_seen_at`, detail, evidence and push `expires_at` on. No
  model call, no ticket. A streak whose card was hidden or started gets no other.
- **The slower check-in.** From the third failure (`DOUBLING_FROM`), `held_back` skips the agent until the doubled
  interval has passed, up to 24 hours or the workspace's `max_interval_minutes` when shorter. Each skip is a
  `skipped_cap` / `failing_streak` run.
- **The reset.** The first run that works expires every open streak card of the agent with a "works again" detail.

Auto's workspace check-in has no agent to key a streak by; its first failure still rings the bell.

## The heartbeat bridge

The heartbeat scheduler runs `agent_job` and `orchestrator_job` (`services/proactive/heartbeat_bridge.py`) instead
of `HeartbeatService._agent_tick` and `_orchestrator_tick`. At every fire the job reads the effective mode and whether
the heartbeat converts:

- **An agent's** converts when `heartbeat_mode(hb_config)` is `check_in` (`modules/proactive/heartbeats.py`: the
  stated `mode`, else `act` when `auto_act` is set, else `check_in`) and the agent is not a CLI agent.
- **The orchestrator's** converts at `proactive_level` `silent` or `notify` (the heartbeat's own, else the
  workspace's orchestrator settings, else `notify`).

| Effective mode | What runs |
|---|---|
| `off`, or a heartbeat that doesn't convert | Today's tick only. |
| `shadow` | Today's tick, then the check-in, which rings nothing. |
| `live` | The check-in only: no ticket, no `report_to` routing, no per-tick report. |

A check-in runs only inside the heartbeat's active hours. A mode that can't be read is off, so the tick runs. The
Run-now routes still run the tick. In live mode `services/proactive/heartbeat_bell.py` rings `heartbeat_complete`
only when the run suggested something or failed for the first time in a streak, and never for the orchestrator at
`silent`. Its copy is built from structured fields only.

## Delivery

`deliver_run(db, run_id)` (`services/proactive/delivery.py`) runs only when the effective mode is `live`:

1. **The thread.** Each new open suggestion with `delivered_at` NULL is claimed (set and committed), then posted once
   through `deliver_background_message` in its addressee's Auto thread: source `{origin: 'suggestion',
   suggestion_id}`, parts `[text, {type: 'suggestion@1', suggestion_id}]`. `suggestion` is not an announced origin,
   so no `auto_message` is raised. A failed post gives the claim back but is not retried.
2. **The bell.** One `suggestion_ready` per addressee per run, through `NotificationDispatcher`, when a just-posted
   suggestion has severity `today`, the person isn't in quiet hours (`services/proactive/quiet.py`), and fewer than
   `PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY` `suggestion_ready` rows were written for them since their local midnight.
   It links to the Command Center card (`link_type` `suggestion`).
3. **The live event.** `suggestion_changed` on the `board_events` channel (`services/proactive/live.py`), after
   commit: workspace, id, addressee and status, never the words. It is also sent on Start, Hide and expiry.

Suggestions recorded in shadow are never delivered later when the mode turns live; they stay in the list and brief.

## The suggestion push

The fourth push type, `suggestion` (event `suggestion_ready`), is built and **ships off**. `services/push/fanout.py`
sends it only when `may_push_suggestion` (`services/push/suggestion_rules.py`) holds:

- `SUGGESTION_PUSH_ENABLED` is true;
- the event names one person's suggestion in its workspace, of severity `today`, still open and not past
  `expires_at`;
- the device registered the `suggestion` category (`mobile_devices.push_categories`);
- the person's `suggestion` push switch isn't off (`GET /api/notification-preferences` lists it; no row means on).

The body is "Auto has a suggestion" in the phone's language, or the title (≤ 120 characters) when the workspace's
`settings.push.suggestion_titles` is true. The title is never a signed field. Discovery lists `suggestion_push` only
while the switch is on.
