# PRD-204 — Auto Watcher: persistent supervision of launched work (own the outcome, not the launch)

> Status: APPROVED FOR BUILD 2026-07-16 (Gerard) — §8 recommendations applied as build defaults; override at PR review.
> Grounded @ `8e0543211` (origin/main, 2026-07-16). Relationship: PRD-200 is the same mechanics one altitude down (per-task verify-gate inside the mission tick); PRD-128 is the delivery plane; PRD-193/196 are the approval plane; PRD-185 built the breaker/telemetry substrate this consumes.
> **Build size: L (one PR, 12 stories) · Risk: Medium** — one new background loop (rides the existing fcntl-locked UnifiedScheduler), all corrective actions policy-gated, every producer hook fail-soft.

---

## 1. What this is

A **Watch** is a first-class, workspace-scoped row that supervises one launched unit of work — a mission, a playbook execution, or a scheduled playbook — from launch to a **verdict**. The loop: watch → check (event-driven at terminal choke points, heartbeat sweep as fallback) → score the output against the original intent → take a bounded corrective action if weak (tweak+rerun, replan, reassign, spawn agent from blueprint — each through the existing approval policy) → notify the user with the verdict.

The product contract: when the user says "can you do X," Auto owns X to a verdict. Not "I started it, good luck" — started, watched, judged, corrected within guardrails, reported. Blueprints and the policy plane stay the boundary: the watcher never grants itself powers; every action rides `evaluate_approval` + the workspace autonomy setting + the §12.3 gate, exactly like a human-triggered action would.

**Framing (CLAUDE.md §3): Extension.** The registry and decision loop are net-new; everything else wires production seams that already exist (dispatch spine, notification dispatcher, cross-model judge, approval grants, retry lineage). **Why now:** the deep review's verdict was "good bones, open loops" — this is the PRD that closes the loop the whole autonomy story hangs on, and four of its stories are straight fixes to silent-failure holes that exist today regardless of the watcher.

**What this is NOT** (terminology guard): not the OS-review "generic watcher primitive" missing-channel (that is an *inbound-trigger* watcher — watch external conditions to *start* work); not infra monitoring (`core/monitoring/` Prometheus/psutil); not a unified event bus (§9).

## 2. Current reality (grounded)

- **No watch concept exists.** Broad grep (`watch|monitor|supervis|follow_up|check_back`) across `services/`, `core/`, `api/`, `modules/` finds only the autonomy vocabulary, scheduling follow-ups, Composio "monitoring" app-actions, and wizard SSE. Nothing supervises a launched unit to terminal state.
- **Missions fail silently.** `_dispatch_mission_event` is called only for `mission_step_complete`, `mission_plan_ready`, `mission_complete` (`services/coordinator_service.py:2113/2432/2601/3998`). There is no `mission_failed` user notification; a failed run emits an internal `RUN_FAILED` audit event and stops.
- **Budget-exceeded pause is silent.** The dispatcher blocks and transitions the run to `PAUSED` (`modules/coordination/dispatcher.py:645-666`) but `escalation_service.notify_budget_exceeded` is dead code — zero non-test callers.
- **A failed mission's board card renders "done".** `_RUN_STATE_TO_BOARD_STATUS` maps `failed` and `cancelled` → `"done"` (`services/orchestration_board_bridge.py:303-315`).
- **Breaker trips are silent.** `PlaybookSchedulerService._fire_playbook` skips the tick with only `logger.warning` when `breaker_is_open` (`services/playbook_scheduler.py:180-188`); the breaker also only gates cron — manual/webhook launches bypass it.
- **Missed cron ticks are lost undetected.** Playbook schedules live in APScheduler `MemoryJobStore`, recreated from DB at boot, no `misfire_grace_time`/`coalesce` (`services/playbook_scheduler.py:127-134`).
- **Playbooks are NOT silent anymore** (problem statement honesty): every terminal run fires `playbook_complete`/`playbook_failed` + an auto-report (`api/recipe_executor.py:1714/1993-2008`, PRD-185 S4). Board tasks notify both outcomes. The silent set is the five bullets above.
- **The judge exists, per-task, advisory-becoming-gated.** `VerificationService` — wired cross-model LLM judge, 4 dims 0–1 (`modules/coordination/verification.py:41`), thresholds `COORDINATOR_VERIFICATION_*` 0.7/0.4, output-hash cached; PRD-200 S1 wired `_apply_verdict_fail` to requeue once (`modules/coordination/reconciler.py:553-625`). There is **no run-level output judging** and no business rubric anywhere.
- **`PlaybookQualityService`** scores execution *mechanics* (5 heuristic dims), manual-only via `POST /assess-quality` — never auto-invoked. `workflow_recipes.success_rate` is declared, dashboard-read, **never written**.
- **Delivery is already session-independent.** PRD-128 `NotificationDispatcher` + `notifications` inbox + telegram/slack/webhook fan-out + `auto_reporting` quiet-hours (`core/services/notification_dispatcher.py:45-62` — 14 event types). In-app is a 30s poll; no email/mobile push; deep links TODO.
- **Approval machinery is ready, one seam unwired.** `evaluate_approval` (always_ask / auto_below_budget / full_auto, fails safe) + durable `ApprovalGrant` + ApprovalsInbox + resume-after-grant. `SUBJECT_PLAYBOOK_RUN` is defined (`core/models/approval_grants.py:53-55`) with **zero references** — `_requeue_subject` handles only `board_task` and `tool_call` (`api/approval_grants.py:182-290`).
- **No rerun primitive.** Re-running means re-POST `/{recipe_id}/execute`; nothing copies a prior execution's inputs. `retry_of`/`attempt_count` columns exist but are written only by the stall auto-retry (`services/task_reconciler.py:218-247`).
- **Step prompts are unversioned shared state.** `workflow_recipes.steps` JSONB, whole-playbook `version` string only; **no per-execution prompt override** — `execute_recipe` accepts `input_data` only (`api/workflow_recipes.py:832-833`). A watcher "tweak" today would mutate every future scheduled run.
- **Chat cannot carry the report-back.** Chat turns are synchronous streaming HTTP (`api/chat.py:461-497`); no background path writes an assistant message. `ScheduledTaskService`'s docstring promises chat injection; `_trigger_agent_chat` executes the agent and **discards the output** (`services/scheduled_task_service.py:386-422`).
- **Event substrate is partial.** LISTEN/NOTIFY push exists for board tasks only (`board_events` / `board_task_available`); missions emit 45 typed events into `orchestration_events` — an audit table with **no subscribers** (read-only REST poll, `api/missions.py:924-1000`). Four parallel event mechanisms, no bus.
- **Cost is readable.** `llm_usage` rows per `execution_id` (tokens + dollars) + `compute_execution_metrics` (`services/report_service.py:39-146`); `orchestration_runs.budget_spent/tokens_used`; live ceilings enforced mid-run.
- **Loop precedents to copy, not reinvent:** board dispatch spine (5s + `pg_notify` wakeup, lease/sweep/SLA-notify, `services/board_dispatcher.py:484-538`), coordinator 5s tick, TaskReconciler 60s stall→fail→backoff-retry, boot reaper, fcntl single-owner scheduler lock (`main.py:353-467`).
- **Integration health has no substrate** (why Watch-type E is out of scope §9): nothing marks `ComposioConnection` `error`/`disconnected` at runtime; no connection event type in the dispatcher; `expires_at` unwatched.

## 3. Findings → fix → story

| # | Finding (grounded §2) | Fix | Story |
|---|---|---|---|
| 1 | No watch entity/loop anywhere | `watches` + `watch_events` registry + WatchService | S1, S2 |
| 2 | Mission failure + budget-pause silent to user | `mission_failed` / `mission_budget_paused` notification events at the coordinator/dispatcher terminal+pause boundaries | S4 |
| 3 | Failed mission board card shows "done" | map `failed→"failed"` (keep `cancelled→"done"`) | S4 |
| 4 | Breaker trip silent; manual runs bypass unnoticed | `playbook_benched` notification when the scheduler skips on open breaker (once per open period) | S4 |
| 5 | Missed cron ticks lost undetected | watcher tick compares croniter expected-fire vs latest `recipe_executions` row for scheduled watches | S5 |
| 6 | Terminal states reach no subscriber (audit-table events only) | fail-soft `watch_ingest` hooks at the three terminal choke points; heartbeat sweep as fallback | S3, S5 |
| 7 | No run-level output judging / business rubric | `RunVerdictService` extending the PRD-200 judge to run level, 6-dim rubric, 0–1 internal / 0–10 display | S6 |
| 8 | No rerun primitive; tweak would mutate shared definition | rerun endpoint copying inputs + `step_overrides` honored per-execution without touching `workflow_recipes.steps` | S7 |
| 9 | `SUBJECT_PLAYBOOK_RUN` defined, unwired | wire the `_requeue_subject` playbook_run resume branch → grant-gated rerun | S7 |
| 10 | "Change direction / find a way" has no owner | watch decision step choosing from allowed actions (rerun / tweak+rerun / replan / reassign / spawn-agent / escalate), all via `evaluate_approval`, bounded by `action_budget` | S8, S10 |
| 11 | Auto has no watch tools; no poll/wait tool exists | `platform_create_watch/list/get/cancel` + auto-create on Auto-launched missions | S9 |
| 12 | No user-visible watchlist | minimal Watchlist surface + API; board/bell stay the action surfaces | S11 |

## 4. Stories (test-first; CI is the only gate — no local runs)

### S1 · Watch registry schema — M · _core/models_
`watches`: `id` UUID PK, `workspace_id` FK NOT NULL, `created_by`, `owner_agent_id` NULL, `watch_type` (`mission|playbook_execution|scheduled_playbook`), `target_type`, `target_id`, `title`, `description`, `status` String(32) (`watching|acting|awaiting_approval|needs_attention|passed|failed|escalated|expired|cancelled`), `success_criteria` Text (intent snapshot), `failure_criteria` Text NULL, `quality_threshold` Float default 0.8, `check_interval_seconds` Int default 300, `last_checked_at`, `next_check_at`, `deadline_at` NULL, `policy` String(32) default `run_and_report` (`run_and_report|score_and_improve|watch_change|persistent`), `allowed_actions` JSONB, `action_budget` Int default 2, `actions_taken` Int default 0, `final_score` Float NULL, `final_verdict` Text NULL, `lineage` JSONB (ordered target chain — reruns/replans append), `version_id` (optimistic, house pattern), timestamps + `closed_at`.
`watch_events`: `id`, `watch_id` FK, `event_type`, `summary`, `snapshot` JSONB, `score` NULL, `action_taken` NULL, `requires_attention` bool, `event_key` String — **UNIQUE(watch_id, event_key)** for idempotent ingest, `created_at`.
Partial unique index: one non-terminal watch per (workspace_id, target_type, target_id). Alembic migration **chained on the current single head** (derive head by reading `alembic/versions` down_revision graph — do not run alembic locally).
**Test:** model round-trip + dedup index + FK integrity (seed `workspaces` first — PRD-158 lesson).

### S2 · WatchService — M · _services/watch_service.py_
Create/get/list/cancel + guarded `transition()` (allowed-transition map, house style of `orchestration_enums`), `ingest(event_key, ...)` idempotent via the unique key, single-writer via `FOR UPDATE SKIP LOCKED` when the tick claims due watches. `record_action()` increments `actions_taken`, hard-stops at `action_budget` → `needs_attention` + escalation. Lineage helper: `follow(new_target)` appends to `lineage` and repoints the live target (a rerun/replanned run stays the SAME watch).
**Test:** transition guard, idempotent double-ingest, budget hard-stop, lineage repoint.

### S3 · Terminal choke-point hooks — S · _fail-soft, knowledge_flywheel pattern_
Call `watch_ingest_terminal(...)` (try/except-log, never breaks the producer) from: mission terminal boundary (`_complete_verified_run` + fail/cancel paths in `coordinator_service`), playbook terminal (`recipe_executor` success block + `_fail_execution`), board-task complete/fail bridge. Payload: target ids, terminal state, output pointer, cost snapshot.
**Test:** hook fires on each terminal path; a raising watch service does NOT fail the producer (assert mission/playbook completes normally).

### S4 · Close the silent-failure holes — M · _independent of the registry; ships value even alone_
1. Add to `NotificationDispatcher.VALID_EVENT_TYPES`: `mission_failed`, `mission_budget_paused`, `playbook_benched`, `watch_verdict`, `watch_action`, `watch_escalation`.
2. Dispatch `mission_failed` (to `run.created_by`, reuse `_dispatch_mission_event`) at the run-failure boundary; `mission_budget_paused` at the budget-pause transition (retire the dead `notify_budget_exceeded` or wire it — one owner).
3. Board mapping: `failed→"failed"` in `_RUN_STATE_TO_BOARD_STATUS`; update any tests asserting the old lie.
4. `playbook_benched` once per breaker-open period from the scheduler skip path (dedupe via latest `watch_events`/notification lookback, not new state).
**Test:** each event lands one `notifications` row with correct `event_type`/`link_id`; mapping test flipped.

### S5 · Watcher tick — M · _services/watch_ticker.py on the UnifiedScheduler_
Config: `WATCHER_ENABLED` default true, `WATCHER_TICK_SECONDS` default 300. Registered in `main.py` lifespan inside the fcntl-locked scheduler block (single owner across workers). Each tick: claim due watches (`next_check_at <= now`, SKIP LOCKED batch), per watch: refresh target state (cheap status read), deadline/`expired` handling, scheduled-playbook checks (croniter expected-fire vs latest execution → missed-run event; `breaker_is_open` → benched event), write `watch_event` only on **meaningful change** (state delta, not "still running"), reschedule `next_check_at`. Event hooks (S3) do the fast path; the tick is the fallback and the trend/missed-run brain. Notify only per §7.9 rules (terminal, threshold breach, action taken, approval needed, expiry).
**Test:** tick idempotence (two ticks, one event), missed-cron detection with frozen clock fixtures, no-noise (running→running writes nothing).

### S6 · Run-level verdict — M · _modules/coordination/run_verdict.py_
`RunVerdictService` reusing `VerificationService`'s cross-model judge selection + output-hash cache, scoring the **run-level output** (mission `output_summary`+task outputs / playbook `output_data.final_output`+auto-report) against the watch's `success_criteria` on 6 dims: `business_usefulness, completeness, evidence_quality, clarity, actionability, reliability` — each 0–1, weighted mean, **internal 0–1, displayed ×10**. `reliability` folds in mechanics (tool failures/retries from `step_results` — reuse `PlaybookQualityService` heuristics rather than re-deriving). Verdict written to the watch (`final_score`, `final_verdict` with reasoning + caveats), `watch_verdict` notification with score + one-paragraph explanation. LLM cost attributed: `request_type='watch'`, execution-id threading like the recipe path so `llm_usage` shows supervision cost.
**Test:** judge-stubbed scoring math, threshold verdict boundaries (0.79 fail / 0.80 pass), cache hit on identical output hash, cost row written.

### S7 · Corrective actions v1: rerun + tweak — L · _the "acting" half_
1. `POST /api/v1/playbooks/{recipe_id}/executions/{execution_id}/rerun` — new `RecipeExecution` copying `input_data`, `retry_of=<original>`, `attempt_count+1`, `triggered_by='watch_rerun'` (or `'rerun'` when human), optional `step_overrides`.
2. `step_overrides` (`{step_id: {prompt_template: ...}}`) stored in `execution_metadata`, merged at execution start in the executor's step resolution — **`workflow_recipes.steps` is never mutated**. Overrides recorded in the watch event for before/after comparison.
3. Gate: estimate rerun cost from the original run's `llm_usage` sum → `evaluate_approval`; auto path launches immediately; ask path creates `ApprovalGrant(subject_kind=SUBJECT_PLAYBOOK_RUN)` + `approval_pending` notification, watch → `awaiting_approval`.
4. Wire `_requeue_subject` `playbook_run` branch: grant → launch the stored rerun spec; deny → watch `needs_attention`.
**Test:** rerun copies inputs + lineage; override applied to prompt assembly without touching the template row (assert template unchanged); grant→resume launches; deny→no launch; budget-exceeded rerun estimate → ask path.

### S8 · Direction-change actions: replan / reassign / spawn-agent — M
Actions on mission watches, same `evaluate_approval` gate + `action_budget`:
- **replan** — drive the existing coordinator replan path (`failed→replanning` is an allowed run transition; reuse the `platform_replan_mission` service method) with the watch's diagnosis as replan context.
- **reassign** — requeue a failed/stalled task to a different capable agent (reuse dispatcher capability matching; skip if none — `no_capable_agent` → escalate).
- **spawn_agent** — instantiate from an existing **blueprint** (never free-form; blueprint validation path stays authoritative), then reassign/replan onto it. **Always grant-gated in v1** regardless of policy (§8 Q5) — `full_auto` workspaces auto-approve under the dollar ceiling via the policy engine, consistent with PRD-193 semantics.
- **escalate** — escalation-service board card (reuse `escalate_stalled_task` shape) + `watch_escalation` notification; terminal for the watch unless renewed.
**Test:** each action once under `full_auto` (auto) and `always_ask` (grant card), budget exhaustion → escalate, spawn honors blueprint `rules` defaulting (onboarding-wall regression guard).

### S9 · Auto tool surface — S · _modules/tools/discovery/actions_watches.py + handlers_
`platform_create_watch` (target, criteria, threshold, policy, deadline), `platform_list_watches`, `platform_get_watch`, `platform_cancel_watch` — registered like `actions_missions`. Auto-create (Q1): `platform_create_mission` and the playbook-execute tool create a `run_and_report` watch by default (workspace setting `watch_auto_create`, default ON), success_criteria seeded from the user's request text (intent snapshot).
**Test:** tool schema `required[]` matches handler defaults (blueprint-rules-wall lesson), auto-create writes the watch, handlers workspace-scoped.

### S10 · Decision step — M · _policy first, LLM only where it earns it_
Deterministic policy table drives the flow per watch `policy` (e.g. `run_and_report`: terminal→score→notify, no actions; `score_and_improve`: below-threshold→diagnose→one tweak+rerun→rescore→final). LLM is used for exactly two bounded jobs: (a) failure/low-score **diagnosis** (inputs: error, step_results compact, verdict reasoning → one-paragraph cause + proposed action + optional `step_overrides` draft), (b) tweak drafting. Both heartbeat-style single calls, cost-attributed to the watch, never unbounded loops — `action_budget` is the hard rail.
**Test:** policy table transitions golden-file tested; diagnosis stubbed; runaway guard (action_budget=0 → straight to escalate).

### S11 · Watchlist surface — M · _frontend + API_
`GET /api/v1/watches` (+`/{id}` detail incl. recent events), `POST /api/v1/watches/{id}/cancel`, ws-scoped perms mirroring the board router. Minimal UI: Watchlist panel in the Command Center (table: title, type, status, last check, score/verdict; cancel action), polling hook (30–60s, house pattern), entries link to the board card / execution. **Update the committed `orchestrator/reports/route-manifest.json` by hand** (sorted `{method,path}` + count bump) — the frontend route-contract lane reads the committed file, it does not regenerate.
**Test:** route-contract lane green; API list filtered by workspace; UI renders empty + populated states.

### S12 · CI hardening — S
Full test sweep for the above; ensure single alembic head; don't trip source-grep guard lanes (repoint any guard that referenced moved lines); no new routes missing from the manifest; migration boots via docker-entrypoint `alembic upgrade heads` semantics (idempotent, additive only).

## 5. Sequencing

S1→S2 (registry) → S3+S4 in parallel (hooks + silent-hole fixes; S4 has standalone value) → S5 (tick) → S6 (verdict) → S7 (rerun/tweak) → S8 (direction-change) → S9+S10 (tools + decision) → S11 (surface) → S12 rolling. Single PR, story-per-commit (`feat(prd-204): S<N> …`).

## 6. Verification (CI is the only gate — no local runs)

orchestrator-tests lane green (new units + integration); migrations lane single-head; frontend route-contract lane green against the updated manifest; CodeQL/security lanes clean (no new path/injection surface — watch APIs are read/cancel + gated actions); malware-scan lane (post-incident) green. Behavioural assertions in tests, not local servers.

## 7. Baseline capture — freeze, then measure the delta

Pre-merge truths to freeze in the PR body (queryable, tenant-safe):
- `mission_failed` notifications possible: **0** (event type absent) → post: present + tested.
- Terminal-notification coverage metric (notifications dossier headline): failure-path mission coverage **0%** → post: 100% of mission terminal states dispatch.
- Missed-cron detectability: **0** (no mechanism) → post: detected within one tick.
- Run-level output verdicts: **0** anywhere → post: every watched terminal run scored.
- Rerun primitive: **absent** → post: endpoint + grant-gated path.
Post-merge product metrics (PRD §14 of the source draft): % Auto-launched work with a watch, time-to-verdict, failures caught unprompted, tweak delta (score before/after), watch cost per verdict (`llm_usage where request_type='watch'`).

## 8. Open questions — Gerard's call (decide, don't let me defer — CLAUDE.md §12)
*Recommendations below are applied as build defaults; flag at PR review to flip.*

1. **Auto-create watches for Auto-launched work?** → **REC: yes**, `run_and_report` policy, workspace-settable (`watch_auto_create` ON). Cheap (one row + existing events).
2. **Default quality threshold?** → **REC: 0.8 internal (displayed 8/10)**, per-watch configurable. Note it is stricter than the task-judge's 0.7 pass — intentional at run level.
3. **Rerun without approval?** → **REC: no blanket yes — route through `evaluate_approval` with the rerun's estimated cost.** `auto_below_budget`/`full_auto` auto-rerun under ceiling; `always_ask` gets the grant card. Consistent with PRD-193 + §12.3.
4. **Report-back in chat?** → **REC: out of this PRD.** Delivery = bell + channels (works session-independently today). "Background→chat message" is a real primitive with threading/UX questions — candidate PRD-205; the `ScheduledTaskService` discard bug gets fixed there, not here.
5. **spawn_agent autonomy?** → **REC: always grant-gated in v1** (blueprints only); `full_auto` + under-ceiling auto-approves via the policy engine. Revisit after live data.
6. **Board mapping change (`failed→failed`) is user-visible** — cards that used to show done now show failed. → **REC: ship it; it was lying.**
7. **Integration/Data watch (source-draft type E)?** → **REC: own PRD** — zero substrate today (no runtime connection-error writer, no event type, `expires_at` unwatched). Needs a connection-health story first.
8. **Scale presentation?** → **REC: 0–1 internal everywhere (matches judge/thresholds), ×10 only at display/notification edge.**
9. **Watch lineage across corrective actions?** → **REC: the watch follows the work** — reruns/replanned runs append to `lineage`, one watch one verdict; a new watch per attempt would fragment the story.

## 9. Explicitly out of scope (each → its own future decision, not silently dropped)

Unified event bus (hooks + sweep suffice at this scale; a fifth event mechanism would be the real risk); chat-message injection (§8 Q4 → PRD-205 candidate); integration/data watch (§8 Q7); multi-week business-outcome watch (source-draft type F); email/mobile-push delivery (notification plane extension, P2-26 digest territory); user-custom rubric UI; anomaly detection/predictive intervention. Source-draft §13 concurs.

---

*Traceability: source draft = Gerard's "Auto Watcher" side-PRD (2026-07-16); grounding = 5-agent repo survey @ `8e0543211` (missions/board, playbook/scheduling, events/notifications, scoring/heartbeats/integrations, PRD-landscape); closes deep-review open loops on missions (advisory-only verification → owned outcomes), playbooks (silent bench/missed-run), notifications (failure-path silence); addresses OS-review F090 (human can't watch Auto act) / F023 (green-over-failed) / F085 (babysit-or-trust-blindly); reuses PRD-128 dispatcher, PRD-193/196 grants+inbox, PRD-200 judge+gate patterns, PRD-185 breaker/telemetry, PRD-161 dispatch-spine idioms. Canonical terms: Mission, Playbook (=Recipe), Auto, Watch. PILOT lens applied — cold-start emptiness is not a defect signal.*
