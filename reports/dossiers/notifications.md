# Module Dossier — Notifications

> Phase-2 component deep-review. Scope: the **notification plane** — the unified event dispatcher, the in-app inbox (bell), routing preferences, the auto-reporting override layer, and the widget-callback destination orchestrator. The per-platform channel drivers (Telegram/Slack/webhook wire protocols) are the **`channels`** dossier's remit; here `channels.sender.send_to_channel` is treated as a dependency.
>
> Tier: standard · Status: live · Prior finding: F090 (board SSE, FIXED). Security/adversarial-input lens is deliberately **out of scope** (separate Opus pass).
>
> Grounding: `file:line` in `automatos-ai/orchestrator` + `frontend`, cross-checked against live production data captured 2026-07-04 (`reports/dossiers/evidence/data/notifications.md`, `board-tasks.md`, `deliverables.md`, `real-data-inventory.md`).

---

## A. What it is

A single in-house notification pipeline (PRD-128) that turns platform events — heartbeat done, task complete/failed, mission plan ready, playbook complete, report submitted, SLA breach — into (a) rows in an in-app inbox surfaced by a bell dropdown, and (b) optional external pushes to Telegram/Slack/webhook via the shared channel sender. Every producer (`heartbeat_service`, `board_tasks`, `coordinator_service`, `recipe_executor`, `report_service`, `board_dispatcher`) funnels through one class, `NotificationDispatcher`, which reads per-workspace/per-user `notification_preferences`, applies a Wave-2 `auto_reporting` override (routes + quiet hours), fans the event out to every enabled destination, and writes the in-app row without committing (the caller owns the transaction). A separate but adjacent piece, the **widget-callback destination dispatcher** (`services/destinations/dispatcher.py`), is the only part with real retry logic: it delivers storefront callback requests to a Site's configured destinations with 3-attempt exponential backoff, logging each attempt to `widget_event_log`. Real-time board push (F090, `board_events.py`) is a *sibling* transport — Postgres `LISTEN/NOTIFY` → SSE — that refreshes the Command Center board, distinct from and not integrated with the bell feed.

---

## B. What it does — real implementation & data path

**The one dispatcher.** `NotificationDispatcher.dispatch()` (`core/services/notification_dispatcher.py:90-241`) is the whole spine:

1. Loads `auto_reporting` settings (`_load_auto_reporting`, l.245-255) → if `enabled`, `route_for_event` can override the per-event preference and `is_quiet_hours` can force non-urgent traffic to `in_app` (l.150-163). `urgent`/`security` severities bypass quiet hours (l.152-153).
2. Reads `notification_preferences` for `(workspace_id, event_type, user_id OR NULL)` with **user-row-shadows-workspace-default** merge semantics (`_get_preferences`, l.271-310).
3. **No prefs at all → one `in_app` row** (l.139-148), so a completion event is never silently dropped *when the code path reaches dispatch*.
4. For each enabled, non-`silent` destination: `in_app` → raw `INSERT INTO notifications … ` **without commit** (`_insert_in_app`, l.327-360); `telegram|slack|webhook` → `_format_external_message` (icon + title + agent + 200-char-truncated body + a hard-coded `"🔗 (deep link pending)"` placeholder, l.364-391) then `send_workspace_notification`; `channel` → look up a specific `channel_connections` row and deliver via its platform (l.395-426). Returns `{"dispatched_to": [...]}`.

**The egress helper.** `send_workspace_notification(workspace_id, message, channel)` (`core/services/notification_service.py:34-86`) is a thin façade over `channels.sender.send_to_channel`. `orchestrator|direct|in_app|silent|None` short-circuit to no-op success (l.31, 52-53); everything else opens a `SessionLocal()`, calls the driver, **never raises** (l.66-68). 24 call sites across heartbeat, webhooks, harness, destinations, and the dispatcher itself.

**Producers (who emits, with what status):**
- Heartbeat (`heartbeat_service.py:1176-1274`) → `heartbeat_complete`; correctly maps `status="ok" if hb_status=="success" else "error"` (l.1193-1194). 1,701 rows, current (07-03).
- Board task complete (`api/board_tasks.py:179-217`) → `task_complete status=ok`; board task failed (`api/board_tasks.py:222-260`, PRD-161 S3) → `task_failed status=error`. Data: 90 `task_complete`, and **only 2 `task_failed` rows ever, both 07-03** — the failure emit is brand-new and barely exercised.
- SLA breach (`services/board_dispatcher.py:344-370`) → `task_sla_breach status=warn`; note it `db.commit()`s its own transaction (l.362), unlike the in-request emitters that never commit.
- Mission (`services/coordinator_service.py:289-315, 2413`) → `mission_step_complete` / `mission_complete` / `mission_plan_ready`. Data: 65 + 14 + **1** `mission_plan_ready action_required`.
- Report (`services/report_service.py:260-263`) → `report_submitted`. **2,099 rows — the single largest event class.**
- Playbook (`api/recipe_executor.py:1624-1634, 1687-1697`) → `playbook_step_complete` (default `silent`) / `playbook_complete` (default `in_app`).

**In-app read/mutate surface.** `api/notifications.py` (631 lines): `GET /api/notifications` (paginated, `dismissed_at IS NULL`), `/unread-count`, `POST /{id}/read`, `/read-all`, `/{id}/dismiss`, plus `/api/notification-preferences` GET/PUT. Every query enforces `workspace_id = ctx.workspace_id AND (user_id = ctx.user_id OR user_id IS NULL)` (l.150-153) — workspace + own-or-broadcast rows only. The `notifications` table (`alembic/versions/prd128_notifications.py`) has a partial index on unread rows (l.83-87). **No retention/purge job exists anywhere** (grep-confirmed).

**Frontend.** `notification-bell.tsx` — a popover; `useUnreadNotificationCount` polls `/unread-count` **every 30 s** (`use-notifications-api.ts:78`), the list loads only when the popover opens (l.87-106). `linkFor()` maps `link_type` → an in-app route (l.42-64). Settings tab at `/settings/notifications` edits preferences.

**Widget-callback destination dispatcher** (`services/destinations/dispatcher.py`, 316 lines) — the retry story: `dispatch_one_destination` tries each destination up to `MAX_ATTEMPTS=3` with `BACKOFF_SECONDS=(0,5,15)` (l.39-40, 172-174), honours `SendResult.retryable`, and writes a `callback_delivered`/`callback_failed` row to `widget_event_log` on every attempt (l.185-216). Submitted fire-and-forget via `asyncio.create_task` (`enqueue_callback_dispatch`, l.294-316). **Writes to `widget_event_log`, never to `notifications`** — grep-confirmed.

---

## C. Honest quality — how good is it *really*?

**Maturity: 2 / 5 (below adequate).** The architecture is right — one dispatcher, one egress helper, preference-driven fan-out, per-tenant read scoping, transaction-joined writes — and that is genuinely better than the sprawl it replaced. But the plane's actual behaviour in production is **misleading**: it confidently reports success and stays silent on the failures that matter, and the live data proves it.

**Defect 1 — the plane went blind to the outage it exists to surface (CONFIRMED, current, production).** `playbook_complete` notifications **stopped 2026-06-16** in the data — the same day `deliverables` production stopped — while `board_tasks` kept closing the daily "Recipe:" playbook tasks as `done` through 07-03. Root cause is structural, not incidental: `playbook_complete` is dispatched **only on the happy path** at `recipe_executor.py:1687`, reached only after every step succeeds. On step failure the executor calls `_fail_execution` and `return`s at `recipe_executor.py:1588` and `:1602` — and **`_fail_execution` dispatches no notification** (grep-confirmed: zero `dispatch`/`NotificationDispatcher` inside it). There is **no `playbook_failed` event type** in `VALID_EVENT_TYPES` (`notification_dispatcher.py:45-60`) at all. So when the daily content playbooks began failing on OpenRouter 402 in mid-June, the notification plane didn't fire a warning — it simply stopped firing. The user's own inbox told them everything was fine for ~2.5 weeks. This is the single worst thing a notification system can do, and it is live right now.

**The asymmetry that proves it was an oversight:** the *board* path got a `_dispatch_task_failed` emitter in PRD-161 S3 (`board_tasks.py:222-260`) — same dispatcher, `status="error"`. The *playbook* path, sharing the identical `_dispatch_playbook_event` helper, never got the equivalent wiring. One producer learned to report failure; the sibling didn't.

**Defect 2 — 70% of volume is machine chatter that buries the signal (CONFIRMED).** Of 5,423 rows: `report_submitted` 2,099 + `heartbeat_complete` 1,701 = 70%. The events a human actually needs to act on — `mission_plan_ready action_required` (**1**), `task_failed` (**2**), `report_submitted critical` (**5**) — are a rounding error drowned in green ticks. There is **no batching, digesting, throttling, or severity-based summarisation** anywhere in the dispatcher; every heartbeat and every report writes its own inbox row. The bell's `9+` badge is permanently saturated with noise, which trains the user to ignore it — defeating the purpose.

**Defect 3 — declared-but-dead event types (CONFIRMED).** `agent_error` and `trigger_fired` are in `VALID_EVENT_TYPES` and carry `in_app` defaults (`hybrid.py:210,212`), the settings UI renders toggles for them, yet **neither has a single emitter** in the codebase (grep-confirmed) and **neither appears in 5,423 production rows**. The plane advertises 12 event types; ~10 are wired; of those, one silently broke. Users can configure notifications that can never arrive.

**Defect 4 — split-brain surfaces; "one event, one story" is not met (CONFIRMED).** The probe asked for consistency between the bell feed, board SSE, and channel egress. It isn't there:
- Widget callback deliveries/failures land in `widget_event_log` only, **never in the `notifications` inbox** (`destinations/dispatcher.py`, grep-confirmed) — a client callback that failed all 3 retries is invisible to the bell.
- Board real-time (`board_events.py` LISTEN/NOTIFY → SSE) is a *separate* transport that refreshes the board grid; it is **not** wired to the bell's unread count, so a `task_failed` row and the board's live status update travel two unconnected paths.
- The only retry/backoff logic in the whole module lives in the widget dispatcher; the main `NotificationDispatcher` external send is **single-shot, fire-and-forget** — if Telegram is down, the push is lost with a log line and `dispatched_to` simply omits it.

**Defect 5 — cosmetic-but-user-visible rough edges.** External messages always append a literal `"🔗 (deep link pending)"` (`notification_dispatcher.py:390`) — every Slack/Telegram notification for ~2 months has shipped a TODO to the end user. `heartbeat_complete` is emitted even for *failed* heartbeats (correct `status=error`, but the event name says "complete"): a semantic mislabel. And the SLA-breach path commits its own transaction while every other in-request emitter deliberately doesn't — inconsistent transaction discipline that means an SLA notification can persist even if the surrounding board sweep later rolls back.

**What is genuinely good (honest positives):** the read surface is clean, correctly scoped per-tenant with a real `user_id = :uid OR user_id IS NULL` predicate and no cross-workspace leak (`api/notifications.py:150-153`); the fan-out/override/merge logic is well-factored and unit-tested (`test_prd128_*`); the dispatcher's "never raise, never block the caller's primary work" contract is correctly implemented everywhere; the transaction-joined in-app write (no separate commit) is the right call; and the widget destination dispatcher is a solid, well-instrumented delivery loop. The bones are sound. The problem is that the plane is **honest about successes and silent about failures**, which is exactly backwards for a system whose job is to make an autonomous platform observable.

---

## D. Competitive teardown

The relevant field is **notification-infrastructure products**: Knock, Novu (OSS), Courier. All three solve, as commodity primitives, the exact gaps above.

**Knock** (`knock.app`) — commercial, the category leader.
- **Workflow steps as first-class primitives:** batch, delay, throttle, digest, fetch, plus conditions — Knock's own report names batch and delay as the most-used steps ([Knock features](https://knock.app/blog/state-of-notification-infrastructure-report-2025)). Automatos has **none** of these; every event is immediate and un-batched. This is precisely Defect 2, solved out-of-the-box.
- **In-app feed over WebSocket** with server-computed aggregate unread/read/seen counts and archive state ([Knock feeds](https://docs.knock.app/in-app-ui/feeds/overview)) — "clients connect… which receives notifications over a websocket." Automatos polls every 30 s and has read/dismiss but no seen/archive distinction.
- **Preference model** is a PreferenceSet across four axes — categories, channels, channel_types, workflows — with **preference conditions** and **channel escalation** ("only notify on channel B if the user hasn't *seen* the message on channel A") ([Knock preferences](https://docs.knock.app/preferences/overview)). Automatos preferences are a flat `(event_type → destination)` table with no categories, no conditions, no escalation.

**Novu** (`github.com/novuhq/novu`, MIT core) — the OSS option, directly self-hostable.
- Visual drag-and-drop **workflow editor** with conditions/delays/**digest**, an embeddable React/Next inbox ("6 lines of code") with archiving + unread filtering + search over **WebSocket**, subscriber preference management, and **delivery observability** (status, delivery rates, **failures**) across 15+ email / 14+ SMS / 5+ push / 4 chat providers ([Novu](https://novu.co/), [Novu GitHub](https://github.com/novuhq/novu)). Novu's "delivery observability → failures" is exactly Defect 1+4 as a shipped feature. It is MIT-licensed and deploys on Railway (the platform's own host) as a one-click template.

**Courier** (`courier.com`) — the API-first integrator.
- **Idempotency keys / deduplication** as a core guarantee: `Idempotency-Key` header, first response stored 24 h (configurable to 1 year), retries return the same result ([Courier idempotency](https://www.courier.com/docs/reference/idempotent-requests)). Automatos has **no idempotency or dedup** — the data shows heartbeat/report summaries stored twice each (per `real-data-inventory.md §2`), which idempotency keys would collapse.
- Automations for time-delayed, multi-channel sequences that **stop as soon as the user reads them**, plus digest/batch/throttle ([Courier guide](https://www.courier.com/blog/developers-guide-notification-apis)).

**Where Automatos actually stands:** it has the *inbox read surface* and *per-event routing* roughly at parity with the basic tier of these products, and its tight coupling to internal events (heartbeat, board, mission, playbook objects with deep-link `link_type`s) is a real advantage a generic vendor can't match cheaply. But on the **five things that make a notification plane trustworthy at scale — batching/digest, delivery retries + status tracking, idempotency/dedup, real-time transport, and a rich preference/escalation model — it is behind all three**, and one of those gaps (no failure emit) is actively misleading users in production. On the industry's universal contract — *at-least-once delivery + idempotent processing* ([webhook reliability 2026](https://www.digitalapplied.com/blog/webhook-reliability-idempotency-retries-engineering-reference-2026)) — Automatos delivers **at-most-once, fire-and-forget** on external channels.

---

## E. Verdict — Build / Extend / Adopt / Replace

**EXTEND now (fix the blindness this week); seriously EVALUATE adopting Novu for the delivery/workflow engine.** Split by layer, because the module is really two things:

**1. The in-app inbox + internal event bindings → EXTEND (keep, do not replace).**
The bell, the per-tenant read surface, the `link_type`→deep-link routing, and the six internal producers are tightly and correctly coupled to Automatos's own objects (BoardTask, Mission, playbook execution). No external product gives you `/assignments?tab=playbooks&execution=<id>` deep links for free — that binding *is* the value, and it makes Auto's work observable to clients, which is the North Star. Keep this and fix it (Section J).

**2. The delivery/workflow engine (batching, digest, retries, delivery-status, idempotency, escalation) → ADOPT-candidate: Novu (MIT, self-host).**
This is the classic build-vs-adopt line, and per the reuse-first rule it leans adopt. Everything in Defects 1–4 and all of Section D's gaps are **commodity primitives Novu ships**: workflow steps with digest/delay/throttle, an embeddable inbox over WebSocket, subscriber preferences, and **delivery observability including failures** — MIT-licensed, self-hostable on the platform's existing Railway host at ~$0 licence cost (infra only). Integration shape: keep `NotificationDispatcher.dispatch()` as the internal seam but have it `POST` a Novu workflow trigger (subscriber = `(workspace_id,user_id)`, payload = the current fields) instead of hand-rolling fan-out; keep the in-app read surface pointed at Automatos's own `notifications` table *or* migrate the inbox to Novu's `@novu/react` component (the bigger lift). Rough effort: a spike + one producer behind a flag ≈ 3–4 days; full cutover with digest/preferences ≈ 2–3 weeks.
- *Honest counter-argument for staying in-house:* the plane is ~1,700 LOC, the read surface already works, and Novu adds an operational dependency (a service to run, upgrade, monitor) plus a preference-model migration. If the team's appetite for another self-hosted service is low, the **EXTEND-only** path (Section J: add `playbook_failed`, a digest tick, dedup keys, and delivery retry in-house) closes the *critical* gaps for ~1 week of work without a new dependency. That is a legitimate choice.

**Do not REPLACE the inbox, do not BUILD a new plane.** Both would throw away correct, well-scoped code.

**Recommendation:** ship the **critical EXTEND fixes immediately** (failure emit + noise reduction don't wait for an adopt decision), then run a **1-day Novu spike** against the digest/delivery-status requirement and let Gerard decide adopt-vs-keep-extending on the evidence.

---

## G. Quality metric

Today "did the user get told?" is a vibe — and the data shows the vibe was wrong for 2.5 weeks. Make it a tracked number:

1. **Terminal-event coverage (the headline metric):** of all *terminal* executions (playbook/task/mission/heartbeat that reached a done **or failed** state), what fraction emitted a notification whose `status` matches the true outcome? **Today, for playbooks: ~0% on the failure path** (no `playbook_failed` exists), and the daily failing playbooks prove it. Target: 100%. This is directly computable by joining `recipe_executions.status` / `board_tasks.status` against `notifications` on `link_id` — a nightly reconciliation query, no new instrumentation.
2. **Signal-to-noise ratio:** `count(action-required events) / count(total notifications)`. Today ≈ `8 / 5,423 ≈ 0.15%`. A digest that collapses heartbeat/report chatter should push the *inbox* SNR to a usable level (target ≥ 5% of surfaced items are actionable).
3. **External delivery success rate:** `dispatched_to` contains the requested external channel ÷ times it was requested. Currently **uncounted** (fire-and-forget, no delivery log for the main dispatcher — only the widget path logs to `widget_event_log`). Adding a delivery-attempt log is a prerequisite to measuring this at all.
4. **Dead-event audit:** count of `VALID_EVENT_TYPES` with zero emitters (today: **2** — `agent_error`, `trigger_fired`). Target: 0. Feeds T3's per-module scorecard.

Metrics 1 and 2 are computable **from existing tables today** and should seed the T3 harness immediately — they'd have caught the outage on day one.

---

## H. Cost note

Negligible per-event compute — the dispatcher is SQL + optional one HTTP POST per external destination; no LLM in the path. The real cost lens is **storage and attention**, not tokens:
- `notifications` grows ~3,800 rows / 2 months with **no purge** (Section B) — unbounded table growth; the partial unread index mitigates query cost but the table itself only grows. A retention/archive job is cheap insurance.
- The 30 s unread-count poll (`use-notifications-api.ts:78`) is one lightweight `COUNT(*)` per active tab per 30 s against a partial-indexed predicate — fine at current scale, but it is *polling where the platform already has LISTEN/NOTIFY* (`board_events.py`); folding the bell onto that transport removes the poll entirely at no token cost.
- External sends inherit the channel driver's cost (out of scope here; see `channels`). The dominant "cost" is the **human cost of a saturated `9+` badge** — the noise problem is a quality cost, not a compute one.

---

## I. UX / surface

The bell is competent but surfaces the wrong things. Concrete changes:

1. **Digest the chatter, elevate the actionable.** Heartbeat/report "ok" events should roll into a single collapsible daily digest row, not 3,800 individual rows. `mission_plan_ready`, `task_failed`, `*critical`, `task_sla_breach` should be visually distinct (the code already has `status` → colour dot at `notification-bell.tsx:70-80`; extend it to a top "Needs attention" section that never gets buried).
2. **Make failures loud.** A failed playbook/task must produce a red inbox row *and* count toward unread — right now a failing playbook produces **nothing**. This is the single highest-value UX change and it's a backend fix (Section J).
3. **Unify with the Command Center.** The board already has real-time LISTEN/NOTIFY (F090); the bell polls a separate path. One event (`task_failed`) should drive both the board's live status *and* the bell badge from the same NOTIFY, so "the same event tells one story" (the probe's own words).
4. **Surface widget callback failures in the bell.** A storefront callback that exhausted its 3 retries is currently invisible outside `widget_event_log`; for the Shopify pilot this is a lost lead the operator never hears about. Route the terminal `callback_failed` through `NotificationDispatcher` too.
5. **Kill the `"🔗 (deep link pending)"` string** in external messages (`notification_dispatcher.py:390`) — resolve `link_type`+`link_id` to the same route `linkFor()` already computes on the frontend, or drop the line. Shipping a visible TODO to end users for 2 months is a credibility tax.
6. **Hide dead event types** from the settings UI until they have emitters (`agent_error`, `trigger_fired`), or wire the emitters — offering toggles for notifications that can't fire erodes trust.

---

## J. Upgrade path (prioritised by North-Star impact × effort)

**P0 — Stop the blindness (critical; ~1 day; highest North-Star impact).**
Add a `playbook_failed` event type to `VALID_EVENT_TYPES` (`notification_dispatcher.py:45-60`) and to `DEFAULT_NOTIFICATION_PREFERENCES` (`hybrid.py:203-213`, default `in_app`), then dispatch it from `_fail_execution` / the failure `return`s in `recipe_executor.py:1583-1602` with `status="error"` — mirroring the existing `_dispatch_task_failed` (`board_tasks.py:222-260`). Effect: the daily failing content playbooks immediately produce a red inbox row instead of silence. *This is a one-file-plus-enum change and it fixes the worst live defect in the module.* (Impact: very high · Effort: low.)

**P0 — Reconciliation metric + alert (critical; ~1 day).**
Ship the nightly "terminal-event coverage" query from Section G as a scheduled check (a playbook or heartbeat), emitting `report_submitted critical` when executions are `done/failed` with no matching notification. This is the safety net that catches the *next* silent break. (Impact: very high · Effort: low.)

**P1 — Reduce noise via a digest tick (high; ~3–5 days).**
Introduce a batching layer: heartbeat/report "ok" events accumulate and flush as one digest row per workspace per interval (hourly/daily, honouring the existing `auto_reporting.digest_frequency` field that's already in the settings schema at `auto_reporting.py:18` but **unused by the dispatcher**). Wire the already-defined `digest_frequency`/`quiet_hours` config into the dispatch path. (Impact: high · Effort: medium.)

**P1 — Delivery retry + status log for external sends (high; ~2–3 days).**
Give the main `NotificationDispatcher` external path the retry/backoff the widget dispatcher already has (`destinations/dispatcher.py:39-40,172-174` is the reference implementation to reuse — do not re-invent), and log each attempt so metric G-3 becomes measurable. Add an idempotency key on `(workspace_id, event_type, link_id)` to collapse the duplicate heartbeat/report writes noted in `real-data-inventory.md §2`. (Impact: high · Effort: medium.)

**P1 — Fold the bell onto LISTEN/NOTIFY (medium-high; ~2 days).**
Replace the 30 s unread poll with a subscription on the board-events transport (or a new `notification_events` channel), so the badge updates sub-second and one NOTIFY drives both board and bell. Reuses `board_events.py`'s proven listener pattern. (Impact: medium-high · Effort: medium.)

**P2 — Novu spike (decision-gating; ~1 day).**
Time-boxed: trigger one Novu workflow from `dispatch()` behind a flag, evaluate its digest + delivery-observability + inbox against the P1 items. Output is a keep-extending-vs-adopt recommendation for Gerard, not a cutover. (Impact: strategic · Effort: low.)

**P2 — Housekeeping (low; hours each).**
Retention/archive job for `notifications` (Section H); remove the `"🔗 (deep link pending)"` string or resolve real deep links (`notification_dispatcher.py:390`); reconcile SLA-breach transaction discipline with the other emitters (`board_dispatcher.py:362`); hide or wire the two dead event types. (Impact: low-medium · Effort: low.)

**Dependency order:** P0 items are independent and ship first (they need no design). P1-digest and P1-retry are independent of each other. The P2 Novu spike should run *before* investing heavily in a bespoke digest engine (P1-digest) — if adopt wins, Novu's digest replaces that work. Sequence: **P0 (both) → P2 spike → then P1 digest/retry per the spike outcome → P1 LISTEN/NOTIFY → P2 housekeeping.**
