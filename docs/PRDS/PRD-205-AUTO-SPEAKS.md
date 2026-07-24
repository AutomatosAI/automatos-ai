# PRD-205 — Auto Speaks: background→chat delivery (verdicts land in the conversation, not just the bell)

> Status: APPROVED FOR BUILD 2026-07-17 (Gerard) — §8 recommendations applied as build defaults; override at PR review.
> Grounded @ `ecfe3a11a` (origin/main, 2026-07-17, post-#553). Relationship: completes PRD-204 (Auto Watcher) — watcher verdicts/actions/escalations become conversational; fixes the PRD-77 scheduled-task output discard at the same primitive; extends PRD-128's delivery plane (bell/channels) with the chat surface; reuses PRD-161's LISTEN/NOTIFY SSE lane.
> **Build size: M–L (one PR, 8 stories) · Risk: Medium** — one additive migration, one new service seam, no new HTTP routes; the live-receive frontend work is deliberately minimal (no state-management migration).

---

## 1. What this is

A **background→chat delivery primitive**: server-side producers (the watcher, scheduled tasks — later anything) can post an assistant-authored message into a chat conversation after the originating HTTP turn is gone. When you tell Auto "run this and watch it," the verdict comes back *in that conversation*: "Watched it. First run scored 6.4 — the report was too operational — so I tightened the SCRIBE step and reran. 8.4/10, passed." Today that sentence has nowhere to go: it lands in the notification bell only, and the PRD-77 scheduled-task runner literally throws its agent's output away.

Delivery targeting: the **originating chat** when known (captured at watch creation), else a per-user **"Auto" thread** per workspace (net-new concept — the canonical place Auto speaks unprompted). Messages carry a persisted `source` so the UI badges them "Auto · background". An open chat receives them live via the existing LISTEN/NOTIFY SSE lane; closed chats surface them through the history list's `updated_at` re-sort (already shipped by PRD-220) plus the existing bell.

**Framing (CLAUDE.md §3): Extension.** `save_message` already works from background sessions (verified — it touches only its own `db`, no request state); the SSE lane exists; the badge slot in `message.tsx` exists. The net-new pieces are the messenger seam, the Auto-thread concept, two additive columns, one executor context injection, and a small live-receive hook.

**What this is NOT:** not a react-query migration of the chat UI (out of scope, §9); not email/mobile push; not multi-agent group chat; not a second LLM pass to "compose" messages (v1 posts the prose producers already write).

## 2. Current reality (grounded)

- **`save_message` is background-safe but only ever called inside a synchronous turn.** Signature + full behavior at `consumers/chatbot/service.py:391-467`: validates chat, inserts the `Message`, commits, bumps `chat.updated_at` (which re-sorts the PRD-220 history list for free, `service.py:333`). Zero request-state dependencies. Its only callers: the user-message write (`api/chat.py:333`) and two assistant writes inside the streaming generator (`service.py:2283/:2402`). No system/welcome/bare-insert endpoint exists anywhere.
- **The PRD-77 discard is total.** `scheduled_task_service._trigger_agent_chat` (`:386-422`) runs `factory.execute_with_prompt(...)`, extracts `llm_text`, and `logger.info`s the first 200 chars (`:408-419`). The assistant text is `result["result"]` (`agent_factory.py:1277-1298`). Nothing is saved anywhere a user can see.
- **`messages` has no author/source signal.** Columns: `role` (CHECK `user|assistant|system`, `core.py:1174`), `parts`, `attachments`, `retrieval_context`, `context_trace` (`core.py:1144-1176`). `GET /{chat_id}/messages` returns `{id, role, parts, attachments, createdAt}` only (`api/chat.py:704-712`). The frontend badge slot exists and renders `message.metadata.source` (`components/chatbot/message.tsx:294-313`) — but `metadata` is frontend-only and not persisted, so any badge dies on reload.
- **`chats.user_id` is INTEGER `users.id`** (`core.py:1105`) — the Clerk-string trap (#513) applies to every background write; the resolution pattern is the coordinator's `User.clerk_user_id` lookup (`coordinator_service.py:306-310`). `Watch.created_by` is a Clerk string (`watches.py:74-76`).
- **No default/system/pinned conversation concept exists** (grep-verified). Chat rows are created in exactly one place: `ChatService.create_chat` (`service.py:279-300`), called only from the streaming POST (`api/chat.py:294/:320`).
- **`origin_chat_id` cannot be captured today.** `caller_context["conversation_id"]` IS the chat id (`service.py:136-137`, fed at `:2238`), but `PlatformActionExecutor` injects only `_agent_id` / `field_id` / `_created_by` into handler params (`platform_executor.py:946-1005`) — the conversation id never reaches `handlers_watches.py`. The `watches` table has no suitable column (`lineage` is strictly the target chain).
- **Live-receive doesn't exist.** Chat messages are `useState`-only — no react-query, no polling, no inbound SSE (`lib/chat/hooks.ts:32-399`); a background insert is invisible until reload. The reusable push lane: LISTEN/NOTIFY channel `board_events` (`services/board_events.py:35-64`) → SSE `GET /api/v1/tasks/stream` (`api/board_tasks.py:458-483`, workspace-filtered `board_events.py:183-194`) → always-mounted hook `use-board-event-stream.ts` (mounted `command-center-shell.tsx:92`) which currently branches only on `board_changed`.
- **Two dead chat routes by declaration order** (the exact PRD-220 `/search` failure mode, unfixed elsewhere): `PATCH /vote` (`api/chat.py:759`) is shadowed by `PATCH /{chat_id}` (`:737`); `GET /agents` (`:812`) is shadowed by `GET /{chat_id}` (`:656`).
- **Alembic: TWO heads at this commit** (`prd204_watch_registry` and `w3_post201_merge_heads` — both merging the same parents; #551 carries the join and is OPEN). Yesterday's lesson applies: never author a second join of the same parents.
- PRD-220 (#549) changed **no schema and no routes** — widget chats are ordinary `chats` rows distinguished only client-side; history ordering is `updated_at DESC`.

## 3. Findings → fix → story

| # | Finding (grounded §2) | Fix | Story |
|---|---|---|---|
| 1 | No background write path; no create/find-chat helper outside the turn | `ChatMessenger` service seam (post + find-or-create Auto thread) | S1, S2 |
| 2 | No persisted author/source; badge dies on reload | `messages.source` JSONB + surface in `/messages` + map to the existing badge slot | S3, S7 |
| 3 | No Auto-thread concept; `chats` can't mark one | `chats.kind` (`user|auto`) + partial unique index | S2, S3 |
| 4 | `conversation_id` never reaches watch handlers; `watches` can't store it | executor injection (`_origin_chat_id`) + `watches.origin_chat_id` column | S4 |
| 5 | Watcher verdicts/actions/escalations are bell-only | `watch_notifications` seam also posts to chat | S5 |
| 6 | Scheduled-task output discarded (PRD-77) | deliver via ChatMessenger + notification; kill the discard | S6 |
| 7 | Open chat can't receive a pushed message | `chat_changed` on the existing SSE lane + minimal refetch-merge | S7 |
| 8 | `PATCH /vote` + `GET /agents` dead-shadowed | reorder above `/{chat_id}` + regression tests | S8 |

## 4. Stories (test-first; CI is the only gate — no local runs)

### S1 · ChatMessenger — the one seam — M · _services/chat_messenger.py_
`post_background_message(db, *, workspace_id, text, source, chat_id=None, clerk_user_id=None, link_type=None, link_id=None) -> Message | None`. Mechanics: resolve integer user via `User.clerk_user_id` (coordinator pattern; never write a Clerk string into `chats.user_id`); validate `chat_id` belongs to the workspace → append via `ChatService.save_message` with `parts=[{"type":"text","text":...}]` (match the AI-SDK shape the existing assistant writes use — verify against `service.py:2283`); no/invalid chat → fall back to the S2 Auto thread for that user; set `messages.source` (S3). A fail-soft wrapper `deliver_background_message(...)` (try/except-log, knowledge_flywheel pattern) is what producers call — a chat failure must never break a watcher tick or scheduled task. Emit the S7 `chat_changed` notify after commit.
**Test:** Clerk resolution, workspace-mismatch rejection, fallback path, parts shape parity with in-turn writes, fail-soft (raising messenger never propagates).

### S2 · The Auto thread — S · _find-or-create, per user per workspace_
`find_or_create_auto_chat(db, workspace_id, user_int_id) -> Chat`: `chats.kind='auto'`, title `"Auto"`, `visibility='private'`, one per (workspace, user) via partial unique index; race-safe (IntegrityError → re-select). Ordinary chat in every other way — history list, deep-link `/chat?chatId=`, deletion allowed (recreated on next post).
**Test:** idempotent find-or-create, uniqueness under simulated race, appears in `/history` ordered by `updated_at`.

### S3 · Migration + source surfacing — S
One additive migration: `messages.source` JSONB nullable (`{"origin": "watcher"|"scheduled_task", "label": "Auto · background", "link_type": ..., "link_id": ...}`), `chats.kind` String(20) NOT NULL default `'user'` + CHECK (`user|auto`) + partial unique index `(workspace_id, user_id) WHERE kind='auto'`, `watches.origin_chat_id` UUID nullable. **Chaining (§8 Q7): check the head graph at build time — if #551's join has merged, chain on it; otherwise chain single-parent on `prd204_watch_registry` and DO NOT author another join of the same parents.** Surface `source` in `GET /{chat_id}/messages` (`api/chat.py:704-712`).
**Test:** migration round-trip on the models; `/messages` returns `source`; old rows return `source: null`.

### S4 · Capture `origin_chat_id` at watch creation — S
In `platform_executor.py`, mirror the `_MISSION_ATTRIBUTED` injection block (`:988-1005`): lift `caller_context.get("conversation_id")` into `params["_origin_chat_id"]` for the watch tool actions (and the mission/playbook launch actions whose handlers auto-create watches — `handlers_missions.py:134`, `handlers_playbooks.py:500`). `create_watch`/`auto_create_watch` persist it to `watches.origin_chat_id`. Never trust an LLM-supplied `chat_id` tool arg for this (the `config.get("chat_id")` precedent at `handlers_missions.py:110` is caller-supplied and stays untouched).
**Test:** executor injects for watch+launch actions only; handler persists; LLM-supplied param cannot spoof the injected key (injection overwrites).

### S5 · The watcher speaks — M
`watch_notifications` (the shared dispatch seam from PRD-204 S6) additionally calls `deliver_background_message` for `watch_verdict`, `watch_action`, `watch_escalation`: target `watch.origin_chat_id`, else the Auto thread of `watch.created_by`; text = the prose the watcher already composes (score + one-paragraph reasoning / action summary / escalation reason), `source.origin="watcher"`, `link_type="watch"`, `link_id=watch.id`. Bell notification unchanged (§8 Q3: both surfaces in v1). Message ends with the target link (board card / execution) where one exists.
**Test:** verdict posts to origin chat; no-origin falls back to Auto thread; bell still fires; chat failure doesn't break the notification (fail-soft ordering).

### S6 · Scheduled tasks deliver — S · _the PRD-77 fix_
`_trigger_agent_chat` (`scheduled_task_service.py:386-422`): posts `result["result"]` via `deliver_background_message` (`source.origin="scheduled_task"`, target = the task creator's Auto thread, or a task-configured chat if the `agent_scheduled_tasks` row carries one — builder verifies the table's columns) and dispatches the existing notification path. The 200-char `logger.info` stays as a trace; the discard dies. Empty/error results post nothing but notify `agent_error` (existing event type).
**Test:** output lands as an assistant message with source; empty result → no message; error path notifies.

### S7 · Live receive + badge — M · _frontend + SSE_
Backend: `notify_chat_event(db, workspace_id, chat_id, user_id)` — a sibling emitter on the SAME `board_events` channel (`event="chat_changed"`, payload `{workspace_id, chat_id, user_id, event}`); called by ChatMessenger post-commit. Frontend: `use-board-event-stream.ts` gains a `chat_changed` branch → dispatches a window-level event; `useChat` (`lib/chat/hooks.ts`) listens and, when the event's `chat_id` matches the open chat, re-runs `getChatMessages` and merges missing messages by id into its `useState` (append-only; never clobbers an in-flight stream); the history sidebar refetches its list (cheap — `updated_at` already re-sorted). Badge: map persisted `source.label` into the existing `message.metadata.source` slot (`message.tsx:294-313`) so background messages render an "Auto · background" chip that survives reload. Per-user targeting is client-side (payload carries `user_id`; the hook drops events for other users) — the SSE lane's workspace filter is unchanged.
**Test:** hook branch (unit), merge-by-id no-dupe, foreign-user event dropped, badge renders from persisted source (component test at existing depth).

### S8 · Chat-router order fixes — S
Move `PATCH /vote` and `GET /agents` above the `/{chat_id}` routes (or convert to non-colliding paths — reorder is the minimal fix), with regression tests asserting both resolve (the PRD-220 `/search` fix pattern, `api/chat.py:608-610`). No manifest change (no route added/removed; paths unchanged).
**Test:** route-order regression tests for `/vote`, `/agents`, and `/search` (lock all three).

## 5. Sequencing

S3 (migration) → S1+S2 (messenger + thread) → S4 (capture) → S5+S6 in parallel (producers) → S7 (live+badge) → S8 anywhere (independent). Single PR, story-per-commit `feat(prd-205): S<N> …`.

## 6. Verification (CI is the only gate — no local runs)

orchestrator-tests green (units above; workspaces FK seeded; dispatcher seam patched in every test that posts — the PRD-204 lesson); Frontend CI green (tsc baselined, vitest, eslint, route-contract — **no manifest delta expected**; if any route is added after all, hand-update the committed manifest); migrations lane: no NEW multi-head introduced by this PR (red-until-#551 is pre-existing); CodeQL/security/malware lanes clean.

## 7. Baseline capture — freeze, then measure

Pre-merge truths (tenant-safe): background-authored chat messages possible: **0** (no code path — `save_message` callers are 3, all in-turn); scheduled-task outputs delivered to any user surface: **0%** (discard at `scheduled_task_service.py:408-419`); watcher verdicts in chat: **0**; live receive of a background message in an open chat: **impossible**; dead chat routes: **2** (`/vote`, `/agents`). Post-merge metrics: % watch verdicts delivered to a chat (target: 100% where origin known); scheduled-task outputs delivered (target: 100% of non-empty); time-to-visible for an open chat (target: ≤ SSE latency + one fetch).

## 8. Open questions — Gerard's call (decide, don't let me defer — CLAUDE.md §12)
*Recommendations applied as build defaults; flip at PR review.*

1. **Delivery target?** → **REC: originating chat when known; per-user Auto thread fallback.** A workspace-shared Auto thread can't satisfy `chats.user_id NOT NULL` cleanly and leaks cross-user context; per-user threads match the notification plane's targeting.
2. **Compose with an LLM pass?** → **REC: no in v1** — producers already write the prose (verdict reasoning, agent output). A composer pass is cost + latency + a new failure mode for zero information gain.
3. **Suppress the bell when chat delivery succeeds?** → **REC: keep both in v1** (different surfaces, different attention models); revisit if users report double-ping fatigue. Escalations must always keep the bell.
4. **Real-time transport?** → **REC: piggyback `chat_changed` on the existing `board_events` channel + always-mounted SSE** — one new event value + client-side user filter; a dedicated channel/stream doubles the moving parts for no v1 benefit. Chat pages outside the Command Center shell fall back to on-open fetch (acceptable v1; note the shell is where Auto-heavy users live).
5. **Fix the two shadowed chat routes here?** → **REC: yes (S8)** — same bug class, two-line moves, regression-tested; leaving known-dead routes in the router we're editing would be negligent.
6. **Persisted signal: `messages.source` column vs marker inside `parts`?** → **REC: column.** `parts` is the AI-SDK render contract — the codebase already keeps `retrieval_context`/`context_trace` off it for exactly this reason (`core.py:1156-1165`).
7. **Migration chaining with #551 in flight?** → **REC: chain on #551's join if merged at build time, else single-parent on `prd204_watch_registry`; never author a second join of the same parents** (yesterday's #545×#548 lesson).
8. **Which producers in v1?** → **REC: watcher + scheduled tasks only.** Missions/playbooks already notify via bell and their outcomes reach chat through the watcher's verdict; wiring them directly would double-post. Heartbeats stay on their existing routing (`report_to`).

## 9. Explicitly out of scope (each → its own future decision, not silently dropped)

React-query migration of the chat message state (S7 does targeted refetch-merge instead); email/mobile-push delivery; widget/embed-SDK chats as background-post targets (anonymous browser sessions — no resolvable user; ordinary logged-in widget chats work via the normal path); LLM-composed conversational summaries (§8 Q2); a workspace-shared (multi-user) Auto channel; backfilling `source` onto historical messages.

---

*Traceability: grounding = single-agent survey @ `ecfe3a11a` (2026-07-17) + PRD-204 review pack @ `8e0543211`; closes the PRD-204 §8-Q4 deferral (chat delivery) and the PRD-77 output-discard defect; extends PRD-128 (delivery plane), PRD-161 (LISTEN/NOTIFY SSE lane), PRD-220 (history updated_at ordering, `/search` route-order precedent). Canonical terms: Auto, Watch, Mission, Playbook, Command Center. Chat-500 lesson #513 (Clerk-string vs integer user id) enforced at the messenger seam. PILOT lens applied.*
