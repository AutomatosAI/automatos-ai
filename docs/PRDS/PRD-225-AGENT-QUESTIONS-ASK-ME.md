# PRD-225 — Agent Questions: the ASK ME Queue & Telegram Bridge

> **Status:** Draft for rollout planning — written 2026-08-27, not yet scheduled.
> **Origin:** Munder Difflin deep review (2026-08-27); the feature Gerard flagged directly ("one tab where all agent questions are queued… plug it into our Telegram channel"). Review artifact:
> https://claude.ai/code/artifact/f31677a8-f2cb-47fe-b7dd-f705d764418b
> **Type (per CLAUDE.md §3):** Extension of the approval-grant system + one net-new surface (the tab). The core design decision honors "no new tables when an existing one fits."

## 1. Overview

When work can only move with a human decision, the question becomes **task state, not a message**: the ask is recorded against the thing it blocks, one tab aggregates every open ask across the floor (each showing the cascade of downstream work stuck behind it), answering unblocks the work automatically, and the Q&A trail stays on the subject forever as decision history. Questions also travel: a `question_pending` notification reaches Telegram through the existing channel stack, and a Telegram **reply answers the question** — the round trip already exists in code; only the correlation is missing.

Adopted from Munder Difflin's ASK ME design (their best feature), with their key rules kept: an entry may be a question *or* a human-only to-do (approve a purchase, provide credentials, test on a device); asks are short markdown decisions, never pasted reports; **agents never idle-wait for an answer** — work parks, other work continues, the answer is picked up when it arrives. Telegram is still an unbuilt roadmap item for them; this ships it first.

## 2. Current reality (grounded)

- **Zero question constructs exist.** `agent_question|pending_question|ask_user|ask_human|request_input|user_input_required` → 0 hits across `orchestrator/` and `frontend/` (verified 2026-08-27).
- **ApprovalGrant is the seed — durable, subject-scoped, auto-resuming, but binary.** `core/models/approval_grants.py:58-123` (`pending|granted|denied|revoked|expired`, subjects `board_task|playbook_run|tool_call`, hot index on subject). On grant, `_requeue_subject` re-queues a blocked board task and `_resume_tool_call` re-executes an interrupted tool call (`api/approval_grants.py:172-309`). **The grant endpoint takes no body — no free-text answer can reach a blocked agent anywhere in the platform.**
- **Notification fan-out is ready.** `NotificationDispatcher` with 21 event types incl. `approval_pending`, destinations `in_app|telegram|slack|webhook`, quiet hours with urgent bypass (`core/services/notification_dispatcher.py:45-255`).
- **Telegram works both directions, uncorrelated.** Outbound `sendMessage` driver (`channels/drivers/telegram.py:105-154`); inbound platform-detecting webhook with replay/dedup guards captures reply context (`api/webhooks.py:186-213, 374-470`) and routes the text to an agent as a *fresh message* via `UniversalRouter` — nothing maps a reply to a pending item.
- **The queue UI patterns exist.** Command Center tabs with count badges — a new tab is one `TABS` entry + one `tabCounts` line + one component (`frontend/components/command-center/command-center-shell.tsx:45-111`); ApprovalsInbox (`governance/approvals-inbox.tsx`) and WatchlistTab are the structural models; `DecisionsNeededWidget` already aggregates "needs a human" KPIs (`api/kpi_api.py:361-450`).
- **No agent-facing ask tool.** `platform_send_notification` is fire-and-forget and its `event_type` enum omits `approval_pending` (`actions_auto_reporting.py:95-145`, `:110-120`).
- Dead vocabulary warning: `RunState.AWAITING_HUMAN` + `TASK_HUMAN_*` events have **zero writers** (the mission review surface was deliberately retired, `api/missions.py:1353-1355`) — this PRD must not build on them.

## 3. Goals

- G1: Any executing lane (board task, playbook run, tool call — same subjects grants cover) can raise a free-text ask that parks the subject and returns immediately.
- G2: One **Questions** tab shows every open ask with markdown rendering, the blocked-cascade, an answer box, and dismiss-without-answering; answered asks keep their trail per subject.
- G3: Answering resumes the subject automatically (reusing grant resume machinery) and the answer reaches the agent's next execution context; a confirmation lands in chat.
- G4: Questions reach Telegram; a Telegram reply (or `/answer <id> …`) answers them from the phone.
- G5: Inbound channel traffic that *directs work* passes a per-channel trust gate (`strict | communication_only | allow_all`), default strict; correlated answers always flow.

## 4. Non-goals

- No email or SMS delivery (no platform mailer exists; out of scope).
- No interactive answer widget in the *background* chat path in v1 — `chat_messenger.py:165` is text-parts-only; the chat surface links to the Questions tab instead. (Widget for in-turn asks may reuse `ToolApprovalWidget` patterns later.)
- No changes to mission verification flow; no resurrection of `AWAITING_HUMAN`.
- No Slack/WhatsApp/Discord answer bridges in v1 — drivers exist; correlation ships Telegram-first and the seam is channel-generic.

## 5. Design

### Component A — the ask model: extend `approval_grants`, don't sibling it

Per CLAUDE.md §4 ("no new tables when an existing one fits"): a question **is** a grant whose decision is words instead of a boolean. Extend `approval_grants`:

- `kind` VARCHAR — `'approval'` (default, existing rows) | `'question'`.
- `question_md` TEXT — the ask, markdown, ≤ ~700 chars encouraged (formatting doctrine in PRD-226).
- `options` JSONB nullable — optional discrete choices (rendered as buttons; free text always allowed).
- `answer_text` TEXT, `answered_by` — the human's answer.
- `asked_by_agent_id` — who raised it.
- `channel_refs` JSONB — outbound delivery correlation, e.g. `{"telegram": {"chat_id": …, "message_id": …}}`.

Status vocabulary is reused: `pending` = open ask; `granted` = answered; `denied` = dismissed-without-answer (no fabricated answer — the subject stays blocked and the asker may re-ask, matching ASK ME's dismiss semantics); `expired` via existing `expires_at`. **The trail is rows**: re-asks are new rows against the same subject; history = query by `(subject_type, subject_id)` on the existing hot index. Alembic migration must land on the single head (house rule after the 4-heads incident).

### Component B — `platform_ask_human` (3-file tool pattern)

`platform_ask_human(subject_type, subject_id, question, options?, expires_hours?)`:
parks the subject (board task → `blocked` + `blocked_reason="Awaiting human answer (ask #N)"`, same pattern as the approval gate at `api/board_tasks.py:1074-1082`), inserts the question row, dispatches a new `question_pending` event through `NotificationDispatcher` (add to `VALID_EVENT_TYPES`), and **returns immediately** — park, don't wait. Registered via `actions_*.py` + `handlers_*.py` + `platform_executor.py` map, like every platform action.

### Component C — answer & resume

- `POST /api/v1/approval-grants/{id}/answer` `{answer_text}` (or a chosen option): writes the answer, flips to `granted`, resumes via the **existing** `_requeue_subject` / `_resume_tool_call`, and appends the Q&A into the subject's execution context (board task `planning_data.human_qa[]`; tool-call resume payload) so the agent's next run sees it verbatim.
- Confirmation into chat via the existing `deliver_background_message` seam ("Answered: … — <agent> resuming <task>").
- Existing grant/deny endpoints remain approval-kind only; `deny` on a question = dismiss.

### Component D — the Questions tab

Command Center tab `questions` (entry + count + component, `command-center-shell.tsx:45-111`), following ApprovalsInbox structure: pending-first, newest ask on top; each card renders `question_md` as markdown, the assignee/asker badge, the answer box (⌘-Enter to send), option buttons when present, dismiss, and **the cascade** — downstream work transitively blocked behind this subject, resolved from `parent_task_id` + `OrchestrationTaskDependency` (cycle-safe, capped list, "+N more"). List reuses the existing grants list route filtered by `kind=question` — no new list endpoint. Bell: add `case 'question'` deep-link to the tab (the missing `approval_grant`/`watch` cases are fixed in PRD-227).

### Component E — the Telegram bridge

- Outbound: `question_pending` → channel stack → store the sent `message_id` per channel in `channel_refs`.
- Inbound (`api/webhooks.py`, before `UniversalRouter`): a message whose `reply_to_message_id` matches a stored ref answers that question through Component C; explicit fallback `/answer <id> <text>` for clients that lose reply threading; everything unmatched flows to existing routing. Replay/dedup guards already present (`:450-470`).

### Component F — the ingress trust gate

Per-channel `trigger_mode` on the channels config (`core/models/channels.py`): `strict` (default — inbound *directives* require operator approval; they land as pending items in the Questions tab, which doubles as the approvals surface for ingress) | `communication_only` (chatter routes, directives held) | `allow_all`. Conservative classifier: unconfident ⇒ directive. Correlated answers (Component E) and explicit `/answer` bypass the gate — they are responses, not directives. All config through `config.py` / channel settings; no bare `os.getenv`.

## 6. Waves & acceptance criteria

**Wave 0 — model + tool.**
- [ ] Migration adds Component A columns; single alembic head; existing grant flows unaffected (regression tests on approve/deny/resume).
- [ ] `platform_ask_human` parks a board task (blocked + reason), creates the row, fires `question_pending`, returns without blocking the tool loop; unit tests cover all three subject types.

**Wave 1 — tab + answer path.**
- [ ] Questions tab lists open asks with markdown + cascade; count badge live; answer resumes the subject (board task re-queued and executed; the run context contains the Q&A); dismiss leaves the subject blocked with trail intact.
- [ ] Route-manifest procedure followed for the answer endpoint: `RouterSpec` if separately mounted, regenerated committed `orchestrator/reports/route-manifest.json` with bumped `route_count`, matching `frontend/lib/api-client.ts` path — route-contract CI green.

**Wave 2 — Telegram.**
- [ ] A question reaches the configured Telegram chat; replying to that message answers it (trail shows `answered_by` with channel provenance); `/answer <id>` works; unmatched messages behave exactly as today.

**Wave 3 — trust gate.**
- [ ] `strict` holds an inbound directive as a pending item and nothing executes; `allow_all` restores current behavior; answers always pass. Gate decisions logged.

CI-only verification throughout; frontend stories verified via the existing component-test patterns (`governance-approvals-inbox.test.tsx` as the model).

## 7. Technical considerations

- Resume machinery is reused, not duplicated — any behavior change to `_requeue_subject`/`_resume_tool_call` needs the approval-kind regression suite.
- Notification quiet hours apply to `question_pending` with the standard urgent bypass; blocked-cascade size may warrant urgency escalation (open question 3).
- The `channel_refs` write happens post-send on the dispatcher path — delivery failure leaves a perfectly usable in-app question (fail-soft, matching `sender.py` semantics).
- PRD-229 depends on this PRD: unanswerable orchestrator clarifications escalate into this queue.

## 8. Open questions (Gerard)

1. Dismiss semantics: subject stays blocked (their model — the god re-asks) vs dismiss-also-unblocks ("proceed with your judgment"). Proposal: stays blocked; an explicit "answer: use your judgment" is one click.
2. Should `question_pending` bypass quiet hours by default, or only when the cascade exceeds N blocked tasks?
3. Telegram target: the workspace default chat (`telegram_default_chat_id`) or a dedicated questions channel?
