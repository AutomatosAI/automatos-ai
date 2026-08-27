# PRD-224 — The Ticket Lane: Auto as Line Manager

> **Status:** Draft for rollout planning — written 2026-08-27, not yet scheduled.
> **Origin:** Munder Difflin deep review (2026-08-27) + Gerard's scoping. Review artifact:
> https://claude.ai/code/artifact/f31677a8-f2cb-47fe-b7dd-f705d764418b
> **Type (per CLAUDE.md §3):** Extension. Every primitive exists; the increment is one routing lane and one enum value.

## 1. Overview

Auto today has two ways to get work done from chat: **delegate the turn** (a specialist answers inline, then it's over) or **launch a mission** (the planner staffs a large multi-step goal with whatever agents fit). Missions are project staffing. This PRD adds the missing middle lane — **line management**: "have my accountant agent chase the invoices" becomes a board ticket assigned to that named agent, started, supervised to a verdict, and reported back into the thread where it was asked.

Decision record (Gerard, 2026-08-27): *missions are designed for larger tasks with multiple agents — that is not agent management or assignment. Auto needs to be able to assign tickets/tasks to single agents or write up playbooks and manage them.*

## 2. Current reality (grounded)

**The ticket primitives already exist and are good:**

- `platform_create_task` accepts `assigned_agent_name`, priority, tags, `parent_task_id`, an optional `approval_action` gate, and initial status — its own examples say *"raise a task for the researcher to check competitor pricing"* (`orchestrator/modules/tools/discovery/actions_board_tasks.py:10-75`).
- `platform_assign_task(task_id, agent_name)` assigns by name (`actions_board_tasks.py:171-198`).
- `platform_update_task_status → "in_progress"` "triggers immediate agent execution if an agent is assigned" — the handler calls `_launch_task_execution` directly (`handlers_board_tasks.py:287-296` → `api/board_tasks.py:1126` → `AgentFactory.execute_with_prompt`).
- The board dispatcher claims `assigned` tasks with leases (`services/board_dispatcher.py:64 claim_tasks`, FOR UPDATE SKIP LOCKED), scans SLA breaches (`:311`), and blocked tasks >24h auto-spawn an escalation card (`services/escalation_service.py:26-92`).
- Playbook authoring is complete from chat: create / update / add / edit / delete steps / schedule / execute / inspect execution (`actions_playbooks.py:12-403`).

**What's missing — the lane and the loop:**

- **AutoBrain has no action for it.** The action space is `RESPOND | DELEGATE | MISSION` (`WORKFLOW` deprecated) — `consumers/chatbot/auto.py:47-65`, consumed at `api/chat.py:391-432`. DELEGATE routes the *chat turn* (agent answers inline); MISSION suggests the big hammer. "Have my accountant do X" has no path that files a managed ticket.
- **Watches cannot supervise a board task.** `platform_create_watch.target_type` enum is `mission | playbook_execution | scheduled_playbook` only (`actions_watches.py:16-110`; `core/models/watches.py:91 target_type`). The whole supervision machine — policies, quality scoring, bounded corrective action, escalation, chat narration — is unreachable for tickets.
- **Ticket completion never reports back.** The PRD-205 background→chat seam (`services/chat_messenger.py:186 deliver_background_message`) has exactly two producers: watch verdicts (`services/watch_notifications.py:87-100`) and scheduled tasks (`services/scheduled_task_service.py:446-465`). A filed ticket goes dark unless the human opens the Command Center.
- **Chat-created tickets start on the fallback poll.** `notify_task_available` fires only from `api/board_tasks.py:398,632,816,862`, never from `handlers_board_tasks.py` — latency, not correctness.

## 3. Goals

- G1: A named single-agent request in chat produces an assigned board ticket, not an inline answer and not a mission.
- G2: Every Auto-assigned ticket is supervised: done / failed / below-bar lands back in the originating thread without the human polling the board.
- G3: Auto resolves agent names against the live roster and honors named routing ("have Jim…", "my coding agent…").
- G4: Recurring asks surface a playbook suggestion (authoring tools already exist; the habit is PRD-226 doctrine — this PRD only guarantees the mechanics work when invoked).

## 4. Non-goals

- No multi-agent tickets — that is a mission, and missions already work from chat (`platform_create_mission`, `handlers_missions.py:92-149`).
- No new board columns, statuses, or tables. No changes to Guardrails/Blueprints/SDLC (covered, per Gerard).
- No CLI-agent wrapping, file mailboxes, or any Munder Difflin mechanism transplant — pattern only.
- No new HTTP routes (this PRD is tool-lane + enum work; zero route-manifest churn).

## 5. Design

### Component A — the ASSIGN lane in AutoBrain

- Add `Action.ASSIGN` to `consumers/chatbot/auto.py` (`WORKFLOW` stays deprecated; do not reuse it).
- Assessment guidance (the three-lane doctrine, also in PRD-226): **DELEGATE** = the specialist answers *this conversation*; **ASSIGN** = a named/single agent does work *off-thread, on the board*; **MISSION** = multi-agent project. Signals for ASSIGN: an explicit agent name or role possessive ("my accountant agent"), a deliverable that outlives the chat turn, no requirement for a conversational answer.
- Handling at `api/chat.py`: ASSIGN does **not** hard-code a bypass. It biases the turn — `tool_hints` steer to `platform_create_task` / `platform_assign_task` / `platform_update_task_status`, and a context directive instructs Auto to file the ticket (4-part contract description per PRD-226), optionally start it, attach supervision (Component C), and confirm in one line with the task reference. `ComplexityAssessment.target_agent_id` already exists (`auto.py:65+`) — populate it from roster name-matching.

### Component B — watches learn `board_task`

- Extend `WatchTargetType` with `board_task` (`core/models/watch_enums.py`; column is a plain string, `watches.py:91` — no schema migration, enum-in-code only).
- `platform_create_watch.target_type` enum gains `board_task`; `target_id` = the integer task id as string (`actions_watches.py:16`).
- `watch_decider`: terminal detection for board tasks — `done` / `failed` statuses; `review` with `review_feedback` counts as output to score; `blocked` past deadline → escalate. Score the task's recorded result against `success_criteria` exactly as mission outputs are scored today.
- `watch_actions`: bounded corrective action for tickets = re-run via the existing run-now path (`api/board_tasks.py:830`), respecting `action_budget`; escalation = the existing escalation card, and (once PRD-225 lands) a question-kind ask.
- Verdict narration rides the existing pipe unchanged: `watch_notifications.py` → `deliver_background_message` → Auto thread / originating chat.

### Component C — auto-supervision on assignment

- When Auto assigns a ticket via the ASSIGN lane, auto-attach a `run_and_report` watch (Component B) targeting it. Config dial `AUTO_TICKET_WATCH` in `config.py` (no `os.getenv` elsewhere), default **on** — an unsupervised ticket is the current failure mode, not a feature.
- Loudly visible activation (per standing feedback): the assignment confirmation line names the watch ("supervised — I'll report back here").

### Component D — sub-second start

- `handlers_board_tasks.py` create/assign/status handlers call `notify_task_available` exactly as the HTTP layer does (`api/board_tasks.py:398,632,816,862`) so an assigned ticket is claimed on the LISTEN wake, not the fallback poll.

## 6. Waves & acceptance criteria

**Wave 0 — mechanics (D + B enum groundwork).** No behavior change visible yet.
- [ ] `notify_task_available` fires from chat-side create/assign/update handlers (pytest asserts the NOTIFY emit, pattern of existing board-event tests).
- [ ] `board_task` accepted by watch create/decider with terminal detection unit-tested (done, failed, review-with-feedback, deadline).

**Wave 1 — the ASSIGN lane.**
- [ ] "Have <existing agent> do <task>" in chat yields: a `BoardTask` with `assigned_agent_name` resolved, `created_by_type="agent"`, a one-line confirmation with the task id — and **no inline answer, no mission**.
- [ ] Ambiguous/unknown agent name → Auto asks in-thread rather than guessing (roster lookup via `platform_list_agents`).
- [ ] Assessment eval: gold-set cases for the three lanes pass (gold sets stay LOCAL — never committed; repo is public).

**Wave 2 — supervision + report-back.**
- [ ] Auto-assigned ticket has an active watch (`target_type='board_task'`); completion/failure produces a chat message in the originating thread via the existing PRD-205 seam, with the provenance badge.
- [ ] `AUTO_TICKET_WATCH=off` produces no watch and says so in the confirmation line.
- [ ] Corrective re-run respects `action_budget`; budget exhaustion escalates (card today; PRD-225 ask once available).

All verification is CI-only (nothing runs locally, per workspace rules); frontend untouched in this PRD except copy in the confirmation line (rendered by existing chat surfaces).

## 7. Technical considerations

- **No migrations, no new routes** — `target_type` is a string column; all new behavior is enum + handlers + prompt. Route manifest untouched.
- Watch uniqueness: partial unique on `(workspace_id, target_type, target_id)` (`watches.py:53`) already prevents double-supervision; re-assign of the same ticket repoints via `lineage` per the existing rerun pattern (`watches.py:145`).
- The dead `RunState.AWAITING_HUMAN` vocabulary is **not** used here (zero writers today; see review §6c) — tickets block via existing `blocked` status only.
- Interaction with PRD-227: agent-driven status moves become visible on the live board once 227's SSE fix lands; the lanes are independent but 227 makes this one legible.

## 8. Open questions (Gerard)

1. Default start mode for an ASSIGN ticket: start immediately (`in_progress`) or wait for the agent's next heartbeat (`assigned`)? Proposal: immediate when the user's phrasing is imperative, heartbeat otherwise — but a single default is simpler.
2. Should ASSIGN tickets created without a named agent (e.g. "get someone to do X") fall back to matcher-based auto-assign, or always ask? Proposal: ask — reflexive auto-assign is how missions behave, and this lane exists to be deliberate.
3. `AUTO_TICKET_WATCH` default on — agreed?
