# PRD-252: Command Centre board — every ticket opens where you click, says why it is where it is, and a review means Reject, Approve or Discuss

> **Status:** DRAFT 2026-10-02.
>
> **The owner's words:**
> - "Users are saying it's confusing."
> - "When I select a review in my Needs you widget it just takes me to the board and I see loads of tickets in review, when one needs me. It should open the actual ticket."
> - On review: "Reject — user enters why it's rejected… Approve — does this just mean the ticket is closed? Discuss — open in the canvas where I can discuss the task with the agent… then the ticket gets updated and put back in the queue."
> - "Tasks need numbers… 001, 002."
> - "The Command Centre doesn't scroll… we need to be able to scroll up so we can hide the stats… full-screen board, full-screen calendar, widgets… same for other pages."
>
> **Source:** TESTER's board-flow review page, traced from `main` at `19a16c722`, 2 Oct. The owner read it ("this reads really well") and asked for this PRD. Every fact below has a file reference under **Evidence**.

## 1. Problem: what the code does today

- **Clicks drop you on the board, not the ticket.**
  - Needs you's review rows link to the whole board. Its question rows link to the Questions tab.
  - The activity feed pushes `?task=<id>`, but nothing reads it (F218).
- **"A review I can't find."** Needs you lists every ticket in `review`, including mission step tickets. The board hides those, and a step in review is waiting for the mission's own checker, not for the owner.
- **Review buttons don't say what they do, and neither takes a note.**
  - The viewer calls Reject with no feedback. The API accepts feedback (it feeds owner corrections), but the agent is told "The owner sent it back without a note."
  - Approve has no note box, and the API ignores a note (F038). If the ticket carries an `approval_action` (for example, publish), Approve runs it without saying so.
- **A stage doesn't say why the ticket is there.**
  - Review has six causes: human mode, llm mode, missing file, nothing done, refused command, retries used up.
  - Blocked has seven: question, approval, spend ceiling, CLI question, mission paused, step failed, stopped by a person.
  - The `llm` review mode has no reviewer and behaves exactly like `human`.
- **Tickets can't be told apart.** Cards show no number. Every card says TASK; a mission step only adds a small `mission` tag. Titles repeat.
- **Five counters, five definitions.**
  - The Board tab badge counts Failed, Cancelled and Closed.
  - ATTENTION counts a grant-blocked ticket twice.
  - Needs you leaves out Blocked.
  - Auto's pill counts questions and decisions only.
  - The feed shows Blocked as "failed" and Review as "pending".
- **Drags skip the buttons.** There are no transition rules. Review → Done by drag never runs Approve's action.
- **No Assign or Cancel on the board.** An Inbox ticket refused with "Assign an agent first" can't be fixed there. Cancelled and Closed tickets open to an empty viewer.
- **The page head never scrolls away.** On Studio full-bleed pages, `<main>` is pinned to the viewport and only the tab body scrolls. On a laptop, the head and the KPI tiles leave the board and the calendar a few centimetres.

## 2. Requirements

### R1 — Every ticket reference opens that ticket
- The board reads `?task=<id>` and opens the viewer. If the ticket isn't in the loaded set (done more than 30 days ago, or a mission step), it fetches it by id.
- In Needs you:
  - a review row opens its ticket;
  - a question row opens that question inside its ticket;
  - an approval row opens the ticket at its approval.
- The activity feed, notifications, chat ticket cards and Auto's replies link the same way.
- **Acceptance:** one click from Needs you opens the right viewer for each of these: a task, a playbook step, a mission card, and a ticket done more than 30 days ago.

### R2 — Review means Reject, Approve or Discuss
- **Reject** has a required "What's wrong?" field, sent as the existing `feedback`. The redo brief leads with the owner's words. After 3 rejects on one ticket, the panel suggests Discuss.
- **Approve** has an optional note, stored on the ticket (fixes F038). The button names its effect from `planning_data.approval_action`, for example "Approve and publish". A toast confirms what happened.
- **Discuss (new)** opens a chat with the ticket's agent, with the ticket as context. Extend `/chat?ticket=<id>`, which today works for session tickets only.
  - "Update ticket and re-queue" writes the agreed brief onto the ticket, as its description plus a correction, and moves it to Assigned.
- Approve and Reject show an error toast when they fail. Today they show nothing.
- **Acceptance:**
  - the next run's prompt contains the reject note word for word;
  - the approve note is readable on the ticket;
  - Discuss re-queues the ticket with the agreed brief.

### R3 — Every card says why it is in its stage
- **A Review card shows its reason:** you asked to review it · the file it named is missing · it did nothing · a held command was refused · retries used up.
  - A mission step being checked by its mission shows "Mission checking", not Review, and never appears in Needs you.
- **A Blocked card shows a reason chip** (Question · Approval · Spend ceiling · Mission paused · Stopped by you) and the one action that unblocks it.
  - Give the spend-ceiling park a real release: raising the ceiling sends the ticket back to Assigned. Otherwise, stop saying it comes back on its own.
- Hide the `llm` review mode until a model reviewer exists.
- **Acceptance:**
  - every Review or Blocked card shows a reason;
  - no `llm` option remains when creating or editing a ticket.

### R4 — Tickets you can tell apart
- **A per-workspace number:** a new `board_tasks.workspace_seq`, assigned on insert from a per-workspace counter and never reused.
  - Backfill existing tickets in `created_at` order.
  - Show it as `#0042`. Mission and playbook steps show parent and step: `#0051.3`.
- **A type chip from `source_type`:**

  | Chip | `source_type` values |
  |---|---|
  | Task | user, chat, agent, agent_output, activity |
  | Playbook | recipe |
  | Mission | mission, orchestration_task |
  | Routine | heartbeat |

  A `>_ session` mark shows when a Claude Code session runs the ticket.
- **Number, type and title everywhere a ticket is named:** cards, the viewer, Needs you, the feed, notifications, chat ticket cards, and Auto's replies.
- **Auto names tickets by number,** and its task tools accept `#0042` wherever they take a task id. Night 6 saw Auto refer to "tasks 2 and 3" by list position.
- **Acceptance:**
  - two tickets with the same title can be told apart on every surface listed;
  - a ticket's number never changes;
  - the migration keeps one Alembic head, and every step survives a database that `create_all` already built (AGENTS.md → Migrations).

### R5 — One "Needs you" number
- **Needs you** = tickets in Review that are the owner's to judge (not mission checking), plus open questions, pending approvals, and failed tickets in the selected period.
- One backend endpoint serves both the count and the list.
- The Board tab badge, ATTENTION, the Needs you widget and Auto's pill all show that number.
- The activity feed uses the board's status words.
- **Acceptance:** for the same period, all four counters show the same number, and it equals the widget's row count.

### R6 — Drags match the buttons
- A drag that needs a decision is refused, with a message pointing to the button:
  - Review → Done: "Use Approve";
  - Review → Assigned: "Use Reject".
- A drag to In progress runs Run Now's path, so the ticket carries its corrections and answers, and the operator's consent is recorded.
- **Acceptance:** Review → Done by drag never skips an approval action.

### R7 — Assign and Cancel on the board
- Assign an agent, and Cancel, from the card menu and from the viewer.
- Cancelled and Closed become one stage, Cancelled. The viewer shows who cancelled the ticket and when.
- **Acceptance:** an Inbox ticket can be assigned without leaving the board.

### R8 — Work surfaces can take the whole screen
- On Studio full-bleed pages, the page head and the stats scroll away with the page.
- The tab bar sticks under the app header.
- The active tab's surface (board, calendar, widgets) then fills the viewport under the tab bar, keeping its own inner scroll for board columns and the calendar grid.
- A collapse control on the stats strip is optional. If added, remember its state per browser in `localStorage`, inside try/catch.
- Apply this to every page that `main-layout.tsx` renders full-bleed, not only the Command Centre. The phone layout stays as it is.
- **Acceptance:** at 1440×900, one scroll leaves only the app header and the tab bar above the board, and the board and the calendar each get at least 85% of the viewport height.

## 3. Order of work: one PR per step, each green in CI with screenshots ("GREEN ≠ RIGHT")

1. **R1, R2 (the Reject and Approve notes only) and R8.** No migration, and they remove the biggest confusion.
2. **R5, R3, R6 and R7.**
3. **R4 (migration and backfill) and R2's Discuss.**

## 4. Defaults for the open decisions (the owner may change any of them)

| | Decision | Default |
|---|---|---|
| D1 | Stages | Keep today's statuses, merge Closed into Cancelled, and show reasons as chips. |
| D2 | Drags | Refuse drags that need a decision (R6). |
| D3 | Notes | The reject note is required, with a Discuss hint after 3 rejects. The approve note stays on the ticket, not in agent memory. |
| D4 | Discuss | On every ticket type. |
| D5 | Numbering | `#0042` per workspace, with a step suffix (`#0051.3`). |
| D6 | Missions | Mission cards stay on the board, but their buttons are mission actions: Approve calls the mission approve endpoint, not ticket approve. Steps stay hidden and out of Needs you. |
| D7 | `llm` review | Hidden until a model reviewer exists. |

## 5. Tests

- **Vitest:**
  - each surface's link opens the right ticket;
  - Needs you row targets;
  - Reject requires a note;
  - type-chip mapping;
  - the counters share one source;
  - the sticky tab bar and the full-height surface (R8).
- **Pytest:**
  - the reject note reaches the redo brief;
  - the approve note is stored;
  - `workspace_seq` is assigned, backfilled and unique per workspace;
  - the task tools accept `#number`;
  - drag refusals.
- **Screenshots in every PR:** the board, a ticket opened from Needs you, the review panel, and R8 before and after at 1440×900.

## Evidence (main @ `19a16c722`)

- Needs you sources and links: `frontend/components/activity/widgets/needs-you-widget.tsx:68-74, 111-127`.
- The board drops step tickets: `frontend/hooks/use-board-tasks.ts:91`. The feed's link: `frontend/components/activity/activity-feed.tsx:241` (F218).
- Reject with no note: `board-task-viewer.tsx:496`. The reject API: `orchestrator/api/board_tasks.py:1146-1216`. The redo text: `services/ticket_redo.py:24`. Approve: `board_tasks.py:1039-1095`.
- Drag refusals are only 404, invalid status, no agent and mission: `board_tasks.py:1425-1460`. No branch anywhere on `review_mode == 'llm'`.
- The tab badge: `command-center-shell.tsx:116-123`. ATTENTION: `services/activity_service.py:640-685`. Feed labels: `activity_service.py:451, 469-474`.
- Cards use `task.id` only for drag and delete: `board-card.tsx:66,130`, `board-tab.tsx:274`.
- The pinned page: `frontend/components/layout/main-layout.tsx:103-106` (`sh-fullbleed flex-1 min-h-0 flex flex-col`) and `command-center-shell.tsx:172-223` (`cc-headrow`, then `cc-tabs`, then `cc-body`).
- The chat ticket seam: `frontend/app/chat/page.tsx:74-75`.
- Ledger findings this closes: F038, F039 (provenance comes from the type chip and number), F092 (sending back), F190 (stale result on re-run; clear it in R6's Run Now path), F218, F219 (re-check after R8).
