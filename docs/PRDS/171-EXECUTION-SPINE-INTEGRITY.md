# PRD-171 — Execution Spine Integrity (Wave 1)

**Status:** Draft v1 — pending approval
**Type:** Bugfix / Coherence (P0 spine repair)
**Priority:** P0 — the highest-leverage work in the plan; no dependencies; precondition for every wave that follows
**Owner:** Gerard Kavanagh
**Author:** Gerard Kavanagh + Claude (Opus 4.8)
**Date:** 2026-07-02
**Phase:** A — Coherence · **Size:** S–M · **Risk:** low (single-line critical + narrow spine fixes)
**Parent:** [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md)
**Source:** review §4 (Critical F001; High F023/F024/F025), §13 Wave 1, §9.5 step 1
**Findings register pinned to:** `37fdecc4e` — re-confirm each `file:line` on current `main` before editing
**Findings in scope:** F001, F023, F024, F025 (+ the §9.5 step-1 dead-import repoint at `tool_loop.py:55`)

---

## Operating Principle

> **One loop; every surface routes into it.** `ToolLoopExecutor` is the single, correct, model-driven loop.
> The spine's job is to deliver every unit of work — board task, mission, scheduled/heartbeat run, webhook,
> inter-agent delegation — *through* `execute_with_prompt` into that loop, and to record the true result of
> the run. This PRD does not add capability; it makes the capability that already exists actually execute.

---

## 1. Purpose

On `origin/main` today **Auto can converse but cannot orchestrate.** `agent_factory.py:1070` passes
`content_truncate_chars=0` to a `ToolLoopExecutor` constructor that PRD-157 renamed to accept only
`content_truncate_tokens` (`tool_loop.py:154`), with no shim. Every non-chat path therefore raises a
`TypeError` after its first LLM response; the retry `except` swallows it, budget is burned, and the call
returns `{"status":"error"}`. PRD-161's exactly-once dispatch spine then faithfully delivers every task into
this dead engine — and the board marks the failed run **`done`** because it never inspects the returned status.

This is why CI never caught it: **all twenty-plus tests construct `ToolLoopExecutor` directly**, bypassing
`execute_with_prompt`. Chat, voice, and widget build their own executor with the correct kwarg and are
unaffected — which is exactly why the regression hid behind green dashboards.

**Fixing F001 (one line) flips "Auto operates everything end-to-end" from false to true.** The three High
findings in this wave (F023, F024, F025) are the rest of the async spine that stays severed even after F001:
failed runs still close as `done`, long runs double-execute, and a kanban drag double-fires mission mirrors.
Wave 1 makes the spine deliver work into the loop **and** report the truth of what happened.

---

## 2. Background

### 2.1 What's working today (must not break)

- **The chat path.** `consumers/chatbot/service.py:1457` constructs the executor with the correct kwarg and
  runs a real model-driven loop. Chat, voice, and widget are unaffected by F001. Do not touch these paths.
- **PRD-161 dispatch spine.** `board_dispatcher.py` claims tasks `FOR UPDATE SKIP LOCKED`, notifies, sweeps,
  and enforces slots/SLA correctly. The spine *delivery* is sound; the engine it delivers into is dead. W1
  fixes the engine and the result-handling, not the dispatch mechanics.
- **Missions' own status handling.** Missions already branch on execution result and mark `failed` with error
  text — F023's fix makes board tasks behave the same way (parity, not invention).

### 2.2 What's broken / blocked

- **F001 — the non-chat agent engine is dead on every surface** (`agent_factory.py:1070` → `tool_loop.py:154`).
  Board tasks, missions, scheduled/heartbeat runs, webhooks, inter-agent delegation all `TypeError` after the
  first LLM response. **This gates the entire "Jarvis acting out" thesis.**
- **F023 — the board spine marks failed executions as `done`** (`board_tasks.py:900-921`). `_launch_task_execution`
  never inspects the `status:error` dict returned to it, so a failed run is closed successfully — masking F001
  and any future execution error as a green task.
- **F024 — no lease renewal, so long runs double-execute** (`board_dispatcher.py:176-205`). Any run exceeding
  the 600s lease is swept back to `assigned` and re-claimed as a duplicate. Breaks the exactly-once bar under
  lease expiry (review §13 Reliability pillar).
- **F025 — a kanban drag double-executes a mission-mirror task** (`board_tasks.py:550-555`). `PATCH status →
  in_progress` fires launch for *any* non-recipe source, so dragging a mission-mirror row re-runs work the
  mission already owns.
- **Dead learning sink** (`tool_loop.py:55`, review §9.5 step 1). The stuck-loop learning path imports a
  nonexistent `modules.memory.task_learning`; the real sink is `modules/memory/tool_outcome_capture.py`.
  Repoint it so the loop's outcome capture is not silently broken.

### 2.3 Why now

W1 has **no dependencies** and is startable in a single afternoon. Until it lands, PRD-161/163/164 deliver
into a dead engine, every enterprise bar that depends on execution is untestable, and every later wave builds
on a spine that does not run. This supersedes the prior review's "spine is done" claim (review §13
reconciliation: WS-6/PRD-161 "delivers into a dead engine until Wave 1").

---

## 3. Findings in scope

| ID | Severity | Location (pinned `37fdecc4e`) | Defect | Fix |
|---|---|---|---|---|
| **F001** | Critical | `orchestrator/modules/agents/factory/agent_factory.py:1070` (→ `tool_loop.py:154`) | Passes renamed-away kwarg `content_truncate_chars`; every non-chat run `TypeError`s and returns `status:error` | Rename to `content_truncate_tokens=0`; add a test that builds the executor **through** `execute_with_prompt` |
| **F023** | High | `orchestrator/api/board_tasks.py:900-921` | `_launch_task_execution` ignores the `status:error` result; failed runs close as `done` | Branch on `exec_result["status"]`; mark `failed` with the error text (as missions already do) |
| **F024** | High | `orchestrator/services/board_dispatcher.py:176-205` | No lease renewal; runs > 600s lease are swept back to `assigned` and re-claimed (duplicate execution) | Renew the lease mid-run on heartbeat |
| **F025** | High | `orchestrator/api/board_tasks.py:550-555` | `PATCH status→in_progress` fires launch for any non-recipe source, double-executing mission mirrors | Exclude orchestration-mirror rows from the drag-to-execute path |
| **(§9.5-1)** | — | `orchestrator/modules/tools/execution/tool_loop.py:55` | Imports nonexistent `modules.memory.task_learning` | Repoint the stuck-loop learning sink to `modules/memory/tool_outcome_capture.py` |

---

## 4. Changes (minimal diff, per finding)

**4.1 F001 — kwarg rename.** At `agent_factory.py:1070`, change `content_truncate_chars=0` to
`content_truncate_tokens=0` to match the `ToolLoopExecutor` constructor (`tool_loop.py:154`). No shim, no
backward-compat alias (CLAUDE.md §4). Verify no other caller passes the old name (`grep -rn
content_truncate_chars orchestrator/`).

**4.2 F023 — inspect status before closing.** In `_launch_task_execution` (`board_tasks.py:900-921`), read
`exec_result["status"]` (or the canonical result key confirmed on `main`); on error, transition the task to
`failed` and persist the error text, mirroring the mission path exactly. Do not close as `done` on a
non-success result.

**4.3 F024 — lease renewal on heartbeat.** In the dispatcher (`board_dispatcher.py:176-205`), extend the
task lease each time the run heartbeats, so a legitimately long run is not swept back to `assigned`. Keep the
sweep for genuinely dead runs (no heartbeat within the window).

**4.4 F025 — exclude orchestration mirrors from drag.** In the `PATCH status→in_progress` path
(`board_tasks.py:550-555`), do not fire launch for orchestration-mirror rows (mission-owned tasks). Gate
launch on the task's source so only user-owned board tasks execute on drag.

**4.5 §9.5 step 1 — repoint the learning sink.** At `tool_loop.py:55`, replace the import of the nonexistent
`modules.memory.task_learning` with `modules/memory/tool_outcome_capture.py`. Confirm the outcome-capture
call sites still type-check against the real module's signature.

> **Scope discipline (CLAUDE.md §12):** Loop *convergence* (routing recipe `_execute_step` and heartbeat onto
> `ToolLoopExecutor`) is **sequenced into the W4 migration order** (review §9.5 step 3), not deferred by this
> PRD — W1 repairs the existing spine; it does not re-plumb the recipe/heartbeat loops. That boundary is the
> review's dependency order, made explicit in the roadmap, not a silent descope. If review re-confirmation on
> `main` shows F001's blast radius reaches a path this PRD doesn't name, that path is added here — not punted.

---

## 5. Test-first acceptance

Write these **failing first**, then implement to green:

1. **The gap test (F001, headline acceptance).** A test drives a tool-using turn **through**
   `execute_with_prompt` — *not* a direct `ToolLoopExecutor` construction — with a stubbed LLM, and asserts a
   **non-error** result. This is the exact gap that let the regression ship; it is the wave's definition of done.
2. **F023.** Given an execution result with `status:error`, the board task transitions to `failed` with the
   error text and is **not** marked `done`.
3. **F024.** A run that heartbeats past the 600s lease is **not** swept back to `assigned` / re-claimed
   (exactly-once holds under lease expiry).
4. **F025.** A drag/`PATCH status→in_progress` on an orchestration-mirror row does **not** fire a second
   execution.
5. **§9.5-1.** The stuck-loop learning path imports and calls `tool_outcome_capture` without ImportError, and
   an outcome is recorded.

**Wave-level bar:** with F001+F023 fixed, a board task submitted through the real dispatch path runs the loop
and reports its true terminal status (`done` on success, `failed` on error) — "Auto operates the platform
end-to-end" is demonstrably true for the board surface.

---

## 6. Risks & rollback

- **Low blast radius.** F001 is a one-line rename; F023/F025 are narrow conditionals; F024 adds a renewal call.
  None touch the chat path.
- **F024 correctness risk:** over-generous renewal could keep a genuinely dead run leased. Mitigate by renewing
  only on a real heartbeat signal and keeping the dead-run sweep for the no-heartbeat case.
- **Rollback:** each finding is an independent commit; revert individually. The gap test (5.1) must stay green
  as a permanent regression guard regardless of the other fixes.

---

## 7. References

- Review §4 — F001 (Critical), F023/F024/F025 (High): `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
- Review §13 Wave 1 (acceptance: drive a turn *through* `execute_with_prompt`)
- Review §9.5 step 1 (resurrect headless execution; repoint the learning sink)
- Review §13 reconciliation (supersedes WS-6/PRD-161 "spine is done")
- CLAUDE.md §4 (no backward-compat shims), §5 (delete what you supersede), §12 (no unilateral descope)
