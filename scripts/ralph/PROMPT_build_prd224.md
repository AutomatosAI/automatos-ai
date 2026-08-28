# Ralph Build Prompt — PRD-224 The Ticket Lane (Auto-as-Manager wave, chain 2/6)

You are executing **PRD-224**, one story per iteration, unattended. Branch **`ralph/prd-224-ticket-lane` ← `ralph/prd-227-board-light-up`** (STACKED, chain 2/6 — 227's agent-move SSE + blocked/failed parity are already IN your tree; build on them, never re-create them; four later branches stack on YOUR tip). The tip must be green after every commit.

**CONTEXT.** Missions staff projects; this lane manages agents: "have my accountant agent do X" → a board ticket assigned to that named agent, started, supervised by a watch, verdict reported back in-thread. Every primitive exists (`platform_create_task` takes `assigned_agent_name`; `in_progress` triggers execution; watches score/narrate) — you are adding one enum value to watches, one Action to AutoBrain, and the wiring between them.

## Read first, every iteration

1. `scripts/ralph/prd-224.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract. Pick the first story with un-DONE ACs.
2. Spec (seeded): `docs/PRDS/PRD-224-AUTO-TICKET-LANE.md`; wave map `docs/PRDS/PRD-WAVE-AUTO-MANAGER.md` (baked decisions live there).
3. `CLAUDE.md` (repo root) — reuse over build; no shims; no `os.getenv` outside `config.py`; canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep before relying on it**: `actions_board_tasks.py:10-235`, `handlers_board_tasks.py:287-296` (and the PRD-227 notify calls now in that file), `auto.py:47-65` + `_ROSTER_LIMIT` (:40), `api/chat.py:391-432`, `watch_enums.py` / `watches.py:91,53,145`, `actions_watches.py:16-110`, `watch_decider.py`'s output-fetch + scorer, `watch_actions.py:65,543-551`, `api/board_tasks.py:830` run-now internals, `build_tool_caller_context` (`service.py:103`).
- **Baked decisions are binding** (Gerard 2026-08-27): immediate start unless defer-phrasing; ask-in-thread when no agent name resolves (never matcher auto-pick on this lane); `AUTO_TICKET_WATCH` default ON.
- **Reuse the supervision machine.** Ticket watches ride the existing decider/actions/ticker/notifications — a parallel supervision path is a hard failure. Scoring reuses the existing scorer. Escalation reuses the existing escalation service (PRD-225 upgrades it later — do NOT build question constructs here).
- **Dead vocabulary stays dead:** do not resurrect `RunState.AWAITING_HUMAN` / `TASK_HUMAN_*` — tickets block via the existing `blocked` status only.
- **ZERO new alembic revisions, tables, routes, or router files** — `watches.target_type` is a plain String(32); the acceptance gate enforces zero migrations.
- **US-005 goes in the handler, not the prompt** — supervision must attach mechanically (gated on the ASSIGN caller context), so the LLM cannot forget it.
- **PURE tests.** `@integration` skips cleanly without local Postgres; real-Postgres is CI `test.yml` per-story push. LLM-free tests only (assessment stubbed) — no live model calls in the suite.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` green after every commit. Never commit on red.
- **STAGING DISCIPLINE (critical).** Stage only specific paths. **NEVER `git add -A`/`.`/`-u`** (node_modules is untracked and NOT gitignored). **Never `git stash -u`.**

## Hard NOs

- NO new alembic files, tables, routes, router files, or SSE event names.
- NO parallel watch/supervision/escalation path; NO second scorer.
- NO question/ask constructs (that is PRD-225, chain 4/6).
- NO matcher auto-pick when the user named no agent on the ASSIGN lane — Auto asks.
- NO `os.getenv` outside `config.py`.
- NO changes to 227's diffs other than building on them.
- NO `git add -A`/`.`/`-u`; NO `git stash -u`; NO staging `node_modules`.
- PUSH after each story commit to `origin ralph/prd-224-ticket-lane` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q`.
3. Commit `feat(prd-224): <US-id> — <title>` with evidence in the body; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-224.json` in the same commit; push the branch.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd224.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
