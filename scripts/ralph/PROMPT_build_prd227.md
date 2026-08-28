# Ralph Build Prompt — PRD-227 Board Light-Up (Auto-as-Manager wave, chain 1/6)

You are executing **PRD-227**, one story per iteration, unattended. Branch **`ralph/prd-227-board-light-up`, cut from `origin/main`** (chain position 1/6 — the five later wave PRDs stack on YOUR tip, so a broken tip poisons the whole chain). The tip must be green after every commit.

**CONTEXT.** This PRD is pure wiring: agent activity on the board must be as live as human activity, missions must narrate into the thread that launched them, and the bell must navigate every link type the backend writes. The infrastructure (SSE lane, PRD-205 messenger, bell) all exists — you are adding producers and cases, never new mechanisms.

## Read first, every iteration

1. `scripts/ralph/prd-227.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract. Pick the first story with un-DONE ACs.
2. Spec (seeded on this branch): `docs/PRDS/PRD-227-BOARD-LIGHT-UP.md`; wave map `docs/PRDS/PRD-WAVE-AUTO-MANAGER.md`.
3. `CLAUDE.md` (repo root) — reuse over build; no shims; no `os.getenv` outside `config.py`; canonical terms (Mission / Task / Auto is a proper noun).

## The execution contract

- **RE-VERIFY every anchor by grep before relying on it** (verified 2026-08-27, they drift): `handlers_board_tasks.py:258-320` (+allowed statuses ~:270), `api/board_tasks.py:912-916` payload shape, `board_events.py:38-70` fail-soft pattern, `coordinator_service.py:3129` / `:2272`, `chat_messenger.py:96-202`, `notification-bell.tsx:42-64`, `command-center-shell.tsx:77-78` tab params.
- **Reuse the seams.** Narration goes through `deliver_background_message` — never a parallel send path, never a direct `ChatService` call. SSE stays `board_changed`/`chat_changed` — NO new event names.
- **Fail-soft is load-bearing.** A NOTIFY or narration failure must never fail the tool call or the coordinator tick — clone the existing try/log pattern, and test it with a monkeypatched raise.
- **JSONB is rebuild-don't-mutate** (PRD-220 class) if you persist `origin_chat_id` on the run's config.
- **ZERO new alembic revisions, tables, or routes** on this branch — the acceptance gate enforces it.
- **PURE tests.** DB-bound `@integration` tests skip cleanly without local Postgres — real-Postgres coverage is CI `test.yml` per-story push. Frontend: `cd frontend && npm run -s test` green after US-003.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` green after every commit. Never commit on red.
- **STAGING DISCIPLINE (critical).** Stage only specific paths. **NEVER `git add -A`/`.`/`-u`** — `node_modules/` is untracked and NOT gitignored; a blind add poisons the branch and every stacked branch after it. **Never `git stash -u`.**

## Hard NOs

- NO new alembic files, tables, routes, or SSE event names.
- NO parallel narration path bypassing `deliver_background_message`.
- NO `os.getenv` outside `config.py` (`MISSION_NARRATION_TASK_CAP` lives in config.py).
- NO touching mission execution semantics — you add producers at transition sites, you do not change transitions.
- NO `git add -A`/`.`/`-u`; NO `git stash -u`; NO staging `node_modules`.
- PUSH after each story commit to `origin ralph/prd-227-board-light-up` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh (grep, read the current code).
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ `cd frontend && npm run -s test` for US-003).
3. Commit `feat(prd-227): <US-id> — <title>` with evidence in the body (grep-proofs); mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-227.json` in the same commit; push the branch.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd227.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
