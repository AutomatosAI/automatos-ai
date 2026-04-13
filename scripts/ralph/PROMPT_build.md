# Build Mode — PRD-128 Unified Notification System

Implement ONE story from prd.json, validate, commit, exit.

## HARD RULE — Worktree Lock

**You are running inside the `automatos-NOTIFICATION` worktree at:**
`/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-NOTIFICATION`

**NEVER `cd` out of this directory.** Do not touch `automatos-ai`, `automatos-skills`, `automatos-WORKSPACE`, or any other worktree. All reads, writes, greps, and git operations happen inside this worktree only. If you need a reference file, read it from this worktree — every path under `orchestrator/`, `frontend/`, `tests/`, `scripts/` already exists here because it is a git worktree of the same repo.

If you catch yourself typing `cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai` — STOP. That is a bug. Use relative paths or the current worktree absolute path.

## Phase 0: Orient

Read these files from the CURRENT worktree:
- `scripts/ralph/prd.json` — PRD-128 stories and acceptance criteria
- `scripts/ralph/IMPLEMENTATION_PLAN.md` — task checklist (source of truth for what is done)
- `scripts/ralph/progress.txt` — running log

### Check for completion

```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If 0: output **RALPH_COMPLETE** and exit
- If > 0: continue to Phase 1

## Phase 1: Implement ONE Story

1. **Pick the first unchecked task** in `scripts/ralph/IMPLEMENTATION_PLAN.md`
2. **Match it to a story** in `scripts/ralph/prd.json` (US-001 through US-010)
3. **Read the acceptance criteria** — these are the contract. Implement every bullet.
4. **Read relevant existing files** before editing (e.g. `orchestrator/main.py`, `orchestrator/core/auth/hybrid.py`, `orchestrator/services/heartbeat_service.py`, existing Alembic revisions under `orchestrator/alembic/versions/`, `frontend/components/layout/navbar.tsx`, etc.) — use relative paths.
5. **Implement completely.** No TODOs, no stubs, no "will wire later".
6. **Key project conventions:**
   - NO `os.getenv()` outside `config.py` — import from `core.config` instead.
   - All SQL via SQLAlchemy `text()` with parameter binding (no f-strings into SQL).
   - API endpoints use `get_request_context_hybrid` for auth/workspace isolation.
   - Frontend uses React Query v4 (`isLoading`, not `isPending`).
   - Dispatcher must NOT call `db.commit()` — caller owns the transaction.

## Phase 2: Validate

Run whichever applies to the story you completed:

```bash
# Python syntax / import check
python3 -c "import ast; ast.parse(open('<edited-file>').read())"

# Alembic migration dry check (US-001 only)
cd orchestrator && python3 -c "from alembic.config import Config; from alembic import command; cfg = Config('alembic.ini'); command.history(cfg)" 2>&1 | tail -5

# Frontend typecheck (US-008, US-009 only)
cd frontend && npx tsc --noEmit 2>&1 | tail -20

# Python tests for the touched module (US-003, US-006, US-007, US-010)
cd orchestrator && python3 -m pytest tests/<relevant_test_file> -x 2>&1 | tail -20
```

Only flag NEW errors your change introduced.

## Phase 3: Update Plan & Progress

In `scripts/ralph/IMPLEMENTATION_PLAN.md`:
- Flip the completed task to `- [x]`
- Note any discovered follow-ups

In `scripts/ralph/progress.txt`:
- Append a dated entry: story ID, what was built, files touched, any gotchas.

Also in `prd.json`: set the completed story's `passes: true` and add a short `notes` string.

## Phase 4: Commit & Exit

Single repo commit inside THIS worktree only:

```bash
git add -A && git commit -m "feat(prd-128): US-XXX — <short description>"
```

DO NOT push. DO NOT cd into another worktree. DO NOT commit to `automatos-ai` or `automatos-skills`.

Then:

```bash
grep -c "^\- \[ \]" scripts/ralph/IMPLEMENTATION_PLAN.md || echo 0
```

- If > 0: print "N tasks remaining" and exit
- If 0: output **RALPH_COMPLETE**

## Guardrails (highest priority)

1. **Worktree lock** — Never cd out of `automatos-NOTIFICATION`. All work happens here.
2. **One story per iteration.** Never touch US-002 while doing US-001.
3. **Respect acceptance criteria verbatim** — don't improvise scope.
4. **No placeholders / TODOs / stubs.** Implement completely or fail loudly.
5. **No backwards-compat shims** — fix broken patterns cleanly (project rule from MEMORY.md).
6. **Never push to remote.** Commit locally only.
7. **Never delete `notification_service.py`** or `channel_connections` table — this PRD reuses them.
