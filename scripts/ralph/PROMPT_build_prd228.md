# Ralph Build Prompt — PRD-228 Fleet State (Auto-as-Manager wave, chain 5/6)

You are executing **PRD-228**, one story per iteration, unattended. Branch **`ralph/prd-228-fleet-state` ← `ralph/prd-225-ask-me`** (STACKED, chain 5/6 — the whole wave so far is IN your tree: surface 224's ticket watches and 225's open asks in the read-model; one later branch stacks on YOUR tip). The tip must be green after every commit.

**CONTEXT.** "Auto, how's the team doing?" must get a grounded answer, and the Agents page must show what each agent is doing right now. Every input exists scattered (leases, running tasks, heartbeats, watches, asks, cost); this PRD is ONE read-model over them, one route, one tool, one view. It writes nothing.

## Read first, every iteration

1. `scripts/ralph/prd-228.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract (baked decisions: fleet tab in Agents page; cost = the canonical source the usage surfaces already read, PIN IT FIRST; rolling 24h).
2. Spec (seeded): `docs/PRDS/PRD-228-FLEET-STATE.md`; wave map `docs/PRDS/PRD-WAVE-AUTO-MANAGER.md`.
3. `CLAUDE.md` (repo root) — no duplicate hooks; reuse over build; canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep**: board lease fields (`core.py:1592-1597`), the matcher's busy derivation (`agent_matcher.py:459` — REUSE it, a rival "busy" definition is drift), heartbeat state access, the grants hot index + 225's `kind`, `agent-management.tsx:244-276` tab set, `modes.py:66-73` heartbeat tool loading, the board list's permission dependency.
- **Cost source is pinned, not guessed.** Grep what the existing credit/usage surfaces query; use THAT. Two genuine rivals → `RALPH_BLOCKED` naming both. Never a new store.
- **READ-ONLY is structural.** No session mutations in `fleet_state.py` — a grep test enforces it. Fail-soft per source (omit fields, never 500).
- **No N+1.** Bounded query set with a query-count assertion test.
- **Route procedure in full** (US-002): existing router preferred; RouterSpec if a new file; regenerate + COMMIT `route-manifest.json` with bumped count; `api-client.ts` parity; both contract tests green.
- **Frontend reuse:** extend the existing agents hooks; reuse the existing details modal; no V2-suffixed anything.
- **ZERO new alembic revisions, ZERO new tables.**
- **PURE tests** (`@integration` skips cleanly; real Postgres in CI per-story push). Frontend: `cd frontend && npm run -s test` green after US-004.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` after every commit; never commit on red.
- **STAGING DISCIPLINE.** Explicit paths only. **NEVER `git add -A`/`.`/`-u`**; **never `git stash -u`.**

## Hard NOs

- NO writes anywhere in the fleet service; NO new state semantics; NO rival "busy" derivation.
- NO new cost store or ad-hoc cost math beyond the pinned source.
- NO new alembic files, tables, modals, or duplicate hooks.
- NO SSE additions (the view polls + reuses the existing board-event CustomEvent).
- NO `os.getenv` outside `config.py`; NO `git add -A`/`.`/`-u`; NO `git stash -u`.
- PUSH after each story commit to `origin ralph/prd-228-fleet-state` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh (US-001 starts with the cost-source pin).
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ `cd frontend && npm run -s test` for US-004).
3. Commit `feat(prd-228): <US-id> — <title>` with evidence; mark AC lines `DONE — <evidence>` in `scripts/ralph/prd-228.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd228.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
