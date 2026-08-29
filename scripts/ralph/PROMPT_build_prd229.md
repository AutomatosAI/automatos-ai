# Ralph Build Prompt — PRD-229 Mid-Run Clarifications (Auto-as-Manager wave, chain 6/6)

You are executing **PRD-229**, one story per iteration, unattended. Branch **`ralph/prd-229-clarifications` ← `ralph/prd-228-fleet-state`** (STACKED, chain 6/6, the last of the wave — 225's question-asks are your escalation target; 228's fleet state is answering context). The tip must be green after every commit.

**CONTEXT.** Today an executing agent that hits ambiguity can ask nobody anything — it guesses or fails. This PRD gives it one tool with a two-step ladder behind it: Auto answers routine questions from retrievable context (grounded, cited, budgeted), and only what Auto cannot answer — or any governance-category question — escalates into the ASK ME queue with the task parked and a draft preserved. It also deletes the dead collaboration machinery, per house rule §5.

## Read first, every iteration

1. `scripts/ralph/prd-229.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract (baked decisions: budget 3; draft-on-park; delete confirmed).
2. Spec (seeded): `docs/PRDS/PRD-229-MID-RUN-CLARIFICATIONS.md`; wave map `docs/PRDS/PRD-WAVE-AUTO-MANAGER.md`.
3. `CLAUDE.md` (repo root) — delete what you replace; reuse over build; canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep**: the task envelope (`coordinator_service.py:2237-2249` — your time-box must fit inside it, document the arithmetic), `modes.py`'s mode-scoped tool admission, 225's ask internals (share the function), the progress ledger, and — before US-004 — the zero-external-callers claim for `inter_agent.py` (ANY caller = `RALPH_BLOCKED`, report it).
- **Grounded-only is absolute.** Empty retrieval short-circuits BEFORE any LLM call; every answer carries source refs; the reviewer hunts invented answers. The LLM call is composition over retrieved context, nothing more.
- **The ladder is vertical.** Worker → orchestrator → human. NOTHING lateral — no agent↔agent messaging, ever.
- **Escalation reuses 225.** The shared service function `platform_ask_human` uses — no HTTP self-call, no parallel ask construction.
- **Draft-on-park:** partial output into the task's EXISTING result JSONB, rebuild-don't-mutate, labeled draft. No schema change.
- **Budget semantics:** `CLARIFICATION_BUDGET` (config.py, default 3) limits ANSWERS; escalations are never budget-limited.
- **US-004 is a real deletion:** file + re-exports + empty scaffolding + declare-the-world configs, same commit, zero orphan imports. Remember: import-linter-style configs break on the MERGED TIP — update them here, not later.
- **ZERO new alembic revisions, tables, or routes.**
- **PURE tests**, LLM-free (the answering LLM is stubbed; a zero-invocation assertion guards the short-circuit). **Green tip:** `cd orchestrator && python3 -m pytest -q` after every commit; never commit on red.
- **STAGING DISCIPLINE.** Explicit paths only. **NEVER `git add -A`/`.`/`-u`**; **never `git stash -u`.**

## Hard NOs

- NO lateral agent messaging; NO mailboxes; NO resurrection of anything from `inter_agent.py`.
- NO ungrounded answers; NO LLM call on empty retrieval; NO fabricated sources.
- NO parallel ask path; NO HTTP self-calls; NO `AWAITING_HUMAN` writers.
- NO new alembic files, tables, routes; NO pgvector touches (RAG is S3 Vectors ONLY).
- NO tool visibility outside TASK_EXECUTION context.
- NO `os.getenv` outside `config.py`; NO `git add -A`/`.`/`-u`; NO `git stash -u`.
- PUSH after each story commit to `origin ralph/prd-229-clarifications` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh (US-004 starts with the caller grep).
2. Implement → `cd orchestrator && python3 -m pytest -q`.
3. Commit `feat(prd-229): <US-id> — <title>` with evidence; mark AC lines `DONE — <evidence>` in `scripts/ralph/prd-229.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd229.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
