# Ralph Build Prompt — PRD-225 Agent Questions: ASK ME + Telegram (Auto-as-Manager wave, chain 4/6)

You are executing **PRD-225**, one story per iteration, unattended. Branch **`ralph/prd-225-ask-me` ← `ralph/prd-226-manager-doctrine`** (STACKED, chain 4/6 — 227 wiring, 224 ticket lane + watches, and 226 doctrine are IN your tree; two later branches stack on YOUR tip). The tip must be green after every commit.

**CONTEXT.** A question to the human is **task state, not a message**: it lives on a grant row against the subject it blocks, one tab aggregates every open ask with the cascade of downstream work stuck behind it, answering resumes the work through the machinery grants already have, and a Telegram reply answers from the phone. This is the wave's ONLY schema change and its most user-visible surface — precision over speed.

## Read first, every iteration

1. `scripts/ralph/prd-225.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract (baked decisions included). Pick the first story with un-DONE ACs.
2. Spec (seeded): `docs/PRDS/PRD-225-AGENT-QUESTIONS-ASK-ME.md`; wave map `docs/PRDS/PRD-WAVE-AUTO-MANAGER.md`.
3. `CLAUDE.md` (repo root) — no new tables (extending `approval_grants` IS the design); no duplicate hooks (US-004 consolidates the existing pair); canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep** — the prd.json lists ~20 with line numbers verified 2026-08-27; they drift. Especially: the resume machinery (`api/approval_grants.py:172-309`), the webhook reply-context capture (`api/webhooks.py:186-213`), the dispatcher's quiet-hours/urgent logic, `command-center-shell.tsx`'s tab plumbing, and whether the Telegram driver's `SendResult` already carries `message_id`.
- **THE MIGRATION IS SINGULAR.** Exactly one new alembic revision (US-001), `alembic heads` == 1 after it. US-006 does NOT get a second revision — channel `trigger_mode` rides the channels settings JSONB.
- **Reuse the resume machinery** — `_requeue_subject` / `_resume_tool_call` are called, never reimplemented. Approval-kind regression tests must pass WITHOUT modification (their files untouched in the diff).
- **Route discipline:** ONE new route (the answer endpoint) on the EXISTING approval-grants router. Regenerate + COMMIT the route manifest with the bumped count; add the `api-client.ts` call; `node frontend/scripts/check-route-contract.js` green. The questions list reuses the grants list route with a kind filter — no new list endpoint.
- **Park, never wait.** `platform_ask_human` returns immediately. No sleep, no polling, anywhere in the handler.
- **Trail is rows.** Re-asks are new rows against the same subject. No qa-history JSONB.
- **JSONB rebuild-don't-mutate** (`channel_refs`, `planning_data.human_qa`, channel settings) — PRD-220 class, the reviewer hunts it.
- **Dead vocabulary stays dead:** no `AWAITING_HUMAN` writers, no `TASK_HUMAN_*` emissions.
- **Secrets hygiene:** webhook/Telegram fixtures use obviously-fake tokens (gitleaks; public repo). Gate logs carry no message bodies.
- **PURE tests** (`@integration` skips cleanly; real Postgres in CI per-story push). Frontend: `cd frontend && npm run -s test` green after US-004.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` after every commit; never commit on red.
- **STAGING DISCIPLINE.** Explicit paths only. **NEVER `git add -A`/`.`/`-u`**; **never `git stash -u`.**

## Hard NOs

- NO second alembic revision; NO new tables; NO new router files; NO new list endpoint.
- NO reimplementation of resume/requeue; NO edits to approval-kind regression test files.
- NO `AWAITING_HUMAN`/`TASK_HUMAN_*` writers.
- NO HTTP self-calls (the Telegram answer path calls the shared service function).
- NO real-looking secrets in fixtures; NO message bodies in gate logs.
- NO `os.getenv` outside `config.py`; NO `git add -A`/`.`/`-u`; NO `git stash -u`.
- PUSH after each story commit to `origin ralph/prd-225-ask-me` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ `cd frontend && npm run -s test` for US-004).
3. Commit `feat(prd-225): <US-id> — <title>` with evidence; mark AC lines `DONE — <evidence>` in `scripts/ralph/prd-225.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd225.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
