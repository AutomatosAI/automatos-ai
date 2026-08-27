# Ralph Build Prompt — PRD-226 The Manager's Doctrine (Auto-as-Manager wave, chain 3/6)

You are executing **PRD-226**, one story per iteration, unattended. Branch **`ralph/prd-226-manager-doctrine` ← `ralph/prd-224-ticket-lane`** (STACKED, chain 3/6 — 227's wiring and 224's ASSIGN lane + ticket watches are IN your tree; the doctrine you write must describe how 224 actually behaves; three later branches stack on YOUR tip). The tip must be green after every commit.

**CONTEXT.** This PRD is content and prompt structure, near-zero mechanism: the 9-point management doctrine into Auto's seeded soul + skill (with a hash-guarded backfill), the assessment rubric extension, and the shared 4-part dispatch-contract fragment (planner emits, verification consumes, ASSIGN shares). The doctrine text itself is specified in the prd.json description — write it well, in the existing CTO voice.

## Read first, every iteration

1. `scripts/ralph/prd-226.json` — story list + THE DOCTRINE content spec; `description` + `acceptanceCriteria` = the BINDING contract.
2. Spec (seeded): `docs/PRDS/PRD-226-AUTO-MANAGER-DOCTRINE.md` §4-5; wave map `docs/PRDS/PRD-WAVE-AUTO-MANAGER.md`.
3. `CLAUDE.md` (repo root) — personas live in the DB, seed files are the source; no shims; canonical terms (Playbook / Mission / Task / Deliverable / Auto is a proper noun — the doctrine text must use them).

## The execution contract

- **RE-VERIFY every anchor by grep**: `seed_auto_agent.py:29-84` and its existing update/sync mechanism (this determines US-001's backfill shape — investigate BEFORE writing), the 224 rubric block in `auto.py`, `planner.py:756/:1099`, `verification.py`'s scoring entry.
- **Baked decisions are binding** (Gerard 2026-08-27): CTO voice kept — doctrine added as sections, not a rewrite; no per-workspace narration dial this wave; backfill only replaces hash-matching (uncustomized) souls, skips + reports the rest.
- **Single source for shared prompt fragments.** The 4-part contract fragment is defined ONCE and imported by the planner builder and the ASSIGN directive — a copy-paste is a hard failure. Same for the rubric: extend 224's block in place.
- **DB-not-files:** the backfill writes rows through the seed path's own mechanism. If that mechanism requires a data migration to do this, STOP — reply `RALPH_BLOCKED` (a migration on this branch is Gerard's call, not yours; PRD-225 owns the wave's only planned revision).
- **Eval discipline:** gold-set fixtures are LOCAL-ONLY (public repo). Eval harness additions load from the gitignored local path and SKIP cleanly when fixtures are absent — CI must stay green without them.
- **Verification back-compat is sacred:** with no `definition_of_done` present, `verification.py` behavior must be regression-identical.
- **PURE tests**, LLM-free (string-presence and parse fixtures, never live model calls). **Green tip:** `cd orchestrator && python3 -m pytest -q` after every commit; never commit on red.
- **STAGING DISCIPLINE.** Stage explicit paths only. **NEVER `git add -A`/`.`/`-u`**; **never `git stash -u`.**

## Hard NOs

- NO persona rewrite — additions in the existing voice only.
- NO duplicated prompt fragments or rubric blocks (1 definition site each).
- NO alembic files (declare `RALPH_BLOCKED` instead if the backfill truly needs one), NO new tables/routes, NO schema change for `definition_of_done` (it lives inside the existing plan JSONB, rebuild-don't-mutate).
- NO committed eval fixtures; NO weakening any existing test to make doctrine changes pass.
- NO `os.getenv` outside `config.py`; NO `git add -A`/`.`/`-u`; NO `git stash -u`.
- PUSH after each story commit to `origin ralph/prd-226-manager-doctrine` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify anchors fresh (especially seed_auto_agent's update mechanism before US-001).
2. Implement → `cd orchestrator && python3 -m pytest -q`.
3. Commit `feat(prd-226): <US-id> — <title>` with evidence; mark AC lines `DONE — <evidence>` in `scripts/ralph/prd-226.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd226.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
