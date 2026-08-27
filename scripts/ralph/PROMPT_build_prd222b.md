# Ralph Build Prompt — PRD-222 Auto-Led Onboarding, Wave 1b (finish the spine)

You are executing **PRD-222 Wave 1b**, one story per iteration, unattended. This branch is **`ralph/prd-222-wave1b`, cut from `origin/main`** (standalone — NOT stacked). The tip must be green after every commit.

**CONTEXT — read this twice.** Wave 1's backend spine (US-001..US-010) is **ALREADY MERGED to main** (PR #613, 2026-07-31). `services/onboarding_state.py`, `services/trial_ledger.py`, `modules/context/sections/onboarding.py`, `platform_update_onboarding` + the intake tools, and the trust guards (`tests/test_prd222_trust_guards.py`) all EXIST on main. Build ON them. Never re-create, fork, or shadow them; never weaken the trust-guard test. The PRD's §3 "current reality" predates that merge — trust main, not §3.

**SCOPE.** 6 stories (`scripts/ralph/prd-222b.json`): the tool-schema truth pass (US-011), four frontend surfaces — modal unmount + opener (US-012), power-up card (US-013), trial pill + exhausted banner (US-014), intake progress card (US-015) — and the dev onboarding reset (US-016, PRD W1·S10/D9: the operator must be able to reset onboarding in ONE workspace and run it again with a single alias account).

## Read first, every iteration

1. `scripts/ralph/prd-222b.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract. Pick the first story with un-DONE ACs.
2. Full spec (seeded on this branch): `docs/PRDS/PRD-222-AUTO-LED-ONBOARDING.md` — §2 decisions (incl. D9) + §5 stage specs + W1·S10 are the intent; the JSON is the contract.
3. `CLAUDE.md` (repo root) — reuse over build; no shims; no `os.getenv` outside `config.py`; canonical terms (Playbook / Mission / Deliverable / Auto).

## The execution contract

- **RE-VERIFY every anchor by grep before relying on it** (verified 2026-08-27, they still drift): `actions_reports.py` required[] (~:99), the either-of error copies in `handlers_assignments.py` / `handlers_marketplace.py`, `providers.tsx` FirstLoginGuard mount (~:12, ~:103), `use-wizard-progress.ts`, `workspace_purge.py` internals, the `TRIAL_*` config block (~config.py:1309), `grant_trial_at_provisioning(db, workspace_id, *, owner_id)`.
- **ZERO new alembic revisions.** US-001 shipped the one migration. Every schema need on this branch lives inside the `workspaces.onboarding` JSONB. A new alembic file is a hard acceptance failure.
- **JSONB is rebuild-don't-mutate.** Assign a NEW dict/whole value on every write (match `_write_trial`'s `jsonb_set` style) — in-place mutation is the PRD-220 silent-loss bug class and the reviewer hunts for it.
- **US-016 trust rules:** `reset_onboarding` is the ONLY sanctioned backward writer of the onboarding doc — do NOT loosen `advance_onboarding_stage`'s monotonic/terminal validator. The wipe REUSES `workspace_purge` internals (parameterized to spare survivors; no duplicated table list) and must NOT inherit the purge's soft-delete precondition, its workspace-row DELETE, its users DELETE, or its Clerk-user deletion. Endpoint 404s when `ONBOARDING_RESET_ENABLED` is off; workspace-admin auth (same bucket as `api/onboarding_agents.py`'s `_require_admin`); committed route manifest hand-add + count bump.
- **US-013 key discipline:** the raw key value is posted to the existing credentials endpoint and appears in NO log statement and NO client-side store. Fixtures use obviously-fake key formats (gitleaks).
- **Out of scope — do not touch:** `frontend/lib/shepherd/**` (US-016 IMPORTS `tour-storage.ts` helpers — modifying the lib is still a violation), deletion of `first-login-guard.tsx`/`welcome-modal.tsx` (US-012 changes the MOUNT only), `frontend/components/wizard/**` behavior (US-015 shares `use-wizard-progress` logic without breaking the wizard's own consumption), `orchestrator/core/seeds/seed_onboarding_agents.py`, `_clone_onboarding_agents` / `source=="mission_zero"` branches, plans/tiers, waitlist/sign-up, academy repos. These are W2/W3 or Gerard's calls.
- **Trust rule (absolute):** no onboarding code path sets `skip_verification` or `auto_approve` — `tests/test_prd222_trust_guards.py` locks it and scans the onboarding-owned file list; if you add a new onboarding-owned file, ADD it to that list — never weaken the guard.
- **PURE tests.** DB-bound `@integration` tests must skip cleanly without a local Postgres — real-Postgres coverage is CI `test.yml` on each per-story push. Frontend: `cd frontend && npm run -s test` green after every frontend story.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` green after every commit. Never commit on red.
- **STAGING DISCIPLINE (critical).** Stage only specific paths. **NEVER `git add -A`/`.`/`-u`** — `node_modules/` is untracked and NOT gitignored; a blind add poisons the branch. **Never `git stash -u`** (drops graphify snapshots). Check a minimal `git status` before each commit.

## Hard NOs

- NO new alembic files; NO new tables; NO new router file (the reset endpoint lives on the existing workspaces router).
- NO `os.getenv` outside `config.py`.
- NO `source` added to `platform_create_mission`'s schema (W2 deletes the special-casing).
- NO weakening `tests/test_prd222_trust_guards.py` or `advance_onboarding_stage`'s validator.
- NO touching the out-of-scope list above; NO deleting anything Shepherd/wizard/onboarding-agents.
- NO `git add -A`/`.`/`-u`; NO `git stash -u`; NO staging `node_modules`.
- PUSH after each story commit to `origin ralph/prd-222-wave1b` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh (grep, read the current code).
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ `cd frontend && npm run -s test` for frontend stories).
3. Commit `feat(prd-222): <US-id> — <title>` with evidence in the body (grep-proofs, measurements); mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-222b.json` in the same commit; push the branch.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd222b.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
