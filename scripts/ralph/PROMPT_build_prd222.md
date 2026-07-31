# Ralph Build Prompt — PRD-222 Auto-Led Onboarding (Mission Zero v2), Wave 1

You are executing **PRD-222 Wave 1**, one story per iteration, unattended. This branch is **`ralph/prd-222-auto-led-onboarding`, cut from `origin/main`** (standalone — NOT stacked). The tip must be green after every commit.

**SCOPE — read this twice.** 15 stories (`scripts/ralph/prd-222.json`): the server-side onboarding state machine + the ONE migration (US-001), the current-workspace surface (US-002), the `platform_update_onboarding` tool (US-003), the $5 trial ledger — config/grant (US-004) and enforcement at the LLM key-resolution choke point (US-005), credential-validation truth (US-006), the capability report (US-007), intake-pipeline-as-tools (US-008), OnboardingSection v2 (US-009), trust guards + summary Deliverable (US-010), the tool-schema truth pass (US-011), and four frontend surfaces (US-012..US-015).

**Why this PRD exists.** Onboarding today is three disconnected surfaces, has never run for a real user, and the first mission a customer would see runs unverified and auto-approved. Wave 1 makes Auto the only guide, funds the first mile with a hard-capped trial, moves the BYOK ask to after the value moment, and instruments every stage. A pilot cohort is waiting on this.

## Read first, every iteration

1. `scripts/ralph/prd-222.json` — story list; `description` + `acceptanceCriteria` = the BINDING contract. Pick the first story with un-DONE ACs.
2. Full spec (seeded on this branch): `docs/PRDS/PRD-222-AUTO-LED-ONBOARDING.md` — §2 decision record + §5 stage specs are the intent; the JSON is the contract.
3. `CLAUDE.md` (repo root) — reuse over build; no shims; no `os.getenv` outside `config.py`; canonical terms (Playbook / Mission / Deliverable / Auto).

## The execution contract

- **RE-VERIFY every anchor by grep before relying on it** (they drift): `hybrid.py` provisioning lines, `workspaces.py` response shape, `onboarding.py` section, the `PLATFORM_KEY_WORKSPACE_ID` resolution seam, the five schema-mismatch line numbers.
- **ONE migration total, on US-001.** Every later schema need lives inside the `workspaces.onboarding` JSONB. A second alembic file on this branch is a hard failure (the acceptance gate counts).
- **JSONB is rebuild-don't-mutate.** Assign a NEW dict to the column every write — in-place mutation is the PRD-220 silent-loss bug class and the reviewer hunts for it.
- **US-005 must not touch the BYOK path.** A workspace with its own working key bypasses the trial ledger entirely — prove it with a test. Document today's keyless resolution behavior (grep-proof) in the commit body BEFORE building the routing.
- **Out of scope — do not touch:** `frontend/lib/shepherd/**`, deletion of `first-login-guard.tsx`/`welcome-modal.tsx` (US-012 changes the MOUNT only), `frontend/components/wizard/**` behavior, `orchestrator/core/seeds/seed_onboarding_agents.py`, `_clone_onboarding_agents` / `source=="mission_zero"` branches in `coordinator_service.py`, plans/tiers, waitlist/sign-up, academy repos. These are W2/W3 or Gerard's calls.
- **Trust rule (absolute):** no onboarding code path sets `skip_verification` or `auto_approve` — US-010 locks it with a guard test; never weaken that test.
- **PURE tests.** DB-bound `@integration` tests must skip cleanly without a local Postgres — real-Postgres coverage is CI `test.yml` on each per-story push. Frontend: `cd frontend && npm run -s test` green after every frontend story.
- **Secrets discipline:** fixtures use obviously-fake key formats; no secret value in any file; capability report returns booleans only.
- **Green tip:** `cd orchestrator && python3 -m pytest -q` green after every commit. Never commit on red.
- **STAGING DISCIPLINE (critical).** Stage only specific paths. **NEVER `git add -A`/`.`/`-u`** — `node_modules/` is untracked and NOT gitignored; a blind add poisons the branch. **Never `git stash -u`** (drops graphify snapshots). Check a minimal `git status` before each commit.

## Hard NOs

- NO second migration; NO new tables; NO new pricing table (reuse the existing per-request cost computation).
- NO `os.getenv` outside `config.py`.
- NO `source` added to `platform_create_mission`'s schema (W2 deletes the special-casing).
- NO touching the out-of-scope list above; NO deleting anything Shepherd/wizard/onboarding-agents.
- NO `git add -A`/`.`/`-u`; NO `git stash -u`; NO staging `node_modules`.
- PUSH after each story commit to `origin ralph/prd-222-auto-led-onboarding` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh (grep, read the current code).
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ `cd frontend && npm run -s test` for frontend stories).
3. Commit `feat(prd-222): <US-id> — <title>` with evidence in the body (grep-proofs, measurements); mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-222.json` in the same commit; push the branch.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd222.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO (e.g. the usage/cost seam genuinely doesn't exist, the choke point is not a single seam) → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
