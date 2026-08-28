# Ralph Build Prompt — PRD-222 Wave 2a (connect · checklist · retirement · gate migration)

You are executing **PRD-222 Wave 2a**, one story per iteration, unattended. This branch is **`ralph/prd-222-w2a`, cut from `origin/main`** with Wave 1b (#628) already merged. The tip must be green after every commit.

**CONTEXT.** Wave 1 is COMPLETE on main: the state machine + trial ledger (US-001..008, #613), and the v2 section, trust guards, frontend surfaces, and dev reset (#628). This wave is Wave 2's Q1-independent half, per Gerard's 2026-08-27 decisions: **wizard UI dies (Q3 = PRD default), retirement proceeds now (Q2)**. Exposure profiles and plan recommendation (W2·S1/S2) are a LATER kit gated on the Q1 tier decision — do NOT build or prepare for them.

## Read first, every iteration

1. `scripts/ralph/prd-222w2a.json` — `description` + `acceptanceCriteria` = the BINDING contract. Pick the first story with un-DONE ACs.
2. Spec (seeded): `docs/PRDS/PRD-222-AUTO-LED-ONBOARDING.md` — §6 W2·S3..S6 + **§10 deletion inventory** are the intent.
3. `CLAUDE.md` (repo root) — reuse over build; delete what you replace; no shims; canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep before relying on it** (verified 2026-08-27, they drift): coordinator lines (~:486/:566/:1580/:1682/:2464), main.py seed call (~:213), wizard.py re-seed (~:556/:636), the 19 `data-tour` files, package.json shepherd deps (~:119/:124), the Composio OAuth initiation route (NOT pinned at kit time — find what the Tools page connect button calls).
- **Deletion discipline (US-021).** Relocate the SSE hook BEFORE deleting the wizard tree (the intake card imports `use-wizard-progress` — move it to `use-intake-progress.ts` first). Delete each surface WITH its tests. After every deletion story: frontend build + vitest + orchestrator suite are your only link-checkers — browser ESM never validates imports (the LX-wave dropped-hunks lesson). End grep-CLEAN: `shepherd|use-auto-tour|mission_zero|OnboardingAgentsTab|seed_onboarding_agents` = zero hits in code dirs.
- **Wave-1 surfaces are load-bearing:** `onboarding-opener`, `power-up-card`, `trial-balance-pill`, `trial-exhausted-banner`, `intake-progress-card`, `reset_onboarding` + `/dev/reset-onboarding`, the v2 section, `test_prd222_trust_guards.py`. Deleting or breaking ANY of them is a hard failure — the acceptance gate checks each by name.
- **EXACTLY ONE new alembic revision** (US-021's seeded-agent cleanup): idempotent, no-op when rows absent, provably scoped — a fixture agent that is NOT a seeded onboarding agent must survive it.
- **Backend intake stays.** `orchestrator/modules/intake/**` and `api/wizard.py`'s pipeline endpoints are the tool substrate — only the lazy re-seed function dies there.
- **Route manifest is HAND-edited** with count bumps (down for deleted routes) — never regenerate; the committed file deliberately carries stale rows.
- **JSONB rebuild-don't-mutate** for every checklist write. No `os.getenv` outside `config.py`. Never weaken `test_prd222_trust_guards.py` or `advance_onboarding_stage`'s validator.
- **PURE tests** locally (@integration skips without Postgres; CI covers). Orchestrator baseline: the documented `prd172|dr_restore|composio` flakes only — no NEW failures. Frontend: strict green after every frontend story.
- **STAGING DISCIPLINE.** Stage explicit paths only — NEVER `git add -A`/`.`/`-u` (node_modules is untracked and NOT gitignored). The package.json + lockfile change for the shepherd dep removal is staged explicitly. Never `git stash -u`.

## Hard NOs

- NO plans/tiers/exposure code (W2·S1/S2 — Q1-gated, later kit). NO Wave-3 work.
- NO second migration; NO new tables; NO new OAuth/redirect endpoints (reuse the existing Composio flow).
- NO deleting: backend intake, `api/wizard.py` pipeline endpoints, `business_profiles`, any Wave-1 surface, the SSE progress backbone.
- NO localStorage for checklist state (D8 — server is the record).
- NO `git add -A`/`.`/`-u`; NO `git stash -u`.
- PUSH after each story commit to `origin ralph/prd-222-w2a` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ `cd frontend && npm run -s test` for frontend stories; + `npm run -s build` after deletion stories).
3. Commit `feat(prd-222): <US-id> — <title>` with evidence in the body (grep-proofs, before/after counts); mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-222w2a.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd222w2a.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` + one line why + grep evidence in the last commit.
