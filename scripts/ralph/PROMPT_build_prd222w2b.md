# Ralph Build Prompt — PRD-222 Wave 2b (tiers v1 · exposure · plan recommendation)

You are executing **PRD-222 Wave 2b**, one story per iteration, unattended. Branch **`ralph/prd-222-w2b` ← `origin/main`** (post-#630: the retirement is merged — Shepherd/wizard/4-agent machinery no longer exist; do not reference them). Tip green after every commit.

**CONTEXT.** Q1 is ANSWERED. `docs/PRDS/PRD-222-Q1-TIER-STRAWMAN.md` (seeded on this branch) is the **approved v1 numbers contract**: Basic $19 / Pro $49 / Business $99 (display-only — NO billing code, Q5), enterprise = coming-soon label only, every number config-driven because Gerard tunes tiers live while testing. Wave 1 (#613+#628) and Wave 2a (#630) are on main and load-bearing.

## Read first, every iteration

1. `scripts/ralph/prd-222w2b.json` — `description` + `acceptanceCriteria` = the BINDING contract. First story with un-DONE ACs.
2. The strawman (numbers contract) + PRD §6 W2·S1/S2 + §5.2 (proposal stage) — intent.
3. `CLAUDE.md` — reuse over build; canonical terms; no shims.

## The execution contract

- **RE-VERIFY anchors by grep** (kit-time notes, they drift): the v2 section's proposal block (`sections/onboarding.py` ~:129), `api/workspaces.py` current-workspace serialization (US-002 pattern), `marketplace-grid.tsx`, the plan column default (`workspaces.py` model :32), `plan_limits` consumers (budget `config.py:794`, max_agents). The nav component and the tool-surface assembly seam were NOT pinned at kit time — FIND them first (grep nav link definitions; grep the discovery module for where per-turn platform-tool schemas are gathered) and cite both in the commit body.
- **Exactly ONE migration** (US-023 rename+backfill). **Hidden ≠ deleted** (D5): exposure never deletes routes or data. **State changes through `platform_update_onboarding` only** (FR-4).
- **Measure the tool-surface trim** for a basic workspace — before/after token counts in the US-024 commit body. Families live in config, not hardcoded in the filter.
- **Load-bearing surfaces** (breaking any = hard failure): opener, power-up card, trial pill, exhausted banner, intake card, connect card, checklist card, `reset_onboarding` + `/dev/reset-onboarding`, the v2 section, `test_prd222_trust_guards.py`.
- JSONB rebuild-don't-mutate; no `os.getenv` outside `config.py`; PURE tests locally (real-Postgres in CI per-story push); frontend vitest + build green after every frontend story.
- **STAGING DISCIPLINE:** explicit paths only; NEVER `git add -A`/`.`/`-u`; never `git stash -u`.

## Hard NOs

- NO billing/checkout/payment/Stripe code (grep-guarded). NO enterprise code path (label only). NO quota-enforcement hardening beyond existing consumers.
- NO second migration; NO new tables; NO new router file.
- NO weakening trust guards or the stage validator; NO touching Wave-3 territory (academy/voice/invitee/partner).
- PUSH each story commit to `origin ralph/prd-222-w2b` ONLY. No PRs mid-run, no merges.

## Per-iteration protocol

1. Pick the first un-DONE story; re-verify anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ frontend vitest/build for frontend stories).
3. Commit `feat(prd-222): <US-id> — <title>` with evidence (grep-proofs, measured token counts); mark ACs `DONE — <evidence>` in `scripts/ralph/prd-222w2b.json` same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd222w2b.sh`. Exit 0 → reply `RALPH_COMPLETE` (final line, alone).
- Hard-NO conflict → `RALPH_BLOCKED` (final line) + one-line why + grep evidence in the last commit.
