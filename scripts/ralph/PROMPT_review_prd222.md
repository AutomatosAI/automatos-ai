# Ralph Review Prompt — PRD-222 Auto-Led Onboarding (Mission Zero v2), Wave 1

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-222 Wave 1 is complete. Your job: find where the trial ledger doesn't actually bite, where BYOK traffic accidentally routes through the trial path, where JSONB was mutated in place, where a "guard test" wouldn't catch the thing it guards, or where scope crept into W2 territory. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-222.json` (`description` + `acceptanceCriteria` = binding contract). Full spec (seeded, reference): `docs/PRDS/PRD-222-AUTO-LED-ONBOARDING.md`.

**Scope guard (important):** Wave 1 builds the spine + trial + instrumentation. If the diff deletes Shepherd files, retires the wizard UI, touches `seed_onboarding_agents.py` or the `source=="mission_zero"` coordinator branches, adds plan/tier code, adds a second migration, or changes sign-up/waitlist — that is a finding (W2/W3/Gerard's call).

## Hunt list — every item is a confirmed-risk class

1. **The trial must BITE (CRITICAL class).** Prove by reading the tests AND the seam:
   - BYOK bypass: a workspace with its own working key never enters `resolve_trial_routing` accounting — if trial spend could accrue against a BYOK workspace, CRITICAL.
   - Allowlist: a trial request naming an off-list model must not reach the provider with that model (substitution or typed error — whichever the story chose, the test must exercise it).
   - Hard stop: at ≥100% the typed `trial_exhausted` error is returned; no code path lets an exhausted trial keep generating.
   - Background burn: heartbeat/scheduled execution provably skips non-converted trial workspaces (find the actual skip site, not just a test asserting a helper) — an idle trial workspace burning platform money is a CRITICAL.
   - Grant: one per Clerk user; global daily cap pauses grants (check the counter actually increments in the request path).
2. **JSONB mutate-in-place (the PRD-220 bug class).** Grep every write to `onboarding` for in-place dict mutation (`ws.onboarding[...] = …`, `.update(`, `append` on nested values) without whole-dict reassignment. Any instance = HIGH.
3. **ONE migration.** `git diff --name-only --diff-filter=A $BASE..HEAD -- orchestrator/alembic/versions/` must list exactly one file; its `down_revision` must be the pre-branch head; the static single-head test must exist and pass.
4. **Trust guards are real.** The US-010 static guard must actually scan the onboarding-owned file list (not a stale hardcoded list missing new files from this very diff); the awaiting_approval pin test must fail if someone flips the default. `skip_verification`/`auto_approve` set anywhere reachable from onboarding = CRITICAL.
5. **Schema truth pass genuinely bites.** The US-011 walker must fail when a handler-defaulted field sits in `required[]` — verify its self-check proves that. `platform_submit_report` required[] is exactly `['title','content']` and its handler still defaults the other two. No `source` added to `platform_create_mission`.
6. **Section v2 honesty.** Completed/skipped → `''` (test). The rendered largest variant fits the registered budget (if the registration was raised, it is ≤1200 with a measurement in a commit body). The OpenRouter copy sits in the `powerup` variant, not earlier (key-before-value would reverse the whole D4 decision — HIGH).
7. **Credential truth.** `test_status` persisted from the live call result; the trial→converted hook fires only on a VALID save. Any path that writes a healthy status without a real validation = HIGH (this is the 2026-07-29 incident class).
8. **Frontend surfaces.** `FirstLoginGuard` unmounted but files not deleted; opener renders only for `not_started`; power-up card never logs/stores the raw key (grep console/log/localStorage in the new components); exhausted banner renders with no LLM dependency; wizard's own progress consumption untouched. `node_modules`/mass-add staging poison = CRITICAL.
9. **No `os.getenv` outside `config.py`** in the diff; capability report returns booleans only (no secret echo).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-222-auto-led-onboarding --workflow test.yml --limit 3`: a NEW failure vs base = finding (arbitrate new-vs-pre-existing honestly).
- Run `bash scripts/ralph/acceptance-prd222.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) `TRIAL_*` values are Railway config Gerard sets before the pilot (defaults: $5 / $25/day / planner-model allowlist); (b) the keyless-resolution ground truth the build documented (unbounded platform fallback or error) and what it means; (c) W2 (plans/exposure + retirement of Shepherd/wizard/4-agent machinery) is gated on Gerard's Q1 tier decision.
- **Findings** → append `P222-RVW-1..n` fix stories to `scripts/ralph/prd-222.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-222): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-222-auto-led-onboarding` (never force, never another ref). Do not re-run the build.
