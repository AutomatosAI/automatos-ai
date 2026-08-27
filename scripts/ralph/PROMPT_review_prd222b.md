# Ralph Review Prompt — PRD-222 Auto-Led Onboarding, Wave 1b

You are a fresh-context **adversarial reviewer**. The build loop claims PRD-222 Wave 1b (US-011..US-016) is complete. Your job: find where the schema walker wouldn't actually bite, where a frontend card leaks the raw key, where the dev reset is reachable in production or deletes a survivor, where JSONB was mutated in place, or where scope crept into W2 territory. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-222b.json` (`description` + `acceptanceCriteria` = binding contract). Full spec (seeded, reference): `docs/PRDS/PRD-222-AUTO-LED-ONBOARDING.md` (W1·S10 + D9 are the reset's intent). Wave 1's US-001..010 are ALREADY ON MAIN (PR #613) — anything in the diff that re-creates, forks, or weakens them is a finding.

**Scope guard (important):** if the diff deletes Shepherd files or MODIFIES `frontend/lib/shepherd/**` (importing from it is fine), retires the wizard UI or breaks `use-wizard-progress`'s wizard consumption, touches `seed_onboarding_agents.py` or the `source=="mission_zero"` branches, adds plan/tier code, adds ANY alembic file, adds a new router file, or changes sign-up/waitlist — that is a finding (W2/W3/Gerard's call).

## Hunt list — every item is a confirmed-risk class

1. **The reset must be UNREACHABLE in production (CRITICAL class).**
   - Endpoint responds with anything but 404 while `ONBOARDING_RESET_ENABLED` is off → CRITICAL. The flag must be read in `config.py` only.
   - Reachable by a non-admin workspace member when on → CRITICAL.
   - Any parameter lets it act on a workspace other than the caller's current one → CRITICAL (cross-workspace scoping test must exist AND the handler must actually scope every DELETE).
2. **The wipe must spare survivors (CRITICAL class).** Read the deletion path end-to-end: if it can delete the workspace row, `users` rows, the Clerk user, system agents, `required_role='onboarding'` agents, operator-workspace (`PLATFORM_KEY_WORKSPACE_ID`) rows, or credentials when `wipe_credentials` is off → CRITICAL. If it inherited `workspace_purge`'s soft-delete precondition or its Clerk-deletion step → CRITICAL. If it hand-duplicates the scoped-table list instead of reusing `workspace_purge` internals → HIGH (two lists WILL drift).
3. **The forward spine stays strict.** Any loosening of `advance_onboarding_stage`'s monotonic/terminal validator to enable resets → HIGH. `reset_onboarding` must be the only backward writer; the reset must rebuild the doc (new value, `jsonb_set`/whole-assignment) — in-place JSONB mutation anywhere in the diff = HIGH (PRD-220 class).
4. **Trial regrant honesty.** `reset_trial` must strip-then-regrant via `grant_trial_at_provisioning` (call-site grep) — a second grant implementation = HIGH. A kill-switch/cap decline must surface as a pause in the response, not an exception or a silent nothing.
5. **Schema truth pass genuinely bites (US-011).** The walker must fail when a handler-defaulted field sits in `required[]` — verify its self-check proves that. `platform_submit_report` required[] is exactly `['title','content']` and its handler still defaults the other two. No `source` added to `platform_create_mission`. The either-of descriptions carry the handlers' exact error copy.
6. **Frontend surfaces (US-012..015).** `FirstLoginGuard` unmounted but files not deleted; opener renders only for `not_started`; power-up card never logs/stores the raw key (grep console/log/localStorage/sessionStorage in the new components); exhausted banner renders with zero LLM/network dependency beyond the credentials POST; wizard's own progress consumption untouched (its tests still green). `node_modules`/mass-add staging poison = CRITICAL.
7. **Trust guards are real.** `tests/test_prd222_trust_guards.py` must still pass and must scan any NEW onboarding-owned file this diff adds (a stale file list that misses the reset code = finding). `skip_verification`/`auto_approve` set anywhere reachable from onboarding = CRITICAL.
8. **Manifest + conventions.** ZERO new alembic files (`git diff --name-only --diff-filter=A $BASE..HEAD -- orchestrator/alembic/versions/` is empty). The committed route manifest carries the reset route and its count is bumped. No `os.getenv` outside `config.py` in the diff. `.env.example` documents `ONBOARDING_RESET_ENABLED` as temporary. The dev page is unlinked (no nav/sidebar references).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on `git diff $BASE..HEAD` — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-222-wave1b --workflow test.yml --limit 3`: a NEW failure vs base = finding (arbitrate new-vs-pre-existing honestly).
- Run `bash scripts/ralph/acceptance-prd222b.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary. Note explicitly: (a) `ONBOARDING_RESET_ENABLED` is default-off — Gerard sets it in Railway ONLY for the test window and removes it after the pilot; (b) `TRIAL_*` Railway values are still Gerard's pre-pilot step; (c) W2 (plans/exposure + Shepherd/wizard/4-agent retirement) stays gated on the Q1 tier decision; (d) the old merged branch `ralph/prd-222-auto-led-onboarding` on origin is safe to delete.
- **Findings** → append `P222B-RVW-1..n` fix stories to `scripts/ralph/prd-222b.json` (title, `file:line` evidence, mechanical ACs, files). Commit `chore(prd-222): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only the fix-story commit to `ralph/prd-222-wave1b` (never force, never another ref). Do not re-run the build.
