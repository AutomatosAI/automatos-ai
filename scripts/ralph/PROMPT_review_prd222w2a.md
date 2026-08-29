# Ralph Review Prompt — PRD-222 Wave 2a

You are a fresh-context **adversarial reviewer**. The build claims Wave 2a (US-019..US-022) is complete: Composio connect card, checklist card, the full retirement of Shepherd/wizard/4-agent machinery, and the `is_new_workspace` migration. Your job: find where a deletion took a survivor with it, where an orphan import waits for a browser to discover it, where the migration over-deletes, or where checklist "detection" is actually a manual tick. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Contract: `scripts/ralph/prd-222w2a.json`. Spec: PRD §6 W2·S3..S6 + §10 inventory. Wave 1 (#613 + #628) is on main — it is the load-bearing floor this wave stands on.

## Hunt list

1. **Collateral deletion (CRITICAL class).** The retirement must take ONLY §10's inventory. Verify each Wave-1 surface exists at HEAD with its tests green: `onboarding-opener`, `power-up-card`, `trial-balance-pill`, `trial-exhausted-banner`, `intake-progress-card` (now on `use-intake-progress`), the v2 section, `reset_onboarding` + `/dev/reset-onboarding`, `test_prd222_trust_guards.py`. Backend intake (`modules/intake/**`, `api/wizard.py` pipeline endpoints, `business_profiles`, SSE backbone) must survive. Any of these gone or broken = CRITICAL.
2. **Orphan imports / dropped hunks (the LX-wave class).** For every deleted file, grep for lingering importers. `npm run -s build` must pass — vitest alone does not exercise all import graphs. `data-tour` = 0; `shepherd` deps gone from package.json AND the lockfile; grep-clean claims in commit bodies re-verified yourself.
3. **The migration must not over-delete (CRITICAL).** Read the cleanup migration: it may only touch seeded `required_role='onboarding'` agents; it must be idempotent; a non-seeded agent fixture must survive it in a test. `down_revision` chains to the current single head. EXACTLY ONE new alembic file in the diff.
4. **Coordinator surgery.** `_clone_onboarding_agents`, `_cleanup_ephemeral_agents`, `source=="mission_zero"` branches, roster injection — gone without breaking non-onboarding mission execution (the mission path's tests still green). The `source` field itself survives (only its mission_zero consumption dies).
5. **Checklist honesty.** Item completion derived from real counts (connections/missions/members) with tests; the Academy item is the only manual-dismiss and says so; single-seat plans never see the invite item; state in `workspaces.onboarding.checklist` via full-reassignment writes (in-place JSONB mutation = HIGH); zero localStorage.
6. **Connect card reuses the existing OAuth flow.** No new OAuth/redirect endpoint in the diff (route manifest gains nothing from US-019); the funnel event fires exactly once per workspace on 0→1 (not on popup close, not repeatedly).
7. **`is_new_workspace` fully gone** from code dirs with contract tests updated; the external-consumer verification (widget-sdk/shopify zero hits) documented in a commit body.
8. **Manifests + conventions.** Route manifest hand-edited (count bumped DOWN, stale rows preserved); `router_manifest.py` entry for onboarding-agents removed; no `os.getenv` outside config.py; trust guards + `advance_onboarding_stage` validator byte-untouched; staging clean (no node_modules, lockfile change scoped to the dep removal).

## Verification

- Run the **code-review** skill (or code-reviewer agent) on the diff — CRITICAL/HIGH findings are findings.
- `gh run list --branch ralph/prd-222-w2a --workflow test.yml --limit 3` — a NEW failure vs base = finding.
- `bash scripts/ralph/acceptance-prd222w2a.sh` — non-zero = automatic CRITICAL.

## Verdict protocol

**Put the sentinel on the FINAL line of your reply, alone.**

- No CRITICAL/HIGH/MEDIUM → 5-line summary noting: (a) W2·S1/S2 remain Q1-gated (untouched here); (b) the cleanup migration runs on deploy — seeded onboarding agents disappear from live workspaces, which is the intended user-visible change; (c) Wave-1 surfaces verified intact. Then the final line: `REVIEW_PASS`
- Findings → append `P222W2A-RVW-1..n` fix stories to `scripts/ralph/prd-222w2a.json` (file:line evidence, mechanical ACs), commit `chore(prd-222): review findings → fix stories`, push. Then the final line: `REVIEW_FINDINGS`
- Fix nothing. Never force-push, never another ref.
