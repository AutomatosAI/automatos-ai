# PRD-222 Wave 1b — acceptance-gate notes (for Gerard)

**Status: all 6 stories (US-011…US-016) are functionally complete and every CI check is green on the branch tip.**

Tip commit: `84fdea724` — CI `test` run **success**, including the **required** `orchestrator-tests`
gate, the `Frontend CI … route-contract` job, the `Orchestrator module + integration tests (F056)`
real-Postgres @integration reset tests, and `Alembic from-zero — exactly one head` (zero new migrations).

`bash scripts/ralph/acceptance-prd222b.sh` does **not** exit 0, but **none** of the three failing
checks is a defect in Wave-1b code. Each needs a call/action from you:

## 1. `.env.example documents the temporary flag` — hook-blocked (your action)

`grep -q ONBOARDING_RESET_ENABLED .env.example` fails: the file has no such line. The repo's
`block-secrets` PreToolUse hook refuses every `.env*` edit (glob `*.env.*` matches `.env.example`),
and the global rule is *do not work around it*. The flag is documented instead in
`orchestrator/config.py:1315-1321` (TEMPORARY / DEV-ONLY). To close the check, paste this into
`.env.example` yourself (outside Claude Code):

```
# TEMPORARY (PRD-222 US-016 / D9): enable the in-place dev onboarding-reset endpoint.
# Off in every real environment. Remove when the single-account test loop is retired.
ONBOARDING_RESET_ENABLED=false
```

## 2. `ls tests/test_prd222_trust_guards.py` — the guard is missing from `main` (your call)

The PRD-222 trust guard — the static scan asserting **no onboarding-owned source file sets
`skip_verification` or `auto_approve` truthy** — is **not on `main` or HEAD** under any name. It
existed pre-squash as `test_prd222_trust_guards.py` (branch commit `82421cb5f`, US-010), but the
**PR #613 squash-merge (`c0dd5a765`) did not land it** — the file that arrived,
`test_prd222_credential_validation.py`, does not carry the `_ONBOARDING_OWNED` scan.

- The **invariant itself is intact**: `grep -rn 'skip_verification\|auto_approve'` over the
  onboarding-owned source (`onboarding_state.py`, `trial_ledger.py`, `workspace_purge.py`,
  `api/workspaces.py` reset endpoint, `sections/onboarding.py`, intake/onboarding tool modules)
  finds **no truthy set** — only unrelated `auto_approve` schema/handler fields in the general
  board-task tool, which is not onboarding-owned. The `advance_onboarding_stage`
  monotonic/terminal validator (`InvalidStageTransition`) is also present.
- I did **not** re-create the guard: this branch's prompt says *"trust guards … all EXIST on main;
  build ON them, **never re-create or fork them**."* Re-creating it would violate that instruction,
  so the fix (restore the static guard on `main`, extended to cover US-016's reset/wipe surface) is
  yours to make in a Wave-1 remediation, not this loop's.

## 3. `orchestrator-full-suite` (local only) — pre-existing env baseline

Run locally, the suite shows the 3 documented env-baseline failures (Clerk-unset SaaS boot,
`pg_dump` 14 vs server 18, Composio webhook) that reproduce on `origin/main`. CI (proper env) is
green — see the tip run above.

## What this iteration changed

Fixed a route-manifest regression introduced by an earlier US-016 iteration: it ran a full
`python3 -m scripts.dump_routes` regen in a divergent env, which **dropped** the PRD-172-F006 stale
rows (`/api/workflows/execute`, `/api/workflows/{workflow_id}/execute`) that
`STALE_MANIFEST_ENTRIES` intentionally keeps and the frontend route-contract baseline still
references — breaking the (non-required) Frontend CI route-contract job. Reverted to the
PRD-mandated **hand-add**: `origin/main` manifest + the single reset route, `route_count` 767→768.
See commit `84fdea724`.

**Out-of-scope finding (flagged, not fixed):** `frontend/lib/api-client.ts` still calls the
PRD-172-deleted `/api/workflows/execute` + `/api/workflows/{workflow_id}/execute` endpoints. The
stale manifest rows mask these dead calls from the route-contract check. Cleaning both sides is the
frontend workflow-surface cleanup, not PRD-222.
