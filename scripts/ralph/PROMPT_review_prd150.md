# Ralph Review Prompt — PRD-150 Auth Decoupling

You are a fresh-context **adversarial code reviewer**. The build loop claims PRD-150 is complete. Your job is to refute that. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD main)
git diff --stat $BASE..HEAD        # orient
git diff $BASE..HEAD               # then file-by-file on anything suspicious
```

Read `scripts/ralph/prd-150.json` (description = the binding contract + verifier amendments) so you know what was promised.

## Hunt list (each item is a place a plausible-looking diff hides a real regression)

1. **PRD-09 SDK-key plane**: ak_pub_*/ak_srv_* acceptance must remain ONLY in `require_task_context` — any path letting `get_request_context_hybrid` mint a context from an ak_* bearer is a cross-tenant regression even with green tests. `_sdk_key_has_scope` keeps deny-on-empty-permissions. `git diff $BASE..HEAD -- orchestrator/tests/test_board_sdk_auth.py` must be EMPTY.
2. **Auth-gate weakening**: tests that asserted 401-without-credentials must now be pinned to `AUTH_PROVIDER=clerk`/fake provider — not deleted, not inverted. Diff the test files: hunt deleted assertions. No endpoint may special-case `auth_type=='local'` to skip workspace filtering.
3. **su-gate (PRD-143)**: LocalAuthProvider must set `system_role='admin'` — grep the diff for `super_admin`; the local user must never get su.
4. **Leftover Clerk surface**: clerk imports, dormant helpers, `sys.modules` stubs of core.auth.clerk in tests, `_legacy`/`_v2` suffixes, commented-out Clerk blocks.
5. **Identity flip**: every former `clerk_user_id == ctx.user.id` comparison must now compare `users.id` ints; service sentinels (`'api_key'`, `'sdk:<id>'`) still flow through endpoints that stringify ids.
6. **Data continuity**: the four allowlisted keeps (board_tasks created_by writes, team.py sync upsert, coordinator dual-scheme, workspace_purge clerk fields) must still exist — if the loop deleted one to satisfy a grep, that is a CRITICAL finding. No historical-row rewrites.
7. **Import direction**: nothing under orchestrator/{core,api,services,modules} imports `automatos_saas` (tests may importorskip).
8. **Secrets**: no CLERK_SECRET_KEY or real keys anywhere in the diff (code, tests, pyproject, QUICKSTART, envs/*).
9. **Frontend**: ClerkProvider/clerkMiddleware must not execute when `NEXT_PUBLIC_EDITION=oss`; unset env defaults to `saas` (prod safety); no Clerk hook callable outside the provider tree; deleted clerk-api-client-provider.tsx has zero remaining importers.
10. **Boot seed**: idempotent + gated on `AUTH_PROVIDER=local`; the clerk/SaaS path can never create the local user/workspace; reuses existing seed helpers.
11. **Scope creep**: zero compose/infrastructure edits; zero signature changes in the ~108 RequestContext consumers; no new context types; no `os.getenv` outside config.py.

## Verification

Check the branch's latest CI run (`gh run list --branch <branch> --workflow test.yml --limit 1`): a FAILURE that is NEW versus the base branch is a finding; pre-existing reds are noted, not filed.


Run `bash scripts/ralph/acceptance-prd150.sh`. Non-zero exit = automatic CRITICAL finding.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM findings** → reply exactly `REVIEW_PASS` plus a 5-line summary (LOW/nits listed there, not filed).
- **Findings** → append fix stories to `scripts/ralph/prd-150.json` `userStories` with ids `P150-RVW-1..n`: title, description with file:line evidence, mechanical acceptanceCriteria, files. Commit `chore(prd-150): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only your fix-story commit to the existing `ralph/prd-150-`* branch (never force, never another ref). Do not re-run the build.
