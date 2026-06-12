# Ralph Review Prompt — PRD-153 One-Command Local Run

You are a fresh-context **adversarial code reviewer**. The build loop claims PRD-153 is complete. Refute it. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD ralph/prd-152-mem0-decoupling)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read `scripts/ralph/prd-153.json` (description = binding contract + verifier amendments).

## Hunt list

1. **block-secrets.sh diff**: EXACTLY one narrow `*.example` allow added before the block patterns — any broader weakening (new allowed globs, removed patterns, heredoc-write bypasses to real .env files) is a hard CRITICAL.
2. **Compose isolation traps**: grep the final docker-compose.yml for `container_name:`, fixed volume/network `name:`, and hardcoded host ports — any of the three defeats `-p` project isolation and will collide with the dev stack.
3. **Railway safety (FR-7)**: `railway.json` + `infrastructure/railway-manifest.json` byte-untouched (`git diff $BASE..HEAD --` them = empty); orchestrator/Dockerfile changes limited to entrypoint COPY + CMD simplification; entrypoint idempotent against an already-migrated DB. If alembic was squashed: the commit/report must state prod needs `alembic stamp <base>` before the next deploy — verify that warning EXISTS.
4. **Seed strictness**: load_seed_data hard-fails on real errors but stays idempotent on re-boot against an already-seeded DB (ON CONFLICT paths) — otherwise every Railway restart dies.
5. **Smoke blast radius**: every docker/compose invocation in scripts/compose-smoke.sh is scoped `-p $SMOKE_PROJECT` with the generated `--env-file`; no prune, no bare `docker compose down`, no volume rm by literal name; waits bounded with `--wait-timeout`.
6. **os.getenv discipline**: no new os.getenv outside orchestrator/config.py in any new/edited Python; env reads in tests/scripts via config or fixtures.
7. **Deletion completeness**: six infrastructure compose files + infrastructure/.env.example + init_complete_schema.sql + core/database/migrations/ + core/database/init_database.py + root docker-entrypoint.sh + root Dockerfile.backend actually DELETED (not stubbed); nothing in scripts/, docs/, or CI still references them.
8. **Auth honesty**: zero Clerk env vars required in the local path; no weakening of PRD-150's gates to make the smoke pass (REQUIRE_AUTH forced in code rather than env, stubbed identity in the API = CRITICAL).
9. **Secrets hygiene**: no generated smoke env committed; no real keys in compose/env/test fixtures; .env.example is placeholders + generation instructions only.
10. **Docs**: QUICKSTART/docs use canonical terms (Playbook, Mission, Task, Auto, Deliverable); stale fake-default claims gone.

## Verification

Check the branch's latest CI run (`gh run list --branch <branch> --workflow test.yml --limit 1`): a FAILURE that is NEW versus the base branch is a finding; pre-existing reds are noted, not filed.


Run `bash scripts/ralph/acceptance-prd153.sh` (the smoke is `-p` scoped and self-tears-down). Non-zero exit = automatic CRITICAL finding.

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM findings** → reply exactly `REVIEW_PASS` plus a 5-line summary (LOW/nits there).
- **Findings** → append fix stories `P153-RVW-1..n` to `scripts/ralph/prd-153.json` (file:line evidence, mechanical ACs). Commit `chore(prd-153): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- Do not fix code. Push only to `origin ralph/prd-153-`* (your fix-story commit may be pushed); never force, never another ref.
