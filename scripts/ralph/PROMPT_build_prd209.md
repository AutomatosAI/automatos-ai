# Ralph Build Prompt — PRD-209 Open-Core Phase 0 (fresh-clone boot + honest CI)

You are executing **PRD-209**, one story per iteration, unattended. Branch **`ralph/prd-209-fresh-clone-boot` ← `origin/main`**. Tip green after every commit.

**THE POINT — read twice.** Open-core is a claim until a stranger can `git clone && docker compose up` and get a running local edition — and until the CI lanes that guard that boot actually bite. Two invariants rule every story: **(A) the SaaS path stays byte-identical** (`AUTH_EDITION` default `saas`; Railway never reads compose; zero behavioural change in saas mode), and **(B) no stamped database is ever stranded** (Railway prod sits at the current 12-head alembic frontier; plain `alembic upgrade heads` must keep working from there with no manual step). A story that violates either is a failure even if its own ACs pass.

**SCOPE.** 9 stories (`scripts/ralph/prd-209.json` = the BINDING contract; the seeded spec `docs/PRDS/PRD-209-PHASE2-WAVE-4-FRESH-CLONE-BOOT-HONEST-CI.md` = intent; program context `docs/PRDS/PRD-WAVE-OPEN-CORE.md`). Baked owner decisions are in the JSON description (Q1 four-constraint lineage rule; Q6 workspace-id from the CI seed convention).

## Read first, every iteration

1. `scripts/ralph/prd-209.json` — first story with un-DONE ACs; ACs are binding.
2. The seeded PRD — Current reality, the story's section, §12 questions (baked answers in the JSON).
3. `CLAUDE.md` — reuse over build; no backward-compat shims; canonical terms.

## The execution contract

- **RE-VERIFY anchors by grep** (grounded @ 182cd6739, they drift): entrypoint mode (`git ls-files -s docker-entrypoint.sh`); compose bind-mount `docker-compose.yml:~188`; `validate_auth_edition()` `orchestrator/config.py:~1536-1575` and what it requires in `local`; the CI seed's workspace convention in `orchestrator/tests/` (init_test_db seeding); smoke script `scripts/ci/smoke-fresh-clone.sh:39-48`; `continue-on-error` sites (`smoke-fresh-clone.yml:~51`, the from-zero step in `test.yml`); lockfile trio under `frontend/`; the six `infrastructure/docker-compose*.yml`.
- **This wave touches**: compose, `envs/*.defaults`, workflows, `scripts/ci/`, alembic **lineage**, docs, guard tests. It does **not** touch runtime intelligence, auth code, routers (except US-005's possible trivial readiness route + manifest bump), or any schema shape.
- **Public repo:** NO secrets in any committed file. `envs/*.defaults` carry non-secret defaults only; compose keeps the three `:?` secrets required.
- **Required lanes are off-limits:** never edit the definitions of `orchestrator-tests` / `ioc-scan`. Only the named non-required lanes (smoke-fresh-clone, alembic-from-zero, the new drift lane).
- Guard tests are pure (read files/git index; no servers, no Docker — this machine runs nothing).
- No `os.getenv` outside `config.py`; no new tables; no backward-compat shims.
- **STAGING DISCIPLINE:** explicit paths only; NEVER `git add -A`/`.`/`-u`; never `git stash -u`.
- PURE tests locally (`cd orchestrator && python3 -m pytest -q`; @integration skips); real-Postgres + the boot lanes run in CI on each per-story push — for US-004/005 the CI result IS the evidence, quote it in the commit body.

## Hard NOs

- NO deleting alembic revision files (strands stamped DBs — constraint B). The merge-revision + initdb-stamp shape is the approved US-002 implementation.
- NO weakening or skipping existing tests to get green; NO `continue-on-error` anywhere new.
- NO edits to Railway config, `core/auth/`, or anything that changes saas-mode behaviour.
- NO new env vars read outside `config.py`; NO secrets in defaults files.
- PUSH each story commit to `origin ralph/prd-209-fresh-clone-boot` ONLY. No PRs mid-run, no merges.

## Per-iteration protocol

1. First un-DONE story; re-verify its anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q` (+ frontend vitest/build for US-007).
3. Commit (`feat(prd-209)`) with evidence in the body; mark ACs `DONE — <evidence>` in `scripts/ralph/prd-209.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd209.sh`. Exit 0 → reply `RALPH_COMPLETE` (final line, alone).
- Hard-NO conflict or an unsatisfiable constraint pair → `RALPH_BLOCKED` (final line) + one-line why + grep evidence in the last commit.
