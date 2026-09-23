# Ralph Build Prompt — PRD-251 Socials, Wave 0

You are executing **PRD-251 Wave 0**, one story per iteration, unattended. Branch **`feat/prd-251-socials` ← `main`** (cut at `af214bc7b`), in its own worktree. Keep the tip green after every commit.

**CONTEXT.** PRD-251 gives every workspace a Socials tab in Deliverables: on-brand posts, approved one by one, on the calendar, published through the workspace's own Composio connections. Wave 0 lays the foundation and makes it visible:
- the two switches;
- the two tables;
- the post lifecycle and its API;
- three platform defects that must be fixed before anything publishes, plus a deny list for Composio actions that spend real money;
- a bare Socials tab, so the owner can click through the rules.

Nothing in Wave 0 renders media, schedules a job or publishes to a channel.

## Read first, every iteration

1. `scripts/ralph/prd-251.json` is the BINDING contract (stories, acceptance criteria, anchors, traps). Take the first story with ACs not yet marked DONE.
2. `docs/PRDS/PRD-251-SOCIALS.md` is the spec.
   - Decisions **D1–D17 are binding**. Since 2026-09-23 it is Composio-first: paid media comes from the customer's connected Composio tools (D12, D15, D16).
   - The status block records the owner's answers of 2026-09-23: **every plan gets Socials**, and **the LinkedIn workaround is workspace-scoped**.
   - §"What exists today (verified)" carries the file:line evidence.
3. `CLAUDE.md`: reuse over build, delete what you replace, canonical terms, no `os.getenv` outside `config.py`, no hardcoded values.

## The execution contract

- **RE-VERIFY every anchor by grep before building on it.** The evidence is from 2026-09-23 on `main` @ `af214bc7b`. If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.**
  - US-001 (switches) → US-002 (tables) → US-003 (service) → US-004 (API) → US-005, US-006, US-007 (defects) → US-008 (tab) → US-009 (the Composio deny list).
  - US-004 needs US-001–003. US-008 needs US-004's routes in the committed route manifest.
- **The two switches gate everything.** Every `/api/socials/*` route depends on `require_socials_enabled`. There is **no plan exposure key**: every plan gets Socials (D1). Do not touch `services/plan_tiers.py`.
- **Tenant isolation is not optional.**
  - Every query filters by the caller's `workspace_id`; another workspace's post is a 404.
  - The LinkedIn fix never falls back to another workspace's credential or token.
- **The Composio deny list (US-009) never depends on the policy plane or the classifier.** It is one helper that every Composio execution path calls, reading a system setting. It is never a code constant.
- **Approval binds to the hash (D6).** `publish_post` calls `assert_publishable` before anything else, and any content edit voids an approval.
- **One migration.** `prd251_socials`, chained onto `kb_multimodal_tables`. Move BOTH head pins. `social_campaigns` is not created.
- **Route manifest.** Every new route the UI calls must be in the COMMITTED `orchestrator/reports/route-manifest.json`. The frontend route-contract job reads that file; backend CI never compares it.
- **Test names.**
  - Backend tests for this PRD are `orchestrator/tests/test_prd251_*.py`. The acceptance gate counts them, and CI runs them.
  - Frontend tests sit beside the existing Deliverables tests.

## Tests: CI is the only gate — nothing runs on this machine

This is the owner's standing rule, and it's also a fact: no Python on this machine has the orchestrator's dependencies. **Never** run pytest, npm, tsc, docker, compose, a server or a browser here, and **never** `pip install` or `npm install` anything.

- **You write the tests. CI runs them.**
  - Backend: `orchestrator/tests/test_prd251_*.py`. Design them for SQLite via `create_all` and the JSON variant, like the existing unit tests. Mark anything that needs real Postgres `integration`.
  - Frontend: vitest files with `socials` in the name, beside the existing Deliverables tests.
- **Every story is proven on its own commit.** `test.yml` runs on every push to this branch, takes about 10 minutes, and a newer push cancels an older run. So:
  1. Commit the story's code and tests: `git commit -s -m 'feat(prd-251): <US-id> — <title>'`.
  2. Push: `git push -q origin feat/prd-251-socials`.
  3. Find the run for that exact SHA: `gh run list --branch feat/prd-251-socials --workflow test.yml --limit 10 --json databaseId,headSha,status`.
  4. Wait for it: `gh run watch <id> --exit-status > /dev/null`.
  5. Read the job results: `gh run view <id> --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'`.
- **The five jobs that must be green:**
  - `orchestrator-tests`
  - `Alembic from-zero — exactly one head`
  - `Schema-drift check (four writers)`
  - `Frontend CI (tsc baselined, vitest, eslint, route-contract)`
  - `Prod images built with default args carry production behaviour`. This one runs `next build`, which catches the JSX errors vitest's mocks hide (the PRD-246 US-001 lesson).
- **If one of the five is red:**
  1. Read it with `gh run view <id> --log-failed | tail -200`.
  2. Fix it in a new signed commit, push, and wait again.
  3. A red that also fails on `main`'s latest run is pre-existing. Say so in the commit body and move on.
- **When the five are green,** mark the story's AC lines `→ DONE — <evidence, including the CI run id>` in `scripts/ralph/prd-251.json`. Commit that alone: `git commit -s -m 'chore(prd-251): <US-id> ACs DONE — CI run <id> green'`, then push.
- **At the START of every iteration,** check the latest `test.yml` run on the branch tip. If it's red because of this branch, fixing it IS this iteration's story.
- **The dead database port.** The runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at 127.0.0.1:1. Ports 5432 and 6379 on this machine belong to the owner's customer-night stack. Never point anything at them.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO, and an unsigned commit fails the `dco` check.
- **STAGING DISCIPLINE:** stage explicit paths only.
  - **NEVER** use `git add -A`, `git add .` or `git add -u`. `node_modules` is untracked and NOT gitignored.
  - **NEVER** use `git stash`. The stash stack is shared with every worktree and every other session on this machine.
- **Two commits per story:**
  1. The code and tests: `feat(prd-251): <US-id> — <title>`, with the evidence in the body.
  2. Once CI is green: `chore(prd-251): <US-id> ACs DONE — CI run <id> green`, marking the story's AC lines in `scripts/ralph/prd-251.json`.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR), and NO pushing anywhere except `origin feat/prd-251-socials`. **Never touch `main` or `test/customer-night`.**
- NO Wave 1+ work: no `media-render`, templates, brand-kit fields, voice, music, scheduler jobs, calendar source or channel publishers. If a story seems to need one, reply `RALPH_BLOCKED` with why.
- NO plan exposure key, and no change to `services/plan_tiers.py`.
- NO new dependency, backend or frontend. Wave 0 needs none.
- NO weakening an existing test or gate to make a story pass. A genuine contradiction is `RALPH_BLOCKED` with evidence.
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py`, and NO hardcoded values. Constants live in config.
- NO docker, compose, servers, browsers or database connections from this machine.

## Per-iteration protocol

1. Check CI on the branch tip (above). Red caused by this branch comes first.
2. Pick the first story with ACs not yet DONE, and re-verify its anchors fresh.
3. Implement it with its tests. Commit (signed) and push.
4. Wait for `test.yml` on that SHA. Fix until the five jobs are green.
5. Commit the DONE marks (signed) and push.

## Completion

- **All ACs DONE:** run `bash scripts/ralph/acceptance-prd251.sh`. It is greps and git plus the CI result for the pushed HEAD; it runs no tests itself. If it exits 0, reply `RALPH_COMPLETE`.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED`, with one line of why and the grep evidence in the last commit.
