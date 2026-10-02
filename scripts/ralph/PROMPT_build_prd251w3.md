# Ralph Build Prompt — PRD-251 Socials, Wave 3 (schedule and publish)

You are executing **PRD-251 Wave 3**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-251-w3-publish`**, in its own worktree. Keep the tip green after every commit.

**The owner builds Socials in three stages** (2026-09-25: "lets keep building on their own branch and we can test each stage one at a time"):
1. Wave 1: the video engine and the agent layer (on `main`).
2. Wave 2: the Socials tab (`feat/prd-251-w2-socials-tab`).
3. **Wave 3 (this run): scheduling and publishing.**

**The base is Wave 2.** While Wave 2 is not on `main`, this branch is **stacked on** `feat/prd-251-w2-socials-tab`; once it is, the branch is cut from `main`.
- The base for every diff and check is what `bash scripts/ralph/acceptance-prd251w3.sh --print-base` prints.
- **Never merge anything into this branch**: not `main`, not the Wave 2 branch. Integration is the owner's job.

**CONTEXT.** Wave 2 built everything publishing stands on:
- media hosting: presigned inline links, public storage for URL-only steps (`modules/socials/media_urls.py`);
- the channel registry and the adapter DATA: every channel's documented action sequence, its file params, its URL-only params (`modules/socials/channel_adapters.py`, `capabilities.py`);
- channels as approved content: the content hash covers the targets, and each target carries its resolved steps (`modules/socials/targets.py`);
- the post gate's way through, `PLATFORM_PUBLISHER` (`core/composio/post_gate.py`), which nothing passes yet.

Wave 3 builds, in order:
1. **The publisher** (US-301, with LinkedIn): the ONE engine that runs a target's steps through Composio.
2. **The other channels** (US-302 X, US-303 Instagram, US-304 TikTok, US-305 YouTube and the generic adapter): data and tests on the same engine, never a code branch per channel.
3. **Scheduling** (US-306): one-shot jobs, reconcile, missed slots.
4. **The calendar** (US-307): the `social` kind, drag to reschedule.
5. **The publish controls** (US-308) in the post view.
6. **One way out, end to end** (US-309), and the docs.

**Reuse first.** The scheduler (`services/scheduler.py`, the fcntl-locked leader), its reconcile tick (`services/schedule_reconcile.py`), the DateTrigger precedent (`services/scheduled_task_service.py`), the boot reaper (`core/boot/reaper.py`), the Composio executor and its file resolver (`core/composio/tool_executor.py`), `launch_guarded` for background work, `NotificationDispatcher`, the Command Center calendar's sources (`services/activity_service.py`) and its kinds (`frontend/components/command-center/calendar-*.ts`). Build new only where a story says nothing fits.

**What stays with the owner** (his test script is `docs/PRDS/prd251-w3-owner-test.md`): one live publish per channel to his own test accounts, and the browser checks at 390 px and 1440 px. CI has no Composio connections. Never fake these, and never claim them.

## Read first, every iteration

1. `scripts/ralph/prd-251w3.json` is the BINDING contract. Take the first story with ACs not yet marked DONE.
2. `docs/PRDS/PRD-251-SOCIALS.md` is the spec. Decisions **D1–D17 are binding**, and the **"Waves 2+3 build" block** at the top amends them:
   - the channel set is approved content;
   - file-first publishing;
   - X needs the customer's own X app;
   - TikTok uploads files;
   - reschedule is `POST /schedule` and keeps the approval;
   - times show in the post's own timezone.
   Read D6 (approval), D8 (publishing), D9 (media), D10 (scheduling), D14 (one way out) and the **Traps** before every story.
3. `CLAUDE.md` and `AGENTS.md`, which it imports: reuse over build, delete what you replace, canonical terms, no `os.getenv` outside `config.py`, no hardcoded values.
4. **Code shape, which CI checks on every line you change** (`AGENTS.md` → Code shape; a required job):
   - a Python function you touch is at most 50 code lines and nests at most 4 levels;
   - a React component is at most 150 lines;
   - a new file is at most 800 lines;
   - ruff is clean on your changed lines (no blind `except Exception` that neither logs with `logger.exception` and re-raises nor fails clearly, no `print`, complexity at most 10);
   - touching an existing function that is already over a limit means splitting it in the same commit.
   - **Big files:** `orchestrator/api/socials.py` (970 lines), `modules/socials/service.py` (992), `services/activity_service.py` (1,327) and `frontend/components/command-center/calendar-tab.tsx` (1,019) are over 800. New code goes in new modules; these files get only the few lines a route, a transition or a hook needs.

## The execution contract

- **This session ends when your turn ends.** Nothing reports back later.
  - Never end your turn while anything runs in the background (an agent, a background shell, a `gh run watch`).
  - Never use ScheduleWakeup, and never reply that you will "pick up when it reports".
  - Wait for CI in the foreground: `gh run watch <id> --exit-status`. If the tool call times out, run it again.
  - **A code review runs in the FOREGROUND, before the commit,** for every story that touches the post gate or the executor (D14), approval (D6), or anything that makes a post leave Automatos: US-301, US-306 and US-309 always; any other story that changes `publisher`/`publishing`, `post_gate.py` or `tool_executor.py`.
  - Commit and push as soon as the story's code and tests are written, then fix forward on CI.
- **RE-VERIFY every anchor by grep before building on it.** If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.** Follow the story order in the JSON: the engine first, then each channel, then scheduling, the calendar, the controls and the end-to-end check.

## Hard rules for publishing

- **Approval is sacred (D6).** Every path that publishes calls `assert_publishable` first: publish now, retry, a scheduled job, a missed post published on recovery. A post whose approval no longer matches its content publishes nothing and makes no executor call. Weakening this = stop and reply `RALPH_BLOCKED`.
- **Publish once.** The publisher claims the post with a compare-and-set (`service.claim_unchanged`) before any executor call; a target already `published` is never run again; its `idempotency_key` goes to the platform where the action takes one. Two workers, a job fired twice, or a retry racing a scheduled fire publish once.
- **One way out (D14).** Only the publisher passes `way_through=PLATFORM_PUBLISHER`. No agent path, no tool, no route handler outside the publisher references it. The Wave 0 deny list is checked first and is never weakened; a denied step fails its target (or is skipped when optional).
- **One engine, channels as data.** Channel differences live in `channel_adapters.py`. The publisher never branches on a toolkit's name. Channel action slugs appear only in the adapter data (the gate greps).
- **File-first.** A step's `files` params go through the executor's file resolver as that call's own upload spec. **Never add a slug to the global `UPLOAD_ACTIONS`.** A step's `urls` params take a presigned inline link and need public storage (D9); without it an optional step is skipped and recorded, a required one fails its target saying so.
- **Never the stale or URL-pull slugs:** `TWITTER_CREATE_TWEET`, `INSTAGRAM_CREATE_MEDIA_CONTAINER`, `INSTAGRAM_CREATE_POST`, `INSTAGRAM_GET_POST_STATUS`, `TIKTOK_PUBLISH_VIDEO`, `TIKTOK_POST_PHOTO`. Never `/api/generated-images`.
- **Retries (D8):** only transient errors (timeouts, 5xx, 429, connection errors), at most `SOCIALS_MAX_TARGET_ATTEMPTS`, with backoff. A 4xx or a platform refusal never retries and keeps the platform's message on the target.
- **Scheduling (D10):** jobs are `social-publish-<post_id>`, registered with a module-level function and plain string args (the RedisJobStore pickles them), never a closure, only on the scheduler leader; the reconcile pass keeps jobs and posts in step. A slot missed by more than `SOCIALS_MISFIRE_GRACE_SECONDS` ends `missed` with a notification. Stale content is never posted silently.
- **Times:** UTC in the database, the post's own timezone in the UI. There is no workspace timezone.
- **No migration is expected.** Every status the wave needs already exists (`SOCIAL_POST_STATUSES`, `SOCIAL_TARGET_STATUSES` in `core/models/socials.py`). If a story truly needs schema, add ONE revision for the whole wave, chained onto the base's single head, create_all-first safe, both head pins moved, and say why in the commit body.
- **Routes.**
  - Every NEW database-touching route is a plain `def`, never `async def` over a sync Session (F105). Background work goes through `launch_guarded`.
  - Every new route the UI calls goes in the COMMITTED `orchestrator/reports/route-manifest.json`, with its HTTP method, and its `route_count` updated.
  - `apiClient` calls each route with the backend's method.
- **Config.** Every new setting goes in `orchestrator/config.py`, and its NAME in `orchestrator/reports/config-surface.json` in sorted position.
- **Frontend.**
  - Fetches go through `apiClient` (raw `fetch` is eslint-banned).
  - Compact (phone) CSS goes ONLY in the compact region of `frontend/app/globals.css`.
  - Media previews use the shared `FilePreview`.
  - **No new npm dependency:** drag to reschedule uses native HTML5 drag events.

## Tests: CI is the gate

**On the owner's machine nothing runs**: no pytest, npm, tsc, docker, compose, server or browser, and no `pip install` / `npm install`. Its Docker engine belongs to the owner's customer-night stack, and the runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at 127.0.0.1:1.

**In a Claude Code cloud session** (a disposable container, not the owner's machine), you may install the dependencies there and run the tests and the changed-line checks before pushing. It catches failures a CI round-trip would cost. CI stays the only evidence: never mark a story DONE on a local run.

- **You write the tests. CI runs them.**
  - Backend tests: `orchestrator/tests/test_prd251w3_*.py`.
  - Frontend tests: vitest files with `socials` or `calendar` in the name, or under `components/deliverables/socials/__tests__/`.
  - Composio is mocked in every test: a fake `ComposioToolExecutor` that records every `execute` call (action, params, `way_through`) and answers from a script. No test reaches a real platform.
  - Concurrency (the publish claim, a double fire) is proven on the CI Postgres, like Wave 1's quota tests.
- **Every story is proven on its own commit:**
  1. `git commit -s`, then push to `origin feat/prd-251-w3-publish`.
  2. Find the `test.yml` run for that SHA: `gh run list --branch feat/prd-251-w3-publish --workflow test.yml --limit 10 --json databaseId,headSha,status`.
  3. Wait for it: `gh run watch <id> --exit-status > /dev/null`.
  4. Read the jobs: `gh run view <id> --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'`.
  - A push cancels the previous run on the branch: never push the next story before this one's run has finished.
- **The jobs that must be green:** `orchestrator-tests`, `Alembic from-zero — exactly one head`, `Schema-drift check (four writers)`, `Frontend CI (tsc baselined, vitest, eslint, route-contract)`, `Prod images built with default args carry production behaviour`, `media-render`, and `Code standards on changed lines (ruff, function length, nesting)` (marked non-required on pull requests, but required for this run; red is red).
- **If one is red:** read it with `gh run view <id> --log-failed | tail -200`, fix it in a new signed commit, and push again. Red that also fails on the base's latest run is pre-existing; say so in the commit body.
- **When the jobs are green,** mark the story's AC lines `→ DONE — <evidence, including the CI run id>` in `scripts/ralph/prd-251w3.json`. Commit that alone: `chore(prd-251): <US-id> ACs DONE — CI run <id> green`. (`scripts/ralph` is gitignored, but the kit files are tracked: `git add` the JSON by its path.)
- **At the START of every iteration,** check the latest `test.yml` run on the branch tip. Red caused by this branch comes first.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO.
- **STAGING DISCIPLINE:** stage explicit paths only.
  - NEVER `git add -A`, `git add .` or `git add -u`. `node_modules` is untracked and NOT gitignored.
  - NEVER `git stash`: the stash stack is shared with every worktree on this machine.
- **Binary files:** only tiny test fixtures (under 100 KB) under `orchestrator/tests/fixtures/`. No rendered media in git.

## Hard NOs

- NO merging, NO PRs (the owner opens them), and NO pushing anywhere except `origin feat/prd-251-w3-publish`. Never touch `main`, the Wave 1 or Wave 2 branches, or `test/customer-night`.
- NO live Composio call, from a test or from this machine. NO real post to any platform.
- NO tool, and no tool parameter, that approves, schedules or publishes: agents draft; people approve and publish in the tab.
- NO weakening of the deny list, the post gate, D6's approval checks, an existing test or a CI job.
- NO direct platform API clients, and NO new keys in Settings. Composio only (D8).
- NO channel slug literal outside the adapter data. NO widening of `UPLOAD_ACTIONS`.
- NO skill content written in this repo, and NO edit to the generated seed `orchestrator/core/seeds/platform-management-skill.md`.
- NO Phase 2 work (comments, mentions, replies, metrics dashboards): it is a separate PRD.
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py` (tests excepted), and NO hardcoded values.
- NO new Python or npm dependency without the owner's OK.

## Per-iteration protocol

1. Check CI on the branch tip. Red caused by this branch comes first.
2. Pick the first story with ACs not yet DONE, and re-verify its anchors.
3. Implement it with its tests (and the foreground code review when the story needs one). Commit (signed) and push.
4. Wait for `test.yml` on that SHA, and fix until the required jobs are green.
5. Commit the DONE marks (signed) and push.
6. **STOP.** One story per session. Unless that was the last story, end your reply with one line, `STORY_DONE <US-id>`.

## Completion

- **All ACs DONE** (this session finished the last story): run `bash scripts/ralph/acceptance-prd251w3.sh`. If it exits 0, reply `RALPH_COMPLETE`.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the grep evidence.
