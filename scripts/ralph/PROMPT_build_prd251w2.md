# Ralph Build Prompt — PRD-251 Socials, Wave 2 (the Socials tab)

You are executing **PRD-251 Wave 2**, ONE story per session, unattended: every story gets a fresh session (owner, 2026-09-25). Branch **`feat/prd-251-w2-socials-tab`**, in its own worktree. Keep the tip green after every commit.

**The owner builds Socials in three stages** (2026-09-25: "lets keep building on their own branch and we can test each stage one at a time"). Each stage runs on its own branch and is tested on the socials stack before the next launches:
1. Wave 1: the video engine and the agent layer.
2. **Wave 2 (this run): the Socials tab.**
3. Wave 3: scheduling and publishing.

**The base is Wave 1.** Wave 1 built the video engine and the agent layer on top of Wave 0 (the switches, the tables, the post lifecycle and API, the bare Socials tab, the Composio deny list). If Wave 1 is on `main`, this branch was cut from `main`; otherwise it is **stacked on** `feat/prd-251-w1-video-engine`.
- The base for every diff and check is what `bash scripts/ralph/acceptance-prd251w2.sh --print-base` prints: the Wave 1 fork point while stacked, the `origin/main` merge-base once Wave 1 is on main.
- **Never merge anything into this branch**: not `main`, not `feat/prd-251-w1-video-engine`. A merge would drag another branch's changes into this run's diff. Integration is the owner's job.

**CONTEXT.** Wave 2 is the tab a person uses, and the owner tests it by hand. It builds, in order:
1. The wave's ONE migration (`social_campaigns`).
2. Wave 1's review fixes, `P251W1-RVW-2` to `P251W1-RVW-7` (see below).
3. What the composer needs:
   - media hosting (presigned inline URLs, checked against a real MinIO in CI);
   - the channel capabilities in Wave 1's registry, plus `GET /api/socials/channels`;
   - channels as approved content: the approval hash covers the post's targets.
4. The tab:
   - a board beside the list, search, and the phone form;
   - the approval UI and approver notifications;
   - the composer: brief → proposal, variables and preview, formats and channels;
   - series approval.

**Wave 1's review fixes come right after the migration** (owner, 2026-09-28). Wave 1's adversarial review (2026-09-26) filed them against code that has been on `main` since #788:
- `P251W1-RVW-2`: voice spend is priced and checked against the caps.
- `P251W1-RVW-3`: the render quota holds when renders run at once.
- `P251W1-RVW-4`: media-render admits a bounded number of jobs per workspace.
- `P251W1-RVW-5`: spend is booked before the provider can charge.
- `P251W1-RVW-6`: the seeded starters are brand-neutral, and the brand rule catches every colour literal.
- `P251W1-RVW-7`: housekeeping.

Build each exactly to its ACs, like any story. Three of them touch money (D13), so their code review runs in the foreground (below). Their line numbers are as of Wave 1's tip `20074ea93`, so re-verify every anchor on the base. `P251W1-RVW-1` (the marketplace persona, and the same hole in the plugins route) was fixed on its own, by PR #792: never rebuild it. If #792 is not on `main` when this run starts, leave those files alone; the owner merges it.

**Nothing publishes in this wave.** `publish-now` keeps answering 501 ("Channel publishing arrives in Wave 3"), and no scheduler job exists. The publisher, the channel publishers, scheduling and the calendar are Wave 3's.

**Reuse first.** Extend Wave 0's service, API and tab, and Wave 1's registry, render lifecycle, sources search, brand kit and built-in skills. Build new only where a story says nothing fits.

**What stays with the owner** (his test script is `docs/PRDS/prd251-w2-owner-test.md`): the browser checks at 390 px and 1440 px, and the composer's real render times. Nothing runs on this machine; never fake these, and never claim them.

## Read first, every iteration

1. `scripts/ralph/prd-251w2.json` is the BINDING contract (stories, acceptance criteria, anchors, traps). Take the first story with ACs not yet marked DONE.
2. `docs/PRDS/PRD-251-SOCIALS.md` is the spec. Decisions **D1–D17 are binding**, and the **"Waves 2+3 build" block** at the top amends them:
   - the channel set is approved content;
   - file-first publishing (for the registry's data);
   - X needs the customer's own X app;
   - TikTok uploads files;
   - the post's own timezone.
3. `docs/PRDS/PRD-251A-SOCIALS-REFERENCE-PIPELINE.md`: how the reference videos were made (Wave 1's ground).
4. `CLAUDE.md` and `AGENTS.md`, which it imports: reuse over build, delete what you replace, canonical terms (Deliverable, Playbook, Command Center, Auto), no `os.getenv` outside `config.py`, no hardcoded values.
5. **Code shape, which CI checks on every line you change** (`AGENTS.md` → Code shape; a required job):
   - A Python function you touch is at most 50 code lines and nests at most 4 levels (`scripts/ci/check_changed_code_shape.py`).
   - A React component is at most 150 lines (`frontend/scripts/check-changed-lines-eslint.js`).
   - A new file is at most 800 lines.
   - ruff (`ruff.toml`) is clean on your changed lines: no blind `except Exception` that neither logs with `logger.exception` and re-raises nor fails clearly, no `print`, and cyclomatic complexity at most 10.
   - Touching an existing function that is already over a limit means splitting it in the same commit: the ratchet judges the whole function, not only your lines. A comment edit inside a long function counts too.

## The execution contract

- **This session ends when your turn ends (`claude --print`).** Nothing reports back later.
  - Never end your turn while anything runs in the background: an agent, a background shell, a `gh run watch`. The process is killed about 600 s later, the story is left uncommitted, and the next session has to recover it. Wave 1's first sessions lost over an hour this way (2026-09-25).
  - Never use ScheduleWakeup, and never reply that you will "pick up when it reports": there is no later.
  - Wait for CI in the foreground: `gh run watch <id> --exit-status`. If the tool call times out, run it again.
  - A code review in a build session runs in the FOREGROUND (`run_in_background: false`), before the commit. It is only for a story that touches money (D13), the deny list or the post gate (D14, D16), approval (D6) or auth. Every other story relies on CI and the wave-end review session.
  - Commit and push as soon as the story's code and tests are written, then fix forward on CI. Uncommitted work at the end of a session is lost time.
- **RE-VERIFY every anchor by grep before building on it.** Anchors marked "Wave 1" name what the Wave 1 loop was asked to build; read what it actually built on the base. If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.** Follow the story order in the JSON:
  1. the migration;
  2. Wave 1's review fixes (`P251W1-RVW-2` to `P251W1-RVW-7`);
  3. media hosting;
  4. the registry;
  5. channels as approved content;
  6. the tab;
  7. the approval UI;
  8. the composer (three stories);
  9. series approval.
- **Approval is sacred (D6).**
  - An approval binds to the content hash, and from US-204 the hash covers the post's channels. Changing an approved or scheduled post's targets voids its approval.
  - Series approval approves each post through the same `service.approve`, hash-bound.
  - Weakening any of this = stop and reply `RALPH_BLOCKED`.
- **One way out (D14).** Agents still cannot post directly: US-203 extends Wave 1's post gate to every action the registry classifies as `publish`. The Wave 0 deny list is checked first and is never weakened.
- **Slugs are data.**
  - Channel action slugs live only in the registry's adapter data.
  - A seeded adapter needs only the slug's presence in `composio_actions_cache`: its `parameters` are often empty after the bulk sync.
  - The data never lists the stale or URL-pull slugs as usable: `TWITTER_CREATE_TWEET`, the deprecated Instagram actions, `TIKTOK_PUBLISH_VIDEO`, `TIKTOK_POST_PHOTO`.
  - **Never widen the global `UPLOAD_ACTIONS`**: the data carries each adapter's own upload spec for Wave 3.
- **Nothing publishes.** `orchestrator/modules/socials/publisher.py` stays exactly as the base has it: no `social-publish-` jobs, no `DateTrigger`. The gate checks this.
- **Routes.**
  - Every NEW database-touching route is a plain `def`, never `async def` over a sync Session (F105).
  - Every new route the UI calls goes in the COMMITTED `orchestrator/reports/route-manifest.json`, with its HTTP method, edited by hand: `scripts/dump_routes.py` needs the app's dependencies, which this machine lacks.
  - `apiClient` calls each route with the backend's method. The gate checks methods, and the platform's own route-contract check compares only paths.
- **Config.** Every new setting goes in `orchestrator/config.py`, and its NAME in `orchestrator/reports/config-surface.json` in sorted position, edited by hand.
- **One migration for the wave (US-201).**
  - It chains onto the single alembic head of this branch's base: read `EXPECTED_HEAD` from `git show $(bash scripts/ralph/acceptance-prd251w2.sh --print-base):orchestrator/tests/test_prd209_alembic_single_head.py`. That is Wave 1's revision, stacked or on main. Never chain onto a revision that is not the head.
  - Move the head pins: `test_prd209_alembic_single_head.py` `EXPECTED_HEAD`, and `test_prd236_w1_routes.py`'s guard. Keep `test_prd251_models.py`'s chained-revisions assertion (~:332) true, unweakened.
  - A later story in this wave that truly needs schema EXTENDS that same revision.
- **The migration is create_all-first safe.** Every step must tolerate what `create_all` already built:
  - keep an existing table and add only its missing indexes (`_has_table` / `_create_missing_indexes` in `prd251_socials.py`);
  - add a FK only when it is absent;
  - `IF NOT EXISTS` / `IF EXISTS`.

  A test proves it: `create_all` first, then the upgrade, twice.
- **Frontend.**
  - Fetches go through `apiClient` (raw `fetch` is eslint-banned).
  - Compact (phone) CSS goes ONLY in the compact region of `frontend/app/globals.css` (`studio-mobile-scope.test.ts` enforces it). Tailwind 3.3.3 has no `dvh`.
  - Media previews use the shared `FilePreview`.
  - Add NO new npm dependency; the board is read-only.

## Tests: CI is the only gate — nothing runs on this machine

This is the owner's standing rule, and it's also a fact: this machine has no Python with the orchestrator's dependencies, and its Docker engine belongs to the owner's customer-night stack.

**Never** run pytest, npm, tsc, docker, compose, hyperframes, ffmpeg, a server or a browser here. **Never** `pip install` or `npm install` anything.

- **You write the tests. CI runs them.**
  - Backend tests: `orchestrator/tests/test_prd251w2_*.py`.
  - Frontend tests: vitest files with `socials` in the name, or under `components/deliverables/socials/__tests__/`.
  - The LLM, media-render and Composio are mocked or faked in every test. No test reaches a real platform.
- **The MinIO check (US-202)** runs in the `orchestrator-tests` job against a pinned MinIO started by a workflow step. The test fails, never skips, when `CI=true` and its endpoint is missing.
- **Every story is proven on its own commit:**
  1. `git commit -s`, then push to `origin feat/prd-251-w2-socials-tab`.
  2. Find the `test.yml` run for that SHA: `gh run list --branch feat/prd-251-w2-socials-tab --workflow test.yml --limit 10 --json databaseId,headSha,status`.
  3. Wait for it: `gh run watch <id> --exit-status > /dev/null`.
  4. Read the jobs: `gh run view <id> --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'`.
- **The jobs that must be green:**
  - `orchestrator-tests`
  - `Alembic from-zero — exactly one head`
  - `Schema-drift check (four writers)`
  - `Frontend CI (tsc baselined, vitest, eslint, route-contract)`
  - `Prod images built with default args carry production behaviour`
  - `media-render` (from Wave 1)
  - `Code standards on changed lines (ruff, function length, nesting)`: marked non-required on pull requests, but required for this run. Red is red.
- **If one is red:** read it with `gh run view <id> --log-failed | tail -200`, fix it in a new signed commit, and push again. Red that also fails on the base's latest run is pre-existing; say so in the commit body. While stacked, the base is `feat/prd-251-w1-video-engine`: `gh run list --branch feat/prd-251-w1-video-engine --workflow test.yml --limit 1`.
- **When the jobs are green,** mark the story's AC lines `→ DONE — <evidence, including the CI run id>` in `scripts/ralph/prd-251w2.json`. Commit that alone: `chore(prd-251): <US-id> ACs DONE — CI run <id> green`.
- **At the START of every iteration,** check the latest `test.yml` run on the branch tip. Red caused by this branch comes first.
- **The dead database port.** The runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at 127.0.0.1:1. Never point anything at 5432 or 6379 on this machine.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO.
- **STAGING DISCIPLINE:** stage explicit paths only.
  - NEVER use `git add -A`, `git add .` or `git add -u`. `node_modules` is untracked and NOT gitignored.
  - NEVER use `git stash`. The stash stack is shared with every worktree and session on this machine.
- **Binary files:** only tiny test fixtures (under 100 KB, generated or licensed) under `orchestrator/tests/fixtures/`. No rendered media in git.

## Hard NOs

- NO merging (nothing into this branch, and this branch into nothing), NO PRs (the runner opens the draft PR when the owner allows it), and NO pushing anywhere except `origin feat/prd-251-w2-socials-tab`. **Never touch `main`, `feat/prd-251-w1-video-engine`, `feat/prd-251-socials` or `test/customer-night`.**
- NO publishing code: no publisher, no channel publishers, no scheduler jobs, no calendar source. All of that is Wave 3's.
- NO tool, and no tool parameter, that approves, schedules or publishes: agents draft; people approve in the tab. Wave 1's agent tools stay draft-only.
- NO weakening of the Wave 0 deny list, the Wave 1 post gate, an existing test or a CI job to make a story pass.
- NO direct platform API clients, and NO new keys in Settings. Composio only (D8).
- NO channel slug literal outside the registry's adapter data. NO widening of `UPLOAD_ACTIONS`. NO `/api/generated-images` for Socials media.
- NO skill content written in this repo, and NO edit to the generated seed `orchestrator/core/seeds/platform-management-skill.md` (skills are authored in `automatos-skills` first; the owner syncs them).
- NO Phase 2 work (comments, mentions, replies, metrics dashboards): it is a separate PRD.
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py` (tests excepted), and NO hardcoded values.
- NO new npm dependency.
- NO docker, compose, servers, browsers, renders or database connections from this machine.

## Per-iteration protocol

1. Check CI on the branch tip. Red caused by this branch comes first.
2. Pick the first story with ACs not yet DONE, and re-verify its anchors (the Wave 1 ones on the base).
3. Implement it with its tests. Commit (signed) and push.
4. Wait for `test.yml` on that SHA, and fix until the required jobs are green.
5. Commit the DONE marks (signed) and push.
6. **STOP.** One story per session. Unless that was the last story (see Completion), end your reply with one line, `STORY_DONE <US-id>`, and do not start the next story: the runner starts a fresh session for it, and state carries over only through git (the commits and the DONE marks in the kit JSON).

## Completion

- **All ACs DONE** (this session finished the last story): instead of `STORY_DONE`, run `bash scripts/ralph/acceptance-prd251w2.sh` (greps, git and the CI result for the pushed HEAD; it runs nothing locally). If it exits 0, reply `RALPH_COMPLETE`.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the grep evidence.
