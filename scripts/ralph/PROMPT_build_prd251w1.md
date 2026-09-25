# Ralph Build Prompt — PRD-251 Socials, Wave 1 (the video engine)

You are executing **PRD-251 Wave 1**, ONE story per session, unattended: every story gets a fresh session (owner, 2026-09-25). Branch **`feat/prd-251-w1-video-engine`**, in its own worktree. Keep the tip green after every commit.

**The base is Wave 0.** Wave 0 is the switches, the tables, the post lifecycle and API, the bare Socials tab and the Composio deny list. It lands on `main` through PR #783 (`test/customer-night` → `main`; #782 closes as landed there). If this branch was cut after that, its base is `main`; if before, it is **stacked on** `feat/prd-251-socials` (owner, 2026-09-23: "push forward").
- The base for every diff and check is what `bash scripts/ralph/acceptance-prd251w1.sh --print-base` prints: the Wave 0 fork point while stacked, the `origin/main` merge-base after the move.
- **Never merge anything into this branch**: not `main`, not `feat/prd-251-socials`. A merge would drag another branch's changes into this wave's diff. Integration is the owner's job.

**CONTEXT.** Wave 1 is what makes a post worth approving. It adds:
- the `media-render` service, which turns a template, the brand kit, the voice and the music into a finished MP4 or PNG;
- social templates as data, with the four reference videos (UI story, cinematic product, app promo, data story) seeded as templates;
- the brand kit extension;
- facts with sources;
- voice: Kokoro, or the workspace's Composio voice toolkit;
- the music library;
- the infographic;
- AI footage and stills from the workspace's own Composio toolkits;
- the `media` cost lane and the per-plan render quotas;
- **the agent layer** (US-115..US-120; owner, 2026-09-23: "reuse what exists"): brand-kit tools, Socials draft tools (S4.1), social formats in `generate_document` plus a general image tool, the post gate (S3.5), built-in skills synced from `automatos-skills`, and the Socials marketplace package (two agents, four playbooks, guided setup). **Reuse first:** extend the existing tools, seeders and installers named in each story; build new only where the story says nothing fits.

Nothing in Wave 1 schedules or publishes a post (Wave 3), and the full composer is Wave 2.

**Skills.** The owner's rule: skills are authored in the `automatos-skills` repo first, and any copy in this repo is synced from it. This loop **never writes skill content** and never edits a generated seed (`orchestrator/core/seeds/platform-management-skill.md`). US-119 builds the loading and sync mechanism, tested with a fixture skill; the owner runs the sync that brings the Socials skills in. When a story needs a skill's wording changed, list the lines in the commit body.

## Read first, every iteration

1. `scripts/ralph/prd-251w1.json` is the BINDING contract (stories, acceptance criteria, anchors, traps). Take the first story with ACs not yet marked DONE.
2. `docs/PRDS/PRD-251-SOCIALS.md` is the spec.
   - Decisions **D1–D17 are binding**.
   - The status block carries the owner's Wave 1 answers:
     - the renderer runs at 4 vCPU / 8 GB, two renders at once overall, one per workspace;
     - quotas are Basic 10, Pro 60, Business 240 minutes a month;
     - both editions ship in Wave 1;
     - S4.4 is in this wave.
3. `docs/PRDS/PRD-251A-SOCIALS-REFERENCE-PIPELINE.md` is how the reference videos were actually made: every setting, every trap, and the measured costs. **The templates port those compositions**, which are in `docs/PRDS/prd251-reference/`.
4. `CLAUDE.md`: reuse over build, delete what you replace, canonical terms, no `os.getenv` outside `config.py`, no hardcoded values.

## The execution contract

- **This session ends when your turn ends (`claude --print`).** Nothing reports back later.
  - Never end your turn while anything runs in the background: an agent, a background shell, a `gh run watch`. The process is killed about 600 s later, the story is left uncommitted, and the next session has to recover it. Wave 1's first sessions lost over an hour this way (2026-09-25).
  - Never use ScheduleWakeup, and never reply that you will "pick up when it reports": there is no later.
  - Wait for CI in the foreground: `gh run watch <id> --exit-status`. If the tool call times out, run it again.
  - A code review in a build session runs in the FOREGROUND (`run_in_background: false`), before the commit. It is only for a story that touches money (D13), the deny list or the post gate (D14, D16), approval (D6) or auth. Every other story relies on CI and the wave-end review session.
  - Commit and push as soon as the story's code and tests are written, then fix forward on CI. Uncommitted work at the end of a session is lost time.
- **RE-VERIFY every anchor by grep before building on it.** If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.** Follow the story order in the JSON: the service and its CI job, then the cost lane, then the client with the quotas (they count the lane's render units), then everything that renders or spends, then the agent layer (US-115..US-120) on top of the finished engine.
- **The renderer assembles; it never generates (D3).**
  - `media-render` does not call a generation provider.
  - Footage, stills and premium voice come from the workspace's Composio toolkits through the orchestrator's per-toolkit recipes (D12), and they reach the renderer as files.
- **Money is guarded (D13, D16).**
  - Estimate, then check the post's cap and the workspace's monthly cap, then submit, then book.
  - Only allowlisted actions per toolkit.
  - The Wave 0 deny list is never weakened.
  - Provider output URLs are copied into our storage the moment a job completes.
- **Templates are data (D4).**
  - `document_templates` rows with `format` `social_image` or `social_video`.
  - No template hardcodes a colour, font or logo: brand tokens are CSS variables from the brand kit.
  - Every word on screen is template text, never generated.
- **The GPL boundary (Traps).** `phonemizer` and espeak-ng live only inside `media-render`. The orchestrator never imports them.
- **One migration for the wave.** It chains onto the single alembic head of this branch's base: read `EXPECTED_HEAD` from `git show $(bash scripts/ralph/acceptance-prd251w1.sh --print-base):orchestrator/tests/test_prd209_alembic_single_head.py`. Stacked on Wave 0 that is `prd251_socials`; on main after the customer-night PR it is `f049_prd251_merge_heads`. Never chain onto a revision that is not the head. Move the head pins: `test_prd209_alembic_single_head.py` `EXPECTED_HEAD`, and `test_prd236_w1_routes.py`'s guard, which reads the head migration file and asserts the `EXPECTED_HEAD` line. Keep `test_prd251_models.py`'s chained-revisions assertion (~:332) true, unweakened.
- **The migration is create_all-first safe.** On the 2026-09-23 refresh, a backend that had already loaded new models ran `create_all` before the migration, and `prd251_socials` crash-looped on DuplicateTable until 89d89c250 fixed it. Every step must tolerate what `create_all` already built:
  - keep a table that exists and add only its missing indexes (`_has_table` / `_create_missing_indexes` in `prd251_socials.py`);
  - `ADD COLUMN IF NOT EXISTS` (`llm_usage_agent_name.py`);
  - `DROP CONSTRAINT IF EXISTS` then `ADD CONSTRAINT` for a changed CHECK;
  - insert-if-absent for seeds.
  A test proves it: `create_all` first, then the upgrade, twice.
- **Route manifest.** Every new route the UI calls must be in the COMMITTED `orchestrator/reports/route-manifest.json`.

## Tests: CI is the only gate — nothing runs on this machine

This is the owner's standing rule, and it's also a fact: this machine has no Python with the orchestrator's dependencies, and its Docker engine belongs to the owner's customer-night stack.

**Never** run pytest, npm, tsc, docker, compose, hyperframes, ffmpeg, a server or a browser here. **Never** `pip install` or `npm install` anything.

- **You write the tests. CI runs them.**
  - Backend tests: `orchestrator/tests/test_prd251w1_*.py`.
  - Frontend tests: vitest files with `socials` in the name.
  - `media-render` tests: live inside `services/media-render/`, and the new CI job runs them.
- **The new CI job (S1.1a) builds the `media-render` image and renders the fixture composition, timed.** From that story on it is one of the gate's required jobs.
- **Every story is proven on its own commit:**
  1. `git commit -s`, then push to `origin feat/prd-251-w1-video-engine`.
  2. Find the `test.yml` run for that SHA: `gh run list --branch feat/prd-251-w1-video-engine --workflow test.yml --limit 10 --json databaseId,headSha,status`.
  3. Wait for it: `gh run watch <id> --exit-status > /dev/null`.
  4. Read the jobs: `gh run view <id> --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'`.
- **The jobs that must be green:**
  - `orchestrator-tests`
  - `Alembic from-zero — exactly one head`
  - `Schema-drift check (four writers)`
  - `Frontend CI (tsc baselined, vitest, eslint, route-contract)`
  - `Prod images built with default args carry production behaviour`
  - the new `media-render` job (once it exists)
- **If one is red:** read it with `gh run view <id> --log-failed | tail -200`, fix it in a new signed commit, and push again. Red that also fails on the base's latest run is pre-existing; say so in the commit body. While stacked, the base is `feat/prd-251-socials`: `gh run list --branch feat/prd-251-socials --workflow test.yml --limit 1`.
- **When the jobs are green,** mark the story's AC lines `→ DONE — <evidence, including the CI run id>` in `scripts/ralph/prd-251w1.json`. Commit that alone: `chore(prd-251): <US-id> ACs DONE — CI run <id> green`.
- **At the START of every iteration,** check the latest `test.yml` run on the branch tip. Red caused by this branch comes first.
- **The dead database port.** The runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at 127.0.0.1:1. Never point anything at 5432 or 6379 on this machine.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO.
- **STAGING DISCIPLINE:** stage explicit paths only.
  - NEVER use `git add -A`, `git add .` or `git add -u`. `node_modules` is untracked and NOT gitignored.
  - NEVER use `git stash`. The stash stack is shared with every worktree and session on this machine.
- **Binary files:**
  - No generated video in the repo.
  - Fixture media is tiny and licensed. Every music track in the library carries its licence and attribution in the manifest.

## Hard NOs

- NO merging (nothing into this branch, and this branch into nothing), NO PRs (the runner opens the draft PR), and NO pushing anywhere except `origin feat/prd-251-w1-video-engine`. **Never touch `main`, `feat/prd-251-socials` or `test/customer-night`.**
- NO Wave 2+ work: no scheduling, calendar source, channel publishers, full composer or approval UI. The owner pulled exactly three later items into this wave: S3.5 (the post gate, US-118), S4.1 (the draft tools, US-116) and S4.2 (the weekly playbook, seeded unscheduled, US-120). Nothing else. If a story seems to need more, reply `RALPH_BLOCKED`.
- NO tool, and no tool parameter, that approves, schedules or publishes a post. People approve in the Socials tab; the platform publishes (Wave 3).
- NO skill content written in this repo, and NO marketplace rows in `marketplace_items`: the marketplace API never lists them. Marketplace agents are real `Agent` rows (`owner_type='marketplace'`); marketplace playbooks are `workflow_recipes` rows.
- NO attaching the old publisher skills (`instagram-curator`, `twitter-engager`, `linkedin-content-creator`) or `html-to-png` to anything.
- NO provider API clients (Higgsfield, fal, ElevenLabs, fish.audio) and NO API keys in Settings. Paid tools are reached ONLY through the workspace's Composio connection (D15).
- NO weakening of the Wave 0 gates, the deny list, an existing test or a CI job to make a story pass.
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py`, and NO hardcoded values.
- NO docker, compose, servers, browsers, renders or database connections from this machine.

## Per-iteration protocol

1. Check CI on the branch tip. Red caused by this branch comes first.
2. Pick the first story with ACs not yet DONE, and re-verify its anchors.
3. Implement it with its tests. Commit (signed) and push.
4. Wait for `test.yml` on that SHA, and fix until the required jobs are green.
5. Commit the DONE marks (signed) and push.
6. **STOP.** One story per session. Unless that was the last story (see Completion), end your reply with one line, `STORY_DONE <US-id>`, and do not start the next story: the runner starts a fresh session for it, and state carries over only through git (the commits and the DONE marks in the kit JSON).

## Completion

- **All ACs DONE** (this session finished the last story): instead of `STORY_DONE`, run `bash scripts/ralph/acceptance-prd251w1.sh` (greps, git and the CI result for the pushed HEAD; it runs nothing locally). If it exits 0, reply `RALPH_COMPLETE`.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the grep evidence.
