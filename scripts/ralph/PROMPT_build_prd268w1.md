# Ralph Build Prompt — PRD-268 W1 Check-ins, Proactive Auto's engine (19 stories: US-C01 … US-C18)

You are building **wave W1 (Check-ins) of PRD-268**. It runs unattended, ONE story per session, and every story gets a fresh session.
- **Branch:** `feat/prd-268-w1-check-ins`, in its own worktree. It is cut from `RALPH_BASE_REF`: `origin/main` once PR #1362 (the PRD) has merged, else `origin/docs/prd-268-proactive-auto`.
- **One PR** for the whole wave (base `main`, tracking issue #1358) and **one CI run** at the end. Nothing in a story waits on CI.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd268w1.sh --print-base` prints.
- **Never merge anything into this branch.** Integration is the owner's job.
- **This wave adds a guardrail** (the check-in lane), changes heartbeat behaviour in `live` mode, and governs background spend. Gerard approved PRD-268 on 28 Oct 2026 (tracking issue #1358; the JSON's K16). Build exactly what the ACs say.
- **This wave HAS ONE migration** (US-C01, revision `prd268_w1_proactive`). Every other story adds none.
- **Core only.** No frontend file changes in this wave.

## Read first, every iteration

1. `scripts/ralph/prd-268w1.json` is the BINDING contract. Its **`decisions`** (K1–K17) are binding.
2. `docs/PRDS/PRD-268-PROACTIVE-AUTO.md`:
   - §6.2 (check-ins, streaks, the lane, the data model, Start and Hide);
   - §6.9 (cost and safety);
   - §6.11.1 (status), §6.11.3 (readiness);
   - §8 (functional requirements);
   - your story's US section in §7.
3. `CLAUDE.md` and `AGENTS.md`.
4. The story's `notes` name the seams by file and line, measured 28 Oct on `2d02ad69a`. Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Facts you need (verified 28 Oct)

### Code shape and size

- **Files over 800 lines never grow** (K3). Re-measure them:
  - `services/heartbeat_service.py` 1825;
  - `modules/tools/discovery/platform_executor.py` 1543;
  - `modules/tools/execution/unified_executor.py` 1390;
  - `main.py` 1666;
  - `modules/agents/factory/agent_factory.py` 1982;
  - `api/missions.py` 2107;
  - `api/board_tasks.py` 2282;
  - `services/activity_service.py` 1267;
  - `core/llm/manager.py` 1063.

  Where a story adds lines to one, it moves at least as many out in the same commit. Exceptions: `orchestrator/config.py` (settings lines), `route-manifest.json` and `config-surface.json`. The gate measures every file.
- **The changed-lines trap.** `scripts/ci/check_changed_code_shape.py` fails ANY touched line inside a function over 50 code lines. The finding spans the whole function, and every function whose span holds the line counts.
  - **Seams that don't count:** a decorator line above `def`; a pure deletion; a new short function; a line in a short function.
  - **Decided seams (K4):**
    - the lane is a **decorator** on `PlatformActionExecutor.clear`, which already carries `@asks_at_the_confirmation_gate`;
    - heartbeats switch through the **short scheduler functions** `schedule_agent_heartbeat` and `schedule_orchestrator_heartbeat`;
    - routers register through **`RouterSpec`** in `orchestrator/router_manifest.py`, so `main.py` never changes.
  - **Never touched**, and the gate compares each with the base: `_agent_tick`, `_orchestrator_tick`, `_orchestrator_tick_llm`, `_dispatch_heartbeat_notification`, `_route_heartbeat_to_auto`, `clear`, `_run_cleared`.
  - **Run the script on COMMITTED work only.** Commit, run it, and fix in a follow-up commit if it's red.

### The machine

- **This Mac cannot import the orchestrator.** `route-manifest.json` and `config-surface.json` are edited BY HAND or with a stdlib-only python snippet (K5).

### Alembic

- **The head** on the base was `prd266_w2_push` on 28 Oct (`EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py:138`; RE-READ it).
- **The two pins to move:**
  - that file, with its dated comment chain at `:120-137`;
  - `orchestrator/tests/test_prd236_w1_routes.py:343-345`.

### Registry and context

- **The action registry** is lazy: `_ensure_initialized` (`action_registry.py:131-137`). `permission_level` defaults to `"read"` at `:64`, and US-C02a removes that default.
  - **The 3 omissions** are the field tools (`actions_field.py:9, :45, :84`). `platform_field_inject` writes (`handlers_field.py:119`).
  - **`scripts/check_hierarchy_gate.py:48, :189`** defaults a missing level to read.
  - **19 test constructions** omit the level.
- **No `lane` key exists** in `caller_context` today; it's a plain dict. **Composio calls never reach `PlatformActionExecutor`**, which is why the UnifiedToolExecutor decorator exists. **The Anthropic memory tool** is added by `core/llm/clients/anthropic_client.py:399` on headless runs and executed outside the executor. That is why the check-in runner never uses AgentFactory (K8).
- **Context modes:** `ModeConfig` (`modes.py:112-119`), `DEFAULT_BUDGETS` (`budget.py:154-213`), `ContextService._load_tools` (`service.py:711-758`) and `ToolLoadingStrategy` (`sections/tools.py:32-38`). No strategy filters by permission level today.

### Heartbeats

- **`_agent_tick`** is `heartbeat_service.py:1020-1189`.
- **`report_to`** is read only in `_dispatch_heartbeat_notification` (`:1327`, auto branch `:1408-1418`).
- **Reports:** `_auto_create_report` (`:1709`) files one report per tick.
- **`proactive_level`** (silent, notify, act_notify, autonomous) lives on the orchestrator tick only (`:746`, `:812`).
- **The scheduler functions:** `schedule_orchestrator_heartbeat` (`:429`) and `schedule_agent_heartbeat` (`:501`) pass `self._orchestrator_tick` / `self._agent_tick` to `add_job`.

### Spend and settings

- **Spend:**
  - `daily_spend_guard.refuse_new_work(db, workspace_id, what)` (`:140`); the ceiling is platform-wide, 0 (off) by default, and fails open;
  - `background_allowed` is in `services/credits/routing.py:170`;
  - `BACKGROUND_LANES` is in `services/credits/asks.py:38`.
- **LLM:** `LLMManager(service_name="heartbeat")` uses the platform `orchestrator_llm` system setting (`core/llm/manager.py:47`; default `claude-haiku-5-5`).
- **Settings:**
  - workspace: `Workspace.settings` (JSONB); the write pattern is `api/mobile_push.py:139-160`;
  - platform kill-switch pattern: `modules/socials/settings.py:108-133`;
  - a new env setting is a `Config` attribute plus its name in `reports/config-surface.json`.
- **Audit:** `AuditService(db).log(...)` COMMITS the session (`core/workspaces/audit.py:97`) and swallows errors (K15).

### Tests

- **Realdb tests** use `tests/conftest.py` `new_session(engine)` and a module `engine` fixture (pattern: `tests/test_prd266_auto_message.py:49-61`).
- **There is no user fixture:** insert one per file.
- **A new router's test convention:** `tests/test_prd266_digest_now.py:358-365`.

## Code rules (AGENTS.md; CI checks them on every line you change)

- **Shape:** functions ≤ 50 code lines; nesting ≤ 4; cyclomatic ≤ 10 on touched code; a new file ≤ 800 lines (aim for 200–400). Type hints and docstrings on public Python functions. No mutation of arguments, and no mutable default arguments.
- **Settings:** only through `orchestrator/config.py`, each new name in `config-surface.json` in sorted position. No `os.getenv` or `os.environ` anywhere else. No hardcoded values.
- **Routes:**
  - thin (Pydantic in, one service call, errors mapped);
  - a route that touches the database without awaiting is a plain `def`;
  - every query is filtered by the caller's `workspace_id`; another workspace's row is 404, never 403;
  - every new route is in `route-manifest.json`;
  - every new endpoint is authenticated and workspace-scoped, with no exceptions in this wave.
- **Errors:** `except Exception` must log (`logger.exception`) and either re-raise or fail clearly. No `print` in application code.
- **Both editions keep working** (`AUTH_EDITION=local` and `saas`).
- **Untrusted input:** anything a check-in reads is untrusted data. Suggestion text is built from structured fields, and evidence is quoted, never followed. Starting a suggestion never posts its text as the person's own message.
- **Guardrails:** no new Python or npm dependency, and no new third-party API client. Never touch the Composio deny list, the generated skill seed, or any guardrail other than the check-in lane this wave adds.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work, never ScheduleWakeup, never "pick up later". Wait in the foreground for anything you start.
- **A code review runs in the FOREGROUND, before the story commit.**
  - **code-reviewer agent:** on every story.
  - **security-reviewer agent:** on US-C02a, US-C02b, US-C02c, US-C03b, US-C03d, US-C04b, US-C06a and US-C07a (the lane, the tool, the runner, the bridge, the Start and settings routes).
  - **database-reviewer agent:** on US-C01.

  Wait for each, and fix what it finds before committing.
- **One story per session.** Build the first story whose ACs are not all DONE, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep.** **The order is the design:** the JSON's `priority` order is the dependency order (K2).
- **Never push anything yourself.** The runner pushes the branch after each session and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine:** no pytest, alembic, uvicorn, npm, docker, server or installs.

Local checks only:
- `python3 -m py_compile <files>`;
- after each commit, `python3 scripts/ci/check_changed_code_shape.py --base $(bash scripts/ralph/acceptance-prd268w1.sh --print-base)`;
- `python3 orchestrator/scripts/check_hierarchy_gate.py`;
- `bash scripts/ralph/acceptance-prd268w1.sh --no-ci`.

You WRITE the tests each AC names, at the exact paths the ACs give; the gate checks they exist and are new on this branch. CI runs them once, at the end. A test proves the change: it fails without it. A story never waits on CI and never claims a test passed.

## Commits

- `git commit -s` on every commit (DCO).
- Messages: `feat(prd-268): description`, or `fix(prd-268):`, `test(prd-268):`, `refactor(prd-268):`.
- **Every story commit ends with ` [skip ci]`.**
- Explicit paths only (`git add <path> …`). Never `git add -A`, `git add .` or `git add -f`. Never `git stash`. Never amend a pushed commit.

## Hard NOs

- **Git:** NO merging, NO PRs, NO pushing. Never touch `main` or any other branch. Never `git checkout`/`switch` away from this branch. Never `git cherry-pick`, `revert` or `rebase`.
- **Migrations:** NO second migration. NO edit to an existing revision. NO table altered.
- **Code shape:**
  - NO line changed inside the K4 functions;
  - NO touched line inside any function still over 50 code lines after your change;
  - NO growth of a file over 800 lines, beyond the K3 exceptions.
- **Tests:** NO weakening, skipping or deleting a test or a CI check. The US-C02a edits to test constructions only ADD an explicit `permission_level`.
- **Dependencies and config:** NO new dependency, NO third-party client, NO `os.getenv` outside config.
- **Scope:** NO building W2, W3 or W4: delivery, the Auto-thread suggestion card, the `suggestion` push, the brief, reply chips, follow-ups, evals, calibration, Mission or Playbook proposals.
- **No AgentFactory in the check-in lane (K8).**

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet DONE.
2. Implement it with its tests, run the foreground review it requires and the local checks above, then commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text, outside backticks and exactly as written: `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-268w1.json`. Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <US-id>` as its last line.

**The runner reads RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE as sentinels**, on your last line only (≤ 80 characters). Never write them anywhere else in a reply, not even in a sentence about them.

## Completion

- **All ACs DONE:**
  1. Run `bash scripts/ralph/acceptance-prd268w1.sh --no-ci`.
  2. Fix what is red in signed ` [skip ci]` commits.
  3. Make ONE signed commit WITHOUT `[skip ci]`: `chore(prd-268): W1 Check-ins built, run CI`. An empty commit is fine: `git commit -s --allow-empty`.
  4. Reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** every AC is DONE, and the last commit touching anything outside `scripts/ralph/` is either the wave's CI commit or a commit without `[skip ci]` whose test.yml run on the branch completed green. Make NO commit; reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO:** reply with one line saying why, with the evidence, then `RALPH_BLOCKED` as the last line. Use it only for a real contradiction. A story that needs a judgement call takes the smaller change and says so in its DONE mark.
