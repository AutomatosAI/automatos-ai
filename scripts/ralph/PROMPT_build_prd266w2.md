# Ralph Build Prompt — PRD-266 W2 Push, the Automatos Go backend (16 stories: US-010a … US-019c)

You are building **wave W2 (Push) of PRD-266**, plus US-019 (Phones & devices in Studio), ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-266-w2-push`**, in its own worktree, cut from `origin/feat/prd-266-w1-connect` once W1 was complete (the wave is **stacked on W1**: its PR targets W1's branch until W1 lands, K1). One PR for the whole wave (tracking issue #1254; US-019 comes from #1255), one CI run at the end: nothing in a story waits on CI.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd266w2.sh --print-base` prints (the W1 tip while stacked).
- **Never merge anything into this branch** (not `main`, not W1's branch, not another wave). Integration after W1 lands is the owner's job (K1).
- **This wave changes who is told what** (push recipients: editors receive needs-an-answer pushes, D16) and adds **one third-party client** (`services/push/expo.py`, AGENTS.md exception 2). Gerard approved both on 20 Oct (#1254; the JSON's `decisions`). Build exactly what the ACs say; a change to auth, permissions or tenancy that no AC asks for is out of scope.
- **This wave HAS ONE migration** (US-010b, revision `prd266_w2_push`, chained onto W1's `prd266_mobile_devices`). Every other story adds none.

## Read first, every iteration

1. `scripts/ralph/prd-266w2.json` is the BINDING contract. Its **`decisions`** are binding: the PRD's §3 decisions that apply (D16, D18–D24, D63), §4.4, §4.5, §4.10, §8, and the kit decisions K1–K21.
2. `docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md` — §3, §4.3 (the web part), §4.4 (push), §4.5 (the digest), §4.10 (where code goes), §8, §11, and your story's US section (W2: US-009..US-014; W3: US-019).
3. `CLAUDE.md` and `AGENTS.md`.
4. The story's `notes` name the seams by file and line, measured 20 Oct on origin/main 7ae677b52 (same lines as W1's base for every file cited) and on W1's branch at 8187583bd. Line numbers drift and W1 finished after the kit was written: **RE-VERIFY every anchor by grep**, and read every "RE-VERIFY after W1" contract from W1's code (K20). Where W1's code differs from a note, follow the code and say so in the DONE mark.

## Facts you need (verified 20 Oct)

- **W1 is in the tree.** `mobile_devices` (`core/models/mobile_devices.py`, `SECRET_COLUMNS`), the pairing kernel (`core/auth/pairing.py`: codes shown `XXXX-XXXX`), the device lane (`core/auth/bearer_registry.py`, `core/auth/device.py`; `auth_type 'device'`, `raw_claims.source 'mobile_device'`), `core.auth.principal.resolve_user_pk` (THE resolver from a request to `users.id`), `services/mobile/` (`pairing.py`, `devices.py`, `events.py` `with_event`, `urls.py`, `discovery.py`), `api/mobile_devices.py` (395 lines: keep it under 400 — new routes go in the new `api/mobile_push.py`), the answerer roles (`core/auth/workspace_admin.py`, US-006a), Phones & devices in Studio (`frontend/components/settings/phones-devices/`, `hooks/use-mobile-devices.ts`). W1's revoke clears the device's Expo token and push key.
- **Files over 800 lines never grow** (§4.10): `core/auth/hybrid.py`, `api/chat.py`, `api/approval_grants.py` (806), `main.py` (1,681), `services/cli_host_service.py`, `consumers/chatbot/service.py` (3,074), `frontend/lib/api-client.ts` (3,847+), `api/workspaces.py` (985). The one exception: `orchestrator/config.py` gains its settings lines, and the generated reports gain their entries. The gate measures every file.
- **The changed-lines trap.** `scripts/ci/check_changed_code_shape.py` fails ANY touched line inside a function over 50 code lines (every function whose span holds the line counts). Seams that do not count: a decorator line above `def`; a pure deletion; a new short function; a hook in a small callee; a rewrite whose result is ≤ 50 lines. The seams decided here (K3): `NotificationDispatcher.dispatch` (129 code lines) is **rewritten** as a short function over helpers; `bulk_update_preferences` (108), `post_background_message` (76) and `requeue_expired_leases` (79) are **never touched** (the gate compares them with the base, as it does `get_request_context_hybrid`, `_stream_response_with_agent_scoped` and `get_current_workspace`); the hooks go in `deliver_background_message` (14), `_notify_swept` (15) and `start_schedule_reconcile` (19); routers register through `RouterSpec` in `orchestrator/router_manifest.py`, so `main.py` never changes. **Run the script on COMMITTED work only**: commit, run it, fix in a follow-up commit if red.
- **The dispatcher's tests patch its helpers.** `tests/test_prd128_notification_dispatcher.py`, `test_f197_a_…`, `test_f197_b_…` and `test_f209_b_…` patch `_load_auto_reporting`, `_get_preferences`, `_insert_in_app` on the class and `send_workspace_notification` by module path, and prd128's fake session hands the preference rows to the FIRST `db.execute`. Those four files are never edited; the rewrite keeps every name and that order.
- **Background work has two homes, never a third** (K6, K10): the push leg runs through `core.best_effort.off_loop` (inline where no loop runs, on its threads where one does; `tests/conftest.py` drains it at every teardown); an after-commit dispatch runs through `services/notification_events/soon.send_soon` (the `modules/socials/notify._send_soon` contract). No thread of your own, no bare `asyncio.create_task`.
- **Settings.** An environment setting goes in `config.py` and `reports/config-surface.json` (K19). The operator's Expo answer is NOT one: it lives in the DB settings table (`system_settings`, category `notifications`), read with `core.llm.manager.get_system_setting`, written as `modules/voice/live_settings.set_voice_setting` writes (K13).
- **This Mac cannot import the orchestrator** (nothing runs here): `route-manifest.json` and `config-surface.json` are edited BY HAND or with a stdlib-only python snippet (K4): routes sorted by `(path, method)`, `route_count` = the number of routes, `json.dumps(manifest, indent=2, sort_keys=True) + "\n"`; setting names in sorted position.
- **The Alembic head** on the base is `prd266_mobile_devices` (`EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py`; RE-READ it). The two pins to move: that file and `orchestrator/tests/test_prd236_w1_routes.py` (copy W1's `prd266_mobile_devices` lines' pattern).
- **Expo.** `https://exp.host/--/api/v2/push/send` (100 messages a request) and `…/getReceipts` (1,000 ids). `httpx` is already a dependency. `services/push/expo.py` is the ONLY module that speaks HTTP to anyone in this wave (the gate checks).
- **The GDPR export** (`GET /api/v1/gdpr/export`, `services/gdpr_service._export_sql_tables`) is a `SELECT *` over every workspace table: W1's hashes and the push key would leave in it until US-011a closes that (K16).
- **No `frontend/node_modules` on this machine:** `frontend/scripts/check-changed-lines-eslint.js` cannot run here; count component (≤ 150) and function (≤ 50) lengths by reading; CI's Frontend CI runs it. Tailwind `dark:` palette classes are refused by `frontend/lib/__tests__/dark-variant-tokens.test.ts`: use theme tokens.
- **No QR encoder exists** and none may be added without Gerard (K17): US-019c is gated on that decision's STATUS line.

## Code rules (AGENTS.md; CI checks them on every line you change)

- Functions ≤ 50 code lines, React components ≤ 150, nesting ≤ 4, cyclomatic ≤ 10 (ruff C901) on touched code; a new file ≤ 800 lines (aim 200–400). Type hints and docstrings on public Python functions. No mutation of arguments (build new lists and dicts); no mutable default arguments.
- Settings only through `orchestrator/config.py`, each new name in `orchestrator/reports/config-surface.json` in sorted position; no `os.getenv`/`os.environ` anywhere else (ruff TID251); no hardcoded values (named constants or config).
- Routes: new routers are thin (Pydantic in, one service call, errors mapped); a route that touches the database without awaiting is a plain `def`; every query is filtered by the caller's `workspace_id`; another workspace's row is 404, never 403. Every route the frontend calls is in `route-manifest.json`. This wave adds NO public route.
- Secrets and personal data: never log a push token, a push key, `EXPO_PUSH_ACCESS_TOKEN`, a question's text, an option or a push body; a push key is read only to sign. Errors: `except Exception` must log (`logger.exception`) and re-raise or fail clearly; no `print` in application code (scripts may print).
- Only the three push types push (FR-7): every decision reads `services/push/payload.PUSH_TYPE_EVENTS`. A local install pushes nothing until the operator agreed (K13). Both editions keep working (`AUTH_EDITION=local` and `saas`); edition behaviour is gated by `AUTH_EDITION`, never by role.
- The frontend calls the backend only through `apiClient` (no raw `fetch('/api…')`); no `innerHTML`/`dangerouslySetInnerHTML`; canonical terms (Auto, Mission, Deliverable, Command Center).
- No new Python or npm dependency; no third-party client beyond `services/push/expo.py`; never touch the Composio deny list, the generated skill seed (`orchestrator/core/seeds/platform-management-skill.md`) or any guardrail.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work; never ScheduleWakeup; never "pick up later". Wait for anything you start in the foreground.
- **A code review runs in the FOREGROUND, before the story commit,** where the story's `notes` say so: the code-reviewer agent, and the security-reviewer agent for the stories marked security (US-010a, US-011a, US-010c, US-011b, US-009); the database-reviewer agent for US-010b. Wait for each; fix what it finds before committing. No other story spawns a reviewer.
- **One story per session.** Build the first story whose ACs are not all marked, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep.** **Order is the design** (the JSON's `priority` order is the dependency order, K2).
- **Never push** anything yourself: the runner pushes the branch after each session (a backup to a branch with no PR, so nothing runs) and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine** (no pytest, alembic, uvicorn, npm, npx, vitest, tsc, next, docker, server, browser, installs). Local checks only: `python3 -m py_compile <files>`; after each commit `python3 scripts/ci/check_changed_code_shape.py --base $(bash scripts/ralph/acceptance-prd266w2.sh --print-base)`; `python3 orchestrator/scripts/check_hierarchy_gate.py`; `bash scripts/ralph/acceptance-prd266w2.sh --no-ci`. (`python3` is Homebrew's 3.14 when `/opt/homebrew/bin` leads PATH, as the launcher sets it; the shape script needs 3.10+.) You WRITE the tests each AC names, at the exact paths the ACs give (the gate checks them, and they must be new on this branch); CI runs them once at the end. A test proves the change: it fails without it. A story never waits on CI and never claims a test passed.

## Commits

- `git commit -s` (DCO: every commit), messages `feat(prd-266): description` (or `fix(prd-266):`, `test(prd-266):`, `docs(prd-266):`, `refactor(prd-266):`), **every story commit ending ` [skip ci]`**; explicit paths only (`git add <path> …`); never `git add -A`, `git add .` or `git add -f`; never `git stash`; never amend a pushed commit.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR at completion), NO pushing (the runner pushes `origin feat/prd-266-w2-push`). Never touch `main`, W1's branch or any other branch; never `git checkout`/`switch` away from this branch; never `git cherry-pick`/`revert`/`rebase`.
- NO second migration; NO edit to an existing revision (W1's `prd266_mobile_devices.py` included); NO table altered.
- NO line changed inside `bulk_update_preferences`, `post_background_message`, `requeue_expired_leases`, `get_current_workspace`, `get_request_context_hybrid` or `_stream_response_with_agent_scoped` (the gate compares each with the base); NO touched line inside any function that is still over 50 code lines after your change (rewriting `dispatch` into a short function over helpers is the allowed way); NO growth of a file over 800 lines (the one exception above).
- NO edit to `tests/test_prd128_notification_dispatcher.py`, `tests/test_f197_a_running_out_of_credit_is_said_once.py`, `tests/test_f197_b_a_failed_tasks_report_is_held_in_an_outage.py`, `tests/test_f209_b_a_notice_is_dated_when_it_is_written.py`, `tests/test_prd128_default_prefs.py`, or W1's kit files (`scripts/ralph/*266w1*`).
- NO weakening, skipping or deleting a test or a CI check; NO new dependency; NO third-party client but `services/push/expo.py`; NO `os.getenv` outside config.
- NO push for an event outside `PUSH_TYPE_EVENTS`; NO push from a local install without the operator's answer; NO new public route.
- NO W3 work (content reports, `ai_providers`, the paused-schedule fix, universal-link files, the `/go/pair` page), NO account deletion (D63), NO US-020; NO QR encoder unless K17 reads APPROVED.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet marked.
2. Implement it with its tests; the foreground review its `notes` require; the local checks above; commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text (outside backticks, exactly as written) `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-266w2.json` (US-019c while K17 is not approved: the `→ OWNER:` mark its first AC gives). Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <US-id>` as its last line. **The words RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE are sentinels the runner reads on your last line only (≤ 80 characters): never write them anywhere else in a reply, not even in a sentence about them** (a PRD-256 session did, and the runner ended that wave early).

## Completion

- **All ACs marked:** run `bash scripts/ralph/acceptance-prd266w2.sh --no-ci`, fix what is red in signed ` [skip ci]` commits, then ONE signed commit WITHOUT `[skip ci]` (`chore(prd-266): W2 Push built, run CI`; an empty commit is fine: `git commit -s --allow-empty`), and reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** if every AC is marked and the last commit that touches anything outside `scripts/ralph/` is the wave's CI commit (`chore(prd-266): W2 Push built, run CI`), OR is a commit without `[skip ci]` whose test.yml run on the branch completed green (`gh run list --branch feat/prd-266-w2-push --workflow test.yml --limit 5 --json headSha,status,conclusion`: its headSha, `completed`, `success`; e.g. the owner's fix after a red CI), make NO commit at all (the runner reads that commit's run; a new commit would start a needless extra run): reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO** (a line inside a long function with no seam, a new dependency, a second migration, a W3 piece, a W1 contract that is missing): reply with one line of why and the evidence, then `RALPH_BLOCKED` as the last line (this stops the wave: no PR, no CI). Use it for a real contradiction only; a story that needs a judgement call takes the smaller change and says so in its DONE mark.
