# Ralph Build Prompt — PRD-266 W3 Store duties, the Automatos Go backend (10 stories: US-016a … US-018d)

You are building **wave W3 (Store duties) of PRD-266**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-266-w3-store-duties`**, in its own worktree, cut from `origin/main` (the PRD is on main since #1257). One PR for the whole wave (base `main`, part of tracking issue #1255), one CI run at the end: nothing in a story waits on CI.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd266w3.sh --print-base` prints.
- **Never merge anything into this branch** (not `main`, not W1's `feat/prd-266-w1-connect`, not a PRD-261 branch). Integration is the owner's job.
- **W1 is being built at the same time on another branch.** Its files do not exist here and this wave never creates or edits them (K5): `services/mobile/`, `api/mobile_devices.py`, `api/well_known.py`, `core/auth/pairing.py`, `core/auth/bearer_registry.py`, `core/auth/device.py`, `core/models/mobile_devices.py`, `components/settings/phones-devices/`, `core/security/rate_limiter.py`, `core/security/ip_bucket.py`, W1's chat files `api/chat.py` and `consumers/chatbot/service.py`, and its voice files `api/voice_retell.py` and `modules/voice/` (US-017b only READS `modules/voice/live_settings.py`).
- **This wave has NO migration** (K4). It changes no authentication, permission or tenancy rule.

## Read first, every iteration

1. `scripts/ralph/prd-266w3.json` is the BINDING contract. Its **`decisions`** are binding: the PRD's decisions (D10–D11, D15, D23, D27, D51, D63), §4.8–4.10, and the kit decisions K1–K20.
2. `docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md` — §3, §4.8 (store duties), §4.9 (fixes), §4.10 (where code goes), and your story's US section (US-016, US-017, US-018). Where the PRD and the kit differ (module names K7, router registration K6, the chat-history AC K3), the kit wins.
3. `CLAUDE.md` and `AGENTS.md`.
4. The story's `notes` name the seams by file and line, measured 20 Oct on 7ae677b52 (origin/main). Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Facts you need (verified on the base, 20 Oct)

- **Files over 800 lines never grow** (§4.10): this wave touches `api/workspaces.py` (985) and `services/activity_service.py` (1,309), and nothing else over 800 lines. `api/workspaces.py` gains its lines only with an equal or larger move out in the same commit (US-017c); `services/activity_service.py` shrinks (US-018a). Never touch `main.py` (1,684), `core/auth/hybrid.py` (954), `api/chat.py` (908), `consumers/chatbot/service.py` (3,074) or `frontend/lib/api-client.ts` (3,921). The one exception: `orchestrator/config.py` gains its setting line, and the two generated reports gain their entries. The gate measures every file.
- **The changed-lines trap.** `scripts/ci/check_changed_code_shape.py` (CI's code-standards lane) fails ANY touched line inside a function over 50 code lines; the finding spans the whole function, and every function whose span holds the line counts. Seams that do not count: a decorator line above `def`; a pure deletion; a new short function; a module constant. Decided seams (K16): `content_reported` joins `VALID_EVENT_TYPES`, a module constant beside `NotificationDispatcher.dispatch` (129 code lines: the gate compares it with the base byte for byte); `EmbeddingManager._load_provider` (59) is never touched (the gate compares it too); `ActivityService._scheduled_task_items` (44) moves out and becomes a delegate; `get_current_workspace` (45) stays ≤ 50; the router registers through **`RouterSpec`** in `orchestrator/router_manifest.py` (K6), so `main.py` never changes. **Run the script on COMMITTED work only** (it diffs merge-base..HEAD but reads spans from the working tree): commit, run it, fix in a follow-up commit if red.
- **This Mac cannot import the orchestrator** (its Python dependencies are not installed; nothing runs here): `route-manifest.json` and `config-surface.json` are edited BY HAND or with a stdlib-only python snippet (K17): routes sorted by `(path, method)`, `route_count` = the number of routes (945 on the base: RE-READ it), written as `json.dumps(manifest, indent=2, sort_keys=True) + "\n"`; setting names in sorted position.
- **The Alembic head** on the base is `clerk_user_relinks_20261019`. This wave adds NO revision and never edits `orchestrator/tests/test_prd209_alembic_single_head.py` or `orchestrator/tests/test_prd236_w1_routes.py` (W1 moves both pins on its branch).
- **Identity:** `core/auth/principal.resolve_user_pk` is the ONE resolver from a request to `users.id`; the local edition's anonymous operator carries its integer PK in `raw_claims["user_id"]` (source `local_operator`). `require_workspace_permission` treats the anonymous local caller as the owner.
- **Writes:** `core.workspaces.audit.AuditService.log` COMMITS its own row and returns `None` (after logging) on failure; `NotificationDispatcher.dispatch` is async and NEVER commits (the caller commits); `check_rate_limit` is async (Redis, fail-open). A route that awaits them is `async def` (the approval-grant routes are the precedent); a route that only touches the session is a plain `def`.
- **The existing outbound webhook** is `channels/drivers/webhook.py` (`get_driver('webhook')`): httpx, 15 s timeout, body `{text, message, ai_generated}`. Its `SendResult.error` can quote the URL (an httpx exception text): never log it verbatim (K10).
- **Frontend:** no `frontend/node_modules` on this machine: `frontend/scripts/check-changed-lines-eslint.js` cannot run here; count component (≤ 150) and function (≤ 50) lengths by reading; CI's Frontend CI runs the checker, tsc and vitest. Tailwind `dark:` palette classes are refused by `frontend/lib/__tests__/dark-variant-tokens.test.ts`: use theme tokens. Every page wears a frame (`app/__tests__/page-frames.test.ts`: `PageHeader`). The real-middleware test pattern is `lib/theme-host/__tests__/files-proxy-middleware.test.ts`. The bell's drift guard is `components/notifications/__tests__/notification-bell-links.test.ts`.
- **Secrets hook (K20):** Write/Edit is refused on any path matching `*.env`, `*.env.*`, `*/credentials*`, `*/secrets/*`, `*/secret/*`, `*.pem`, `*.key`, `*_rsa`, `*/service-account*.json`, `*/sa-key*.json`. No path this wave needs matches. Never write an `.env`/`.env.example` file: document a variable in `docs/deployment-infrastructure/environment-variables.md`. The PRD on this base names `core/auth/credentials.py`: this wave never writes it (or `bearer_registry.py`). Never work around the hook.
- **gitleaks (K19):** no test or doc holds a Slack-webhook-shaped URL (`hooks.slack.com/services/…`) or anything that looks like a real credential; use `https://moderation.example.test/…`.

## Code rules (AGENTS.md; CI checks them on every line you change)

- Functions ≤ 50 code lines, React components ≤ 150, nesting ≤ 4, cyclomatic ≤ 10 on touched code; a new file ≤ 800 lines (aim 200–400). Type hints and docstrings on public Python functions. No mutation of arguments; no mutable default arguments.
- Settings only through `orchestrator/config.py`, each new name in `orchestrator/reports/config-surface.json` in sorted position; no `os.getenv`/`os.environ` anywhere else (ruff TID251); no hardcoded values (named constants or config). The frontend's one new server variable (`ANDROID_APP_SHA256_CERT_FINGERPRINTS`) is read in its route handler, as `app/api/chat/route.ts` reads `BACKEND_INTERNAL_URL`.
- Routes: the new router is thin (Pydantic in, one service call, errors mapped); every query is filtered by the caller's `workspace_id`; another workspace's row is 404, never 403. Every new backend route is in `route-manifest.json`. This wave adds NO unauthenticated backend route; its only public surfaces are the frontend's `/.well-known/apple-app-site-association`, `/.well-known/assetlinks.json` and `/go/pair`, each an exact path in the saas middleware's public list.
- Secrets and personal data: never log `MODERATION_WEBHOOK_URL`, a key, a token or a report's excerpt or note; the moderation post carries no reporter identity and no workspace name; `ai_providers` reads key PRESENCE only (no key is decrypted, logged or returned). Errors: `except Exception` must log (`logger.exception`) and re-raise or fail clearly; no `print` in application code.
- Both editions keep working (`AUTH_EDITION=local` and `saas`); hosted-only behaviour (the moderation post) is gated by `AUTH_EDITION`, never by role.
- The frontend calls the backend only through `apiClient` and the existing hooks (no raw `fetch('/api…')`); `/go/pair` calls nothing at all; no `innerHTML`/`dangerouslySetInnerHTML`; canonical terms (Auto, Mission, Deliverable, Command Center).
- No new Python or npm dependency; no new third-party API client; never touch the Composio deny list, the generated skill seed (`orchestrator/core/seeds/platform-management-skill.md`), `HOST_OWNED_ROUTES` (PRD-261's theme contract) or any guardrail.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work; never ScheduleWakeup; never "pick up later". Wait for anything you start in the foreground.
- **A code review runs in the FOREGROUND, before the story commit,** where the story's `notes` say so: the code-reviewer agent AND the security-reviewer agent for US-016a, US-016b, US-017a, US-017b, US-018c and US-018d. Wait for each; fix what it finds before committing. No other story spawns a reviewer.
- **One story per session.** Build the first story whose ACs are not all DONE, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep.** **Order is the design** (the JSON's `priority` order is the dependency order, K2).
- **Never push** anything yourself: the runner pushes the branch after each session (a backup to a branch with no PR, so nothing runs) and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine** (no pytest, alembic, uvicorn, npm, npx, vitest, tsc, next, docker, server, browser, installs). Local checks only: `python3 -m py_compile <files>`; after each commit `python3 scripts/ci/check_changed_code_shape.py --base $(bash scripts/ralph/acceptance-prd266w3.sh --print-base)`; `python3 orchestrator/scripts/check_hierarchy_gate.py`; `bash scripts/ralph/acceptance-prd266w3.sh --no-ci`. (`python3` is Homebrew's 3.14 when `/opt/homebrew/bin` leads PATH, as the launcher sets it; the shape script needs 3.10+.) You WRITE the tests each AC names, at the exact paths the ACs give (the gate checks them, and they must be new on this branch); CI runs them once at the end. A test proves the change: it fails without it. A story never waits on CI and never claims a test passed.

## Commits

- `git commit -s` (DCO: every commit), messages `feat(prd-266): description` (or `fix(prd-266):`, `test(prd-266):`, `docs(prd-266):`, `refactor(prd-266):`), **every story commit ending ` [skip ci]`**; explicit paths only (`git add <path> …`); never `git add -A`, `git add .` or `git add -f`; never `git stash`; never amend a pushed commit.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR at completion), NO pushing (the runner pushes `origin feat/prd-266-w3-store-duties`). Never touch `main` or any other branch; never `git checkout`/`switch` away from this branch; never `git cherry-pick`/`revert`/`rebase`.
- NO migration of any kind; NO edit to an existing revision or to the two head-pin tests; NO new table.
- NO line changed inside `NotificationDispatcher.dispatch`, `EmbeddingManager._load_provider` or `get_request_context_hybrid` (the gate compares all three with the base); NO touched line inside any function that is still over 50 code lines after your change; NO growth of a file over 800 lines (the one exception above).
- NO W1 file (the list above), NO edit to `api/chat.py` or `consumers/chatbot/service.py` (the chat-history AC is W1's US-007b: K3), NO edit to `core/security/rate_limiter.py` (pass explicit limits: K8), NO edit to `HOST_OWNED_ROUTES` (frontend `lib/theme-host/package-manifest.ts` or `orchestrator/services/themes/contract.py`), NO `next.config.js` change.
- NO building US-019 (Settings → Phones & devices), W2 (push, the Expo sender, `review_pending`/`auto_message`, digests, preferences) or anything of W1 (pairing, the device lane, discovery, remote access). The discovery `report_content` capability and the `/go/pair` exemption in W1's browser gate are merge items (K5), not stories here.
- NO weakening, skipping or deleting a test or a CI check (adding an entry to the bell's drift-guard fixture is not weakening it); NO new dependency; NO third-party client; NO `os.getenv` outside config; NO `.env*` file.
- NO new unauthenticated backend route; NO custom URL scheme; NO API call from `/go/pair`.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet DONE.
2. Implement it with its tests; the foreground review its `notes` require; the local checks above; commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text (outside backticks, exactly as written) `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-266w3.json`. Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <US-id>` as its last line. **The words RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE are sentinels the runner reads on your last line only (≤ 80 characters): never write them anywhere else in a reply, not even in a sentence about them** (a PRD-256 session did, and the runner ended that wave early).

## Completion

- **All ACs DONE:** run `bash scripts/ralph/acceptance-prd266w3.sh --no-ci`, fix what is red in signed ` [skip ci]` commits, then ONE signed commit WITHOUT `[skip ci]` (`chore(prd-266): W3 Store duties built, run CI`; an empty commit is fine: `git commit -s --allow-empty`), and reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** if every AC is DONE and the last commit that touches anything outside `scripts/ralph/` is the wave's CI commit (`chore(prd-266): W3 Store duties built, run CI`), OR is a commit without `[skip ci]` whose test.yml run on the branch completed green (`gh run list --branch feat/prd-266-w3-store-duties --workflow test.yml --limit 5 --json headSha,status,conclusion`: its headSha, `completed`, `success`; e.g. the owner's fix after a red CI), make NO commit at all (the runner reads that commit's run; a new commit would start a needless extra run): reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO** (a line inside a long function with no seam, a new dependency, a migration, a W1 file, a path the secrets hook refuses): reply with one line of why and the evidence, then `RALPH_BLOCKED` as the last line (this stops the wave: no PR, no CI). Use it for a real contradiction only; a story that needs a judgement call takes the smaller change and says so in its DONE mark.
