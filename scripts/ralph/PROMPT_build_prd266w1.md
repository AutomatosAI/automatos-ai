# Ralph Build Prompt — PRD-266 W1 Connect, the Automatos Go backend (16 stories: US-003a … US-008b)

You are building **wave W1 (Connect) of PRD-266**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-266-w1-connect`**, in its own worktree, cut from `origin/docs/prd-266-automatos-go-backend-2026-10-20` (the docs branch of PR #1257, so the PRD is in the tree). One PR for the whole wave (base `main`, tracking issue #1253), one CI run at the end: nothing in a story waits on CI.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd266w1.sh --print-base` prints.
- **Never merge anything into this branch** (not `main`, not another wave, not a PRD-261 branch). Integration is the owner's job.
- **This wave changes authentication, permissions and tenancy.** Gerard approved those changes on 20 Oct (tracking issue #1253; the JSON's `decisions`). Build exactly what the ACs say; a change to auth that no AC asks for is out of scope.
- **This wave HAS ONE migration** (US-002, revision `prd266_mobile_devices`). Every other story adds none.

## Read first, every iteration

1. `scripts/ralph/prd-266w1.json` is the BINDING contract. Its **`decisions`** are binding: the PRD's §3 decisions (D1 … D53), §4.1–4.3, §4.10–4.11 and §8, and the kit decisions K1–K16.
2. `docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md` — §3, §4.1 (discovery), §4.2 (the table), §4.3 (pairing, the device lane, remote access), §4.6 (editors), §4.7 (chat and voice), §4.10 (where code goes), §4.11 (enterprise alignment), and your story's US section.
3. `CLAUDE.md` and `AGENTS.md`.
4. The story's `notes` name the seams by file and line, measured 20 Oct on b5628af98. Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Facts you need (verified on the base, 20 Oct)

- **Files over 800 lines never grow** (§4.10): `core/auth/hybrid.py` 954, `api/chat.py` 908, `api/approval_grants.py` 806, `main.py` 1,681, `services/cli_host_service.py` 2,799, and the others this wave may touch: `consumers/chatbot/service.py` 3,074, `frontend/lib/api-client.ts` 3,847. They gain only a call or an import; where that would grow them, something moves out in the same commit. The one exception: `orchestrator/config.py` gains its settings lines, and the generated reports gain their entries. The gate measures every file.
- **The changed-lines trap.** `scripts/ci/check_changed_code_shape.py` (CI's code-standards lane) fails ANY touched line inside a function over 50 code lines; the finding spans the whole function, and every function whose span holds the line counts (a nested callback counts against its parent). Seams that do not count: a decorator line above `def`; a pure deletion; a new short function; a hook in a small callee. Decided seams (K3): the device lane enters through a **decorator** on `get_request_context_hybrid` (136 code lines; never touch a line inside it: the gate compares its source with the base's); routers register through **`RouterSpec`** in `orchestrator/router_manifest.py` (as `api.cli_hosts` and `api.provenance` do), so `main.py` never changes; the agent-info fix lands in `consumers/chatbot/streaming.py`, never inside `_stream_response_with_agent_scoped` (365 code lines; the gate compares it too); `mint_web_call` (137 code lines) is rewritten as a short route over helpers. **Run the script on COMMITTED work only** (it diffs merge-base..HEAD but reads spans from the working tree): commit, run it, fix in a follow-up commit if red.
- **This Mac cannot import the orchestrator** (its Python dependencies are not installed; nothing runs here): `route-manifest.json` and `config-surface.json` are edited BY HAND or with a stdlib-only python snippet (K4): routes sorted by `(path, method)`, `route_count` = the number of routes, written as `json.dumps(manifest, indent=2, sort_keys=True) + "\n"` (the format of `scripts/dump_routes.py`); setting names in sorted position.
- **The Alembic head** on the base is `clerk_user_relinks_20261019` (`EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py`; RE-READ it). The two pins to move: that file and `orchestrator/tests/test_prd236_w1_routes.py` (copy the `clerk_user_relinks_20261019` lines' pattern).
- **The local edition's identity today:** every request without a credential is the anonymous operator, `super_admin`, bound to the operator's `users` row with `id` = the EMAIL and the integer PK in `raw_claims["user_id"]` (`hybrid._local_operator_user_context`). `core/auth/principal.resolve_user_pk` is the ONE resolver from a request to `users.id`; it trusts `raw_claims.user_id` only for `source == "local_operator"` (US-004a adds `"mobile_device"`).
- **The pairing precedent** is the CLI host lane: `services/cli_host_service.py` (`hash_secret`, `_new_pairing_code`, `PAIRING_CODE_TTL_SECONDS = 600`, `create_pairing_code`, `pair_host`, `resolve_host_by_token`) and `api/cli_hosts.py` (`/pairing-codes`, `/pair` with `check_rate_limit("instance", "cli_host_pair")`). The public, per-IP rate-limited route precedent is `api/provenance.py` (`get_client_ip`, `rate_bucket_for`, `RateLimitStore`).
- **`core/security/rate_limiter.check_rate_limit` is async.** A route stays a plain `def` when the limiter runs in an async dependency (`dependencies=[Depends(...)]`).
- **Compose:** the API's service is `backend` (not `api`), the web app's `frontend`; every published port is `${BIND_ADDRESS:-127.0.0.1}` today and `orchestrator/tests/test_prd234_s1a_compose_bind.py` enforces it on every compose file (K10 says how it changes). `AUTH_EDITION=local` comes from `envs/api.defaults` (the backend's env_file); `environment:` overrides it.
- **`config.PUBLIC_API_HOST` defaults to `api.automatos.app`** (a host, no scheme); `config.FRONTEND_URL` defaults to `http://localhost:3000`; `config.CLERK_PUBLISHABLE_KEY` reads `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` (K15).
- **The frontend's one token seam** is `apiClient.setClerkTokenGetter` (`lib/api-client.ts:1236`); the local edition installs a getter that returns `null` in `components/local-auth-provider.tsx`. Next.js proxies: `app/api/chat/route.ts` forwards the browser's Authorization; `app/api/themes/files/[...path]/route.ts` sends none in the local edition (K8).
- **No QR encoder exists** in the repo (K9): show the code, the countdown and the link as text.
- **PRD-261 W-M1** (`GET /api/themes/mobile/{install_id}`) is not on this base (K12).
- **No `frontend/node_modules` on this machine:** `frontend/scripts/check-changed-lines-eslint.js` needs eslint and cannot run here. Count component (≤ 150) and function (≤ 50) lengths by reading; CI's Frontend CI runs the checker. Tailwind `dark:` palette classes are refused by `frontend/lib/__tests__/dark-variant-tokens.test.ts`: use theme tokens.

## Code rules (AGENTS.md; CI checks them on every line you change)

- Functions ≤ 50 code lines, React components ≤ 150, nesting ≤ 4, cyclomatic ≤ 10 on touched code; a new file ≤ 800 lines (aim 200–400). Type hints and docstrings on public Python functions. No mutation of arguments; no mutable default arguments.
- Settings only through `orchestrator/config.py`, each new name in `orchestrator/reports/config-surface.json` in sorted position; no `os.getenv`/`os.environ` anywhere else (ruff TID251); no hardcoded values (named constants or config).
- Routes: new routers are thin (Pydantic in, one service call, errors mapped); a route that touches the database without awaiting is a plain `def`; every query is filtered by the caller's `workspace_id`; another workspace's row is 404, never 403. Every route the frontend calls is in `route-manifest.json`. The ONLY unauthenticated routes this wave adds are `GET /.well-known/automatos.json` and `POST /api/v1/mobile/pair` (AGENTS.md exception 1; each module's docstring names it).
- Secrets: pairing codes, device tokens and their previous hashes are stored ONLY as hashes (the kernel's `hash_secret`), compared ONLY in constant time, returned ONLY once; `push_key` is returned only on first registration (K14). Never log a code, a token, a hash or a push key. Errors: `except Exception` must log (`logger.exception`) and re-raise or fail clearly; no `print` in application code (scripts may print).
- Both editions keep working (`AUTH_EDITION=local` and `saas`); hosted-only or local-only behaviour is gated by `AUTH_EDITION`, never by role. Identity is a credential, never an address.
- The frontend calls the backend only through `apiClient` (no raw `fetch('/api…')`); no `innerHTML`/`dangerouslySetInnerHTML`; canonical terms (Auto, Mission, Deliverable, Command Center).
- No new Python or npm dependency; no new third-party API client; never touch the Composio deny list, the generated skill seed (`orchestrator/core/seeds/platform-management-skill.md`) or any guardrail.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work; never ScheduleWakeup; never "pick up later". Wait for anything you start in the foreground.
- **A code review runs in the FOREGROUND, before the story commit,** where the story's `notes` say so: the code-reviewer agent, and the security-reviewer agent for the stories marked security (US-003b, US-004a, US-003c, US-004b, US-005a, US-005b, US-006a); the database-reviewer agent for US-002. Wait for each; fix what it finds before committing. No other story spawns a reviewer.
- **One story per session.** Build the first story whose ACs are not all DONE, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep.** **Order is the design** (the JSON's `priority` order is the dependency order, K2).
- **Never push** anything yourself: the runner pushes the branch after each session (a backup to a branch with no PR, so nothing runs) and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine** (no pytest, alembic, uvicorn, npm, npx, vitest, tsc, next, docker, server, browser, installs). Local checks only: `python3 -m py_compile <files>`; after each commit `python3 scripts/ci/check_changed_code_shape.py --base $(bash scripts/ralph/acceptance-prd266w1.sh --print-base)`; `python3 orchestrator/scripts/check_hierarchy_gate.py`; `bash scripts/ralph/acceptance-prd266w1.sh --no-ci`. (`python3` is Homebrew's 3.14 when `/opt/homebrew/bin` leads PATH, as the launcher sets it; the shape script needs 3.10+.) You WRITE the tests each AC names, at the exact paths the ACs give (the gate checks them, and they must be new on this branch); CI runs them once at the end. A test proves the change: it fails without it. A story never waits on CI and never claims a test passed.

## Commits

- `git commit -s` (DCO: every commit), messages `feat(prd-266): description` (or `fix(prd-266):`, `test(prd-266):`, `docs(prd-266):`, `refactor(prd-266):`), **every story commit ending ` [skip ci]`**; explicit paths only (`git add <path> …`); never `git add -A`, `git add .` or `git add -f`; never `git stash`; never amend a pushed commit.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR at completion), NO pushing (the runner pushes `origin feat/prd-266-w1-connect`). Never touch `main` or any other branch; never `git checkout`/`switch` away from this branch; never `git cherry-pick`/`revert`/`rebase`.
- NO second migration; NO edit to an existing revision; NO table altered other than the new `mobile_devices`.
- NO line changed inside `get_request_context_hybrid` or `_stream_response_with_agent_scoped` (the gate compares both with the base); NO touched line inside any function that is still over 50 code lines after your change (rewriting one into a short function over helpers, as US-008a does for `mint_web_call`, is the allowed way); NO growth of a file over 800 lines (the one exception above).
- NO edit to the CLI-host tests US-003a lists, except `test_prd234_s1a_compose_bind.py`'s port check in US-005a (K10).
- NO weakening, skipping or deleting a test or a CI check; NO new dependency; NO third-party client; NO `os.getenv` outside config.
- NO building W2 or W3 (push: the `push` destination, the Expo sender, fan-out, digests, preferences; content reports; `ai_providers`; the QR image; `/go/pair`; universal-link files) and NO US-020 (the `azp` marker on #1216).
- NO new public route beyond the two named; NO trust in a source address; NO custom URL scheme.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet DONE.
2. Implement it with its tests; the foreground review its `notes` require; the local checks above; commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text (outside backticks, exactly as written) `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-266w1.json`. Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <US-id>` as its last line. **The words RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE are sentinels the runner reads on your last line only (≤ 80 characters): never write them anywhere else in a reply, not even in a sentence about them** (a PRD-256 session did, and the runner ended that wave early).

## Completion

- **All ACs DONE:** run `bash scripts/ralph/acceptance-prd266w1.sh --no-ci`, fix what is red in signed ` [skip ci]` commits, then ONE signed commit WITHOUT `[skip ci]` (`chore(prd-266): W1 Connect built, run CI`; an empty commit is fine: `git commit -s --allow-empty`), and reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** if every AC is DONE and the last commit that touches anything outside `scripts/ralph/` is the wave's CI commit (`chore(prd-266): W1 Connect built, run CI`), OR is a commit without `[skip ci]` whose test.yml run on the branch completed green (`gh run list --branch feat/prd-266-w1-connect --workflow test.yml --limit 5 --json headSha,status,conclusion`: its headSha, `completed`, `success`; e.g. the owner's fix after a red CI), make NO commit at all (the runner reads that commit's run; a new commit would start a needless extra run): reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO** (a line inside a long function with no seam, a new dependency, a second migration, a W2 piece): reply with one line of why and the evidence, then `RALPH_BLOCKED` as the last line (this stops the wave: no PR, no CI). Use it for a real contradiction only; a story that needs a judgement call takes the smaller change and says so in its DONE mark.
