# Ralph Build Prompt — PRD-270 W1 The open Marketplace (19 stories: US-270-00 … US-270-05c)

You are building **Wave 1 of PRD-270**: Sync with Automatos Cloud for themes on every local install, and the public `/marketplace` on SaaS. It runs unattended, ONE story per session, and every story gets a fresh session.
- **Branch:** `feat/prd-270-w1-open-marketplace`, in its own worktree, cut from `origin/main` (which carries #1373, the theme phone previews, and #1377, the PRD). **Not stacked.**
- **One PR** for the whole wave (base `main`; the PRD is PR #1377) and **one CI run** at the end. Nothing in a story waits on CI. The PR stays a **draft** until the review passes; you never open, edit, push or undraft it.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd270w1.sh --print-base` prints.
- **The gate reads COMMITTED work** (HEAD), never your working tree: commit, then run it.
- **Never merge anything into this branch.** Integration is the owner's job (K1).
- **This wave HAS ONE migration** (US-270-00, revision `prd270_w1_cloud_sync`). Every other story adds none.
- **Wave 1 = the PRD's Phase 1 (US-270-01..06) and Part A (US-270-07..10).** Phase 2 (US-270-11..15) is the PRD's own second wave, with its own kit (K28): build none of it.
- **Core and web.** No file outside this repo; the Design Kit and Automatos Go need nothing.

## Read first, every iteration

1. `scripts/ralph/prd-270w1.json` is the BINDING contract. Its **`decisions`** (K1–K31) are binding.
2. `docs/PRDS/PRD-270-SYNC-WITH-AUTOMATOS-CLOUD.md`:
   - §4 (what exists), §5.1–5.5 (the flow, the hosted catalogue, the local side, installing, the UI), §5.7 (the public Marketplace), §5.8 (security);
   - §8 (every recommendation is the default: Q1 owners and admins, Q2 the button only, Q4 local listings, Q6 on by default, Q7 the app's own `/marketplace`, Q8 land on the item with Add ready, Q9 every tab public);
   - your story's line in §6.
3. `CLAUDE.md`, `AGENTS.md`, and for a web story `frontend/DESIGN_SYSTEM.md` §22.
4. The story's `notes` name the seams by file and line, measured 30 Oct 2026 on `origin/main` `66d06cdb8`. Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Facts you need (verified 30 Oct 2026 on origin/main 66d06cdb8)

### Code shape and size

- **Files over 800 lines never grow** (K3). Re-measure them on the base:
  - `frontend/lib/api-client.ts` 4119, `orchestrator/main.py` 1672, `orchestrator/api/marketplace.py` 1106, `docs/getting-started/self-hosting.md` 1898, `orchestrator/tests/test_prd261_theme_models.py` 868.

  Exceptions: `orchestrator/config.py` (settings lines), `route-manifest.json`, `config-surface.json`. The gate measures every modified file.
- **api-client.ts:** new calls go through its generic `get`/`post` (`:1571-1577`) from small modules in `frontend/lib/marketplace/`. A field the server adds to `ThemeListing` is appended to an existing line of the interface (`:1141`).
- **The changed-lines trap.** `scripts/ci/check_changed_code_shape.py` (Python) and `frontend/scripts/check-changed-lines-eslint.js` (TS) fail ANY touched line inside a function over 50 code lines, or a React component over 150. The finding spans the whole function.
  - **Seams that don't count:** a decorator line above `def`; an import line; a pure deletion; a new short function or component; a line in a short function.
  - **Untouchable (K4):** `SystemSettingsTab.tsx`; the seven big Marketplace tab components (`marketplace-{tools,plugins,llms,skills,playbooks,agents,packages}-tab.tsx`); the auth forms and the sign-in, sign-up and SSO pages; `orchestrator/api/marketplace.py`; the revocation modules (`revocation_pull.py`, `revocation_feed.py`, `theme_revocation_scheduler.py`, `api/themes_revocations_feed.py`, `api/admin_theme_revocation_sync.py`).
  - **Decided seams (K4):** `SettingsPanel` gains one tab; `MarketplaceHomepage` gains the Sync button and one deep-link mount; `Providers` gains `ResumeAfterSignIn`; `MarketplaceThemesTab` reads `item`/`add`; `Hero` reads its detail through `listing-source.tsx`; `ListingBadges` gains the Cloud badge; `themeFileUrl` accepts Cloud files URLs; `install_theme_listing` and `update_theme_install` gain one `ensure_stored` call each.
  - **Byte-identical with the base** (the gate compares): `review.approve`, `listings.install_listing`, `listing_updates.offer_listed_update`, `auto_update.apply_listed_update`, `updates.apply_update`, `serve.authorised_package`, `serve.read_stored_file`, `serve.verified_file`, `review_package.rebuilt_package`, `signed_upload.check_upload_signature`, `signature_verify.verify`, `package.read_theme_package`.
  - **Run the Python script on COMMITTED work only.** Commit, run it, and fix in a follow-up commit if it's red. The TS script needs `node_modules` and runs in CI only: count component (≤ 150) and function (≤ 50) lengths by reading.

### The machine

- **This Mac cannot import the orchestrator.** `route-manifest.json` and `config-surface.json` are edited BY HAND or with a stdlib-only python snippet, in their generated formats (K5).
- **No `next build`, no npm, no vitest, no tsc, no playwright here.** The build proof is the owner's `gh workflow run images.yml --ref feat/prd-270-w1-open-marketplace` (K26).

### Alembic

- **The head** on the base is `prd268_w1_proactive` (RE-READ `EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py` on the base: PRD-268 W2, PR #1375, may have landed `prd268_w2_delivery`).
- **The pins to move (K31):** that file (its dated comment chain plus `EXPECTED_HEAD`), `orchestrator/tests/test_prd236_w1_routes.py` (read the new file, assert its down_revision, assert the guard's `EXPECTED_HEAD`), and `orchestrator/tests/test_prd261_theme_models.py` (`MIGRATIONS` at `:85` gains the revision; `_feed_alters` at `:159-160` reads every migration's `_ADD_ORIGIN` and `_ADD_SOURCE`; both IN PLACE: the file stays 868 lines).
- **The revision does two things** (K6): `theme_packages.source` (`VARCHAR(16)`, NULL = local, `'cloud'` only on a Marketplace row, a CHECK) in a guarded DO block named `_ADD_SOURCE`, and `marketplace_cloud_sync` (one row, mirroring `theme_revocation_sync`). Its downgrade deletes no row.

### Pinned tests that move (K31)

Only literals change (plus the one helper line K6 names), and no test function is removed or loosened; the gate checks each file's test count:
- `test_prd209_alembic_single_head.py`, `test_prd236_w1_routes.py`, `test_prd261_theme_models.py`;
- `frontend/lib/__tests__/csp-media-sources.test.ts:52-53` (the bare `img-src` literal gains the Cloud origin).

### The hosted side (SaaS only: 404 on a local edition)

- **The catalogue** (K8): `GET /api/marketplace/catalog`, schema 1, themes only, one item per listed version whose signature verifies, built from `listings.py`'s own functions. Each item carries the `.sig.json` envelope, `files` (the file manifest) and `themeJson` (base64 of theme.json), because a signature covers `{id, version, digest, files}`. Every URL field is a PATH.
- **The files** (K9): `GET /api/marketplace/catalog/files/{digest}/{path:path}` serves `package.autotheme` (`review_package.rebuilt_package`: core stores no zip, and the digest depends on the files alone) and the listed screenshots, nothing else.
- **The public reads** (K10): `GET /api/public/marketplace/{type}` for agents, playbooks, packages, applications, plugins, skills, llms. Each type has its OWN query and a field-by-field allowlist; the member routes' answers leak configuration, steps and a creator's email.
- **All public routes** are plain `def`s with no `get_request_context_hybrid`, cache with `Cache-Control: public, max-age=300` and a weak ETag, and are rate-limited per IP with `MARKETPLACE_PUBLIC_RATE_PER_MINUTE` through `api/provenance.py`'s `RateLimitStore` pattern (K7, K10, K11).

### The local side (local edition only)

- **The pull** (K13, K14): `services/marketplace_cloud/fetch.py` copies `revocation_pull`'s guards (no redirect, caps, timeouts, identity encoding, no credentials). Each item is checked with `check_upload_signature` on a `CheckedThemePackage(manifest, {}, files, digest)`. Accepted items are written in ONE transaction as `source='cloud'` Marketplace rows. A theme id held by a local-origin Marketplace row is skipped whole.
- **Install and update** (K16): `ensure_stored` downloads from the DERIVED URL (the configured origin + the catalogue files path + the digest), never a URL from the catalogue. It checks digest and signature, then stores. The auto-update after a Sync calls `offer_listed_update` + `apply_listed_update`, never `review.approve`.
- **The storefront** (K15): a Cloud row's screenshots load from Automatos Cloud (`cloud_file_url`). `themeFileUrl` accepts them, and the CSP's `img-src` allows the Cloud origin.

### The web

- **Who sees Sync** (K18): `isLocal` and `canManageThemes(role)` and the status says enabled. Settings › Automatos Cloud is a SettingsPanel tab, local only.
- **`/marketplace` public** (K19): exactly one path added to `isPublicRoute`. The page becomes a server component with a client switch: signed out on SaaS → the visitor view; anything else → today's page, unchanged.
- **The visitor storefront** (K20): reuses `Storefront` through a listing-source context whose default is today's hooks.
- **Add while signed out** (K21): sessionStorage return path (only `/marketplace…`), then `/sign-in?redirect_url=…`. `ResumeAfterSignIn` in Providers. The auth pages and forms are untouched.
- **Back with `?item=&add=1`** (K22): themes open the DetailDialog; the other tabs through ONE deep-link component.
- **Share cards** (K23): `generateMetadata` in `app/marketplace/page.tsx`, and `app/robots.ts`.

### Tests

- **Backend:**
  - realdb tests use `tests/conftest.py` `new_session(engine)` and a module `engine` fixture (pattern: `tests/test_prd266_auto_message.py:49-61`);
  - signed packages and keys come from `tests/helpers_theme_publish.py` (`keys_file`, `envelope`, `package`; the RFC 8032 vector);
  - the HTTP mock is httpx's `MockTransport` through the client-factory seam (`tests/test_prd261_revocation_pull.py`);
  - the wire probe is `tests/test_prd261_c3_wire.py`'s;
  - every new mutating route carries `require_workspace_permission`, so the authz sweep classifies it (K11).
- **Frontend:** vitest in `__tests__` beside the code; the middleware test pattern is `lib/theme-host/__tests__/files-proxy-middleware.test.ts`; the browser lane's specs are in `e2e/browser/specs/` and need the `@host` tag to run (K30).

## Code rules (AGENTS.md; CI checks them on every line you change)

- **Shape:** functions ≤ 50 code lines and React components ≤ 150; nesting ≤ 4; cyclomatic ≤ 10 on touched code; a new file ≤ 800 lines (aim for 200–400). Type hints and docstrings on public Python functions. No mutation of arguments, and no mutable default arguments.
- **Settings:** only through `orchestrator/config.py`, each new name in `config-surface.json` in sorted position. No `os.getenv` or `os.environ` anywhere else.
- **Routes:**
  - thin (Pydantic in, one service call, errors mapped);
  - a route that touches the database without awaiting is a plain `def`;
  - every query is filtered as its story says (a workspace's row is never public; the local routes are the caller's workspace's);
  - registered through `RouterSpec` (`main.py` untouched);
  - every new route is in `route-manifest.json`.
- **Errors:** `except Exception` must log (`logger.exception`) and either re-raise or fail clearly. Messages are plain sentences that echo nothing a remote sent. No `print` in application code.
- **Web:**
  - `apiClient` only (its generic methods; `lib/api-client.ts` doesn't grow), never a raw `fetch('/api…')`; the share-card read is server-side and uses the files proxy's backend address (K23);
  - tokens only, no hex, no `orange-*`, no `rounded-3xl` in new surfaces, no fixed palette colour in a `dark:` class;
  - one ink primary per view; no `innerHTML`; nothing auto-plays (K29).
- **Both editions keep working** (`AUTH_EDITION=local` and `saas`): hosted routes 404 locally, local routes 404 on SaaS, the web gates by edition, never by role.
- **Untrusted input:** everything in a catalogue is data. Its URLs are never fetched (K16), its text is checked and rendered as text, and its signature is verified before anything is listed.
- **Guardrails:** no new Python or npm dependency, no new HTTP client (httpx only, in `services/marketplace_cloud/fetch.py`), no model call. Never touch the Composio deny list, the generated skill seed, or any guardrail this wave doesn't name.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work, never ScheduleWakeup, never "pick up later". Wait in the foreground for anything you start.
- **A code review runs in the FOREGROUND, before the story commit.**
  - **code-reviewer agent:** on every story.
  - **security-reviewer agent:** on US-270-01, US-270-02, US-270-07a, US-270-07b (what the public routes leave out), US-270-03a, US-270-03b (the pull and its routes), US-270-04a, US-270-04b (the download and the installs it moves), US-270-08a (the middleware), US-270-09a (no open redirect) and US-270-09b (nothing installs without the click). K24 lists what each must check.
  - **database-reviewer agent:** on US-270-00 (the migration), US-270-03a (the one transaction) and US-270-04b (the update moves).

  Wait for each, and fix what it finds before committing.
- **One story per session.** Build the first story whose ACs are not all marked, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep.** **The order is the design:** the JSON's `priority` order is the dependency order (K2).
- **Never push anything yourself.** The runner pushes the branch after each session and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine:** no pytest, alembic, uvicorn, npm, npx, vitest, tsc, next, playwright, docker, server or installs.

Local checks only:
- `python3 -m py_compile <files>`;
- after each commit, `python3 scripts/ci/check_changed_code_shape.py --base $(bash scripts/ralph/acceptance-prd270w1.sh --print-base)`;
- `python3 orchestrator/scripts/check_hierarchy_gate.py`;
- `bash scripts/ralph/acceptance-prd270w1.sh --no-ci` (on committed work).

You WRITE the tests each AC names, at the exact paths the ACs give; the gate checks they exist and are new on this branch. CI runs them once, at the end. A test proves the change: it fails without it. A story never waits on CI and never claims a test passed.

## Commits

- `git commit -s` on every commit (DCO).
- Messages: `feat(prd-270): description`, or `fix(prd-270):`, `test(prd-270):`, `refactor(prd-270):`, `docs(prd-270):`.
- **Every story commit ends with ` [skip ci]`.**
- Explicit paths only (`git add <path> …`). Never `git add -A`, `git add .` or `git add -f`. Never `git stash`. Never amend a pushed commit.
- **The secrets hook (K25):** never write a path matching `*.env`, `*.env.*`, `*/credentials*`, `*/secrets/*`, `*.pem`, `*.key`, `*_rsa*`, `*/service-account*.json` or `*/sa-key*.json`; never write a key file; tests sign with `tests/helpers_theme_publish.py`'s vector.

## Hard NOs

- **Git:** NO merging, NO PRs, NO pushing, NO `gh pr ready`. Never touch `main` or any other branch. Never `git checkout`/`switch` away from this branch. Never `git cherry-pick`, `revert` or `rebase`.
- **Kits:** NO edit to any other kit's files (`scripts/ralph/*268*`, `*266*`, …).
- **Migrations:** NO second migration. NO edit to an existing revision. NO schema change beyond K6's two.
- **Code shape:**
  - NO line changed inside the byte-identical functions (K4), nor in the untouchable files K4 names;
  - NO touched line inside any function still over 50 code lines, or a component over 150, after your change;
  - NO growth of a file over 800 lines, beyond the K3 exceptions.
- **Tests:** NO weakening, skipping or deleting a test or a CI check. The K31 pins change only their literals (and the one helper line).
- **Dependencies and config:** NO new dependency, NO new HTTP client, NO `os.getenv` outside config, NO model call.
- **Security:**
  - NO public route that carries `get_request_context_hybrid`;
  - NO field outside its allowlist on a public answer;
  - NO URL from a catalogue ever fetched;
  - NO byte stored before its digest and signature check;
  - NO second path in `isPublicRoute`;
  - NO return path but `/marketplace…`.
- **Product:**
  - NO Sync on a schedule (the button only, §8-Q2);
  - NO write to a local-origin row from a Sync;
  - NO `review.approve` call for a Cloud row;
  - NO publishing from a local install.
- **Scope (K28):** NO Phase 2 work (item bundles, export or import of agents, Playbooks, packages or plugins), NO Design Kit or Automatos Go file.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet marked.
2. Implement it with its tests, run the foreground reviews it requires and the local checks above, then commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text, outside backticks and exactly as written: `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-270w1.json`. An AC only the owner can check gets `→ OWNER: <what to check>`. Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <US-id>` as its last line.

**The runner reads RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE as sentinels**, on your last line only (≤ 80 characters). Never write them anywhere else in a reply, not even in a sentence about them.

## Completion

- **All ACs marked:**
  1. Run `bash scripts/ralph/acceptance-prd270w1.sh --no-ci`.
  2. Fix what is red in signed ` [skip ci]` commits.
  3. Make ONE signed commit WITHOUT `[skip ci]`: `chore(prd-270): W1 open Marketplace built, run CI`. An empty commit is fine: `git commit -s --allow-empty`.
  4. Reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** every AC is marked, and the last commit touching anything outside `scripts/ralph/` is either the wave's CI commit or a commit without `[skip ci]` whose test.yml run on the branch completed green. Make NO commit; reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO:** reply with one line saying why, with the evidence, then `RALPH_BLOCKED` as the last line. Use it only for a real contradiction. A story that needs a judgement call takes the smaller change and says so in its DONE mark.
