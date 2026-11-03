# Ralph Build Prompt — PRD-270 W2 Cloud bundles (20 stories: US-270-11a … US-270-14f)

You are building **Wave 2 of PRD-270**: Sync with Automatos Cloud brings the approved community agents, Playbooks, packages and plugins of Automatos Cloud to every local install, as item bundles, labelled Reviewed on Automatos Cloud, shown before they are added. It runs unattended, ONE story per session, and every story gets a fresh session.
- **Branch:** `feat/prd-270-w2-cloud-bundles`, in its own worktree, cut from `origin/feat/prd-270-w1-open-marketplace`. The wave is **stacked on W1**: W1's work is already in this branch, and W1's kit files (`*270w1*`) are never edited.
- **One PR** for the whole wave (the PRD is #1377; base W1's branch while W1's PR is open) and **one CI run** at the end. Nothing in a story waits on CI. The PR stays a **draft** until the review passes; you never open, edit, push or undraft it.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd270w2.sh --print-base` prints.
- **The gate reads COMMITTED work** (HEAD), never your working tree: commit, then run it.
- **Never merge anything into this branch.** Integration is the owner's job (K1).
- **This wave HAS ONE migration** (US-270-11a, revision `prd270_w2_cloud_bundles`: one new table, `marketplace_cloud_items`). Every other story adds none.
- **Every item is Reviewed on Automatos Cloud, never Official** (§8-Q3a). Core gets no private key, signs nothing, verifies no bundle signature.
- **Core and web.** No file outside this repo; the Design Kit and Automatos Go need nothing.

## Read first, every iteration

1. `scripts/ralph/prd-270w2.json` is the BINDING contract. Its **`decisions`** (K1–K31) are binding.
2. `docs/PRDS/PRD-270-SYNC-WITH-AUTOMATOS-CLOUD.md`:
   - §4.1 (what exists, by type), §5.1–5.3 (the flow, the hosted catalogue, the local side), §5.6 (Phase 2: all of it), §5.8 (security);
   - §8 (every recommendation is the default: Q1 owners and admins, Q2 the button only, **Q3 (a)**, Q6 on by default);
   - §6's Phase 2 line (US-270-11..15).
3. `CLAUDE.md`, `AGENTS.md`, and for a web story `frontend/DESIGN_SYSTEM.md` §22.
4. **W1's code on this branch: what W1 built is read, never assumed (K27).** Notes mark each W1 seam **RE-VERIFY after W1** and name the W1 story that built it. W1 was still building when this kit was written. If a W1 contract differs from what the kit expects, read it from the code, take the smaller change, and say what differed in the DONE mark.
5. The story's `notes` name the seams by file and line, measured 30 Oct 2026 on `origin/main` `684ea7ef4` (byte-identical with W1's base for every file cited, except config.py, the two reports, router_manifest.py and the two head pins). Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Facts you need (verified 30 Oct 2026 on origin/main 684ea7ef4)

### Code shape and size

- **Files over 800 lines never grow** (K3). Re-measure them on the base:
  - `frontend/lib/api-client.ts` 4119, `orchestrator/main.py` 1672, `orchestrator/api/marketplace.py` 1106, `orchestrator/api/workflow_recipes.py` 1809, `orchestrator/api/admin_plugins.py` 880, `orchestrator/core/models/core.py` 1867, `docs/getting-started/self-hosting.md` 1898, `frontend/components/marketplace/marketplace-plugins-tab.tsx` 898.

  Exceptions: `orchestrator/config.py` (settings lines), `route-manifest.json`, `config-surface.json`. The gate measures every modified file. The two decided moves (K3):
  - the plugins tab's two interfaces go to `components/marketplace/plugin-types.ts`;
  - api-client.ts's gate call and its import replace two comment lines of `request()`.
- **The changed-lines trap.** `scripts/ci/check_changed_code_shape.py` (Python) and `frontend/scripts/check-changed-lines-eslint.js` (TS) fail ANY touched line inside a function over 50 code lines, or a React component over 150. The finding spans the whole function.
  - **Seams that don't count:** a decorator line above `def`; an import line; a pure deletion; a new short function or component; a line in a short function.
  - **Untouched (K4):**
    - the install machinery: `api/marketplace.py`, `api/workflow_recipes.py`, `api/admin_plugins.py`, `api/marketplace_plugins.py`, `api/workspace_plugins.py`, `cascade_installer.py`, `handlers_marketplace.py`, `handlers_packages.py`, `package_installer.py`, `marketplace_packages.py`, the plugin and skill services, `core/models/core.py` and the package, plugin and persona models, every seed;
    - `services/themes/**` and W1's `services/marketplace_cloud/fetch.py`;
    - on the web: the Playbooks, tools, LLMs, skills and agents tab components, `package-detail-modal.tsx`, `marketplace-plugin-detail-modal.tsx`, `SystemSettingsTab.tsx`, the auth surfaces and `middleware.ts`.
  - **Decided seams (K4):** one badge line each in `AgentListRow`, `AgentGridCard`, `PackageCard`, `PluginCard`, `FeaturedPluginCard` and `ItemModalHeader`; one `CloudItemsLayer` mount in `MarketplaceHomepage`; one gate call in `apiClient.request()`; one `cloud_install_asks` call in `confirmation_gate.py`'s `wrapped` (`platform_executor.py`, `tool_grants.py`, `api/approval_grants.py` and the install actions untouched). W1's files at one named seam each, RE-VERIFIED: the build line of `catalog.catalog_snapshot` (`with_items`), `feed.py` (the type dispatch), `pull.py` (the parsed catalogue on its result, only if absent), `api/marketplace_cloud_sync.py` (`run_item_passes`), the status answer (`items`), `core/models/marketplace_cloud.py` (the model), `use-cloud-sync.ts` (one invalidation), `sync-button.tsx` (the status line).
  - **Run the Python script on COMMITTED work only.** Commit, run it, and fix in a follow-up commit if it's red. The TS script needs `node_modules` and runs in CI only: count component (≤ 150) and function (≤ 50) lengths by reading.

### The machine

- **This Mac cannot import the orchestrator.** `route-manifest.json` and `config-surface.json` are edited BY HAND or with a stdlib-only python snippet, in their generated formats (K5).
- **No `next build`, no npm, no vitest, no tsc, no playwright here.** The build proof is the owner's `gh workflow run images.yml --ref feat/prd-270-w2-cloud-bundles` (K25).

### Alembic

- **The head** on the base is W1's `prd270_w1_cloud_sync` (RE-READ `EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py` on the base: if W1 re-chained after merging main, read what it says).
- **The pins to move (K29):** that file (its dated comment chain plus `EXPECTED_HEAD`) and `orchestrator/tests/test_prd236_w1_routes.py` (read the new file, assert its down_revision, assert the guard's `EXPECTED_HEAD`).
- **The revision creates ONE table** (K6): `marketplace_cloud_items`, with its two CHECKs, its unique (type, external_id) and four partial unique indexes. It alters no table and changes no row; its downgrade drops the table only. **Why a table and not W1's `source` column:** K6 (core.py can't grow; `agents` and `workflow_recipes` are the busiest tables; packages and plugins have no `owner_type`).

### The format (K8) and what SaaS exports (K9, K10)

- **One bundle per item**, schema 1, the same bytes on both sides.
  - A strict validator refuses any key outside the spec, at any depth: the allowlist by construction.
  - Canonical bytes: sorted keys, compact separators, UTF-8. The revision is their sha256.
  - Refs are `{slug}`, `{templateId}`, `{item}`, `{name}`, `{modelId}` or `{app}`; a Playbook's steps name its agents by local ids (`a1`…).
- **Never in a bundle:** configuration, schedule_config, an id of any row, an email, a credential, a connection, a path, an install count.
- **Exported:** approved community rows a local edition does not seed itself, the newest per lineage. Plugins only when approved, active and their scan said safe.
- **externalId** is a hash of the lineage (K10): a resubmission on Automatos Cloud is an update, not a new item. A local Cloud row's version column is `<version>+<revision[:12]>`, so a workspace copy records what it was cloned from.

### The hosted side (SaaS only: 404 on a local edition)

- **The catalogue** (K11): W1's `GET /api/marketplace/catalog` gains item entries (schema 1, additive) through ONE `with_items` call in `catalog_snapshot`, so W1's bytes, ETag, 30-second snapshot and size cap cover them. Entries are metadata only: W1's route refuses the whole catalogue, themes included, over `MARKETPLACE_CATALOG_FEED_MAX_KB`.
- **The bundle route** (K12): `GET /api/marketplace/catalog/bundles/{revision}` answers a listed bundle's canonical bytes, immutable, ETag the revision.
- **The plugin package** (K12): `GET /api/marketplace/catalog/plugins/{slug}/{version}/package.zip` answers a deterministic zip of an exported plugin's stored files, read raw from S3.
- **Both** are plain `def`s with no `get_request_context_hybrid`, with SERVING_HEADERS, rate-limited per IP through W1's `api/public_rate_limit.per_ip_limit` with their own buckets, in the route manifest.

### The local side (local edition only)

- **Bundles** (K13): only new or changed entries are fetched, from the DERIVED URL (the configured origin + the path built from the revision), through W1's `fetch.py`, capped, sha256 equal to the revision, validated, the entry's type, id and version matched. An entry claiming Official is refused. An unknown type is skipped, uncounted.
- **Plugins first** (K15), one at a time, each committed on its own (the materializer commits inside):
  - checks: the derived package URL, the upload's zip rules, manifest.json's slug and version, and the STATIC scan here (never the LLM scan: no model call);
  - then: stored with `extract_plugin`; Automatos Cloud's report kept in a `PluginSecurityScan` row; approved on arrival (the local edition has no plugin review page); skills materialised;
  - limits: up to `AUTOMATOS_CLOUD_PLUGINS_PER_SYNC` per Sync.
- **Agents, Playbooks, packages** (K14), in ONE transaction:
  - upserted by their `marketplace_cloud_items` mapping; a row with no mapping is never touched;
  - a name held by a local marketplace agent, or a package slug held locally, refuses the item;
  - refs resolve to local rows; what doesn't resolve is recorded in `unresolved`, except a package member, which refuses the package.
- **The Sync** (K16): W1's POST runs themes, then `run_item_passes` (plugins, items), then W1's auto-update. The catalogue is fetched once. The answer and the status gain `items`.
- **Installs are the existing installs** (K17): unchanged routes, permissions, warnings. A Sync never writes a workspace copy. An update notice appears when a copy's version differs from its Cloud row's and no copy is at the current version, and when a Cloud plugin this workspace enabled has moved to a newer version (by reference, K15).
- **Auto's chat installs** (K17, US-270-14g): when `platform_install_marketplace_agent`, `platform_install_package` or `platform_install_plugin` names a Cloud item, the call runs only on a single-use grant for that exact call. Otherwise it returns the EXISTING ask: a pending tool_call grant (`tool_grants.attach_ask_grant`) whose `question_md` is the disclosure as inert text. Approve re-dispatches the call (`api/approval_grants.py`); Deny runs nothing. Neither the full-autonomy dial nor the owner's own turn skips it; seeded and operator items are as today.

### The web (K19)

- **Data:** `lib/marketplace/cloud-items.ts` and `hooks/use-cloud-items.ts` (local only).
- **Badges:** 'From Automatos Cloud' and 'Reviewed on Automatos Cloud' at the K4 seams. A Playbook's badge is in its detail header (the Playbooks tab is untouchable).
- **The install gate** (`lib/marketplace/cloud-install-gate.ts`), called once at the top of `apiClient.request()`:
  - which calls: exactly the four install calls (agent, Playbook, package, plugin enable); for a Cloud item, it waits for the disclosure;
  - on Add the request goes; on Close it is refused with a plain sentence;
  - everything else, and everything on SaaS, passes unawaited;
  - it imports nothing from `api-client.ts`.
- **The layer** (`components/marketplace/cloud/cloud-items-layer.tsx`), mounted once in `MarketplaceHomepage`, local only: it registers the opener, renders the disclosure dialog and the update notices.

### Tests

- **Backend:**
  - realdb tests use `tests/conftest.py` `new_session(engine)` and a module `engine` fixture (pattern: `tests/test_prd266_auto_message.py:49-61`);
  - the HTTP mock is httpx's `MockTransport` through W1's client-factory seam (W1's `tests/test_prd270w1_cloud_pull.py`; before it, `tests/test_prd261_revocation_pull.py`);
  - S3 is a monkeypatched client (the `tests/test_prd261_theme_files.py:632` pattern);
  - async calls go through the `run_async` seam (`asyncio.run` in tests);
  - the wire probe is `tests/test_prd261_c3_wire.py`'s;
  - every route this wave adds is a GET, so the authz sweep needs no entry (K22).
- **Frontend:** vitest in `__tests__` beside the code. The browser lane's spec is in `e2e/browser/specs/` and needs the `@host` tag to run (K31).
- **Fixtures:** `orchestrator/tests/fixtures/marketplace_bundles/` (four canonical bundles and `cases.json`), fake data only.

## Code rules (AGENTS.md; CI checks them on every line you change)

- **Shape:** functions ≤ 50 code lines and React components ≤ 150; nesting ≤ 4; cyclomatic ≤ 10 on touched code; a new file ≤ 800 lines (aim for 200–400). Type hints and docstrings on public Python functions. No mutation of arguments, and no mutable default arguments.
- **Settings:** only through `orchestrator/config.py`, each new name in `config-surface.json` in sorted position. No `os.getenv` or `os.environ` anywhere else.
- **Routes:**
  - thin (Pydantic or plain answers, one service call, errors mapped);
  - a route that touches the database without awaiting is a plain `def`;
  - local reads filter copies by the caller's workspace;
  - registered through `RouterSpec` (`main.py` untouched);
  - every new route is in `route-manifest.json`.
- **Errors:** `except Exception` must log (`logger.exception`) and either re-raise or fail clearly. Messages are plain sentences that echo nothing a remote sent. No `print` in application code.
- **Web:**
  - `apiClient` only (its generic methods; `lib/api-client.ts` gains no method and does not grow), never a raw `fetch('/api…')`;
  - tokens only, no hex, no `orange-*`, no `rounded-3xl` in new surfaces, no fixed palette colour in a `dark:` class;
  - one ink primary per view; no `innerHTML`; server text rendered as text (a prompt in a whitespace-pre-wrap block, never markdown); nothing auto-plays (K30).
- **Both editions keep working** (`AUTH_EDITION=local` and `saas`): the hosted routes 404 locally, the local routes 404 on SaaS, the web gates by edition, never by role.
- **Untrusted input:** everything in a catalogue or a bundle is data. Its URLs are never fetched (derived URLs only), it is validated before any row is written, and it is never rendered as HTML.
- **Guardrails:** no new Python or npm dependency, no new HTTP client (W1's `fetch.py` only, never edited), no model call, no private key. Never touch the Composio deny list, the generated skill seed, or any guardrail this wave doesn't name.

## The execution contract

- **This session ends when your turn ends.** Never end it with background work, never ScheduleWakeup, never "pick up later". Wait in the foreground for anything you start.
- **A code review runs in the FOREGROUND, before the story commit.**
  - **code-reviewer agent:** on every story.
  - **security-reviewer agent:** on US-270-11b, 11c (the validator's strictness), 12a, 12b (what leaves SaaS), 12c, 12d (the public routes), 13a (SSRF, the derived URL, the revision), 15a (the plugin bytes), 13b, 13c (rows without a mapping untouched), 13d (the order, the one-Sync rule), 14a (local only, workspace-filtered), 14g (Auto's chat installs: an install path) and 14d (the gate: no install without the click, no bypass). K23 lists what each must check.
  - **database-reviewer agent:** on US-270-11a (the migration), 15a (the per-plugin commits and cleanup), 13b and 13c (the one transaction, the unique keys, the upsert).

  Wait for each, and fix what it finds before committing.
- **One story per session.** Build the first story whose ACs are not all marked, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep**, and every W1 seam by reading W1's code. **The order is the design:** the JSON's `priority` order is the dependency order (K2).
- **Never push anything yourself.** The runner pushes the branch after each session and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine:** no pytest, alembic, uvicorn, npm, npx, vitest, tsc, next, playwright, docker, server or installs.

Local checks only:
- `python3 -m py_compile <files>`;
- after each commit, `python3 scripts/ci/check_changed_code_shape.py --base $(bash scripts/ralph/acceptance-prd270w2.sh --print-base)`;
- `python3 orchestrator/scripts/check_hierarchy_gate.py`;
- `bash scripts/ralph/acceptance-prd270w2.sh --no-ci` (on committed work).

You WRITE the tests each AC names, at the exact paths the ACs give; the gate checks they exist and are new on this branch. CI runs them once, at the end. A test proves the change: it fails without it. A story never waits on CI and never claims a test passed.

## Commits

- `git commit -s` on every commit (DCO).
- Messages: `feat(prd-270): description`, or `fix(prd-270):`, `test(prd-270):`, `refactor(prd-270):`, `docs(prd-270):`.
- **Every story commit ends with ` [skip ci]`.**
- Explicit paths only (`git add <path> …`). Never `git add -A`, `git add .` or `git add -f`. Never `git stash`. Never amend a pushed commit.
- **The secrets hook (K24):** never write a path matching `*.env`, `*.env.*`, `*/credentials*`, `*/secrets/*`, `*.pem`, `*.key`, `*_rsa*`, `*/service-account*.json` or `*/sa-key*.json` (never name a new file `credentials*`); never write a key file; fixtures hold fake data only.

## Hard NOs

- **Git:** NO merging, NO PRs, NO pushing, NO `gh pr ready`. Never touch `main`, W1's branch or any other branch. Never `git checkout`/`switch` away from this branch. Never `git cherry-pick`, `revert` or `rebase`.
- **Kits:** NO edit to any other kit's files (`scripts/ralph/*270w1*`, `*268*`, …).
- **Migrations:** NO second migration. NO edit to an existing revision (W1's included). NO `ALTER TABLE`; NO column added to `agents`, `workflow_recipes`, `marketplace_packages`, `marketplace_plugins` or `theme_packages`.
- **Code shape:**
  - NO line changed in the K4 untouched files;
  - NO touched line inside any function still over 50 code lines, or a component over 150, after your change;
  - NO growth of a file over 800 lines, beyond the K3 exceptions.
- **Tests:** NO weakening, skipping or deleting a test or a CI check. The K29 pins change only their literals.
- **Dependencies and config:** NO new dependency, NO new HTTP client, NO edit to W1's `fetch.py`, NO `os.getenv` outside config, NO model call (NEVER `PluginScanService.scan_plugin` or `llm_security_scan`).
- **Security:**
  - NO public route that carries `get_request_context_hybrid`;
  - NO bundle field outside the strict spec;
  - NO URL from a catalogue or a bundle ever fetched;
  - NO row written before its bundle is validated;
  - NO plugin byte stored before its zip, manifest and static checks;
  - NO write to a row without a mapping, or to a workspace copy;
  - NO install of a Cloud item from the web without the disclosure's Add, or from Auto's chat without its approval;
  - NO 'Official' on a bundle item; NO private key, NO signing.
- **Product:**
  - NO Sync on a schedule (the button only, §8-Q2);
  - NO publishing from a local install;
  - NO change to who can Sync (W1's local edition + workspace:manage).
- **Scope (K28):** NO Design Kit or Automatos Go file, NO SaaS version flow. K28's three decisions are settled: build US-270-14g, K15's by-reference plugin update with its notice, and the public bundle routes as written.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet marked.
2. Implement it with its tests, run the foreground reviews it requires and the local checks above, then commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text, outside backticks and exactly as written: `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-270w2.json`. Where a W1 seam differed from the kit, say how. An AC only the owner can check gets `→ OWNER: <what to check>`. Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <US-id>` as its last line.

**The runner reads RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE as sentinels**, on your last line only (≤ 80 characters). Never write them anywhere else in a reply, not even in a sentence about them.

## Completion

- **All ACs marked:**
  1. Run `bash scripts/ralph/acceptance-prd270w2.sh --no-ci`.
  2. Fix what is red in signed ` [skip ci]` commits.
  3. Make ONE signed commit WITHOUT `[skip ci]`: `chore(prd-270): W2 Cloud bundles built, run CI`. An empty commit is fine: `git commit -s --allow-empty`.
  4. Reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** every AC is marked, and the last commit touching anything outside `scripts/ralph/` is either the wave's CI commit or a commit without `[skip ci]` whose test.yml run on the branch completed green. Make NO commit; reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO:** reply with one line saying why, with the evidence, then `RALPH_BLOCKED` as the last line. Use it only for a real contradiction (a W1 seam that cannot take the one change K4 names counts). A story that needs a judgement call takes the smaller change and says so in its DONE mark.
