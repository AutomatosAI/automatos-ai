# Ralph Build Prompt — PRD-259 Skins in Studio, Wave 1 (S-001, S-002, S-003, S-009, S-004a, S-004b, S-010a, S-010b)

You are building **Wave 1 of PRD-259**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-259-w1-skins`**, in its own worktree, cut from `origin/main`. One PR for the whole wave, one CI run at the end: nothing in a story waits on CI.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd259w1.sh --print-base` prints.
- **Never merge anything into this branch** (not `main`, not another wave). Integration is the owner's job.
- **This wave HAS ONE migration** (S-002). Every other story adds none.

## Read first, every iteration

1. `scripts/ralph/prd-259w1.json` is the BINDING contract. Its **`decisions`** are binding: D1–D6 from PRD §10 (Gerard, 14 Oct) and the kit decisions K1–K5 (the base cookie, fonts through FontFace, the private storage prefix, the single-tone lock, pairs on the skin's own values).
2. `docs/PRDS/PRD-259-SKINS-IN-STUDIO.md` (§3 the format, §4 how Studio applies a skin, §5 the tables and routes, §7 radius and dark:, §10 the decisions).
3. `CLAUDE.md` and `AGENTS.md`.
4. The story's `notes` name the seams by file on the base, measured 14 Oct on `origin/main` f2be2abb0. Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Facts you need (verified on the base, 14 Oct)

- **Studio's tokens are HSL triplets** (`--primary: 30 12% 9%`) consumed as `hsl(var(--x))` by `frontend/tailwind.config.ts`. Classic's live in `:root` and `.dark`, Studio's in `.studio` and `.studio.dark` (`frontend/app/globals.css`, 3,398 lines: add only what a story names).
- **The pre-wrapped Studio aliases are derived, never set by a skin:** `--paper, --bg-1, --bg-2, --fg, --fg-2, --fg-3` hold `hsl(...)` literals in the first `.studio` block; a skin's CSS derives them from its own core tokens (S-004a). `--border-2`, `--accent-2` (with an alpha: 0.10 light, 0.14 dark) and `--navy` are allowed skin tokens but are ALSO pre-wrapped in globals.css, so the generator writes them wrapped; `--olive`, `--warn-ink`, `--on-strong` are raw triplets.
- **About 12 files branch on `isStudio` and swap layouts** (app/agents, assignments, chat, command-center, deliverables, playbooks, tools pages; components/documents/document-management.tsx, layout/main-layout.tsx, shared/filter-tabs.tsx, shared/page-header.tsx, shared/stats-bar.tsx). So **a skin's `base` must drive the `studio` class and `isStudio`**: a skin never changes a layout.
- **The server sets the style class** from the `automatos-style` cookie in `frontend/app/layout.tsx` (`parseUiStyle`, `STUDIO_HTML_CLASS`); `frontend/contexts/ui-style-context.tsx` switches it in the browser; `frontend/components/ui/theme-toggle.tsx` is the one dropdown (Style + Tone). Tone is next-themes 0.2.1 (`.dark` on `<html>`, `ThemeProvider` in `components/providers.tsx`, which wraps `UiStyleProvider`).
- **CSP already allows inline style and script** (`frontend/next.config.js`: `script-src 'self' 'unsafe-inline' …`, `style-src 'self' 'unsafe-inline' …`): the pre-paint `<script>` needs no CSP change. The API's own responses carry `nosniff` and `default-src 'none'` (main.py `add_security_headers`).
- **An asset route needs the bearer token**, which a CSS `url()` or `<img src>` cannot send: fetch to an object URL through apiClient's seam (`apiFileUrl(apiClient.getBaseUrl(), path)` + `apiClient.getAuthHeaders()`, the `fetchBrandFileObjectUrl` pattern in `components/documents/blocks/api.ts`, used by `BrandKitFonts.tsx` with the FontFace API). Reuse or extract it; never copy it; never grow `lib/api-client.ts` (3,429 lines).
- **The transport cap:** `orchestrator/main.py` refuses bodies over 10 MB (`MAX_BODY_SIZE`) unless the path is in `PATH_BODY_LIMITS`; the import route needs its entry (S-003).
- **Tailwind is 3.3.3:** `node_modules/tailwindcss/lib/corePlugins.js` `darkVariants` knows only `'class'` and `'media'`, so today's `darkMode: ['variant', …]` adds NO dark variant and every `dark:` class is dead. `['class', '.dark']` gives `:is(.dark &)`. `@tailwindcss/typography` is NOT installed, so every `prose*` class (and `dark:prose-invert`) is inert. Tailwind 3 compiles `@layer base` to plain CSS: specificity decides, not cascade layers.
- **The migration head** on the base is `prd258_voice_timings` (`EXPECTED_HEAD` in `orchestrator/tests/test_prd209_alembic_single_head.py`; RE-VERIFY). Pins to move: that file and `orchestrator/tests/test_prd236_w1_routes.py` (copy the `prd258_voice_timings` lines' pattern).
- **Storage** goes through `core.storage` (`get_s3_client`, `ensure_bucket`) and `config.MARKETPLACE_S3_BUCKET`, exactly as `core/services/marketplace_s3.py` does. Never a new client.
- **woff2 sniffing already exists:** `modules/documents/brand_fonts.check_woff2` (check `.importlinter` before importing it from `services/`).
- **The Play Looks engine to port from** (read-only reference: re-implement in the core's style, never import, never edit): `/Users/gkavanagh/Development/Automatos-AI-Platform/.worktrees/automatos-play/ralph+play-pages-a/lib/looks/{contrast,prepaint,css}.ts` (also `validate.ts`, `types.ts`). If that worktree is gone, the same files are staged read-only at `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/scripts/ralph/prd259-spec/play-looks-ref/`. Play works in `#rrggbb`; Studio in HSL triplets: convert HSL → sRGB before the WCAG maths. Play's mascot states/shells/lights are in its `lib/mascot/types.ts` and `lib/looks/catalogue.ts` (the JSON names them).

## Code rules (AGENTS.md; CI checks them on every line you change)

- Functions ≤ 50 code lines, React components ≤ 150, nesting ≤ 4, complexity ≤ 10 on touched code; a new file ≤ 800 lines (aim 200–400); a file already over 800 lines is not grown except by the lines a story names (`config.py` settings, `main.py` router and body-limit lines, `globals.css` `--radius`).
- **The changed-lines trap (frontend):** `scripts/check-changed-lines-eslint.js` fails on ANY touched line inside a function that is over the limit (the finding spans the whole function: `Message` in `components/chatbot/message.tsx`, long functions in `chat.tsx`). Inside one, only a pure line deletion is safe. Run it before every frontend commit.
- Config only via `orchestrator/config.py`, each new name in `orchestrator/reports/config-surface.json` in sorted position; no `os.getenv`/`os.environ` elsewhere; no hardcoded values (named constants).
- **Routes:** authenticated with `get_request_context_hybrid` and the existing workspace-permission dependency for writes; a route that only touches the database is a plain `def`; every query filtered by the request context's `workspace_id` (tenant isolation); 404, never 403, for another workspace's private skin. Every new route goes in the COMMITTED `orchestrator/reports/route-manifest.json` with its method and the exact FastAPI `route.path`, sorted, `route_count` updated, **edited by hand** (`cd orchestrator && python -m scripts.dump_routes` needs the app's dependencies, which this machine lacks).
- **Migrations:** exactly one new revision (`prd259_skins`), chained onto the base's single head, `CREATE TABLE IF NOT EXISTS` / `CREATE [UNIQUE] INDEX IF NOT EXISTS` (survives a `create_all` database), both head pins moved in the same commit. Never edit an existing revision.
- The frontend calls the backend only through `apiClient` (no `fetch('/api…')`); a binary asset goes through apiClient's `getBaseUrl`/`getAuthHeaders` seam as above. No `innerHTML`; `dangerouslySetInnerHTML` only for the pre-paint constant in `app/layout.tsx`.
- **A skin is data:** the CSS is written by Studio from validated numbers only, never from a string the skin carried. No `@font-face`, no `url()` in the generated CSS.
- Both editions keep working (`AUTH_EDITION=local` and `saas`). No new Python or npm dependency. Canonical terms (AGENTS.md).

## The execution contract

- **This session ends when your turn ends.** Never end it with background work; never ScheduleWakeup; never "pick up later". Wait for anything you start in the foreground.
- **A code review runs in the FOREGROUND, before the commit,** for **S-002** (the migration), **S-003** (uploads, path handling, tenant isolation: use the security-reviewer agent too) and **S-004a** (the CSS generator and the pre-paint script: the injection surface); use the code-reviewer agent and wait for it. No other story spawns a reviewer.
- **One story per session.** Build the first story whose ACs are not all DONE, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep.** **Order is the design** (the JSON's priority order is the dependency order).
- **Never push** anything yourself: the runner pushes the branch after each session (a backup to a branch with no PR, so nothing runs) and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine** (no pytest, npm, npx, vitest, tsc, next build, docker, server, browser, installs, alembic). Local checks only: `python3 -m py_compile <files>`, `ruff check <files>` if on PATH, `python3 orchestrator/scripts/check_hierarchy_gate.py`, `cd frontend && node scripts/check-changed-lines-eslint.js origin/main` when you touched the frontend, and `bash scripts/ralph/acceptance-prd259w1.sh --no-ci`. You WRITE the tests each AC names (`orchestrator/tests/test_prd259_*.py`, `frontend/**/__tests__/*.test.ts(x)`); CI runs them once at the end. A story never waits on CI and never claims a test passed. `→ OWNER:` ACs stay as they are.

## Commits

- `git commit -s` (DCO: every commit), messages `feat(prd-259): description` (or `fix(prd-259):`, `test(prd-259):`, `docs(prd-259):`), **every story commit ending ` [skip ci]`**; explicit paths only (`git add <path> …`); never `git add -A` or `git add .`; never `git stash`; never amend a pushed commit.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR at completion), NO pushing (the runner pushes `origin feat/prd-259-w1-skins`). Never touch `main` or any other branch, never `git checkout`/`switch` away from this branch.
- NO edit under `automatos-play` or its worktrees (read-only reference); NO import from them.
- NO second migration; NO change to an existing revision; NO table altered other than the two new ones.
- NO new boto3/MinIO client; NO `os.getenv`/`os.environ` outside `config.py`; NO hardcoded values; NO new dependency.
- NO weakening, skipping or deleting a test or a CI check; NO growing a function already over the limit.
- NO building Wave 2 (the Marketplace routes and tab, Auto in chat, the colour sweep, docs: `scripts/ralph/prd-259w2.json`), and NO workspace default skin (PRD §10.6 is open: Gerard's call).
- NO edit to the generated seed (`core/seeds/platform-management-skill.md`), the Composio deny list or any guardrail.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet DONE.
2. Implement it with its tests; the local checks above; commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text (outside backticks, exactly as written) `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-259w1.json` (`→ OWNER:` ACs stay). Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <S-id>` as its last line. **The words RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE are sentinels the runner reads on your last line only (≤ 80 characters): never write them anywhere else in a reply, not even in a sentence about them** (a PRD-256 session did, and the runner ended that wave early).

## Completion

- **All ACs DONE** (`→ OWNER:` excepted): run `bash scripts/ralph/acceptance-prd259w1.sh --no-ci`, fix what is red in signed ` [skip ci]` commits, then ONE signed commit WITHOUT `[skip ci]` (`chore(prd-259): Wave 1 built, run CI`; an empty commit is fine: `git commit -s --allow-empty`), and reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** if every AC is DONE and the last commit that touches anything outside `scripts/ralph/` is the wave's CI commit (`chore(prd-259): Wave 1 built, run CI`), make NO commit at all (the runner reads that commit's run; a new commit would start a needless second run): reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO:** reply with one line of why and the evidence, then `RALPH_BLOCKED` as the last line (this stops the wave: no PR, no CI). Use it for a real contradiction only; a story that needs a judgement call takes the smaller change and says so in its DONE mark.
