# Ralph Build Prompt — PRD-255 Brand Kit v2, Wave 1 (the v2 kit, and every renderer reading it)

You are executing **PRD-255 Wave 1**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-255-w1-brand-kit-v2`**, in its own worktree, cut from `main`. Keep the tip green after every commit.

**The thesis (owner, 4 Oct): Automatos becomes your brand.** Every document, spreadsheet and social post a business makes should look like it came from the same studio. Nights 10 and 10b showed the kit is too thin: four colours with no roles, so every renderer paints `primary` on everything ("a lot of orange in there"). This wave makes the kit a role-based system and makes every renderer read it. The standard is restraint: an off-white page, near-black text, one accent used sparingly, strong size contrast, generous space.

**The base is `main`.** The base for every diff and check is what `bash scripts/ralph/acceptance-prd255w1.sh --print-base` prints. **Never merge anything into this branch.** Integration is the owner's job.

Wave 1 builds, in the JSON's priority order:
1. **US-003** the derivation: `derive_palette` / `effective_palette` in `core/brand_palette.py` (pure; every v1 kit reads as v2, a stored role wins).
2. **US-001** the colour roles and `accent_use` in the kit, validated on PUT, reported by GET (set/derived per role).
3. **US-002** the type scale, spacing, margins, logo rules, logo-dark / logo-mono uploads, currency, date style, tone meanings.
4. **US-004** documents (block PDF/DOCX, the legacy Jinja starters) on the tokens; currency, date style, `brand.sign_off`; render tests on real pages.
5. **US-005** spreadsheets, **US-006** socials on the same tokens.
6. **US-008** the agents' tools and rules block carry the whole kit.
7. **US-007** the Brand kit page shows and edits v2.

## Read first, every iteration

1. `scripts/ralph/prd-255w1.json` is the BINDING contract. Its **`decisions`** (the answers to the PRD's §9 open questions) are binding: accent `sparing` by default for EVERY kit; no hosted re-seed job; one default type scale, no presets; no onboarding change. Take the first story (by `priority`) with ACs not yet marked DONE. Each story's `notes` names its anchors and the loop's choices for the PRD's ambiguities: follow them.
2. `docs/PRDS/PRD-255-BRAND-KIT-V2.md` is the spec (the FRs in §4 and the technical notes in §7 bind every story).
3. `CLAUDE.md` and `AGENTS.md`, which it imports: reuse over build, delete what you replace, canonical terms, no `os.getenv` outside `config.py`, no hardcoded values.
4. **F356 may be on the base.** FIXER's F356 (branch `fix/f356-documents-look-professional`, 5 Oct) adds `orchestrator/modules/documents/blocks/design_tokens.py` (a type scale, a 4 pt grid, colours by role — the primary on the title and table headers), `docx_style.py`, `docx_tables.py`, `page_starters.py`, `xlsx_letterhead.py`, `variables/chip_text.py` and restyles the old-style starters. **Grep the base.** If `design_tokens.py` exists, it IS the token mapping: extend it to read the kit (the roles from `effective_palette`, the kit's `type_scale` / spacing / margins / logo rules), delete its now-duplicated constants, and build nothing parallel. If it does not exist, build the mapping as small new functions next to `page_style.py`.

## The code this wave changes (all on main; RE-VERIFY each by grep before building on it)

- `orchestrator/core/brand_palette.py`: `parse_hex`, `to_hex`, `luminance`, `contrast`, `mix`, `_least`, `_most` (the contrast search), `paper_palette`, `stage_palette`, `PAPER_TOKENS`, `STAGE_TOKENS`. **`derive_palette` reuses the contrast search: no second WCAG implementation anywhere.** It is the single source of colour roles for documents, spreadsheets and socials (FR-3).
- `orchestrator/core/media_render_bundle.py`: `brand_tokens` (`COLOUR_TOKENS`, `BODY_FONT_TOKEN`, `HEADING_FONT_TOKEN`, the stage and paper tokens), `_logos`, `build_bundle`. **Keep every existing token name** (`--brand-ink`, `--brand-on-paper`, …): all 18 social templates read them.
- `orchestrator/modules/documents/brand_kit.py` (518 lines): `BrandKit` and `BrandVoice` and `CompanyContact` (`extra="forbid"`), `BrandKitPatch`, `PATCH_FIELDS`, `SERVER_MANAGED_FIELDS`, `get_brand_kit` (lenient read), `validate_brand_kit` (strict write), `update_brand_kit`, `brand_kit_errors`. New models go in a NEW module (e.g. `modules/documents/brand_system.py`; never a `_v2` name), so `brand_kit.py` stays small.
- `orchestrator/api/document_brand_kit.py`: GET/PUT `/brand-kit`, the logo and logo-mark upload/stream/delete helpers (`_store_logo_upload`, `_stream_logo`, `_remove_logo` on a `path_field`).
- `orchestrator/modules/documents/blocks/page_style.py` (`build_styles`: raw primary on headings and table headers today), `blocks/docx_renderer.py`, `blocks/html_renderer.py`, `letterhead.py`, `presets.py` (`LETTERHEAD_LOGO_MM`, the starters' block ids), `seed_templates.py`, `legacy_jinja.py`, `templates/basic_report.html`, `invoice.html`, `executive_summary.html`, `amounts.py`, `variables/catalog.py`, `variables/resolver.py`, `xlsx_render.py`, `thumbnails/render.py` (the F353 page-1 renderer: the render tests reuse it).
- `orchestrator/modules/documents/templates/social/` (18 social templates) and `scripts/ci/social_template_previews.py` (the media-render CI job renders every seeded social template through `hyperframes check`).
- `orchestrator/services/brand_rules.py`: `rules_for_kit`, `_colours_line`, `_logo_line`, `_look_lines`, `_voice_lines`, `brand_assets`, `sign_off_name`.
- `orchestrator/modules/tools/discovery/actions_brand_kit_update.py` (`_PARAMETERS`, `admin_only=True`) and the parity test `orchestrator/tests/test_prd251w1_brand_kit_tools.py` (`set(schema properties) == PATCH_FIELDS`): **every field you add to `BrandKitPatch` goes into `_PARAMETERS` in the same commit, or that test goes red.**
- The frontend Brand kit tab: `frontend/components/deliverables/brand/` (`brand-kit-tab.tsx`, `brand-kit-basics.tsx`, `brand-kit-fields.tsx`, `brand-kit-images.tsx`, `brand-kit-preview.tsx`, `use-brand-kit-form.ts`, `__tests__/`), `frontend/components/documents/blocks/api.ts` and `types.ts`, `frontend/lib/api-client.ts`.

## Code rules (AGENTS.md; CI checks them on every line you change)

- **Functions:** a Python function you touch is at most **50 code lines** and nests at most **4 levels**; a React component at most 150 lines; cyclomatic complexity at most 10 (ruff `C901`). Touching an existing function that is already over a limit means splitting it in the same commit. Where a function is already long, put the new work in a decorator or a new small module rather than growing it.
- **Files:** a new file is at most 800 lines (aim 200–400); never grow a file that is already over 800 lines (PRD §7: `page_style.py`, `docx_renderer.py` and `xlsx_render.py` get the token mapping as small new functions).
- **Config only via `orchestrator/config.py`**, its name in `orchestrator/reports/config-surface.json` in sorted position. No `os.getenv` / `os.environ` anywhere else (ruff `TID251`).
- **No hardcoded values:** every contrast target, tint, default size, bound and limit is a named constant (or config). No colour hex literal in a renderer outside a named constant.
- **No migration.** The kit is JSON on `workspace.settings['brand_kit']`; v2 is derived at read time (FR-2). Adding an Alembic revision is a Hard NO for this wave.
- **Errors:** never swallow one (an `except Exception` logs with `logger.exception` and re-raises, or returns a clear failure); no `print`; validate at the boundary (Pydantic, fail fast, a message that leaks nothing).
- **Tenant isolation:** every read and write is scoped to the caller's workspace.
- **Routes:** every new route is authenticated with the existing request-context and workspace-permission dependencies (`_MANAGE` for writes, as the logo routes); a route that only touches the database is a plain `def` (F105); the upload routes await the file read exactly as `/brand-kit/logo-mark` does. Every new route goes in the COMMITTED `orchestrator/reports/route-manifest.json` with its method, and `route_count` updated.
- **Frontend:** calls only through `apiClient` (raw `fetch('/api…')` is an ESLint error); compact (phone) CSS only in the compact region of `frontend/app/globals.css`; no new npm dependency.
- **Both editions** (local and hosted) keep working; anything hosted-only is gated by `AUTH_EDITION` / `isSaaS`, never by role.
- **Replace, don't shim.** When a token mapping supersedes a raw-colour path, delete the old path in the same commit. No `_legacy`, no `V2` copies.

## The execution contract

- **This session ends when your turn ends.** Nothing reports back later.
  - Never end your turn while anything runs in the background (an agent, a background shell, a `gh run watch`).
  - Never use ScheduleWakeup, and never reply that you will "pick up when it reports".
  - Wait for CI in the foreground: `gh run watch <id> --exit-status`. If the tool call times out, run it again.
  - **A code review runs in the FOREGROUND, before the commit,** only for the stories that touch a gate, an approval, auth or money: **US-002** (new authenticated upload routes) and **US-008** (the agent write path to the kit, `admin_only`). No other story spawns a reviewer.
  - Commit and push as soon as the story's code and tests are written, then fix forward on CI.
- **RE-VERIFY every anchor by grep before building on it.** If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.** Follow `priority` in the JSON.
- **US-004 is large.** If it cannot finish green in one session, commit what is green, push, wait for CI, and end with `STORY_PARTIAL P255-W1-US-004`; the next session continues it. No other story may be partial.

## Tests: CI is the gate

**On the owner's machine nothing runs**: no pytest, npm, tsc, docker, compose, server or browser, and no `pip install` / `npm install`. Its Docker engine belongs to the owner's customer-night stack, and the runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at 127.0.0.1:1. The one local exception is a standard-library script the gate itself runs (e.g. `python3 orchestrator/scripts/check_hierarchy_gate.py`).

**In a Claude Code cloud session** (a disposable container, not the owner's machine), you may install the dependencies there and run the tests and the changed-line checks before pushing. CI stays the only evidence: never mark a story DONE on a local run.

- **You write the tests. CI runs them.** Backend: `orchestrator/tests/test_prd255w1_*.py`. Frontend: vitest files under `frontend/components/deliverables/brand/__tests__/`. **A test must fail without your change.**
- **Render tests are real:** a real PDF rendered by the platform's own path, page 1 rasterised with the F353 renderer (`pypdfium2`, already a dependency), colours and sizes read back. No new dependency.
- **Every story is proven on its own commit:**
  1. `git commit -s`, then push to `origin feat/prd-255-w1-brand-kit-v2`. CI runs on the wave's draft PR (the runner opened it before the first story; `test.yml` runs on pull requests and on main only).
  2. Find the `test.yml` run for that SHA: `gh run list --branch feat/prd-255-w1-brand-kit-v2 --workflow test.yml --limit 10 --json databaseId,headSha,status`.
  3. Wait for it: `gh run watch <id> --exit-status > /dev/null`.
  4. Read the jobs: `gh run view <id> --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'`.
  - A push cancels the previous run on the branch: never push the next story before this one's run has finished.
- **The jobs that must be green:** `orchestrator-tests`, `Alembic from-zero — exactly one head`, `Schema-drift check (four writers)`, `Frontend CI (tsc baselined, vitest, eslint, route-contract)`, `Prod images built with default args carry production behaviour`, `media-render`, `Code standards on changed lines (ruff, function length, nesting)` (marked non-required on pull requests, but required for this run; red is red), and `Which image lanes this change needs`. The two image lanes (`Prod images built…`, `media-render`) run on a PR only when it touches their paths: `skipped` there is fine, anything else must be `success`. This wave touches `core/brand_palette.py`, `core/media_render_bundle.py` and `templates/social/`, which ARE on those paths: expect both lanes to run, and keep them green.
- **If one is red:** read it with `gh run view <id> --log-failed | tail -200` (empty mid-run: use `gh api repos/{owner}/{repo}/actions/jobs/<job id>/logs`), fix it in a new signed commit, and push again. A red job only proves its FIRST failing step: after a fix, read the whole job again. Red that also fails on main's latest run is pre-existing; say so in the commit body. Typecheck: diff tsc's error LIST against main, never the count.
- **When the jobs are green,** mark the story's AC lines `→ DONE — <evidence: the test names, and the CI run id>` in `scripts/ralph/prd-255w1.json`. Commit that alone: `chore(prd-255): <US-id> ACs DONE — CI run <id> green`. (`scripts/ralph` is gitignored, but the kit files are tracked: `git add -f` the JSON by its path.)
- **An AC marked `→ OWNER:` stays as it is.** It is the owner's browser check: never mark it DONE, never claim it, never run a browser.
- **At the START of every iteration,** check the latest `test.yml` run on the branch tip. Red caused by this branch comes first.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO. Messages: `type(prd-255): description` (feat, fix, refactor, test, chore, docs).
- **STAGING DISCIPLINE:** stage explicit paths only. NEVER `git add -A`, `git add .` or `git add -u` (`node_modules` is untracked and NOT gitignored). NEVER `git stash`: the stash stack is shared with every worktree on this machine.
- **Binary files:** only tiny test fixtures (under 100 KB) under `orchestrator/tests/fixtures/`. No rendered output in git.

## Hard NOs

- NO merging, NO PRs (the runner keeps the draft PR; the owner merges), and NO pushing anywhere except `origin feat/prd-255-w1-brand-kit-v2`. Never touch `main`, F356's branch or `test/customer-night`.
- NO Alembic revision (the kit is JSON; FR-2).
- NO renderer reading `primary_color` for an element that has a role (FR-3). NO heading or table header fill in the accent under `sparing` (FR-6).
- NO currency a kit doesn't have (FR-7). NO invented logo variant, NO AI-generated or altered logo (FR-9, non-goals).
- NO type-scale presets, NO hosted re-seed job, NO onboarding change (the decisions).
- NO new font hosting (fonts are the kit's uploads or system / web-safe families). NO print/CMYK output.
- NO weakening, skipping or deleting a test or a CI check. If an existing test asserted the old one-colour look, change its expectation in the same commit and say why in the body.
- NO edit to the generated seed `orchestrator/core/seeds/platform-management-skill.md`, the Composio deny list, or any guardrail.
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py` (tests excepted), NO hardcoded values.
- NO new Python or npm dependency (WeasyPrint stays pinned `<70`, #966).

## Per-iteration protocol

1. Check CI on the branch tip. Red caused by this branch comes first.
2. Pick the first story (by `priority`) with ACs not yet DONE, and re-verify its anchors.
3. Implement it with its tests (and the foreground code review when the story needs one). Commit (signed) and push.
4. Wait for `test.yml` on that SHA, and fix until the required jobs are green.
5. Commit the DONE marks (signed) and push.
6. **STOP.** One story per session. Unless that was the last story, end your reply with one line, `STORY_DONE <US-id>` (or `STORY_PARTIAL P255-W1-US-004`, as above). Do not start the next story.

## Completion

- **All ACs DONE** (the `→ OWNER:` ACs excepted; this session finished the last story): run `bash scripts/ralph/acceptance-prd255w1.sh`. If it exits 0, reply `RALPH_COMPLETE`. If it fails, fix what it names (a new signed commit, CI green) and run it again.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the grep evidence.
