# Ralph Build Prompt — PRD-255 Brand Kit v2, Wave 2 (the brand board, the Brand designer, Auto's brand routing)

You are executing **PRD-255 Wave 2**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-255-w2-brand-board-designer`**, in its own worktree. Keep the tip green after every commit.

**The thesis (owner, 4 Oct): Automatos becomes your brand; Auto is the one voice and delegates, agents do the work.** Wave 1 made the kit a role-based system every renderer reads. This wave shows the kit as one page (the brand board), and gives the owner a designer on the team: Auto files the brand ask to the Brand designer, who proposes, renders, LOOKS at the result, revises, and changes the kit only after the owner approves.

**The base is Wave 1.** While Wave 1 is not on `main`, this branch is **stacked on** `feat/prd-255-w1-brand-kit-v2`; once it is, the branch is cut from `main`.
- The base for every diff and check is what `bash scripts/ralph/acceptance-prd255w2.sh --print-base` prints.
- **Never merge anything into this branch**: not `main`, not the Wave 1 branch. Integration is the owner's job.

Wave 2 builds, in the JSON's priority order:
1. **US-009** the Brand Board block starter (PDF/DOCX) and the social "Brand board" (4:5, 9:16), from the kit only.
2. **US-010** the board on the Brand kit page, with PDF and PNG downloads.
3. **US-012** `render_preview`: an agent renders a page into its session folder and opens it.
4. **US-013** `create_template` / `update_template` through the studio's own validation.
5. **US-011** the Brand designer, seeded per workspace (it uses the tools above).
6. **US-014** Auto routes a brand ask to the designer; the propose → approve → save → sample set flow.

## Read first, every iteration

1. `scripts/ralph/prd-255w2.json` is the BINDING contract. Its **`decisions`** are binding: accent `sparing` by default; no hosted re-seed job (the Brand Board reaches workspaces through the existing starter-refresh rule); **the designer does the kit plus DOCUMENT templates only** (the template tools refuse social formats); no onboarding change; one default type scale. Take the first story (by `priority`) with ACs not yet marked DONE. Each story's `notes` names its anchors and the loop's choices for the PRD's ambiguities: follow them.
2. `docs/PRDS/PRD-255-BRAND-KIT-V2.md` is the spec (FR-9..FR-12 bind this wave).
3. `CLAUDE.md` and `AGENTS.md`, which it imports: reuse over build, delete what you replace, canonical terms (Deliverable, Task, Auto), no `os.getenv` outside `config.py`, no hardcoded values, data lives in the database.
4. **What Wave 1 built** (grep the base to confirm): `derive_palette` / `effective_palette` in `core/brand_palette.py`; the v2 `BrandKit` fields (roles, `accent_use`, `type_scale`, spacing, `logo_rules`, `logo_dark_path` / `logo_mono_path`, `currency`, `date_style`, tone meanings) in `modules/documents/brand_kit.py` and its models module; the renderers on the tokens; the tools and the rules block carrying the whole kit. Read the base's code, not this summary, before using a name.

## The code this wave builds on (RE-VERIFY each by grep before building on it)

- Starters and blocks: `orchestrator/modules/documents/presets.py`, `seed_templates.py` (the seeder and its refresh rule), `blocks/schema.py`, `blocks/validation.py`, `blocks/html_renderer.py`, `blocks/docx_renderer.py`, `blocks/page_style.py`, `social_starters.py`, `templates/social/` (the media-render CI job renders every seeded social template: `scripts/ci/social_template_previews.py`).
- Rendering a page: `orchestrator/modules/documents/thumbnails/render.py` (`pdf_first_page_png`, `render_png_isolated`: the render in a child process under a time limit), `thumbnails/job.py` (`load_output`), `template_preview.py`, `orchestrator/api/deliverable_thumbnails.py` (the workspace-scoped streaming pattern).
- The session's folder: `orchestrator/modules/tools/execution/session_document_folder.py` (F349: `sessions/<ticket>/<file>` through `WorkspaceClient.write_binary`; the ticket from the server-built context, `session_task_id`).
- Session tools: `orchestrator/api/session_tools.py`, `services/session_tools.py`, `services/session_tool_groups.py` (`SESSION_TOOL_GROUPS`; the documents group).
- Platform tools: `orchestrator/modules/tools/README.md` (the 3-file pattern), `modules/tools/discovery/actions_templates.py`, `template_tools.py`, `actions_asks.py` (`platform_ask_human`), `actions_brand_kit_update.py`, `platform_executor.py` (the routing table), `orchestrator/scripts/check_hierarchy_gate.py`.
- Templates: `orchestrator/api/document_generation.py` (POST/PUT `/templates`: `_validate_blocks_or_422`, `is_social_format`, `SOCIAL_TEMPLATE_ERRORS`), `modules/documents/template_service.py`, `template_summary.py` (`STARTER_CREATOR = "system"`), `core/models/core.py` `DocumentTemplate` (category, tags, created_by: no migration).
- Agents and Auto: `orchestrator/core/seeds/seed_auto_agent.py` (the per-workspace seed pattern), `core/seeds/seed_socials_package.py` (`_BRAND_DESIGNER_PERSONA`, the Socials package's "Brand Designer": REUSE it, one persona source, never a duplicate agent), `core/seeds/skills/brand-kit-builder.md`, `core/cli_runtime.py` (runtime validation), `orchestrator/config.py` (`CLI_RUNTIME_ENABLED`: local edition only), `consumers/chatbot/paperwork_to_the_team.py` (F337(c), the routing-note pattern) and its wiring in `consumers/chatbot/service.py`, `core/models/approval_grants.py` (`KIND_QUESTION`; always pass the kind, F259).
- The Brand kit tab: `frontend/components/deliverables/brand/`, `frontend/lib/api-client.ts`, the authenticated blob hook (`useAuthenticatedBlobUrl`).

## Code rules (AGENTS.md; CI checks them on every line you change)

- **Functions:** a Python function you touch is at most **50 code lines** and nests at most **4 levels**; a React component at most 150 lines; cyclomatic complexity at most 10. Touching an existing function already over a limit means splitting it in the same commit; where a function is long, put the new work in a decorator or a new small module (e.g. `actions_documents.py`'s `@with_read_document` precedent).
- **Files:** a new file at most 800 lines (aim 200–400); never grow a file already over 800 lines.
- **New platform actions (US-012, US-013, US-014) use the 3-file tool pattern:** the `ActionDefinition` inside `registry.register(ActionDefinition(...))` in an `actions_<domain>.py` (the hierarchy gate reads registrations with `ast` and exits 2 on an `ActionDefinition` built outside `registry.register(...)`), the handler in `handlers_<domain>.py` (or the domain's existing tools module), the route in `platform_executor.py`. Every `write` action is hierarchy-gated, flag-gated, or on `ALLOW_LIST` in `orchestrator/scripts/check_hierarchy_gate.py` with a comment saying why (precedent: `platform_create_social_post`, "REST POST documents:create, editor and up"). Session names drop the `platform_` prefix and go in `SESSION_TOOL_GROUPS`. **Run `python3 orchestrator/scripts/check_hierarchy_gate.py` before every push that touches `modules/tools/discovery/`** (stdlib only; `orchestrator-tests` runs it BEFORE pytest).
- **Config only via `orchestrator/config.py`**, its name in `orchestrator/reports/config-surface.json` in sorted position. No `os.getenv` / `os.environ` elsewhere.
- **No hardcoded values:** limits (page caps, render timeouts, sizes) are named constants or config.
- **Tenant isolation:** every query and every file read or write is scoped to the caller's workspace (a ticket's workspace for a session tool). Another workspace's template or Deliverable answers "not found".
- **Routes:** authenticated with the existing request-context and workspace-permission dependencies; a database-touching route is a plain `def` (F105); render work runs through `render_png_isolated` (never the CPU-bound layout on the event loop or the API process); every new route in the COMMITTED `orchestrator/reports/route-manifest.json` with its method, `route_count` updated.
- **Data lives in the database:** the Brand designer's persona and instructions are seeded into their table (`orchestrator/core/seeds/`), never read from a file at runtime.
- **Frontend:** calls only through `apiClient`; compact CSS only in the compact region of `frontend/app/globals.css`; no new npm dependency.
- **Both editions** keep working; anything hosted-only is gated by `AUTH_EDITION` / `isSaaS`, never by role. Session mode (`runtime: cli`) exists only where `config.CLI_RUNTIME_ENABLED` is on (the local edition): see US-011's note for the loop's choice.
- **No migration is expected.** If a story truly needs schema, add ONE revision for the whole wave, chained onto the base's single head, create_all-first safe, both head pins moved (`test_prd209_alembic_single_head.py`, `test_prd236_w1_routes.py`), and say why in the commit body.

## The execution contract

- **This session ends when your turn ends.** Nothing reports back later.
  - Never end your turn while anything runs in the background (an agent, a background shell, a `gh run watch`).
  - Never use ScheduleWakeup, and never reply that you will "pick up when it reports".
  - Wait for CI in the foreground: `gh run watch <id> --exit-status`. If the tool call times out, run it again.
  - **A code review runs in the FOREGROUND, before the commit,** only for the stories that touch a gate, an approval or auth: **US-012** (tenant isolation, writes into a session's folder), **US-013** (write tools, the hierarchy gate, starter protection) and **US-014** (no kit change before the owner approves, FR-11). No other story spawns a reviewer.
  - Commit and push as soon as the story's code and tests are written, then fix forward on CI.
- **RE-VERIFY every anchor by grep before building on it.** If an anchor moved, adapt and say so in the commit body. If a story's premise is gone, reply `RALPH_BLOCKED` with the grep.
- **Order is the design.** Follow `priority` in the JSON.

## Hard rules for the designer and its tools

- **The kit changes only after the owner approves (FR-11).** The designer saves through `platform_update_brand_kit` only after a granted Approve on its proposal card; a proposal is rendered WITHOUT saving. Auto never changes the kit itself on a brand ask: it files the ticket (thesis: Auto delegates).
- **Templates only through the validated actions (FR-11):** the same validation as POST/PUT `/api/documents/templates` (one shared validator; no second one). A starter (`created_by == STARTER_CREATOR`) is never changed or deleted: copy-to-customise only. No delete tool. Social formats are refused (Decision Q3).
- **`render_preview` is read-only** apart from the PNG it writes into the ticket's own `sessions/<ticket>/` folder; outside a session's ticket it refuses and writes nothing.
- **The board is rendered only from the kit (FR-10)** and refreshed on save. No AI-generated logo, no change to the owner's logo; an unset variant is never invented (FR-9).

## Tests: CI is the gate

**On the owner's machine nothing runs**: no pytest, npm, tsc, docker, compose, server or browser, and no `pip install` / `npm install`. Its Docker engine belongs to the owner's customer-night stack, and the runner exports `DATABASE_URL`, `POSTGRES_*` and `REDIS_*` pointing at 127.0.0.1:1. The one local exception is a standard-library script the gate itself runs (`python3 orchestrator/scripts/check_hierarchy_gate.py`).

**In a Claude Code cloud session** (a disposable container, not the owner's machine), you may install the dependencies there and run the tests and the changed-line checks before pushing. CI stays the only evidence: never mark a story DONE on a local run.

- **You write the tests. CI runs them.** Backend: `orchestrator/tests/test_prd255w2_*.py`. Frontend: vitest files under `frontend/components/deliverables/brand/__tests__/`. **A test must fail without your change.** No test calls a real model, a real host or Composio: the session folder's worker is faked, as F349's tests do.
- **Every story is proven on its own commit:**
  1. `git commit -s`, then push to `origin feat/prd-255-w2-brand-board-designer`. CI runs on the wave's draft PR (the runner opened it before the first story).
  2. Find the `test.yml` run for that SHA: `gh run list --branch feat/prd-255-w2-brand-board-designer --workflow test.yml --limit 10 --json databaseId,headSha,status`.
  3. Wait for it: `gh run watch <id> --exit-status > /dev/null`.
  4. Read the jobs: `gh run view <id> --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"'`.
  - A push cancels the previous run on the branch: never push the next story before this one's run has finished.
- **The jobs that must be green:** `orchestrator-tests`, `Alembic from-zero — exactly one head`, `Schema-drift check (four writers)`, `Frontend CI (tsc baselined, vitest, eslint, route-contract)`, `Prod images built with default args carry production behaviour`, `media-render`, `Code standards on changed lines (ruff, function length, nesting)` (marked non-required on pull requests, but required for this run; red is red), and `Which image lanes this change needs`. The two image lanes run on a PR only when it touches their paths: `skipped` there is fine, anything else must be `success`. US-009's social "Brand board" is on the media-render paths: expect that lane to run and keep it green.
- **If one is red:** read it with `gh run view <id> --log-failed | tail -200` (empty mid-run: `gh api repos/{owner}/{repo}/actions/jobs/<job id>/logs`), fix it in a new signed commit, and push again. A red job only proves its FIRST failing step. Red that also fails on the base's latest run is pre-existing; say so in the commit body. Typecheck: diff tsc's error LIST against the base, never the count.
- **When the jobs are green,** mark the story's AC lines `→ DONE — <evidence: the test names, and the CI run id>` in `scripts/ralph/prd-255w2.json`. Commit that alone: `chore(prd-255): <US-id> ACs DONE — CI run <id> green`. (`git add -f` the JSON by its path.)
- **An AC marked `→ OWNER:` stays as it is.** It is the owner's check: never mark it DONE, never claim it, never run a browser or a live session.
- **At the START of every iteration,** check the latest `test.yml` run on the branch tip. Red caused by this branch comes first.

## Commits

- **SIGN EVERY COMMIT: `git commit -s`.** CI enforces DCO. Messages: `type(prd-255): description`.
- **STAGING DISCIPLINE:** stage explicit paths only. NEVER `git add -A`, `git add .` or `git add -u` (`node_modules` is untracked and NOT gitignored). NEVER `git stash` (shared with every worktree).
- **Binary files:** only tiny test fixtures (under 100 KB) under `orchestrator/tests/fixtures/`. No rendered output in git.

## Hard NOs

- NO merging, NO PRs (the runner keeps the draft PR; the owner merges), and NO pushing anywhere except `origin feat/prd-255-w2-brand-board-designer`. Never touch `main`, the Wave 1 branch or `test/customer-night`.
- NO kit change by Auto, and NO kit change by the designer without the owner's Approve (FR-11).
- NO tool that changes or deletes a starter; NO template delete tool; NO social-format template from the designer's tools (Decision Q3).
- NO hosted re-seed job, NO onboarding change, NO type-scale presets (the decisions).
- NO AI-generated logo, NO change to the owner's logo, NO invented logo variant. NO image generator for the board.
- NO second Brand Designer persona or duplicate agent: one persona source, seeded insert-if-absent.
- NO weakening, skipping or deleting a test or a CI check; NO edit to the hierarchy gate's rules (adding an `ALLOW_LIST` entry with its reason is the documented path, not a weakening).
- NO edit to the generated seed `orchestrator/core/seeds/platform-management-skill.md`, the Composio deny list, or any guardrail. NO skill content written in this repo (skills come from automatos-skills).
- NO `os.getenv`/`os.environ` outside `orchestrator/config.py` (tests excepted), NO hardcoded values.
- NO new Python or npm dependency.

## Per-iteration protocol

**Owner, 6 Oct: ONE CI run per wave.** Do NOT wait for CI on a story. **Every story commit message ends with ` [skip ci]`** (GitHub then skips test.yml on that push). Never run `gh run watch`, and never poll `test.yml` during a story.

1. Pick the first story (by `priority`) with ACs not yet DONE, and re-verify its anchors.
2. Implement it with its tests. Run `python3 -m py_compile` on the changed Python files and `ruff check` on the changed paths if ruff is on PATH. Commit (signed) and push.
3. Mark its ACs DONE (the CI AC reads "pushed; CI runs at the end of the wave"). Commit (signed) and push.
4. **STOP.** One story per session. Unless that was the last story, end your reply with one line, `STORY_DONE <US-id>`. Do not start the next story.

## Completion

- **All ACs DONE** (the `→ OWNER:` ACs excepted; this session finished the last story): make ONE signed commit WITHOUT `[skip ci]` (`chore(prd-255): Wave 2 built, run CI`), push it, then run `bash scripts/ralph/acceptance-prd255w2.sh`. If it exits 0, reply `RALPH_COMPLETE`. If it fails, fix what it names (a new signed commit) and run it again. This is the one place where you wait for CI.
- **A story can't be built without breaking a Hard NO:** reply `RALPH_BLOCKED` with one line of why and the grep evidence.
