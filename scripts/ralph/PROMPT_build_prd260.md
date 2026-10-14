# Ralph Build Prompt — PRD-260 Play, a mascot theme in Automatos (one wave: T-001 … T-010)

You are building **PRD-260**, ONE story per session, unattended: every story gets a fresh session. Branch **`feat/prd-260-play-theme`**, in its own worktree, cut from `origin/main`. One PR for the whole wave, one CI run at the end: nothing in a story waits on CI.

- The base for every diff and check is what `bash scripts/ralph/acceptance-prd260.sh --print-base` prints.
- **Never merge anything into this branch** (not `main`, not `feat/prd-259-w1-skins`). Integration is the owner's job.
- **The seed already did three things — never redo or undo them:** cherry-picked PRD-259 W1's `--radius` (cc08db61c) and `dark:` fixes (ad2d3fee7, 15c70c94d) with `-x` and a sign-off; committed Auto's 36 clips under `frontend/public/auto/`; placed the READ-ONLY Play reference at `scripts/ralph/prd260-spec/play-ref/` and the fonts at `scripts/ralph/prd260-spec/play-fonts/` (untracked, gitignored).
- **This wave is frontend only:** no migration, no file under `orchestrator/` or `services/`, no new dependency.

## Read first, every iteration

1. `scripts/ralph/prd-260.json` is the BINDING contract. Its **`decisions`** are binding: D1–D11 from PRD-260 §3–§7 (Gerard, 14 Oct) and the kit decisions K1–K11.
2. `docs/PRDS/PRD-260-PLAY-THEME.md` (§3 how it fits, §4 the pages, §5 the cherry-picks, §6 non-goals).
3. `CLAUDE.md` and `AGENTS.md`.
4. `scripts/ralph/prd260-spec/play-ref/README.md`, then the Play files the story's ACs and `notes` name. The Ember mocks are PNGs in `play-ref/docs/design/ember/screens/` (read them as images).
5. The story's `notes` name the seams by file, measured 14 Oct on `origin/main` d7affad60. Line numbers drift: RE-VERIFY every anchor by grep before editing.

## Port, don't import (K1)

- `scripts/ralph/prd260-spec/play-ref/` is reading material. **Never import from it, never copy a file in unchanged, never edit it, never `git add` it** (gates N9 and N11; the guard test of T-002a fails on an import of it). Nothing under `automatos-play` or its worktrees is touched.
- Port into the core's own structure and style: kebab-case file names; `frontend/lib/looks`, `frontend/lib/mascot`, `frontend/components/looks`, `frontend/components/mascot` as the PRD names them; Play-only screens under `frontend/components/play/<area>/`; hooks as `frontend/hooks/use-*.ts`. The exact file names are in the ACs: the gate checks them.
- **Data comes from the core, never Play's client:** chat through `useChatWithReceipts` (`lib/chat/use-chat-with-receipts.ts`, the hook `chat.tsx` uses) and `stores/chat-session-store.ts`; Deliverables through `hooks/use-deliverables-api.ts`; history through `lib/chat/api.ts`; health through `apiClient.request('/health')`. Play's proxy routes, its `lib/client` and its stream parser are NOT ported (its stream TYPES are, for the mascot reducer).
- **React 18.2 here** (Play is React 19.3: no ref-as-prop — use `forwardRef`; no `use()`), **react-markdown 9** (Play has 10), **Next 15.5**, **Tailwind 3.3.3**, **next-themes 0.2.1**, **vitest 2 with `css: false`** (a CSS-module import returns class-name proxies: tests assert roles and text, never class names).

## Facts you need (verified on the base, 14 Oct)

- **The Style axis today** is `frontend/lib/ui-style.ts` (`'classic' | 'studio'`, cookie `automatos-style`, `STUDIO_HTML_CLASS`), read on the server in `frontend/app/layout.tsx` and switched in the browser by `frontend/contexts/ui-style-context.tsx` (`applyUiStyleToDocument` toggles only `studio` today). `frontend/hooks/use-studio-theme.ts` `useIsStudio()` is `style === 'studio'`. About 12 files branch on it (app/agents, assignments, chat, command-center, deliverables, playbooks, tools pages; components/documents/document-management.tsx, layout/main-layout.tsx, shared/filter-tabs.tsx, shared/page-header.tsx, shared/stats-bar.tsx): under Play they must keep Studio's layout, so `isStudio` stays true (D2).
- **Studio's tokens are HSL triplets** (`--primary: 30 12% 9%`) consumed as `hsl(var(--x))` by `tailwind.config.ts`; `:root`/`.dark` (Classic) and `.studio`/`.studio.dark` in `frontend/app/globals.css` (3,398 lines: never grow it beyond the cherry-picked `--radius` lines; gate N16). The pre-wrapped aliases `--paper, --bg-1, --bg-2, --fg, --fg-2, --fg-3, --border-2, --accent-2` hold `hsl(...)` literals.
- **The variable clash (K2):** Play's `--accent` (a hex) and `--ring` would clobber Studio's HSL `--accent`/`--ring`. Every Play variable is renamed `--pl-*`; Play's CSS reads only `--pl-*` plus `--font-geist-sans`, `--font-geist-mono`, `--font-newsreader`, `--radius` (gate N19 and the T-002a guard test).
- **Where the look lives (K3):** inline custom properties on `<html>`, set by the pre-paint script before paint and by the LookProvider after hydration, ONLY under `html.play`; leaving Play removes every one.
- **The tone (K4):** under Play the ground decides — Paper → Light, the others → Dark — through next-themes' `forcedTheme` on `ThemeProvider` (`frontend/components/providers.tsx`, which wraps `UiStyleProvider`), with the `automatos-play-tone` cookie so the server starts it for the first paint. `forcedTheme` never overwrites the stored tone.
- **CSP already allows inline script and style** (`frontend/next.config.js`). The pre-paint `<script>` is the ONE `dangerouslySetInnerHTML` this wave may add, exactly `<script dangerouslySetInnerHTML={{ __html: LOOK_PREPAINT_SCRIPT }} />` in `app/layout.tsx` (gate N8). No `innerHTML` anywhere (N7).
- **Fonts (K6):** `app/layout.tsx` uses `next/font/google` for Geist, Geist Mono and Newsreader today: leave that import line byte-for-byte (gate N18). Play's Instrument Serif, Bricolage Grotesque and Fredoka come from `scripts/ralph/prd260-spec/play-fonts/` → `frontend/app/fonts/play/` → `next/font/local` in `frontend/app/play-fonts.ts`. Never `next/font/google` for them (the 7 Oct Turbopack trap), never an `@fontsource` package.
- **The code-shape trap (K7):** `scripts/check-changed-lines-eslint.js` fails on ANY touched line inside a function over its limit (50 lines, 150 for a component), and the finding spans the whole function. `ChatPage` (`app/chat/page.tsx`) is far over; `MainLayout` (`components/layout/main-layout.tsx`) sits at ~150. So: Play's chat and Deliverables mount through the existing `app/chat/layout.tsx` and `app/deliverables/layout.tsx` via `components/play/play-route-switch.tsx`; `app/chat/page.tsx` is never edited; MainLayout gains the Play frame only with an extraction that leaves it under the limit. Run the checker before every frontend commit.
- **Routes:** every backend route Play needs is already in `orchestrator/reports/route-manifest.json` (`/api/chat*`, `/api/deliverables*`, `/health`). The frontend route-contract check runs in CI. A Deliverable's page is `/deliverables?deliverable=<id>` (K8): no new `/deliverables/[id]` route.
- **No `next build` runs in CI for this PR** (the image lanes are path-gated). So: no server-only import in a client component, `'use client'` where hooks are used, a server component (`app/conversations/page.tsx`, the layouts) imports only what a server component may. Write as if a build will judge it — the owner's local build will.
- **The cherry-picked guard test** `frontend/lib/__tests__/dark-variant-tokens.test.ts` fails on a `dark:` class with a fixed palette colour anywhere in `frontend/{app,components,lib,hooks,contexts}`: Play's code uses its CSS modules and tokens, never `dark:` palette classes.

## Code rules (AGENTS.md; CI checks them on every line you change)

- Functions ≤ 50 code lines, React components ≤ 150, nesting ≤ 4, complexity ≤ 10 on touched code; a new file ≤ 800 lines (aim 200–400; the ACs say ≤ 400 for new TS files): split Play's long CSS modules and components.
- The frontend calls the backend only through `apiClient` and the core's hooks (no `fetch('/api…')`; no `fetch` of anything else either: assets through the core's existing authenticated paths).
- No hardcoded values: named constants (intervals, limits, breakpoints). Literal colours only in `lib/looks/catalogue.ts` (the T-002a guard).
- Canonical terms in identifiers and copy (AGENTS.md): Mission, Deliverable, Auto (a name), Playbook, Command Center (the menu label stays as `lib/studio-menu.ts` has it).
- A guard test checks imports through the TypeScript compiler API (`typescript` is already a dev dependency), **never by raw text**: a comment or a string that mentions a forbidden path must never fail it (14 Oct: a docstring saying 'MinIO' failed PRD-259's text-grep guard). If you ever touch Python, ruff F403 forbids `import *` (gate N20).
- No new npm or Python dependency (gates N5, N6). Both editions keep working (`AUTH_EDITION=local` and `saas`: the profile control and edition filters are reused, never re-derived).

## The execution contract

- **This session ends when your turn ends.** Never end it with background work; never ScheduleWakeup; never "pick up later". Wait for anything you start in the foreground.
- **A code review runs in the FOREGROUND, before the commit,** for **T-002a** (look files are untrusted input: code-reviewer, and security-reviewer on `transfer.ts`/`validate.ts`) and **T-002b** (the pre-paint script and the provider: code-reviewer and security-reviewer). Wait for each. No other story spawns a reviewer.
- **One story per session.** Build the first story whose ACs are not all DONE, then STOP (step 4 below), even if time remains.
- **RE-VERIFY every anchor by grep.** **Order is the design** (the JSON's priority order is the dependency order).
- **Never push** anything yourself: the runner pushes the branch after each session (a backup to a branch with no PR, so nothing runs) and opens the wave's draft PR at completion.

## Tests: CI is the gate, once per wave

**Nothing runs on the owner's machine** (no npm, npx, vitest, tsc, next build/dev, docker, server, browser, installs, pytest). Local checks only: `cd frontend && node scripts/check-changed-lines-eslint.js origin/main` after every frontend change, and `bash scripts/ralph/acceptance-prd260.sh --no-ci`. You WRITE the tests each AC names (`frontend/**/__tests__/*.test.ts(x)`, ported from `play-ref/tests/` where they fit, adapted to the core's hooks and fixtures); CI runs them once at the end. A story never waits on CI and never claims a test passed. `→ OWNER:` ACs stay as they are.

## Commits

- `git commit -s` (DCO: every commit), messages `feat(prd-260): description` (or `fix(prd-260):`, `test(prd-260):`, `docs(prd-260):`), **every story commit ending ` [skip ci]`**; explicit paths only (`git add <path> …`); never `git add -A`, `git add .` or `git add -f`; never `git stash`; never amend a pushed commit (the seed's cherry-picks included).
- Binary files (the woff2 fonts) are copied with `cp`, never rewritten.

## Hard NOs

- NO merging, NO PRs (the runner opens the draft PR at completion), NO pushing (the runner pushes `origin feat/prd-260-play-theme`). Never touch `main` or any other branch, never `git checkout`/`switch` away from this branch, never `git cherry-pick`/`revert`/`rebase` (the seed's picks are done).
- NO file under `orchestrator/` or `services/`; NO migration; NO new dependency; NO change to `frontend/package.json` or `package-lock.json`.
- NO import from, edit of, or commit of `scripts/ralph/prd260-spec/`; NO edit under `automatos-play` or its worktrees.
- NO edit to `frontend/app/chat/page.tsx`; NO growth of `frontend/app/globals.css` or `frontend/lib/api-client.ts`; NO change to `DEFAULT_UI_STYLE`, `APPEARANCE_DEFAULTS_KEY` or `APPEARANCE_DEFAULTS_VERSION` (Play is opt-in: D11).
- NO weakening, skipping or deleting a test or a CI check (the cherry-picked tests included); NO growing a function already over the limit.
- NO building the non-goals (PRD §6: Play designs for Missions, Agents, Knowledge; phones; the Design Kit; server-side looks) and NO making Play the default (§7.1 is Gerard's).
- NO edit to the generated seed (`orchestrator/core/seeds/platform-management-skill.md`), the Composio deny list or any guardrail.

## Per-iteration protocol

1. Pick the first story (by `priority`) with ACs not yet DONE.
2. Implement it with its tests; the local checks above; commit (signed, ` [skip ci]`).
3. Mark its ACs by APPENDING plain text (outside backticks, exactly as written) `→ DONE — <evidence: files, test names, commit>` in `scripts/ralph/prd-260.json` (`→ OWNER:` ACs stay). Commit that alone (signed, ` [skip ci]`).
4. **STOP.** Unless that was the last story, end your reply with `STORY_DONE <T-id>` as its last line. **The words RALPH_COMPLETE, RALPH_BLOCKED and STORY_DONE are sentinels the runner reads on your last line only (≤ 80 characters): never write them anywhere else in a reply, not even in a sentence about them** (a PRD-256 session did, and the runner ended that wave early).

## Completion

- **All ACs DONE** (`→ OWNER:` excepted): run `bash scripts/ralph/acceptance-prd260.sh --no-ci`, fix what is red in signed ` [skip ci]` commits, then ONE signed commit WITHOUT `[skip ci]` (`chore(prd-260): Play built, run CI`; an empty commit is fine: `git commit -s --allow-empty`), and reply with `RALPH_COMPLETE` as the last line.
- **Already built and tested:** if every AC is DONE and the last commit that touches anything outside `scripts/ralph/` is the wave's CI commit (`chore(prd-260): Play built, run CI`), make NO commit at all (the runner reads that commit's run; a new commit would start a needless second run): reply `RALPH_COMPLETE` and nothing else.
- **A story can't be built without breaking a Hard NO** (a backend change, a new dependency, editing `app/chat/page.tsx`): reply with one line of why and the evidence, then `RALPH_BLOCKED` as the last line (this stops the wave: no PR, no CI). Use it for a real contradiction only; a story that needs a judgement call takes the smaller change and says so in its DONE mark.
