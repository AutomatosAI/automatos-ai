# Ralph Build Prompt — PRD-246 Studio on mobile

You are executing **PRD-246**, one story per iteration, unattended. Branch **`ralph/prd-246-studio-mobile` ← `studio`**. Tip green after every commit.

**CONTEXT.** PRD-244 made Studio the default style and Dark the default tone for every browser. Below 1024 px every Studio route still returns the *classic* component while the chrome and the tokens stay Studio — so a phone gets the hybrid Gerard rejected on the desktop. You give each Studio surface a compact form, then flip the route forks. Classic mobile is untouched throughout: it is the permanent fallback, not a thing to convert.

## Read first, every iteration

1. `scripts/ralph/prd-246.json` — the BINDING contract (stories, acceptance criteria, traps). Take the first story with un-DONE ACs.
2. `docs/PRDS/PRD-246-STUDIO-MOBILE.md` — the spec. §"What the audit found" carries the file:line evidence; decisions **M1–M8 are LOCKED**.
3. `CLAUDE.md` — reuse over build, delete what you replace, canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep before building on it.** The evidence is from 2026-09-17 on `studio` @ `ac8a44113`. If an anchor moved, adapt and say so in the commit body; if a story's premise is gone, `RALPH_BLOCKED` with the grep.
- **Order is the design.** US-007 flips the route forks and must run last; starting it while US-002…US-006 are un-DONE puts a desktop-shaped Studio on every phone.
- **CSS first (M5).** A surface responds through media queries in the single "── Studio compact" region US-001 creates. A component fork is justified only where the information architecture changes (the chat's columns). Scattered `@media` blocks across the file are a review CRITICAL.
- **Scope every rule.** Studio rules hang off `:is(.studio, <own root>)` — a bare `.entry-grid` or `.cc-stats` rule leaks into Classic and fails the pass.
- **Never weaken a PRD-244 gate.** The chrome guard (`flex-shrink: 0` on every named chrome row), `prd244-w0-honesty`, `studio-honest-chrome`, `cc-page-scope`, `studio-surfaces-scope`, `markdown-typography` and `page-frames` all stay green as written. If a mobile requirement genuinely contradicts one, `RALPH_BLOCKED` with the evidence — do not edit the gate to pass.
- **No new primitives.** Sheets use the existing `Sheet`; the rail is the one `AutoNowRail`; safe-area uses the one `.safe-bottom` helper; markdown is `MarkdownView`. A second drawer, a second rail or a second safe-area helper is an automatic CRITICAL.
- **Tests are vitest + jsdom.** Width is mocked through `@/hooks/use-mobile`; style through `UiStyleProvider initialStyle=…`. **Nothing runs a server, a browser or a database.** Adding `StatsBar` to a page means that page's test needs `@/hooks/use-system-config-api` stubbed (it reads react-query).
- **`npm run build` MUST PASS before every commit, not just `npm run test`.** vitest mocks `@/components/layout/main-layout` in most suites, so a syntax error there passes the test run and still breaks the production build — that is exactly what US-001 shipped (a `{/* … */}` JSX comment placed inside `return (` before the root element; fixed in US-003). The Docker image build in CI is the check that catches it. Run `cd frontend && npm run build` before you commit; a red build is a red story.
- **Green tip:** `cd frontend && npm run test` after every commit; never commit on red. Pre-existing unrelated red is not yours to fix.
- **SIGN EVERY COMMIT: `git commit -s`.** This repo enforces DCO in CI — an unsigned commit fails the `dco` check.
- **STAGING DISCIPLINE:** explicit paths only. **NEVER `git add -A` / `.` / `-u`** (node_modules is untracked and NOT gitignored). Never `git stash -u`.

## Hard NOs

- NO merging, NO PRs, NO pushing anywhere except `origin ralph/prd-246-studio-mobile`. **NEVER touch `main`** — prod is frozen.
- NO changes to Classic components, Classic branches of the shared primitives, or the classic mobile sidebar.
- NO deleting a classic page, route or component (M6).
- NO new breakpoints — 768 and 1024 exist in `hooks/use-mobile.ts` and are enough (M2).
- NO new dependencies. In particular NOT `@tailwindcss/typography`: this app styles markdown through `.md-view`, and a gate asserts the plugin's absence.
- NO backend changes. This PRD is frontend-only.
- NO `git add -A`; NO `git stash -u`.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh.
2. Implement → `cd frontend && npm run test` (story-scoped first, full suite before commit).
3. Commit `git commit -s -m 'feat(prd-246): <US-id> — <title>'` with evidence in the body (the `-s` is required — CI enforces DCO); mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-246.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd246.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why plus the grep evidence in the last commit.
