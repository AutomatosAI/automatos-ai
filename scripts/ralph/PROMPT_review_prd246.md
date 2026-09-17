# Ralph Review Prompt — PRD-246 Studio on mobile

You are a fresh-context **adversarial reviewer**. The build claims PRD-246 is complete. Your job: find where the compact form is a shrunken desktop, where Classic got dragged in, or where a PRD-244 gate was quietly weakened to make a story pass. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD studio)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Diff against **`studio`**, never `main`. Read `scripts/ralph/prd-246.json` (binding; M1–M8 locked) and `docs/PRDS/PRD-246-STUDIO-MOBILE.md` (the audit's file:line evidence).

## Hunt list — every item is a confirmed-risk class

1. **The hybrid must be gone, not moved (M1).** After US-007, does Studio-on-a-phone ever render a classic component, and does Classic-on-a-phone ever render a Studio one? Either direction = CRITICAL. Grep `app/**/page.tsx` for any surviving `isStudio && !is…` component choice.
2. **Compact is designed, not shrunk (M3).** For each surface, does the diff show a real compact form — stacking, sheets, scrolling strips — or only font/padding reductions? A three-column grid that merely narrows = HIGH. A table that keeps four columns at 390 px = HIGH.
3. **Classic is untouched (M6).** Any diff hunk inside a classic component, a classic branch of `PageHeader`/`StatsBar`/`FilterTabs`, or `components/layout/mobile-sidebar.tsx` = CRITICAL unless the story explicitly authorised it. The classic tests must still assert the gradient title, the card grid and the Radix tablist.
4. **The chrome guard survives (PRD-244).** Every named chrome row still carries `flex-shrink: 0`, and no compact rule reintroduces shrinking on one. A collapsed tab strip is the exact bug this PRD inherited — if any surface can crush its chrome at any width, CRITICAL.
5. **Scoping leaks.** Every new rule hangs off `:is(.studio, <own root>)` or sits inside a `.studio`-scoped block. A bare `.cc-stats`, `.entry-grid` or `.sh-chat*` rule reaches Classic = CRITICAL.
6. **One compact region (M5).** All compact rules live in the single region US-001 creates. `@media` blocks scattered through `globals.css` = HIGH; a second region = HIGH.
7. **No second primitive.** One `Sheet`, one `AutoNowRail`, one `.safe-bottom`, one `MarkdownView`. A new drawer/rail/safe-area helper = CRITICAL. A new breakpoint beside 768/1024 = HIGH.
8. **Gates intact.** `prd244-w0-honesty`, `studio-honest-chrome`, `cc-page-scope`, `studio-surfaces-scope`, `markdown-typography`, `page-frames` — all unchanged and green. Any edit to a gate's assertions to make a story pass = CRITICAL (the story should have been `RALPH_BLOCKED`).
9. **Tests prove behaviour, not wishes.** Width must be mocked through `@/hooks/use-mobile` and style through `UiStyleProvider`; a test that renders at the default width and claims a mobile assertion is worthless = HIGH. Any page test mounting `StatsBar` without stubbing `@/hooks/use-system-config-api` will fail on react-query — if suites were made to pass by deleting assertions, CRITICAL.
10. **Conventions.** Frontend only (a backend diff = CRITICAL); no new dependency, and emphatically not `@tailwindcss/typography`; no `git add -A` poisoning (`node_modules` in the diff = CRITICAL); no PRs, merges or pushes outside `ralph/prd-246-studio-mobile`.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff — any CRITICAL/HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd246.sh` yourself; a green build with a red gate is a finding.
- Read the acceptance criteria marked `DONE` in `prd-246.json` and spot-check three at random against the code. Evidence that does not exist = CRITICAL.

## Verdict

Reply exactly one of:

- `REVIEW_CLEAN` — no CRITICAL or HIGH findings.
- `REVIEW_FINDINGS` followed by a numbered list, each with severity, `file:line`, and the one-line reason. No fixes, no diffs.
