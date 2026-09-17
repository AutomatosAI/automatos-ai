# PRD-246: Studio on mobile — the compact form of the design system, in both tones

> **Status:** DRAFT → build-ready, written 2026-09-17 for an overnight Ralph run on branch `ralph/prd-246-studio-mobile` ← **`studio`** (never `main`: prod is frozen until Gerard's desktop pass and this one are both signed). Grounded in the tree at `studio` @ `ac8a44113`; every file reference below was read, not recalled.
>
> **Numbering note.** PRD-244 §D6 promised this work as "PRD-245". That number is already in use by the session-lane workstream (`feat/prd-245-w0-session-lane` … `-w3-session-composio`). The mobile pass is **PRD-246**; the stale reference in PRD-244 is corrected in the same commit as this file.

## The problem (2026-09-17)

Studio + Dark became the default for every browser earlier today (PRD-244 §D1). On a phone that default currently produces **the exact hybrid Gerard rejected on the desktop**: Studio chrome and Studio colours wrapped around classic page bodies.

It is not a styling gap. It is the deliberate holding pattern from PRD-244 §D6 ("mobile stays classic until a pass is scoped") meeting a default flip that happened before the pass was written.

## What the audit found (facts, not opinions)

- **Every Studio route fork is `isStudio && !isTabletOrBelow`** — `app/chat/page.tsx`, `app/command-center/page.tsx`, `app/assignments/page.tsx`, `app/agents/page.tsx`, `app/deliverables/page.tsx`, `app/tools/page.tsx`. Below 1024 px they all return the classic component.
- **The chrome does not fork the same way.** `components/layout/main-layout.tsx:143-160` opens a `Sheet` on mobile and fills it with `StudioSidebar` when the style is Studio. So the rail is Studio, the page is classic, and the tokens are Studio's. That is the hybrid.
- **Breakpoints already exist and are enough:** `hooks/use-mobile.ts:3-4` — `MOBILE_BREAKPOINT = 768`, `TABLET_BREAKPOINT = 1024`. No new ones are needed.
- **The Studio stylesheet has almost no compact story:** 8 `@media` blocks in ~3,050 lines, and the Studio-scoped ones stop at 700 px — `sh-chat-grid` narrows and `sh-chat-rail` hides at 1279 (`globals.css:1271`), `cc-stats` steps 6→3→2 (`:1621`, `:1625`), `cc-sum-grid` collapses at 1199 (`:1697`), `cc-roster-row` at 700 (`:1753`), `cc-act-card` drops two columns at 1199 (`:2369`). The only sub-768 rule in the file reduces `backdrop-filter` on `.glass-card` (`:2981`) and is Classic's.
- **The designed surfaces assume width.** The chat is a three-column ledger (`.sh-chat-grid`), the Command Centre is a seven-tab strip over a six-cell stats grid, the Assignments hub is a card grid, the board is `grid-auto-flow: column` with `minmax(240px, 1fr)` columns (`globals.css`, the one-row fix from the review), and the activity table is a real table.
- **Touch and notch handling is Classic-only.** `MobileSidebar` uses `min-h-[48px]` targets; the Studio rail's `sh-item` is sized for a pointer. A `.safe-bottom` helper exists (`globals.css:3034`) and no Studio surface uses it.
- **The Auto now rail is already honest on small screens:** it hides under 1280 and leaves the pill (PRD-244 §D5), which is the pattern the rest should follow.

## Framing (CLAUDE.md §3)

**Rehouse + extension.** Every surface, hook and component already exists. This adds a second form for the Studio surfaces and the chrome that carries them. Nothing new is invented; Classic is not touched.

## Decisions

- **M1 · Mobile follows the Style axis, like the desktop.** Studio on a phone renders the **Studio** surfaces in their compact form. Classic on a phone renders today's classic mobile, unchanged. Neither borrows from the other; the hybrid ends.
- **M2 · Two widths inside Studio, using the hooks that exist.** *Compact* is `< 1024` (`useIsTabletOrBelow`); *phone* is `< 768` (`useIsMobile`) and is used only where a surface needs a third form. No new breakpoints, no new hooks.
- **M3 · Compact is a designed form, not a shrink.** Each surface states what it drops, what it stacks, and what moves behind a sheet. A horizontally-scrolling desktop layout is a last resort, never the default answer.
- **M4 · The chrome carries the phone.** The Studio rail stays in the sheet, with 48 px targets; the Studio header keeps the page title and collapses actions into one overflow; every bottom-anchored surface (the chat composer above all) respects `env(safe-area-inset-bottom)`.
- **M5 · CSS first, TSX only for information architecture.** A surface responds through media queries on its own class family (`sh-chat*`, `cc-*`, the hub families). A component fork is justified only where the structure changes — the chat's three columns becoming one is the single expected case.
- **M6 · Classic mobile is untouched and permanent.** It is the fallback for anyone who wants it and the reference for "did we break something".
- **M7 · The desktop gates stay green and gain a mobile sibling.** The chrome guard (no chrome row absorbs page overflow), the page-frame gate, the markdown typography gate and the scope tests all keep passing; a new gate asserts every Studio class family has a compact rule set.
- **M8 · Nothing reaches prod on this branch.** `studio` only. Gerard's pass on a real phone, in both tones, is the gate — it is the last thing before the single `studio` → `main` PR.

## Stories (the Ralph waves)

Each is one iteration, one commit, tip green. The binding contract with acceptance criteria is `scripts/ralph/prd-246.json`.

**US-001 — Foundations.** The compact conventions: where mobile rules live in `globals.css`, the safe-area utility applied to the Studio shell and composer, 48 px minimum targets on `sh-item` and every Studio control, and `.md-view` defaulting to its compact density under 768 px. No surface changes yet.

**US-002 — Command Centre compact.** The tab strip scrolls with its active tab in view; stats go 2-up under 768; the summary widgets become one column; the board keeps one row per status but snaps, with the column head sticky; the activity table becomes the card list it already has at 1199; the calendar gets a month/day toggle rather than a shrunken grid.

**US-003 — Chat compact.** One column: the conversation. Threads move behind the existing left sheet, Auto now stays the pill it already becomes, the composer is safe-area aware and never covered by the keyboard, and the breadcrumb bar collapses to the active thread's title.

**US-004 — Assignments hub compact.** The entry grid is 1-up, the flip tabs scroll, mission and playbook cards stack with their meta beneath the title, and the status head wraps instead of truncating.

**US-005 — Agents, Deliverables and Tools compact.** The in-page `cc-tabs` scroll, stats 2-up, roster and app cards 1-up, and the toolbars wrap rather than overflow.

**US-006 — The framed pages, in one move.** `PageHeader`, `StatsBar` and `FilterTabs` get their compact forms, which converts Team, Analytics, Settings and its sub-pages, Knowledge Base, Marketplace and its sub-pages, mission detail and the three admin surfaces at once — the same leverage that converted them to Studio in PRD-244 Waves 5d–5g.

**US-007 — The forks flip.** With every surface compact, the six routes fork on **style alone**: Studio renders the Studio component at any width, Classic renders the classic one. `useIsTabletOrBelow` stops choosing the *component* and only informs the *layout* inside it. Classic components stay exactly where they are.

**US-008 — Gates and the checklist.** The mobile scope gate (every Studio family has compact rules), a route test that Studio-on-phone renders the Studio component, and the manual checklist below written into the PRD's tail for Gerard's pass.

## Manual test checklist (Gerard's pass — the gate for prod)

| Surface | On a phone, in Studio Light and Studio Dark |
|---|---|
| Chrome | Rail opens from the sheet, every item is thumb-sized, nothing hides under the notch or the home bar |
| Chat | One column, threads reachable, composer visible with the keyboard up, Auto now pill opens the rail's content |
| Command Centre | Tabs reachable without pinching, stats readable, board usable, an activity row opens the right thing |
| Assignments | Start-something cards stack, flip tabs reachable, a mission opens |
| Agents / Deliverables / Tools | Tabs scroll, cards stack, modals fit the screen |
| Team / Analytics / Settings / Admin | Head, stats and tabs read correctly at 390 px |
| Both tones | Nothing unreadable, no Classic remnant inside a Studio page |
| Classic style | Unchanged from today on the same phone |

## How it reads when it's done

Open the app on a phone. It is the same app as the desktop, in the same design system: paper or ink, the serif heads, the rail one tap away. The chat is a conversation with the floor one tap behind a pill. The Command Centre's tabs slide under your thumb. Nothing is a shrunken desktop and nothing is a leftover from the old design.

## Open questions (Gerard's call — the build proceeds on the stated default)

1. **The board on a phone.** Sideways scroll of status columns (today's shape, with snap) *or* a single list with a status filter? **Default while unanswered: sideways scroll with snap** — it keeps one board, and the filter is a bigger change than a pass should make.
2. **Does Classic stay selectable on a phone** once Studio mobile is signed, or does the picker hide the style axis on small screens? **Default: Classic stays selectable**, per M6.
