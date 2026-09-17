# PRD-244: Studio is the desktop UI — converge the two shells, test every page, and give chat a live view of the floor

> **Status:** APPROVED 2026-09-17 by Gerard with two amendments (D1 split, D6 sequencing — recorded below). Build and test on the **local** edition first; nothing here reaches prod before Wave 4 is signed. Grounded @ the Studio audit of `main` @ `2aad76154` the same day (every file reference below was read, not recalled). PR #397 (`feat/studio-rebrand-phase1`, opened 2026-05-29, 119 commits behind, 9 of 18 files since rewritten on `main`) stays **parked**; §D4 harvests what is still unique and closes it.

## The review (2026-09-17)

1. *"I can't see the Questions tab on either local or prod."* — PRD-225's tab exists, but only in the Command Centre shell that renders when the theme is **Studio** and the viewport is **≥ 1024 CSS px** (`app/command-center/page.tsx:12-17`). The default theme is `system` (`components/providers.tsx:91`), the picker labels Studio "preview" (`components/ui/theme-toggle.tsx:66-72`), and nothing ever defaults to it. Everyone else gets the legacy `ActivityPage` (Summary · Board · Calendar · Feed · History) — no Questions, no Watchlist, no Governance; the bell's `?tab=questions` link lands there and does nothing.
2. *"Studio was parked for a while as it needs proper testing … Studio can't go live until it's fully tested."* — Parked in name, load-bearing in practice: three of the six PRD-224–229 surfaces (Questions, Watchlist, Governance) and the round-4 Command Centre (stats strip, "is it working" strip, setup checklist, live tab counts) ship **only** in Studio; `app/globals.css` carries **599** Studio-scoped lines; seven files fork on `useIsStudio`. Every page is two pages.
3. *"Could some of those be shortcuts on the /chat page … a quick view of the command centre and what Auto is doing."* — Nothing operational is shown in chat today: no consumer of `useActivityStats`, `useFleetState`, `useQuestions`, `useWatches` or `useDecisionsNeeded` exists under `components/chatbot`, `lib/chat` or `app/chat`. The only live objects are the per-message `TaskCard` (one ticket) and the Studio-only mission rail (`components/chatbot/studio-chat-shell.tsx:337-421`), which reads the active mission and nothing else.

## What the audit found (facts, not opinions)

**Mechanics.** `useIsStudio()` returns `false` until mounted (`hooks/use-studio-theme.ts:41`), so every forked page paints the classic tree first and swaps after hydration. The only way in is the picker or `?theme=studio-preview` (`hooks/use-studio-theme.ts:16-26`), persisted by next-themes under `automatos-theme`. Clerk's appearance is already the Studio palette in every theme (`components/providers.tsx:41-69`).

**The seven forks.** `components/layout/main-layout.tsx:34,91` (shell, sidebar, header, page tabs; mobile Sheet at `:151,156`); `app/command-center/page.tsx:12-17`; `app/chat/page.tsx:55,277` (three-column ledger vs `ChatTabs` + history sheet); `app/assignments/page.tsx:12-17`; `app/missions/page.tsx:13,22-26` (redirect keeps query params in Studio, drops them in classic); `app/playbooks/page.tsx:14-28` (Studio renders `null` while redirecting). Breakpoint inconsistency: chat forks on `useIsMobile` (768 px) while every other page forks on `useIsTabletOrBelow` (1024 px) — `hooks/use-mobile.ts:3-4`.

**Studio-only.** Command Centre tabs Watchlist · Questions · Governance and their components (`command-center-shell.tsx:51-64,217-219`; `watchlist-tab.tsx`, `questions-tab.tsx`, `governance-tab.tsx` have no other consumer); `StatsStrip`, `IsItWorkingStrip`, `SetupChecklistCard`, `TrialBalancePill`, live tab counts (`:107-125,172-191`); the Assignments hub's "Start something" grid and Missions/Playbooks flip tabs (`components/assignments/studio/assignments-hub.tsx:99-144`); the chat mission rail.

**Classic-only.** Feed and History tabs plus the period selector (`components/activity/activity-page.tsx:89-95,266-281`); a second Board implementation (`BoardView`, `activity-page.tsx:48`) beside Studio's `BoardTab`; the `?from=assignments` back button in chat (`app/chat/page.tsx:302-313`).

**CSS.** 521 selector lines under `.studio`, 63 dual-scoped `:is(.studio, .cc-cal-root)` (the PR #710 calendar fix, `globals.css:2096-2108`), two token blocks (`:225`, `:338`). Buckets: `cc-*` 232, `sh-*` 122, `mis-*` 37, `studio-*` 26, `pb-card` 13, `entry-*` 13, `mkt-*` 10, plus legacy overrides (`glass-*`, `stage-*`, `log-entry*`, `glow-*`). No classic mount consumes a Studio-only selector today; the calendar was the one case and is fixed.

**Dead or drifted config.** `STUDIO_PAGE_TABS.assign` never renders (`/assignments` mounts `MainLayout fullBleed`, `app/assignments/page.tsx:19`, and page tabs mount only when `!fullBleed`, `main-layout.tsx:100`); `STUDIO_PAGE_TABS.deliv` lists an Explorer tab absent from `app/deliverables/page.tsx:32`; `STUDIO_PAGE_TABS.agents` links `?tab=` that `agent-management.tsx:46,133-139` never reads; `lib/studio-menu.ts:6` says 13 primary items, there are 11; `lib/studio-menu.ts:109` maps `/activity*` to a route that does not exist. Studio's Workspace Admin item has no admin role gate (classic has one, `components/layout/sidebar.tsx:128,146-147`) and Studio's footer Settings is always shown (classic: admin-only, `mobile-sidebar.tsx:216`). The mobile sidebar carries 10 items with shorter labels (`mobile-sidebar.tsx:36-115`).

**Mobile.** No mobile rendering exists for the Command Centre shell, the Assignments hub or the Studio layout itself — all fall back to classic under 1024 px; the chat rail is hidden by CSS under 1280 px (`globals.css:1342-1346`).

**Tests.** Nine files touch Studio (chrome honesty, sidebar edition gating, calendar scope, header edition, placebo relics, both Command Centre shells mount `AutosRead`, classic calendar tree deleted, chat resume on the classic path). **None covers `hooks/use-studio-theme.ts`.**

## Framing (CLAUDE.md §3)

**Refactor / consolidation.** Two desktop UIs exist; one is canonical. Pick it, migrate the survivors, delete the losers (§5). Plus one **extension**: a live rail in chat built from reads that already exist.

## Decisions

- **D1 · Studio is the desktop UI — in two steps (Gerard, 09-17).** **D1a, now:** the default stays `system`; the picker keeps a **Studio** entry with the word "preview" removed; Light and Dark remain the palette *inside* Studio (tokens, `globals.css:225`); Matte and the `?theme=studio-preview` flag go; when Studio is the theme the shell renders server-side with no classic flash. **D1b, only after Wave 4 is signed:** the default becomes `studio` for viewports ≥ 1024 px and the picker offers Light / Dark / System. Until then Studio is opt-in and is tested hard on the local edition.
- **D2 · One Command Centre.** The shell is the Command Centre at every desktop width. Feed and History move into the shell's Activity tab (one feed, one history, the period selector kept); `ActivityPage`, `BoardView` and the classic activity tree are deleted in the same PR. The bell's deep-links then work for everyone — **this closes the Questions-tab report.**
- **D3 · One chat, one Assignments.** The Studio chat shell and Assignments hub are the only desktop implementations; the classic `ChatTabs` + history sheet and the classic `AssignmentsPage` are deleted after parity (the `?from=assignments` back button moves into the shell; missions/playbooks redirects keep query params in both).
- **D4 · #397 is harvested, then closed.** Eight files apply cleanly and untouched by `main` (assignments hub, playbook card, page header, studio page tabs, dialog centring, system-config hook, premium icon, activity tab — ~240 lines); they land as a fresh small PR if still wanted after D3. The mission-detail, deliverables and board-viewer pieces are redone on today's `main`, deliberately, or dropped. The stale branch is never merged.
- **D5 · Chat gets "Auto now", a live rail, not a dashboard.** One collapsible rail in the chat shell (the existing `sh-chat-rail` pattern, `studio-chat-shell.tsx:55-56,96-101,318-329`) whose sections are the floor's live objects, each a one-line row that deep-links to its Command Centre tab: **Working now** (`useActivityStats('1d')`: working_now · agents_active · tasks_in_queue · needs_attention, `hooks/use-activity-api.ts:192-197`) and the top three fleet rows with their live line (`useFleetState`, 10 s + `automatos:board-changed`, `hooks/use-agent-api.ts:178-207`); **Questions** (`useQuestions`, count + the newest three, answer inline through `useAnswerQuestion`, `hooks/use-approval-grants.ts:43,75`); **Watchlist** (`useWatches`, count + next due, `hooks/use-watches-api.ts:22`); **Decisions** (`useDecisionsNeeded`, count → Governance, `hooks/use-kpi-api.ts:124-128`); **Mission** (today's rail, unchanged). No new endpoint; every read exists with its own cadence. Empty sections say so ("Nothing waiting on you"); the rail never fabricates a count. Collapse state persists as today. Under 1280 px the rail becomes a header pill with the two counts that need a human (questions + decisions).
- **D6 · Mobile stays classic for now, and a mobile pass precedes prod (Gerard, 09-17).** Below 1024 px the classic tree remains through the Studio pass; once Wave 4 is signed on local, a mobile pass (PRD-245) is built and tested on local **before** D1b ships to prod. The breakpoint is the same everywhere (1024), and the chat page moves from 768 to it.
- **D7 · The audit's dead config is fixed, not documented.** `STUDIO_PAGE_TABS` either drives the pages (agents reads `?tab=`, deliverables gains or drops Explorer, assignments renders tabs) or is deleted; the menu comment matches the list; the `/activity` mapping goes; Workspace Admin and Settings carry the same role gates in both rails.

## Stories

**Wave 0 — honesty (small, first).** D7 in full; chat breakpoint → 1024; a vitest for `hooks/use-studio-theme.ts` (flag, persistence, mounted gate); a `STUDIO_PAGE_TABS` ↔ page `VALID_TABS` consistency test; the `calendar-tab-scope` test pattern generalised into one test that fails when a classic mount consumes a `.studio`-only class. Acceptance: no inert Studio link, no route to nowhere, CI green.

**Wave 1 — one Command Centre (D2 + the theme default, D1).** Shell at every desktop width and theme; Feed + History + period selector inside the Activity tab; `ActivityPage`/`BoardView` deleted; bell deep-links verified from a fresh browser on both editions; Studio default with Light/Dark tokens; Matte and the preview flag removed; SSR renders the shell (no post-mount swap). Acceptance: a new user on either edition opens Command Centre and sees the seven tabs; `?tab=questions` from the bell lands on Questions; a manual pass of the checklist below on the Command Centre rows.

**Wave 2 — one chat, one Assignments (D3) + the harvest (D4).** Classic chat layout and `AssignmentsPage` deleted after parity; #397's eight clean files landed or dropped; #397 closed with the note. Acceptance: chat resume, tabs, history, `?from=assignments`, missions/playbooks deep-links all pass on both editions.

**Wave 3 — "Auto now" (D5).** The rail, its sections, the small-width pill, empty states; a vitest per section on fake hook data; no new endpoint. Acceptance: with one open question, one due watch and one running ticket, the rail shows three honest rows and each row lands on the right tab; with nothing, it says so.

**Wave 4 — the test pass (Gerard; local first, then SaaS).** The checklist below, executed on local and SaaS, at desktop (≥ 1024) and tablet (768–1023), from a fresh browser profile and from an existing one, on each of Light and Dark inside Studio. CI is the only gate for code; this pass is the gate for go-live — Studio is not the default until it is signed.

## Manual test checklist (the pass Wave 4 executes)

| Page | What to check |
|---|---|
| Sign-in → first paint | The shell renders on first load with no classic flash; theme persists across reload and across tabs; Light/Dark switch inside Studio; System follows the OS |
| Sidebar + header | 11 items, same labels both rails, Team/Analytics exposure gates, Workspace Admin + Settings admin gates identical to classic; collapse persists; account slot per edition |
| Chat | resume last conversation, open several as tabs, unread dot, history panel, `?from=assignments`, plan mode, mission suggestion/created cards, voice dock, attachments, "Auto now" rail sections + collapse + small-width pill |
| Command Centre | seven tabs; Summary strips and live counts; Board (blocked/failed visible, drag, approve); Calendar (styled, schedule for later); Activity incl. feed + history + period; Watchlist (open, cancel); Questions (answer, option buttons, dismiss, cascade); Governance (approve, deny); every bell deep-link |
| Assignments | Missions/Playbooks flip tabs with counts, "Start something" grid, mission row → detail, playbook run → execution, `?tab=` deep-links |
| Deliverables | tabs incl. whatever Explorer becomes, preview, download, share links |
| Agents | Roster (runtime badge), Fleet (live lines, CLI host card), Org Chart, Configuration modal (all tabs incl. Model/runtime, Heartbeat), Skills |
| Tools & Integrations | connect/manage state after OAuth, marketplace, categories |
| Knowledge, Marketplace, Team, Analytics, Workspace Admin, Settings | render, gates per edition, no unstyled block, Channels tab per edition |
| Document Template Studio | picker, layouts, preview, generate (PRD-242/243 flows) |
| Errors | a failed turn notice, a 403 page, an empty workspace, offline backend banner |

## How it reads now

Open the app: Studio, no flash. Chat on the left, the conversation in the middle, and on the right a quiet rail: **Working now 2 · 1 needs you**, then *OPS — working: Draft the Q3 vendor email*, one open question from TRACKER with an answer box, a watch due at 16:00, one approval waiting. Click any row and you are on that Command Centre tab; the tab is there whatever theme or window you use. Nothing is "preview" any more.

## Open questions (Gerard's call)

1. ~~Mobile~~ — decided: PRD-245 after the Studio pass, before prod (D6).
2. ~~Matte~~ — decided: retired (D1a).
3. **"Auto now" cadence.** The rail inherits each hook's polling (10 s fleet, 30 s questions, 60 s watches). Fine for a pilot; a single aggregate read (`/api/command-center/pulse`) would cut chat's request count if it matters on SaaS. Not built unless you say.
4. **Harvest scope for #397.** Land the eight clean files, or drop them if D3's hub already covers what they added? The mission-detail changes are the only ones I could not map to something already on `main`.
