# PRD-244: Two styles, two tones — Classic and Studio, each in Light and Dark; Studio redesigned on every page; chat gets a live view of the floor

> **Status:** APPROVED 2026-09-17 by Gerard; **model amended the same evening after the local test of Waves 1–2** — see *The second review* below. Build and test on the **local** edition first; nothing here reaches prod before the manual pass is signed. **Mechanism (Gerard, 09-17):** `main` deploys to prod on every push (Railway), so every wave merges into the `studio` integration branch (cut from `main` @ `757814531`, CI gates widened to it); the local stack runs a test branch that merges the unmerged wave branches (Gerard tests **before** merging); one PR takes `studio` → `main` after the pass, and `main` is merged into `studio` whenever it moves. Grounded @ the Studio audit of `main` @ `2aad76154` (every file reference below was read, not recalled). PR #397 (`feat/studio-rebrand-phase1`, opened 2026-05-29) stays **parked**; §D4 harvested what was still unique (PR #755).

> **Mobile pass = PRD-246**, not 245: that number belongs to the session-lane workstream. See `docs/PRDS/PRD-246-STUDIO-MOBILE.md`.

## The review (2026-09-17)

1. *"I can't see the Questions tab on either local or prod."* — PRD-225's tab exists, but only in the Command Centre shell that renders when the theme is **Studio** and the viewport is **≥ 1024 CSS px** (`app/command-center/page.tsx:12-17`). The default theme is `system` (`components/providers.tsx:91`), the picker labels Studio "preview" (`components/ui/theme-toggle.tsx:66-72`), and nothing ever defaults to it. Everyone else gets the legacy `ActivityPage` (Summary · Board · Calendar · Feed · History) — no Questions, no Watchlist, no Governance; the bell's `?tab=questions` link lands there and does nothing.
2. *"Studio was parked for a while as it needs proper testing … Studio can't go live until it's fully tested."* — Parked in name, load-bearing in practice: three of the six PRD-224–229 surfaces (Questions, Watchlist, Governance) and the round-4 Command Centre (stats strip, "is it working" strip, setup checklist, live tab counts) ship **only** in Studio; `app/globals.css` carries **599** Studio-scoped lines; seven files fork on `useIsStudio`. Every page is two pages.
3. *"Could some of those be shortcuts on the /chat page … a quick view of the command centre and what Auto is doing."* — Nothing operational is shown in chat today: no consumer of `useActivityStats`, `useFleetState`, `useQuestions`, `useWatches` or `useDecisionsNeeded` exists under `components/chatbot`, `lib/chat` or `app/chat`. The only live objects are the per-message `TaskCard` (one ticket) and the Studio-only mission rail (`components/chatbot/studio-chat-shell.tsx:337-421`), which reads the active mission and nothing else.

## The second review (2026-09-17, evening — after testing Waves 1–2 on local)

The first cut of D2/D3 rendered the Studio-designed Command Centre shell, chat ledger and Assignments hub on **every** desktop theme, inside whatever chrome the theme had. On Dark that produced the Studio ledger inside the classic dark rail — *"a hybrid menu system and hybrid pages … Agents in Dark and Studio look like different apps, and chat is a mix-match of matte black and the old menu."* Matte had also been removed against *"keep them separate until fully tested."* Two facts the screenshots made plain:

1. **Studio is not a palette; it is the other design system** (paper, ink, serif headings, sectioned sidebar, breadcrumbs). Putting it in the same picker list as Light/Dark/Matte made one control do two jobs. Studio has exactly one palette today — the stylesheet has **zero** Studio-dark rules.
2. **Studio is itself a hybrid on most pages.** Only chat, Command Centre and Assignments have Studio-designed components; every other page renders the classic component inside Studio chrome with `.studio` overrides (e.g. Agent Management shows two tab strips — the Studio page tabs and the page's own).

**Gerard's decision:** *two styles, two tones.* **Classic** (Light · Dark) is the original design, rolled back to `main`, permanent. **Studio** (Light · Dark) is the full Studio redesign on **all** pages, permanent. Nothing crosses between the styles. Feature parity is delivered *in each style's own idiom*: surfaces are built from shared theme-neutral components (the Questions and Watchlist tabs already are — zero Studio classes), and only chrome and page layout are style-specific. Matte retires; Studio Dark takes the dark role on the Studio side.

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
**Rehouse + extension.** Two design systems, both kept: Classic stays as it is on `main`; Studio is completed page by page in two tones. Plus one **extension**: a live rail in chat built from reads that already exist, mounted in both styles.

## Decisions
- **D1 · Two styles × two tones (Gerard, 09-17).** The picker offers **Style** — Classic | Studio — and **Tone** — Light | Dark | System. **Defaults (Gerard, 09-17, after seeing Studio Dark on local): Studio + Dark — for everyone, once.** A browser with no stored choice gets them (the style on the server, the tone from next-themes); every browser is moved to them one time on its next load via a versioned marker (`automatos-appearance-defaults`) and picks freely afterwards. Mechanism: next-themes keeps the tone exactly as today (`light`/`dark`/`system`, the `.dark` class every Tailwind rule depends on); the style is its own persisted setting applied as the `.studio` class on `<html>` by a pre-hydration script, so switching style never changes tone and there is no classic flash. `useIsStudio()` reads the style. Matte and the `?theme=studio-preview` flag are gone (Wave 0). **Studio Dark** is new: a dark token set for the Studio system (ink ground, paper-coloured text, the same orange) plus a sweep of the hard-coded colours in the Studio rules; drafted first, judged on local.
- **D2 · Feature parity, each style in its own idiom.** Classic is never rendered with Studio components and vice versa. Surfaces are built from shared theme-neutral components; chrome and page layout are style-specific. Command Centre: the Studio shell in Studio; the Classic page carries **Watchlist · Questions · Governance** in Classic style, mounting the same tab bodies (Governance's sub-tabs via a `variant`). The bell's deep-links land in both styles. The period selector lives in the Studio shell (it always lived in the Classic page).
- **D3 · The Studio-designed pages render only in Studio; Classic rolls back to `main`.** Chat ledger shell and Assignments hub: Studio desktop only; classic chat layout and `AssignmentsPage`: Classic desktop and every style below 1024 px. `/missions` keeps its query params in every style (a fix). The Studio shell gains the `?from=assignments` back button.
- **D4 · #397 harvested (PR #755), then closed.** Four files were still unique: the playbook card's real `schedule_config` shape + cron humaniser + honest success %, dialog centring, and the two icon-style guards. Everything else was rewritten on `main` or superseded.
- **D5 · Chat gets "Auto now", a live rail, not a dashboard — in both styles.** One shared rail component whose sections are the floor's live objects, each a one-line row deep-linking to its Command Centre tab: **Working now** (`useActivityStats('1d')` + top fleet rows via `useFleetState`), **Questions** (`useQuestions`, answer inline through `useAnswerQuestion`), **Watchlist** (`useWatches`), **Decisions** (`useDecisionsNeeded` → Governance), **Mission** (today's rail). Studio mounts it in the ledger's `sh-chat-rail`; Classic mounts it in an aside of its own layout (the existing `motion.aside` pattern). No new endpoint; empty sections say so; under 1280 px a header pill with the two counts that need a human.
- **D6 · Mobile stays classic in both styles for now; a mobile pass precedes prod.** Below 1024 px the classic trees serve both styles through the Studio pass; PRD-246 is built and tested on local **before** anything reaches prod.
- **D7 · The audit's dead config is fixed (Wave 0, done).**
- **D8 · Studio: the full redesign on every page, in two tones (Gerard, 09-17).** Pages with a Studio design today: chat, Command Centre, Assignments. Pages that still render classic components under Studio chrome: Deliverables, Agent Management, Tools & Integrations, Knowledge Base, Marketplace, Analytics, Settings, Workspace Admin (Docs is external). Each gets a Studio page — a Studio layout of the shared components, no overrides on classic markup — in both tones, with the duplicated tab strips resolved (the Studio page tabs are the only tabs). Design source: there is no Figma for Studio (`FigmaDesigns/` holds academy and markets only); the Studio system lives in code — its tokens, chrome and the three designed pages — and each page is designed from it and judged on local, page by page.

## Stories
**Wave 0 — honesty (PR #753, green, passed local).** D7; chat breakpoint → 1024; Matte and the flag removed; tests for the theme hook, the tab config and the role gates.

**Wave 1 — Command Centre (PR #754, restructured 09-17).** Studio-only shell with the period selector; the Classic page gains Watchlist · Questions · Governance in its own style; bell deep-links land in both. Acceptance: Classic looks exactly as on `main` plus three tabs; `?tab=questions` lands on Questions in either style.

**Wave 2 — chat and Assignments (PR #756, restructured 09-17) + the harvest (PR #755).** Classic chat, Assignments and Playbooks are `main`'s files; the Studio shell and hub render only in Studio; `/missions` keeps params; the back button in the shell. Acceptance: Classic unchanged from `main`; Studio unchanged plus the button.

**Wave 3 — the picker and Studio Dark.** Style × Tone controls; the style class set before hydration; `useIsStudio()` on the style; a Studio Dark token draft covering the three designed pages and the chrome; tests: switching style never changes tone, System follows the OS in both styles, no classic flash on reload. Acceptance: Gerard judges Studio Dark on local; every Classic screen is byte-identical in behaviour to `main`.

**Wave 4 — "Auto now" (D5), shared component, both styles.** Sections, empty states, the small-width pill; a vitest per section on fake hook data. Acceptance: with one open question, one due watch and one running ticket, the rail shows three honest rows in either style and each row lands on the right tab.

**Waves 5a–5g — Studio on every page, two tones (D8). Built 09-17, three mechanisms by page shape:**
- **5a Agent Management (#760)** and **5b Deliverables (#761)** — bespoke Studio pages (`components/<page>/studio/…`): the editorial head, **in-page** `cc-tabs` driven by `?tab=`, the same tab bodies. With 5b the header sub-nav strip (`StudioPageTabs`, `STUDIO_PAGE_TABS`) is deleted: every Studio page composes its own tabs.
- **5c Tools & Integrations (#762)** — a **frame variant** on the existing 1,035-line component (`variant="studio"`): head, stats and toolbar swapped, the grid and modals shared.
- **5d Knowledge Base, 5e Marketplace, 5f Analytics, 5g Settings + Workspace Admin (#763, one PR)** — the **shared frame primitives** `PageHeader`, `StatsBar` and `FilterTabs` render the Studio frame in the Studio style and their classic markup otherwise, so every page built from them is a Studio page in both tones with no per-page copy (24 / 11 / 9 pages respectively, Team included).
Acceptance per page: Gerard's local pass in Studio Light and Dark; Classic byte-for-byte as on `main`. Fixes from the review land as follow-up commits on the wave branches.

**Wave 6 — the manual pass (Gerard; local first, then SaaS).** The checklist below on both styles × both tones × both editions, desktop and tablet, fresh and existing browser. CI is the only gate for code; this pass is the gate for `studio` → `main`. Then PRD-246 (mobile), then the one PR.

## Manual test checklist (the pass Wave 4 executes)

| Page | What to check |
|---|---|
| Sign-in → first paint | Style and tone persist across reload and tabs; switching one never changes the other; System follows the OS in both styles; Studio renders on first load with no classic flash; Classic renders exactly as on `main` |
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
Pick **Classic** and the app is exactly the app on `main`, in Light or Dark, with three more tabs in the Command Centre and, later, the "Auto now" aside in chat. Pick **Studio** and every page is the Studio design, in cream paper or in ink, with the same features in Studio's idiom. Nothing is "preview" any more, and nothing is a mixture.

## Open questions (Gerard's call)
1. ~~Mobile~~ — decided: PRD-246 after the Studio pass, before prod (D6).
2. ~~Matte~~ — decided: retired; Studio Dark takes the dark role on the Studio side (D1).
3. **"Auto now" cadence.** The rail inherits each hook's polling (10 s fleet, 30 s questions, 60 s watches). Fine for a pilot; a single aggregate read would cut chat's request count if it matters on SaaS. Not built unless you say.
4. ~~Harvest scope for #397~~ — decided: the four unique fixes, PR #755.
5. ~~Order of the Studio page waves~~ — decided (09-17): by visibility — Agent Management, Deliverables, Tools & Integrations, Knowledge Base, Marketplace, Analytics, Settings + Workspace Admin.
6. ~~Default style~~ — decided (09-17): Studio + Dark, for everyone, once (D1).

