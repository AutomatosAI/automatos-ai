# Automatos AI — Platform Redesign PRD

**Status:** Draft v1 · 2026-05-14
**Author:** Gerard + Claude (engineering lens) + Claude Design (visual lens)
**Direction:** Ledger Studio — locked after Round 2, validated by Round 3 mocks
**Supersedes:** none. Companion to `REBRAND_BRIEF.md`, `REBRAND_BRIEF_ROUND2.md`, `REBRAND_BRIEF_ROUND3.md`, `REBRAND_CONTEXT.md`, `REBRAND_PAGE_TIERS.md`
**Lives in:** `automatos-ai/` (this repo). Working tree of mocks/spec: `DUMPING AREA/DesignKIT/`

---

## 0 · Why this document exists

We have a locked design direction (Ledger Studio), 5 Round 3 artefacts, and 42 routes to migrate. We need a single PRD the three collaborators — Gerard, Claude (engineering), Claude Design — can work from across many sessions without losing context. The PRD splits the work into 11 phases:

- **Phase 1** — token foundation. Pure visual swap. No IA changes.
- **Phases 2 → 11** — one phase per top-10 page. Each phase examines use case, UI/UX, and requirements for that page and its sub-pages, then ships it with Claude Design's specific mocks.

After Phase 11 we sweep Tier 1–2 (the remaining 30 routes) in a single inheritance pass.

---

## 1 · Vision

> **Easy enough for a non-technical user to feel oriented in 60 seconds.**
> **Powerful enough for an operator to debug a failed mission in 5.**
> **Beautiful enough that the screenshot reads as the brand.**

These are not in tension — they're in **layers**. Studio's design language is editorial on top (serif headlines, plain-English ledes, named status pills) and operator-precise on the bottom (mono tool calls, mission IDs, cost lines). Non-techy users live on the top layer. Operators flip into the bottom layer when they need it.

### Non-negotiable principles

1. **Editorial-first per page.** Every page leads with a serif headline + plain-English lede paragraph (1–2 sentences) before any data table or technical surface.
2. **Two-layer information.** Mono detail is always present but never the default reader experience. Progressive disclosure rules from §3.
3. **Semantic-locked colour.** Burnt orange = consequence. Near-black = positive primary. Olive = good. Navy = info/queued. Tan = neutral. No drift.
4. **Auto is the human interpreter.** Where the system has technical state to expose, Auto narrates it in plain English alongside (not replacing) the technical surface.
5. **Operator power is preserved.** Cmd-K, dense tables, keyboard nav, mono detail — never removed in the name of "simplification." Hidden by default, present on demand.
6. **No emoji. No hype. No "Oops!" copy.** Operator-first voice: direct, named, receipts on every claim.

### What "non-techy easy" means in practice

The three concrete moves we apply across every Tier 0 page:

- **Move 1 — Progressive disclosure on the detail layer.** Each technical surface has a `Show details` toggle. Default view is human-readable. Toggle persists per user.
- **Move 2 — Inline glossary tooltips.** Mission IDs, event types, model names, cosine scores get hover tooltips with a plain-English definition for the first 3 sightings per user. After that, tooltip suppresses.
- **Move 3 — Auto's narration panel.** Auto runs a "what's happening" plain-English ticker alongside operational pages (mission detail, command centre, activity). Real-time, sentence-form.

Move 1 lands in every Phase 2–11. Move 2 in Phase 1 as cross-cutting infrastructure, applied per phase. Move 3 design ships **before Phase 3 build** (concurrent with Phase 3 design — see `phase1/narration.jsx`, already delivered).

---

## 2 · Top 10 pages, in phase order

| Phase | Route | Why this page | Sub-pages in scope |
|---|---|---|---|
| 2 | `/chat`, `/chat/[id]` | Most-used surface. The on-ramp. If a new user only opens one page, it's this one. | Thread list, conversation view, mission-context side rail |
| 3 | `/command-center` | The "today" view. Sets the tone for the whole product. First page post-signin. | Widgets, ticker, agent statuses, KPI sliver |
| 4 | `/missions/[id]` | Where the live work shows. The msn_8f3a failure-recovery story. | DAG view, event stream, retry queue, fork/pause modals |
| 5 | `/activity/execution` | Audit log. The operator power tool. Highest density. | Filters, anchor, replay window, export, books integration |
| 6 | `/agents`, `/agents/[id]` | Workforce management. List + detail (modal or standalone TBD). | Roster, agent config (System / Tools / Skills / Memory / Advanced / Telemetry tabs) |
| 7 | `/playbooks` | Reusable mission templates. Where the workforce becomes a system. | Playbook list, playbook detail/edit, version history |
| 8 | `/marketplace`, `/marketplace/*` | Extending the workspace. 100+ agents/skills/tools. | Browse, widget detail, publish, developer dashboard |
| 9 | `/settings` (+ sub-pages) | Workspace config. The boring-crud-survival test. | 6 sections × 6 fields. Settings/profile and settings/notifications stay as Tier 1 inheritance unless flagged. |
| 10 | `/analytics` | Cost-per-agent, cost-per-request, decision economics. Proves the value. | Time-range charts, cost breakdowns, projections |
| 11 | `/deliverables`, `/deliverables/explorer` | Outputs from missions. Where the work lives after it's done. | List view, explorer (tree/detail), deliverable detail |

**Out of scope for the 11-phase deep-dive** (inherit from Tier 1–2 sweep after Phase 11):
- `/dashboard` (likely redundant with `/command-center` — decide in Phase 3)
- `/documents`, `/tools`, `/team`, `/onboarding/wizard`, `/admin/*`, `/api-debug`, `/api-diagnostics`, `/context`, `/field-theory`, `/assignments`
- `/sign-in`, `/sign-up`, `/sso-callback`, `/accept-invitation`, `/reset-password`, `/tools/callback` (Clerk config swap)

---

## 3 · Phase 1 — Token foundation (L1+L2)

**Goal:** Ship the Studio visual system across the whole platform behind a `?theme=studio-preview` flag. No IA changes. Every page already looks ~80% rebranded before any Phase 2+ work starts.

### Source material

- Tokens: `DUMPING AREA/redesign/styles.css` (the base cream/serif/mono foundation)
- Studio scope overrides: `DUMPING AREA/round2/styles.css` (panel, pill, DAG primitives)
- Round 3 extensions: `DUMPING AREA/DesignKIT/round3/styles.css` (settings nav, chat bubbles, kit grid)
- Spec sheet: `DUMPING AREA/DesignKIT/round3/kit.jsx` (Components + States artboards)
- Semantic lock (memorise):
  - `--bg: #f4eee2` cream paper
  - `--paper: #ffffff` cards
  - `--fg: #1a1814` near-black ink (primary)
  - `--accent: #c44a1a` burnt orange (consequence)
  - `--olive: #5b6f3a` (good)
  - `--navy: #1d3658` (info/queued)
  - `--border: #dcd2bd` tan
  - serif: Tiempos Headline → Charter → Iowan Old Style → Georgia
  - mono: JetBrains Mono → SF Mono

### Deliverables

1. **`frontend/app/globals.css`** — Studio-only token set. Drop the orange/glass/dark variables. Three theme variants: `:root` (Studio light), `.dark` (Studio pitch — defer), `.matte` (defer or remove).
2. **`frontend/tailwind.config.ts`** — extend with Studio colour tokens, serif/mono font families, status icon vocabulary, spacing scale.
3. **`frontend/components/ui/*`** — restyle all 53 shadcn primitives against the spec sheet. Button (5 hierarchies × 3 sizes), Input (5 states), Card, Badge (6 semantic tones), Tabs, Segments, Dialog, Toast, Skeleton, etc. The kit.jsx is the visual ground truth.
4. **`frontend/components/shared/*`** — restyle the 12 shared components: PageHeader (serif title + eyebrow + lede paragraph), StatsBar (KPI sliver), ItemCard, FilterTabs, SearchInput, StatusBadge, EmptyState, GlobalSearch, IconSelector, PremiumIcon, ViewToggle.
5. **`?theme=studio-preview` feature flag** — opt-in via user menu ("Try Studio preview"). Toggle persists per user. Old theme remains default until Phase 11 lock.
6. **In-product `/styleguide` page** — mirror of the v2 landing styleguide, but for the platform. Documents tokens + primitives as we build them. Becomes living spec.
7. **Move 2 infrastructure** — `<GlossaryTooltip>` primitive + glossary content map (8 core terms: mission, agent, playbook, deliverable, router, skill, handoff, deliverable). First 3 hovers per term show the tooltip; after that, suppressed. Used in Phases 2–11.

### Out of scope for Phase 1

- Any layout changes
- Any IA changes
- Any new features
- Phase-specific UX moves (those land in their phase)

### Success criteria

- Every page in the platform renders without visual bugs under `?theme=studio-preview`
- Pilot user can flip between old and new theme without functional regression
- All 53 primitives match the Round 3 kit.jsx visual spec at all states
- `<GlossaryTooltip>` works on three test terms in the dev environment

### Timeline

- **8–10 working days** (revised from 5 after CD pushback — accepted). Days 1–5 cover tokens + ~70% of primitives (the high-frequency ones: Button, Input, Card, Badge, Dialog, Tabs, etc.). Days 6–10 cover the long-tail primitives (Combobox, DataTable, Sheet, Carousel, ContextMenu, etc.) where decisions get less mechanical.
- **Checkpoints:** Day 3 (easy primitives review) + Day 7 (long-tail decisions). CD on-call for ad-hoc Qs throughout.
- The kit.jsx is the visual ground truth; `phase1/microspec.jsx` (see below) pins the remaining implicit decisions.

### Phase 1 micro-spec — locked

`DUMPING AREA/phase1/microspec.jsx` (CD-delivered) pins the implicit visual decisions that kit.jsx left to engineer judgment. Engineering reference, build to this. Locked rules:

- **Card hover** — 220ms ease. Border darkens to `--border-2 (#c9bea4)`. Shadow lifts to `0 4px 12px rgba(60,40,10,.06)`. Background stays `--paper`. No translate, no scale.
- **Card-as-button focus** — `outline: none; box-shadow: 0 0 0 3px rgba(91,111,58,.18)` (olive @ 18%). Only on `:focus-visible`. Non-interactive cards do NOT get a focus ring.
- **Ticker height** — 36px including 1px bottom border. Vertical-center text.
- Plus 4–5 additional pinned decisions (toast positioning, modal scrim opacity, segmented control disabled state, table row hover) — see file.

### Claude Design asks during Phase 1

- **§9.1 RESOLVED** — `phase1/microspec.jsx` delivered. Pings only on net-new edge cases.
- **Open:** final lock on the `.dark` (Studio pitch) variant — build it in Phase 1 or defer? CD: please call.

---

## 4 · Per-phase template (used for Phases 2–11)

Each subsequent phase fills this template. Phases 2 and 3 are filled in detail below as exemplars; Phases 4–11 are stubs that Claude Design and Claude (engineering) fill in collaboration with Gerard before that phase starts.

```
### Phase N · /page-route

A. Purpose (1 line)
   Why this page exists for the user. What it lets them do.

B. Use case — who, when, why, what they need
   — Primary persona (one of: pilot operator · tooling lead · ops-finance buyer · non-techy user)
   — When they open it (frequency, trigger)
   — What they want to accomplish in <60 seconds
   — What they want to accomplish across a session

C. Current state (what's there today)
   — Brief description of existing IA and what works/doesn't
   — Known confusing elements (the things a non-techy user trips on)

D. Target state (what we're building)
   — New IA in 3–5 bullets
   — Editorial-first treatment: serif headline + lede paragraph
   — Detail-layer surfaces and where they live

E. Sub-pages in scope
   — Routes covered in this phase
   — Modals/overlays covered
   — Out-of-scope explicit list

F. Non-techy moves applied
   — Move 1 (progressive disclosure): what gets the toggle
   — Move 2 (glossary tooltips): which terms
   — Move 3 (Auto narration): does this page get a narration panel? If yes, what does it say?

G. Round 3 mock pointer (or Round 4 if commissioned)
   — Which .jsx in DesignKIT/round3/
   — What's missing that needs Round 4

H. Build deliverables
   — Files to create/modify
   — Components reused vs. new

I. Success criteria
   — Behavioural parity checks (golden-path QA list, 3–5 user actions)
   — Visual parity checks (matches Studio mock at desktop + tablet)
   — Non-techy readability check (60-second-orient test)

J. Open questions for Claude Design
   — 2–4 specific design questions only Claude Design can answer

K. Estimated timeline
   — Design (Claude Design): N days
   — Build (Claude eng): M days
   — QA (Gerard): O days
```

---

## 5 · Phase 2 — `/chat`, `/chat/[id]` (the on-ramp)

**A. Purpose.** Talk to your agents in plain English. Auto routes the request, runs the work, reports back. The first page a new user touches.

**B. Use case.**
- **Primary persona:** non-techy user (and pilot operator returning daily)
- **When:** every session. Average 8–15 conversations per day per operator.
- **<60 second goal:** type a request like "draft the Q3 update" and see Auto start handling it
- **Session goal:** review thread history, fork conversations, share with team, escalate to a mission

**C. Current state.** Existing chat is a single-column conversation. No thread list, no mission context, no inline tool calls visible. Streaming bubbles. Probably feels like ChatGPT — but doesn't show *what Auto is doing*.

**D. Target state.** 3-column layout per Round 3 mock:
1. **Left rail — Threads (220px).** Recent missions/conversations with status pip + mono mission ID
2. **Centre — Conversation.** Serif headline of active mission, status pill, user/Auto bubbles with inline `tool.call` cards (mono detail, olive border = good, orange border = error)
3. **Right rail — Mission context (280px).** Pipeline steps preview, deliverables list, "Open mission detail" CTA

**E. Sub-pages in scope.**
- `/chat` (new thread)
- `/chat/[id]` (active thread)
- Inline tool-call detail expand
- Out of scope: agent-specific chat (deferred)

**F. Non-techy moves applied.**
- **Move 1:** inline `tool.call` cards have a `Show technical details` toggle. Default: "Halberd pulled the Q3 payload from logs." Toggle on: full mono `research.collect` payload preview
- **Move 2:** glossary on first sightings of "Auto", "mission", "handoff", "tool call"
- **Move 3:** Auto narration is *native* to chat — every Auto bubble IS the narration. No separate panel.

**G. Round 3 mock pointer.** `DesignKIT/round3/chat.jsx`. Mock is complete; no Round 4 needed.

**H. Build deliverables.**
- `frontend/app/chat/page.tsx` — restructure to 3-column grid
- `frontend/app/chat/[id]/page.tsx` — restructure to 3-column grid with active mission context
- New: `components/chat/ThreadList.tsx`, `MissionContextRail.tsx`, `ToolCallCard.tsx` (with toggle), `ChatComposer.tsx`
- Restyle: `ChatBubble.tsx`

**I. Success criteria.**
- Golden path: open chat → type request → see Auto respond with tool calls inline → click "Open mission detail" → land on Phase 4 page
- Visual parity with `round3/chat.jsx` desktop + tablet
- New user orient test: a non-techy user can describe what the page does after 60 seconds without help

**J. Resolved by Claude Design (§9.2).**
1. **Mobile collapse order** — right rail (mission context) collapses first behind a `</>` button. Left rail (threads) collapses second as slide-in from edge. Centre stays. Below 640px, threads becomes a top sheet.
2. **Composer model selector** — closed = mono "Model · Auto · routes best-fit" label. Click opens a popover with the 5-tier list + cost/tok per tier. Override is per-message; resets on send.
3. **Forked thread visualisation** — new thread header carries the parent `msn_8f3a` mono pill below the title. Hover reveals the parent message snippet. Parent thread gets a `→ forked` line where the fork happened.

**K. Timeline.** Design Q&A 0d (resolved) · Build 4d · QA 1d = **~1 week**.

---

## 6 · Phase 3 — `/command-center` (the "today" view)

**A. Purpose.** First page post-signin. Glance at what your workforce did, is doing, and needs your attention on. The mission-control dashboard.

**B. Use case.**
- **Primary persona:** pilot operator, every morning
- **When:** session-start, plus 2–4 check-ins per day
- **<60 second goal:** know whether anything is failing right now and whether the night was clean
- **Session goal:** triage errors, replay missions, open the books, fork a recovery flow

**C. Current state.** Existing command centre has draggable widgets — stats + activity feed + agent statuses. Feels like a tools palette. Doesn't lead with a narrative of "is the workforce healthy."

**D. Target state.** Per Round 2 Pass-2 Ledger Studio mock:
1. **Top — Ticker bar.** 7 KPIs (UPTIME, CACHE-HIT, $/DEC, P50, ERR/HR, T2.5 HIT, QUEUE) updating every 5s
2. **Headline — serif greeting + status.** "Good morning, Yann. Your workforce is running, with one held step."
3. **Lede paragraph.** "3 of 5 launch-mission steps complete. 1 failed · retry in 23s. Pilot tracking 68% cache-hit and $0.0027 per routing decision vs target."
4. **4 KPI cards** with sparklines (CACHE HIT, $/DEC, P50 LATENCY, ERRORS LAST 1H)
5. **Live taps section** — last 9 audit rows (anchored to current incident)
6. **Routing T-tier panel** — which tier handled what
7. **Agents leaderboard** — top 5 by success rate

**E. Sub-pages in scope.**
- `/command-center` only
- Decide in this phase: is `/dashboard` redundant? If yes, kill it.

**F. Non-techy moves applied.**
- **Move 1:** the live taps section has a `Show details` toggle for the mono columns
- **Move 2:** glossary on first sightings of "router", "T2.5", "cache hit", "$/decision"
- **Move 3:** Auto narration runs in a slim panel below the ticker — "Sentinel just tried to open a PR but branch protection blocked it. I've queued a retry that will ask you to approve."

**G. Mock pointers.** Round 2 Pass-2 Studio Command Centre + **`DUMPING AREA/phase1/narration.jsx`** (CD-delivered Move-3 mini-round, includes 03 command-centre placement + 04 mission-detail placement + 05 voice cadence reference). Round 4 NOT required for this phase.

**H. Build deliverables.**
- `frontend/app/command-center/page.tsx` — new layout
- New: `components/dashboard/TickerBar.tsx`, `KpiCardWithSparkline.tsx`, `RoutingTierPanel.tsx`, `AgentLeaderboard.tsx`, `AutoNarrationPanel.tsx`
- Decide: kill `/dashboard` and add a redirect to `/command-center`

**I. Success criteria.**
- 60-second orient test passes for non-techy
- Operator can identify all failing/queued items in <5 seconds
- Ticker updates without layout shift

**J. Resolved by Claude Design (§9.3).**
1. **Auto narration panel** — slim strip below the ticker, paper card, ~52px tall. Format: `[avatar-A] [serif italic phrase in olive] · [mono · 14s ago]`. Example: *"Sentinel just tried to open a PR but branch protection blocked it. I've queued a retry."* Phrases live ~30s then fade. Click to expand into 5-message context drawer. Full design: `phase1/narration.jsx`.
2. **Ticker collapse** — below 1024px scrollable horizontally with left-edge fade. Below 640px collapses to ERR/HR + $/DEC + a chevron that opens the rest as a sheet.
3. **Leaderboard sort** — mono `↓` arrow next to active sort column header. Click to flip. Three sort options: success rate, tasks, spend.
4. **Day-one empty state** — kit "all clear" pattern. Copy: *"Your workforce is ready."* + near-black primary `Hire your first agent`. Status pill says `READY` in muted tan, not olive (no work happening yet = no good news to report).

**K. Timeline.** Design 0d (resolved) · Build 5d · QA 1d = **~1 week**.

---

## 6a · Track A decision — the on-ramp (resolve before Phase 2 build)

**The tension CD surfaced:** Phase 2 calls `/chat` "the on-ramp" but `/command-center` is the actual first page post-signin. These can't both be the on-ramp. The decision has more leverage than any other in the redesign — it sets what a new user sees in their first 10 seconds.

Three options. We must pick one before Phase 2 ships, ideally during Phase 1 build:

### Option A — Redirect new users to `/chat` on first session
A `hasCompletedFirstChat` flag flips after their first message. After that, `/command-center` becomes home. Pro: conversation is the most-forgiving on-ramp; non-techy users start in plain English. Con: first impression is "this is just another chatbot" — until Auto responds and shows the workforce.

### Option B — Sticky-collapsed chat composer on `/command-center`
The composer lives on command-centre as a slim bar, expands on focus, posts to `/chat/[new]`. Pro: keeps command-centre as home for everyone; gives the conversation affordance prominent placement. Con: command-centre is already dense; adds a surface to design.

### Option C — First-run wizard
3-step wizard (workspace name → first agent install → "what should I ask Auto to do?"). Ends in chat. Pro: explicit onboarding; teaches the model. Con: another flow to design + maintain; users hate forced wizards.

**Default recommendation (write up, but Gerard decides):** **Option B**, because:
- The command-centre's serif headline + lede is already the "easy AND informative" first impression we want — putting a composer there extends that surface, doesn't fork it
- Avoids a flag-based redirect that makes day-2+ behaviour confusing
- Wizards are a known anti-pattern for operator-flavoured tools

**Action.** Add to Phase 1 mini-spec / Phase 3 design: a sticky-composer treatment on command-centre. CD: low-cost extra artboard. Gerard: pick A/B/C by end of Phase 1.

---

## 7 · Phases 4–11 — stubs (to be filled phase-by-phase)

For each remaining phase, the template in §4 is filled at the START of that phase, not now. Doing them all upfront produces a stale doc by Phase 6. The phase opens with a working session between Gerard, Claude (engineering), Claude Design.

What's locked now per phase:

### Phase 4 · `/missions/[id]`
- Mock: `DesignKIT/round3/` (missions wasn't in Round 3, use Round 2 ledger.jsx + studio.jsx)
- Non-techy moves: all three apply heavily — this is the operator-deep surface that non-techy users need translation on
- Open question: standalone route vs modal-from-/missions list?

### Phase 5 · `/activity/execution`
- Mock: `DesignKIT/round3/audit.jsx` ✓ complete
- Non-techy moves: Move 1 critical (toggle between "human events" and "raw log"), Move 2 critical (glossary), Move 3 optional (could be a sidebar narration)
- Open question: do we ship the "Open in books" CTA in Phase 5 or defer until /analytics?

### Phase 6 · `/agents`, `/agents/[id]`
- Mock: Round 2 atlas.jsx (Agent Config modal). Confirm whether `/agents/[id]` is a modal-from-list or a standalone route — Track B candidate to decide
- Non-techy moves: Move 2 on form labels (what is "max tokens"? what is "temperature"?)

### Phase 7 · `/playbooks`
- Mock: not commissioned. **Round 4 candidate.**
- Non-techy moves: Move 3 — Auto can suggest playbooks from natural-language descriptions

### Phase 8 · `/marketplace`, `/marketplace/*`
- Mock: Round 1 marketplace + Round 2 already done. Sub-pages need Round 4.
- Non-techy moves: Move 2 on capability tags, Move 1 on the publish form

### Phase 9 · `/settings`
- Mock: `DesignKIT/round3/settings.jsx` ✓ complete
- Non-techy moves: Move 2 on every setting label (the descriptions are already good — extend with hover glossary)

### Phase 10 · `/analytics`
- Mock: not commissioned. **Round 4 candidate.** This is the buyer-conviction page.
- Non-techy moves: this is the page where non-techy and operator views diverge most — likely needs two presentation modes

### Phase 11 · `/deliverables`, `/deliverables/explorer`
- Mock: not commissioned. **Round 4 candidate.**
- Non-techy moves: Move 1 on the explorer detail panel

---

## 8 · Cross-cutting concerns

### 8.1 · The `?theme=studio-preview` feature flag

- Defaults to old theme; opt-in via user menu ("Try Studio preview")
- Persists per user
- Both themes share the same routes — no code duplication
- Old theme stays available until Phase 11 lands and we're confident
- After Phase 11: flip the default; offer a "back to classic" toggle for 60 days, then remove

### 8.2 · QA approach

- **Per-phase golden-path manual QA.** 3–5 user actions per page, run by Gerard after each phase.
- **Screenshot diff** on the 10 top-tier pages with Playwright + pixelmatch
- **Non-techy readability check** per phase — read the page back to a non-pilot user and ask "what does this do?" — if they can't answer in 60s, the page fails

### 8.3 · Voice and copy rules

- Editorial top: serif headline + 1–2 sentence lede paragraph per page
- No exclamation marks, no questions in headlines, no "Oops!" / "Welcome!" / emoji
- Status pills are the only place icons appear in copy: `✓` done · `!` error · `↻` queued · `◐` running · `·` pending · `◦` paused
- British spelling (centre, optimise, customise) per existing brand
- Auto is a proper noun. Never "the assistant" / "the AI"

### 8.4 · Tier 1–2 inheritance sweep (post-Phase 11)

The 30 remaining routes inherit from L1+L2 primitives and shared components. No bespoke design. One-week sweep:
- Visual cleanup pass per page (1–2 hours each)
- Screenshot diff catches regressions
- Anything that fails the orient test gets flagged for Round 5

### 8.5 · Track B — IA and flow improvements

Parallel backlog. As we go through Phases 2–11, we **write down** every "this page shouldn't exist" / "these two screens should merge" / "this nav makes no sense" observation. We do NOT act during Phases 2–11. Track B opens after Phase 11 stabilises.

Current candidates:
- `/dashboard` vs `/command-center` redundancy (decide in Phase 3)
- `/sign-in/*` vs `/auth/signin/*` duplicates (decide in Tier 3 Clerk pass)
- `/context`, `/field-theory`, `/assignments` — still stabilising in product

---

## 9 · Claude Design's view — RESOLVED (round 0.5 reply, 2026-05-14)

CD reviewed the PRD draft and replied with three pushbacks + direct answers to every §9 ask. All resolved. Outcomes:

1. **§9.1 — Phase 1 ambiguity → RESOLVED.** CD delivered `phase1/microspec.jsx` (two sheets pinning card hover, focus rings, ticker height, toast positioning, modal scrim opacity, segmented control disabled state, table row hover). Engineering reference for L1 build.
2. **§9.2 — Chat (Phase 2) → RESOLVED.** All 3 answers locked into §5.J: mobile collapse order, composer model selector behaviour, forked thread visualisation.
3. **§9.3 — Command Centre (Phase 3) → RESOLVED.** All 4 answers locked into §6.J: Auto narration panel spec, ticker collapse, leaderboard sort, day-one empty state.
4. **§9.4 — Round 4 commission → CONFIRMED.** Batch A (Playbooks + Analytics, ~6 artboards) lands before Phase 7 design. Batch B (Deliverables + Marketplace sub-pages, ~7 artboards) lands before Phase 11 design.
5. **§9.5 — Move 3 mini-round → DELIVERED.** `phase1/narration.jsx` shipped — 3 artboards: command-centre placement, mission-detail placement, voice-cadence reference sheet. Turnaround: 1 working day as promised.
6. **§9.6 — Phase 1 availability → CONFIRMED.** Ad-hoc Qs any time. Day-3 + Day-7 checkpoints booked.

### CD's three pushbacks (accepted into PRD)

- **Phase 1 timeline: 5d → 8–10d.** Long-tail primitives (Combobox, DataTable, Sheet) bleed into days 6–10. Updated §3.
- **The on-ramp tension.** `/command-center` is the first page post-signin, not `/chat`. Demands a Track A decision. New §6a added with three options + default recommendation (Option B: sticky composer on command-centre). Gerard decides by end of Phase 1.
- **Move 3 timing.** Run mini-round before Phase 3 *build*, not before Phase 3 *design* — concurrent. Updated §1. Already obsolete because CD shipped it the same day.

### Path note

CD flagged: their internal deliverables live at `round3/` in their project; in this repo they're staged at `DUMPING AREA/DesignKIT/round3/` and the new Phase 1 + Move 3 artefacts at `DUMPING AREA/phase1/`. References in this PRD use the local repo paths.

### Open before Phase 1 starts

- [ ] Decide: build `.dark` Studio pitch variant in Phase 1 or defer? (CD waiting on Gerard's call)
- [ ] Decide: on-ramp option A / B / C from §6a? (Gerard, by end of Phase 1)

---

## 10 · Approval gate

This PRD locks when:
- [ ] Gerard signs off on the phase order and top-10 selection
- [ ] Claude Design responds to §9 with the asks resolved or scheduled
- [ ] Claude (engineering) confirms Phase 1 timeline is achievable

After lock: Phase 1 starts. Document is then **append-only by phase** — each phase adds its filled-in §4 template under §11 below, plus a "what we learned" retro paragraph.

---

## 11 · Phase journal (append-only as we ship)

### Round 0.5 · PRD review with Claude Design — 2026-05-14

**What happened.** PRD v1 drafted. CD reviewed, called the 11-phase shape correct and the three non-techy moves "the strongest thing in this doc." Pushed back on three items (Phase 1 timeline, the on-ramp tension, Move 3 timing) and answered all §9 asks with locked specs the same day. Also pre-shipped two deliverables:

- `DUMPING AREA/phase1/microspec.jsx` — Phase 1 build spec (component states, focus rings, edge-case primitives)
- `DUMPING AREA/phase1/narration.jsx` — Move 3 mini-round (3 artboards: cmd-centre + mission + voice cadence)

**What we learned.** CD's review made every Phase 1 + Phase 2 + Phase 3 decision concrete in one round. The collaborative model — PRD draft → CD review → spec lock → build — is working at the cadence we hoped. Open: Gerard's two pending decisions (on-ramp option, `.dark` variant) gate Phase 1 start.

**Phase 1 status:** pre-lock. Awaiting Gerard's two decisions, then build starts.

---

## Appendices

- **A.** Top-10 page route list with current line counts: see `find frontend/app -name "page.tsx"` (40 total)
- **B.** Shadcn primitive list: `frontend/components/ui/*` (53 files)
- **C.** Shared component list: `frontend/components/shared/*` (12 files)
- **D.** Studio token foundation: `DUMPING AREA/redesign/styles.css`
- **E.** Studio scope overrides: `DUMPING AREA/round2/styles.css`
- **F.** Round 3 mocks: `DUMPING AREA/DesignKIT/round3/`
- **G.** Semantic lock summary: `kit.jsx` Sheet 1 status-vocab block
- **H.** Phase 1 micro-spec (CD-delivered): `DUMPING AREA/phase1/microspec.jsx`
- **I.** Move 3 Auto narration mini-round (CD-delivered): `DUMPING AREA/phase1/narration.jsx`
- **J.** Phase 1 styles extension: `DUMPING AREA/phase1/styles.css`
