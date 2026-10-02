# Platform Page Audit — Studio Rebrand Gap Analysis

**Date:** 2026-05-15
**Branch:** `feat/studio-rebrand-phase1`
**Coverage:** 40 routes, organised by tier × phase
**Companion doc:** `PRD_PLATFORM_REDESIGN.md`

---

## 0 · How to read this doc

Each entry follows a fixed template:
- **Current state** — what the page looks like today
- **Studio target** — what it should look like after its phase ships
- **Gap** — specific changes needed
- **Phase / Priority** — PRD phase number + tier + use frequency
- **Effort** — S (≤1 day) / M (2–4 days) / L (1+ week)
- **Mock pointer** — which mock file to build against (or "no mock yet")

Routes are grouped by tier (hot path → long tail → out of scope). Within each tier, ordered by PRD phase.

**Use this doc as the working punch list for Phases 2–11.** As each route lands, mark it ✅ in the heading.

---

## 1 · Executive summary

**Three findings dominate this audit:**

1. **The Clerk `colorPrimary` is hardcoded to legacy orange `#ff6b35`** in `frontend/components/providers.tsx` (line 40). This affects every Clerk-rendered surface — sign-in, sign-up, accept-invitation, the user-avatar ring in the top nav. **Single 5-minute change unblocks all of Tier 3.**

2. **Pervasive orange + glass legacy** in custom auth/onboarding flows (`/sign-up`, `/reset-password`, `/accept-invitation`, `/onboarding/wizard`, `/tools/callback`). These pages hardcode `text-orange-500`, `shadow-orange-500/20`, and `glass-card` classes — they don't pick up Studio tokens automatically. Each needs a per-page colour pass.

3. **Editorial-first treatment isn't adopted anywhere yet.** Every page that uses `<PageHeader>` still passes `title + titleAccent + subtitle` (the classic API). None pass `eyebrow + lede` (the new Studio API added in Phase 1). Wiring up editorial-first on the 10 hot-path pages is the highest-leverage move for the "looks like the Round 2 mocks" outcome.

**Two cross-cutting Track B items surfaced:**
- `/auth/signin` and `/auth/signup` are legacy redirects to `/sign-in` and `/sign-up`. Consolidate post-rebrand.
- `/dashboard` and `/command-center` have ≥60% functional overlap. Decision needed in Phase 3 (see PRD §6a Track B).

**Bottom line:** the Phase 1 token swap is healthy. To close the visual gap to the Round 2 mocks, we need (a) the Clerk fix, (b) the orange-hardcoding sweep, and (c) Phases 2–11 page-by-page work as planned.

---

## 2 · Cross-cutting issues (fix once, fixes many)

### A. Clerk `colorPrimary` blocker
**Where:** `frontend/components/providers.tsx` line 40
```ts
colorPrimary: '#ff6b35',     // legacy bright orange
```
**Studio replacement:** `#1a1814` (near-black, positive primary) — or read theme via `useIsStudio()` and conditionally swap.
**Impact:** All Clerk surfaces — sign-in, sign-up, accept-invitation, user-menu avatar ring, OAuth callback screens.
**Effort:** S (single file, ~10 lines).
**Recommendation:** ship in Phase 1.5 polish pass.

### B. Orange hardcoding in custom auth/onboarding
**Where:** `/sign-up`, `/reset-password`, `/accept-invitation`, `/onboarding/wizard`, `/tools/callback`
**Pattern:**
- `text-orange-500` for spinners
- `shadow-orange-500/20`, `shadow-orange-500/40` for button glow
- `gradient-accent` class (renders orange gradient — neutralised under `.studio` but still rendered in default theme)
- Hardcoded dark gradients (`from-black via-zinc-900`) for full-page backgrounds — don't inherit from `--background`
**Fix:** systematic replace pass — orange → semantic Studio colour (navy for info/loading, olive for success, burnt orange only for consequence), remove `shadow-orange-*`, replace dark gradients with `bg-background`.
**Effort:** M (across 5 routes).

### C. Glass card pattern still everywhere
**Pattern:** Many pages use `glass-card`, `glass-panel`, `shadow-2xl`, `backdrop-blur`.
**Under `.studio`:** I already neutralised these in `globals.css` (flatten to paper card with tan border, kill shadow/glow). Cards render correctly under Studio scope.
**Issue:** Some custom pages bypass `.glass-card` and apply dark glass inline (`bg-zinc-900/80 border border-zinc-800`). These do NOT inherit the Studio fix.
**Fix:** per-page audit to remove inline dark-glass styling. Mostly affects auth pages.
**Effort:** M.

### D. PageHeader editorial API not adopted anywhere
**Where:** every page using `<PageHeader>` still uses `title + titleAccent + subtitle`.
**New API (Phase 1):** `eyebrow` (mono uppercase label) + `lede` (1–2 sentence editorial paragraph).
**Fix:** as each phase ships, update its page to pass `eyebrow` + `lede` instead of `subtitle`. Mechanical change per page.
**Effort:** S per page. Spread across Phases 2–11.

### E. Icon registry full-saturation
**Where:** agent cards (`/agents`), tool logos (`/tools`, `/marketplace`)
**Pattern:** SVG/PNG icons render at full original saturation — bright primary colours clash with Studio cream.
**Fix:** Studio-scoped CSS filter, e.g.
```css
.studio .agent-icon img,
.studio .tool-logo img {
  filter: saturate(0.65) brightness(0.95);
}
```
**Effort:** S (one CSS rule, ship in Phase 1.5 polish).

### F. Brand mark (orange sailboat) is bright orange
**Where:** top-left of every page; `/brand/automatos-mark-hi.png`
**Fix options:**
1. Add Studio variant asset (`/brand/automatos-mark-studio.png`) — warm brown/burnt orange on cream
2. Apply CSS filter under `.studio` to desaturate the bright orange
3. Keep as-is (single brand-identity anchor, accepts the legacy orange tone)
**Recommendation:** option 1 (cleanest). Defer to Phase 1.5 polish.
**Effort:** S (asset + one CSS swap).

### G. Duplicate auth routes (Track B)
- `/auth/signin/[[...rest]]` → redirects to `/sign-in`
- `/auth/signup/[[...rest]]` → redirects to `/sign-up`
**Action:** consolidate post-rebrand. Remove legacy routes, update any deep links. Out of scope for Phase 1–11.

---

## 3 · Tier 0 — Hot path (10 routes · Phases 2–8 + 11)

These are where pilot users live every day. Each gets a bespoke design + build pass per the PRD phase plan. Order matches PRD §2 phase order.

### /chat ✅ Phase 2 mock ready

**Current state.** `<Chat>` component wrapped in MainLayout with a client-side history sidebar (Sheet on mobile, AnimatePresence panel on desktop). 20-chat history grouped by date. State managed locally (currentChatId/Messages/selectedChat). Deep-link via `?mode=plan&from=assignments`. Dark glass theme, orange accents, no PageHeader lede.

**Studio target.** Per `round3/chat.jsx`: cream background, serif "Launch · Q3 product update" headline + mission ID pill in mono. 3-column layout: (1) left threads list with semantic dot indicators (olive/orange/navy), (2) main chat with human bubbles (sans) + Auto/agent bubbles (with inline tool-call detail in mono showing timestamp/model/duration/cost/status icon), (3) right rail with mission KPIs (3/5 steps, elapsed, spend, retry countdown) + mini-DAG + deliverables list. Breadcrumb bar at top.

**Gap.**
- No editorial PageHeader (eyebrow + lede)
- Sidebar: no status dot indicators per thread
- Chat bubbles: no inline tool-call detail surfaces
- Right rail (KPIs + mini-DAG + deliverables) does not exist
- Breadcrumb navigation missing
- Border colours: `gray-800` → `--border` (tan)

**Phase / Priority.** Phase 2 / Tier 0 hot / 20+ min per session (highest frequency)
**Effort.** L
**Mock pointer.** `DUMPING AREA/DesignKIT/round3/chat.jsx`

---

### /chat/[id] ✅ Phase 2 mock ready

**Current state.** Async server component fetching chat metadata + messages. Renders AppSidebar + Chat in a flex container with dark glass borders. No PageHeader, minimal back navigation.

**Studio target.** Same 3-column layout as `/chat` but pre-loaded for a specific thread. PageHeader with conversation title (serif), started-by metadata (mono small), and mission context. Right rail KPIs/DAG/deliverables active. Breadcrumb back to `/chat` or `/command-center`.

**Gap.**
- No PageHeader on this specific route
- AppSidebar not wired for chat selection on this route
- Right rail panel missing entirely
- No breadcrumb
- Dark `border-gray-800` → `--border`
- `isReadonly` prop set to `false` but ambiguous intent — confirm

**Phase / Priority.** Phase 2 / Tier 0 hot / 5–15 min per session
**Effort.** M
**Mock pointer.** `DUMPING AREA/DesignKIT/round3/chat.jsx` (same template)

---

### /command-center ⚠️ /dashboard overlap decision needed

**Current state.** Renders ActivityPage (a tabbed dashboard: Summary, Board, Calendar, Feed, History). Summary uses lazy-loaded CommandCentreDashboard. PageHeader without lede. Dark glass theme.

**Studio target.** Per Round 2 Pass-2 Studio Command Centre: cream paper, serif greeting *"Good morning, Yann. Your workforce is running, with one held step."*, lede with embedded stats, ticker bar (7 KPIs), Auto narration strip (Move 3), 4 KPI cards with sparklines, live taps audit slice, routing T-tier panel, agents leaderboard.

**Gap.**
- PageHeader needs eyebrow + lede
- No ticker bar component
- No Auto narration strip (design exists at `phase1/narration.jsx`)
- KPI cards have no sparklines
- Stats card semantics: Working Now → navy, Active → olive, Needs Attention → burnt orange
- Dashboard panels flatten from glass to paper + tan border
- Agent leaderboard sort affordance missing
- Day-one empty state needs implementation ("Your workforce is ready" per CD's spec)

**Phase / Priority.** Phase 3 / Tier 0 hot / Daily — first page post-signin
**Effort.** L (1.5 weeks)
**Mock pointer.** Round 2 Pass-2 Studio Command Centre + `DUMPING AREA/phase1/narration.jsx`
**Cross-link:** **Decide if `/dashboard` is redundant** (see Tier 1 entry below). If redundant, kill `/dashboard` in this phase.

---

### /missions/[id]

**Current state.** Server async route → MissionDetailPage. ResizablePanelGroup splits DAG canvas (MissionDAGCanvas) + right panel (activity feed / field viz). MissionBudgetBar shows token spend. No PageHeader, dark orange/glass.

**Studio target.** Per Round 2 studio.jsx (Mission Detail): cream, serif headline with mission ID, mono eyebrow. 2-column split: left = MissionBudgetBar + DAG canvas (task nodes coloured by status: olive done, navy queued, orange error, grey pending), right = KPI section (3/5 steps · $spend · elapsed · retry in) + mini-DAG + Activity Feed (mono detail rows). Flat tan-bordered surfaces throughout.

**Gap.**
- No editorial PageHeader
- MissionBudgetBar uses orange — must be olive for spend metric
- DAG node colours not semantically mapped to Studio palette
- Right panel KPI section missing formal structure
- Activity feed rows: agent name needs serif, IDs/timestamps/costs need mono
- ResizableHandle styling tan-bordered
- No glossary tooltips on mission IDs / agent names (Phase 1 primitive shipped but not wired in)

**Phase / Priority.** Phase 4 / Tier 0 hot / 10–30 min per session
**Effort.** M
**Mock pointer.** `DUMPING AREA/round2/studio.jsx` + `DUMPING AREA/DesignKIT/round3/audit.jsx` (event row styling)

---

### /activity/execution ✅ Phase 5 mock ready

**Current state.** ExecutionKitchen — 9-stage playbook execution log with streaming events, progress bars, stage-filtered log table. PageHeader says "Playbook Execution" / "Monitor live playbook run progress". Hardcoded `#ff6b35` etc. No ticker.

**Studio target.** Per `round3/audit.jsx`: cream, serif "Activity" headline + mono eyebrow *"Execution log · today, 09:00 → now · tick 5s"*. Ticker bar at top. Filter pill bar with semantic colours and counts. Table columns: TIMESTAMP (mono) · LVL (pill) · Agent (serif 600) · EVENT (mono) · TOOL · TARGET (mono navy) · Detail · MISSION (mono accent for anchor) · COST (mono right-align). Row highlighting for anchor mission. Footer with row count + streaming + spend totals.

**Gap.**
- PageHeader missing eyebrow
- No ticker bar above content
- Table column styling mismatched to audit.jsx spec
- Stage progress bars use hardcoded `#ff6b35` — replace with semantic CSS variables
- Filter chips need semantic pill styling
- Footer text needs mono detail treatment
- No glossary tooltips on event types / agent names / tool IDs

**Phase / Priority.** Phase 5 / Tier 0 hot / 5–20 min per active run
**Effort.** M
**Mock pointer.** `DUMPING AREA/DesignKIT/round3/audit.jsx`

---

### /agents

**Current state.** AgentManagement: PageHeader ("My" + "Agents"), StatsBar (4 stats), FilterTabs (roster/configuration/skills/org-chart), SearchInput. Glass-card styling, legacy orange across all UI. No editorial framing.

**Studio target.** Editorial PageHeader (eyebrow "WORKFORCE MANAGEMENT" + lede), paper-card stats with semantic colours (Active = olive, Needs Attention = burnt orange), paper-card filter tabs with hover state, mono detail layer on agent IDs / model versions / status codes.

**Gap.**
- No editorial PageHeader; gradient-text on titleAccent must be neutralised (already done under `.studio` but render fallback)
- Stat cards still use legacy orange/blue — need semantic lock
- Filter tabs lack Studio paper transitions
- Roster / configuration tabs hardcoded to light glass
- No mono detail layer on IDs / model names
- Search input + filter chips still orange-focused
- Agent icon backgrounds full saturation (see cross-cutting E)

**Phase / Priority.** Phase 6 / Tier 0 hot / Very high frequency
**Effort.** L
**Mock pointer.** `DUMPING AREA/DesignKIT/round3/settings.jsx` (form density pattern) + Round 2 atlas.jsx (Agent Config modal)

---

### /marketplace

**Current state.** MarketplaceHomepage (not directly audited; inferred from /marketplace/widgets). Likely featured grid + category filters + search + widget cards. Glass-card legacy orange.

**Studio target.** Editorial PageHeader (eyebrow "WIDGET ECOSYSTEM" + lede on extending workspace via widgets). Featured carousel in paper cards. Category browse in paper-badge chips. Studio search input. Card hierarchy: icon + serif title + sans description + mono metadata footer + pricing badge (olive=free, info=paid).

**Gap.**
- PageHeader needs editorial framing
- Widget cards lack Studio polish (paper bg, hover states)
- Featured carousel pattern undefined
- Pricing badges + install counts need semantic colours + mono fonts
- Category chips need Studio styling

**Phase / Priority.** Phase 8 / Tier 0 hot / High frequency
**Effort.** L
**Mock pointer.** No dedicated mock — Round 4 batch B

---

### /marketplace/widgets

**Current state.** PageHeader (title="Widget", titleAccent="Marketplace"), search bar, featured carousel, category filter chips, sort dropdown, widget grid, pagination. Glass-card legacy orange.

**Studio target.** Editorial PageHeader + paper search + paper filter chips + paper featured carousel + paper widget grid + Studio pagination buttons.

**Gap.**
- PageHeader basic — needs eyebrow + lede
- Search input needs paper bg + muted border + serif placeholder
- Featured "Featured" badge orange — should be paper with no accent
- Category chip selected state still orange — switch to near-black background
- Widget cards use glass — paper styling needed
- Pagination buttons legacy primary

**Phase / Priority.** Phase 8 / Tier 0 hot / Very high frequency
**Effort.** M
**Mock pointer.** No dedicated mock — Round 4 batch B

---

### /deliverables

**Current state.** Tab-based hub: PageHeader "Deliverables", FilterTabs (outputs/blogs/templates/explorer), tab content renders OutputsFeed (gallery) / DeliverablesBlog / TemplateManager / routes to /deliverables/explorer.

**Studio target.** Editorial PageHeader (eyebrow "YOUR OUTPUTS" + lede on deliverables as artifacts/reports/templates). Paper tabs (outlined, no fill until active). Outputs gallery in paper cards (title + type icon + date in mono + preview). Blogs as article list (serif title + sans synopsis + date mono). Templates browser. Explorer routes through.

**Gap.**
- PageHeader basic — needs eyebrow + lede
- FilterTabs Studio refinement needed
- Tab content containers need paper-card cohesion
- OutputsFeed likely uses glass-card — audit nested components
- DeliverablesBlog needs serif/mono hierarchy
- No mono detail on dates / artifact types
- No empty-state editorial copy per tab

**Phase / Priority.** Phase 11 / Tier 0 hot / Medium-high frequency
**Effort.** L
**Mock pointer.** No dedicated mock — Round 4 batch B

---

### /marketplace/widgets/[id]

**Current state.** Widget detail: back button + header card (icon + name + developer + description + rating + installs + install button) + tabs (overview/screenshots/reviews/changelog) + sidebar (details/categories/permissions). Glass-card legacy orange, framer-motion. Review form + screenshot gallery.

**Studio target.** Detail page with editorial framing. Back button as Studio outline. Header is paper card with icon + serif name + sans description + mono metadata (developer, version, published, bundle size). Install/manage button prominent (studio primary). Studio tabs. Paper review form + paper details sidebar.

**Gap.**
- No editorial PageHeader / framing
- Back button ghost variant — needs Studio outline
- Header glass-card → paper with serif name
- Metadata not in mono font
- Tab navigation needs Studio refinement
- Review form glass-card → paper with serif heading
- Sidebar details/categories/permissions cards glass-card → paper
- Uninstall button uses legacy destructive red — switch to burnt orange `#c44a1a`
- Rating stars yellow via legacy warning variable — confirm CSS variable correctness

**Phase / Priority.** Phase 8 / Tier 0 hot / Medium-high frequency (post-browse)
**Effort.** M
**Mock pointer.** No dedicated mock — Round 4 batch B

---

## 4 · Tier 1 — Important inheritance + template work (15 routes · Phases 7, 9, 10 + sweep)

### /dashboard ⚠️ Decide vs /command-center

**Current state.** System overview with metric cards (system health, agents, documents, performance), system health panel, quick actions, performance chart. PageHeader (title="System" + titleAccent="Dashboard"), "System Online" badge. Glass-card dark theme.

**Studio target.** Editorial PageHeader (eyebrow "REAL-TIME MONITORING"). Paper metric cards (warm ink labels + near-black numbers). System Online badge in olive. Quick action buttons in near-black with tan borders. Performance chart with warm ink axes + olive growth / burnt orange warnings.

**Gap.**
- Major overlap with `/command-center` (estimated >60%). Decide which page survives.
- Otherwise: PageHeader editorial, metric cards paper styling, semantic colour lock, glass removal

**Phase / Priority.** Phase 3 (decide redundancy) / Tier 1 / Currently high frequency
**Effort.** L if kept, S if killed
**Mock pointer.** No mock — Round 4 batch A if kept
**Recommendation:** **kill `/dashboard`** and redirect to `/command-center`. The Studio Command Centre mock already covers the "real-time monitoring + KPIs" use case. Maintaining two surfaces doubles the work.

---

### /playbooks

**Current state.** PlaybooksPanel: PageHeader ("My" + "Playbooks"), SearchInput, grid of basic bordered playbook cards (name, support, tenant_id, created_at). No tabs / stats / visual hierarchy.

**Studio target.** Editorial PageHeader (eyebrow "WORKFLOW LIBRARY"). Paper playbook cards: title (serif bold) + description (sans) + metadata footer (support count + creation date in mono). Status indicators (draft = muted, active = olive). 2–3 column responsive grid.

**Gap.**
- PageHeader needs eyebrow + lede
- Playbook cards are minimal divs → Studio paper cards
- No semantic colour layer for status
- Metadata not in mono
- No stats bar / aggregate view
- Grid layout rigid

**Phase / Priority.** Phase 7 / Tier 1 / Medium frequency
**Effort.** M
**Mock pointer.** No dedicated mock — Round 4 batch A

---

### /tools

**Current state.** ToolsDashboard: PageHeader (no eyebrow/lede), StatsBar (available/deprecated/beta/connected), SearchInput, Tabs (available-tools/installed/composio-apps), tool grid with status icons. Glass-card legacy orange.

**Studio target.** Editorial PageHeader (eyebrow "INTEGRATIONS" + lede on tools as connectable integrations). Paper cards with semantic status: green (available), olive (connected), yellow (beta), red (deprecated). Tool version/metadata in mono.

**Gap.**
- PageHeader missing editorial framing
- Stat cards + tab badges still orange-primary
- No mono on tool IDs / versions / metadata
- Glass-card → paper-card
- Category badges + status icons need Studio semantic alignment
- Search + filter chips legacy
- Tool logo full saturation (cross-cutting E)

**Phase / Priority.** Tier 1 sweep / Tier 1 / Medium frequency
**Effort.** M
**Mock pointer.** No dedicated mock — inheritance + Studio principles

---

### /team

**Current state.** TeamManagement: PageHeader ("Team"), tabs (roster / invite / settings), member cards (avatar + name + role + status), action buttons. Minimal styling, legacy orange buttons.

**Studio target.** Editorial PageHeader (eyebrow "WORKSPACE MEMBERS" + lede on roles + invitations). Paper member cards: avatar + name (serif) + role (sans) + status badge (olive = active, muted = pending, red = inactive). Mono on member ID / email / joined date. Studio button + input.

**Gap.**
- PageHeader basic
- Member cards are bare divs → paper-card
- Status badges not semantically coloured
- No mono on IDs / emails / metadata
- Buttons legacy orange
- No empty-state editorial copy

**Phase / Priority.** Tier 1 sweep / Tier 1 / Medium frequency
**Effort.** S
**Mock pointer.** No mock — apply audit.jsx editorial header pattern

---

### /marketplace/publish

**Current state.** 5-step submission wizard: PageHeader ("Publish" + "Widget"), ProgressIndicator (custom), per-step form cards. Glass-card layout, orange gradient text + buttons. Save Draft (ghost) + Submit (primary).

**Studio target.** Editorial onboarding wizard. Editorial PageHeader (eyebrow "SHARE YOUR WIDGET"). Paper step cards with serif step titles. ProgressIndicator: completed = olive filled, current = navy outlined, future = muted. Studio form inputs (paper bg, muted border, serif labels). Studio submit buttons.

**Gap.**
- PageHeader editorial framing missing
- ProgressIndicator circles need semantic colours
- Step cards glass → paper
- Form fields need Studio refinement (categories/permissions/plans checkboxes + radios)
- Pricing section needs unified paper layout
- Save Draft + Submit buttons need Studio styles
- Error messages need prominent treatment (icon + red text)
- Success screen copy: serif heading + sans body

**Phase / Priority.** Phase 8 / Tier 1 / Low frequency (developer-focused)
**Effort.** M
**Mock pointer.** No mock — Round 4 batch B

---

### /marketplace/developer

**Current state.** Developer dashboard: PageHeader ("Developer" + "Dashboard"), 4-up stat cards (published / installs / rating / reviews), table of developer's widgets. Status badges (draft/review/published/suspended) semantically correct. Glass-card icon badges.

**Studio target.** Editorial PageHeader (eyebrow "YOUR MARKETPLACE"). Paper stat cards (simple icon, no coloured bg). Status badges Studio paper-badge style. Widget table: row hover lift, serif title, mono version, mono installs, mono + star icons. Studio Create Widget primary button.

**Gap.**
- PageHeader editorial framing missing
- Stat cards glass + legacy small-icon badges → paper + simple icon
- Status badges correct semantic but style refinement needed
- Widget table rows need hierarchy: serif title + mono metadata + icon actions
- Table header → small caps sans
- Version display should use explicit JetBrains Mono class

**Phase / Priority.** Phase 8 / Tier 1 / Low frequency
**Effort.** S
**Mock pointer.** No mock — table + stat card from audit.jsx + settings.jsx

---

### /deliverables/explorer

**Current state.** Full-page file browser: PageHeader ("Workspace Explorer") + WorkspaceExplorer (file tree + editor + terminal). Back button (ghost) routes to /deliverables. ESC key hooks back.

**Studio target.** PageHeader with serif title + eyebrow "FILE BROWSER". Back button Studio outline. WorkspaceExplorer surfaces use paper-card + mono on code/paths.

**Gap.**
- PageHeader could use eyebrow
- Back button ghost → Studio outline
- WorkspaceExplorer audit deferred (out of route scope)
- Page header border-bottom subtle

**Phase / Priority.** Phase 11 / Tier 1 / Medium frequency (power users)
**Effort.** S
**Mock pointer.** No mock — inherit page-header editorial pattern

---

### /documents

**Current state.** Minimal route wrapper → DocumentManagement component. No PageHeader visible at route level.

**Studio target.** Editorial PageHeader (eyebrow "DOCUMENT LIBRARY" + lede on documents as persistent workspace records). DocumentManagement renders paper document cards (title serif + type icon + author/date mono + preview). Studio search/filter inputs. Studio upload/create primary button. Empty state editorial copy.

**Gap.**
- No PageHeader at route level
- DocumentManagement styling unknown — likely lacks paper-card hierarchy
- No visible search/filter at route level
- No empty-state copy
- IA pattern (list/grid/table) undefined

**Phase / Priority.** Tier 1 sweep / Tier 2 / Low-medium frequency
**Effort.** S
**Mock pointer.** No mock — inherit editorial header

---

### /analytics

**Current state.** AnalyticsPage with 6+ tabs (overview/agents/missions/documents/costs/tools/admin). PageHeader (title="" + titleAccent="Analytics"). Dark theme. Admin workspace switcher + time range select in header actions.

**Studio target.** Editorial PageHeader (eyebrow "INSIGHTS" + lede). Tab headers serif + olive underline for active. Data visualisations with warm ink labels + burnt orange for cost warnings. Time range select as tan-border box. Admin switcher as muted pill.

**Gap.**
- PageHeader needs editorial framing
- Tab labels generic — need serif + olive active underline
- Admin workspace switcher needs muted styling
- Time range select needs tan border
- Data cards need warm ink labels + olive (positive trends) + burnt orange (cost overruns)
- All blue/purple accents → olive (success) / burnt orange (cost/waste)

**Phase / Priority.** Phase 10 / Tier 1 / High frequency (daily ops check)
**Effort.** L (updates all 6 child tab components)
**Mock pointer.** No mock — Round 4 batch A

---

### /settings ✅ Phase 9 mock ready

**Current state.** Light shell wrapping SettingsPanel — 9+ tabs (system-settings, orchestrator, webhooks, API keys, credentials, channels, notifications, voice profiles, widget SDK). Generic icons + grey labels, no serif.

**Studio target.** Per `round3/settings.jsx`: serif "System Settings" headline + lede paragraph. Tabs styled tan-bordered with olive underline on active. Mono on API key hashes + webhook event types. Warm ink throughout. Studio toggle/switch styles.

**Gap.**
- Tab styling: olive active underline instead of orange
- PageHeader needs eyebrow + lede
- SettingsPanel wrapper needs cream bg + serif
- Icon colours generic grey/orange → warm ink with accent for destructive only
- Tab buttons: muted ink off / near-black on with olive underline
- Mono detail on technical strings

**Phase / Priority.** Phase 9 / Tier 1 / High frequency (daily for admins)
**Effort.** M
**Mock pointer.** `DUMPING AREA/DesignKIT/round3/settings.jsx`

---

### /settings/profile

**Current state.** Custom full-page editor (not wrapped in SettingsPanel). Generic PageHeader, orange/purple gradient CTAs, glass-card, dark theme. Personal info + workspace ID copy + security + danger zone.

**Studio target.** Editorial PageHeader (serif Profile + subtitle lede). Cream paper. Personal info section with warm ink labels + plain text values. No avatar orange border (or thin accent on hover). Workspace ID in mono. Danger zone with burnt orange. Edit/Save buttons minimal Studio.

**Gap.**
- Wrap in `.studio` scope
- PageHeader: serif (inherits) + remove gradient
- Avatar: drop orange gradient border
- Section headings (Personal / Email / Workspace / Security / Danger Zone) → serif + near-black
- Form labels: small mono uppercase muted (#5a5448)
- Workspace ID → `[data-studio-mono]`
- Edit/Save/Cancel → Studio button styles, not gradient
- Danger Zone only uses burnt orange `#c44a1a`

**Phase / Priority.** Phase 9 / Tier 1 / Medium frequency
**Effort.** M
**Mock pointer.** No mock — inherits from `round3/settings.jsx`

---

### /settings/notifications

**Current state.** Wrapper around `<NotificationsSettingsTab>`. PageHeader (title="Notification" + titleAccent="Preferences"). No editorial framing.

**Studio target.** Editorial PageHeader (eyebrow + lede on event types + channel routes). Notification rule cards with warm ink labels, olive (enabled), burnt orange (mute), navy (info). Event type badges in mono.

**Gap.**
- Wrap in `.studio`
- PageHeader subtitle → lede
- NotificationsSettingsTab needs warm ink + tan borders
- Event type badges → mono + muted
- Toggle states: enabled = olive, disabled = muted, mute = burnt orange
- Remove orange/blue gradients

**Phase / Priority.** Phase 9 / Tier 1 / Medium frequency (setup-only)
**Effort.** S
**Mock pointer.** No mock — Round 4 batch B (or inheritance)

---

### /onboarding/wizard

**Current state.** Custom React WizardShell — 6-step business intake wizard (Goals → Domain Scan → Scanning → Page Checklist → Intake → Profile Editor). Framer Motion. Defaults to `open={true}`, redirects to `/assignments?tab=missions` on close. Glass-card + heavy orange accents (`text-orange-500`, `shadow-orange-500`). Dark gradient background.

**Studio target.** Redirect-first onboarding with serif headline + lede. Flat cream paper (replace dark gradient). Step headers serif on titles + mono on code/detail zones. Orange spinners → navy (loading/info), olive (success), burnt orange (consequence only). Shadows/glow stripped.

**Gap.**
- Heavy orange hardcoded throughout (spinners, shadows, text accents)
- Glass-card styles need Studio paper replacement
- Dark gradient background → cream
- Step typography sans default → serif on titles
- Modal wrapping (`min-h-screen` dark) → cream page or full-bleed cream shell

**Phase / Priority.** Tier 1 sweep / Tier 1 / First-run UX (critical path)
**Effort.** L
**Mock pointer.** No mock — inline component refactor

---

### /reset-password

**Current state.** Custom form (no Clerk components). Uses `useSignIn` hook. Glass-card + orange branding (`shadow-orange-500/20`, gradient text title). Motion animations.

**Studio target.** Flat cream paper. White paper card. Accents: navy for info (code input label), olive for success. Orange only if consequence. Serif headline + lede. No glow.

**Gap.**
- Glass-card + dark border → flat paper
- `shadow-orange-500/*` → remove
- Gradient title → serif black on cream
- Button gradient-accent → solid navy (info flow)
- Dark background → cream
- Form inputs styled for dark → light inputs on cream
- No serif headlines

**Phase / Priority.** Tier 2 — Standalone (not Clerk-dependent)
**Effort.** M
**Mock pointer.** No mock — inherits CSS vars only

---

## 5 · Tier 2 — Long tail (11 routes · post-Phase 11 sweep)

These inherit from Phase 1 tokens + Phase 2–11 patterns. Each gets a 1–2 hour cleanup pass during the post-Phase-11 sweep.

### /admin/workspaces
Workspace admin table with stats, search/filter, pagination. Glass-card legacy colour. **Gap:** wrap in `.studio`, PageHeader editorial, stat cards paper, search/filter tan borders, table small-caps mono header + regular mono body + tan borders + cream/tan row hover, workspace state badges olive/muted/burnt-orange, action buttons icon-only pills. **Phase / Effort:** Tier 2 sweep / M. **Mock:** none (apply Studio tokens).

### /admin/plugins
Plugin approval queue with risk score badges + LLM summaries + finding cards. Glass-card gradient-text. **Gap:** wrap `.studio`, serif "Plugin Approval Queue" + eyebrow "SECURITY REVIEW" + lede, plugin cards paper + thin warm border, risk progress bar olive→muted→burnt-orange, verdict badges olive/muted/burnt-orange, finding severity small-caps mono, batch action buttons olive/burnt-orange. **Phase / Effort:** Tier 2 sweep / M. **Mock:** none.

### /admin/plugins/upload
Plugin upload form with zip/GitHub toggle + scan progress + results display. Glass-card orange gradient. **Gap:** same patterns as `/admin/plugins`: editorial header, near-black toggle buttons (no gradient), progress phase mono labels (olive/muted/burnt-orange), risk score bar same gradient, finding cards mono severity, Approve = olive / Reject = burnt orange. **Phase / Effort:** Tier 2 sweep / M. **Mock:** none.

### /accept-invitation
Custom React invitation-accept using Clerk SignUp. Dark gradient `from-black via-zinc-900 to-black` + orange spinners. **Gap:** dark bg → cream paper, orange spinners → navy, Clerk SignUp appearance inherits from global Clerk config (blocked by cross-cutting issue A), Shell layout → light, no serif headlines, error icons need olive/burnt-orange semantic. **Phase / Effort:** Tier 3 Clerk batch / M (depends on Clerk fix). **Mock:** none.

### /sso-callback
Clerk `AuthenticateWithRedirectCallback`. Minimal page, no custom UI. **Gap:** `afterSignInUrl="/"` lands on `/` → `/chat` (indirect); no visible UI during normal flow, but no fallback for errors. Add serif loading screen + Studio empty state. **Phase / Effort:** Tier 3 Clerk batch / S. **Mock:** none.

### /tools/callback
Custom Composio OAuth callback handler. Dark `bg-black` + orange spinner. **Gap:** dark bg → cream (if user sees it >1s), orange spinner → navy, add error UI with serif headline + lede + burnt orange accent. **Phase / Effort:** Tier 3 / S. **Mock:** none.

### /api-debug
Internal debug page. Incomplete implementation (empty `fetch()` + `alert()`). Inline blue button. **Gap:** likely non-functional — flag for cleanup. If kept, apply paper bg + serif title + navy/slate button. **Phase / Effort:** Tier 3 / S (or delete). **Mock:** none.

### /api-diagnostics
Internal diagnostics for testing backend endpoints. Uses design-system classes. **Gap:** emoji status icons not themeable → SVG with CSS bindings, plain h1 → serif + lede, ensure buttons use slate/navy not bright primary. **Phase / Effort:** Tier 3 / S. **Mock:** none.

### /context
Experimental client-side page → ContextEngineering component inside MainLayout. **Gap:** likely inherits, but verify component has no hardcoded colours / dark glass / orange accents. **Phase / Effort:** Tier 2 sweep / S. **Mock:** none.

### /field-theory
Experimental page → FieldVisualization directly (no MainLayout wrapper). **Gap:** no layout wrapper means full-screen viz; verify cream bg; component-level audit deferred. **Phase / Effort:** Tier 2 sweep / S. **Mock:** none.

### /assignments
Client-side AssignmentsPage inside MainLayout. **Gap:** inherits, but audit component for hardcoded orange / dark glass / typography. **Phase / Effort:** Tier 2 sweep / S. **Mock:** none.

---

## 6 · Tier 3 — Clerk batch (2 routes · half-day sweep)

### /sign-in/[[...rest]]
Clerk SignIn wrapper. Inline `appearance` override on card only. **Critical gap:** depends on global Clerk `colorPrimary: '#ff6b35'` in `providers.tsx` line 40 — fix that and 90% of this lands. Otherwise: dark gradient bg → cream, add serif headline + lede outside the Clerk component. **⚠️ Duplicate of `/auth/signin` (legacy redirect). Consolidate post-rebrand.** **Phase / Effort:** Tier 3 / S. **Mock:** Clerk provider config.

### /sign-up/[[...rest]]
Custom SignUpForm component (not Clerk). Full sign-up with OAuth + email verification + waitlist. Glass-card + heavy orange. **Gap:** entire form needs Studio paper-card + serif title + navy/olive buttons (depending on flow step), remove all orange shadows + gradient text, OAuth buttons light bordered on cream, dark gradient bg → cream. **⚠️ Duplicate of `/auth/signup` (legacy redirect). Consolidate post-rebrand.** **Phase / Effort:** Tier 2 / M (full custom form). **Mock:** none.

---

## 7 · Out of scope (3 routes · redirects only)

### `/` — redirects to `/chat`. No UI. Out of scope.
### `/auth/signin/[[...rest]]` — legacy redirect to `/sign-in`. **Track B: consolidate post-rebrand.**
### `/auth/signup/[[...rest]]` — legacy redirect to `/sign-up`. **Track B: consolidate post-rebrand.**

---

## 8 · Recommended action sequence

### Sprint 0 — Phase 1.5 polish (1–2 days · ship before Phase 2 design)
1. **Clerk `colorPrimary`** swap to near-black `#1a1814` (cross-cutting A) — 10 min
2. **Brand mark Studio variant** asset or CSS filter (cross-cutting F) — 1 hour
3. **Icon desaturation** under `.studio` scope (cross-cutting E) — 30 min CSS
4. **Orange hardcoding sweep** on the 5 problem auth/onboarding pages (cross-cutting B) — 1 day
5. **Inline dark-glass styling** audit on the same 5 pages (cross-cutting C) — 0.5 day

After Sprint 0, the platform under Studio mode looks significantly more cohesive.

### Sprints 1–10 — Phases 2 through 11 (per PRD §2)
Top-10 page deep dives in the order locked in PRD §2:
1. Phase 2 → `/chat` + `/chat/[id]` ← **mock ready**
2. Phase 3 → `/command-center` (kill `/dashboard`) ← needs Round 4 narration mocks
3. Phase 4 → `/missions/[id]`
4. Phase 5 → `/activity/execution` ← **mock ready**
5. Phase 6 → `/agents` + `/agents/[id]`
6. Phase 7 → `/playbooks` ← needs Round 4 batch A
7. Phase 8 → `/marketplace` + sub-pages ← needs Round 4 batch B
8. Phase 9 → `/settings` ← **mock ready**
9. Phase 10 → `/analytics` ← needs Round 4 batch A
10. Phase 11 → `/deliverables` ← needs Round 4 batch B

### Sprint 11 — Tier 1 + 2 sweep (1 week)
Apply inheritance + cleanup to the remaining 15 + 11 routes. Per-page 1–2 hour pass.

### Sprint 12 — Tier 3 Clerk batch (0.5 day)
Once cross-cutting A is in, the Clerk pages mostly land on their own.

### Sprint 13 — Track B (post-rebrand)
- Consolidate `/auth/signin` → `/sign-in` redirects
- Consolidate `/auth/signup` → `/sign-up` redirects
- Kill `/dashboard` if confirmed redundant

---

## 9 · Three immediate calls for you

1. **Approve Sprint 0 polish list** — should I run the 5 polish items as the next overnight, or wait for your visual review of Phase 1 first?
2. **Kill `/dashboard`?** — recommendation is yes (merge functionality into `/command-center`). Confirm or push back.
3. **Round 4 commission** — go ahead and tell CD to start Batch A (Playbooks + Analytics + Command-Centre Round 4) now so they're ready when we hit Phase 3?
