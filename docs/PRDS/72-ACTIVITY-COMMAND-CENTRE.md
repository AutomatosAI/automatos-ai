# PRD-72: Activity Command Centre

**Version:** 1.0
**Status:** Draft
**Priority:** P1
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-03-07
**Updated:** 2026-03-07
**Dependencies:** PRD-10 (Workflow Engine — COMPLETE), PRD-55 (Agent Heartbeats — COMPLETE), PRD-06 (Monitoring & Analytics — COMPLETE)
**Branch:** `feat/activity-command-centre`

---

## Executive Summary

Automatos has four execution modes — chat conversations, agent heartbeat routines, recipe executions, and complex dynamic missions — but users have no single place to see all of them. The Workflow Management page only shows recipes and "cooking" (running executions). Agent heartbeats are invisible. Chat history is buried. There's no way to answer the simplest question: **"What is my AI workforce doing right now?"**

This PRD replaces `/workflows` with `/activity` — the **Activity Command Centre**. One page, four tabs, unified feed. Non-technical users see their workforce in action. Power users drill into any execution, configure the source, or replay runs.

### What We're Building

1. **Unified Activity Feed** — a real-time timeline showing all execution types (chats, routines, recipes, missions) with status, duration, agents involved, and drill-down links
2. **Routines Tab** — surfaces agent heartbeats (currently invisible) as manageable recurring tasks with pause/resume/edit
3. **Enhanced Recipes Tab** — keeps existing recipe management, adds run history inline
4. **Missions Tab** — placeholder for complex multi-agent workflows (future PRD), with a "Coming Soon" state that explains the concept

### What We're NOT Building

- New execution engine (existing recipe executor + heartbeat service are sufficient)
- Analytics replacement (Activity is operational monitoring; Analytics stays for cost/performance trends)
- Chat page changes (Chat remains at `/chat`, but chat executions appear in the Activity feed)

---

## 1. Naming Convention

All user-facing terminology follows this model:

| Technical Concept | User-Facing Name | Icon | Description |
|---|---|---|---|
| Chat + tool calls | **Chat** | `MessageCircle` | "Just ask" — conversations with AI agents |
| Agent heartbeat | **Routine** | `RefreshCw` | "Keep checking this" — recurring background tasks |
| Recipe execution (single/multi-step) | **Recipe** | `ChefHat` / `CookingPot` | "Follow these steps" — defined automations |
| Complex dynamic workflow | **Mission** | `Rocket` | "Go handle this" — multi-agent projects (future) |

### Status Labels

| Technical | User-Facing | Badge Colour | CSS Variable |
|---|---|---|---|
| `pending` | Waiting | `--muted` | `bg-muted/10 text-muted-foreground` |
| `running` | Working... | `--info` | `bg-[hsl(var(--info))]/10 text-[hsl(var(--info))]` |
| `completed` | Done | `--success` | `bg-[hsl(var(--success))]/10 text-[hsl(var(--success))]` |
| `failed` | Needs Attention | `--destructive` | `bg-destructive/10 text-destructive` |
| `cancelled` | Cancelled | `--muted` | `bg-muted/10 text-muted-foreground` |
| `scheduled` | Upcoming | `--agent` | `bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))]` |
| `paused` | Paused | `--warning` | `bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))]` |

---

## 2. Design System Compliance

All components must follow the established Automatos design language from `globals.css`. No custom colours or one-off styling.

### Required Patterns

| Pattern | Class / Token | Used By |
|---|---|---|
| Page header | `<PageHeader>` shared component | Title with `gradient-text` accent |
| Stat cards | `<StatsBar>` shared component | 4-up hero stats, `glass-card` + `card-glow` |
| Tab navigation | `<FilterTabs>` shared component | Icon + label tabs, `bg-secondary/40 backdrop-blur` |
| Content cards | `glass-card` class | All card surfaces |
| Drill-down panels | `glass-panel` class | Execution detail views |
| Time period toggle | `PeriodToggle` (from analytics-costs) | `1D / 7D / 30D / 90D` pill selector |
| Status badges | Semantic `Badge` variants | Using CSS variable palette above |
| Charts | Recharts with `MODEL_COLORS` palette | Area/Line charts for trend data |
| Motion | `framer-motion` staggered entrance | `initial → animate` with `delay: index * 0.08` |
| Empty states | Centered icon + two-line text | Muted icon (30% opacity) + description |
| Tables | `glass-card overflow-hidden` wrapper | `[11px] uppercase tracking-wider` headers |
| Loading | `<Skeleton>` matching card layout | Skeleton grid matching real content shape |
| Mobile | Reduced `backdrop-blur(8px)` | Per `globals.css` `@media (max-width: 767px)` |

### Typography Scale (from existing pages)

- Page title: `text-2xl md:text-3xl font-bold`
- Stat value: `text-2xl font-bold leading-none`
- Stat label: `text-sm text-muted-foreground`
- Card title: `CardTitle` with icon + `gap-2`
- Table header: `text-[11px] font-medium text-muted-foreground uppercase tracking-wider`
- Body text: `text-sm`
- Timestamps: `text-xs text-muted-foreground`

### Colour Semantics (DO NOT hardcode hex)

```
Primary/Orange:  hsl(var(--primary))       — active states, CTAs, accents
Success/Green:   hsl(var(--success))        — completed, healthy
Info/Blue:       hsl(var(--info))           — running, in-progress
Warning/Amber:   hsl(var(--warning))        — paused, attention
Destructive/Red: hsl(var(--destructive))    — failed, errors
Agent/Purple:    hsl(var(--agent))          — agent-related, scheduled
Muted:           hsl(var(--muted-foreground)) — inactive, secondary text
```

---

## 3. Page Layout

### Route Change

```
/workflows  →  /activity
```

Sidebar navigation item rename:
- Label: `Workflow Management` → `Activity`
- Icon: `GitBranch` → `Activity` (from lucide-react)
- `href`: `/workflows` → `/activity`

301 redirect from `/workflows` → `/activity` for bookmarks.

### Page Structure

```
┌──────────────────────────────────────────────────────────────┐
│  <PageHeader                                                  │
│    title="Command"                                            │
│    titleAccent="Centre"                                       │
│    subtitle="Your AI workforce at a glance"                   │
│    actions={<PeriodToggle /> <RefreshButton />}               │
│  />                                                           │
├──────────────────────────────────────────────────────────────┤
│  <StatsBar stats={[                                           │
│    { label: "Working Now", value: "3", icon: Activity }       │
│    { label: "Channels Live", value: "3", icon: Radio }        │
│    { label: "Completed Today", value: "24", icon: CheckCircle}│
│    { label: "Needs Attention", value: "1", icon: AlertTriangle│
│  ]} />                                                        │
├──────────────────────────────────────────────────────────────┤
│  <FilterTabs tabs={[                                          │
│    { value: "feed", label: "Feed", icon: Activity }           │
│    { value: "routines", label: "Routines", icon: RefreshCw }  │
│    { value: "recipes", label: "Recipes", icon: CookingPot }   │
│    { value: "missions", label: "Missions", icon: Rocket }     │
│  ]}>                                                          │
│    <TabsContent value="feed">      → Section 4              │
│    <TabsContent value="routines">  → Section 5              │
│    <TabsContent value="recipes">   → Section 6              │
│    <TabsContent value="missions">  → Section 7              │
│  </FilterTabs>                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Feed Tab (Default)

The unified timeline. Shows all execution types, most recent first. This is what the user sees when they click "Activity" in the sidebar.

### 4.0 Channels Live Stat Card

The "Channels Live" stat card shows the count of connected channels (Telegram, WhatsApp, Slack, email, webchat) that are actively receiving messages. Data comes from Settings > Channels config via `/api/activity/stats`.

- **Click action:** Navigates to `/settings?tab=channels`
- **Icon:** `Radio` (lucide-react) — matches Settings > Channels tab icon
- **Colour:** `text-[hsl(var(--info))]` (blue) — indicates connectivity
- **"Needs Attention" includes:** Any channel marked as connected but not receiving messages for >1h (stale) gets counted in the "Needs Attention" stat alongside failed executions

### 4.1 Data Sources

| Type | Source | Polling |
|---|---|---|
| Chat | `GET /api/chats?workspace_id=X&limit=20` | On mount + 30s interval |
| Routine | `GET /api/heartbeats/executions?workspace_id=X` | On mount + 15s interval |
| Recipe | `GET /api/recipes/executions?workspace_id=X` | On mount + 15s interval |
| Mission | Future — empty for now | — |

All sources merged client-side, sorted by `started_at DESC`.

### 4.2 Feed Item Card

Each feed item is a `glass-card` with a coloured left border indicating type:

```
┌─ border-l-3 ──────────────────────────────────────────────┐
│  [TypeIcon]  Type Label · "Item Name"              [Time] │
│              Agent: {name} · Status Badge · Duration       │
│              ───────────────────────────────────            │
│              [Context line — varies by type]                │
│                                        [View] [Configure]  │
└────────────────────────────────────────────────────────────┘
```

**Left border colours by type:**
- Chat: `border-l-[hsl(var(--primary))]` (orange)
- Routine: `border-l-[hsl(var(--agent))]` (purple)
- Recipe: `border-l-[hsl(var(--info))]` (blue)
- Mission: `border-l-[hsl(var(--success))]` (green)

**Channel badge:** When `channel` is present, a small muted badge appears inline after the status: `via Telegram`, `via WhatsApp`, etc. Uses `text-xs text-muted-foreground` — informational, not prominent. Channel icon from the existing Settings > Channels icon set.

**Context line by type:**
- Chat: First 100 chars of the user's message, truncated
- Routine: "Checked 3 min ago · Next run: 11:00"
- Recipe: Step pipeline → `✓ Step 1 → ✓ Step 2 → ● Step 3...`
- Mission: Progress bar `████░░░░ 65% · 8 of 12 tasks`

**Actions:**
- **View** — navigates to the detail view (chat opens `/chat/{id}`, recipe opens execution detail, routine opens routine detail)
- **Configure** — navigates to the source's config page (agent config for routines, recipe editor for recipes)

### 4.3 Filters

Above the feed, a row of filter chips:

```
[All]  [💬 Chats]  [🔄 Routines]  [📋 Recipes]  [🚀 Missions]  |  [Status ▼]
```

- Type chips are toggle buttons (multiple can be active)
- Status dropdown: All / Working / Done / Needs Attention / Upcoming

### 4.4 Empty State

```
<Activity className="w-12 h-12 mx-auto mb-3 text-muted-foreground/30" />
"No activity yet"
"Start a chat, create a routine, or run a recipe to see your workforce in action"
```

### 4.5 Real-Time Updates

- Feed items with status `running` show the `stage-active` pulse animation on their status badge
- New items prepend to the feed with `log-slide-in` animation (from `globals.css`)
- Polling interval: 15s for recipe/routine executions, 30s for chats
- Optional: WebSocket/SSE for truly real-time updates (enhancement, not MVP)

---

## 5. Routines Tab

Surfaces agent heartbeats as user-manageable recurring tasks. Currently these are invisible — users configure a heartbeat on an agent but have no way to see it running or manage it outside the agent config modal.

### 5.1 Data Source

```
GET /api/heartbeats?workspace_id=X
```

Returns all heartbeat configurations with their last execution time and status.

### 5.2 Routine Card

```
┌─ glass-card ──────────────────────────────────────────────┐
│  [AgentAvatar]  Routine Name                    ● Active   │
│                 Agent: {name}                               │
│                 Every {interval} · Last ran {ago}           │
│                 Next: {time}                                │
│                 "{description}"                             │
│                                          [Pause] [Edit]    │
└────────────────────────────────────────────────────────────┘
```

- **Active** badge: `bg-[hsl(var(--success))]/10` with green dot
- **Paused** badge: `bg-[hsl(var(--warning))]/10` with amber dot
- **Edit** navigates to the agent configuration page (heartbeat section)
- **Pause/Resume** toggles heartbeat via `PATCH /api/heartbeats/{id}/toggle`

### 5.3 Routine History (Expandable)

Clicking a routine card expands to show last 10 executions:

```
│  ▼ Execution History                                       │
│  ┌─ log-entry log-entry-success ─────────────────────────┐ │
│  │  ✓ Completed · Mar 7, 10:00 · 4.2s · No issues       │ │
│  ├─ log-entry log-entry-success ─────────────────────────┤ │
│  │  ✓ Completed · Mar 7, 09:00 · 3.8s · No issues       │ │
│  ├─ log-entry log-entry-error ───────────────────────────┤ │
│  │  ✗ Failed · Mar 7, 08:00 · "API rate limit"          │ │
│  └───────────────────────────────────────────────────────┘ │
```

Uses `log-entry`, `log-entry-success`, `log-entry-error` classes from `globals.css`.

### 5.4 Create Routine CTA

```
[+ New Routine] button in tab header
```

Opens a modal or navigates to agent config with heartbeat section focused. MVP: navigates to `/agents` with a toast "Select an agent to configure its routine."

### 5.5 Empty State

```
<RefreshCw className="w-12 h-12 mx-auto mb-3 text-muted-foreground/30" />
"No routines set up"
"Routines let your agents check things automatically — like monitoring your inbox or tracking sales"
[Set Up a Routine →]
```

---

## 6. Recipes Tab

Migrates the existing `<RecipesTab>` component with enhancements.

### 6.1 What Stays

- Existing recipe grid/list view with `<ViewToggle>`
- Recipe cards with step count, agent count, trigger type
- Create Recipe dialog
- Search input

### 6.2 What Changes

**Trigger type badges** — clearer labels with icons:

| Trigger | Badge | Icon |
|---|---|---|
| Scheduled | `⏰ Scheduled: Mon 9am` | `Clock` |
| Event trigger | `⚡ Trigger: New CRM contact` | `Zap` |
| Manual | `🖐 Manual` | `Hand` |
| Webhook | `🔗 Webhook` | `Link` |

**Inline run history** — each recipe card shows its last 3 runs as mini status dots:

```
Last runs: ● ● ○   (green = done, red = failed, gray = cancelled)
           3h  1d  3d
```

Clicking opens the execution history for that recipe.

**Run count** — total executions shown: `Ran 47 times`

### 6.3 Recipe Execution Detail (Drill-Down)

When a user clicks a running or completed recipe execution (from Feed or from recipe card), they see:

```
┌─ glass-panel ─────────────────────────────────────────────┐
│  ← Back to Activity                                        │
│                                                            │
│  📋 Weekly Team Summary                                    │
│  Recipe · Started Mar 7, 09:15 · Done ✓ · 2m 34s          │
│                                                            │
│  ┌─ Step Pipeline ─────────────────────────────────────┐   │
│  │ ✓ Pull Slack highlights (Agent: Slack Bot, 45s)     │   │
│  │ ✓ Draft summary (Agent: Content Writer, 1m 12s)     │   │
│  │ ✓ Post to #team-updates (Agent: Slack Bot, 37s)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─ Output ────────────────────────────────────────────┐   │
│  │ {rendered output or artifact links}                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  ┌─ Execution Log ─────────────────────────────────────┐   │
│  │ 09:15:00  Started execution (triggered: schedule)   │   │
│  │ 09:15:02  Step 1: Calling Slack API...              │   │
│  │ 09:15:47  Step 1: Complete (45s)                    │   │
│  │ ...                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│                         [Re-run] [Edit Recipe] [View Logs] │
└────────────────────────────────────────────────────────────┘
```

**Step pipeline** uses the existing `stage-completed`, `stage-active`, `stage-pending` CSS classes with `stage-connector` between them.

### 6.4 Older Runs

A "History" sub-tab or expandable section within the recipe detail:

```
┌─ Run History ──────────────────────────────────────────────┐
│  Run #47  · Mar 7, 09:15  · Done ✓   · 2m 34s   [View]  │
│  Run #46  · Mar 6, 09:15  · Done ✓   · 2m 12s   [View]  │
│  Run #45  · Mar 5, 09:15  · Failed ✗ · 0m 45s   [View]  │
│  Run #44  · Mar 4, 09:15  · Done ✓   · 2m 48s   [View]  │
│                                          [Load More]       │
└────────────────────────────────────────────────────────────┘
```

Each row is clickable → opens that execution's detail view.

---

## 7. Missions Tab

Placeholder for complex multi-agent workflows. Not building the execution engine yet — just the UI frame so users know it's coming and understand the concept.

### 7.1 Coming Soon State

```
┌─ glass-card ──────────────────────────────────────────────┐
│                                                            │
│  🚀                                                        │
│                                                            │
│  Missions — Coming Soon                                    │
│                                                            │
│  Missions are big, multi-agent projects that run for       │
│  hours or days. Give a complex brief — like "Prepare       │
│  the Q1 board deck" — and your AI workforce figures        │
│  out the steps, assigns agents, and delivers results.      │
│                                                            │
│  ┌─ What Missions Can Do ──────────────────────────────┐   │
│  │  • Break complex goals into tasks automatically      │   │
│  │  • Assign the right agents to each task              │   │
│  │  • Track progress with a live dashboard              │   │
│  │  • Produce artifacts (docs, spreadsheets, reports)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                            │
│  Want early access? Let us know what you'd use it for.     │
│  [Request Early Access]                                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

The "Request Early Access" button opens the ChatWidget with a pre-filled message: "I'm interested in Missions — here's what I'd use it for:"

---

## 8. API Endpoints

### 8.1 New Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/activity/feed` | Unified feed — merges chats, routines, recipes. Query params: `type`, `status`, `period`, `limit`, `offset` |
| `GET` | `/api/activity/stats` | Stats for hero cards (includes channel health). Query params: `period` |
| `GET` | `/api/heartbeats` | List all heartbeat configs for workspace |
| `GET` | `/api/heartbeats/{id}/executions` | Execution history for a specific heartbeat |
| `PATCH` | `/api/heartbeats/{id}/toggle` | Pause/resume a heartbeat |

### 8.2 Existing Endpoints (No Changes)

| Method | Path | Used For |
|---|---|---|
| `GET` | `/api/recipes` | Recipe list (Recipes tab) |
| `GET` | `/api/recipes/{id}/executions` | Recipe execution history |
| `GET` | `/api/recipes/executions/{exec_id}` | Single execution detail |
| `POST` | `/api/recipes/{id}/execute` | Manual recipe run |
| `GET` | `/api/chats` | Chat list for feed |

### 8.3 Unified Feed Response Schema

```typescript
interface ActivityFeedItem {
  id: string
  type: 'chat' | 'routine' | 'recipe' | 'mission'
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
  started_at: string           // ISO timestamp
  completed_at: string | null
  duration_seconds: number | null
  agent: {
    id: number
    name: string
    avatar_url: string | null
  } | null
  agents: Array<{              // For multi-agent executions
    id: number
    name: string
  }>
  summary: string              // Context line (first message, step progress, etc.)
  source_id: string            // Recipe ID, heartbeat ID, chat ID
  source_url: string           // Deep link to configure/view source
  trigger: 'manual' | 'scheduled' | 'event' | 'webhook' | 'heartbeat' | null
  channel?: {                    // Present when activity came via an external channel
    type: 'telegram' | 'whatsapp' | 'slack' | 'email' | 'webchat'
    name: string                 // e.g. "Support Bot", "#sales-alerts"
  } | null
  step_progress?: {
    current: number
    total: number
    steps: Array<{
      name: string
      status: 'pending' | 'running' | 'completed' | 'failed'
    }>
  }
  error_message: string | null
}

interface ActivityStats {
  working_now: number
  channels_live: number          // connected channels actively receiving messages
  completed_today: number
  needs_attention: number        // failed executions + stale routines + disconnected channels
  period: string
}
```

---

## 9. File Structure

```
frontend/
  app/
    activity/
      page.tsx                  # NEW — route entry, replaces /workflows
    workflows/
      page.tsx                  # MODIFY — 301 redirect to /activity
  components/
    activity/
      activity-page.tsx         # NEW — main page component (like analytics-page.tsx)
      activity-feed.tsx         # NEW — Feed tab content
      activity-feed-item.tsx    # NEW — individual feed card
      activity-stats.tsx        # NEW — hero stats with polling
      activity-routines.tsx     # NEW — Routines tab
      routine-card.tsx          # NEW — individual routine card with history
      activity-missions.tsx     # NEW — Missions coming-soon placeholder
      execution-detail.tsx      # NEW — drill-down view for any execution
    workflows/
      recipes-tab.tsx           # MODIFY — add inline run history dots
      workflow-management.tsx   # KEEP — imported by activity-page for RecipesTab
  hooks/
    use-activity-feed.ts        # NEW — SWR/React Query hook for unified feed
    use-activity-stats.ts       # NEW — polling stats hook
    use-heartbeats.ts           # NEW — heartbeat CRUD hook
  lib/
    activity-service.ts         # NEW — API client methods for activity endpoints

orchestrator/
  api/
    activity.py                 # NEW — /api/activity/feed + /api/activity/stats
    heartbeats.py               # MODIFY — add GET list, GET executions, PATCH toggle
  services/
    activity_service.py         # NEW — merges data from chats, heartbeats, recipe_executions
```

---

## 10. Implementation Phases

### Phase 1: Route + Shell (no new data)
1. Create `/activity` route and `activity-page.tsx` with `PageHeader`, `StatsBar` (hardcoded), `FilterTabs`
2. Update sidebar: rename Workflow Management → Activity, update href + icon
3. Add redirect from `/workflows` → `/activity`
4. Move existing `<RecipesTab>` into the Recipes tab (zero functional change)
5. Add Missions tab with coming-soon placeholder
6. Wire `data-tour` attributes for the SHEPHERD tour system

### Phase 2: Routines Tab
7. Create `GET /api/heartbeats` endpoint (list all for workspace)
8. Create `use-heartbeats.ts` hook
9. Build `activity-routines.tsx` + `routine-card.tsx`
10. Create `PATCH /api/heartbeats/{id}/toggle` endpoint
11. Add execution history expansion per routine

### Phase 3: Unified Feed
12. Create `activity_service.py` backend — merges `chats`, `heartbeat_executions`, `recipe_executions` into a single sorted response
13. Create `/api/activity/feed` and `/api/activity/stats` endpoints
14. Build `use-activity-feed.ts` + `use-activity-stats.ts` hooks with polling
15. Build `activity-feed.tsx` + `activity-feed-item.tsx`
16. Wire live stats into `<StatsBar>`
17. Add filter chips (type + status)

### Phase 4: Execution Detail + History
18. Build `execution-detail.tsx` — step pipeline, output, logs
19. Add inline run history dots to recipe cards
20. Add "View" links from feed items → detail view
21. Add "Configure" links → source page (agent config, recipe editor)
22. Add "Re-run" action for completed recipe executions

### Phase 5: Polish
23. Real-time pulse animation for running items
24. `log-slide-in` animation for new feed items
25. Mobile responsive pass — reduced blur, stacked cards
26. Loading skeletons matching each tab's layout
27. Empty states for each tab
28. `prefers-reduced-motion` compliance (disable animations)

---

## 11. Navigation Drill-Down Map

From any item in the Activity Command Centre, users can reach the source configuration:

```
Feed Item (Chat)      → [View]      → /chat/{id}
Feed Item (Routine)   → [View]      → Routine detail (inline expand)
                      → [Configure] → /agents/{id}#heartbeat
Feed Item (Recipe)    → [View]      → Execution detail (inline or slide-over)
                      → [Configure] → Recipe editor modal
                      → [Re-run]    → POST /api/recipes/{id}/execute
Feed Item (Mission)   → [View]      → Execution theater (future)
                      → [Configure] → Mission editor (future)

Routine Card          → [Edit]      → /agents/{id}#heartbeat
                      → [Pause]     → PATCH /api/heartbeats/{id}/toggle
                      → [History ▼] → Expand execution log

Recipe Card           → [Run Now]   → POST /api/recipes/{id}/execute
                      → [Edit]      → Recipe editor modal
                      → [History]   → Expand run list
```

---

## 12. Success Metrics

| Metric | Target | How to Measure |
|---|---|---|
| Time to answer "what's running?" | < 3 seconds (one click from sidebar) | User testing |
| Routine visibility | 100% of active heartbeats visible | Compare DB heartbeat count vs UI |
| Recipe run findability | Any run reachable in ≤ 2 clicks | UI audit |
| Failed execution awareness | 0 silently failed runs | "Needs Attention" stat card always accurate |
| Page load time | < 1.5s initial, < 500ms tab switch | Lighthouse + RUM |
| Mobile usability | All tabs functional on 375px width | Manual test |

---

## 13. Open Questions

1. **WebSocket for real-time?** Polling at 15s is good enough for MVP. WebSocket/SSE upgrade is a future enhancement — the UI should be designed to support both without refactoring (just swap the data source in the hook).

2. **Feed pagination strategy?** Infinite scroll vs "Load More" button. Recommendation: "Load More" button — simpler, more predictable, works better on mobile.

3. **Mission execution engine?** Out of scope for this PRD. When we build it, it plugs into the Missions tab and the Feed seamlessly because the `ActivityFeedItem` schema already supports it.

4. **Chat entries in feed — all or just tool-using?** Recommendation: only show chats where an agent + tools were invoked (not simple Q&A). This keeps the feed focused on "work being done" rather than casual conversation. Configurable via a toggle later.
