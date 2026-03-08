# Command Centre Dashboard — Layout Design v2

**Status:** Design Review
**Replaces:** Flat activity feed list (current "Feed" tab)
**Builds on:** PRD-38.* widget infrastructure, `react-grid-layout` v2.2.2

---

## The Problem

Current Feed tab is a flat chronological list of routine/recipe executions. It's not useful as a "Command Centre" — you can't see what's happening at a glance, what's coming up, or get agent reports.

## The Vision

A **widget-based dashboard** that answers 3 questions instantly:
1. **What's happening right now?** (active executions, live stats)
2. **What's coming up?** (scheduled routines/recipes on a calendar)
3. **What did my agents report?** (pinnable agent report cards)

---

## Desktop Layout (12-column grid)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Command Centre                                  [1 Day ▾] [Customize ⚙]    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ Working   │  │ Channels  │  │ Completed │  │ Needs     │   KPI Stats    │
│  │ Now    3  │  │ Live   2  │  │ Today  12 │  │ Attn    1 │   (existing)   │
│  │     ↑ 2   │  │           │  │     ↑ 5   │  │     ↓ 2   │               │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘               │
│                                                                              │
│  ┌──────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ Active Now              3 ▪  │  │ Schedule                    This Wk │  │
│  │ ─────────────────────────── │  │ ──────────────────────────────────── │  │
│  │                              │  │                                      │  │
│  │  ● Social Media Scan        │  │  Mon   Tue   Wed   Thu   Fri   Sat  │  │
│  │    Running · Step 2/4 · 2m  │  │  ───   ───   ───   ───   ───   ─── │  │
│  │    [████████░░░░░░░] 50%    │  │   ·     ●●    ·    ●·     ·     ·  │  │
│  │                              │  │               ▲                      │  │
│  │  ● Nightly Build Report     │  │            (today)                   │  │
│  │    Running · Step 3/5 · 4m  │  │                                      │  │
│  │    [████████████░░░] 60%    │  │  UPCOMING                            │  │
│  │                              │  │  ─────────                           │  │
│  │  ○ Email Digest             │  │  14:00  Social Media Scan     daily  │  │
│  │    Pending · Starts in 12m  │  │  18:00  Nightly Build Report  daily  │  │
│  │                              │  │  09:00  Email Digest       → tmrw   │  │
│  │                              │  │  10:30  Weekly Research    → Thu    │  │
│  │          [View All →]       │  │                                      │  │
│  └──────────────────────────────┘  └─────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │ Agent Reports                                    [Pin Agents ⚙] 3 of 8 ││
│  │ ─────────────────────────────────────────────────────────────────────── ││
│  │                                                                          ││
│  │  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐  ││
│  │  │ 🤖 Social Manager │ │ 🤖 DevOps Bot     │ │ 🤖 Research AI    │  ││
│  │  │ ────────────────── │ │ ────────────────── │ │ ────────────────── │  ││
│  │  │ Last run: 2h ago   │ │ Last run: 6h ago   │ │ Last run: 1d ago   │  ││
│  │  │ Status: ● Done     │ │ Status: ● Done     │ │ Status: ● Done     │  ││
│  │  │                    │ │                    │ │                    │  ││
│  │  │ "Found 3 trending  │ │ "All 8 builds     │ │ "Compiled 12      │  ││
│  │  │  topics on AI      │ │  passing. Zero     │ │  sources on AI    │  ││
│  │  │  governance.       │ │  critical alerts.  │ │  governance       │  ││
│  │  │  Drafted 2 posts   │ │  Deps up to date." │ │  regulation..."   │  ││
│  │  │  for review."      │ │                    │ │                    │  ││
│  │  │                    │ │                    │ │                    │  ││
│  │  │ [View Report]  📌  │ │ [View Report]  📌  │ │ [View Report]  📌  │  ││
│  │  └────────────────────┘ └────────────────────┘ └────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │ Recent Activity                                         [View All →]    ││
│  │ ─────────────────────────────────────────────────────────────────────── ││
│  │                                                                          ││
│  │  ✅ Social Media Scan      Done     2m 34s    10 min ago    [View]     ││
│  │  ❌ Data Pipeline          Failed   timeout   45 min ago    [View]     ││
│  │  ✅ Email Digest           Done     1m 12s    2h ago        [View]     ││
│  │  ✅ Nightly Build Report   Done     4m 51s    6h ago        [View]     ││
│  │  ✅ Weekly Research        Done     8m 22s    1d ago        [View]     ││
│  │                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
```

## Grid Spec (12-column, `react-grid-layout`)

| Widget              | Grid Position      | Size (cols x rows) | Min Size  |
|---------------------|--------------------|--------------------|-----------|
| KPI Stats           | (0,0)              | 12 x 1             | 12 x 1    |
| Active Now          | (0,1)              | 5 x 3              | 4 x 2     |
| Schedule            | (5,1)              | 7 x 3              | 5 x 2     |
| Agent Reports       | (0,4)              | 12 x 3             | 6 x 2     |
| Recent Activity     | (0,7)              | 12 x 2             | 6 x 2     |

All widgets are **draggable and resizable** via `react-grid-layout`.
Users can rearrange, resize, and the layout persists to localStorage.

---

## Mobile Layout (stacked)

On screens < 768px, everything stacks vertically (1 column):

```
┌────────────────────────┐
│ Command Centre  [1d ▾] │
├────────────────────────┤
│ ┌──┐ ┌──┐ ┌──┐ ┌──┐  │  KPIs in 2x2 grid
│ │3 │ │2 │ │12│ │1 │  │
│ └──┘ └──┘ └──┘ └──┘  │
│                        │
│ ┌────────────────────┐ │
│ │ Active Now     (3) │ │
│ │ ● Social Scan 2/4  │ │
│ │ ● Build      3/5   │ │
│ │ ○ Email (in 12m)   │ │
│ └────────────────────┘ │
│                        │
│ ┌────────────────────┐ │
│ │ Schedule   This Wk │ │
│ │ 14:00 Social Scan  │ │
│ │ 18:00 Build Report │ │
│ │ 09:00 Email  tmrw  │ │
│ └────────────────────┘ │
│                        │
│ ┌────────────────────┐ │
│ │ Agent Reports [⚙]  │ │
│ │ ┌──────┐ ┌──────┐  │ │  Horizontal scroll
│ │ │Social│ │DevOps│→ │ │
│ │ └──────┘ └──────┘  │ │
│ └────────────────────┘ │
│                        │
│ ┌────────────────────┐ │
│ │ Recent Activity    │ │
│ │ ✅ Social   10m    │ │
│ │ ❌ Pipeline 45m    │ │
│ │ ✅ Email    2h     │ │
│ │     [View All →]   │ │
│ └────────────────────┘ │
└────────────────────────┘
```

---

## Widget Specifications

### 1. KPI Stats Row (existing — keep as-is)
- Already built as `StatsBar` component
- 4 cards: Working Now, Channels Live, Completed Today, Needs Attention
- Animated value changes (via `RealtimeMetricCard` pattern)
- **Change:** Add trend arrows (up/down vs yesterday)

### 2. Active Now Widget (NEW)
- Shows currently running + pending executions
- Each item shows: name, status (running/pending), step progress bar, elapsed time
- Clicking an item → deep-links to ExecutionKitchen (`/workflows?openExecution=...`)
- Auto-refreshes every 10s (only when items are running)
- Empty state: "All quiet — nothing running right now"
- **Data source:** `GET /api/activity/feed?status=running,pending&limit=5`

### 3. Schedule Widget (NEW)
- Mini week view: Mon-Sat with dots for scheduled items
- Today column highlighted
- "Upcoming" list below: next 4-5 scheduled routines/recipes with time + frequency
- Clicking an item → navigates to that routine's agent config or recipe page
- **Data source:** `GET /api/activity/schedule?range=7d` (new endpoint)
  - Returns: `{ scheduled: [{ name, type, next_run_at, frequency, agent_name }] }`

### 4. Agent Reports Widget (NEW — key feature)
- Shows pinned agents' latest execution summaries
- User can **pin/unpin agents** via settings gear → opens a multi-select popover
- Default: first 3 agents that have heartbeat routines
- Each card shows: agent avatar/icon, name, last run time, status, summary excerpt
- "View Report" button → opens ExecutionDetail inline or navigates to full view
- Pin icon shows pinned state, click to unpin
- **Data source:** `GET /api/activity/agent-reports?agent_ids=1,2,3` (new endpoint)
  - Returns: `{ reports: [{ agent_id, agent_name, last_run, status, summary, execution_id }] }`
- **Persistence:** Pinned agent IDs saved to localStorage + user preferences API

### 5. Recent Activity Widget (simplified feed)
- Compact table/list: 5 most recent completed/failed executions
- Columns: status icon, name, result, duration, time ago, [View] button
- "View All →" link switches to full Feed tab (or could be infinite scroll)
- Clicking [View] on a recipe → deep-links to ExecutionKitchen
- **Data source:** Existing `GET /api/activity/feed?status=completed,failed&limit=5`

---

## Customize Mode

The `[Customize ⚙]` button in the header enables a layout edit mode:

```
┌──────────────────────────────────────────────────────────────────────┐
│ Customizing Dashboard                  [Reset Layout] [Done ✓]      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Widgets now show drag handles and resize grips.                     │
│  Blue dashed outlines appear around each widget.                     │
│  User can drag widgets to reorder, resize by pulling corners.        │
│                                                                      │
│  Sidebar panel slides in from right:                                 │
│  ┌──────────────────────┐                                            │
│  │ Available Widgets    │                                            │
│  │                      │                                            │
│  │ ☑ KPI Stats          │                                            │
│  │ ☑ Active Now         │                                            │
│  │ ☑ Schedule           │                                            │
│  │ ☑ Agent Reports      │                                            │
│  │ ☑ Recent Activity    │                                            │
│  │ ☐ Token Usage (new)  │  ← future widgets can be toggled on       │
│  │ ☐ Cost Tracker (new) │                                            │
│  └──────────────────────┘                                            │
│                                                                      │
│  Layout saved to localStorage on [Done].                             │
│  [Reset Layout] restores default grid positions.                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What Changes from Current Page

| Current                        | New Dashboard                           |
|--------------------------------|-----------------------------------------|
| Feed tab = flat list of items  | Feed tab = widget grid dashboard        |
| No calendar/schedule view      | Schedule widget with week view          |
| No agent reports               | Pinnable agent report cards             |
| No live progress               | Active Now with step progress bars      |
| No drag-and-drop               | `react-grid-layout` with persistence    |
| Routines tab (unchanged)       | Stays as-is                             |
| Recipes tab (unchanged)        | Stays as-is                             |
| Missions tab (unchanged)       | Stays as-is                             |

---

## Implementation Approach

### Reuse from PRD-38.*
- `react-grid-layout` (already in package.json)
- `workspace-store.ts` patterns for position/size persistence
- `WidgetBase.tsx` chrome (title, actions bar, drag handle)
- `LayoutPresets.tsx` pattern for grid calculations
- `RealtimeMetricCard` for KPI animations

### New Components
```
frontend/components/activity/
  command-centre-dashboard.tsx    # Grid layout orchestrator (replaces ActivityFeed in Feed tab)
  widgets/
    active-now-widget.tsx         # Running/pending executions
    schedule-widget.tsx           # Mini calendar + upcoming list
    agent-reports-widget.tsx      # Pinned agent summaries
    recent-activity-widget.tsx    # Compact completed/failed list
    dashboard-customize.tsx       # Layout edit mode + widget toggle panel
```

### New Backend Endpoints
```
GET /api/activity/schedule?range=7d        # Scheduled routines/recipes
GET /api/activity/agent-reports?agent_ids=  # Latest reports from pinned agents
```

### No Changes To
- Routines tab, Recipes tab, Missions tab
- ExecutionKitchen, ExecutionDetail
- Sidebar navigation
- Activity stats endpoint (already serves KPI data)

---

## Open Questions for Review

1. **Agent Reports — fallback for agents with no routines?** Show "No reports yet — set up a routine" or hide the card?
2. **Schedule — show past items?** Or only future scheduled items?
3. **Customize — persist server-side?** localStorage only (fast, simple) or sync to user preferences API (works across devices)?
4. **Widget count cap?** Allow users to add unlimited widgets or cap at 6-8 for performance?
