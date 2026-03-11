# PRD-72: Activity Command Centre v2

**Version:** 2.0
**Status:** Draft
**Priority:** P1
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-03-07
**Updated:** 2026-03-11
**Dependencies:** PRD-10 (Workflow Engine — COMPLETE), PRD-55 (Agent Heartbeats — COMPLETE), PRD-06 (Monitoring & Analytics — COMPLETE), PRD-76 (Agent Reporting — COMPLETE), PRD-77 (Agent Self-Scheduling — IN PROGRESS)
**Branch:** `feat/activity-command-centre-v2`
**Inspiration:** [crshdn/mission-control](https://github.com/crshdn/mission-control) — Kanban patterns, `@hello-pangea/dnd`, SSE real-time

---

## Executive Summary

The Activity page is the nerve centre of Automatos — but v1 was a list-based feed that required clicking through multiple pages to understand what's happening. Users couldn't track task progress, manage workload, review agent output, or see their schedule without navigating 3-4 different views.

v2 rebuilds Activity as a **five-tab operational dashboard** that replaces the need for Jira, combines Feed + Reports into a Kanban board, adds a Calendar for scheduled tasks, enhances Memory with a daily journal view, and introduces a Projects page for multi-agent initiatives.

### What's Changing

| v1 Tab | v2 Tab | What Changed |
|---|---|---|
| Dashboard | **Summary** | Rename + new analytics widgets (status donut, priority breakdown, workload, types of work) |
| Feed | **Board** | List → Kanban board with drag-and-drop, agent sidebar, unified task viewer |
| Reports | *(merged into Board)* | Reports are now viewable inside task cards at Review/Done stages |
| Memory | **Memory** | Enhanced: two-panel layout, daily journal, organized by day/agent/type |
| Missions | **Projects** | Evolved from placeholder into project cards with progress tracking |
| *(new)* | **Calendar** | Full scheduler view: always-running tasks, week grid, next-up list |

### What's New (Cross-Cutting)

- **Global Search** (Cmd+K) — spotlight-style overlay searching tasks, memories, documents, agents
- **SSE real-time updates** — Board receives live task status changes via Server-Sent Events instead of polling
- **Unified Task Viewer** — one slide-over component replaces separate Feed detail, Reports viewer, and Execution detail

### What We're NOT Building

- New execution engine (existing recipe executor + heartbeat service are sufficient)
- Agent Zero / Orchestrator rename (separate effort, out of scope)
- Chat entries on the Board (chat history stays at `/chat`)
- Analytics replacement (Analytics page stays for cost/performance trends)

---

## 1. Naming Convention

### User-Facing Terminology

UI displays "Task" — backend remains "Recipe". No backend rename.

| Technical Concept | User-Facing Name | Icon | Description |
|---|---|---|---|
| Recipe execution (single/multi-step) | **Task** | `CheckSquare` | A unit of work — one step or many |
| Agent heartbeat | **Routine** | `RefreshCw` | Recurring background check |
| Complex dynamic workflow | **Project** | `FolderKanban` | Multi-task initiative with progress tracking |
| Chat + tool calls | **Chat** | `MessageCircle` | Conversations (not shown on Board) |

### Task Status Labels (Board Columns)

| Status | Column | Badge Colour | CSS Variable |
|---|---|---|---|
| `inbox` | Inbox | `--muted` | `bg-muted/10 text-muted-foreground` |
| `assigned` | Assigned | `--agent` | `bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))]` |
| `in_progress` | In Progress | `--info` | `bg-[hsl(var(--info))]/10 text-[hsl(var(--info))]` |
| `review` | Review | `--warning` | `bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))]` |
| `done` | Done | `--success` | `bg-[hsl(var(--success))]/10 text-[hsl(var(--success))]` |
| `failed` | Done (with error) | `--destructive` | `bg-destructive/10 text-destructive` |

### Priority Levels

| Priority | Colour | Indicator |
|---|---|---|
| `urgent` | `--destructive` | Red dot / border |
| `high` | `--warning` | Amber badge |
| `medium` | `--info` | Blue badge |
| `low` | `--muted` | Gray badge |

---

## 2. Design System Compliance

All components follow the established Automatos design language from `globals.css`. No custom colours or one-off styling.

### Required Patterns

| Pattern | Class / Token | Used By |
|---|---|---|
| Page header | `<PageHeader>` shared component | Title with `gradient-text` accent |
| Stat cards | `<StatsBar>` shared component | 4-up hero stats, `glass-card` + `card-glow` |
| Tab navigation | `<FilterTabs>` shared component | Icon + label tabs, `bg-secondary/40 backdrop-blur` |
| Content cards | `glass-card` class | All card surfaces |
| Drill-down panels | `glass-panel` class | Task viewer slide-over |
| Status badges | Semantic `Badge` variants | Using CSS variable palette above |
| Charts | Recharts with `MODEL_COLORS` palette | Donut, bar charts on Summary |
| Motion | `framer-motion` staggered entrance | `initial → animate` with `delay: index * 0.08` |
| Empty states | Centered icon + two-line text | Muted icon (30% opacity) + description |
| Loading | `<Skeleton>` matching card layout | Skeleton grid matching real content shape |
| Mobile | Reduced `backdrop-blur(8px)` | Per `globals.css` `@media (max-width: 767px)` |

### Typography Scale

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
Warning/Amber:   hsl(var(--warning))        — review, attention
Destructive/Red: hsl(var(--destructive))    — failed, errors
Agent/Purple:    hsl(var(--agent))          — agent-related, assigned
Muted:           hsl(var(--muted-foreground)) — inactive, secondary text
```

---

## 3. Page Layout

### Route

```
/activity (unchanged from v1)
```

### Tab Structure

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
│    { label: "Agents Active", value: "11", icon: Users }       │
│    { label: "Tasks in Queue", value: "35", icon: ListTodo }   │
│    { label: "Needs Attention", value: "1", icon: AlertTriangle│
│  ]} />                                                        │
├──────────────────────────────────────────────────────────────┤
│  <FilterTabs tabs={[                                          │
│    { value: "summary",  label: "Summary",  icon: LayoutDashboard }│
│    { value: "board",    label: "Board",    icon: Columns3 }   │
│    { value: "calendar", label: "Calendar", icon: Calendar }   │
│    { value: "memory",   label: "Memory",   icon: Brain }      │
│    { value: "projects", label: "Projects", icon: FolderKanban }│
│  ]}>                                                          │
│    <TabsContent value="summary">   → Section 4              │
│    <TabsContent value="board">     → Section 5              │
│    <TabsContent value="calendar">  → Section 6              │
│    <TabsContent value="memory">    → Section 7              │
│    <TabsContent value="projects">  → Section 8              │
│  </FilterTabs>                                                │
└──────────────────────────────────────────────────────────────┘
```

Deep-link support: `?tab=board&task_id=123` opens Board with task viewer for that task.

---

## 4. Summary Tab (Default)

Renamed from "Dashboard". Draggable widget grid using `react-grid-layout` (already installed).

### 4.1 Existing Widgets (Keep)

| Widget | Component | Description |
|---|---|---|
| Active Now | `active-now-widget.tsx` | Top 5 currently running tasks with progress bars |
| Schedule | `schedule-widget.tsx` | Week calendar dots + upcoming 5 items. **Click "View All" → Calendar tab** |
| Agent Reports | `agent-reports-widget.tsx` | Pinned agent report cards |
| Recent Activity | `recent-activity-widget.tsx` | Last 5 completed + 3 failed. **Click "View All" → Board tab** |

### 4.2 New Widgets

#### Status Overview (Donut Chart)

```
┌─ glass-card ──────────────────────────────┐
│  Status Overview              [View all →] │
│                                            │
│         ┌─────────┐                        │
│        ╱    77     ╲    ● In Review: 23    │
│       │  Total work │   ● Done: 17         │
│        ╲   items   ╱    ● To Do: 20        │
│         └─────────┘     ● In Progress: 17  │
│                                            │
└────────────────────────────────────────────┘
```

- Recharts `PieChart` with `innerRadius` for donut
- Counts from Board data (`GET /api/activity/board/stats`)
- Click segment → navigates to Board tab filtered by that status
- Click "View all" → Board tab

#### Priority Breakdown (Bar Chart)

```
┌─ glass-card ──────────────────────────────┐
│  Priority Breakdown                        │
│                                            │
│  60 ┤                                      │
│  40 ┤            ██                        │
│  20 ┤      ██    ██                        │
│   0 ┤  ██  ██    ██    ██    ██            │
│     Urgent High Medium  Low  None          │
└────────────────────────────────────────────┘
```

- Recharts `BarChart`
- Shows task count by priority level
- Colour-coded bars using priority colours from Section 1

#### Types of Work (Horizontal Bar)

```
┌─ glass-card ──────────────────────────────┐
│  Types of Work                 [View all →]│
│                                            │
│  Type          Distribution                │
│  ⚡ Routine    ████████████████████  65%    │
│  ✅ Task       ███████████  28%             │
│  📁 Project    ███  7%                      │
└────────────────────────────────────────────┘
```

- Shows distribution of work items by type
- Horizontal stacked bars

#### Team Workload (Agent Distribution)

```
┌─ glass-card ──────────────────────────────┐
│  Team Workload                             │
│                                            │
│  Assignee        Work distribution         │
│  🤖 Sentinel     ████████████████  12      │
│  📝 Loki         ██████████  8             │
│  🔍 Vision       ████████  6               │
│  ⚡ Pepper       ██████  4                 │
│  ○ Unassigned    ████  3                   │
└────────────────────────────────────────────┘
```

- Horizontal bars per agent
- Click agent row → Board filtered by that agent

### 4.3 Widget Grid

All 8 widgets rendered via `react-grid-layout` with drag-to-reorder. Layout persisted in localStorage (`automatos:dashboard-layout`). "Customize" toggle + "Reset Layout" button (existing pattern from v1).

Default layout (2-column on desktop):

```
Row 1: [Active Now          ] [Status Overview    ]
Row 2: [Schedule             ] [Priority Breakdown ]
Row 3: [Recent Activity      ] [Types of Work      ]
Row 4: [Agent Reports        ] [Team Workload      ]
```

---

## 5. Board Tab (Replaces Feed + Reports)

The centrepiece of v2. A Kanban board with agent sidebar, drag-and-drop task management, and a unified task viewer that replaces the separate Feed detail and Reports pages.

### 5.1 Layout

```
┌──────────────────────────────────────────────────────────────┐
│  [Filter: agent | priority | type | date]   11 agents | 35 tasks │
├──────────┬───────────────────────────────────────────────────┤
│ ● AGENTS │  ● TASK BOARD                                     │
│    (12)  │                                                    │
│          │  ● INBOX  ● ASSIGNED  ● IN PROGRESS  ● REVIEW  ● DONE │
│ Bhanu    │    (11)     (10)        (7)           (5)      (0) │
│ LEAD     │                                                    │
│ ●WORKING │  ┌────┐   ┌────┐     ┌────┐        ┌────┐        │
│          │  │card│   │card│     │card│        │card│        │
│ Friday   │  │    │   │    │     │    │        │    │        │
│ INT      │  └────┘   └────┘     └────┘        └────┘        │
│ ●WORKING │  ┌────┐   ┌────┐     ┌────┐        ┌────┐        │
│          │  │card│   │card│     │card│        │card│        │
│ Fury     │  └────┘   └────┘     └────┘        └────┘        │
│ SPC      │                                                    │
│ ●WORKING │           ┌────┐     ┌────┐                       │
│          │           │card│     │card│                       │
│ ...      │           └────┘     └────┘                       │
└──────────┴───────────────────────────────────────────────────┘
```

### 5.2 Agent Sidebar

Left panel showing all workspace agents:

```
┌─ AGENTS (12) ────────────────┐
│                               │
│  🤖 Bhanu          LEAD      │
│     Founder        ● WORKING │
│                               │
│  ⚔️ Friday         INT       │
│     Developer      ● WORKING │
│                               │
│  🔥 Fury           SPC       │
│     Customer Rese… ● WORKING │
│                               │
│  ...                          │
└───────────────────────────────┘
```

- Agent avatar, name, role badge (LEAD/INT/SPC), status dot
- Click agent → filters board to show only that agent's tasks
- Active filter shown as highlighted agent row
- Click again to clear filter
- Collapsible on mobile (hamburger toggle)

Data source: `GET /api/agents?workspace_id=X` (existing endpoint)

### 5.3 Board Columns

5 columns using `@hello-pangea/dnd` (fork of react-beautiful-dnd, actively maintained):

| Column | Status | Header Colour | Description |
|---|---|---|---|
| **Inbox** | `inbox` | `--muted` | New tasks not yet assigned |
| **Assigned** | `assigned` | `--agent` | Assigned to an agent, waiting to start |
| **In Progress** | `in_progress` | `--info` | Currently executing |
| **Review** | `review` | `--warning` | Awaiting human or LLM review |
| **Done** | `done` | `--success` | Completed (success or failure) |

Each column header shows: `● COLUMN NAME  (count)`

Columns are scrollable vertically. Cards within a column are ordered by priority (urgent first), then by `updated_at DESC`.

### 5.4 Board Card

Each task is a draggable card:

```
┌─ glass-card draggable ────────────────────┐
│  ↕ Explore SiteGPT Dashboard &            │
│    Document All Features                   │
│                                            │
│    Thoroughly explore the entire            │
│    SiteGPT dashboard...                    │
│                                            │
│  🤖 Loki              1 day ago            │
│  [research] [documentation] [sitegpt]      │
│                                    [→]     │
└────────────────────────────────────────────┘
```

**Card fields:**
- Task name (line-clamp-2)
- Description snippet (line-clamp-2, `text-sm text-muted-foreground`)
- Agent avatar + name (bottom-left)
- Time ago (bottom-right)
- Tags as pills (`text-xs` badges)
- Priority indicator: left border colour or small dot next to title
- Arrow button → opens Task Viewer
- For "In Progress": mini progress bar showing step X of Y

**Drag behaviour:**
- `@hello-pangea/dnd` `<DragDropContext>` wrapping the board
- Each column is a `<Droppable>`
- Each card is a `<Draggable>`
- On drag end: optimistic UI update → `PATCH /api/activity/tasks/{id}/status`
- On API failure: revert to previous position + toast error
- Cards in "In Progress" with `running` backend status: draggable (user can force-move to Review/Done)
- Cards in "Done" with `completed` status: draggable back to Inbox for re-run

### 5.5 Task Viewer (Slide-Over)

Replaces: `activity-feed-item.tsx` detail, `report-viewer.tsx`, `execution-detail.tsx`

Opens as a right slide-over panel when clicking a card's arrow button or double-clicking a card.

Content adapts based on the task's current column:

#### Inbox / Assigned View

```
┌─ Task Viewer ─────────────────────────────────────┐
│  ← Back                                    [Edit] │
│                                                    │
│  Explore SiteGPT Dashboard                        │
│  Created by Henry · 1 day ago                      │
│                                                    │
│  Description:                                      │
│  Thoroughly explore the entire SiteGPT dashboard   │
│  and document all features...                      │
│                                                    │
│  ┌─ Details ──────────────────────────────────┐   │
│  │ Assignee:  [🤖 Loki ▼]                     │   │
│  │ Priority:  [● High ▼]                      │   │
│  │ Due Date:  [Mar 15, 2026]                   │   │
│  │ Tags:      [research] [documentation] [+]   │   │
│  │ Steps:     3 steps defined                  │   │
│  │ Review:    [Human ▼] (Human / LLM / Auto)  │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  [Start Task →]                                    │
└────────────────────────────────────────────────────┘
```

- Editable fields: assignee (agent dropdown), priority, due date, tags, review mode
- "Start Task" moves to In Progress and triggers execution

#### In Progress View

```
┌─ Task Viewer ─────────────────────────────────────┐
│  ← Back                                   [Stop]  │
│                                                    │
│  Explore SiteGPT Dashboard                        │
│  🤖 Loki · Working... · Step 2 of 3              │
│                                                    │
│  ┌─ Progress ─────────────────────────────────┐   │
│  │ ✓ Step 1: Pull dashboard data (45s)         │   │
│  │ ● Step 2: Analyze features...               │   │
│  │ ○ Step 3: Write documentation               │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ Live Logs ────────────────────────────────┐   │
│  │ 09:15:00  Started execution                 │   │
│  │ 09:15:02  Step 1: Fetching dashboard...     │   │
│  │ 09:15:47  Step 1: Complete (45s)            │   │
│  │ 09:15:48  Step 2: Analyzing features...     │   │
│  │ ● (streaming...)                            │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  [Move to Review]                                  │
└────────────────────────────────────────────────────┘
```

- Reuses internals from existing `execution-detail.tsx`
- Live log streaming via SSE
- Step pipeline with `stage-completed`, `stage-active`, `stage-pending` CSS classes
- "Stop" button cancels execution
- "Move to Review" manually advances (or auto-advances on completion)

#### Review View

```
┌─ Task Viewer ─────────────────────────────────────┐
│  ← Back                              Review Mode: │
│                                      [Human ▼]    │
│  Explore SiteGPT Dashboard                        │
│  🤖 Loki · Completed in 2m 34s · Awaiting Review │
│                                                    │
│  ┌─ Report ───────────────────────────────────┐   │
│  │ ## SiteGPT Dashboard Analysis              │   │
│  │                                             │   │
│  │ Key findings:                               │   │
│  │ - Feature A does X                          │   │
│  │ - Feature B does Y                          │   │
│  │ ...                                         │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ Grade ────────────────────────────────────┐   │
│  │ Rating: ★★★★☆                              │   │
│  │ Notes:  [                                ]  │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  [✗ Reject → Inbox] [✓ Approve → Done]            │
└────────────────────────────────────────────────────┘
```

- Renders report content as markdown (reuses `report-viewer.tsx` internals)
- Grade form with star rating + notes (reuses `report-grade-form.tsx`)
- Review mode selector: Human / LLM / Auto
  - **Human**: requires manual approve/reject
  - **LLM**: another agent reviews and auto-approves/rejects based on quality threshold
  - **Auto**: moves to Done automatically on completion (no review step)
- Reject sends back to Inbox with rejection notes
- Approve moves to Done with grade saved

#### Done View

```
┌─ Task Viewer ─────────────────────────────────────┐
│  ← Back                                           │
│                                                    │
│  Explore SiteGPT Dashboard          ✓ Completed   │
│  🤖 Loki · 2m 34s · Mar 7, 09:17                 │
│  Grade: ★★★★☆                                     │
│                                                    │
│  ┌─ Results ──────────────────────────────────┐   │
│  │ ## SiteGPT Dashboard Analysis              │   │
│  │ (full report content)                       │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ Execution Summary ────────────────────────┐   │
│  │ Duration: 2m 34s                            │   │
│  │ Tokens:   4,521                             │   │
│  │ Steps:    3/3 completed                     │   │
│  │ Trigger:  Manual                            │   │
│  └────────────────────────────────────────────┘   │
│                                                    │
│  [↻ Re-run] [📥 Download Report] [📋 Copy]       │
└────────────────────────────────────────────────────┘
```

- Full report content rendered
- Execution metadata: duration, tokens, steps, trigger
- Re-run creates a new execution of the same recipe
- Download as markdown
- Failed tasks show error message + stack trace in collapsible section

### 5.6 Creating Tasks

"New Task" button in Board header → opens existing `create-recipe-modal.tsx`:
- 1-step recipe = simple task
- N-step recipe = multi-step workflow
- **No backend changes needed** — recipe model already supports both

Task can also be created by:
- Agents spawning sub-tasks via `platform_schedule_task` tool (PRD-77)
- QA agent adding bug-fix tasks to Inbox
- Redis queue pushing new tasks (existing infrastructure)

### 5.7 Board Filters

Filter bar above the board:

```
[Agent ▼] [Priority ▼] [Type ▼] [Date Range ▼]  |  🔍 Search tasks...
```

- **Agent**: multi-select dropdown of workspace agents
- **Priority**: multi-select (Urgent/High/Medium/Low)
- **Type**: Routine / Task (single-step) / Workflow (multi-step)
- **Date Range**: Today / This Week / This Month / Custom
- **Search**: instant filter by task name/description

Filters persist in URL query params for shareability.

### 5.8 Real-Time Updates (SSE)

Board uses Server-Sent Events instead of polling:

```
GET /api/activity/board/stream?workspace_id=X
```

**SSE Event Types:**

| Event | Payload | UI Effect |
|---|---|---|
| `task_created` | Full task object | Card appears in Inbox with `log-slide-in` animation |
| `task_updated` | `{ id, status, agent_id, ... }` | Card moves to new column with animation |
| `task_progress` | `{ id, step_current, step_total, log_line }` | Progress bar updates, log line appends |
| `task_completed` | `{ id, status, duration, report_id }` | Card moves to Review or Done |
| `task_failed` | `{ id, error_message }` | Card shows error badge, moves to Done |

**Fallback:** If SSE connection drops, fall back to 60s polling via `useActivityFeed()` (existing hook). Reconnect SSE with exponential backoff (1s, 2s, 4s, max 30s).

**Backend implementation:**
- New SSE endpoint in `orchestrator/api/activity.py`
- Publishes events from: recipe executor, heartbeat service, manual status changes
- Uses Redis pub/sub as event bus (existing Redis infrastructure)

---

## 6. Calendar Tab

Full-page scheduler view showing all of Automatos's scheduled and recurring tasks. Lets users verify what's running proactively, spot unwanted schedules, and click through to task details.

### 6.1 Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Scheduled Tasks                         [Week] [Today] [↻] │
│  {user}'s automated routines                                 │
├──────────────────────────────────────────────────────────────┤
│  ⚡ Always Running                                           │
│  ┌────────────────────────────────────────────┐              │
│  │ mission control check • Every 30 min       │              │
│  └────────────────────────────────────────────┘              │
├──────────────────────────────────────────────────────────────┤
│  Sun    │  Mon    │  Tue    │  Wed    │  Thu    │ *Fri* │ Sat│
│ ┌─────┐│┌─────┐  │┌─────┐  │┌─────┐  │┌─────┐  │┌─────┐│┌──┐│
│ │ai sc│││ai sc│  ││ai sc│  ││ai sc│  ││ai sc│  ││ai sc││    │
│ │5:00A│││5:00A│  ││5:00A│  ││5:00A│  ││5:00A│  ││5:00A││    │
│ ├─────┤│├─────┤  │├─────┤  │├─────┤  │├─────┤  │├─────┤│    │
│ │morn │││morn │  ││morn │  ││morn │  ││morn │  ││morn ││    │
│ │8:00A│││8:00A│  ││8:00A│  ││8:00A│  ││8:00A│  ││8:00A││    │
│ ├─────┤│├─────┤  │├─────┤  │├─────┤  │├─────┤  │├─────┤│    │
│ │comp │││comp │  ││newsl│  ││comp │  ││comp │  ││comp ││    │
│ │10:00│││10:00│  ││9:00A│  ││10:00│  ││10:00│  ││10:00││    │
│ └─────┘│└─────┘  │├─────┤  │└─────┘  │└─────┘  │└─────┘│    │
│        │         ││comp │  │         │         │       │    │
│        │         ││10:00│  │         │         │       │    │
│        │         │└─────┘  │         │         │       │    │
├──────────────────────────────────────────────────────────────┤
│  📅 Next Up                                                  │
│  mission control check .......................... In 30 min  │
│  competitor youtube scan ........................ In 1 hours  │
│  ai scarcity research .......................... In 20 hours │
│  morning brief .................................. In 23 hours │
│  newsletter reminder ............................ In 4 days  │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Three Sections

#### Always Running
- Shows high-frequency routines (interval < 1 hour)
- Pill badge: `{name} • Every {interval}`
- Click → opens task viewer

#### Week Grid
- 7-column day grid (Sun-Sat)
- Task pills colour-coded per agent (consistent colours from agent config)
- Shows task name (truncated) + scheduled time
- Parsed from cron expressions in `schedule_config`
- Current day column highlighted (today indicator)
- Click pill → opens task viewer slide-over

#### Next Up
- Sorted list of upcoming tasks by soonest
- Shows task name + relative time ("In 30 min", "In 1 hours")
- Colour-coded text matching agent colour
- Click row → opens task viewer

### 6.3 Data Source

Extend existing `useActivitySchedule()` hook:
- Already returns `next_run_at`, `frequency`, `agent_name`, `agent_id`
- Add: cron expression parsing to plot recurring items on week grid
- Add: "always running" filter for sub-hourly intervals
- Endpoint: `GET /api/activity/schedule?range=7d` (existing)

### 6.4 View Toggle

- **Week** (default): 7-day grid view
- **Today**: single-day view with hourly timeline
- **Refresh** button: re-fetches schedule data

### 6.5 Library Decision

Build custom (no external calendar library needed). The design is a simple week grid with pills — `@fullcalendar` is overkill. A CSS grid with mapped cron data is sufficient. If we later need month view or drag-to-reschedule, we can add `@fullcalendar` then.

---

## 7. Memory Tab

Two-panel layout inspired by daily journal apps. Browse and search through all of Automatos's memories, organized chronologically with rich content rendering.

### 7.1 Layout

```
┌─────────────────────────┬──────────────────────────────────────┐
│ 🔍 Search memory...     │  2026-02-26 — Thursday               │
│                         │  Thursday, Feb 26 · 4.8 KB · 772 w  │
│ 🧠 Long-Term Memory    │  ─────────────────────────────────── │
│    1,608 words          │                                      │
│    Updated 22 hours ago │  ⏰ 9:00 AM — Qwen 3.5 Medium       │
│                         │     Series Research                  │
│ 📖 DAILY JOURNAL       │                                      │
│    37 entries           │  What we discussed: Alex shared the  │
│                         │  Qwen 3.5 Medium announcement...     │
│ ▼ Yesterday (1)        │                                      │
│   📅 Mon, Mar 2        │  Key findings:                       │
│   8.7 KB • 1,253 words │  - 35B-A3B beats old 235B flagship   │
│                         │  - 122B-A10B matches 397B            │
│ ▼ This Week (1)        │  - 27B dense gets best SWE-bench     │
│   📅 Sun, Mar 1        │                                      │
│   6.8 KB • 1,068 words │  Recommendations given:              │
│                         │  1. Keep 397B on Studio 2            │
│ ▼ February 2026 (25)   │  2. Add 35B-A3B on Studio 1          │
│   📅 Sat, Feb 28       │  3. 122B-A10B as potential upgrade    │
│   12.6 KB • 2,001 w    │                                      │
│   📅 Fri, Feb 27       │  Decision: Pending                   │
│   3.9 KB • 614 words   │  ─────────────────────────────────── │
│   📅 Thu, Feb 26  ◀    │                                      │
│   4.8 KB • 772 words   │  Overnight — Reborn Factory Results  │
│   ...                   │  ...                                  │
└─────────────────────────┴──────────────────────────────────────┘
```

### 7.2 Left Sidebar

#### Search
- Full-text search across all memories
- **Fix required:** verify `Mem0Client.search()` POST fix merged from `fix-memory` branch
- Results replace the journal list with search results, ranked by relevance score
- Keyboard: Enter to search, Esc to clear

#### Long-Term Memory Card
- Shows total word count, last updated timestamp
- Click → displays consolidated long-term memory in right panel
- Badge: `🧠` emoji or Brain icon

#### Daily Journal
- Grouped by time period:
  - **Yesterday** (collapsible)
  - **This Week** (collapsible)
  - **{Month Year}** (collapsible) — e.g., "February 2026 (25)"
- Each entry row: `📅 {Day, Mon DD}` + `{size} • {word count}`
- Click entry → loads full content in right panel
- Active entry highlighted with `◀` indicator and `bg-primary/10` background

### 7.3 Right Content Panel

- Renders selected memory entry as rich markdown
- Header: `{date} — {day name}` + metadata (date, size, word count, modified time)
- Content sections with timestamp headers: `⏰ 9:00 AM — {Topic}`
- Structured entries with: "What we discussed", "Key findings", "Recommendations", "Decision"
- Code blocks syntax-highlighted
- Images rendered inline

### 7.4 Organize By (Toggle)

Above the journal list, a segmented control:

```
[By Day] [By Agent] [By Type]
```

- **By Day** (default): grouped by date as shown above
- **By Agent**: grouped by agent name, each agent section shows their memories
- **By Type**: grouped by category (conversation, task, document, insight, research)

### 7.5 Enhanced Filters

- **Agent** dropdown (multi-select)
- **Date range** picker
- **Type/Category** filter (global / agent-specific)
- **Sort**: Newest (default) / Oldest / Relevance (when searching)

### 7.6 API Changes

| Endpoint | Change |
|---|---|
| `GET /api/memory/browse` | Add `group_by` param: `day` / `agent` / `type` |
| `GET /api/memory/browse` | Add `date_from`, `date_to` params |
| `POST /api/memory/search` | Verify working (Mem0 POST fix) |
| `GET /api/memory/health` | Existing — no changes |
| `GET /api/memory/journal/{date}` | **New** — returns full day's memory content as structured markdown |

---

## 8. Projects Tab

Evolved from the v1 "Missions" placeholder into a project management view. Projects are multi-task initiatives with progress tracking, assigned agents, and priority.

### 8.1 Layout

```
┌──────────────────────────────────────────────────────────────┐
│  📁 Projects                                     [+ New]     │
│  5 total • 2 active • 3 planning                             │
├──────────────────┬──────────────────┬───────────────────────┤
│ Agent Org Infra  │ Mission Control  │ Skool AI Extension    │
│ [Active]         │ [Active]         │                       │
│ Core infra for   │ Central dash-    │ Chrome ext for Vibe   │
│ the autonomous   │ board for agent  │ Code Academy. RAG     │
│ agent org...     │ activity, docs...│ pipeline over...      │
│                  │                  │                       │
│ ████████ 100%    │ ██████░░ 70%     │ ░░░░░░░░ 0%          │
│ 10/10            │ 0/0              │                       │
│                  │                  │                       │
│ 🟢 Charlie       │ 🟢 Henry         │ 🟢 Henry              │
│ [high]   8d ago  │ [high]   8d ago  │         8d ago        │
├──────────────────┼──────────────────┤                       │
│ Micro-SaaS       │ Even G2 Integr.  │                       │
│ [Planning]       │ [Planning]       │                       │
│ Violet's opp     │ Smart glasses    │                       │
│ engine — market  │ bridge app...    │                       │
│ gaps, validate...│                  │                       │
│                  │                  │                       │
│ ░░░░░░░░ 0%     │ ░░░░░░░░ 0%     │                       │
│ 0/0              │ 0/0              │                       │
│                  │                  │                       │
│ 🟣 Violet        │ ○ Unassigned     │                       │
│ [medium]  8d ago │ [medium]  8d ago │                       │
└──────────────────┴──────────────────┴───────────────────────┘
```

### 8.2 Project Card

```
┌─ glass-card ──────────────────────────────┐
│  Project Name                    [Active] │
│                                            │
│  Description text (line-clamp-3)...        │
│                                            │
│  ████████████░░░░░░░░  70%       7/10     │
│                                            │
│  🟢 Agent Name                             │
│  [priority]                    time ago    │
└────────────────────────────────────────────┘
```

**Card fields:**
- Project name
- Status badge: `Active` (green) / `Planning` (blue) / `Complete` (muted) / `On Hold` (amber)
- Description (line-clamp-3)
- Progress bar: completed tasks / total tasks, percentage
- Lead agent avatar + name
- Priority badge
- Time since creation

### 8.3 Project Detail (Click-Through)

Clicking a project card opens a detail view:

```
┌──────────────────────────────────────────────────────────────┐
│  ← Projects                                                  │
│                                                              │
│  Agent Org Infrastructure                         [Active]   │
│  Core infrastructure for the autonomous agent organization   │
│  Lead: 🟢 Charlie · Priority: High · Created 8 days ago     │
│                                                              │
│  ████████████████████ 100%    10/10 tasks                    │
│                                                              │
│  ┌─ Tasks ────────────────────────────────────────────────┐  │
│  │ ✓ Set up base agent framework                          │  │
│  │ ✓ Implement heartbeat service                          │  │
│  │ ✓ Configure inter-agent messaging                      │  │
│  │ ✓ Deploy monitoring stack                              │  │
│  │ ...                                                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  [+ Add Task] [Open in Board →]                              │
└──────────────────────────────────────────────────────────────┘
```

- Lists all tasks belonging to this project
- Each task row links to the Board task viewer
- "Add Task" creates a new task pre-linked to this project
- "Open in Board" navigates to Board tab filtered by this project's tasks

### 8.4 Data Model

Projects are a **new concept** — a group of recipes/tasks with shared metadata:

```python
class Project(Base):
    __tablename__ = 'projects'

    id: int  # PK
    workspace_id: UUID
    name: str
    description: str
    status: str  # 'planning', 'active', 'complete', 'on_hold'
    lead_agent_id: int | None  # FK to agents
    priority: str  # 'low', 'medium', 'high', 'urgent'
    created_at: datetime
    updated_at: datetime
```

Link table for project ↔ task (recipe execution) relationship:

```python
class ProjectTask(Base):
    __tablename__ = 'project_tasks'

    project_id: int  # FK to projects
    execution_id: str  # FK to recipe_executions
    order: int  # Display order within project
```

### 8.5 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/projects` | List projects for workspace |
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/projects/{id}` | Project detail with tasks |
| `PATCH` | `/api/projects/{id}` | Update project |
| `POST` | `/api/projects/{id}/tasks` | Link task to project |
| `DELETE` | `/api/projects/{id}/tasks/{exec_id}` | Unlink task |

---

## 9. Global Search (Cmd+K)

Spotlight-style search overlay accessible from anywhere in the app.

### 9.1 Trigger

- Keyboard: `Cmd+K` (Mac) / `Ctrl+K` (Windows/Linux)
- Click: Search icon in top navigation bar

### 9.2 Layout

```
┌──────────────────────────────────────────────────┐
│  🔍 Mac Studio                          ✕   ESC │
├──────────────────────────────────────────────────┤
│  TASKS                                           │
│  ┌──────────────────────────────────────────┐    │
│  │ ✅ Flesh out $10K Mac Studio use cases  ●│    │
│  │    Clawdbot                            → │    │
│  ├──────────────────────────────────────────┤    │
│  │ ✅ Local model recommendations for Mac ● │    │
│  │    Infrastructure                      → │    │
│  ├──────────────────────────────────────────┤    │
│  │ ✅ Research Exo Labs dual-Studio clust ● │    │
│  │    Mac Studio Launch                   → │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  MEMORIES                                        │
│  ┌──────────────────────────────────────────┐    │
│  │ 🧠 Mac Studio M2 Ultra performance...    │    │
│  │    Feb 26, 2026                         → │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ↑↓ navigate  ↵ select  esc close               │
├──────────────────────────────────────────────────┤
```

### 9.3 Search Categories

Results grouped by type:

| Category | Icon | Source |
|---|---|---|
| **Tasks** | `CheckSquare` | Recipe executions (name, description) |
| **Memories** | `Brain` | Memory search (content) |
| **Documents** | `FileText` | Document metadata (name) |
| **Agents** | `Bot` | Agent names, roles |
| **Projects** | `FolderKanban` | Project names, descriptions |

### 9.4 Result Item

Each result shows:
- Category icon
- Title (highlighted match)
- Subtitle (agent name, category, or date)
- Status dot (colour = current status for tasks)
- Arrow → navigates to item

### 9.5 Keyboard Navigation

- `↑↓` — move selection
- `Enter` — open selected item
- `Esc` — close search
- Type to filter in real-time (debounced 300ms)

### 9.6 API

```
GET /api/search?q={query}&workspace_id=X&types=tasks,memories,documents,agents,projects&limit=10
```

Returns results grouped by type, ordered by relevance. Backend searches across:
- `recipe_executions` (name, description via ILIKE)
- `memories` (Mem0 vector search)
- `documents` (name, metadata via ILIKE)
- `agents` (name, role via ILIKE)
- `projects` (name, description via ILIKE)

---

## 10. API Endpoints (Complete)

### 10.1 Existing Endpoints (No Changes)

| Method | Path | Used For |
|---|---|---|
| `GET` | `/api/recipes` | Recipe list (used by Board internally) |
| `GET` | `/api/recipes/{id}/executions` | Recipe execution history |
| `POST` | `/api/recipes/{id}/execute` | Manual recipe run |
| `GET` | `/api/activity/feed` | Feed data (fallback for Board polling) |
| `GET` | `/api/activity/stats` | Stats for hero cards |
| `GET` | `/api/activity/schedule` | Calendar data source |
| `GET` | `/api/agents` | Agent list for sidebar |
| `GET` | `/api/memory/health` | Memory health stats |
| `DELETE` | `/api/memory/{id}` | Delete memory |
| `POST` | `/api/memory/consolidate` | Merge/summarize memories |

### 10.2 Modified Endpoints

| Method | Path | Change |
|---|---|---|
| `GET` | `/api/activity/feed` | Add `status` filter for board columns (inbox, assigned, in_progress, review, done) |
| `GET` | `/api/memory/browse` | Add `group_by`, `date_from`, `date_to` params |

### 10.3 New Endpoints

| Method | Path | Description |
|---|---|---|
| `PATCH` | `/api/activity/tasks/{id}/status` | Update task status (drag-and-drop) |
| `PATCH` | `/api/activity/tasks/{id}` | Update task fields (assignee, priority, tags, due_date, review_mode) |
| `GET` | `/api/activity/board/stats` | Board column counts + priority/type/agent breakdowns for Summary widgets |
| `GET` | `/api/activity/board/stream` | **SSE** — real-time task events stream |
| `GET` | `/api/memory/journal/{date}` | Full day's memory as structured markdown |
| `GET` | `/api/projects` | List projects |
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/projects/{id}` | Project detail + linked tasks |
| `PATCH` | `/api/projects/{id}` | Update project |
| `POST` | `/api/projects/{id}/tasks` | Link task to project |
| `DELETE` | `/api/projects/{id}/tasks/{exec_id}` | Unlink task |
| `GET` | `/api/search` | Global search across all entity types |

### 10.4 Database Migrations

#### Migration 1: Board statuses + task fields

```sql
-- Add new statuses to recipe_executions
ALTER TABLE recipe_executions
  ADD COLUMN priority VARCHAR(10) DEFAULT 'medium',
  ADD COLUMN tags JSONB DEFAULT '[]',
  ADD COLUMN assignee_agent_id INTEGER REFERENCES agents(id),
  ADD COLUMN review_mode VARCHAR(10) DEFAULT 'human',
  ADD COLUMN due_date TIMESTAMP,
  ADD COLUMN reviewed_at TIMESTAMP,
  ADD COLUMN reviewed_by VARCHAR(50);

-- Update status enum to include new values
-- (status is VARCHAR, so just document valid values)
-- Valid: 'inbox', 'assigned', 'pending', 'running', 'in_progress', 'review', 'completed', 'done', 'failed', 'cancelled'
```

#### Migration 2: Projects table

```sql
CREATE TABLE projects (
  id SERIAL PRIMARY KEY,
  workspace_id UUID NOT NULL REFERENCES workspaces(id),
  name VARCHAR(255) NOT NULL,
  description TEXT,
  status VARCHAR(20) DEFAULT 'planning',
  lead_agent_id INTEGER REFERENCES agents(id),
  priority VARCHAR(10) DEFAULT 'medium',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE project_tasks (
  project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
  execution_id VARCHAR(255) REFERENCES recipe_executions(execution_id),
  "order" INTEGER DEFAULT 0,
  PRIMARY KEY (project_id, execution_id)
);

CREATE INDEX idx_projects_workspace ON projects(workspace_id);
CREATE INDEX idx_project_tasks_project ON project_tasks(project_id);
```

### 10.5 Key Type Definitions

```typescript
// Board task (extends ActivityFeedItem)
interface BoardTask {
  id: string
  type: 'routine' | 'recipe'
  name: string
  description?: string
  status: 'inbox' | 'assigned' | 'in_progress' | 'review' | 'done'
  priority: 'urgent' | 'high' | 'medium' | 'low'
  tags: string[]
  assignee?: {
    agent_id: number
    agent_name: string
    agent_icon?: string
  }
  creator?: string
  due_date?: string
  review_mode: 'human' | 'llm' | 'auto'
  started_at?: string
  completed_at?: string
  duration_ms?: number
  step_progress?: { current: number; total: number }
  error_message?: string
  report_id?: string
  source_id: string
  project_id?: number
}

// SSE event
interface BoardEvent {
  type: 'task_created' | 'task_updated' | 'task_progress' | 'task_completed' | 'task_failed'
  task: Partial<BoardTask> & { id: string }
  timestamp: string
}

// Project
interface Project {
  id: number
  name: string
  description: string
  status: 'planning' | 'active' | 'complete' | 'on_hold'
  lead_agent?: { id: number; name: string; icon?: string }
  priority: 'urgent' | 'high' | 'medium' | 'low'
  task_count: number
  completed_count: number
  progress_pct: number
  created_at: string
  updated_at: string
}

// Global search result
interface SearchResult {
  type: 'task' | 'memory' | 'document' | 'agent' | 'project'
  id: string
  title: string
  subtitle: string
  status?: string
  url: string
}
```

---

## 11. File Structure

```
frontend/
  app/
    activity/
      page.tsx                          # KEEP — route entry
  components/
    activity/
      activity-page.tsx                 # MODIFY — update tabs to Summary|Board|Calendar|Memory|Projects
      activity-feed.tsx                 # DELETE after Board is live (keep as fallback initially)
      activity-feed-item.tsx            # DELETE after Board is live
      activity-reports.tsx              # DELETE — merged into task viewer
      report-card.tsx                   # DELETE — replaced by board-card
      activity-missions.tsx             # MODIFY → activity-projects.tsx
      activity-memory.tsx               # REWRITE — two-panel layout
      memory-card.tsx                   # MODIFY → sidebar entry row
      execution-detail.tsx              # KEEP — internals reused in task viewer
      report-viewer.tsx                 # KEEP — internals reused in task viewer
      report-grade-form.tsx             # KEEP — embedded in Review view

      # NEW — Board
      board/
        board-view.tsx                  # Main kanban container with DragDropContext
        board-column.tsx                # Single droppable column
        board-card.tsx                  # Draggable task card
        board-filters.tsx              # Filter bar (agent, priority, type, date)
        board-agent-sidebar.tsx         # Agent roster sidebar
        board-task-viewer.tsx           # Unified slide-over (Inbox/InProgress/Review/Done views)

      # NEW — Calendar
      calendar/
        activity-calendar.tsx           # Main calendar container
        calendar-week-grid.tsx          # 7-day CSS grid with task pills
        calendar-always-running.tsx     # High-frequency routines section
        calendar-next-up.tsx            # Upcoming tasks list

      # NEW — Memory (rewrites)
      memory/
        memory-sidebar.tsx              # Left panel: search + journal list
        memory-viewer.tsx               # Right panel: content renderer
        memory-journal-entry.tsx        # Sidebar entry row (date + size)

      # NEW — Projects
      projects/
        activity-projects.tsx           # Project card grid
        project-card.tsx                # Individual project card
        project-detail.tsx              # Project detail with task list

      # NEW — Summary widgets
      widgets/
        command-centre-dashboard.tsx     # KEEP — drag grid container
        active-now-widget.tsx            # KEEP
        recent-activity-widget.tsx       # KEEP — add click-through to Board
        schedule-widget.tsx              # KEEP — add click-through to Calendar
        agent-reports-widget.tsx         # KEEP
        status-overview-widget.tsx       # NEW — donut chart
        priority-breakdown-widget.tsx    # NEW — bar chart
        types-of-work-widget.tsx         # NEW — horizontal bars
        team-workload-widget.tsx         # NEW — agent distribution bars

      # NEW — Global Search
    global-search/
      global-search.tsx                 # Cmd+K overlay
      search-result-item.tsx            # Individual result row

  hooks/
    use-activity-api.ts                 # MODIFY — add board stats, 60s polling
    use-reports-api.ts                  # KEEP
    use-memory-explorer-api.ts          # MODIFY — add group_by, date range
    use-board-sse.ts                    # NEW — SSE hook for real-time board
    use-board-tasks.ts                  # NEW — React Query + SSE hybrid
    use-projects-api.ts                 # NEW — project CRUD hooks
    use-global-search.ts               # NEW — debounced search hook
    use-memory-journal.ts              # NEW — journal day content hook

  types/
    board.ts                            # NEW — BoardTask, BoardEvent types
    project.ts                          # NEW — Project types
    search.ts                           # NEW — SearchResult types

orchestrator/
  api/
    activity.py                         # MODIFY — add board/stream SSE, tasks PATCH, board/stats
    projects.py                         # NEW — project CRUD endpoints
    search.py                           # NEW — global search endpoint
    memory.py                           # MODIFY — add journal/{date} endpoint, group_by param
  core/
    models/
      core.py                           # MODIFY — add Project, ProjectTask models + recipe_execution fields
  services/
    activity_service.py                 # MODIFY — add board stats aggregation
    project_service.py                  # NEW — project business logic
    search_service.py                   # NEW — cross-entity search
```

---

## 12. Implementation Phases

### Phase 1: Tab Restructure + Summary Widgets (PR1 + PR2)

1. Rename tabs in `activity-page.tsx`: Summary | Board | Calendar | Memory | Projects
2. Change polling from 15s → 60s across all activity hooks
3. Update deep-link support for new tab names
4. Build 4 new Summary widgets: status donut, priority breakdown, types of work, team workload
5. Add `GET /api/activity/board/stats` endpoint
6. Wire "View All" click-throughs: Schedule widget → Calendar, Recent Activity → Board

### Phase 2: Board — Backend (PR3)

7. Database migration: add `priority`, `tags`, `assignee_agent_id`, `review_mode`, `due_date` to `recipe_executions`
8. Add `inbox`, `assigned`, `review` as valid status values
9. Create `PATCH /api/activity/tasks/{id}/status` endpoint
10. Create `PATCH /api/activity/tasks/{id}` endpoint (update fields)
11. Extend `GET /api/activity/feed` with board status filters
12. Implement SSE endpoint `GET /api/activity/board/stream` using Redis pub/sub

### Phase 3: Board — Frontend (PR4)

13. Install `@hello-pangea/dnd`
14. Build `board-view.tsx` — DragDropContext + 5 columns
15. Build `board-column.tsx` — Droppable column with header + count
16. Build `board-card.tsx` — Draggable card with task details
17. Build `board-agent-sidebar.tsx` — agent roster with filter-on-click
18. Build `board-filters.tsx` — filter bar
19. Implement optimistic drag-and-drop with revert on failure
20. Build `use-board-sse.ts` hook with reconnect + 60s polling fallback

### Phase 4: Task Viewer (PR5)

21. Build `board-task-viewer.tsx` — slide-over with 4 context-aware views
22. Inbox/Assigned view: editable fields (assignee, priority, due date, tags, review mode)
23. In Progress view: live logs + step pipeline (reuse `execution-detail.tsx` internals)
24. Review view: report content + grade form + approve/reject (reuse `report-viewer.tsx` + `report-grade-form.tsx`)
25. Done view: results + execution summary + re-run/download actions
26. Wire task viewer to board card clicks

### Phase 5: Calendar (PR6)

27. Build `activity-calendar.tsx` — container with view toggle
28. Build `calendar-always-running.tsx` — sub-hourly routines
29. Build `calendar-week-grid.tsx` — CSS grid with cron-parsed task pills
30. Build `calendar-next-up.tsx` — upcoming sorted list
31. Wire calendar events to task viewer slide-over
32. Wire Schedule widget "View All" → Calendar tab

### Phase 6: Memory Enhancement (PR7)

33. Verify Mem0 search POST fix is merged
34. Rewrite `activity-memory.tsx` → two-panel layout
35. Build `memory-sidebar.tsx` — search + grouped journal list
36. Build `memory-viewer.tsx` — right panel markdown renderer
37. Add `group_by` param to `GET /api/memory/browse`
38. Create `GET /api/memory/journal/{date}` endpoint
39. Add organize-by toggle (Day / Agent / Type)
40. Enhanced filters: agent, date range, type, sort

### Phase 7: Projects (PR8)

41. Database migration: `projects` + `project_tasks` tables
42. Create `projects.py` API with CRUD endpoints
43. Build `activity-projects.tsx` — card grid with stats header
44. Build `project-card.tsx` — card with progress bar + agent + priority
45. Build `project-detail.tsx` — detail view with linked task list
46. Wire "Open in Board" → Board tab filtered by project

### Phase 8: Global Search (PR9)

47. Create `GET /api/search` endpoint — cross-entity search
48. Build `global-search.tsx` — Cmd+K overlay with keyboard navigation
49. Build `search-result-item.tsx` — categorized result rows
50. Wire results to navigation (Board task viewer, Memory entry, Agent page, Project detail)
51. Add Cmd+K listener to app layout

### Phase 9: Cleanup (PR10)

52. Delete deprecated components: `activity-feed.tsx`, `activity-feed-item.tsx`, `activity-reports.tsx`, `report-card.tsx`
53. Update all internal links referencing old Feed/Reports tabs
54. Mobile responsive pass for all new components
55. Loading skeletons for Board, Calendar, Memory, Projects
56. Empty states for each tab
57. `prefers-reduced-motion` compliance

---

## 13. Dependencies

### New npm Packages

| Package | Purpose | Size |
|---|---|---|
| `@hello-pangea/dnd` | Drag-and-drop for Kanban board | ~45KB gzipped |

### Existing Packages (Already Installed)

| Package | Used For |
|---|---|
| `react-grid-layout` | Summary widget drag grid |
| `recharts` | Donut, bar charts on Summary |
| `framer-motion` | Animations |
| `lucide-react` | Icons |

No `@fullcalendar` needed — Calendar is custom CSS grid.

---

## 14. Success Metrics

| Metric | Target | How to Measure |
|---|---|---|
| Time to answer "what's running?" | < 3 seconds (one click from sidebar) | User testing |
| Task status visibility | 100% of active tasks visible on Board | Compare DB count vs UI count |
| Drag-and-drop latency | < 200ms perceived (optimistic update) | Performance testing |
| SSE delivery latency | < 2s from event to UI update | Instrumentation |
| Calendar shows all schedules | 100% of cron jobs visible | Compare scheduler DB vs Calendar |
| Memory search returns results | > 0 results for any known topic | Regression test |
| Global search response time | < 500ms | API monitoring |
| Page load time | < 1.5s initial, < 500ms tab switch | Lighthouse + RUM |
| Mobile usability | All tabs functional on 375px width | Manual test |

---

## 15. Open Questions

1. **SSE vs WebSocket:** SSE is simpler and sufficient for server→client push. If we later need client→server streaming (e.g., collaborative editing), upgrade to WebSocket. For now, SSE.

2. **Board pagination:** If a column has 100+ cards, do we paginate or virtual-scroll? Recommendation: show latest 50 per column with "Load More" button. Virtual scrolling is a future optimization.

3. **Project ↔ Mission relationship:** Projects in this PRD are the UI evolution of Missions. The backend `projects` table is new. Do we migrate existing mission data or start fresh? Recommendation: start fresh — Missions tab was a placeholder with no data.

4. **Review mode "LLM":** Which agent performs LLM review? Options: (a) dedicated QA agent, (b) any agent with "reviewer" skill, (c) configurable per-task. Recommendation: configurable per-task with a workspace-level default.

5. **Calendar — month view?** v2 ships with Week and Today views only. Month view is a future enhancement if users request it. Keeps scope manageable.
