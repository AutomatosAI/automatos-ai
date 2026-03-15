# Mission Control UX Specification

**PRD:** 82A Sequential Mission Coordinator
**Version:** 1.1 (aligned to backend API + board bridge)
**Date:** 2026-03-15

---

## Design Principles

1. **Chat-native** — Missions start in conversation, not forms
2. **Progressive disclosure** — Simple overview first, details on demand
3. **Real-time confidence** — Users always know what's happening and what needs their attention
4. **Existing patterns** — Reuse PageHeader, StatsBar, FilterTabs, glass-card everywhere
5. **Dark + orange + glass** — Match the existing Automatos design system exactly

---

## Design Tokens (Additive)

Mission-specific semantic colors layered on existing design system:

```css
/* Mission state colors — mapped to RunState enum values */
--mission-pending:           hsl(var(--muted-foreground));      /* gray-400 */
--mission-planning:          hsl(var(--info));                  /* blue-500 */
--mission-awaiting-approval: hsl(var(--warning));               /* amber-500 */
--mission-running:           hsl(var(--primary));               /* orange-500 */
--mission-paused:            hsl(217 91% 60% / 0.5);           /* blue-500/50 */
--mission-verifying:         hsl(var(--info));                  /* blue-500 */
--mission-awaiting-human:    hsl(var(--warning));               /* amber-500 */
--mission-completed:         hsl(var(--success));               /* green-500 */
--mission-failed:            hsl(var(--destructive));           /* red-500 */
--mission-cancelled:         hsl(var(--muted-foreground) / 0.5);

/* Task DAG node states */
--task-done:          hsl(var(--success));               /* green-500 */
--task-active:        hsl(var(--primary));               /* orange-500 — pulsing */
--task-upcoming:      hsl(var(--muted-foreground));      /* gray-500 */
--task-failed:        hsl(var(--destructive));           /* red-500 */
--task-review:        hsl(var(--warning));               /* amber-500 */
--task-skipped:       hsl(var(--muted-foreground) / 0.4);
```

---

## Screen 1: Chat with Mission Mode

### Entry Point

**Location:** Bottom toolbar of `MultimodalInput` (left side, after mic button)

Add a `MissionToggle` button alongside existing controls:

```
┌─────────────────────────────────────────────────────────────┐
│  Type a message...                                          │
│                                                             │
│  [Clip] [Mic] [Call] [Agent▾]      [tools...]  [▶ Send]    │
│                            ↑                                │
│                     Add here: [🎯 Mission] toggle           │
└─────────────────────────────────────────────────────────────┘
```

**Component:** `MissionModeToggle`
- **Idle state:** Ghost button with `Target` icon + "Mission" label
- **Active state:** Filled orange pill `bg-primary/20 border-primary/40 text-primary`
- **Behavior:** Click toggles `isMissionMode` in chat state. Also activatable via `/mission` command typed in input.
- **When active:** Orange glow ring appears on the entire input card border (same as recording state pattern but orange)

### Mission Mode Active — Conversation Flow

When mission mode is ON, the chat becomes a structured conversation:

```
┌──────────────────────────────────────────────────────┐
│  🎯 Mission Mode                              [✕]    │  ← Dismissible banner
├──────────────────────────────────────────────────────┤
│                                                      │
│  USER: "Research the top 5 competitors to Notion     │
│  and create a comparison report with pricing,        │
│  features, and market positioning"                   │
│                                                      │
│  PLANNER (Claude): I have a few questions before     │
│  creating your mission plan:                         │
│                                                      │
│  1. What's the target audience for this report?      │
│     (investors, product team, marketing?)            │
│  2. Should I include free/open-source alternatives?  │
│  3. How deep should the pricing analysis go?         │
│     (just tiers, or detailed per-seat calculations?) │
│                                                      │
│  USER: Product team. Yes include OSS. Just tiers     │
│  and key differentiators.                            │
│                                                      │
│  PLANNER: Got it. Here's the plan:                   │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │  📋 Mission Plan: Competitor Analysis      │      │
│  │  5 tasks · 3 agents · ~2 hours est.        │      │
│  │                                            │      │
│  │  [View Full Plan]  [Approve]  [Reject]     │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
MultimodalInput (MODIFIED)
├── MissionModeToggle              NEW — toggle button in toolbar
│
ChatMessages (MODIFIED — new message types)
├── MissionBanner                  NEW — top banner when mode active
├── PlannerMessage                 NEW — renders planner questions
│   └── (standard message bubble styling)
├── MissionPlanCard                NEW — inline plan preview card
│   ├── plan title + stats
│   ├── [View Full Plan] → opens PlanReviewModal
│   ├── [Approve] → POST /api/missions/{id}/approve
│   └── [Reject] → opens feedback textarea, POST /api/missions/{id}/reject
```

### Interaction Details

| Action | Behavior |
|--------|----------|
| Toggle Mission ON | Sets `isMissionMode=true`, sends system context to planner |
| User types goal | Normal message, but backend routes to planner agent |
| Planner responds | Renders as assistant message with planner avatar |
| Plan generated | `MissionPlanCard` appears inline in chat |
| "View Full Plan" | Opens `PlanReviewModal` (Screen 2) as fullscreen overlay |
| "Approve" | POST approve, exits mission mode, shows toast "Mission started" |
| "Reject" | Expands inline textarea for feedback, POST reject, planner iterates |
| `/mission` typed | Same as clicking toggle ON |
| `✕` on banner | Exits mission mode, normal chat resumes |

### State Management

```ts
// Add to chat Zustand store or local state in chat component
interface MissionChatState {
  isMissionMode: boolean
  activeMissionId: string | null      // Set after planner creates mission
  planStatus: 'idle' | 'planning' | 'reviewing' | 'approved' | 'rejected'
}
```

---

## Screen 2: Plan Review (ReactFlow DAG)

### Layout

Fullscreen modal or dedicated route (`/missions/[id]/plan`). Split view:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Mission: Competitor Analysis Report          [Approve] [Reject ▾] │
│  5 tasks · 3 agents · ~2h estimated                                │
├───────────────────────────────────────┬─────────────────────────────┤
│                                       │                             │
│         ReactFlow DAG Canvas          │    Task Detail Panel        │
│                                       │                             │
│    [Research] ──→ [Analyze] ──→       │    📋 Task: Research        │
│                                       │    Agent: Researcher        │
│         [Compare] ──→ [Report]        │    Est: 30 min              │
│                                       │                             │
│              ──→ [Review]             │    Description:             │
│                                       │    "Search for top 5..."    │
│                                       │                             │
│                                       │    [Edit Description]       │
│                                       │    [Reassign Agent ▾]       │
│                                       │                             │
├───────────────────────────────────────┴─────────────────────────────┤
│  Legend: ● Queued  ○ Sequential dependency  👤 Agent avatar         │
└─────────────────────────────────────────────────────────────────────┘
```

### ReactFlow DAG Node — `MissionTaskNode`

Custom node component for the DAG:

```
┌─────────────────────────────┐
│  👤  Research Competitors    │  ← Agent avatar + task title
│  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈  │
│  🤖 Researcher Agent         │  ← Assigned agent name
│  ⏱ ~30 min                   │  ← Time estimate (if available)
└─────────────────────────────┘
```

**Visual states for plan review:**
- All nodes start as `queued` (gray border, `border-muted-foreground/30`)
- Selected node: `ring-2 ring-primary` (orange highlight)
- Hover: `border-muted-foreground/60` transition

**Node sizing:** `min-w-[220px]` — enough for title + agent name

### Component Hierarchy

```
PlanReviewModal | PlanReviewPage
├── header
│   ├── mission title + stats (tasks, agents, time)
│   ├── ApproveButton                → POST /api/missions/{id}/approve
│   └── RejectButton                 → expands RejectFeedbackForm
│       └── RejectFeedbackForm       → textarea + submit
│
├── ResizablePanelGroup (horizontal, react-resizable-panels)
│   ├── ResizablePanel (DAG — default 65%)
│   │   └── ReactFlow
│   │       ├── MissionTaskNode[]     CUSTOM node type
│   │       ├── Background (dots)
│   │       ├── Controls
│   │       └── MiniMap
│   │
│   ├── ResizableHandle
│   │
│   └── ResizablePanel (Detail — default 35%)
│       └── TaskDetailPanel
│           ├── task title + description (editable)
│           ├── agent assignment dropdown (AgentSelector reuse)
│           ├── verification criteria (read-only chips)
│           └── dependencies list
│
└── Legend bar (bottom)
```

### Approve with Modifications

When user edits tasks before approving, modifications use the backend's allowed keys:

```ts
// Tracked in local state, sent with approve request
// Backend validates keys against ALLOWED_MODIFICATION_KEYS:
//   { "task_overrides", "notes", "agent_overrides" }
// Total payload capped at 10KB

interface PlanModifications {
  task_overrides?: Record<string, {   // task_id → overrides
    description?: string
    agent_role?: string
  }>
  agent_overrides?: Record<string, number>  // task_id → agent_id
  notes?: string
}

// POST /api/missions/{id}/approve
{ modifications: PlanModifications }
```

### ReactFlow Configuration

```ts
const nodeTypes = { missionTask: MissionTaskNode }

// Layout: Use dagre or elkjs for automatic DAG layout
// Direction: TB (top-to-bottom) for sequential clarity
// Edge type: 'smoothstep' with MarkerType.ArrowClosed
// Edge color: hsl(var(--muted-foreground) / 0.3)
// Animated edges: false during plan review (no execution yet)
```

---

## Screen 3: Mission List

### Location

Activity page → existing "Projects" tab (rename to "Missions")

**Tab change in `activity-page.tsx`:**
```ts
// Before:
{ value: 'projects', label: 'Projects', icon: FolderKanban }

// After:
{ value: 'missions', label: 'Missions', icon: Target, count: activeMissionCount }
```

### Layout

Replace `ActivityMissions` content with real mission data:

```
┌─────────────────────────────────────────────────────────────────┐
│  Command Centre                                    [Refresh]    │
│  Your AI workforce at a glance                                  │
├─────────────────────────────────────────────────────────────────┤
│  [Summary] [Board] [Calendar] [Memory] [Missions (3)]          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌── Filters ──────────────────────────────────────┐            │
│  │ [All] [Active] [Needs Review] [Completed] [⌕]  │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
│  ┌─ MissionCard ────────────┐ ┌─ MissionCard ────────────┐     │
│  │ ● Executing              │ │ ⏸ Awaiting Review        │     │
│  │                          │ │                          │     │
│  │ Competitor Analysis      │ │ Blog Content Pipeline    │     │
│  │ Report                   │ │                          │     │
│  │                          │ │ 4/6 tasks done           │     │
│  │ 3/5 tasks done           │ │ ████████░░ 67%           │     │
│  │ ██████░░░░ 60%           │ │                          │     │
│  │                          │ │ 👤👤👤 3 agents           │     │
│  │ 👤👤👤 3 agents           │ │ ⏱ 1h 20m elapsed        │     │
│  │ ⏱ 45m elapsed            │ │                          │     │
│  │                          │ │ [Review Now]             │     │
│  │ [View Details →]         │ │                          │     │
│  └──────────────────────────┘ └──────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MissionCard Component

```
┌────────────────────────────────────────┐
│  ● Executing                     ⋯     │  ← Status badge + overflow menu
│                                        │
│  Competitor Analysis Report            │  ← Title (truncate 2 lines)
│  Research top 5 competitors...         │  ← Description (truncate 1 line)
│                                        │
│  ████████░░░░ 60%  (3/5 tasks)         │  ← Progress bar + fraction
│                                        │
│  👤 👤 👤  ·  ⏱ 45m  ·  💰 $0.12      │  ← Agent avatars · elapsed · cost
│                                        │
│  [View Details →]                      │  ← CTA (or [Review Now] if awaiting)
└────────────────────────────────────────┘
```

**Status badge colors (RunState enum values):**
| RunState | Badge Label | Color |
|----------|-------------|-------|
| pending | Pending | `bg-muted text-muted-foreground` |
| planning | Planning | `bg-info/10 text-info border-info/20` |
| awaiting_approval | Needs Approval | `bg-warning/10 text-warning border-warning/20` |
| running | Running | `bg-primary/10 text-primary border-primary/20` (pulse) |
| paused | Paused | `bg-muted text-muted-foreground` |
| verifying | Verifying | `bg-info/10 text-info border-info/20` (pulse) |
| awaiting_human | Needs Review | `bg-warning/10 text-warning border-warning/20` (pulse) |
| completed | Completed | `bg-success/10 text-success border-success/20` |
| failed | Failed | `bg-destructive/10 text-destructive border-destructive/20` |
| cancelled | Cancelled | `bg-muted text-muted-foreground line-through` |

### Component Hierarchy

```
ActivityMissions (REWRITE)
├── filter bar
│   ├── status pills (All | Active | Needs Review | Completed)
│   └── SearchInput
│
├── grid (md:grid-cols-2 xl:grid-cols-3)
│   └── MissionCard[]                 NEW
│       ├── StatusBadge (reuse existing)
│       ├── title + description
│       ├── Progress bar
│       ├── agent AvatarGroup
│       ├── meta row (elapsed, cost)
│       └── CTA button
│           ├── "View Details" → router.push(`/missions/${id}`)
│           └── "Review Now" → router.push(`/missions/${id}?tab=review`)
│
└── empty state (reuse existing pattern with Target icon)
```

### Data Hook

```ts
// hooks/use-missions-api.ts
export const missionQueryKeys = {
  all: ['missions'] as const,
  list: (filters: MissionFilters) => ['missions', 'list', filters] as const,
  detail: (id: string) => ['missions', id] as const,
  events: (id: string) => ['missions', id, 'events'] as const,
}

export function useMissions(filters: MissionFilters = {}) {
  return useQuery({
    queryKey: missionQueryKeys.list(filters),
    queryFn: () => apiClient.request<MissionListResponse>('/api/missions', { params: filters }),
    staleTime: 15_000,
    refetchInterval: 30_000,  // Poll every 30s
  })
}

export function useMission(id: string) {
  return useQuery({
    queryKey: missionQueryKeys.detail(id),
    queryFn: () => apiClient.request<MissionDetail>(`/api/missions/${id}`),
    staleTime: 5_000,
    refetchInterval: 10_000,  // Faster polling for active missions
  })
}

export function useMissionEvents(id: string) {
  return useQuery({
    queryKey: missionQueryKeys.events(id),
    queryFn: () => apiClient.request<MissionEvent[]>(`/api/missions/${id}/events`),
    staleTime: 5_000,
    refetchInterval: 10_000,
  })
}

export function useApproveMission() {
  return useMutation({
    mutationFn: ({ id, modifications }: { id: string; modifications?: PlanModification[] }) =>
      apiClient.request(`/api/missions/${id}/approve`, { method: 'POST', body: { modifications } }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: missionQueryKeys.all }),
  })
}

export function useRejectMission() {
  return useMutation({
    mutationFn: ({ id, feedback }: { id: string; feedback: string }) =>
      apiClient.request(`/api/missions/${id}/reject`, { method: 'POST', body: { feedback } }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: missionQueryKeys.all }),
  })
}

// Similar mutations for: pause, resume, cancel, review
```

---

## Screen 4: Mission Detail

### Route

`/app/missions/[id]/page.tsx`

### Layout

Three-zone layout using `react-resizable-panels`:

```
┌─────────────────────────────────────────────────────────────────────┐
│  ← Back to Missions                                                │
│                                                                    │
│  Competitor Analysis Report                                        │
│  ● Executing · 3/5 tasks · 45m elapsed · $0.12 spent              │
│                                                                    │
│  [Pause] [Cancel]                               [Review] (if due)  │
├─────────────────────────────────┬───────────────────────────────────┤
│  ┌─ Stats ────────────────────┐ │                                   │
│  │ [3 Done] [1 Active]       │ │                                   │
│  │ [1 Queued] [0 Failed]     │ │                                   │
│  └────────────────────────────┘ │                                   │
│                                 │                                   │
│      Live DAG Canvas            │       Activity Feed               │
│                                 │                                   │
│   ✅ Research ──→ ✅ Analyze    │  10:24  Task "Analyze" completed  │
│                        │        │         Score: 0.92               │
│             🟠 Compare ←─┘      │  10:23  Agent Analyst started     │
│                  │              │         "Analyze Pricing"         │
│             ⬜ Report            │  10:15  Task "Research"           │
│                  │              │         completed (verified)      │
│             ⬜ Review            │  10:01  Mission started           │
│                                 │         3 agents assigned         │
│                                 │                                   │
│  Legend: ✅ Done  🟠 Active     │  ─── Load More ───                │
│          ⬜ Queued ❌ Failed     │                                   │
├─────────────────────────────────┴───────────────────────────────────┤
│  War Room (v2 — currently read-only feed)                    [▴]   │
│  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈  │
│  [Researcher] Found 12 sources for competitive analysis            │
│  [Analyst] Starting pricing comparison matrix                      │
│  [System] Task dependency resolved: Compare unblocked              │
└─────────────────────────────────────────────────────────────────────┘
```

### DAG Node States (Execution View)

`MissionTaskNode` adapts its visuals based on task state:

| State | Border | Background | Icon | Animation |
|-------|--------|------------|------|-----------|
| QUEUED | `border-muted-foreground/20` | `bg-muted/5` | `Circle` (empty) | none |
| ASSIGNED | `border-info/40` | `bg-info/5` | `UserCheck` | none |
| RUNNING | `border-primary/60` | `bg-primary/5` | `Loader2` | `animate-spin` on icon, `animate-pulse` on border glow |
| COMPLETED | `border-success/40` | `bg-success/5` | `CheckCircle2` | none |
| VERIFIED | `border-success/60` | `bg-success/10` | `ShieldCheck` | none |
| FAILED | `border-destructive/40` | `bg-destructive/5` | `XCircle` | none |
| RETRYING | `border-warning/40` | `bg-warning/5` | `RefreshCw` | `animate-spin` |
| SKIPPED | `border-muted/20` | `bg-muted/5` | `SkipForward` | opacity-50 |
| AWAITING_HUMAN | `border-warning/60` | `bg-warning/10` | `Eye` | `animate-pulse` |

**Edge animation:** Edges TO the active (RUNNING) node are `animated: true` with `stroke: hsl(var(--primary))`

### Activity Feed Component

```
ActivityFeedItem
├── timestamp (relative, e.g. "2m ago")
├── event icon (color-coded by type)
├── event description
│   ├── Task events: "Task 'Research' completed · Score: 0.92"
│   ├── Agent events: "Agent Researcher assigned to 'Research'"
│   ├── System events: "Verification passed · Model: gemini-2.5-pro"
│   └── Human events: "Mission approved by user"
└── optional detail (expandable)
```

**Event types → icons:**
| Type | Icon | Color |
|------|------|-------|
| task_started | `Play` | `text-info` |
| task_completed | `CheckCircle2` | `text-success` |
| task_failed | `XCircle` | `text-destructive` |
| task_verified | `ShieldCheck` | `text-success` |
| agent_assigned | `UserPlus` | `text-agent` |
| verification_started | `Search` | `text-info` |
| human_review_needed | `Eye` | `text-warning` |
| mission_paused | `Pause` | `text-muted-foreground` |
| mission_resumed | `Play` | `text-primary` |
| mission_completed | `Trophy` | `text-success` |

### War Room Panel (v1 — Read-Only)

Collapsible bottom panel using `react-resizable-panels` (vertical group):

- **v1:** Read-only feed of agent messages and system events
- Collapsed by default (just header bar visible), expandable to ~200px
- Each entry: `[AgentAvatar] [AgentName] message text [timestamp]`
- Auto-scrolls to bottom on new entries
- **v2 marker:** Input field placeholder "Type to agents... (coming in v2)" — disabled

### Component Hierarchy

```
MissionDetailPage
├── header
│   ├── Back link → /activity?tab=missions
│   ├── PageHeader (title=mission.title, subtitle=status string)
│   └── action buttons: [Pause] [Resume] [Cancel] [Review]
│
├── StatsBar (4 cards)
│   ├── Tasks Done (count / total)
│   ├── Active Now (running task count)
│   ├── Time Elapsed (formatted duration)
│   └── Cost ($X.XX)
│
├── ResizablePanelGroup (vertical)
│   ├── ResizablePanelGroup (horizontal)  ← main content
│   │   ├── ResizablePanel (DAG — 60%)
│   │   │   └── MissionDAGCanvas
│   │   │       ├── ReactFlow
│   │   │       │   ├── MissionTaskNode[] (status-aware)
│   │   │       │   ├── Background
│   │   │       │   ├── Controls
│   │   │       │   └── MiniMap
│   │   │       └── DAGLegend
│   │   │
│   │   ├── ResizableHandle
│   │   │
│   │   └── ResizablePanel (Feed — 40%)
│   │       └── MissionActivityFeed
│   │           ├── ActivityFeedItem[]
│   │           └── "Load more" button
│   │
│   ├── ResizableHandle
│   │
│   └── ResizablePanel (War Room — collapsed default)
│       └── WarRoomPanel
│           ├── header: "War Room" + collapse toggle
│           ├── message feed (read-only)
│           └── disabled input (v2 placeholder)
```

### Click-to-Inspect on DAG Node

When user clicks a node in the execution DAG, show a slide-over or popover:

```
┌─ Task Inspector ───────────────────────┐
│  Research Competitors              [✕]  │
│  Status: ✅ Verified                    │
│  Agent: 🤖 Researcher                  │
│                                        │
│  Description:                          │
│  Search for top 5 Notion competitors   │
│  including pricing, features...        │
│                                        │
│  Output:                               │
│  ┌──────────────────────────────────┐  │
│  │ Found 8 competitors:            │  │
│  │ 1. Coda — $10/user/mo          │  │
│  │ 2. Craft — $5/user/mo          │  │
│  │ ...                             │  │
│  └──────────────────────────────────┘  │
│                                        │
│  Verification:                         │
│  Score: 0.92 · Model: gemini-2.5-pro   │
│  Relevance: 0.95 · Completeness: 0.88  │
│                                        │
│  Tokens: 4,230 · Cost: $0.03          │
│  Duration: 3m 12s                      │
└────────────────────────────────────────┘
```

---

## Screen 5: Human Review Mode

### Trigger

When mission reaches `awaiting_human` state:
1. **MissionCard** shows "Review Now" CTA (amber pulse)
2. **MissionDetailPage** shows "Review" button in header
3. **Optional:** Toast notification "Mission X needs your review"

### Layout

Reuses MissionDetailPage but overlays a review panel:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Competitor Analysis Report — Human Review                         │
│  5/5 tasks verified · Review each task to complete mission         │
├───────────────────────────────────┬─────────────────────────────────┤
│                                   │                                 │
│      DAG Canvas                   │   Review Panel                  │
│   (all nodes in review state)     │                                 │
│                                   │   Task 1: Research        [✅]  │
│   ✅ Research ──→ ✅ Analyze      │   Score: 0.95                   │
│                        │          │   "Found 8 competitors..."      │
│              ✅ Compare ←─┘        │   [Accept] [Reject ▾]          │
│                   │               │                                 │
│              ✅ Report             │   Task 2: Analyze         [✅]  │
│                   │               │   Score: 0.92                   │
│              ✅ Review             │   "Pricing matrix created..."   │
│                                   │   [Accept] [Reject ▾]          │
│                                   │                                 │
│   (Selected node highlighted)     │   Task 3: Compare         [⏳]  │
│                                   │   Score: 0.88                   │
│                                   │   "Feature comparison..."       │
│                                   │   [Accept] [Reject ▾]          │
│                                   │                                 │
│                                   │   ─────────────────────         │
│                                   │   3/5 reviewed                  │
│                                   │   [Submit Review]               │
├───────────────────────────────────┴─────────────────────────────────┤
│  ⚠ Rejecting a task sends it back for retry with your feedback     │
└─────────────────────────────────────────────────────────────────────┘
```

### Per-Task Review Item

```
┌────────────────────────────────────────────────────┐
│  Task 1: Research Competitors              [✅ 0.95]│
│  Agent: Researcher                                 │
│                                                    │
│  Output preview (3 lines, expandable):             │
│  "Found 8 competitors: Coda ($10/user), Craft..."  │
│  [Show full output ▾]                              │
│                                                    │
│  Verification: relevance 0.95, completeness 0.92,  │
│  accuracy 0.96, format 0.98                        │
│                                                    │
│  ┌─────────┐  ┌──────────┐                         │
│  │ Accept  │  │ Reject ▾ │                         │
│  └─────────┘  └──────────┘                         │
│                                                    │
│  (If Reject clicked:)                              │
│  ┌────────────────────────────────────────────┐    │
│  │ What needs to change?                      │    │
│  │ ___________________________________________│    │
│  │                                            │    │
│  │                        [Submit Feedback]   │    │
│  └────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────┘
```

### Component Hierarchy

**IMPORTANT:** Backend review endpoint uses a SINGLE verdict (accept/reject) for the
entire mission, with optional per-task feedback strings for rejected tasks. The UI
lets users annotate individual tasks with feedback, then submit one decision.

```
HumanReviewPanel
├── header: "Review Mission" + task count
│
├── ScrollArea
│   └── TaskReviewItem[]          (one per verified task)
│       ├── task title + verification score badge
│       ├── agent name
│       ├── output preview (Collapsible, 3 lines default)
│       ├── verification scores row (4 dimensions)
│       └── FeedbackToggle
│           ├── "Flag for revision" checkbox
│           └── RejectFeedbackForm (shown when flagged)
│               └── Textarea (placeholder="What needs to change?")
│
├── Separator
│
├── summary: "2 tasks flagged for revision"
│
├── Mission Verdict Buttons (mutually exclusive)
│   ├── AcceptButton → POST /api/missions/{id}/review { verdict: "accept" }
│   └── RejectButton → POST /api/missions/{id}/review
│       body: {
│         verdict: "reject",
│         task_feedback: { [task_id]: "feedback string", ... }
│       }
│   Note: Reject is disabled unless at least 1 task is flagged with feedback
│
└── Info banner: "Rejecting sends flagged tasks back for retry"
```

### Review State Machine (Local)

```ts
interface ReviewState {
  // Per-task feedback (only for tasks user flags)
  taskFeedback: Record<string, string>  // task_id → feedback text
}

// User flags tasks they want revised, adds feedback text
// Then clicks Accept (all good) or Reject (sends flagged tasks for retry)
// Accept: POST { verdict: "accept" }
// Reject: POST { verdict: "reject", task_feedback: reviewState.taskFeedback }
// Reject button disabled if taskFeedback is empty
```

### Backend API Contract (existing)

```ts
// POST /api/missions/{id}/review
interface MissionReviewRequest {
  verdict: 'accept' | 'reject'           // Single verdict for whole mission
  task_feedback?: Record<string, string>  // task_id → feedback (max 50 entries, 2000 char each)
}
// On reject: tasks with feedback get re-queued for retry
// On accept: mission transitions to completed
```

---

## Shared Components

### MissionTaskNode (ReactFlow custom node)

Used in both Plan Review (Screen 2) and Mission Detail (Screen 4):

```tsx
interface MissionTaskNodeData {
  id: string
  title: string
  agent: {
    id: number
    name: string
    avatar?: string
  } | null
  status: TaskState          // from orchestration_enums
  verificationScore?: number
  isSelected: boolean
  mode: 'plan' | 'execution' | 'review'
}
```

**Rendering logic:**
- `mode=plan`: All nodes gray, selected has orange ring
- `mode=execution`: Nodes colored by status, active pulses
- `mode=review`: Nodes show verification score badge, accept/reject state

### MissionDAGCanvas

Wrapper around ReactFlow with mission-specific config:

```tsx
interface MissionDAGCanvasProps {
  tasks: MissionTask[]
  dependencies: TaskDependency[]
  mode: 'plan' | 'execution' | 'review'
  selectedTaskId?: string
  onTaskSelect: (taskId: string) => void
}
```

**Auto-layout:** Use `dagre` library for automatic top-to-bottom layout:
```ts
import dagre from 'dagre'

function getLayoutedElements(nodes, edges) {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 80 })

  nodes.forEach(node => g.setNode(node.id, { width: 220, height: 80 }))
  edges.forEach(edge => g.setEdge(edge.source, edge.target))

  dagre.layout(g)

  return {
    nodes: nodes.map(node => {
      const pos = g.node(node.id)
      return { ...node, position: { x: pos.x - 110, y: pos.y - 40 } }
    }),
    edges,
  }
}
```

### StatusBadge (Mission variant)

Reuse existing `StatusBadge` component but add mission-specific status mappings.

### AgentAvatarGroup

Small row of overlapping agent avatars (max 4 visible + "+N"):

```tsx
function AgentAvatarGroup({ agents, max = 4 }: { agents: Agent[]; max?: number }) {
  const visible = agents.slice(0, max)
  const overflow = agents.length - max

  return (
    <div className="flex -space-x-2">
      {visible.map(agent => (
        <Avatar key={agent.id} className="w-6 h-6 border-2 border-background">
          <AvatarFallback className="text-[10px] bg-agent/20 text-agent">
            {agent.name[0]}
          </AvatarFallback>
        </Avatar>
      ))}
      {overflow > 0 && (
        <div className="w-6 h-6 rounded-full bg-muted border-2 border-background flex items-center justify-center text-[10px] text-muted-foreground">
          +{overflow}
        </div>
      )}
    </div>
  )
}
```

---

## State Management

### Zustand Store: `useMissionStore`

```ts
interface MissionStore {
  // Chat integration
  isMissionMode: boolean
  setMissionMode: (on: boolean) => void
  activePlanningMissionId: string | null
  setActivePlanningMissionId: (id: string | null) => void

  // Plan review
  selectedTaskId: string | null
  setSelectedTaskId: (id: string | null) => void
  planModifications: PlanModification[]
  addModification: (mod: PlanModification) => void
  clearModifications: () => void

  // Human review
  reviewDecisions: Record<string, { verdict: 'accepted' | 'rejected'; feedback?: string }>
  setReviewDecision: (taskId: string, verdict: 'accepted' | 'rejected', feedback?: string) => void
  clearReviewDecisions: () => void
}
```

### React Query Keys

All mission data fetching through React Query v4 hooks in `hooks/use-missions-api.ts` (defined above in Screen 3).

### WebSocket / Polling Strategy

**v1 (polling):**
- Mission list: 30s interval
- Active mission detail: 10s interval
- Activity feed: 10s interval
- Increase to 5s when mission is RUNNING

**v2 (WebSocket):**
- Real-time events via existing WebSocket infrastructure
- Push task state changes, verification results, agent messages

---

## Navigation & Routing

```
/activity?tab=missions              → Mission List (Screen 3)
/missions/[id]                      → Mission Detail (Screen 4)
/missions/[id]?tab=review           → Mission Detail with Review panel open (Screen 5)
/missions/[id]/plan                 → Plan Review fullscreen (Screen 2)

Chat (existing /chat/[id])          → Mission mode toggle (Screen 1)
```

### Sidebar Navigation

Add "Missions" as a sub-item under Activity in the sidebar, or as a standalone item:

```
Activity (existing)
├── Summary
├── Board
├── Calendar
├── Memory
└── Missions  ← count badge when missions need attention
```

---

## Responsive Behavior

### Mobile (< 768px)

- **Mission List:** Single column cards, full width
- **Mission Detail:** Stacked layout — DAG on top (scrollable, 300px min-height), feed below
- **Plan Review:** Full-screen DAG with bottom sheet for task detail
- **Human Review:** Full-screen scrollable list, no split view
- **Chat Mission Mode:** Same as desktop (chat is already mobile-optimized)

### Tablet (768px - 1024px)

- **Mission List:** 2-column grid
- **Mission Detail:** Side-by-side but narrower panels
- **Plan Review:** Side-by-side with collapsible detail panel

---

## Accessibility

- All interactive nodes in DAG have `aria-label` with task name and status
- Tab navigation between DAG nodes (ReactFlow supports this)
- Screen reader summary of mission status at top of each screen
- Review buttons have clear labels: "Accept task: Research Competitors"
- Feedback textarea has `aria-describedby` linking to helper text
- Status colors always paired with icon + text (never color-only)
- `prefers-reduced-motion`: Disable pulse animations, use opacity changes instead

---

## File Structure (New Files)

```
frontend/
├── app/
│   └── missions/
│       └── [id]/
│           ├── page.tsx              Mission Detail page
│           └── plan/
│               └── page.tsx          Plan Review page (optional, can be modal)
│
├── components/
│   └── missions/
│       ├── mission-card.tsx          Mission list card
│       ├── mission-dag-canvas.tsx    ReactFlow wrapper with dagre layout
│       ├── mission-task-node.tsx     Custom ReactFlow node
│       ├── mission-activity-feed.tsx Activity event feed
│       ├── mission-stats-bar.tsx     Mission-specific StatsBar config
│       ├── mission-detail-page.tsx   Main detail page component
│       ├── plan-review-modal.tsx     Plan review overlay/modal
│       ├── human-review-panel.tsx    Review panel with per-task toggles
│       ├── task-review-item.tsx      Individual task review card
│       ├── task-inspector.tsx        Click-to-inspect popover
│       ├── war-room-panel.tsx        Read-only agent message feed
│       ├── mission-mode-toggle.tsx   Chat input toggle button
│       ├── mission-plan-card.tsx     Inline plan preview in chat
│       ├── agent-avatar-group.tsx    Overlapping avatar row
│       └── dag-legend.tsx            DAG color legend bar
│
├── hooks/
│   └── use-missions-api.ts          React Query hooks for missions
│
├── stores/
│   └── mission-store.ts             Zustand store
│
└── types/
    └── missions.ts                  TypeScript types
```

---

## TypeScript Types

Aligned 1:1 with backend Pydantic models in `orchestrator/api/missions.py`.

```ts
// types/missions.ts

// ── Enums (match orchestration_enums.py exactly) ──────────────

export type RunState =
  | 'pending'
  | 'planning'
  | 'awaiting_approval'
  | 'running'
  | 'paused'
  | 'verifying'
  | 'awaiting_human'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type TaskState =
  | 'pending'
  | 'queued'
  | 'assigned'
  | 'running'
  | 'completed'        // NOT terminal — awaiting verification
  | 'verifying'
  | 'verified'         // BLOCKED not terminal — human can reject
  | 'failed'
  | 'skipped'
  | 'stalled'
  | 'retrying'

export type StateType = 'initial' | 'active' | 'blocked' | 'terminal'

// ── API Response Types (match Pydantic models) ────────────────

// GET /api/missions → MissionListResponse.missions[]
// Also used in GET /api/missions/{id} base fields
export interface MissionResponse {
  id: string
  workspace_id: string
  goal: string
  state: RunState
  state_type: StateType
  plan: Record<string, unknown> | null
  config: Record<string, unknown> | null
  output_summary: Record<string, unknown> | null
  token_budget_estimate: number | null
  tokens_used: number
  max_retries: number
  created_by: string
  started_at: string | null
  completed_at: string | null
  created_at: string | null
  updated_at: string | null
}

// GET /api/missions/{id} (extends MissionResponse)
export interface MissionDetailResponse extends MissionResponse {
  tasks: TaskResponse[]
  recent_events: EventResponse[]
  // NOTE: dependencies not yet in API — needs backend addition (see below)
  dependencies?: TaskDependencyResponse[]
}

export interface TaskResponse {
  id: string
  title: string
  description: string | null
  task_type: string | null
  sequence_number: number
  agent_role: string | null
  state: TaskState
  state_type: StateType
  assigned_agent_id: number | null
  attempt_number: number
  tokens_used: number
  failure_reason_code: string | null
  failure_detail: string | null
  output_excerpt: string | null       // First 500 chars of output
  started_at: string | null
  completed_at: string | null
  created_at: string | null
}

export interface EventResponse {
  id: string
  event_type: string
  actor_type: string
  actor_id: string | null
  old_state: string | null
  new_state: string | null
  task_id: string | null
  payload: Record<string, unknown> | null  // NEEDS BACKEND: add to _event_to_response
  created_at: string | null
}

export interface TaskDependencyResponse {
  task_id: string
  depends_on_task_id: string
  trigger_rule: string
}

export interface MissionListResponse {
  missions: MissionResponse[]
  total: number
  limit: number
  offset: number
}

// ── Request Types ─────────────────────────────────────────────

export interface MissionCreateRequest {
  goal: string
  config?: Record<string, unknown>
}

export interface MissionApproveRequest {
  modifications?: {
    task_overrides?: Record<string, unknown>
    notes?: string
    agent_overrides?: Record<string, unknown>
  }
}

export interface MissionRejectRequest {
  reason: string
}

export interface MissionReviewRequest {
  verdict: 'accept' | 'reject'
  task_feedback?: Record<string, string>  // task_id → feedback
}

// ── Computed Frontend Types (derived from API data) ───────────

// Frontend computes these from MissionDetailResponse.tasks[]
export interface MissionStats {
  taskCount: number
  tasksDone: number       // state in ['verified', 'failed', 'skipped']
  tasksActive: number     // state in ['assigned', 'running', 'completed', 'verifying']
  tasksFailed: number     // state === 'failed'
  tokensUsed: number      // sum of task.tokens_used
  elapsedMs: number       // Date.now() - started_at
}

export function computeMissionStats(mission: MissionDetailResponse): MissionStats {
  const tasks = mission.tasks
  const doneStates: TaskState[] = ['verified', 'failed', 'skipped']
  const activeStates: TaskState[] = ['assigned', 'running', 'completed', 'verifying']

  return {
    taskCount: tasks.length,
    tasksDone: tasks.filter(t => doneStates.includes(t.state)).length,
    tasksActive: tasks.filter(t => activeStates.includes(t.state)).length,
    tasksFailed: tasks.filter(t => t.state === 'failed').length,
    tokensUsed: tasks.reduce((sum, t) => sum + t.tokens_used, 0),
    elapsedMs: mission.started_at
      ? Date.now() - new Date(mission.started_at).getTime()
      : 0,
  }
}
```

---

## Implementation Priority

| Phase | Screen | Effort | Dependencies |
|-------|--------|--------|--------------|
| 1 | Types + Hooks + Store | S | Backend API deployed |
| 2 | Mission List (Screen 3) | M | Phase 1 |
| 3 | Mission Detail (Screen 4) | L | Phase 1, ReactFlow node |
| 4 | Plan Review (Screen 2) | M | Phase 1, ReactFlow node |
| 5 | Human Review (Screen 5) | M | Phase 3 |
| 6 | Chat Integration (Screen 1) | M | Phase 1, backend planner |
| 7 | War Room v2 | L | WebSocket infra |

**Phase 1-5 = v1 launch. Phase 6 = fast-follow. Phase 7 = v2.**

Start with Screen 3 (Mission List) because it gives immediate visibility into existing missions and requires the least custom UI (reuses PageHeader, StatsBar, FilterTabs, card patterns). Screen 4 (Detail) follows naturally with the ReactFlow DAG component which is then reusable for Screen 2 (Plan Review).

---

## Board Bridge Integration

Mission tasks automatically appear on the existing kanban board via `orchestration_board_bridge.py`:

| Orchestration | Board Status | Kanban Column |
|---------------|-------------|---------------|
| pending → | backlog → | inbox |
| queued → | todo → | inbox |
| assigned/running → | in_progress → | in_progress |
| completed/verifying → | in_review → | review |
| verified → | done → | done |
| failed/stalled → | blocked → | review |
| skipped → | cancelled → | done |

**UX implications:**
- Mission tasks are visible on the Board tab (existing kanban) AND on the Mission Detail DAG
- Board cards for mission tasks show `source_type: 'orchestration_task'` tag
- Parent mission shows as `source_type: 'orchestration'` board task
- Users can see mission progress from either view — they complement each other
- No special handling needed: board bridge auto-syncs on every task state change

---

## Required Backend Changes

Small additions to `missions.py` to support the frontend:

### 1. Add dependencies to detail response

```python
# In get_mission endpoint, after loading tasks:
deps = (
    db.query(OrchestrationTaskDependency)
    .filter(OrchestrationTaskDependency.task_id.in_([t.id for t in tasks]))
    .all()
)
result["dependencies"] = [
    {"task_id": str(d.task_id), "depends_on_task_id": str(d.depends_on_task_id), "trigger_rule": d.trigger_rule}
    for d in deps
]
```

### 2. Add payload to event response

```python
# In _event_to_response:
"payload": event.payload,  # Already JSONB, just include it
```

### 3. Add verification data to task response (for review screen)

```python
# In _task_to_response, add:
"verification_criteria": task.verification_criteria,
"output_metadata": task.output_metadata,  # Contains verification scores
```

These are additive changes — no breaking modifications to existing API contracts.
