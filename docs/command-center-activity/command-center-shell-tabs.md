# Command Center Shell & Tabs

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/app/command-center/page.tsx](frontend/app/command-center/page.tsx)
- [frontend/app/globals.css](frontend/app/globals.css)
- [frontend/components/__tests__/prd197-substrate-tile.test.tsx](frontend/components/__tests__/prd197-substrate-tile.test.tsx)
- [frontend/components/auth/profile-menu.tsx](frontend/components/auth/profile-menu.tsx)
- [frontend/components/command-center/activity-tab.tsx](frontend/components/command-center/activity-tab.tsx)
- [frontend/components/command-center/agent-tones.ts](frontend/components/command-center/agent-tones.ts)
- [frontend/components/command-center/calendar-tab.tsx](frontend/components/command-center/calendar-tab.tsx)
- [frontend/components/command-center/command-center-shell.tsx](frontend/components/command-center/command-center-shell.tsx)
- [frontend/components/command-center/is-it-working-strip.tsx](frontend/components/command-center/is-it-working-strip.tsx)
- [frontend/components/command-center/sparkline.tsx](frontend/components/command-center/sparkline.tsx)
- [frontend/components/command-center/stats-strip.tsx](frontend/components/command-center/stats-strip.tsx)
- [frontend/components/command-center/summary-tab.tsx](frontend/components/command-center/summary-tab.tsx)
- [frontend/components/layout/__tests__/prd184-us006-placebo-relics.test.tsx](frontend/components/layout/__tests__/prd184-us006-placebo-relics.test.tsx)
- [frontend/components/layout/__tests__/studio-sidebar.test.tsx](frontend/components/layout/__tests__/studio-sidebar.test.tsx)
- [frontend/components/layout/main-layout.tsx](frontend/components/layout/main-layout.tsx)
- [frontend/components/layout/studio-sidebar.tsx](frontend/components/layout/studio-sidebar.tsx)
- [frontend/hooks/use-analytics-api.ts](frontend/hooks/use-analytics-api.ts)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/api/workflows.py](orchestrator/api/workflows.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/models/substrate_metrics.py](orchestrator/core/models/substrate_metrics.py)
- [orchestrator/core/observability/substrate_metrics.py](orchestrator/core/observability/substrate_metrics.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/reports/route-manifest.json](orchestrator/reports/route-manifest.json)
- [orchestrator/router_manifest.py](orchestrator/router_manifest.py)
- [orchestrator/tests/authz_sweep_probe.py](orchestrator/tests/authz_sweep_probe.py)
- [orchestrator/tests/test_p2w2_authz_boundary_sweep.py](orchestrator/tests/test_p2w2_authz_boundary_sweep.py)
- [orchestrator/tests/test_prd154_s5_missions.py](orchestrator/tests/test_prd154_s5_missions.py)
- [orchestrator/tests/test_prd222_w2s1_plan_tiers.py](orchestrator/tests/test_prd222_w2s1_plan_tiers.py)

</details>



The Command Center is the operational cockpit of Automatos AI, providing a centralized view of system activity, agent performance, and ongoing tasks. This page details the `CommandCenterShell` component, its various tabs (Summary, Activity, Calendar, Board, Watchlist, Governance, Questions), the `StatsStrip`, `IsItWorkingStrip`, `Sparkline` visualizations, and the `agent-tones` system.

## Purpose and Scope

The Command Center serves as the primary dashboard for users to monitor and interact with their Automatos AI deployment. It aggregates real-time data from various backend services, presenting it through a user-friendly interface. This page covers the frontend components responsible for rendering this interface and the backend APIs that supply the necessary data.

## Command Center Shell (`CommandCenterShell`)

The `CommandCenterShell` component [frontend/components/command-center/command-center-shell.tsx:81-223]() is the main container for the Command Center. It orchestrates the display of various sub-components, including the header, stats strips, tab navigation, and the content of the currently active tab.

The shell dynamically updates its content based on the active tab, which is controlled by the `tab` query parameter in the URL [frontend/components/command-center/command-center-shell.tsx:86-87]().

### Layout

The layout of the `CommandCenterShell` is structured as follows:
1.  **Editorial Top**: Contains an eyebrow, main heading (`h1`), subheading, and action buttons. The subheading (`lede`) dynamically changes based on the system's activity status (e.g., "A quiet hour" or "X agents working") [frontend/components/command-center/command-center-shell.tsx:131-153]().
2.  **StatsStrip**: Displays high-level operational metrics.
3.  **Tab Strip**: Provides navigation between different Command Center views.
4.  **Tab Body**: Renders the content of the selected tab.

### Tab Management

The `CommandCenterShell` defines a set of `TABS` [frontend/components/command-center/command-center-shell.tsx:52-64]() that users can navigate through. The active tab is determined by the `tab` query parameter in the URL. The `setTab` function [frontend/components/command-center/command-center-shell.tsx:155-159]() updates this parameter, triggering a re-render of the appropriate tab content.

Each tab can display a live count of relevant items, fetched from various data hooks. For example, the "Board" tab shows the count of active tasks, and the "Questions" tab shows the number of open questions [frontend/components/command-center/command-center-shell.tsx:107-123]().

Sources:
* [frontend/components/command-center/command-center-shell.tsx:81-223]()

## Tabs

The Command Center features several tabs, each providing a different perspective on the system's operations.

### Summary Tab (`SummaryTab`)

The `SummaryTab` [frontend/components/command-center/summary-tab.tsx]() provides an overview of key metrics and actionable insights. It often includes widgets and cards that summarize the current state of agents, workflows, and pending actions.

Sources:
* [frontend/components/command-center/summary-tab.tsx]()

### Activity Tab (`ActivityTab`)

The `ActivityTab` [frontend/components/command-center/activity-tab.tsx]() displays a chronological feed of system events, agent actions, and workflow executions. This tab leverages `useActivityFeed` [frontend/components/command-center/command-center-shell.tsx:92]() to fetch activity data.

Sources:
* [frontend/components/command-center/activity-tab.tsx]()
* [frontend/components/command-center/command-center-shell.tsx:92]()

### Calendar Tab (`CalendarTab`)

The `CalendarTab` [frontend/components/command-center/calendar-tab.tsx:1-834]() visualizes scheduled tasks, heartbeat routines, cron playbooks, mission SLA deadlines, and board task SLA deadlines. It supports day, week, and month views.

**Data Source**: The calendar data is sourced from `useActivitySchedule` [frontend/components/command-center/calendar-tab.tsx:48](), which queries the `/api/activity/schedule` endpoint [orchestrator/reports/route-manifest.json:38](). This ensures consistency across all workers.

**Event Types**: The calendar displays five types of items:
*   Heartbeat routines with structured recurrence.
*   Cron playbooks.
*   Agent-scheduled tasks.
*   Mission SLA deadlines.
*   Board task SLA deadlines.

**Recurrence Parsing**: The `parseIntervalMinutes` function [frontend/components/command-center/calendar-tab.tsx:115-126]() parses human-readable frequencies like "Every 30m" into minutes. `cronDaysOfWeek` [frontend/components/command-center/calendar-tab.tsx:131-147]() and `cronHourMin` [frontend/components/command-center/calendar-tab.tsx:153-158]() handle cron expressions.

Sources:
* [frontend/components/command-center/calendar-tab.tsx:1-834]()
* [frontend/components/command-center/calendar-tab.tsx:48]()
* [frontend/components/command-center/calendar-tab.tsx:115-126]()
* [frontend/components/command-center/calendar-tab.tsx:131-147]()
* [frontend/components/command-center/calendar-tab.tsx:153-158]()
* [orchestrator/reports/route-manifest.json:38]()

### Board Tab (`BoardTab`)

The `BoardTab` [frontend/components/command-center/board-tab.tsx]() presents a Kanban-style view of tasks, leveraging `useBoardTasks` [frontend/components/command-center/command-center-shell.tsx:91]() for data. It supports real-time updates via `useBoardEventStream` [frontend/components/command-center/command-center-shell.tsx:100-103](), which subscribes to SSE events and invalidates the board cache on new events.

Sources:
* [frontend/components/command-center/board-tab.tsx]()
* [frontend/components/command-center/command-center-shell.tsx:91]()
* [frontend/components/command-center/command-center-shell.tsx:100-103]()

### Watchlist Tab (`WatchlistTab`)

The `WatchlistTab` [frontend/components/command-center/watchlist-tab.tsx]() displays items that Automatos AI is actively supervising. This tab uses `useWatches` [frontend/components/command-center/command-center-shell.tsx:96]() to fetch the list of active watches. The badge on this tab indicates the number of items currently under supervision [frontend/components/command-center/command-center-shell.tsx:116-117]().

Sources:
* [frontend/components/command-center/watchlist-tab.tsx]()
* [frontend/components/command-center/command-center-shell.tsx:96]()
* [frontend/components/command-center/command-center-shell.tsx:116-117]()

### Questions Tab (`QuestionsTab`)

The `QuestionsTab` [frontend/components/command-center/questions-tab.tsx]() lists all open questions or pending agent asks, along with any blocked downstream tasks. The badge on this tab shows a live count of open questions, fetched via `useQuestions` [frontend/components/command-center/command-center-shell.tsx:98]() [frontend/components/command-center/command-center-shell.tsx:118-119](). The `ApprovalGrant` interface [frontend/lib/api-client.ts:169-200]() defines the structure for these questions, which can include `question_md`, `options`, and `cascade` information.

Sources:
* [frontend/components/command-center/questions-tab.tsx]()
* [frontend/components/command-center/command-center-shell.tsx:98]()
* [frontend/components/command-center/command-center-shell.tsx:118-119]()
* [frontend/lib/api-client.ts:169-200]()

### Governance Tab (`GovernanceTab`)

The `GovernanceTab` [frontend/components/command-center/governance-tab.tsx]() provides a human-facing interface for the policy plane, including approvals, audit logs, and compliance information. This tab is typically accessible only to workspace administrators. It does not have a live badge count on the main tab strip [frontend/components/command-center/command-center-shell.tsx:120-122](). The `GovernanceStatus` [frontend/lib/api-client.ts:206-215]() and `ApprovalGrant` [frontend/lib/api-client.ts:169-200]() interfaces define the data structures for this tab.

Sources:
* [frontend/components/command-center/governance-tab.tsx]()
* [frontend/components/command-center/command-center-shell.tsx:120-122]()
* [frontend/lib/api-client.ts:206-215]()
* [frontend/lib/api-client.ts:169-200]()

## Stats Strip (`StatsStrip`)

The `StatsStrip` component [frontend/components/command-center/stats-strip.tsx]() displays a row of high-level operational statistics. These statistics are fetched using `useActivityStats` [frontend/components/command-center/command-center-shell.tsx:90]() and include metrics like agents working, tasks in queue, and items needing attention.

Sources:
* [frontend/components/command-center/stats-strip.tsx]()
* [frontend/components/command-center/command-center-shell.tsx:90]()

## Is-It-Working Strip (`IsItWorkingStrip`)

The `IsItWorkingStrip` component [frontend/components/command-center/is-it-working-strip.tsx]() provides a quick, at-a-glance status of the system's health. It leverages various metrics defined in `api-client.ts`, such as `ActivationMetric`, `MissionSuccessRateMetric`, `ErrorsBySubsystemMetric`, `WidgetEngagementMetric`, `PrimitiveHealthMetric`, `SubstrateHealthMetric`, `SlosMetric`, `WorkspaceActivationMetric`, `DeliverableFreshnessMetric`, and `CommerceIntegrityMetric` [frontend/lib/api-client.ts:38-135](). These metrics are fetched from the `/api/analytics/real` endpoints [orchestrator/main.py:97-98]().

Sources:
* [frontend/components/command-center/is-it-working-strip.tsx]()
* [frontend/lib/api-client.ts:38-135]()
* [orchestrator/main.py:97-98]()

## Sparkline

The term "sparkline" refers to small, high-density charts that visualize trends over time. While a dedicated `Sparkline` component is mentioned [frontend/components/command-center/sparkline.tsx](), its specific integration within the Command Center shell is typically for visualizing trends within the `StatsStrip` or `IsItWorkingStrip` for metrics like agent activity or mission success rates.

Sources:
* [frontend/components/command-center/sparkline.tsx]()

## Agent Tones (`agent-tones`)

The `agent-tones` system [frontend/components/command-center/agent-tones.ts]() provides a way to assign distinct visual "tones" or colors to agents. This helps in quickly identifying agents in various UI components, such as the Calendar tab where `toneFor` [frontend/components/command-center/calendar-tab.tsx:54]() is used to color-code events based on the agent involved.

Sources:
* [frontend/components/command-center/agent-tones.ts]()
* [frontend/components/command-center/calendar-tab.tsx:54]()

## Command Center Data Flow

The Command Center relies on a variety of backend API endpoints to fetch the data displayed in its shell and tabs.

```mermaid
graph TD
    subgraph Frontend
        A[CommandCenterShell] --> B[StatsStrip]
        A --> C[IsItWorkingStrip]
        A --> D[Tab Navigation]
        D --> D1[SummaryTab]
        D --> D2[ActivityTab]
        D --> D3[CalendarTab]
        D --> D4[BoardTab]
        D --> D5[WatchlistTab]
        D --> D6[QuestionsTab]
        D --> D7[GovernanceTab]
        D3 --> E[agent-tones]
        B --> F[Sparkline]
        C --> F
    end

    subgraph Backend (orchestrator)
        G[API Endpoints]
        G1[/api/activity/stats]
        G2[/api/analytics/real/*]
        G3[/api/activity/schedule]
        G4[/api/board/tasks]
        G5[/api/activity/feed]
        G6[/api/kpi/decisions-needed]
        G7[/api/watches]
        G8[/api/governance/grants]
        G9[/api/governance/status]
        G10[/api/governance/audit-log]
    end

    B -- "useActivityStats" --> G1
    C -- "useAnalyticsApi hooks" --> G2
    D3 -- "useActivitySchedule" --> G3
    D4 -- "useBoardTasks" --> G4
    D4 -- "useBoardEventStream (SSE)" --> G4
    D2 -- "useActivityFeed" --> G5
    D1 -- "useDecisionsNeeded" --> G6
    D5 -- "useWatches" --> G7
    D6 -- "useQuestions" --> G8
    D7 -- "useGovernance hooks" --> G9
    D7 -- "useGovernance hooks" --> G10

    style A fill:#ace,stroke:#333,stroke-width:2px
    style B fill:#bce,stroke:#333,stroke-width:2px
    style C fill:#bce,stroke:#333,stroke-width:2px
    style D fill:#cde,stroke:#333,stroke-width:2px
    style D1 fill:#def,stroke:#333,stroke-width:2px
    style D2 fill:#def,stroke:#333,stroke-width:2px
    style D3 fill:#def,stroke:#333,stroke-width:2px
    style D4 fill:#def,stroke:#333,stroke-width:2px
    style D5 fill:#def,stroke:#333,stroke-width:2px
    style D6 fill:#def,stroke:#333,stroke-width:2px
    style D7 fill:#def,stroke:#333,stroke-width:2px
    style E fill:#fef,stroke:#333,stroke-width:2px
    style F fill:#fef,stroke:#333,stroke-width:2px
    style G fill:#eec,stroke:#333,stroke-width:2px
    style G1 fill:#ffe,stroke:#333,stroke-width:2px
    style G2 fill:#ffe,stroke:#333,stroke-width:2px
    style G3 fill:#ffe,stroke:#333,stroke-width:2px
    style G4 fill:#ffe,stroke:#333,stroke-width:2px
    style G5 fill:#ffe,stroke:#333,stroke-width:2px
    style G6 fill:#ffe,stroke:#333,stroke-width:2px
    style G7 fill:#ffe,stroke:#333,stroke-width:2px
    style G8 fill:#ffe,stroke:#333,stroke-width:2px
    style G9 fill:#ffe,stroke:#333,stroke-width:2px
    style G10 fill:#ffe,stroke:#333,stroke-width:2px
```
**Title: Command Center Data Flow**

Sources:
* [frontend/components/command-center/command-center-shell.tsx]()
* [orchestrator/reports/route-manifest.json]()
* [orchestrator/main.py]()
* [frontend/lib/api-client.ts]()

## Frontend Component Interaction

The `MainLayout` component [frontend/components/layout/main-layout.tsx:29-193]() is responsible for rendering the overall application structure, including the sidebar and header. When the application is in "Studio" mode and on a desktop, it renders the `StudioSidebar` [frontend/components/layout/studio-sidebar.tsx:64-194]() and `StudioHeader` [frontend/components/layout/studio-header.tsx]() components. The `CommandCenterShell` is then rendered within this layout.

The `StudioSidebar` [frontend/components/layout/studio-sidebar.tsx:64-194]() provides the main navigation, including links to the Command Center. It dynamically filters navigation items based on the user's plan tier and edition [frontend/components/layout/studio-sidebar.tsx:55-61]().

```mermaid
graph TD
    A[MainLayout] --> B{isStudio && !isMobileLayout?}
    B -- Yes --> C[StudioSidebar]
    B -- Yes --> D[StudioHeader]
    B -- Yes --> E[StudioPageTabs]
    B -- Yes --> F[main.sh-fullbleed / main.px-4]
    F --> G[CommandCenterShell]
    B -- No --> H[Sidebar]
    B -- No --> I[MobileSidebar]
    B -- No --> J[Header]
    B -- No --> K[main.px-4]
    K --> G

    G --> L[StatsStrip]
    G --> M[IsItWorkingStrip]
    G --> N[Tab Components (Summary, Activity, etc.)]

    style A fill:#ace,stroke:#333,stroke-width:2px
    style C fill:#bce,stroke:#333,stroke-width:2px
    style D fill:#bce,stroke:#333,stroke-width:2px
    style E fill:#bce,stroke:#333,stroke-width:2px
    style F fill:#cde,stroke:#333,stroke-width:2px
    style G fill:#def,stroke:#333,stroke-width:2px
    style H fill:#bce,stroke:#333,stroke-width:2px
    style I fill:#bce,stroke:#333,stroke-width:2px
    style J fill:#bce,stroke:#333,stroke-width:2px
    style K fill:#cde,stroke:#333,stroke-width:2px
    style L fill:#fef,stroke:#333,stroke-width:2px
    style M fill:#fef,stroke:#333,stroke-width:2px
    style N fill:#fef,stroke:#333,stroke-width:2px
```
**Title: Frontend Layout and Command Center Integration**

Sources:
* [frontend/components/layout/main-layout.tsx:29-193]()
* [frontend/components/layout/studio-sidebar.tsx:64-194]()
* [frontend/components/layout/studio-sidebar.tsx:55-61]()
* [frontend/components/layout/studio-header.tsx]()

---