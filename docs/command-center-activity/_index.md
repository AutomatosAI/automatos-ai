# Command Center & Activity

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/app/command-center/page.tsx](frontend/app/command-center/page.tsx)
- [frontend/components/activity/__tests__/feed-progress-followup.test.ts](frontend/components/activity/__tests__/feed-progress-followup.test.ts)
- [frontend/components/activity/activity-feed-item.tsx](frontend/components/activity/activity-feed-item.tsx)
- [frontend/components/activity/activity-feed.tsx](frontend/components/activity/activity-feed.tsx)
- [frontend/components/activity/execution-detail.tsx](frontend/components/activity/execution-detail.tsx)
- [frontend/components/activity/routine-card.tsx](frontend/components/activity/routine-card.tsx)
- [frontend/components/activity/widgets/agent-reports-widget.tsx](frontend/components/activity/widgets/agent-reports-widget.tsx)
- [frontend/components/activity/widgets/recent-activity-widget.tsx](frontend/components/activity/widgets/recent-activity-widget.tsx)
- [frontend/components/command-center/agent-tones.ts](frontend/components/command-center/agent-tones.ts)
- [frontend/components/command-center/command-center-shell.tsx](frontend/components/command-center/command-center-shell.tsx)
- [frontend/components/command-center/sparkline.tsx](frontend/components/command-center/sparkline.tsx)
- [frontend/components/shared/item-card.tsx](frontend/components/shared/item-card.tsx)
- [frontend/hooks/use-activity-api.ts](frontend/hooks/use-activity-api.ts)
- [frontend/hooks/use-scheduled-tasks-api.ts](frontend/hooks/use-scheduled-tasks-api.ts)
- [orchestrator/modules/tools/discovery/actions_scheduling.py](orchestrator/modules/tools/discovery/actions_scheduling.py)
- [orchestrator/services/activity_service.py](orchestrator/services/activity_service.py)
- [orchestrator/tests/test_schedule_endpoint.py](orchestrator/tests/test_schedule_endpoint.py)

</details>



The Command Center serves as the operational cockpit for Automatos AI, providing a centralized view of all ongoing activities, agent performance, and system health. It is designed to give users a comprehensive overview of their AI operations, enabling them to monitor, manage, and respond to events efficiently. This page offers a high-level introduction to the Command Center's main components, with detailed explanations available in linked child pages.

## Command Center Shell & Tabs

The Command Center shell (`frontend/components/command-center/command-center-shell.tsx`[]) provides the main layout and navigation for the operational cockpit. It features an editorial top section with an eyebrow, title, subtitle, and action cluster, followed by a `StatsStrip` and a tab strip for navigating different views.

The available tabs include:
*   **Summary**: Offers a high-level overview of key metrics and actionable insights.
*   **Board**: Displays tasks and their statuses, often integrating with the `BoardTask` model [orchestrator/services/activity_service.py:20]().
*   **Calendar**: Shows scheduled activities, routines, and playbook executions.
*   **Activity**: Presents a unified feed of all system activities.
*   **Watchlist**: Monitors specific items or agents under supervision.
*   **Questions**: Lists open questions or requests requiring human input.
*   **Governance**: Provides tools for managing policies, approvals, and compliance (workspace-admin only).

The `CommandCenterShell` also includes a `StatsStrip` to show real-time operational statistics and an `IsItWorkingStrip` for quick health checks. Agent tones (`frontend/components/command-center/agent-tones.ts`[]) and sparklines (`frontend/components/command-center/sparkline.tsx`[]) contribute to the visual feedback of system status. Tab counts are dynamically updated using data hooks, ensuring they reflect the current state of the backend [frontend/components/command-center/command-center-shell.tsx:107-125]().

For details, see [Command Center Shell & Tabs](#27.1).

Sources:
* `frontend/components/command-center/command-center-shell.tsx`
* `frontend/components/command-center/agent-tones.ts`
* `frontend/components/command-center/sparkline.tsx`
* `orchestrator/services/activity_service.py`

## Activity Feed & Execution Detail

The Activity Feed provides a unified, chronological stream of all significant events within the Automatos AI system. This includes chats, routine executions, recipe (playbook) executions, and board tasks. The `ActivityService` [orchestrator/services/activity_service.py:67]() is responsible for merging these disparate sources into a single feed, which is then consumed by the frontend via the activity API.

The frontend component `ActivityFeed` (`frontend/components/activity/activity-feed.tsx`[]) uses the `useActivityFeed` hook (`frontend/hooks/use-activity-api.ts`[]) to fetch and display these items. Each item in the feed (`ActivityFeedItem` [frontend/hooks/use-activity-api.ts:32]()) provides a summary of the event, its status, associated agents, and timestamps.

When a user selects an activity item, the `ExecutionDetail` component (`frontend/components/activity/execution-detail.tsx`[]) provides an in-depth view of the execution. This detail includes step-by-step progress, logs, and relevant metadata. Routine cards (`frontend/components/activity/routine-card.tsx`[]) and memory cards are specialized displays for specific activity types. The system also supports reports viewing and grading, allowing for evaluation of agent and playbook performance.

### Activity Feed Data Flow

```mermaid
graph TD
    subgraph "Backend (orchestrator)"
        A[Chat] --> AS
        B[Heartbeat Executions] --> AS
        C[Recipe Executions] --> AS
        D[Board Tasks] --> AS
        AS(ActivityService) -- get_feed() --> API_ACTIVITY_FEED[/api/activity/feed]
        AS -- get_stats() --> API_ACTIVITY_STATS[/api/activity/stats]
    end

    subgraph "Frontend (Next.js)"
        API_ACTIVITY_FEED -- useActivityFeed() --> UAF[useActivityFeed Hook]
        API_ACTIVITY_STATS -- useActivityStats() --> UAS[useActivityStats Hook]
        UAF --> AF[ActivityFeed Component]
        UAS --> CC_SHELL[CommandCenterShell]
        AF --> AFI[ActivityFeedItem Component]
        AFI --> ED[ExecutionDetail Component]
        AF --> RC[RoutineCard Component]
    end

    style AS fill:#f9f,stroke:#333,stroke-width:2px
    style API_ACTIVITY_FEED fill:#bbf,stroke:#333,stroke-width:2px
    style API_ACTIVITY_STATS fill:#bbf,stroke:#333,stroke-width:2px
    style UAF fill:#ccf,stroke:#333,stroke-width:2px
    style UAS fill:#ccf,stroke:#333,stroke-width:2px
    style AF fill:#cfc,stroke:#333,stroke-width:2px
    style AFI fill:#cfc,stroke:#333,stroke-width:2px
    style ED fill:#cfc,stroke:#333,stroke-width:2px
    style RC fill:#cfc,stroke:#333,stroke-width:2px
    style CC_SHELL fill:#cfc,stroke:#333,stroke-width:2px
```
Sources:
* `orchestrator/services/activity_service.py`
* `frontend/hooks/use-activity-api.ts`
* `frontend/components/activity/activity-feed.tsx`
* `frontend/components/activity/activity-feed-item.tsx`
* `frontend/components/activity/execution-detail.tsx`
* `frontend/components/activity/routine-card.tsx`

## Dashboard Widgets

The Command Center dashboard (`frontend/app/command-center/page.tsx`[]) integrates various widgets to provide a quick overview of system status and agent performance. These widgets are designed to be modular and provide focused insights.

Key dashboard widgets include:
*   **Agent Reports**: Displays recent reports generated by agents, often summarizing their activities or findings. The `AgentReportsWidget` (`frontend/components/activity/widgets/agent-reports-widget.tsx`[]) uses the `useAgentReports` hook (`frontend/hooks/use-activity-api.ts`[]) to fetch and display these reports, allowing users to pin specific agents for quick monitoring.
*   **Recent Activity**: Shows a condensed list of the most recent system activities, similar to the full activity feed but with a smaller scope. The `RecentActivityWidget` (`frontend/components/activity/widgets/recent-activity-widget.tsx`[]) leverages the `useActivityFeed` hook to present this data.
*   **Decisions Needed**: Highlights tasks or situations requiring human intervention or approval.
*   **Playbook Metrics**: Provides performance metrics for automated workflows.
*   **Self-Learning Health**: Offers insights into the health and effectiveness of the system's self-learning mechanisms.

These widgets are hooked into various API endpoints and data services to ensure they display up-to-date information.

### Scheduled Task Management

The system also provides robust scheduling capabilities, allowing agents and users to schedule one-shot or recurring tasks. The `platform_schedule_task` action [orchestrator/modules/tools/discovery/actions_scheduling.py:9]() enables agents to programmatically schedule follow-up tasks, which can either open a chat with a target agent or file a board ticket. The `platform_list_scheduled_tasks` [orchestrator/modules/tools/discovery/actions_scheduling.py:86]() and `platform_cancel_scheduled_task` [orchestrator/modules/tools/discovery/actions_scheduling.py:117]() actions provide management capabilities for these scheduled items.

The `useUpdateScheduledTaskStatus` hook (`frontend/hooks/use-scheduled-tasks-api.ts`[]) allows the frontend to change the status of scheduled tasks (active, paused, cancelled), while `useCreateScheduledBoardTask` (`frontend/hooks/use-scheduled-tasks-api.ts`[]) facilitates the creation of new scheduled board tasks. The `ActivityService.get_schedule` method [orchestrator/services/activity_service.py:1000]() (tested in `orchestrator/tests/test_schedule_endpoint.py`[]) is crucial for populating the calendar view with these scheduled items, ensuring a stateless and consistent read across all instances.

For details, see [Dashboard Widgets](#27.3).

Sources:
* `frontend/app/command-center/page.tsx`
* `frontend/components/activity/widgets/agent-reports-widget.tsx`
* `frontend/hooks/use-activity-api.ts`
* `frontend/components/activity/widgets/recent-activity-widget.tsx`
* `orchestrator/modules/tools/discovery/actions_scheduling.py`
* `frontend/hooks/use-scheduled-tasks-api.ts`
* `orchestrator/services/activity_service.py`
* `orchestrator/tests/test_schedule_endpoint.py`

---