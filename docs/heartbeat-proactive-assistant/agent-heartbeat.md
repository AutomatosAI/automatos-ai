# Agent Heartbeat

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/activity/board/board-card.tsx](frontend/components/activity/board/board-card.tsx)
- [frontend/components/activity/board/board-column.tsx](frontend/components/activity/board/board-column.tsx)
- [frontend/components/activity/board/board-task-viewer.tsx](frontend/components/activity/board/board-task-viewer.tsx)
- [frontend/components/activity/board/board-view.tsx](frontend/components/activity/board/board-view.tsx)
- [frontend/components/activity/board/index.ts](frontend/components/activity/board/index.ts)
- [frontend/components/command-center/__tests__/calendar-actions.test.ts](frontend/components/command-center/__tests__/calendar-actions.test.ts)
- [frontend/components/command-center/__tests__/calendar-tab-feed.test.tsx](frontend/components/command-center/__tests__/calendar-tab-feed.test.tsx)
- [frontend/components/command-center/__tests__/calendar-tab-scope.test.tsx](frontend/components/command-center/__tests__/calendar-tab-scope.test.tsx)
- [frontend/components/command-center/board-tab.tsx](frontend/components/command-center/board-tab.tsx)
- [frontend/components/command-center/calendar-actions.ts](frontend/components/command-center/calendar-actions.ts)
- [frontend/hooks/use-board-tasks.ts](frontend/hooks/use-board-tasks.ts)
- [frontend/hooks/use-heartbeats-api.ts](frontend/hooks/use-heartbeats-api.ts)
- [frontend/types/board.ts](frontend/types/board.ts)
- [orchestrator/api/board_tasks.py](orchestrator/api/board_tasks.py)
- [orchestrator/services/orchestration_board_bridge.py](orchestrator/services/orchestration_board_bridge.py)

</details>



## Purpose and Scope

Agent Heartbeat is the scheduled proactive execution system enabling Automatos agents to act autonomously without waiting for direct user interactions. It implements a specialized execution pipeline integrated with the **BoardTask** system, allowing agents to function as persistent workers that pull, process, and complete tasks from a workspace kanban board.

This page details the implementation of the `Agent Heartbeat` logic, its integration with `BoardTask` status transitions, the orchestration-to-board bridging layer, and the context injection mechanism that provides agents with task-specific instructions during autonomous ticks.

Sources: [orchestrator/api/board_tasks.py:1-7](), [orchestrator/services/heartbeat_service.py:1-15]()

---

## System Architecture & Data Flow

The heartbeat system is managed by `HeartbeatService`, leveraging `APScheduler` with an optional `RedisJobStore` for distributed scheduling [orchestrator/services/heartbeat_service.py:126-133](). It handles workspace-level orchestrator ticks and agent-specific heartbeat routines [orchestrator/services/heartbeat_service.py:128-133]().

### Heartbeat Execution Flow

The diagram below bridges the natural language concepts of agent routines to the underlying code entities executing the heartbeat loop.

```mermaid
graph TD
    subgraph "Scheduling Subsystem"
        ["APScheduler"]
        ["CronTrigger"]
    end

    subgraph "Heartbeat Services orchestrator/services/"
        ["HeartbeatService._agent_tick"]
        ["HeartbeatService._is_within_active_hours"]
        ["BoardTask_Scanner"]
    end

    subgraph "Execution & Reporting orchestrator/modules/agents/"
        ["AgentFactory.execute_with_prompt"]
        ["ContextService_HEARTBEAT_mode"]
        ["ReportService._auto_create_task_report"]
    end

    subgraph "Data Persistence core/models/"
        ["Agent"]
        ["BoardTask"]
        ["agent_reports"]
    end

    ["APScheduler"] -->|Trigger| ["HeartbeatService._agent_tick"]
    ["CronTrigger"] --> ["APScheduler"]
    ["HeartbeatService._agent_tick"] --> ["HeartbeatService._is_within_active_hours"]
    ["HeartbeatService._is_within_active_hours"] -->|Within Hours| ["BoardTask_Scanner"]
    ["BoardTask_Scanner"] -->|Fetch Tasks| ["BoardTask"]
    ["HeartbeatService._agent_tick"] -->|Activate| ["AgentFactory.execute_with_prompt"]
    ["AgentFactory.execute_with_prompt"] -->|Build Context| ["ContextService_HEARTBEAT_mode"]
    ["AgentFactory.execute_with_prompt"] -->|Persist Metrics| ["ReportService._auto_create_task_report"]
    ["ReportService._auto_create_task_report"] -->|Generate| ["agent_reports"]
```

Sources: [orchestrator/services/heartbeat_service.py:126-187](), [orchestrator/api/board_tasks.py:61-165]()

---

## BoardTask Integration & Lifecycle

The primary function of the Agent Heartbeat is processing tasks from the workspace board. The `BoardTask` model supports an expanded Kanban lifecycle: `inbox`, `assigned`, `in_progress`, `review`, `blocked`, `done`, `failed`, and `cancelled` [orchestrator/api/board_tasks.py:43-43](), [frontend/types/board.ts:6-6]().

### Status Transitions & Mapping

When a heartbeat picks up a task, its status transitions to `in_progress`. Upon completion, it moves to `done`, `review`, or `failed` depending on execution outcome [orchestrator/api/board_tasks.py:118-121]().

| Transition | Event | Implementation Symbol |
|:---|:---|:---|
| `assigned` → `in_progress` | Heartbeat selects task for execution. | `update_board_task_status` [orchestrator/api/board_tasks.py:115-125]() |
| `in_progress` → `done` | Agent execution succeeds. | `BoardTask.status = 'done'` [orchestrator/api/board_tasks.py:118-121]() |
| `in_progress` → `review` | Task requires human/LLM approval. | `ReviewMode` validation [orchestrator/api/board_tasks.py:38-45]() |
| `in_progress` → `failed` | Unhandled error during execution. | `record_error` / `error_message` stamping [orchestrator/api/board_tasks.py:28-28]() |

### Mission to Board Bridging

Missions and orchestration runs sync with the kanban board via `Orchestration Board Bridge` (`orchestration_board_bridge.py`) [orchestrator/services/orchestration_board_bridge.py:1-16](). Mission runs map to parent `BoardTask` records (`source_type='orchestration'`), while individual orchestration tasks map to child `BoardTask` records (`source_type='orchestration_task'`) [orchestrator/services/orchestration_board_bridge.py:71-114](), [orchestrator/services/orchestration_board_bridge.py:131-195]().

```mermaid
graph LR
    subgraph "Orchestration Domain core/models/"
        ["OrchestrationRun"]
        ["OrchestrationTask"]
    end

    subgraph "Bridge Service orchestrator/services/"
        ["create_mission_board_task"]
        ["create_task_board_task"]
        ["_resolve_board_status"]
    end

    subgraph "Kanban Board core/models/"
        ["BoardTask_Parent"]
        ["BoardTask_Child"]
    end

    ["OrchestrationRun"] -->|Triggers| ["create_mission_board_task"]
    ["create_mission_board_task"] --> ["BoardTask_Parent"]
    ["OrchestrationTask"] -->|Triggers| ["create_task_board_task"]
    ["create_task_board_task"] -->|Maps State| ["_resolve_board_status"]
    ["_resolve_board_status"] --> ["BoardTask_Child"]
```

Sources: [orchestrator/api/board_tasks.py:43-53](), [orchestrator/services/orchestration_board_bridge.py:1-195](), [frontend/types/board.ts:6-6]()

---

## Execution Pipeline & Auto-Reporting

When an agent executes via heartbeat or task assignment, the execution engine wraps the prompt generation and triggers downstream telemetry and reporting.

### Auto-Reporting (`_auto_create_task_report`)
Upon completing a task, `_auto_create_task_report` persists an `agent_reports` row so deliverables appear in the Activity Feed and Reports dashboard [orchestrator/api/board_tasks.py:61-71](). 
1. **Result Harvesting**: Pulls text from execution results (`exec_result.get("result")`, `task.result`) [orchestrator/api/board_tasks.py:83-90]().
2. **Metrics Rollup**: Aggregates token usage, duration, and cost through `compute_execution_metrics` [orchestrator/api/board_tasks.py:93-105]().
3. **Report Generation**: Formats markdown lines via `ReportService.create_report()` [orchestrator/api/board_tasks.py:161-165]().

Sources: [orchestrator/api/board_tasks.py:60-165]()

---

## Frontend Monitoring & Live State

The frontend interacts with board tasks via React Query hooks (`use-board-tasks.ts`) and real-time Server-Sent Events (SSE) [frontend/hooks/use-board-tasks.ts:1-75]().

### Key Hooks & Components
- `useBoardTasks`: Fetches filtered tasks from `/api/v1/tasks` and groups them by Kanban column [frontend/hooks/use-board-tasks.ts:55-112]().
- `useUpdateTaskStatus`: Provides optimistic updates for drag-and-drop status changes [frontend/hooks/use-board-tasks.ts:117-157]().
- `BoardTaskViewer`: Slideover component that polls `/api/v1/tasks/{id}` when a task is `in_progress` and renders special execution blocks such as CLI sessions (`runtime_ref` for `runtime: cli` tickets like Claude Code) [frontend/components/activity/board/board-task-viewer.tsx:28-135]().

Sources: [frontend/hooks/use-board-tasks.ts:1-209](), [frontend/components/activity/board/board-task-viewer.tsx:28-135]()

---