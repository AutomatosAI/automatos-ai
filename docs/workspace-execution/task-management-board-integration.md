# Task Management & Board Integration

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/activity/board/board-agent-sidebar.tsx](frontend/components/activity/board/board-agent-sidebar.tsx)
- [frontend/components/activity/board/board-card.tsx](frontend/components/activity/board/board-card.tsx)
- [frontend/components/activity/board/board-column.tsx](frontend/components/activity/board/board-column.tsx)
- [frontend/components/activity/board/board-filters.tsx](frontend/components/activity/board/board-filters.tsx)
- [frontend/components/activity/board/board-task-viewer.tsx](frontend/components/activity/board/board-task-viewer.tsx)
- [frontend/components/activity/board/board-view.tsx](frontend/components/activity/board/board-view.tsx)
- [frontend/components/activity/board/index.ts](frontend/components/activity/board/index.ts)
- [frontend/hooks/use-board-tasks.ts](frontend/hooks/use-board-tasks.ts)
- [frontend/types/board.ts](frontend/types/board.ts)
- [orchestrator/api/board_tasks.py](orchestrator/api/board_tasks.py)
- [orchestrator/services/board_task_bridge.py](orchestrator/services/board_task_bridge.py)
- [orchestrator/services/orchestration_board_bridge.py](orchestrator/services/orchestration_board_bridge.py)

</details>



The Task Management system provides a centralized Kanban-style interface for tracking agent activities, manual tasks, and mission executions. It bridges high-level user goals with the low-level execution layer, allowing for real-time status tracking and lifecycle management from "Inbox" to "Done".

## BoardTask Data Model

The `BoardTask` model is the central entity for the Kanban board. It supports manual tasks created by users, automated tasks from Recipes, and multi-agent coordination from the Mission (Orchestration) system. [orchestrator/api/board_tasks.py:5-7]()

*   **Statuses**: `inbox`, `assigned`, `in_progress`, `review`, `blocked`, `done`. [orchestrator/api/board_tasks.py:28-28](), [frontend/types/board.ts:6-6]()
*   **Priorities**: `urgent`, `high`, `medium`, `low`. [orchestrator/api/board_tasks.py:29-29](), [frontend/types/board.ts:8-8]()
*   **Review Modes**: `human`, `llm`, `auto`. [orchestrator/api/board_tasks.py:30-30](), [frontend/types/board.ts:10-10]()
*   **SLA Tracking**: Tasks include an `sla_deadline` calculated based on priority (e.g., 4 hours for urgent, 72 hours for low). [orchestrator/api/board_tasks.py:33-38](), [orchestrator/services/orchestration_board_bridge.py:34-39]()
*   **Planning Data**: A `JSONB` field storing step-by-step progress, execution IDs, and approval actions. [orchestrator/services/board_task_bridge.py:54-58](), [frontend/types/board.ts:42-42]()

### Task Entity Mapping
The following diagram maps the logical task concepts to the physical database and API entities.

**Diagram: Task Entity Mapping**
```mermaid
graph TD
    subgraph "NaturalLanguageSpace"
        ["User Goal / Prompt"]
        ["Board Card UI"]
    end

    subgraph "CodeEntitySpace"
        BT["class BoardTask (core.models.core)"]
        BTRouter["/api/v1/tasks (orchestrator/api/board_tasks.py)"]
        BTBridge["board_task_bridge.py (orchestrator/services)"]
        OBBridge["orchestration_board_bridge.py (orchestrator/services)"]
        
        BT --- BTRouter
        BT --- BTBridge
        BT --- OBBridge
        ["User Goal / Prompt"] --> BTRouter
        ["Board Card UI"] --- BT
    end

    subgraph "ExecutionSpace"
        RE["RecipeExecution (core.models.workflow)"]
        OR["OrchestrationRun (core.models.orchestration)"]
        OT["OrchestrationTask (core.models.orchestration)"]
        
        BTBridge --- RE
        OBBridge --- OR
        OBBridge --- OT
    end
```
Sources: [orchestrator/api/board_tasks.py:21-26](), [orchestrator/services/board_task_bridge.py:17-18](), [orchestrator/services/orchestration_board_bridge.py:24-31]()

## Board Integration Bridges

Two primary bridge services synchronize external execution states with the Kanban board.

### 1. Recipe Bridge
The `board_task_bridge.py` service handles standard recipe (playbook) executions. [orchestrator/services/board_task_bridge.py:1-9]()

| Function | Purpose |
| :--- | :--- |
| `create_recipe_board_task` | Initializes a `BoardTask` when a recipe starts, linking it via `source_id`. [orchestrator/services/board_task_bridge.py:22-67]() |
| `update_recipe_board_task_progress` | Updates `step_progress` in `planning_data` JSONB field. [orchestrator/services/board_task_bridge.py:70-93]() |
| `complete_recipe_board_task` | Moves task to `done` and attaches `result` or `error_message`. [orchestrator/services/board_task_bridge.py:95-128]() |

### 2. Orchestration (Mission) Bridge
The `orchestration_board_bridge.py` service integrates complex multi-agent missions. [orchestrator/services/orchestration_board_bridge.py:1-16]()

*   **Mission Mapping**: A mission (`OrchestrationRun`) creates a parent `BoardTask` with `source_type='orchestration'`. [orchestrator/services/orchestration_board_bridge.py:76-128]()
*   **Task Mapping**: Individual mission steps (`OrchestrationTask`) create child `BoardTask` rows linked via `parent_task_id`. [orchestrator/services/orchestration_board_bridge.py:136-216]()
*   **Status Sync**: Uses `BOARD_STATUS_MAP` and `_resolve_board_status` to translate internal mission states (e.g., `todo`, `in_review`) to Kanban terms (e.g., `inbox`, `review`). [orchestrator/services/orchestration_board_bridge.py:49-68]()

Sources: [orchestrator/services/board_task_bridge.py:1-128](), [orchestrator/services/orchestration_board_bridge.py:1-216]()

## Task Lifecycle & Notifications

Tasks follow a state machine from creation to completion. Upon completion, the system utilizes the `NotificationDispatcher` and `ReportService` to persist results and alert users. [orchestrator/api/board_tasks.py:42-160](), [orchestrator/api/board_tasks.py:163-196]()

**Diagram: Task Status Transitions**
```mermaid
stateDiagram-v2
    [*] --> inbox: "User/System Create"
    inbox --> assigned: "Agent Assigned"
    assigned --> in_progress: "Execution Start"
    in_progress --> review: "Requires Approval / Failed"
    review --> done: "Approved / Auto-Resolved"
    in_progress --> blocked: "Error / Dependency"
    blocked --> in_progress: "Resolved"
    in_progress --> done: "Success"
```
Sources: [orchestrator/api/board_tasks.py:28-28](), [orchestrator/services/orchestration_board_bridge.py:49-58](), [frontend/types/board.ts:6-6]()

## Board View UI Components

The frontend provides a real-time Kanban board using `@hello-pangea/dnd` for drag-and-drop interactions. [frontend/components/activity/board/board-view.tsx:4-4]()

### Component Architecture
*   **BoardView**: Orchestrates filters (agent, priority, type) and manages the `DragDropContext`. It uses `useUpdateTaskStatus` for optimistic UI updates during drag-and-drop. [frontend/components/activity/board/board-view.tsx:21-49](), [frontend/hooks/use-board-tasks.ts:99-139]()
*   **BoardCard**: Displays task type (Mission, Playbook, or Task), assignee icons, and SLA indicators via `SlaIndicator`. It handles visual indicators for overdue tasks. [frontend/components/activity/board/board-card.tsx:18-52](), [frontend/components/activity/board/board-card.tsx:87-140]()
*   **BoardTaskViewer**: A modal providing deep introspection. It uses the `useLiveTask` hook to poll the backend every 5 seconds for live output and progress updates when a task is `in_progress`. [frontend/components/activity/board/board-task-viewer.tsx:26-49]()
*   **BoardFiltersBar**: Provides multi-dimensional filtering by priority, task type (Mission/Playbook/Task), and assigned agent using `useAgents` hook. [frontend/components/activity/board/board-filters.tsx:51-135](), [frontend/components/activity/board/board-filters.tsx:38-39]()

### Live Data Flow
The frontend maintains synchronization with the backend through a combination of polling and optimistic updates.

**Diagram: UI Synchronization Flow**
```mermaid
sequenceDiagram
    participant UI as "BoardTaskViewer (React)"
    participant Hook as "useLiveTask (React Query)"
    participant API as "/api/v1/tasks/{id} (FastAPI)"
    participant DB as "PostgreSQL (BoardTask)"

    UI->>Hook: "Mount (status='in_progress')"
    loop "Every 5 Seconds"
        Hook->>API: "GET /api/v1/tasks/{id}"
        API->>DB: "Query current status/result"
        DB-->>API: "BoardTask record"
        API-->>Hook: "JSON data (step_progress, result)"
        Hook-->>UI: "Update progress bar & output"
    end
```
Sources: [frontend/components/activity/board/board-task-viewer.tsx:26-49](), [frontend/hooks/use-board-tasks.ts:39-57](), [orchestrator/api/board_tasks.py:16-26]()

## Task Completion & Approval Flow

Tasks with `review_mode` enabled or specific `approval_action` triggers require manual intervention.
*   **Approval**: `useApproveTask` triggers a `POST` to `/api/v1/tasks/{taskId}/approve`, which executes associated logic and transitions status. [frontend/hooks/use-board-tasks.ts:144-158]()
*   **Rejection**: `useRejectTask` triggers a `POST` to `/api/v1/tasks/{taskId}/reject` with optional feedback, typically moving the task back to `blocked`. [frontend/hooks/use-board-tasks.ts:163-177]()
*   **Auto-Reporting**: Completed tasks trigger `_auto_create_task_report`, which calculates execution metrics (cost, tokens, duration) and creates a `Report` entry. [orchestrator/api/board_tasks.py:42-160]()

Sources: [frontend/hooks/use-board-tasks.ts:141-177](), [orchestrator/api/board_tasks.py:42-160]()

---