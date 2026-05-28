# Agent Heartbeat

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/alembic/versions/wave3_escalation_level.py](orchestrator/alembic/versions/wave3_escalation_level.py)
- [orchestrator/core/services/escalation.py](orchestrator/core/services/escalation.py)
- [orchestrator/modules/tools/discovery/actions_reports.py](orchestrator/modules/tools/discovery/actions_reports.py)
- [orchestrator/modules/tools/discovery/actions_workspace.py](orchestrator/modules/tools/discovery/actions_workspace.py)
- [orchestrator/modules/tools/discovery/handlers_reports.py](orchestrator/modules/tools/discovery/handlers_reports.py)
- [orchestrator/modules/tools/discovery/handlers_workspace.py](orchestrator/modules/tools/discovery/handlers_workspace.py)
- [orchestrator/services/heartbeat_service.py](orchestrator/services/heartbeat_service.py)
- [orchestrator/services/report_service.py](orchestrator/services/report_service.py)

</details>



## Purpose and Scope

Agent Heartbeat is the scheduled proactive execution system that enables Automatos agents to act autonomously without waiting for direct user messages. It implements a specialized execution pipeline that integrates with the **BoardTask** system, allowing agents to function as always-on workers that pull, process, and complete tasks from a workspace board.

This page details the implementation of the `Agent Heartbeat` logic, its integration with `BoardTask` status transitions, and the context injection mechanism that provides agents with task-specific instructions during autonomous ticks.

---

## System Architecture

The heartbeat system is managed by the `HeartbeatService`, which uses `APScheduler` with a `RedisJobStore` for persistent, distributed scheduling [orchestrator/services/heartbeat_service.py:24-31](). It manages both workspace-level "Orchestrator Heartbeats" and agent-specific "Agent Heartbeats" [orchestrator/services/heartbeat_service.py:96-121]().

### Agent Heartbeat Data Flow

The diagram below maps the heartbeat lifecycle from the scheduling trigger to the final task status update in the database.

**Heartbeat Execution Flow**
```mermaid
graph TD
    subgraph "Scheduling_Layer"
        ["APScheduler"]
        ["CronTrigger"]
    end

    subgraph "HeartbeatService_Logic"
        ["_agent_tick"]
        ["_is_within_active_hours"]
        ["BoardTask_Scanner"]
    end

    subgraph "Execution_Runtime"
        ["AgentFactory_execute_with_prompt"]
        ["ContextService_HEARTBEAT_mode"]
        ["OrchestrationBoardBridge"]
    end

    subgraph "Persistence_Layer"
        ["Agent_Table"]
        ["BoardTask_Table"]
        ["heartbeat_results"]
    end

    ["APScheduler"] -->|Trigger| ["_agent_tick"]
    ["CronTrigger"] --> ["APScheduler"]
    ["_agent_tick"] --> ["_is_within_active_hours"]
    ["_is_within_active_hours"] -->|Within Hours| ["BoardTask_Scanner"]
    ["BoardTask_Scanner"] -->|Fetch Tasks| ["BoardTask_Table"]
    ["_agent_tick"] -->|Activate| ["AgentFactory_execute_with_prompt"]
    ["AgentFactory_execute_with_prompt"] -->|Build| ["ContextService_HEARTBEAT_mode"]
    ["ContextService_HEARTBEAT_mode"] -->|Inject Task Metadata| ["AgentFactory_execute_with_prompt"]
    ["AgentFactory_execute_with_prompt"] -->|Update| ["OrchestrationBoardBridge"]
    ["OrchestrationBoardBridge"] -->|assigned to in_progress to done| ["BoardTask_Table"]
    ["_agent_tick"] -->|Log Result| ["heartbeat_results"]
```

**Sources:** [orchestrator/services/heartbeat_service.py:17-31](), [orchestrator/services/heartbeat_service.py:59-63](), [orchestrator/services/heartbeat_service.py:129-161]()

---

## BoardTask Integration

The primary function of the Agent Heartbeat is to process tasks from the workspace board. The service identifies tasks where `assigned_agent_id` matches the current agent and the status is specifically `assigned`.

### Task Status Transitions

The heartbeat logic enforces a strict state machine for tasks to ensure visibility in the UI and prevent double-processing. The `BoardTask` model includes an `escalation_level` column (L0-L4) to allow for unified triaging of heartbeat-driven tasks [orchestrator/alembic/versions/wave3_escalation_level.py:20-24]().

| Transition | Event | Implementation |
|:---|:---|:---|
| `assigned` → `in_progress` | Heartbeat selects task for execution. | Sets `started_at` timestamp and updates `BoardTask.status`. |
| `in_progress` → `done` | Agent execution completes successfully. | Sets `completed_at` and `result`. |
| `in_progress` → `review` | Task requires human approval or has failed. | Status changed to `review` based on `requires_approval` flag. [orchestrator/core/services/escalation.py:95-96]() |

### Escalation and Classification
During heartbeat execution, if an agent encounters an issue or completes a critical task, the event is processed through the `classify` function [orchestrator/core/services/escalation.py:72-85](). This function maps event payloads to the `EscalationLevel` ladder:
- **L0 FYI**: Informational only.
- **L1 TASK**: Needs execution/work.
- **L2 APPROVAL**: Requires human intervention [orchestrator/core/services/escalation.py:29]().
- **L3 URGENT**: Immediate attention required [orchestrator/core/services/escalation.py:30]().

**Sources:** [orchestrator/core/services/escalation.py:26-31](), [orchestrator/core/services/escalation.py:72-113](), [orchestrator/alembic/versions/wave3_escalation_level.py:20-39]()

---

## Heartbeat Execution Pipeline

When a heartbeat tick occurs, the agent is provided with a specific execution context designed for autonomous work.

### Context Injection
The `HeartbeatService` triggers a tick that activates the agent via `AgentFactory.execute_with_prompt` using the `HEARTBEAT` context mode.

1. **Identity & Configuration**: The agent's core persona and heartbeat settings are loaded from the `Agent.configuration` JSONB field [orchestrator/services/heartbeat_service.py:126-131]().
2. **Task Context**: If a task is assigned, metadata is injected. The agent can also use `platform_submit_report` to persist its findings [orchestrator/modules/tools/discovery/actions_reports.py:9-16]().
3. **Report Generation**: Agents are encouraged to submit reports after significant heartbeat work using `platform_submit_report`, which stores the content in the workspace filesystem and database [orchestrator/services/report_service.py:156-173]().

**Code Entity Mapping: Agent Actions to Persistence**
```mermaid
graph LR
    subgraph "Agent_Runtime"
        ["HeartbeatService"]
        ["PlatformActionExecutor"]
    end

    subgraph "Action_Handlers"
        ["submit_report"]
        ["store_memory"]
    end

    subgraph "Data_Persistence"
        ["ReportService"]
        ["UnifiedMemoryService"]
        ["agent_reports_table"]
    end

    ["HeartbeatService"] -->|Triggers Tick| ["PlatformActionExecutor"]
    ["PlatformActionExecutor"] -->|Calls| ["submit_report"]
    ["PlatformActionExecutor"] -->|Calls| ["store_memory"]
    ["submit_report"] -->|Invokes| ["ReportService"]
    ["store_memory"] -->|Invokes| ["UnifiedMemoryService"]
    ["ReportService"] -->|Writes to| ["agent_reports_table"]
```

**Sources:** [orchestrator/services/heartbeat_service.py:129-161](), [orchestrator/modules/tools/discovery/handlers_reports.py:14-64](), [orchestrator/services/report_service.py:156-173]()

---

## Configuration & Scheduling

Heartbeat behavior is controlled via the `agent.configuration` JSONB field in the `Agent` model [orchestrator/services/heartbeat_service.py:126-131]().

### Configuration Schema

| Field | Description |
|:---|:---|
| `enabled` | Enables/disables the scheduler for this agent [orchestrator/services/heartbeat_service.py:128](). |
| `interval_minutes` | Frequency of ticks (default 30-60m) [orchestrator/services/heartbeat_service.py:178](). |
| `active_hours` | Timezone-aware window for autonomous activity [orchestrator/services/heartbeat_service.py:30](). |

### Interval to Cron Conversion
To ensure agents fire at predictable times, intervals are converted to fixed cron patterns using the `_interval_to_cron_trigger` helper [orchestrator/services/heartbeat_service.py:139-151]().

| Interval | Resulting Cron Logic |
|:---|:---|
| < 60 min | Distribute evenly within the hour (e.g., 15m → `0,15,30,45`) [orchestrator/services/heartbeat_service.py:155-159](). |
| 1440 min (Daily) | Fixed at 9 AM daily [orchestrator/services/heartbeat_service.py:163-165](). |
| 10080 min (Weekly)| Fixed at Monday 9 AM [orchestrator/services/heartbeat_service.py:160-162](). |

**Sources:** [orchestrator/services/heartbeat_service.py:139-172](), [orchestrator/services/heartbeat_service.py:178-183]()

---

## Reporting and Memory Persistence

During a heartbeat, agents often generate long-term value through reports and memory storage.

### Report Submission
Agents use the `platform_submit_report` tool to document their heartbeat activity. The `ReportService` handles the dual-write pattern:
1. **Workspace File**: A markdown file is written to `reports/{agent_slug}/{timestamp}_{title}.md` [orchestrator/services/report_service.py:179-181]().
2. **Database Row**: Metadata including `escalation_level`, `status`, and `metrics` is inserted into the `agent_reports` table [orchestrator/services/report_service.py:211-220]().

### Memory Storage
Agents can also persist facts discovered during heartbeat ticks using `platform_store_memory` [orchestrator/modules/tools/discovery/actions_workspace.py:61-70](). This tool allows agents to set `source_type` (e.g., `platform_verified`, `inference`) and `confidence` levels, which are stored in the `UnifiedMemoryService` [orchestrator/modules/tools/discovery/handlers_workspace.py:145-182]().

**Sources:** [orchestrator/services/report_service.py:156-200](), [orchestrator/modules/tools/discovery/handlers_workspace.py:145-185](), [orchestrator/modules/tools/discovery/actions_reports.py:9-110]()

---