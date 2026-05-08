# Heartbeat & Proactive Assistant

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

The Heartbeat & Proactive Assistant system enables scheduled, autonomous checks for both workspace-level orchestrator health and individual agent task completion. This system transforms Automatos from reactive to proactive, allowing agents to monitor assigned work and the orchestrator to perform periodic workspace health assessments and autonomous actions.

For multi-channel message handling that enables heartbeat notifications, see [Channel Integrations](#12). For agent execution details, see [Agents](#5). For tool execution used during heartbeat ticks, see [Tools & Integrations](#8). For the unified notification pipeline used to alert users of heartbeat completions, see [Unified Notification System](#24).

**Sources:** [orchestrator/services/heartbeat_service.py:1-10](), [orchestrator/services/heartbeat_service.py:24-31]()

---

## Architecture Overview

The heartbeat system consists of two primary components: **orchestrator heartbeats** (workspace-level health checks) and **agent heartbeats** (task completion monitors). Both use `APScheduler` with cron triggers for precise scheduling and respect active-hours configurations to prevent off-hours resource consumption.

### System Topology

```mermaid
graph TB
    subgraph "Scheduling Layer"
        Scheduler["APScheduler<br/>(AsyncIOScheduler)"]
        JobStore["Redis JobStore<br/>Persistent Jobs"]
        Scheduler --> JobStore
    end
    
    subgraph "Heartbeat Service"
        HB["HeartbeatService"]
        OrchestratorTick["_orchestrator_tick()"]
        AgentTick["_agent_tick()"]
        ActiveHoursGuard["_is_within_active_hours()"]
        
        HB --> OrchestratorTick
        HB --> AgentTick
        HB --> ActiveHoursGuard
    end
    
    subgraph "Orchestrator Tick Pipeline"
        LLMMode["_orchestrator_tick_llm()"]
        ShallowMode["_orchestrator_tick_shallow()"]
        ContextSvc["ContextService<br/>HEARTBEAT mode"]
        PlatformActions["PlatformActionExecutor<br/>47 platform_* tools"]
        
        OrchestratorTick --> LLMMode
        OrchestratorTick --> ShallowMode
        LLMMode --> ContextSvc
        LLMMode --> PlatformActions
    end
    
    subgraph "Agent Tick Pipeline"
        BoardScan["BoardTask Query<br/>status=assigned"]
        AgentFactory["AgentFactory.execute_with_prompt()"]
        StatusUpdate["Update BoardTask<br/>assigned → in_progress → done"]
        
        AgentTick --> BoardScan
        BoardScan --> AgentFactory
        AgentFactory --> StatusUpdate
    end
    
    subgraph "Storage & Delivery"
        HeartbeatResults["heartbeat_results table"]
        NotificationDelivery["_deliver_notification()"]
        ChannelManager["ChannelManager<br/>Multi-channel delivery"]
        ReportSvc["ReportService<br/>platform_submit_report"]
        
        OrchestratorTick --> HeartbeatResults
        AgentTick --> HeartbeatResults
        OrchestratorTick --> NotificationDelivery
        NotificationDelivery --> ChannelManager
        AgentTick --> ReportSvc
    end
    
    Scheduler -->|"schedule_orchestrator_heartbeat()"| OrchestratorTick
    Scheduler -->|"schedule_agent_heartbeat()"| AgentTick
```

**Sources:** [orchestrator/services/heartbeat_service.py:24-91](), [orchestrator/services/heartbeat_service.py:173-222](), [orchestrator/services/report_service.py:149-172]()

---

## Heartbeat Service

The `HeartbeatService` class manages all heartbeat scheduling and execution. It initializes with a shared `AsyncIOScheduler` instance and loads active heartbeat configurations from the database on startup. It enforces rate limits, such as a maximum of 5 concurrent heartbeats per workspace.

### Initialization and Lifecycle

| Method | Purpose |
|--------|---------|
| `start(scheduler)` | Initialize scheduler, load configs, schedule daily summary [orchestrator/services/heartbeat_service.py:43-84]() |
| `stop()` | Remove heartbeat jobs, shutdown owned scheduler [orchestrator/services/heartbeat_service.py:85-91]() |
| `_load_heartbeat_configs()` | Query database for active heartbeat settings [orchestrator/services/heartbeat_service.py:96-133]() |

The service supports both standalone mode (creates its own scheduler with `RedisJobStore`) and shared mode where it integrates with a system-wide scheduler.

**Sources:** [orchestrator/services/heartbeat_service.py:24-91]()

### Cron Trigger Conversion

The service converts interval minutes to `CronTrigger` expressions for predictable execution at fixed times (e.g., top of the hour):

```mermaid
graph LR
    Input["interval_minutes"]
    Convert["_interval_to_cron_trigger()"]
    Output["CronTrigger"]
    
    Input -->|"15 min"| Convert
    Convert -->|"'0,15,30,45 * * * *'"| Output
    
    Input -->|"60 min"| Convert
    Convert -->|"'0 * * * *'"| Output
    
    Input -->|"1440 min"| Convert
    Convert -->|"'0 9 * * *'"| Output
```

**Sources:** [orchestrator/services/heartbeat_service.py:139-171]()

---

## Orchestrator Heartbeat

Orchestrator heartbeats perform workspace-level health checks. They use `ContextService` in `HEARTBEAT` mode to build a system prompt and execute a tool loop (max 5 iterations) to analyze workspace health and perform maintenance tasks.

### LLM-Powered Tick

The orchestrator tick executes via `_orchestrator_tick_llm()`, which handles complex multi-turn interactions and context trimming to avoid token overflow while analyzing workspace state.

```mermaid
sequenceDiagram
    participant Scheduler
    participant HB as HeartbeatService
    participant CS as ContextService
    participant LLM as LLMManager
    participant PE as PlatformActionExecutor
    participant DB as Database
    
    Scheduler->>HB: _orchestrator_tick(workspace_id, config)
    HB->>HB: _is_within_active_hours()
    
    HB->>CS: build_context(mode=HEARTBEAT)
    CS-->>HB: ContextResult(system_prompt, tools)
    
    HB->>LLM: generate_response(messages, tools)
    
    loop Tool Loop (max 5 iterations)
        LLM-->>HB: response with tool_calls
        HB->>PE: execute(tool_name, params)
        PE-->>HB: tool_result
        HB->>LLM: continue with tool results
    end
    
    LLM-->>HB: final response
    HB->>DB: store_heartbeat_result()
```

**Sources:** [orchestrator/services/heartbeat_service.py:382-546]()

### Tool Loop Implementation

The loop in `_orchestrator_tick_llm()` manages message accumulation and exchange boundary detection to keep the context window clean while allowing the orchestrator to call `platform_*` actions to inspect the system.

**Sources:** [orchestrator/services/heartbeat_service.py:461-537]()

---

## Agent Heartbeat

Agent heartbeats monitor the `BoardTask` table for work assigned to specific agents and execute tasks autonomously. This allows agents to work on long-running tasks without direct user interaction in a chat.

### Task Execution Flow

```mermaid
graph TB
    Trigger["Agent Heartbeat Trigger"]
    ActiveHours{"Within Active Hours?"}
    
    subgraph "Task Discovery"
        Query["Query BoardTask<br/>WHERE status='assigned'"]
        TaskContext["Build task_context"]
    end
    
    subgraph "Agent Execution"
        StatusInProgress["Update status: 'in_progress'"]
        ExecutePrompt["AgentFactory.execute_with_prompt()"]
        StatusDone["Update status: 'done'"]
    end
    
    subgraph "Output Handling"
        ReportSvc["ReportService.create_report()"]
        PlatformAction["platform_submit_report"]
    end
    
    Trigger --> ActiveHours
    ActiveHours -->|Yes| Query
    Query --> TaskContext
    TaskContext --> StatusInProgress
    StatusInProgress --> ExecutePrompt
    ExecutePrompt --> StatusDone
    StatusDone --> PlatformAction
    PlatformAction --> ReportSvc
```

**Sources:** [orchestrator/services/heartbeat_service.py:591-735](), [orchestrator/modules/tools/discovery/handlers_reports.py:14-92]()

### Escalation and Reporting
During heartbeat runs, agents use `platform_submit_report` to persist findings. These reports are classified using an `EscalationLevel` (L0-L4) to determine triage priority.
- **L0 (FYI):** Informational standups.
- **L2 (APPROVAL):** Recommendations requiring human decision.
- **L3 (URGENT):** Critical errors or budget exceeded.

**Sources:** [orchestrator/core/services/escalation.py:26-31](), [orchestrator/modules/tools/discovery/actions_reports.py:29-38]()

---

## Configuration & Scheduling

Heartbeat configurations are stored as JSON blobs within the `agents` and `workspaces` tables, allowing per-agent and per-workspace customization of intervals and prompts.

| Scope | Storage Location |
|-------|-----------------|
| Orchestrator | `workspaces.settings['orchestrator']['heartbeat']` [orchestrator/services/heartbeat_service.py:116-121]() |
| Agent | `agents.configuration['heartbeat']` [orchestrator/services/heartbeat_service.py:126-131]() |

### Scheduling API

The `HeartbeatService` provides methods to dynamically schedule or remove jobs when configurations change via the UI:
- `schedule_orchestrator_heartbeat(workspace_id, hb_config)` [orchestrator/services/heartbeat_service.py:173-199]()
- `schedule_agent_heartbeat(agent_id, workspace_id, hb_config)` [orchestrator/services/heartbeat_service.py:200-232]()

---

## Heartbeat API Reference

The heartbeat API provides endpoints for managing configurations, triggering manual runs, and monitoring execution history.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/{agent_id}/config` | GET/PUT | Manage agent heartbeat settings |
| `/api/agents/{agent_id}/last` | GET | Get most recent result for an agent |
| `/api/orchestrator/run` | POST | Trigger immediate orchestrator tick |
| `/api/orchestrator/history` | GET | List recent orchestrator results |

---

## Results and Notifications

### Heartbeat Results

All executions store detailed telemetry in the `heartbeat_results` table, including `findings` (the LLM's analysis), `actions_taken` (tools called), and financial metrics like `cost` and `tokens_used`.

**Sources:** [orchestrator/services/heartbeat_service.py:346-380]()

### Notification Delivery

The `_deliver_notification()` method routes results to configured destinations. It supports internal orchestrator logging, direct webhooks, or external channels like Slack and Telegram via the `ChannelManager`. It also triggers the `NotificationDispatcher` for in-app alerts.

**Sources:** [orchestrator/services/heartbeat_service.py:737-780]()

---