# Self-Management Harness

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/api/harness.py](orchestrator/api/harness.py)
- [orchestrator/api/widget_workflows.py](orchestrator/api/widget_workflows.py)
- [orchestrator/modules/knowledge/primitive_heartbeat.py](orchestrator/modules/knowledge/primitive_heartbeat.py)
- [orchestrator/modules/memory/operations/__init__.py](orchestrator/modules/memory/operations/__init__.py)
- [orchestrator/modules/tools/discovery/actions_harness.py](orchestrator/modules/tools/discovery/actions_harness.py)
- [orchestrator/modules/tools/discovery/actions_routing.py](orchestrator/modules/tools/discovery/actions_routing.py)
- [orchestrator/modules/tools/discovery/handlers_harness.py](orchestrator/modules/tools/discovery/handlers_harness.py)
- [orchestrator/modules/tools/discovery/handlers_routing.py](orchestrator/modules/tools/discovery/handlers_routing.py)
- [orchestrator/services/harness_service.py](orchestrator/services/harness_service.py)
- [orchestrator/tests/test_graph_single_store.py](orchestrator/tests/test_graph_single_store.py)
- [orchestrator/tests/test_harness_api.py](orchestrator/tests/test_harness_api.py)
- [orchestrator/tests/test_harness_power_mode.py](orchestrator/tests/test_harness_power_mode.py)
- [orchestrator/tests/test_harness_routing_rule.py](orchestrator/tests/test_harness_routing_rule.py)
- [orchestrator/tests/test_harness_self_management.py](orchestrator/tests/test_harness_self_management.py)
- [orchestrator/tests/test_p2w1_relics_deleted.py](orchestrator/tests/test_p2w1_relics_deleted.py)
- [orchestrator/tests/test_w1s9_idle_in_tx.py](orchestrator/tests/test_w1s9_idle_in_tx.py)

</details>



The Self-Management Harness, implemented by the `HarnessService` and its associated API, provides a self-optimizing organizational loop for the Automatos AI platform. Its purpose is to autonomously monitor, diagnose, and prescribe configuration changes to agents and the platform itself, ensuring continuous improvement and governance. This system operates largely invisibly to users, running weekly to collect metrics, identify regressions, and apply safe optimizations automatically. Risky changes are flagged for human review via board tasks, integrating governance gates into the self-management process. The harness also enables dynamic routing rule mutation and power mode adjustments from within the product.

Sources:
- [orchestrator/services/harness_service.py:1-12]()

## Harness Service (`HarnessService`)

The `HarnessService` is the core component of the self-management harness. It runs a weekly sweep across all active, opted-in workspaces to analyze performance metrics and propose configuration changes.

### Lifecycle and Scheduling

The `HarnessService` is initialized and started at server startup, similar to `HeartbeatService` and `CoordinatorService` [orchestrator/services/harness_service.py:88-90](). It registers a single cron job, `_harness_sweep`, which runs every Sunday at 2 AM UTC [orchestrator/services/harness_service.py:34,97-104](). This single job design prevents a "thundering herd" problem by iterating through workspaces sequentially and automatically includes new workspaces without requiring a restart.

The `_harness_sweep` method retrieves all active workspaces and filters them based on whether they have opted into the HARNESS service [orchestrator/services/harness_service.py:107-127](). For each eligible workspace, it calls `_harness_tick` to perform the self-optimization process.

### Self-Optimization Loop

The `_harness_tick` method (not fully shown in provided code, but implied by the service description) orchestrates the self-optimization loop, which involves several phases:

1.  **Collect Metrics**: Gathers organization-wide metrics.
2.  **Diagnose Regressions**: Identifies performance degradations or inefficiencies.
3.  **Prescribe Configuration Changes**: Proposes adjustments to agent configurations or platform settings. Each prescription includes a risk score.
4.  **Auto-Apply Safe Changes**: Automatically applies changes with a low risk score (e.g., risk ≤ 2) [orchestrator/services/harness_service.py:6]().
5.  **Queue Risky Changes**: For changes with higher risk scores (e.g., risk ≥ 3), it creates board tasks for human review and approval [orchestrator/services/harness_service.py:7]().
6.  **Snapshot Baseline**: Records a new baseline for comparison in the next weekly run [orchestrator/services/harness_service.py:8-9]().

### Status and Configuration

The `HarnessService` provides a `get_status` method to retrieve the current state of the HARNESS for a given workspace [orchestrator/services/harness_service.py:132-135](). Possible statuses include:
*   `disabled`: Workspace explicitly opted out.
*   `running`: A tick is currently in progress.
*   `failed`: The last tick encountered an exception.
*   `dormant_insufficient_agents`: Fewer than `_MIN_AGENTS` are active [orchestrator/services/harness_service.py:35]().
*   `dormant_insufficient_data`: Insufficient heartbeat history (`_MIN_DATA_DAYS`) [orchestrator/services/harness_service.py:36]().
*   `scheduled_not_run_yet`: Eligible but the cron job hasn't fired.
*   `completed`: The last tick successfully produced a baseline.

Workspaces can opt-in or out of the HARNESS service via their settings [orchestrator/services/harness_service.py:169-173]().

Sources:
- [orchestrator/services/harness_service.py:1-175]()

### Harness Service Diagram

```mermaid
graph TD
    subgraph "Automatos AI Platform"
        A[Server Startup] --> B{HarnessService.start()}
        B --> C{_register_workspace_jobs()}
        C --> D[Scheduler.add_job(harness_sweep, cron)]
        D -- "Every Sunday 2 AM UTC" --> E[Scheduler Trigger]
        E --> F{_harness_sweep()}
        F -- "Iterate active, opted-in workspaces" --> G{_harness_tick(workspace_id)}
        G --> H[Collect Metrics]
        H --> I[Diagnose Regressions]
        I --> J[Prescribe Changes (Risk Score)]
        J -- "Risk <= 2" --> K[Auto-Apply Changes]
        J -- "Risk >= 3" --> L[Queue Board Task for Human Review]
        K --> M[Snapshot New Baseline]
        L --> M
        M --> N[Next Week's Comparison]
    end
    subgraph "Harness Status"
        O[get_status(workspace_id)] --> P{Current State}
        P --> Q[disabled]
        P --> R[running]
        P --> S[failed]
        P --> T[dormant_insufficient_agents]
        P --> U[dormant_insufficient_data]
        P --> V[scheduled_not_run_yet]
        P --> W[completed]
    end
```

Sources:
- [orchestrator/services/harness_service.py:1-175]()

## Harness API

The HARNESS API provides authenticated endpoints for managing high-risk prescriptions generated by the `HarnessService`. This API is crucial for governance, allowing human administrators to review and approve or reject proposed changes.

### API Endpoints

The `api/harness.py` router defines the following endpoints:

*   `POST /api/harness/prescriptions/{rx_id}/approve`: Approves a queued HARNESS prescription. This action applies the change, which is audited and reversible [orchestrator/api/harness.py:80-87]().
*   `POST /api/harness/prescriptions/{rx_id}/reject`: Rejects a queued HARNESS prescription. This prevents the system from re-proposing the same change in the future [orchestrator/api/harness.py:90-97]().
*   `GET /api/harness/self-learning`: Provides a health check for the self-learning components, including HARNESS status, tool-routing recorder stats, and a prescription summary [orchestrator/api/harness.py:100-129]().

### Authentication and Authorization

These endpoints are secured and require `workspace:manage` permission [orchestrator/api/harness.py:80,90](). The system resolves the authenticated principal (user) to an internal `users.id` [orchestrator/api/harness.py:41-58](). This `user_id` is then passed to `handle_harness_command` [orchestrator/api/harness.py:68-70](), which performs further authorization checks.

A key design principle is that approval is **human-admin-only**. If the principal cannot be resolved to an active owner/admin `users` row (e.g., an API key or service principal), the request fails with a `403 Forbidden` status [orchestrator/api/harness.py:17-20,71-72](). If the HARNESS self-management is disabled, a `409 Conflict` status is returned [orchestrator/api/harness.py:73-76]().

### `handle_harness_command`

The actual logic for approving or rejecting prescriptions resides in `api.harness_commands.handle_harness_command` [orchestrator/api/harness.py:30](). This function receives the workspace ID, command (`/approve` or `/reject`), prescription ID, and the resolved `identity` (including the `user_id`).

Sources:
- [orchestrator/api/harness.py:1-129]()
- [orchestrator/tests/test_harness_api.py:1-191]()

### Harness API Flow

```mermaid
graph TD
    subgraph "Command Center Frontend"
        A[User Action: Approve/Reject Prescription] --> B{HTTP POST /api/harness/prescriptions/{rx_id}/(approve|reject)}
    end

    subgraph "FastAPI Backend"
        B --> C{RequestContext (get_request_context_hybrid)}
        C --> D{DB Session (get_db)}
        D --> E{require_workspace_permission("workspace:manage")}
        E -- Authorized --> F{_run_command(db, ctx, command, rx_id)}
        F --> G{_resolve_internal_user_id(db, ctx)}
        G -- "user.id (integer)" --> H{handle_harness_command(db, workspace_id, command, rx_id, identity)}
        H -- "Result: success, unauthorized, message" --> I{Map Result to HTTP Status}
        I -- "unauthorized=True" --> J[HTTPException 403 Forbidden]
        I -- "success=False" --> K[HTTPException 409 Conflict]
        I -- "success=True" --> L[200 OK]
    end

    subgraph "Harness Service (Internal)"
        H --> M[Apply/Reject Prescription Logic]
        M --> N[Update Configuration / Create Audit Log]
    end
```

Sources:
- [orchestrator/api/harness.py:1-129]()
- [orchestrator/tests/test_harness_api.py:1-191]()

## Self-Management Flag and Execution

The HARNESS self-management capabilities are controlled by the `HARNESS_SELF_MANAGEMENT_ENABLED` configuration flag [orchestrator/tests/test_harness_self_management.py:14,124](). When this flag is `False`, no HARNESS tasks are listed or applied [orchestrator/tests/test_harness_self_management.py:165-177](). When enabled, approved board tasks are parsed, their targets resolved, and the prescribed changes are executed.

The `HarnessService` includes a method `_apply_approved_board_tasks` which is responsible for processing approved tasks. It uses an executor to perform actions like `platform_list_tasks` and `platform_list_agents` to gather necessary information, and then applies the changes [orchestrator/tests/test_harness_self_management.py:168-173]().

The `_parse_harness_task` method extracts structured information from a board task's title and description, such as `change_type`, `target_name`, `current_value`, `proposed_value`, `risk_score`, `rationale`, and `expected_improvement` [orchestrator/tests/test_harness_self_management.py:81-96](). It also attempts to resolve the `target_name` to a `target_id` (e.g., an agent ID) [orchestrator/tests/test_harness_self_management.py:83,88](). If a target name cannot be resolved, `target_id` will be `None`, and the system will not attempt to apply the change to an unknown target [orchestrator/tests/test_harness_self_management.py:111-121]().

Sources:
- [orchestrator/tests/test_harness_self_management.py:1-126]()

## Governance Gates and Power Modes

The HARNESS system incorporates governance gates through its risk scoring and human approval process. Changes with higher risk scores are not automatically applied but are instead converted into board tasks, requiring explicit human review and approval via the HARNESS API.

### Power Mode Adjustments

The HARNESS can propose changes to the platform's power modes. For example, a `power_mode_upgrade` or `power_mode_downgrade` prescription maps to the `platform_set_power_mode` action, which updates the `workspace.settings['power_mode']` [orchestrator/tests/test_harness_routing_rule.py:151-159](). This allows the system to autonomously adjust its operational intensity based on observed performance and resource utilization.

Sources:
- [orchestrator/tests/test_harness_routing_rule.py:151-159]()

## Routing Rule Mutation

One of the self-management capabilities of the HARNESS is the ability to mutate routing rules. This allows the system to dynamically optimize how messages are routed to agents or workflows based on observed patterns and performance.

### `platform_create_routing_rule` Action

The `platform_create_routing_rule` action is used to add new routing rules to the system [orchestrator/modules/tools/discovery/actions_routing.py](). This action takes parameters such as `source_channel`, `source_pattern`, `target_agent_id`, `target_workflow_id`, `intent_keywords`, and `priority` [orchestrator/tests/test_harness_routing_rule.py:92-94,135-138]().

The `create_routing_rule` handler function (in `modules/tools/discovery/handlers_routing.py`) inserts a new row into the `routing_rules` table, ensuring it is workspace-scoped [orchestrator/tests/test_harness_routing_rule.py:91-99](). It validates that the rule includes both a target (agent or workflow) and a matcher (e.g., `source_channel` or `source_pattern`), failing closed if these are missing [orchestrator/tests/test_harness_routing_rule.py:106-118](). It also handles invalid priority values by coercing them to a default (e.g., 0) [orchestrator/tests/test_harness_routing_rule.py:120-127]().

### HARNESS Integration

When the HARNESS prescribes a `routing_rule_add` change, the `_auto_apply_prescription` method in `HarnessService` maps this prescription to the `platform_create_routing_rule` action with the appropriate parameters [orchestrator/tests/test_harness_routing_rule.py:131-149](). This enables the HARNESS to autonomously create or modify routing rules to improve system efficiency or agent allocation.

Sources:
- [orchestrator/modules/tools/discovery/actions_routing.py]()
- [orchestrator/modules/tools/discovery/handlers_routing.py]()
- [orchestrator/tests/test_harness_routing_rule.py:1-149]()

## Platform Actions for HARNESS

The HARNESS service exposes several platform actions that allow agents or administrators to interact with its functionality:

*   `platform_harness_status`: Returns the current status of the HARNESS optimization loop, including whether it's disabled, running, failed, dormant, or completed [orchestrator/modules/tools/discovery/actions_harness.py:9-37](). This is useful for checking team performance, optimization status, or overall organizational health. The `harness_status` handler retrieves this information from `HarnessService.get_status` [orchestrator/modules/tools/discovery/handlers_harness.py:14-26]().
*   `platform_harness_trigger`: Manually triggers a HARNESS optimization run outside its weekly cron schedule [orchestrator/modules/tools/discovery/actions_harness.py:39-61](). This is useful after significant organizational changes or incidents. The `harness_trigger` handler calls `HarnessService.trigger_now` [orchestrator/modules/tools/discovery/handlers_harness.py:29-45]().
*   `platform_harness_history`: Lists past HARNESS optimization runs, including dates, prescription counts, applied/queued counts, and convergence status [orchestrator/modules/tools/discovery/actions_harness.py:63-89](). This allows review of historical optimizations. The `harness_history` handler reads baseline files from workspace storage to compile this history [orchestrator/modules/tools/discovery/handlers_harness.py:48-89]().

These actions are registered in the `ActionRegistry` and have defined permission levels and example utterances for discovery [orchestrator/modules/tools/discovery/actions_harness.py:9-89]().

Sources:
- [orchestrator/modules/tools/discovery/actions_harness.py:1-89]()
- [orchestrator/modules/tools/discovery/handlers_harness.py:1-90]()

## Idle-in-Transaction Prevention

A critical aspect of the HARNESS service's robustness, along with other long-running services like the coordinator and heartbeat, is the prevention of "idle in transaction" database states. This issue, where a database connection holds an open transaction across long `await` points (e.g., LLM calls), can block DDL and lead to performance problems.

The solution involves calling `end_open_transaction(db)` immediately before any long `await` operation. This commits the current session, releasing the transaction and returning the connection to an idle state. The `test_w1s9_idle_in_tx.py` suite specifically tests this behavior, ensuring that the commit happens *between* database reads and subsequent `await` calls [orchestrator/tests/test_w1s9_idle_in_tx.py:1-23,170-177]().

Sources:
- [orchestrator/tests/test_w1s9_idle_in_tx.py:1-23,170-177]()

## Knowledge Graph Single Store and Heartbeat

While the HARNESS service itself is distinct from the Knowledge Graph, it interacts with the platform's self-learning and monitoring mechanisms. The `primitive_heartbeat.py` module provides a stateless helper, `_emit_graph_primitive`, to report the health status of the Knowledge Graph build process to the `HeartbeatService` [orchestrator/modules/knowledge/primitive_heartbeat.py:1-23]().

This helper emits a "green" status on successful graph builds/imports and a "down" status with error details on failures [orchestrator/modules/knowledge/primitive_heartbeat.py:34-38,52-54](). This ensures that the health of the Knowledge Graph, a critical component for agent intelligence, is continuously monitored and reported, contributing to the overall self-management and observability of the platform.

Sources:
- [orchestrator/modules/knowledge/primitive_heartbeat.py:1-65]()
- [orchestrator/tests/test_graph_single_store.py:1-75]()

---