# Workflow API Reference

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/agents/org-chart-tab.tsx](frontend/components/agents/org-chart-tab.tsx)
- [frontend/components/knowledge/BusinessGraphPanel.tsx](frontend/components/knowledge/BusinessGraphPanel.tsx)
- [frontend/components/workflows/execution-kitchen.tsx](frontend/components/workflows/execution-kitchen.tsx)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/api/knowledge_graph.py](orchestrator/api/knowledge_graph.py)
- [orchestrator/api/marketplace.py](orchestrator/api/marketplace.py)
- [orchestrator/api/workflow_recipes.py](orchestrator/api/workflow_recipes.py)
- [orchestrator/api/workflows.py](orchestrator/api/workflows.py)
- [orchestrator/core/seeds/platform-management-skill.md](orchestrator/core/seeds/platform-management-skill.md)
- [orchestrator/modules/context/sections/graph_context.py](orchestrator/modules/context/sections/graph_context.py)
- [orchestrator/modules/knowledge/graph_extraction.py](orchestrator/modules/knowledge/graph_extraction.py)
- [orchestrator/modules/knowledge/graph_service.py](orchestrator/modules/knowledge/graph_service.py)
- [orchestrator/modules/tools/discovery/actions_graph.py](orchestrator/modules/tools/discovery/actions_graph.py)
- [orchestrator/modules/tools/discovery/handlers_graph.py](orchestrator/modules/tools/discovery/handlers_graph.py)

</details>



This page documents the REST API endpoints for workflow and recipe management, including CRUD operations, execution control, quality assessment, and learning analysis. The system provides two primary execution paths: the **Workflow Pipeline** (legacy 9-stage/PRD-59 dynamic) and the **Recipe Direct Executor** (modern step-by-step).

---

## API Architecture Overview

The workflow API is organized into modular routers within the FastAPI application. These routers handle distinct responsibilities from template management to real-time execution tracking.

**Workflow API Routers Architecture**

```mermaid
graph TB
    subgraph "FastAPI Application [main.py]"
        Main["FastAPI App"]
    end
    
    subgraph "Recipe Management [/api/workflow-recipes]"
        RecipeRouter["workflow_recipes.py"]
        RecipeCRUD["CRUD Operations"]
        RecipeExec["Direct Execution<br/>POST /{id}/execute"]
        RecipeWebhooks["Webhook Triggers<br/>/webhook/{webhook_id}"]
        
        RecipeRouter --> RecipeCRUD
        RecipeRouter --> RecipeExec
        RecipeRouter --> RecipeWebhooks
    end
    
    subgraph "Workflow Management [/api/workflows]"
        WorkflowRouter["workflows.py"]
        WorkflowCRUD["Legacy CRUD"]
        WorkflowActive["Active Monitoring<br/>GET /active"]
        WorkflowTracker["WorkflowStageTracker<br/>SSE Progress"]
        
        WorkflowRouter --> WorkflowCRUD
        WorkflowRouter --> WorkflowActive
        WorkflowRouter --> WorkflowTracker
    end
    
    subgraph "Execution Backend"
        RecipeExecutor["recipe_executor.py<br/>_execute_recipe_direct()"]
        
        RecipeExec --> RecipeExecutor
    end
    
    subgraph "Data Layer [core/models]"
        WorkflowRecipeModel["WorkflowTemplate<br/>(WorkflowRecipe)"]
        RecipeExecutionModel["RecipeExecution"]
        WorkflowModel["Workflow"]
        
        RecipeCRUD --> WorkflowRecipeModel
        RecipeExec --> RecipeExecutionModel
        WorkflowCRUD --> WorkflowModel
    end
    
    Main --> RecipeRouter
    Main --> WorkflowRouter
```

Sources: [orchestrator/api/workflow_recipes.py:22-31](), [orchestrator/api/workflows.py:34-35]()

---

## Recipe CRUD Endpoints

The recipe management endpoints provide full lifecycle control for workflow recipes, which are stored as `WorkflowTemplate` models in the database.

### List Recipes
**Endpoint**: `GET /api/workflow-recipes`

Lists all workflow recipes in the current workspace with filtering and sorting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_featured`| boolean | - | Filter by featured status |
| `is_public`  | boolean | `true` | Filter by public visibility |
| `search`     | string  | - | Search in name and description |
| `sort_by`    | string  | `popularity` | `popularity`, `created_at`, `use_count`, `name` |

**Implementation Details**:
The list endpoint applies workspace isolation via `get_request_context_hybrid` and enriches step data with agent information through the `_enrich_steps_with_agents()` helper. This helper fetches the `Agent` ORM objects and attaches model configurations and tool counts to each step.

Sources: [orchestrator/api/workflow_recipes.py:177-187](), [orchestrator/api/workflow_recipes.py:140-174]()

### Create and Manage Recipes
**Endpoint**: `POST /api/workflow-recipes`

Creates a new workflow recipe. If the `schedule_config` uses a Composio trigger, the system calls `_auto_register_trigger()` to subscribe via the Composio API and store a `TriggerSubscription`.

**Scheduling Logic**:
The system uses `_sync_cron_schedule()` to interface with the `PlaybookSchedulerService`. If a recipe is configured with a cron expression, it is added to the system scheduler via `scheduler.schedule_playbook(recipe)`; otherwise, it is unscheduled.

Sources: [orchestrator/api/workflow_recipes.py:34-48](), [orchestrator/api/workflow_recipes.py:50-89]()

---

## Recipe Execution Endpoints

The execution system provides a direct step-by-step path that uses the same components as the chatbot (ContextService, ToolRouter) for consistency.

### Execute Recipe
**Endpoint**: `POST /api/workflow-recipes/{recipe_id}/execute`

Launches a recipe execution as an asynchronous background task.

**Execution Flow**:
1. **Concurrency Control**: Uses a per-workspace semaphore (`_workspace_semaphores`) to bound concurrent recipe execution (default: 3).
2. **Activation**: The `AgentFactory` activates the agent for each step via `activate_agent(agent.id)`, providing its LLM manager.
3. **Context Construction**: `ContextService(RECIPE)` builds the system prompt and base tools.
4. **Tool Injection**: Injects the `scratchpad_write` and `scratchpad_read` tools into the agent's context for inter-step communication.
5. **Iteration Limit**: Defaults to 25 LLM tool-call turns per step.

Sources: [orchestrator/api/workflow_recipes.py:155-172](), [orchestrator/api/workflow_recipes.py:50-89]()

---

## Workflow Stage Tracking (SSE)

For complex missions and legacy workflows, the `WorkflowStageTracker` provides real-time progress updates via Server-Sent Events (SSE).

**Stage and Phase Architecture**:
The tracker supports both the legacy 9-stage pipeline and the PRD-59 dynamic phases.

| Phase | Label | Stages Included |
|-------|-------|-----------------|
| `PLAN` | Planning | 1 (Decomposition), 2 (Selection), 2b (Negotiation) |
| `PREPARE` | Preparation | 3 (Context Engineering), 3b (Optimization) |
| `EXECUTE` | Execution | 4 (Execution), 4b (Coordination) |
| `EVALUATE`| Evaluation | 5 (Aggregation), 6 (Learning Update) |
| `LEARN` | Learning | 7 (Quality), 8 (Memory), 9 (Response) |

**Real-time Event Flow**:
```mermaid
sequenceDiagram
    participant API as Workflows API [/api/workflows]
    participant Tracker as WorkflowStageTracker
    participant Redis as Redis Pub/Sub
    participant SSE as SSE Stream Manager
    
    API->>Tracker: start_phase("PLAN")
    Tracker->>SSE: broadcast_event("phase_start")
    Tracker->>Redis: publish_workflow_event()
    API->>Tracker: start_stage(1)
    Tracker->>SSE: broadcast_event("stage_start")
    API->>Tracker: complete_stage(1, result)
    Tracker->>SSE: broadcast_event("stage_complete")
```

Sources: [orchestrator/api/workflows.py:37-68](), [orchestrator/api/workflows.py:88-106](), [orchestrator/api/workflows.py:126-141]()

---

## Marketplace Integration

Workflows can be published to the community marketplace by setting their `owner_type` to `marketplace`.

**Marketplace API Flow**:
1. **Listing**: `GET /api/marketplace/items?type=recipe` queries the `WorkflowTemplate` table where `owner_type == 'marketplace'`.
2. **Installation**: `POST /api/marketplace/install` clones the marketplace item into the user's workspace.
3. **Platform Actions**: Agents can browse and manage marketplace items using tools like `platform_browse_marketplace_agents` and `platform_install_plugin`.

Sources: [orchestrator/api/marketplace.py:152-155](), [orchestrator/api/marketplace.py:89-98](), [orchestrator/core/seeds/platform-management-skill.md:8-15]()

---

## Knowledge Graph API

The Workflow system integrates with the Knowledge Graph for advanced entity and relationship management.

**Knowledge API to Code Mapping**

```mermaid
graph LR
    subgraph "Frontend [BusinessGraphPanel.tsx]"
        UI["BusinessGraphPanel"]
        Viz["BusinessGraphVisualization"]
    end

    subgraph "API Layer [knowledge_graph.py]"
        EntitiesAPI["/api/knowledge/entities"]
        SearchAPI["/api/knowledge/entities/search"]
    end

    subgraph "Service Layer [graph_service.py]"
        GS["GraphifyService"]
        Build["build_graph()"]
        Load["load_graph()"]
    end

    subgraph "Extraction [graph_extraction.py]"
        Extract["_DOCUMENT_EXTRACTION_PROMPT"]
        Report["_REPORT_EXTRACTION_PROMPT"]
    end

    UI -- "apiClient.buildBusinessGraph()" --> Build
    EntitiesAPI -- "DB Query" --> GS
    Build -- "LLM Extraction" --> Extract
    Build -- "LLM Extraction" --> Report
```

Sources: [orchestrator/api/knowledge_graph.py:84-142](), [orchestrator/modules/knowledge/graph_service.py:128-150](), [orchestrator/modules/knowledge/graph_extraction.py:101-187](), [frontend/components/knowledge/BusinessGraphPanel.tsx:174-186]()

---

## Frontend Integration

The frontend interacts with these endpoints via the `apiClient` and specialized React components for monitoring.

**Key Components**:
- **ExecutionKitchen**: A real-time theater for viewing execution logs. It maps incoming SSE events to `STAGE_NAMES` for display and uses `TheaterStageProgress` for visualization.
- **OrgChartTab**: Visualizes agent relationships and teams within a workspace using `OrgChartCanvas`, allowing users to see the "company structure" created by workflows like Mission Zero.
- **StreamingLog**: A sub-component within `ExecutionKitchen` that renders `LogEntry` items with icons based on event types like `agent_spawn`, `task_progress`, and `memory_write`.

Sources: [frontend/components/workflows/execution-kitchen.tsx:74-84](), [frontend/components/workflows/execution-kitchen.tsx:36-38](), [frontend/components/agents/org-chart-tab.tsx:16-24](), [frontend/components/workflows/execution-kitchen.tsx:119-131]()

---