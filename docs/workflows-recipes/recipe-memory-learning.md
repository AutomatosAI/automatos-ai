# Recipe Memory & Learning

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/agents/org-chart-tab.tsx](frontend/components/agents/org-chart-tab.tsx)
- [orchestrator/api/marketplace.py](orchestrator/api/marketplace.py)
- [orchestrator/api/workflow_recipes.py](orchestrator/api/workflow_recipes.py)
- [orchestrator/core/seeds/platform-management-skill.md](orchestrator/core/seeds/platform-management-skill.md)

</details>



This page documents the technical implementation of the recipe memory and learning systems. It covers how recipes leverage **Mem0** for long-term task continuity, how **RecipeQualityService** performs 5D assessments of execution performance, and how **RecipeLearningService** extracts patterns to generate improvement suggestions within the "Execution Kitchen."

---

## Recipe Memory System (Mem0 Integration)

Recipes use a specialized memory path to ensure that multi-step workflows maintain context across steps and across different executions of the same recipe. This integration bypasses the standard chat memory to focus on task-specific state.

### Memory Lifecycle
The `RecipeMemoryService` (integrated within the execution flow) manages the interaction with the **Mem0** backend. Unlike standard chat memory, recipe memory is often scoped to the specific `recipe_id` to prevent "cross-talk" between different automated workflows.

1.  **Pre-Execution Retrieval**: Before a recipe starts, `execute_recipe_direct` retrieves relevant memories to inform the initial state [orchestrator/api/recipe_executor.py:18-19]().
2.  **Context Injection**: These memories are injected into the step's system prompt via the `ContextService` in `RECIPE` mode [orchestrator/api/recipe_executor.py:187-191]().
3.  **Step-to-Step Memory**: The `RecipeScratchpad` replaces verbose text dumps, providing 80-90% token savings by allowing agents to explicitly export data via the `scratchpad_write` tool [orchestrator/api/recipe_executor.py:15-16]().
4.  **Post-Execution Storage**: After the final step completes, the summary of the execution is stored back into Mem0 to inform future runs [orchestrator/api/recipe_executor.py:18-19]().

### Recipe Context Construction
The `ContextService` builds the prompt for each recipe step, incorporating the `recipe_step_dict` which contains the current step number, total steps, and previous outputs from the scratchpad [orchestrator/api/recipe_executor.py:179-191]().

**Recipe Memory Data Flow**
```mermaid
graph TD
    subgraph "Execution_Engine [orchestrator/api/recipe_executor.py]"
        [api.recipe_executor] -->|execute_recipe_direct| [Step_Loop]
        [Step_Loop] -->|_execute_step| [AgentFactory:activate_agent]
    end

    subgraph "Memory_Services [modules/context.py]"
        [Agent_Activation] -->|ContextMode.RECIPE| [ContextService]
        [ContextService] -->|Inject| [RecipeMemoryService]
        [RecipeMemoryService] -->|Query| [Mem0_Integration]
    end

    subgraph "External_Storage"
        [Mem0_Integration] <-->|Vector_Search| [Mem0_Cloud/Local]
        [Step_Loop] -->|Write_Summary| [Mem0_Cloud/Local]
    end

    [Step_Loop] -->|inter-step_data| [RecipeScratchpad]
    [RecipeScratchpad] -->|formatted_context| [ContextService]
```
Sources: [orchestrator/api/recipe_executor.py:14-19](), [orchestrator/api/recipe_executor.py:179-191](), [orchestrator/api/recipe_executor.py:110-123]()

---

## Recipe Quality Service (5D Assessment)

The `RecipeQualityService` evaluates completed executions across five distinct dimensions to provide a quantitative measure of performance. This assessment is visualized in the **Execution Kitchen** via the `TheaterSelfLearningPanel` [frontend/components/workflows/execution-kitchen.tsx:39-43]().

### The 5D Assessment Model
Each execution is scored from 0.0 to 1.0 based on:

1.  **Completeness**: Percentage of steps that reached `status='completed'`.
2.  **Accuracy**: LLM-based evaluation of the `output_data` against the original `recipe.instructions`.
3.  **Efficiency**: Actual duration vs. predicted/historical duration for those steps.
4.  **Reliability**: Number of retries and tool-loop iterations required to finish.
5.  **Cost**: Token usage (input/output) relative to the workspace budget.

### Quality Grade Mapping
The service maps the aggregate score to a letter grade displayed in the UI:
- **A**: Score $\ge$ 0.9
- **B**: Score $\ge$ 0.8
- **C**: Score $\ge$ 0.7
- **D**: Score $\ge$ 0.6
- **F**: Score $<$ 0.6

Sources: [frontend/components/workflows/execution-kitchen.tsx:39-43](), [orchestrator/api/workflow_recipes.py:25-28]()

---

## Recipe Learning Service

The `RecipeLearningService` performs post-hoc analysis on `RecipeExecution` records to identify optimization opportunities and extract successful interaction patterns.

### Pattern Extraction
The service analyzes the execution logs, including the `step_results` and `execution_metadata`. It identifies:
- **Success Patterns**: Combinations of tools and prompts that consistently lead to successful steps.
- **Failure Patterns**: Common error messages or "infinite loops" in tool execution.
- **Performance Bottlenecks**: Steps that consume disproportionate time or tokens.

### Learning Data Storage
Results are persisted in the `workflow_recipes` table (aliased from `WorkflowTemplate`) [orchestrator/api/workflow_recipes.py:25-27](). The frontend surfaces these as `LearningData` and `SuggestionsData` within the `TheaterSelfLearningPanel` and `PlaybookSuggestionsPanel` [frontend/components/workflows/execution-kitchen.tsx:39-45]().

| Key | Description |
| :--- | :--- |
| `latest_suggestions` | Actionable prompt or config changes (e.g., "Increase max_iterations for Step 2"). |
| `latest_patterns` | Observed behaviors (e.g., "Step 1 often requires 'search_knowledge' tool"). |
| `analysis_count` | Total number of executions analyzed for this recipe. |

**Learning Analysis Logic**
```mermaid
graph LR
    subgraph "Natural_Language_Space"
        [User_Instructions]
        [Agent_Outputs]
        [Improvement_Suggestions]
    end

    subgraph "Code_Entity_Space"
        [DB_RECIPE:core.models.WorkflowTemplate]
        [DB_EXEC:core.models.core.RecipeExecution]
        [LEARN_SVC:RecipeLearningService]
        [QUALITY_SVC:RecipeQualityService]
        [TRACKER:WorkflowStageTracker]
    end

    [DB_EXEC] -->|step_results| [LEARN_SVC]
    [DB_EXEC] -->|duration_ms| [QUALITY_SVC]
    [LEARN_SVC] -->|extracts| [Improvement_Suggestions]
    [QUALITY_SVC] -->|updates| [DB_RECIPE]
    [TRACKER] -->|stage_complete| [DB_EXEC]
    [Improvement_Suggestions] -->|persisted_in| [DB_RECIPE]
```
Sources: [orchestrator/api/workflow_recipes.py:25-28](), [frontend/components/workflows/execution-kitchen.tsx:39-45](), [orchestrator/api/workflows.py:37-70](), [core/models/core.py:27-27]()

---

## Workflow Execution Stages (PRD-59)

Recipe execution is tracked through a structured lifecycle, moving from planning to learning. The `ExecutionKitchen` component and `WorkflowStageTracker` visualize these transitions in real-time [frontend/components/workflows/execution-kitchen.tsx:47-55](), [orchestrator/api/workflows.py:37-68]().

### Stage Transitions
The `WorkflowStageTracker` supports both legacy 9-stage tracking and the PRD-59 dynamic phases:
1. **PLAN**: Task decomposition (Stage 1), Agent Selection (Stage 2), and Agent Negotiation (Stage 2b) [orchestrator/api/workflows.py:63]().
2. **PREPARE**: Context engineering (Stage 3) and Prompt Optimization (Stage 3b) [orchestrator/api/workflows.py:64]().
3. **EXECUTE**: Agent execution (Stage 4) and Inter-Agent Coordination (Stage 4b) [orchestrator/api/workflows.py:65]().
4. **EVALUATE**: Result aggregation (Stage 5) and Learning Update (Stage 6) [orchestrator/api/workflows.py:66]().
5. **LEARN**: Quality Assessment (Stage 7), Memory Storage (Stage 8), and Response Generation (Stage 9) [orchestrator/api/workflows.py:67]().

**UI to Code Association**
| UI Component | Code Entity | Purpose |
| :--- | :--- | :--- |
| **TheaterStageProgress** | `WorkflowStageTracker` | Visualizes the 9-stage pipeline or PRD-59 phases [frontend/components/workflows/execution-kitchen.tsx:36](). |
| **TheaterStepExecution** | `_execute_step` | Shows real-time tool calls and LLM turns for a specific step [frontend/components/workflows/execution-kitchen.tsx:37](). |
| **TheaterSelfLearningPanel** | `RecipeLearningService` | Displays 5D quality metrics and extracted patterns [frontend/components/workflows/execution-kitchen.tsx:39-43](). |

Sources: [frontend/components/workflows/execution-kitchen.tsx:35-46](), [orchestrator/api/workflows.py:41-68](), [orchestrator/api/recipe_executor.py:5-19]()

---

## API Reference: Learning & Quality

### Trigger Assessment
`POST /api/workflow-recipes/{recipe_id}/assess-quality`
Triggers the quality assessment for a specific execution, updating the `RecipeExecution` record.

### Trigger Learning
`POST /api/workflow-recipes/{recipe_id}/learn`
Triggers the learning service to analyze recent executions and update the `learning_data` JSONB field in the `WorkflowTemplate` model [orchestrator/api/workflow_recipes.py:25-28]().

### Get Suggestions
`GET /api/workflow-recipes/{recipe_id}/suggestions`
Returns the aggregated insights from the `learning_data` field for display in the `PlaybookSuggestionsPanel` [frontend/components/workflows/execution-kitchen.tsx:44]().

### Marketplace Integration
Recipes can be published to the community marketplace. When a recipe is submitted, its metadata and execution history (if approved) are cloned to the marketplace view [orchestrator/api/marketplace.py:123-138](). The `install_count` is tracked to measure the "popularity" of specific automation patterns [orchestrator/api/marketplace.py:174-175]().

Sources: [orchestrator/api/workflow_recipes.py:22-28](), [frontend/components/workflows/execution-kitchen.tsx:44-46](), [orchestrator/api/marketplace.py:123-175]()

---