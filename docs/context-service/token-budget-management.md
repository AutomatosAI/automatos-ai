# Token Budget Management

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/modules/context/adapters/redis_context.py](orchestrator/modules/context/adapters/redis_context.py)
- [orchestrator/modules/context/adapters/vector_field.py](orchestrator/modules/context/adapters/vector_field.py)
- [orchestrator/modules/context/budget.py](orchestrator/modules/context/budget.py)
- [orchestrator/modules/context/experiment.py](orchestrator/modules/context/experiment.py)
- [orchestrator/modules/context/factory.py](orchestrator/modules/context/factory.py)
- [orchestrator/modules/context/instrumentation.py](orchestrator/modules/context/instrumentation.py)
- [orchestrator/modules/context/modes.py](orchestrator/modules/context/modes.py)
- [orchestrator/modules/context/sections/__init__.py](orchestrator/modules/context/sections/__init__.py)
- [orchestrator/modules/context/sections/agent_roster.py](orchestrator/modules/context/sections/agent_roster.py)
- [orchestrator/modules/context/sections/composio.py](orchestrator/modules/context/sections/composio.py)
- [orchestrator/modules/context/sections/mission_context.py](orchestrator/modules/context/sections/mission_context.py)
- [orchestrator/modules/context/sections/onboarding.py](orchestrator/modules/context/sections/onboarding.py)
- [orchestrator/modules/context/sections/plugins.py](orchestrator/modules/context/sections/plugins.py)
- [orchestrator/modules/tools/discovery/handlers_field.py](orchestrator/modules/tools/discovery/handlers_field.py)
- [orchestrator/tests/demo_ab_comparison.py](orchestrator/tests/demo_ab_comparison.py)
- [orchestrator/tests/test_context/__init__.py](orchestrator/tests/test_context/__init__.py)
- [orchestrator/tests/test_context/conftest.py](orchestrator/tests/test_context/conftest.py)
- [orchestrator/tests/test_context/test_budget_manager.py](orchestrator/tests/test_context/test_budget_manager.py)
- [orchestrator/tests/test_context/test_estimator.py](orchestrator/tests/test_context/test_estimator.py)
- [orchestrator/tests/test_context/test_identity_section.py](orchestrator/tests/test_context/test_identity_section.py)
- [orchestrator/tests/test_context/test_memory_section.py](orchestrator/tests/test_context/test_memory_section.py)
- [orchestrator/tests/test_context/test_modes.py](orchestrator/tests/test_context/test_modes.py)
- [orchestrator/tests/test_context/test_service.py](orchestrator/tests/test_context/test_service.py)
- [orchestrator/tests/test_vector_field.py](orchestrator/tests/test_vector_field.py)

</details>



**Purpose**: This page documents the token budget management system in `ContextService`, which controls how much context can be included in LLM prompts across different execution modes. Token budgets prevent prompt assembly from exceeding model context windows while ensuring critical sections (identity, task context, skills) are never dropped.

---

## Overview

The token budget manager allocates a fixed token budget across rendered sections during prompt assembly. Each context mode (e.g., `CHATBOT`, `TASK_EXECUTION`, `HEARTBEAT_ORCHESTRATOR`) defined in `ContextMode` [orchestrator/modules/context/modes.py:13-22]() has a specific `ModeConfig` which dictates the sections included and optional `max_tokens` constraints [orchestrator/modules/context/modes.py:25-33](). During the assembly phase, the `TokenBudgetManager` ensures that the final prompt fits within the model's window. Sections with priority 1-2 are **never dropped**, while sections with higher priority numbers (lower importance) may be trimmed or excluded if the budget is exhausted [orchestrator/modules/context/budget.py:8-9]().

### Context to Code Mapping

```mermaid
graph TB
    subgraph "Natural Language Space"
        Mode["Context Mode<br/>(e.g. Chatbot, Task)"]
        Constraint["Budget Constraints<br/>(Total, Reserved)"]
    end
    
    subgraph "Code Entity Space"
        CM["ContextMode (Enum)<br/>orchestrator/modules/context/modes.py"]
        TB["TokenBudget (Dataclass)<br/>orchestrator/modules/context/budget.py"]
        TBM["TokenBudgetManager (Class)<br/>orchestrator/modules/context/budget.py"]
        DB["DEFAULT_BUDGETS (Dict)<br/>orchestrator/modules/context/budget.py"]
        MC["ModeConfig (Dataclass)<br/>orchestrator/modules/context/modes.py"]
    end
    
    Mode --> CM
    Constraint --> TB
    CM --> MC
    CM --> DB
    TB --> TBM
    TBM -->|"allocate()"| Result["ContextResult<br/>orchestrator/modules/context/result.py"]
    
    Note1["P1-P2: NEVER DROPPED<br/>Identity, Task/Recipe Context"]
    Note2["P3-P10: DROPPABLE<br/>Tools, Skills, Memory, etc."]
    
    TBM -.-> Note1
    TBM -.-> Note2
```

**Sources**:
- [orchestrator/modules/context/modes.py:13-22]()
- [orchestrator/modules/context/modes.py:25-33]()
- [orchestrator/modules/context/budget.py:25-40]()
- [orchestrator/modules/context/budget.py:53-64]()

---

## Budget Configuration

Token budgets are configured per context mode in `DEFAULT_BUDGETS` [orchestrator/modules/context/budget.py:152-202](). A `TokenBudget` defines the total window, tokens reserved for the model's response, and tokens reserved for the conversation message history [orchestrator/modules/context/budget.py:25-35](). The property `available_for_sections` is dynamically computed to determine the remaining space for system prompt sections [orchestrator/modules/context/budget.py:37-40]().

### Default Budgets by Mode

| ContextMode | Total Budget | Reserved (Response) | Reserved (Messages) | Available for Sections |
| :--- | :--- | :--- | :--- | :--- |
| `CHATBOT` | 128,000 | 4,096 | 60,000 | 63,904 |
| `TASK_EXECUTION` | 128,000 | 4,096 | 20,000 | 103,904 |
| `HEARTBEAT_ORCHESTRATOR`| 128,000 | 2,048 | 0 | 125,952 |
| `COORDINATOR` | 131,072 | 4,096 | 0 | 126,976 |
| `NL2SQL` | 128,000 | 2,048 | 2,000 | 123,952 |

**Sources**:
- [orchestrator/modules/context/budget.py:152-202]()
- [orchestrator/modules/context/budget.py:25-40]()

---

## TokenBudgetManager Implementation

The `TokenBudgetManager` [orchestrator/modules/context/budget.py:53-64]() manages token allocation across `RenderedSection` objects. It follows a multi-step trimming algorithm to protect critical information.

### Data Flow and Trimming Logic

```mermaid
graph TD
    Start["allocate(sections, budget)"] --> Cap["Step 1: Apply per-section max_tokens<br/>(Truncate content via estimate)"]
    Cap --> Sum["Step 2: Calculate Total Tokens"]
    Sum --> Check{"Total <= Available?"}
    Check -- "Yes" --> Done["Return All Sections"]
    Check -- "No" --> Sort["Step 3: Sort by Priority Descending<br/>(P10 -> P1)"]
    Sort --> DropLoop{"Drop Section?"}
    DropLoop -- "Priority > 2" --> Remove["Remove Section<br/>Update Total"]
    DropLoop -- "Priority <= 2" --> Skip["Keep Section<br/>(Protected P1-P2)"]
    Remove --> Check
    Skip --> Done
    
    subgraph "Context Classes"
        RS["RenderedSection<br/>(name, priority, content, estimate)"]
        TB["TokenBudget<br/>(total, reserved_resp, reserved_msgs)"]
    end
```

**Key Implementation Details**:
1. **Per-Section Caps**: Before dropping entire sections, the manager applies individual `max_tokens` constraints. If a section exceeds its limit, it is truncated (roughly 4 chars per token) and re-estimated [orchestrator/modules/context/budget.py:81-93]().
2. **Priority-Based Dropping**: If the total still exceeds the budget, it drops sections starting from the highest priority number (least important) [orchestrator/modules/context/budget.py:113-115]().
3. **Protected Tiers**: Sections with `priority <= 2` are **never dropped** [orchestrator/modules/context/budget.py:122-123](). This ensures the agent always knows who it is and what its immediate goal is.
    - `IdentitySection` (Priority 1) [orchestrator/modules/context/sections/identity.py:27]()
    - `OnboardingSection` (Priority 2) [orchestrator/modules/context/sections/onboarding.py:43]()
    - `TaskContextSection` (Priority 2) [orchestrator/modules/context/sections/task_context.py:39]()

**Sources**:
- [orchestrator/modules/context/budget.py:66-145]()
- [orchestrator/modules/context/sections/identity.py:26-29]()
- [orchestrator/modules/context/sections/onboarding.py:42-43]()
- [orchestrator/modules/context/sections/task_context.py:39]()

---

## Section Priority System

The system uses a 1-10 priority scale where lower numbers indicate higher importance. This mapping is enforced in `SECTION_REGISTRY` [orchestrator/modules/context/sections/__init__.py:28-45]().

| Priority | Section Name | Role | File Reference |
| :--- | :--- | :--- | :--- |
| **1** | `identity` | Agent name, role, and core persona | [identity.py:29]() |
| **2** | `onboarding` | Mission Zero setup (Empty workspace flow) | [onboarding.py:42-43]() |
| **2** | `task_context` | Current active task details | [task_context.py:39]() |
| **3** | `agent_roster` | Available agents for coordination | [agent_roster.py:28]() |
| **3** | `mission_context`| High-level mission goals/DAG state | [mission_context.py:18]() |
| **4** | `skills` | `SKILL.md` content and tool instructions | [skills.py:23]() |
| **5** | `composio` | External app descriptions (OAuth apps) | [composio.py:27]() |
| **5** | `plugins` | Plugin tier-1/tier-2 summaries | [plugins.py:32]() |
| **5** | `platform_actions`| System-level tools (e.g., `list_agents`) | [platform_actions.py:20]() |
| **6** | `memory` | User facts, session context, and logs | [memory.py:17]() |
| **8** | `datetime_context`| Temporal grounding (current time) | [datetime_context.py:14]() |
| **9** | `conversation` | Formatted chat history | [conversation.py:12]() |
| **10** | `custom` | Dynamic overrides or one-off prompts | [custom.py:13]() |

**Sources**:
- [orchestrator/modules/context/sections/__init__.py:28-45]()
- [orchestrator/modules/context/budget.py:121-123]()
- [orchestrator/modules/context/sections/agent_roster.py:28]()
- [orchestrator/modules/context/sections/plugins.py:31-33]()
- [orchestrator/modules/context/sections/composio.py:26-28]()

---

## Token Estimation

The system uses a `TokenEstimator` utility [orchestrator/modules/context/estimator.py:22]() for rapid calculations without requiring a full tokenizer in the critical path.

- **Heuristic**: Roughly 4 characters per token estimate used for truncation [orchestrator/modules/context/budget.py:83]().
- **Usage**: `TokenBudgetManager` uses it to re-calculate estimates after truncating section content to fit `max_tokens` [orchestrator/modules/context/budget.py:84-85]().
- **Safety**: If the total tokens remain over budget after all droppable sections (P3+) are removed, a warning is logged, but priority 1-2 sections are preserved to maintain agent coherence [orchestrator/modules/context/budget.py:137-143]().

**Sources**:
- [orchestrator/modules/context/budget.py:22]()
- [orchestrator/modules/context/budget.py:82-92]()
- [orchestrator/modules/context/budget.py:137-143]()

---

## Integration in Prompt Building

In the `ContextService`, budget management is applied after all sections have been rendered.

1. `ContextService.build_context()` determines the `ContextMode` and retrieves the corresponding `ModeConfig` [orchestrator/modules/context/modes.py:35-134]().
2. It retrieves the corresponding `TokenBudget` from `DEFAULT_BUDGETS` [orchestrator/modules/context/budget.py:152-202]().
3. All sections required by the mode are rendered into `RenderedSection` objects [orchestrator/modules/context/budget.py:43-50]().
4. `TokenBudgetManager.allocate()` is called to trim or drop sections based on priority [orchestrator/modules/context/budget.py:66-70]().
5. The final `ContextResult` includes the list of `sections_included` and `sections_trimmed` for observability [orchestrator/tests/test_context/test_service.py:202-204]().

**Sources**:
- [orchestrator/modules/context/modes.py:35-134]()
- [orchestrator/modules/context/budget.py:152-202]()
- [orchestrator/modules/context/budget.py:66-75]()
- [orchestrator/tests/test_context/test_service.py:192-204]()

---