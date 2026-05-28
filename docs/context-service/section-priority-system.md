# Section Priority System

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/123-HARNESS-PATTERN-ADOPTION.md](docs/PRDS/123-HARNESS-PATTERN-ADOPTION.md)
- [orchestrator/modules/context/budget.py](orchestrator/modules/context/budget.py)
- [orchestrator/modules/context/modes.py](orchestrator/modules/context/modes.py)
- [orchestrator/modules/context/sections/__init__.py](orchestrator/modules/context/sections/__init__.py)
- [orchestrator/modules/context/sections/agent_roster.py](orchestrator/modules/context/sections/agent_roster.py)
- [orchestrator/modules/context/sections/composio.py](orchestrator/modules/context/sections/composio.py)
- [orchestrator/modules/context/sections/memory.py](orchestrator/modules/context/sections/memory.py)
- [orchestrator/modules/context/sections/mission_context.py](orchestrator/modules/context/sections/mission_context.py)
- [orchestrator/modules/context/sections/onboarding.py](orchestrator/modules/context/sections/onboarding.py)
- [orchestrator/modules/context/sections/platform_actions.py](orchestrator/modules/context/sections/platform_actions.py)
- [orchestrator/modules/context/sections/playbook_context.py](orchestrator/modules/context/sections/playbook_context.py)
- [orchestrator/modules/context/sections/plugins.py](orchestrator/modules/context/sections/plugins.py)
- [orchestrator/scripts/eval/tool_routing/seed_telemetry.py](orchestrator/scripts/eval/tool_routing/seed_telemetry.py)
- [orchestrator/tests/test_context/__init__.py](orchestrator/tests/test_context/__init__.py)
- [orchestrator/tests/test_context/conftest.py](orchestrator/tests/test_context/conftest.py)
- [orchestrator/tests/test_context/test_budget_manager.py](orchestrator/tests/test_context/test_budget_manager.py)
- [orchestrator/tests/test_context/test_estimator.py](orchestrator/tests/test_context/test_estimator.py)
- [orchestrator/tests/test_context/test_identity_section.py](orchestrator/tests/test_context/test_identity_section.py)
- [orchestrator/tests/test_context/test_memory_section.py](orchestrator/tests/test_context/test_memory_section.py)
- [orchestrator/tests/test_context/test_modes.py](orchestrator/tests/test_context/test_modes.py)
- [orchestrator/tests/test_context/test_service.py](orchestrator/tests/test_context/test_service.py)
- [orchestrator/tests/test_platform_actions_section.py](orchestrator/tests/test_platform_actions_section.py)
- [orchestrator/tests/test_platform_actions_section_graph.py](orchestrator/tests/test_platform_actions_section_graph.py)

</details>



The **Section Priority System** is the `ContextService`'s mechanism for ensuring critical agent identity, capabilities, and tools are never omitted from the system prompt — even when token budgets are tight. It assigns a **priority (1-10)** to each context section and uses a `TokenBudgetManager` to selectively trim lower-priority sections (P6-10) while guaranteeing high-priority sections (P1-5) always render in full.

This system replaces fragmented prompt-building paths with unified, predictable context assembly. For details on the overall ContextService architecture, see [Context Service]().

---

## Overview

Every context section inherits from `BaseSection` [orchestrator/modules/context/sections/base.py:10-10]() and declares a **priority** integer. When the assembled prompt exceeds the mode's token budget, the `TokenBudgetManager` [orchestrator/modules/context/budget.py:53-53]() trims sections in **reverse priority order** (highest priority number = lowest importance) until the prompt fits.

**Key guarantee:** Priorities 1-2 are **never dropped**, even if the total remains over budget [orchestrator/modules/context/budget.py:8-9](), [orchestrator/modules/context/budget.py:121-123]().

**Sources:** [orchestrator/modules/context/budget.py:1-64](), [orchestrator/modules/context/sections/base.py:10-30]()

---

## Priority Tier Breakdown

The system organizes context into three functional tiers based on their necessity for agent operation.

```mermaid
graph TB
    subgraph "Critical Tier - Never Dropped"
        P1["P1: IdentitySection<br/>Agent name, role, persona"]
        P2["P2: Task/Recipe/Mission Context<br/>Current goal/step instructions"]
    end
    
    subgraph "Important Tier - High Priority"
        P3["P3: ToolsSection<br/>Tool schemas (separate from prompt)"]
        P4["P4: SkillsSection<br/>SKILL.md content"]
        P5["P5: PlatformActionsSection<br/>Platform tool summaries"]
    end
    
    subgraph "Trimmable Tier - Dropped if Over Budget"
        P6["P6: MemorySection<br/>User memories + daily logs"]
        P8["P8: DatetimeContextSection<br/>Current UTC time"]
        P9["P9: ConversationSection<br/>Chat history formatting"]
        P10["P10: CustomSection<br/>Extra metadata"]
    end
    
    Budget["TokenBudgetManager.allocate()"]
    
    P1 --> Budget
    P2 --> Budget
    P3 --> Budget
    P4 --> Budget
    P5 --> Budget
    P6 --> Budget
    P8 --> Budget
    P9 --> Budget
    P10 --> Budget
    
    Budget -->|"Fits budget"| FullPrompt["ContextResult<br/>All sections included"]
    Budget -->|"Exceeds budget"| Trim["Drop P10→P6<br/>until budget met"]
    
    Trim --> TrimmedPrompt["ContextResult<br/>P1-P2 guaranteed"]
```

**Sources:** [orchestrator/modules/context/budget.py:53-145](), [orchestrator/modules/context/sections/identity.py:158-160](), [orchestrator/modules/context/sections/platform_actions.py:39-40](), [orchestrator/modules/context/sections/memory.py:48-49](), [orchestrator/modules/context/sections/composio.py:26-27]()

---

## Section Priority Registry

The `SECTION_REGISTRY` maps section names to their implementation classes [orchestrator/modules/context/sections/__init__.py:28-45]().

| Priority | Section Name | Class | Max Tokens (Default) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `identity` | `IdentitySection` | None | Agent name, role, persona, and workspace context [orchestrator/tests/test_context/test_identity_section.py:158-164]() |
| **2** | `task_context` | `TaskContextSection` | - | Current task description and board metadata [orchestrator/modules/context/sections/__init__.py:39-39]() |
| **2** | `playbook_context` | `PlaybookContextSection` | - | Current recipe step and execution context [orchestrator/modules/context/sections/__init__.py:40-40]() |
| **2** | `mission_context` | `MissionContextSection` | - | Goal decomposition and DAG task graph context [orchestrator/modules/context/sections/__init__.py:35-35]() |
| **2** | `onboarding` | `OnboardingSection` | - | Mission Zero onboarding flow for new workspaces [orchestrator/modules/context/sections/__init__.py:36-36]() |
| **3** | `tools` | `ToolsSection` | None | Manages tool loading (not rendered in prompt text) [orchestrator/modules/context/sections/__init__.py:38-38]() |
| **4** | `skills` | `SkillsSection` | - | SKILL.md content assigned to the agent [orchestrator/modules/context/sections/__init__.py:23-23]() |
| **5** | `platform_actions` | `PlatformActionsSection` | Configurable | Catalog of available `platform_execute` actions [orchestrator/modules/context/sections/platform_actions.py:39-41]() |
| **5** | `composio` | `ComposioSection` | Configurable | Connected external apps (Composio) descriptions [orchestrator/modules/context/sections/composio.py:26-28]() |
| **6** | `memory` | `MemorySection` | Configurable | User memories, session context, and daily logs [orchestrator/modules/context/sections/memory.py:48-50]() |
| **8** | `datetime_context`| `DatetimeContextSection` | - | Current system date and time [orchestrator/modules/context/sections/__init__.py:41-41]() |
| **9** | `conversation` | `ConversationSection` | - | Formatting hints for message history [orchestrator/modules/context/sections/__init__.py:43-43]() |
| **10** | `custom` | `CustomSection` | - | Arbitrary key-value metadata [orchestrator/modules/context/sections/__init__.py:44-44]() |

**Sources:** [orchestrator/modules/context/sections/__init__.py:28-45](), [orchestrator/modules/context/budget.py:121-123](), [orchestrator/modules/context/sections/platform_actions.py:39-41]()

---

## Token Budget Allocation Algorithm

The `TokenBudgetManager.allocate()` method implements a three-step trimming algorithm to fit sections into the `available_for_sections` budget [orchestrator/modules/context/budget.py:37-40]().

### 1. Per-Section Capping
Every `RenderedSection` is first truncated to its own `max_tokens` limit if one is defined [orchestrator/modules/context/budget.py:81-83](). This ensures no single section consumes the entire budget before the priority-based dropping begins. The `TokenEstimator` [orchestrator/modules/context/budget.py:22-22]() provides approximate counts for this calculation.

### 2. Priority-Based Dropping
If the total token estimate still exceeds the budget, the manager sorts sections by priority in descending order (highest number = lowest importance) [orchestrator/modules/context/budget.py:113-115](). It iterates through this list and drops sections until the budget is met, with the strict exception that **Priority 1 and 2 sections are never dropped** [orchestrator/modules/context/budget.py:121-123]().

### 3. Order Preservation
After trimming, the manager rebuilds the section list to preserve the original relative order of the remaining sections, ensuring logical flow in the final system prompt [orchestrator/modules/context/budget.py:134-135]().

**Sources:** [orchestrator/modules/context/budget.py:66-145]()

---

## Code Entity Mapping

The following diagram maps the logical priority system to the specific code classes and their interactions within the `ContextService`.

```mermaid
classDiagram
    class ContextService {
        +build_context(mode, agent, workspace_id)
    }
    class TokenBudgetManager {
        +allocate(sections, budget)
    }
    class BaseSection {
        <<abstract>>
        +str name
        +int priority
        +int max_tokens
        +render(ctx)
    }
    class IdentitySection {
        +priority = 1
    }
    class PlatformActionsSection {
        +priority = 5
        +render(ctx)
    }
    class MemorySection {
        +priority = 6
        +render(ctx)
    }
    class RenderedSection {
        +str name
        +int priority
        +str content
        +int token_estimate
    }

    ContextService --> TokenBudgetManager : uses
    ContextService ..> BaseSection : instantiates via SECTION_REGISTRY
    TokenBudgetManager ..> RenderedSection : filters and trims
    BaseSection <|-- IdentitySection
    BaseSection <|-- PlatformActionsSection
    BaseSection <|-- MemorySection
```

**Sources:** [orchestrator/modules/context/service.py:27-27](), [orchestrator/modules/context/budget.py:43-51](), [orchestrator/modules/context/sections/base.py:18-25](), [orchestrator/modules/context/sections/__init__.py:28-45]()

---

## Mode-Specific Budget Configurations

Budgets are defined per `ContextMode` in `DEFAULT_BUDGETS` [orchestrator/modules/context/budget.py:152-152](). Each mode calculates its available space by subtracting reserved tokens for the response and message history from the total context window.

| Mode | Total Budget | Reserved (Response) | Reserved (Messages) |
| :--- | :--- | :--- | :--- |
| `CHATBOT` | 128,000 | 4,096 | 60,000 |
| `TASK_EXECUTION` | 128,000 | 4,096 | 20,000 |
| `HEARTBEAT_AGENT`| 128,000 | 4,096 | 0 |
| `COORDINATOR` | 131,072 | 4,096 | 0 |

**Sources:** [orchestrator/modules/context/budget.py:152-202](), [orchestrator/modules/context/modes.py:35-134]()

---

## Implementation Details

### Identity Protection
The `IdentitySection` [orchestrator/modules/context/sections/identity.py:9-10]() handles agent name, role, and persona rendering. It includes logic for custom personas and database-stored personas [orchestrator/tests/test_context/test_identity_section.py:81-112](). Because it is Priority 1, these core identity traits are never sacrificed during budget trimming.

### Platform Action Cataloging
`PlatformActionsSection` [orchestrator/modules/context/sections/platform_actions.py:30-30]() (Priority 5) provides a catalog of available `platform_execute` actions. It supports **Semantic Tool Routing** (PRD-138) where actions are ranked by similarity to the user query using `ActionSemanticIndex` [orchestrator/modules/context/sections/platform_actions.py:7-12](). It also implements **Graph-based Routing** (PRD-139) to rank action chains for complex multi-step tasks [orchestrator/modules/context/sections/platform_actions.py:68-72]().

### Memory Injection
`MemorySection` [orchestrator/modules/context/sections/memory.py:32-32]() (Priority 6) injects user memories and daily logs. For `CHATBOT` mode, it attempts to use the **Context Router** (PRD-79) for high-fidelity retrieval including session summaries and temporal awareness [orchestrator/modules/context/sections/memory.py:7-9](). If the router fails, it falls back to `SmartMemoryManager` [orchestrator/modules/context/sections/memory.py:41-43]().

### External App Integration
`ComposioSection` [orchestrator/modules/context/sections/composio.py:19-19]() (Priority 5) renders a markdown block of connected external apps. It queries `AgentAppAssignment` for active apps and retrieves descriptions from `ComposioAppCache` [orchestrator/modules/context/sections/composio.py:57-67]().

**Sources:** [orchestrator/modules/context/sections/platform_actions.py:1-193](), [orchestrator/modules/context/sections/memory.py:1-180](), [orchestrator/modules/context/sections/composio.py:1-104](), [orchestrator/modules/context/budget.py:1-203]()

---