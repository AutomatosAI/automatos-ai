# Context Service

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/123-HARNESS-PATTERN-ADOPTION.md](docs/PRDS/123-HARNESS-PATTERN-ADOPTION.md)
- [orchestrator/consumers/chatbot/integration.py](orchestrator/consumers/chatbot/integration.py)
- [orchestrator/consumers/chatbot/prompt_analyzer.py](orchestrator/consumers/chatbot/prompt_analyzer.py)
- [orchestrator/consumers/chatbot/smart_orchestrator.py](orchestrator/consumers/chatbot/smart_orchestrator.py)
- [orchestrator/modules/agents/queries.py](orchestrator/modules/agents/queries.py)
- [orchestrator/modules/context/budget.py](orchestrator/modules/context/budget.py)
- [orchestrator/modules/context/modes.py](orchestrator/modules/context/modes.py)
- [orchestrator/modules/context/sections/__init__.py](orchestrator/modules/context/sections/__init__.py)
- [orchestrator/modules/context/sections/agent_roster.py](orchestrator/modules/context/sections/agent_roster.py)
- [orchestrator/modules/context/sections/composio.py](orchestrator/modules/context/sections/composio.py)
- [orchestrator/modules/context/sections/identity.py](orchestrator/modules/context/sections/identity.py)
- [orchestrator/modules/context/sections/memory.py](orchestrator/modules/context/sections/memory.py)
- [orchestrator/modules/context/sections/mission_context.py](orchestrator/modules/context/sections/mission_context.py)
- [orchestrator/modules/context/sections/onboarding.py](orchestrator/modules/context/sections/onboarding.py)
- [orchestrator/modules/context/sections/platform_actions.py](orchestrator/modules/context/sections/platform_actions.py)
- [orchestrator/modules/context/sections/plugins.py](orchestrator/modules/context/sections/plugins.py)
- [orchestrator/modules/context/sections/skills.py](orchestrator/modules/context/sections/skills.py)
- [orchestrator/modules/context/sections/task_context.py](orchestrator/modules/context/sections/task_context.py)
- [orchestrator/modules/context/sections/tools.py](orchestrator/modules/context/sections/tools.py)
- [orchestrator/tests/test_context/__init__.py](orchestrator/tests/test_context/__init__.py)
- [orchestrator/tests/test_context/conftest.py](orchestrator/tests/test_context/conftest.py)
- [orchestrator/tests/test_context/test_budget_manager.py](orchestrator/tests/test_context/test_budget_manager.py)
- [orchestrator/tests/test_context/test_estimator.py](orchestrator/tests/test_context/test_estimator.py)
- [orchestrator/tests/test_context/test_identity_section.py](orchestrator/tests/test_context/test_identity_section.py)
- [orchestrator/tests/test_context/test_memory_section.py](orchestrator/tests/test_context/test_memory_section.py)
- [orchestrator/tests/test_context/test_modes.py](orchestrator/tests/test_context/test_modes.py)
- [orchestrator/tests/test_context/test_service.py](orchestrator/tests/test_context/test_service.py)

</details>



## Purpose and Scope

`ContextService` is the unified prompt-building layer for the Automatos AI platform. It provides a single entry point for assembling LLM contexts, consolidating fragmented code paths into a modular system. It manages the assembly of system prompts, message history formatting, and tool schema loading based on specific execution **modes** (e.g., chatbot, task execution, heartbeat, coordinator). [orchestrator/modules/context/service.py:1-19]()

This service ensures that every LLM call across the platform benefits from consistent identity injection, skill awareness, platform action discovery, and memory retrieval while strictly adhering to token budgets through a priority-based trimming system. [orchestrator/modules/context/budget.py:1-9]()

**Sources:** [orchestrator/modules/context/service.py:1-19](), [orchestrator/modules/context/budget.py:1-9]()

---

## Architecture Overview

The Context Service operates by orchestrating several specialized components to produce a `ContextResult`.

```mermaid
graph TB
    subgraph "ContextService [orchestrator/modules/context/service.py]"
        BuildContext["build_context()<br/>mode, agent, workspace_id"]
    end
    
    subgraph "Configuration Layer"
        ModeConfig["ModeConfig [orchestrator/modules/context/modes.py]<br/>sections: list[str]<br/>tool_loading: str<br/>personality: bool"]
        ContextMode["ContextMode [orchestrator/modules/context/modes.py]<br/>CHATBOT, TASK_EXECUTION,<br/>COORDINATOR, etc."]
    end
    
    subgraph "Section Assembly [orchestrator/modules/context/sections/]"
        Registry["SECTION_REGISTRY [orchestrator/modules/context/sections/__init__.py]"]
        P1["Priority 1: IdentitySection"]
        P2["Priority 2: TaskContextSection / MissionContextSection"]
        P4["Priority 4: SkillsSection"]
        P5["Priority 5: PlatformActionsSection"]
        P6["Priority 6: MemorySection"]
        P8["Priority 8: DatetimeContextSection"]
    end
    
    subgraph "Budget & Tools"
        TBM["TokenBudgetManager [orchestrator/modules/context/budget.py]<br/>Sequential allocation<br/>Priority-based trimming"]
        Tools["ToolsSection [orchestrator/modules/context/sections/tools.py]<br/>Loading Strategies:<br/>full, filtered, dispatcher_only"]
    end

    subgraph "Output"
        Result["ContextResult [orchestrator/modules/context/result.py]<br/>system_prompt<br/>messages<br/>tools"]
    end
    
    BuildContext --> ContextMode
    ContextMode --> ModeConfig
    ModeConfig --> Registry
    Registry --> P1 & P2 & P4 & P5 & P6 & P8
    P1 & P2 & P4 & P5 & P6 & P8 -- RenderedSection --> TBM
    TBM -- IncludedSections --> Result
    ModeConfig --> Tools
    Tools --> Result
```

**Diagram 1: Context Service Internal Flow**

### Core Components
1.  **ContextMode**: An enum defining the high-level intent of the LLM call, such as `CHATBOT`, `TASK_EXECUTION`, or `NL2SQL`. [orchestrator/modules/context/modes.py:13-22]()
2.  **ModeConfig**: A declarative configuration for each mode, specifying which sections to include and how tools should be loaded. [orchestrator/modules/context/modes.py:26-33]()
3.  **Section Registry**: A mapping of string identifiers to `BaseSection` subclasses like `IdentitySection` or `SkillsSection`. [orchestrator/modules/context/sections/__init__.py:27-45]()
4.  **TokenBudgetManager**: Logic for trimming or dropping sections based on priority when context exceeds model limits. [orchestrator/modules/context/budget.py:53-64]()

**Sources:** [orchestrator/modules/context/modes.py:13-33](), [orchestrator/modules/context/sections/__init__.py:27-45](), [orchestrator/modules/context/budget.py:53-64]()

---

## Context Modes

Modes define the "flavor" of the prompt. For example, `CHATBOT` mode enables personality and conversation history, while `COORDINATOR` mode focuses on mission goals and agent rosters. [orchestrator/modules/context/modes.py:35-49]()

| Mode | Personality | Tool Loading | Key Sections |
| :--- | :--- | :--- | :--- |
| `CHATBOT` | `True` | `filtered` | identity, skills, memory, platform_actions, conversation |
| `TASK_EXECUTION` | `False` | `full` | identity, skills, task_context, platform_actions, conversation |
| `COORDINATOR` | `False` | `full` | identity, mission_context, agent_roster, platform_actions |
| `HEARTBEAT_ORCHESTRATOR`| `False` | `dispatcher_only` | identity, skills, platform_actions, task_context |
| `RECIPE` | `False` | `full` | identity, skills, memory, playbook_context |

For a full list of modes and their specific configurations, see [Context Modes](#4.1).

**Sources:** [orchestrator/modules/context/modes.py:35-134]()

---

## Section Priority System

The service uses a 10-tier priority system (1 = highest importance). When the `TokenBudgetManager` encounters a context that is too large, it drops sections starting from the highest priority number (lowest importance). [orchestrator/modules/context/budget.py:110-115]()

### Priority Guarantees
*   **Priority 1-2**: Critical sections (e.g., `IdentitySection` at P1, `TaskContextSection` or `MissionContextSection` at P2) are **NEVER** dropped, even if the budget is exceeded. [orchestrator/modules/context/budget.py:121-123]()
*   **Priority 3-10**: Sections like `SkillsSection` (P4), `PlatformActionsSection` (P5), or `MemorySection` (P6) may be dropped to save space. [orchestrator/modules/context/budget.py:118-132]()

For details on how each section is rendered, see [Section Priority System](#4.2) and [Section Types](#4.4).

**Sources:** [orchestrator/modules/context/budget.py:110-132](), [orchestrator/modules/context/sections/identity.py:69](), [orchestrator/modules/context/sections/task_context.py:27](), [orchestrator/modules/context/sections/skills.py:26]()

---

## Token Budget Management

Each mode has a `TokenBudget` defining the total allowed tokens and reservations for messages and responses. [orchestrator/modules/context/budget.py:152-157]()

```mermaid
graph LR
    subgraph "TokenBudget Calculation [orchestrator/modules/context/budget.py]"
        Total["Total Budget (e.g. 128k)"]
        ReservedResp["Reserved for Response"]
        ReservedMsg["Reserved for Messages"]
        Available["Available for Sections"]
        
        Total --- ReservedResp
        Total --- ReservedMsg
        Total --> Available
    end
    
    subgraph "Trimming Logic"
        Capping["1. Apply Section max_tokens<br/>(Truncation)"]
        Dropping["2. Drop by Priority<br/>(P10 -> P3)"]
    end

    Available --> Capping
    Capping --> Dropping
```

**Diagram 2: Token Budget Allocation**

The `TokenBudgetManager` first applies local `max_tokens` caps defined on individual sections (e.g., `IdentitySection` is capped at 600 tokens) via truncation. [orchestrator/modules/context/sections/identity.py:70-71]() If the total is still over budget, it begins dropping entire sections based on the priority system. [orchestrator/modules/context/budget.py:80-132]()

For more on allocation strategies, see [Token Budget Management](#4.3).

**Sources:** [orchestrator/modules/context/budget.py:26-40](), [orchestrator/modules/context/budget.py:80-132](), [orchestrator/modules/context/sections/identity.py:70-71]()

---

## Personality and Identity

The `IdentitySection` manages how an agent perceives itself. In `CHATBOT` mode, it uses `AutomatosPersonality` to generate greetings and response rules. [orchestrator/modules/context/sections/identity.py:122-133]() For non-chatbot modes, it provides a professional, role-based identity. [orchestrator/modules/context/sections/identity.py:87-103]()

The personality system supports:
*   **Presets**: `friendly`, `professional`, and `technical`. [orchestrator/modules/context/sections/identity.py:21-46]()
*   **Communication Styles**: `concise`, `balanced`, and `detailed`. [orchestrator/modules/context/sections/identity.py:48-52]()

**Sources:** [orchestrator/modules/context/sections/identity.py:21-162]()

---

## Tool Loading and Platform Actions

The service handles tool schema preparation through the `ToolsSection` and `PlatformActionsSection`.
*   **PlatformActionsSection**: Injects a markdown catalog of available `platform_execute` actions like `list_agents` or `query_data`. [orchestrator/modules/context/sections/platform_actions.py:23-35]()
*   **SkillsSection**: Loads the `SKILL.md` content associated with an agent's active skills. [orchestrator/modules/context/sections/skills.py:18-27]()
*   **ToolsSection**: Implements strategies like `FULL`, `FILTERED` (via `SmartToolRouter`), and `DISPATCHER_ONLY`. [orchestrator/modules/context/sections/tools.py:32-38]()

```mermaid
graph TD
    subgraph "Natural Language Space"
        UserQuery["User Query: 'How do I create an agent?'"]
    end

    subgraph "Code Entity Space [orchestrator/modules/context/sections/tools.py]"
        ToolsSec["ToolsSection.load_tools()"]
        Strategy["ToolLoadingStrategy.FILTERED"]
        Router["SmartToolRouter.route() [orchestrator/consumers/chatbot/smart_tool_router.py]"]
        Registry["ActionRegistry.to_dispatcher_schema() [orchestrator/modules/tools/discovery/action_registry.py]"]
    end

    UserQuery --> ToolsSec
    ToolsSec --> Strategy
    Strategy --> Router
    Router --> Registry
    Registry --> Result["ContextResult.tools (platform_create_agent)"]
```

**Diagram 3: Natural Language to Tool Schema Mapping**

**Sources:** [orchestrator/modules/context/sections/platform_actions.py:23-35](), [orchestrator/modules/context/sections/skills.py:18-27](), [orchestrator/modules/context/sections/tools.py:32-193]()

---

## Integration and Usage

The `ContextService` is the standard way to prepare prompts for any consumer in the system, replacing fragmented logic in chat, recipes, and heartbeats. [orchestrator/modules/context/service.py:1-19]()

```python
# Standard Usage Pattern in SmartChatOrchestrator
from modules.context import ContextService, ContextMode

context = await ContextService(self._db_session).build_context(
    mode=ContextMode.CHATBOT,
    agent=agent,
    workspace_id=self.workspace_id,
    messages=messages,
    widget_mode=self.widget_mode
)
# context.system_prompt contains the assembled sections
# context.tools contains the loaded schemas
```

For details on the migration from legacy prompt builders, see [Migration & Integration](#4.5).

**Sources:** [orchestrator/consumers/chatbot/smart_orchestrator.py:190-200](), [orchestrator/modules/context/service.py:1-19]()

---