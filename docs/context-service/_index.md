# Context Service

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/consumers/chatbot/prompt_analyzer.py](orchestrator/consumers/chatbot/prompt_analyzer.py)
- [orchestrator/core/attachment_refs.py](orchestrator/core/attachment_refs.py)
- [orchestrator/modules/attachments/extract.py](orchestrator/modules/attachments/extract.py)
- [orchestrator/modules/attachments/resolver.py](orchestrator/modules/attachments/resolver.py)
- [orchestrator/modules/context/__init__.py](orchestrator/modules/context/__init__.py)
- [orchestrator/modules/context/budget.py](orchestrator/modules/context/budget.py)
- [orchestrator/modules/context/modes.py](orchestrator/modules/context/modes.py)
- [orchestrator/modules/context/planning.py](orchestrator/modules/context/planning.py)
- [orchestrator/modules/context/sections/__init__.py](orchestrator/modules/context/sections/__init__.py)
- [orchestrator/modules/context/sections/agent_roster.py](orchestrator/modules/context/sections/agent_roster.py)
- [orchestrator/modules/context/sections/composio.py](orchestrator/modules/context/sections/composio.py)
- [orchestrator/modules/context/sections/conversation.py](orchestrator/modules/context/sections/conversation.py)
- [orchestrator/modules/context/sections/field_memory.py](orchestrator/modules/context/sections/field_memory.py)
- [orchestrator/modules/context/sections/planning_history.py](orchestrator/modules/context/sections/planning_history.py)
- [orchestrator/modules/context/sections/planning_knowledge.py](orchestrator/modules/context/sections/planning_knowledge.py)
- [orchestrator/modules/context/sections/playbook_context.py](orchestrator/modules/context/sections/playbook_context.py)
- [orchestrator/modules/context/sections/plugins.py](orchestrator/modules/context/sections/plugins.py)
- [orchestrator/modules/context/service.py](orchestrator/modules/context/service.py)
- [orchestrator/tests/test_context/test_modes.py](orchestrator/tests/test_context/test_modes.py)
- [orchestrator/tests/test_context/test_service.py](orchestrator/tests/test_context/test_service.py)
- [orchestrator/tests/test_prd164_planning_pack.py](orchestrator/tests/test_prd164_planning_pack.py)
- [orchestrator/tests/test_prd179_field_read.py](orchestrator/tests/test_prd179_field_read.py)
- [orchestrator/tests/test_prd223_attachment_refs.py](orchestrator/tests/test_prd223_attachment_refs.py)

</details>



## Purpose and Scope

`ContextService` serves as the unified prompt-building layer for the Automatos AI platform [orchestrator/modules/context/service.py:1-6](). It provides a single entry point for assembling LLM contexts, replacing fragmented prompt construction code paths with a modular, mode-based architecture [orchestrator/modules/context/service.py:86-91](). It coordinates section rendering, token budget allocation, tool loading, and conversation message formatting into an immutable `ContextResult` [orchestrator/modules/context/service.py:111-118]().

Sources: [orchestrator/modules/context/service.py:1-6](), [orchestrator/modules/context/service.py:86-91](), [orchestrator/modules/context/service.py:111-118]()

---

## Architecture Overview

The Context Service orchestrates configuration lookup, section rendering, token budgeting, and tool resolution through dedicated internal modules [orchestrator/modules/context/service.py:96-183]().

Title: Context Service Natural Language to Code Entity Mapping
```mermaid
graph TB
    subgraph "NaturalLanguageSpace ["User Prompt & Agent Goal"]"
        NL_Chat["Chat / Conversation Request"]
        NL_Task["Autonomous Task Execution"]
        NL_Heartbeat["Agent / Orchestrator Tick"]
    end

    subgraph "CodeEntitySpace ["Context Pipeline classes"]"
        CS["ContextService<br/>[orchestrator/modules/context/service.py:86]"]
        ModeCfg["ModeConfig<br/>[orchestrator/modules/context/modes.py:113]"]
        Registry["SECTION_REGISTRY<br/>[orchestrator/modules/context/sections/__init__.py:31]"]
        TBM["TokenBudgetManager<br/>[orchestrator/modules/context/budget.py:51]"]
        Res["ContextResult<br/>[orchestrator/modules/context/result.py]"]
    end

    NL_Chat -->|build_context| CS
    NL_Task -->|build_context| CS
    NL_Heartbeat -->|build_context| CS

    CS -->|lookup| ModeCfg
    ModeCfg -->|instantiate| Registry
    Registry -->|render & allocate| TBM
    TBM -->|immutable result| Res
```

Sources: [orchestrator/modules/context/service.py:86-183](), [orchestrator/modules/context/modes.py:113-120](), [orchestrator/modules/context/sections/__init__.py:31-51](), [orchestrator/modules/context/budget.py:51-62]()

---

## 4.1. Context Modes

Modes define the operational context and declarative section requirements for an LLM call via `ContextMode` and `ModeConfig` [orchestrator/modules/context/modes.py:13-120](). Supported modes include `CHATBOT`, `TASK_EXECUTION`, `HEARTBEAT_ORCHESTRATOR`, `HEARTBEAT_AGENT`, `RECIPE`, `ROUTER`, `ORCHESTRATOR_STAGE`, and `NL2SQL` [orchestrator/modules/context/modes.py:13-24](). Each configuration controls whether conversational personality is injected and how tools are filtered or loaded [orchestrator/modules/context/modes.py:113-120]().

For detailed mode definitions, tool pruning rules (`EXECUTION_ONLY_TOOLS`), and configuration properties, see [Context Modes](#4.1).

Sources: [orchestrator/modules/context/modes.py:13-120]()

---

## 4.2. Section Priority System

Sections are assigned execution priorities that determine their importance during token budget trimming [orchestrator/modules/context/budget.py:51-62](). Critical sections like `IdentitySection` (Priority 1) and task or mission context (Priority 2) are strictly protected and never dropped [orchestrator/modules/context/budget.py:124-125](). Lower-priority sections such as skills, tools, memory, and datetime context are evaluated and dropped sequentially if token limits are exceeded [orchestrator/modules/context/budget.py:112-134]().

For a complete breakdown of section priorities, execution rules, and how critical context is protected, see [Section Priority System](#4.2).

Sources: [orchestrator/modules/context/budget.py:51-62](), [orchestrator/modules/context/budget.py:112-134]()

---

## 4.3. Token Budget Management

Token budget management is handled by `TokenBudgetManager`, which enforces per-mode budgets (`TokenBudget`) encompassing total window sizes, response reservations, and message reservations [orchestrator/modules/context/budget.py:24-62](). It applies token-boundary truncation caps (`max_tokens`) to individual sections and performs priority-based dropping when necessary [orchestrator/modules/context/budget.py:77-134](). It also separates cache-stable system prompt prefixes from volatile per-turn sections to optimize prompt caching [orchestrator/modules/context/service.py:66-81]().

For detailed budget allocations, sequential section allocation algorithms, and caching strategies, see [Token Budget Management](#4.3).

Sources: [orchestrator/modules/context/budget.py:24-148](), [orchestrator/modules/context/service.py:66-81]()

---

## 4.4. Section Types

The platform implements a diverse registry of modular sections (`SECTION_REGISTRY`) under `orchestrator/modules/context/sections/` [orchestrator/modules/context/sections/__init__.py:31-51](). These include core components like `IdentitySection`, `SkillsSection`, `ToolsSection`, `MemorySection`, `GraphSection`, `PlanningHistorySection`, and `OnboardingSection` [orchestrator/modules/context/sections/__init__.py:31-51](). Each section encapsulates its own rendering logic, error handling, and token limits [orchestrator/modules/context/sections/agent_roster.py:31-38]().

For comprehensive documentation on each section class, its implementation, and rendering behavior, see [Section Types](#4.4).

Sources: [orchestrator/modules/context/sections/__init__.py:31-51](), [orchestrator/modules/context/sections/agent_roster.py:31-38]()

---

## 4.5. Migration & Integration

`ContextService` replaces fragmented prompt-building paths across chat handlers, channel adapters, and planning engines with a centralized architecture [orchestrator/modules/context/service.py:86-91](). It integrates attachment resolution via `AttachmentResolver` for multimodal inputs, page context injection, and conversation history formatting via `ConversationSection` [orchestrator/modules/context/service.py:54-59](), [orchestrator/modules/context/sections/conversation.py:29-54]().

For details on migration history, attachment handling workflows, and consumer integration patterns, see [Migration & Integration](#4.5).

Sources: [orchestrator/modules/context/service.py:54-91](), [orchestrator/modules/context/sections/conversation.py:29-54]()

---