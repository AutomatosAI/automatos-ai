# Agent Factory & Runtime

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/api/chat.py](orchestrator/api/chat.py)
- [orchestrator/api/routing.py](orchestrator/api/routing.py)
- [orchestrator/consumers/chatbot/auto.py](orchestrator/consumers/chatbot/auto.py)
- [orchestrator/consumers/chatbot/service.py](orchestrator/consumers/chatbot/service.py)
- [orchestrator/core/llm/manager.py](orchestrator/core/llm/manager.py)
- [orchestrator/core/routing/engine.py](orchestrator/core/routing/engine.py)
- [orchestrator/modules/agents/factory/agent_factory.py](orchestrator/modules/agents/factory/agent_factory.py)
- [orchestrator/modules/tools/discovery/platform_actions.py](orchestrator/modules/tools/discovery/platform_actions.py)
- [orchestrator/modules/tools/discovery/platform_executor.py](orchestrator/modules/tools/discovery/platform_executor.py)
- [orchestrator/scripts/setup_jira_trigger.py](orchestrator/scripts/setup_jira_trigger.py)
- [orchestrator/services/heartbeat_service.py](orchestrator/services/heartbeat_service.py)
- [orchestrator/services/page_context.py](orchestrator/services/page_context.py)
- [orchestrator/tests/test_prd221_page_context.py](orchestrator/tests/test_prd221_page_context.py)
- [orchestrator/tests/test_prd221_page_prior_tools.py](orchestrator/tests/test_prd221_page_prior_tools.py)

</details>



This document covers the **AgentFactory** system, which manages the complete lifecycle of agent instances from creation to execution. It handles LLM provider configuration, tool loading, and the execution loop that processes user prompts with tool calling support.

**Related pages:**
- For agent creation UI flows, see [Creating Agents](5.1)
- For agent configuration options, see [Agent Configuration](5.2)
- For LLM provider and credential management details, see [LLM Provider Management](5.6)
- For tool discovery and routing, see [Tools API Reference](8.7)

---

## Overview

**AgentFactory** is the core runtime execution engine for agents. It provides a pure execution layer where users define their own agent types while the orchestrator handles prompt engineering using Context Engineering [orchestrator/modules/agents/factory/agent_factory.py:1-11]().

| Capability | Description |
|------------|-------------|
| **Lifecycle Management** | Create, activate, hibernate, and retire agent instances [orchestrator/modules/agents/factory/agent_factory.py:53-60]() |
| **LLM Configuration** | 3-tier API key resolution (BYOK → credential store → env vars) [orchestrator/core/llm/manager.py:135-178]() |
| **Tool Integration** | Unified tool execution via `UnifiedToolExecutor` with single-source tool schemas [orchestrator/modules/agents/factory/agent_factory.py:44-46]() |
| **Prompt Assembly** | System prompt building from persona + plugins + skills [orchestrator/modules/agents/factory/agent_factory.py:107-148]() |
| **Execution Loop** | Multi-iteration tool loop with deduplication and loop prevention [orchestrator/consumers/chatbot/service.py:30-37]() |
| **Metrics Tracking** | Token usage, execution counts, success rates, and avg execution time [orchestrator/modules/agents/factory/agent_factory.py:179-185]() |

The factory maintains a registry of active agents in memory for fast execution without repeated database queries [orchestrator/modules/agents/factory/agent_factory.py:161-178]().

Sources: [orchestrator/modules/agents/factory/agent_factory.py:1-50](), [orchestrator/modules/agents/factory/agent_factory.py:155-192]()

---

## Agent Lifecycle States

Agents transition through well-defined lifecycle states managed by the `AgentLifecycle` enum [orchestrator/modules/agents/factory/agent_factory.py:53-60]():

Title: "Agent Lifecycle State Machine"
```mermaid
stateDiagram-v2
    direction LR
    [*] --> INITIALIZING: "AgentFactory.create_agent()"
    INITIALIZING --> ACTIVE: "activate_agent()"
    ACTIVE --> BUSY: "execute_with_prompt()"
    BUSY --> ACTIVE: "Execution complete"
    ACTIVE --> LEARNING: "AgentService.update_agent_learning()"
    LEARNING --> ACTIVE: "Learning complete"
    ACTIVE --> HIBERNATING: "Inactivity timeout"
    HIBERNATING --> ACTIVE: "Re-activation"
    ACTIVE --> RETIRED: "AgentFactory.retire_agent()"
    RETIRED --> [*]
```

### Lifecycle State Definitions

| State | Description | Triggers |
|-------|-------------|----------|
| `INITIALIZING` | Agent being created, LLM verification in progress | `create_agent()` called [orchestrator/modules/agents/factory/agent_factory.py:54]() |
| `ACTIVE` | Ready to accept tasks | `activate_agent()` completed [orchestrator/modules/agents/factory/agent_factory.py:55]() |
| `BUSY` | Currently executing a task | `execute_with_prompt()` running [orchestrator/modules/agents/factory/agent_factory.py:56]() |
| `LEARNING` | Undergoing training or optimization | Feedback loop or optimization job [orchestrator/modules/agents/factory/agent_factory.py:57]() |
| `HIBERNATING` | Inactive but preserved in memory | Configurable inactivity timeout [orchestrator/modules/agents/factory/agent_factory.py:58]() |
| `RETIRED` | Permanently deactivated | Manual retirement [orchestrator/modules/agents/factory/agent_factory.py:59]() |

Sources: [orchestrator/modules/agents/factory/agent_factory.py:51-60]()

---

## Core Data Structures

### ModelConfiguration

Complete LLM configuration for an agent, supporting per-agent model overrides (PRD-15) [orchestrator/modules/agents/factory/agent_factory.py:63-73]():

```python
@dataclass
class ModelConfiguration:
    provider: str              # "openai", "anthropic", "google", etc.
    model_id: str             # e.g., "gpt-4", "claude-3-opus-20240229"
    temperature: float = 0.7
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    fallback_model_id: Optional[str] = None  # Automatic fallback on failure
```

Sources: [orchestrator/modules/agents/factory/agent_factory.py:63-84]()

---

### AgentRuntime

Runtime representation of an active agent, cached in memory [orchestrator/modules/agents/factory/agent_factory.py:161-178]():

```python
@dataclass
class AgentRuntime:
    agent_id: int
    metadata: AgentMetadata
    llm_manager: LLMManager                    # Pre-configured LLM client
    lifecycle_state: AgentLifecycle
    execution_count: int = 0
    total_tokens_used: int = 0
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = {}   # avg_execution_time, success_rate
    tools: List[Dict[str, Any]] = []           # Composio app assignments
    tool_executor: Any = None                  # UnifiedToolExecutor instance
    is_byok: bool = False                      # Using workspace's own API key
    resolved_provider: str = ""
    workspace_id: Optional[Any] = None
```

Sources: [orchestrator/modules/agents/factory/agent_factory.py:161-178]()

---

## Natural Language Space to Code Entity Bridge

Title: "Natural Language to Code Entity Space - Chat and Agent Routing"
```mermaid
flowchart TD
    subgraph NaturalLanguageSpace ["Natural Language Space"]
        NL1["User sends chat message or prompt"]
        NL2["Determine complexity (Atom -> Organism)"]
        NL3["Resolve target agent or tools"]
    end

    subgraph CodeEntitySpace ["Code Entity Space"]
        CE1["api/chat.py:router POST /api/chat"]
        CE2["consumers/chatbot/auto.py:AutoBrain.assess()"]
        CE3["modules/agents/factory/agent_factory.py:AgentFactory.execute_with_prompt()"]
        CE4["modules/tools/execution/tool_loop.py:ToolLoopExecutor"]
        CE5["modules/tools/discovery/platform_executor.py:PlatformActionExecutor"]
    end

    NL1 --> CE1
    CE1 --> CE2
    CE2 --> CE3
    CE3 --> CE4
    CE4 --> CE5
```

Sources: [orchestrator/api/chat.py:1-37](), [orchestrator/consumers/chatbot/auto.py:7-22](), [orchestrator/modules/agents/factory/agent_factory.py:1-11]()

---

## Agent Creation & Activation

### LLM Configuration Resolution

The factory follows this priority order for LLM configuration:
1. **Agent's `model_config`**: If the agent has an explicit model defined in its metadata [orchestrator/modules/agents/factory/agent_factory.py:122-138]().
2. **System settings**: Fetched via `SystemSetting` table for categories like `orchestrator_llm` or `system_llm` [orchestrator/core/llm/manager.py:57-96]().
3. **Config defaults**: Fallback to `DEFAULT_LLM_PROVIDER` and `DEFAULT_LLM_MODEL` [orchestrator/modules/agents/factory/agent_factory.py:100-104]().

Sources: [orchestrator/modules/agents/factory/agent_factory.py:100-148](), [orchestrator/core/llm/manager.py:57-96]()

---

## API Key Resolution

### 3-Tier Resolution Strategy

Title: "API Key Resolution Code Bridge"
```mermaid
flowchart TD
    subgraph NL ["Natural Language Request"]
        UReq["Request LLM Generation"]
    end

    subgraph CodeSpace ["Code Entity Space"]
        C1["core/llm/manager.py:get_credential_data()"]
        C2["core/credentials/resolver.py:get_credential_resolver()"]
        C3["SystemSetting table lookup"]
        C4["Environment Variables fallback"]
    end

    UReq --> C1
    C1 --> C3
    C3 -->|Explicit mapping| C2
    C1 -->|Fallback| C4
```

The `get_credential_data` function ensures that workspaces can provide their own keys (BYOK) or use platform-provided credentials through structured priorities [orchestrator/core/llm/manager.py:135-155]().

Sources: [orchestrator/core/llm/manager.py:135-178]()

---

## Agent Execution & Tool Loop

### execute_with_prompt() & Tool Loop Spine

Executes a task with a multi-iteration tool loop. It utilizes `get_tools_for_agent_async` as the single source of truth for tool schemas [orchestrator/consumers/chatbot/service.py:52-57]().

Title: "Execution Loop and Tool Deduplication"
```mermaid
flowchart TD
    Start["AgentFactory.execute_with_prompt()"] --> BuildMsgs["Build messages array"]
    BuildMsgs --> LoopStart["Loop via ToolLoopExecutor"]
    LoopStart --> LLMCall["llm_manager.generate_response()"]
    LLMCall --> CheckTools{"tool_calls?"}
    
    CheckTools -->|No| Final["Extract final content"]
    CheckTools -->|Yes| Dedupe["ToolExecutionTracker / _queries_are_similar()"]
    
    Dedupe -->|Duplicate| SkipMsg["Add 'Already executed' error"]
    Dedupe -->|New| ExecTool["UnifiedToolExecutor.execute_tool()"]
    
    ExecTool --> NextIter["Next iteration"]
    SkipMsg --> NextIter
    NextIter --> LoopStart
```

### Tool Loop Prevention & Deduplication

Tool deduplication utilizes query normalization and similarity metrics:
1. **Normalization**: `_normalize_query(query)` strips punctuation and lowercases input [orchestrator/consumers/chatbot/service.py:76-83]().
2. **Semantic Deduplication**: `_queries_are_similar(query1, query2, threshold)` checks semantic overlap using `SequenceMatcher` [orchestrator/consumers/chatbot/service.py:85-95]().
3. **Argument Extraction**: `_extract_query_from_args` pulls relevant search terms from tool payloads [orchestrator/consumers/chatbot/service.py:97-104]().

Sources: [orchestrator/consumers/chatbot/service.py:30-104]()

---

## Platform Actions Runtime

Agents can manage the Automatos platform itself through `platform_*` actions. These are registered in `ActionRegistry` via `register_all_actions` and routed via `PlatformActionExecutor` [orchestrator/modules/tools/discovery/platform_actions.py:57-60](), [orchestrator/modules/tools/discovery/platform_executor.py:2-9]().

### Categories of Platform Actions
- **Agents**: `list_agents`, `create_agent`, `delete_agent` [orchestrator/modules/tools/discovery/platform_executor.py:19-29]()
- **Playbooks**: `list_playbooks`, `create_playbook`, `execute_playbook` [orchestrator/modules/tools/discovery/platform_executor.py:30-42]()
- **Analytics**: `get_llm_usage`, `get_cost_breakdown` [orchestrator/modules/tools/discovery/platform_executor.py:43-48]()
- **Workspace**: `get_workspace_info`, `list_connected_apps` [orchestrator/modules/tools/discovery/platform_executor.py:70-74]()

Sources: [orchestrator/modules/tools/discovery/platform_actions.py:57-103](), [orchestrator/modules/tools/discovery/platform_executor.py:1-80]()

---

## Tool Integration Architecture

### UnifiedToolExecutor

The `UnifiedToolExecutor` routes calls based on tool name patterns and execution contexts [orchestrator/modules/agents/factory/agent_factory.py:44-46]():
- **Platform Actions**: Prefixed with `platform_*`, allowing agents to manage the system itself [orchestrator/modules/tools/discovery/platform_executor.py:2-9]().
- **Composio Actions**: External app integrations managed via `ComposioAppCache` and `AgentAppAssignment` [orchestrator/core/models/composio_cache.py]() (referenced in factory: [orchestrator/modules/agents/factory/agent_factory.py:29]()).

Sources: [orchestrator/modules/agents/factory/agent_factory.py:44-46](), [orchestrator/modules/tools/discovery/platform_executor.py:1-10]()

---