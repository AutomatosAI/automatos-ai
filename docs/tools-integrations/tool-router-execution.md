# Tool Router & Execution

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/DoctorsNotes.docx](docs/DoctorsNotes.docx)
- [orchestrator/api/tools.py](orchestrator/api/tools.py)
- [orchestrator/consumers/chatbot/tool_router.py](orchestrator/consumers/chatbot/tool_router.py)
- [orchestrator/core/composio/client.py](orchestrator/core/composio/client.py)
- [orchestrator/modules/tools/execution/unified_executor.py](orchestrator/modules/tools/execution/unified_executor.py)
- [orchestrator/modules/tools/registry/tool_registry.py](orchestrator/modules/tools/registry/tool_registry.py)
- [orchestrator/modules/tools/services/composio_hint_service.py](orchestrator/modules/tools/services/composio_hint_service.py)
- [orchestrator/modules/tools/services/composio_tool_service.py](orchestrator/modules/tools/services/composio_tool_service.py)
- [orchestrator/modules/tools/tool_router.py](orchestrator/modules/tools/tool_router.py)
- [orchestrator/services/metadata_sync_service.py](orchestrator/services/metadata_sync_service.py)

</details>



This document covers the tool router service and tool execution subsystem, which processes LLM tool calls and routes them to the appropriate execution handlers. The tool router handles built-in tools, Composio integrations, deduplication, and result formatting for both chat and recipe execution contexts.

For information about how tools are resolved and selected for LLM context, see [Tool Resolution Strategies](#6.2). For details on Composio-specific integration patterns, see [Composio Integration](#6.1).

---

## Architecture Overview

The tool router acts as a central dispatcher that receives tool call requests from LLM responses and routes them to appropriate executors. It provides deduplication, tracking, and consistent result formatting across all tool types.

```mermaid
graph TB
    subgraph "LLM Response Processing"
        LLM["LLM Response<br/>(with tool_calls)"]
        Parser["Tool Call Parser<br/>(extract name + args)"]
    end
    
    subgraph "Tool Router Layer"
        Router["ToolRouter<br/>execute_and_format()"]
        Tracker["ToolExecutionTracker<br/>(deduplication)"]
        Registry["Tool Registry<br/>(built-in tools)"]
    end
    
    subgraph "Execution Handlers"
        Builtin["Built-in Tool<br/>Executor"]
        ComposioMega["composio_execute<br/>Mega-Tool"]
        CompositoDirect["Direct Composio<br/>Action Execution"]
        ScratchpadTool["scratchpad_write<br/>Inline Handler"]
    end
    
    subgraph "Result Processing"
        Formatter["Result Formatter<br/>llm_context builder"]
        ErrorHandler["Error Handler<br/>(fallback responses)"]
    end
    
    LLM --> Parser
    Parser --> Router
    Router --> Tracker
    
    Tracker -->|"not duplicate"| Registry
    Tracker -->|"duplicate"| ErrorHandler
    
    Registry -->|"built-in"| Builtin
    Registry -->|"composio_execute"| ComposioMega
    Registry -->|"JIRA_*, SLACK_*, etc"| CompositoDirect
    Registry -->|"scratchpad_write"| ScratchpadTool
    
    Builtin --> Formatter
    ComposioMega --> Formatter
    CompositoDirect --> Formatter
    ScratchpadTool --> Formatter
    ErrorHandler --> Formatter
    
    Formatter --> Result["Formatted Result<br/>(for LLM context)"]
```

**Sources:**
- [orchestrator/consumers/chatbot/service.py:34-34]()
- [orchestrator/api/recipe_executor.py:207-332]()
- [orchestrator/modules/agents/factory/agent_factory.py:1-520]()

---

## Tool Router Service

### Core Interface

The tool router is accessed via `get_tool_router()` which returns a singleton instance:

```mermaid
graph LR
    GetRouter["get_tool_router()"]
    Instance["ToolRouter Instance<br/>(singleton)"]
    
    ExecuteMethod["execute_and_format()<br/>- tool_name<br/>- tool_args<br/>- agent_id<br/>- workspace_id<br/>- original_intent"]
    
    FormatResult["formatted result<br/>{llm_context, status, data}"]
    
    GetRouter --> Instance
    Instance --> ExecuteMethod
    ExecuteMethod --> FormatResult
```

The `execute_and_format()` method is the primary interface for tool execution:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `tool_name` | `str` | Name of the tool to execute (e.g. `"read_file"`, `"JIRA_GET_ISSUE"`) |
| `tool_args` | `Dict[str, Any]` | Tool arguments as parsed from LLM response |
| `agent_id` | `int` | Agent ID for permission checks and tracking |
| `workspace_id` | `UUID` | Workspace ID for multi-tenancy and Composio entity resolution |
| `original_intent` | `str` | Original user prompt for context-aware error messages |

Returns a dictionary with:
- `llm_context`: Formatted string result for LLM message history
- `status`: Execution status (`"success"` or `"error"`)
- `data`: Structured result data (optional)

**Sources:**
- [orchestrator/consumers/chatbot/service.py:34]()
- [orchestrator/api/recipe_executor.py:314-332]()

---

## Tool Execution Strategies

The tool router supports multiple execution strategies, automatically selected based on the tool name:

### Strategy Selection Logic

```mermaid
graph TD
    Start["Tool Call<br/>(name + args)"]
    
    CheckScratchpad{"tool_name ==<br/>scratchpad_write?"}
    CheckComposioDirect{"Matches Composio<br/>action pattern?<br/>(APP_ACTION_NAME)"}
    CheckComposioMega{"tool_name ==<br/>composio_execute?"}
    CheckBuiltin{"Registered in<br/>tool registry?"}
    
    ScratchpadHandler["Inline Handler<br/>(no router needed)"]
    DirectExec["ComposioToolService<br/>execute_action()"]
    MegaTool["composio_execute<br/>with param mapping"]
    BuiltinExec["Built-in Tool<br/>Executor"]
    FallbackError["Unknown Tool<br/>Error Response"]
    
    Start --> CheckScratchpad
    CheckScratchpad -->|Yes| ScratchpadHandler
    CheckScratchpad -->|No| CheckComposioDirect
    
    CheckComposioDirect -->|Yes| DirectExec
    CheckComposioDirect -->|No| CheckComposioMega
    
    CheckComposioMega -->|Yes| MegaTool
    CheckComposioMega -->|No| CheckBuiltin
    
    CheckBuiltin -->|Yes| BuiltinExec
    CheckBuiltin -->|No| FallbackError
```

### Direct Composio Action Execution (Primary Path)

When SDK search returns per-action tools (see [Tool Resolution Strategies](#6.2)), the recipe executor and chat service use direct execution to bypass the `composio_execute` mega-tool and its parameter mapping overhead:

```mermaid
sequenceDiagram
    participant LLM
    participant Executor as Recipe/Chat Executor
    participant Service as ComposioToolService
    participant Client as ComposioClient
    
    LLM->>Executor: tool_call: JIRA_GET_ISSUE
    Note over LLM,Executor: {issue_id_or_key: "PILOT-123"}
    
    Executor->>Executor: Check if action in<br/>composio_result.action_set
    
    alt Direct execution path
        Executor->>Service: execute_action(action_name,<br/>params, entity_id)
        Service->>Client: SDK.execute_action()
        Client-->>Service: {success, data}
        Service-->>Executor: Result dict
        
        Executor->>Executor: Cache result by<br/>action+args hash
        Note over Executor: Prevents repeat calls<br/>with same params
    end
    
    Executor->>LLM: tool_call_result
```

The deduplication cache in recipes prevents identical calls within the same execution:

**Key Implementation Details:**
- Cache key: `f"{tool_name}|{json.dumps(tool_args, sort_keys=True, default=str)}"`
- Cache scope: Per-execution (not persistent)
- Cache hit: Returns cached result, logs `"Composio dedup hit"`
- Applies to: All Composio actions in recipe execution

**Sources:**
- [orchestrator/api/recipe_executor.py:256-312]()
- [orchestrator/modules/tools/services/composio_tool_service.py:193-212]()

### Built-in Tools

Built-in tools (e.g., `read_file`, `search_knowledge`, `query_database`) are registered in the tool registry and executed directly:

| Tool Category | Examples | Handler Location |
|--------------|----------|------------------|
| File Operations | `read_file`, `write_file`, `list_directory` | Tool executor modules |
| RAG/Search | `search_knowledge`, `semantic_search`, `search_codebase` | RAG service |
| Database | `query_database`, `smart_query_database` | NL2SQL module |
| Scratchpad | `scratchpad_write` | Inline handler (recipes only) |

**Sources:**
- [orchestrator/modules/agents/factory/agent_factory.py:54-230]()
- [orchestrator/api/recipe_executor.py:236-254]()

### Scratchpad Tool (Recipe-Specific)

The `scratchpad_write` tool is handled inline during recipe execution, bypassing the tool router entirely:

```python
# Inline handling in recipe executor
if tool_name == SCRATCHPAD_TOOL_NAME and scratchpad:
    result_text = handle_scratchpad_write(
        key=tool_args.get("key", "unknown"),
        value=tool_args.get("value", ""),
        scratchpad=scratchpad,
        step_order=step_order,
    )
    # No tool_router.execute_and_format() call
```

This tool allows agents to explicitly export key-value pairs to the recipe scratchpad for downstream steps. See [Recipe Scratchpad](#4.6) for details.

**Sources:**
- [orchestrator/api/recipe_executor.py:236-254]()
- [orchestrator/core/services/recipe_scratchpad.py:144-161]()

---

## Deduplication & Loop Prevention

The `ToolExecutionTracker` class implements three-tier deduplication to prevent infinite tool loops:

### Deduplication Strategies

```mermaid
graph TB
    subgraph "ToolExecutionTracker"
        IncomingCall["Incoming Tool Call<br/>(name + args)"]
        
        Check1["Check 1:<br/>Retry Limit"]
        Check2["Check 2:<br/>Exact Deduplication"]
        Check3["Check 3:<br/>Semantic Deduplication<br/>(search tools only)"]
        
        ExactSet["exact_executions<br/>Set[(tool_name, args_hash)]"]
        QueryCache["search_queries<br/>Dict[tool_name, List[query]]"]
        CountMap["tool_counts<br/>Dict[tool_name, count]"]
        
        Decision{"Should Skip?"}
        Execute["Execute Tool"]
        Skip["Skip Execution<br/>(return cached or error)"]
    end
    
    IncomingCall --> Check1
    Check1 --> CountMap
    CountMap --> Decision
    
    Check1 --> Check2
    Check2 --> ExactSet
    ExactSet --> Decision
    
    Check2 --> Check3
    Check3 --> QueryCache
    QueryCache --> Decision
    
    Decision -->|"No conflicts"| Execute
    Decision -->|"Duplicate/limit"| Skip
    
    Execute --> ExactSet
    Execute --> QueryCache
    Execute --> CountMap
```

### Implementation Details

**Per-Tool Retry Limits:**

| Tool Type | Limit | Rationale |
|-----------|-------|-----------|
| `composio_execute` | 2 | Expensive API calls; fail fast |
| Search tools (`search_knowledge`, `semantic_search`, etc.) | 2 | Prevent query loops |
| `query_database`, `smart_query_database` | 2 | Database load protection |
| File operations (`read_file`, `write_file`) | 3 or 2 | Moderate retry tolerance |
| Default (unlisted tools) | 3 | Conservative limit |

**Exact Deduplication:**

Uses MD5 hash of JSON-serialized arguments (sorted keys) to detect identical calls:

```python
def _hash_args(self, tool_args: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()
```

**Semantic Deduplication (Search Tools Only):**

For search tools (`search_knowledge`, `semantic_search`, `search_codebase`, etc.), compares normalized query strings using `SequenceMatcher` with a 0.75 similarity threshold:

```python
def _queries_are_similar(query1: str, query2: str, threshold: float = 0.75) -> bool:
    norm1 = _normalize_query(query1)  # Lowercase, strip punctuation
    norm2 = _normalize_query(query2)
    
    if norm1 == norm2:
        return True
    
    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold
```

This prevents scenarios like:
1. `search_knowledge("Python async patterns")`
2. `search_knowledge("python async patterns")` ← Blocked as similar
3. `search_knowledge("async patterns in python")` ← Blocked if ratio ≥ 0.75

**Sources:**
- [orchestrator/consumers/chatbot/service.py:88-186]()
- [orchestrator/consumers/chatbot/service.py:44-86]()

---

## Execution Context Differences

Tool execution behaves differently in chat vs. recipe contexts due to different requirements:

### Chat Context

```mermaid
graph LR
    subgraph "Chat Execution Flow"
        ChatService["StreamingChatService"]
        ToolList["get_chat_tools()<br/>(filtered to top-N)"]
        Tracker["ToolExecutionTracker<br/>(per-conversation)"]
        Router["tool_router.execute_and_format()"]
        
        ChatService --> ToolList
        ToolList --> Tracker
        Tracker --> Router
    end
```

**Characteristics:**
- Tool list filtered to top 25 relevant tools using `rank_tools_for_query()`
- `ToolExecutionTracker` scoped to single conversation turn
- All tool execution via `tool_router.execute_and_format()`
- Composio tools use mega-tool pattern or per-action tools based on SDK search results
- Tool results inserted into message history with `role: "tool"`

**Sources:**
- [orchestrator/consumers/chatbot/service.py:492-850]()
- [orchestrator/consumers/chatbot/service.py:544-585]()

### Recipe Context

```mermaid
graph LR
    subgraph "Recipe Execution Flow"
        RecipeExec["Recipe Executor"]
        ToolList["Full tool set<br/>(no filtering)"]
        DirectExec["Direct Composio Execution<br/>(when SDK search succeeds)"]
        CacheCheck["Per-Execution Cache<br/>(action+args hash)"]
        FallbackRouter["tool_router (fallback)"]
        
        RecipeExec --> ToolList
        ToolList --> DirectExec
        DirectExec --> CacheCheck
        CacheCheck --> FallbackRouter
    end
```

**Characteristics:**
- No tool filtering (recipe steps are curated)
- Direct Composio action execution when SDK search returns per-action tools
- Per-execution deduplication cache (action+args hash)
- `tool_router` used only for built-in tools
- `scratchpad_write` handled inline (no router)
- No persistent `ToolExecutionTracker` (single-step execution)

**Sources:**
- [orchestrator/api/recipe_executor.py:44-364]()
- [orchestrator/api/recipe_executor.py:256-312]()

---

## Result Formatting

The tool router formats all tool execution results into a consistent `llm_context` string suitable for LLM message history:

### Standard Format

```mermaid
graph TB
    subgraph "Result Types"
        Success["Success Result<br/>{status: 'success', data: ...}"]
        Error["Error Result<br/>{status: 'error', error: ...}"]
        StructuredData["Structured Data<br/>(JSON, dict, list)"]
        TextData["Text Data<br/>(string response)"]
    end
    
    subgraph "Formatting Rules"
        Truncate["Truncate to 8000 chars<br/>(LLM context limit)"]
        JSONStringify["JSON.stringify()<br/>for structured data"]
        ErrorPrefix["Prefix: 'Error executing...'"]
        SuccessWrap["Wrap in result object"]
    end
    
    Success --> StructuredData
    Success --> TextData
    StructuredData --> JSONStringify
    TextData --> Truncate
    
    Error --> ErrorPrefix
    ErrorPrefix --> Truncate
    
    JSONStringify --> Truncate
    Truncate --> LLMContext["llm_context string<br/>(returned to LLM)"]
```

### Example Formatting

**Success with structured data:**
```python
# Input (from Composio execution)
{
    "success": True,
    "data": {
        "issue_key": "PILOT-123",
        "summary": "Implement feature X",
        "status": "In Progress"
    }
}

# Formatted llm_context
'{"issue_key": "PILOT-123", "summary": "Implement feature X", "status": "In Progress"}'
```

**Error:**
```python
# Input (from failed tool call)
{
    "success": False,
    "error": "Authentication failed: Invalid credentials"
}

# Formatted llm_context
'Error executing JIRA_GET_ISSUE: Authentication failed: Invalid credentials'
```

**Truncation:**
- Applied to final `llm_context` string
- Limit: 8000 characters for message history, 4000 for tool result storage
- Prevents context window overflow

**Sources:**
- [orchestrator/api/recipe_executor.py:287-311]()
- [orchestrator/consumers/chatbot/service.py:314-332]()

---

## Integration Points

### Chat Service Integration

The chat service uses the tool router within a tool execution loop, processing multiple tool calls per LLM response:

```mermaid
sequenceDiagram
    participant Chat as StreamingChatService
    participant LLM as LLMManager
    participant Tracker as ToolExecutionTracker
    participant Router as ToolRouter
    
    Chat->>Chat: Create ToolExecutionTracker<br/>(per-turn scope)
    
    loop Tool Loop (max 6 iterations)
        Chat->>LLM: generate_response(messages, tools)
        LLM-->>Chat: response with tool_calls[]
        
        alt No tool calls
            Chat->>Chat: Break loop (final response)
        else Has tool calls
            loop For each tool_call
                Chat->>Tracker: should_skip_execution(tool_name, args)
                
                alt Should skip
                    Tracker-->>Chat: (skip, reason)
                    Chat->>Chat: Add error to message history
                else Can execute
                    Chat->>Router: execute_and_format(tool_name, args, ...)
                    Router-->>Chat: {llm_context, status}
                    Chat->>Tracker: record_execution(tool_name, args)
                    Chat->>Chat: Append tool result to messages
                end
            end
        end
    end
    
    Chat->>Chat: Return final LLM response
```

**Sources:**
- [orchestrator/consumers/chatbot/service.py:640-850]()
- [orchestrator/consumers/chatbot/service.py:130-164]()

### Recipe Executor Integration

Recipe execution uses selective tool router integration, preferring direct Composio execution when per-action tools are available:

```mermaid
graph TD
    Start["Recipe Step Execution"]
    
    GetTools["Get tools via<br/>ComposioToolService"]
    
    CheckStrategy{"SDK search<br/>returned tools?"}
    
    DirectPath["Direct Execution Path<br/>(bypass router)"]
    FallbackPath["Router Path<br/>(mega-tool + builtin)"]
    
    CheckToolName{"Tool name<br/>matches pattern?"}
    
    CompositoDirect["Execute via<br/>ComposioToolService.execute_action()"]
    RouterCall["Execute via<br/>tool_router.execute_and_format()"]
    ScratchpadInline["Handle inline<br/>(scratchpad_write only)"]
    
    Start --> GetTools
    GetTools --> CheckStrategy
    
    CheckStrategy -->|Yes| DirectPath
    CheckStrategy -->|No| FallbackPath
    
    DirectPath --> CheckToolName
    CheckToolName -->|"Composio action"| CompositoDirect
    CheckToolName -->|"scratchpad_write"| ScratchpadInline
    CheckToolName -->|"Built-in"| RouterCall
    
    FallbackPath --> RouterCall
```

**Key Difference from Chat:**
- No persistent `ToolExecutionTracker` (steps execute independently)
- Per-execution deduplication cache (action+args hash)
- Inline `scratchpad_write` handling
- Direct Composio execution when SDK search succeeds

**Sources:**
- [orchestrator/api/recipe_executor.py:206-332]()
- [orchestrator/api/recipe_executor.py:256-312]()

---

## Code Entity Reference

### Primary Classes and Functions

| Entity | Location | Purpose |
|--------|----------|---------|
| `get_tool_router()` | `consumers/chatbot/tool_router.py` | Factory function returning singleton ToolRouter instance |
| `ToolRouter.execute_and_format()` | `consumers/chatbot/tool_router.py` | Primary tool execution interface |
| `ToolExecutionTracker` | `consumers/chatbot/service.py:88-186` | Deduplication and loop prevention for chat |
| `ComposioToolService.execute_action()` | `modules/tools/services/composio_tool_service.py:193-212` | Direct Composio action execution |
| `get_chat_tools()` | `consumers/chatbot/tool_router.py` | Returns available tool schemas for chat context |
| `handle_scratchpad_write()` | `modules/tools/builtin/scratchpad_tool.py` | Inline handler for scratchpad exports in recipes |

### Tool Execution Sites

| Context | File | Lines | Notes |
|---------|------|-------|-------|
| Chat tool loop | `consumers/chatbot/service.py` | 640-850 | Uses ToolExecutionTracker + tool_router |
| Recipe tool loop | `api/recipe_executor.py` | 206-332 | Direct Composio + selective router use |
| Direct Composio exec | `api/recipe_executor.py` | 256-312 | Bypasses router for per-action tools |
| Built-in tools | Various tool modules | - | Registered in tool registry |

**Sources:**
- [orchestrator/consumers/chatbot/service.py:88-186]()
- [orchestrator/api/recipe_executor.py:44-364]()
- [orchestrator/modules/tools/services/composio_tool_service.py:1-301]()

---