# Recipe Scratchpad

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/workflows/create-recipe-modal.tsx](frontend/components/workflows/create-recipe-modal.tsx)
- [frontend/components/workflows/execution-kitchen.tsx](frontend/components/workflows/execution-kitchen.tsx)
- [frontend/components/workflows/recipe-execution-config.tsx](frontend/components/workflows/recipe-execution-config.tsx)
- [frontend/components/workflows/recipe-preview-panel.tsx](frontend/components/workflows/recipe-preview-panel.tsx)
- [frontend/components/workflows/recipe-step-builder.tsx](frontend/components/workflows/recipe-step-builder.tsx)
- [frontend/components/workflows/recipes-tab.tsx](frontend/components/workflows/recipes-tab.tsx)
- [frontend/components/workflows/view-recipe-modal.tsx](frontend/components/workflows/view-recipe-modal.tsx)
- [frontend/hooks/use-recipe-form.ts](frontend/hooks/use-recipe-form.ts)
- [orchestrator/alembic/versions/20260202_add_workspace_id_to_skills_patterns_models.py](orchestrator/alembic/versions/20260202_add_workspace_id_to_skills_patterns_models.py)
- [orchestrator/api/recipe_executor.py](orchestrator/api/recipe_executor.py)
- [orchestrator/api/workflow_recipes.py](orchestrator/api/workflow_recipes.py)
- [orchestrator/core/models/core.py](orchestrator/core/models/core.py)
- [orchestrator/core/services/recipe_memory_service.py](orchestrator/core/services/recipe_memory_service.py)
- [orchestrator/core/services/workspace_manager.py](orchestrator/core/services/workspace_manager.py)

</details>



## Purpose and Scope

The Recipe Scratchpad is an ephemeral, structured context-sharing system for multi-step recipe executions. It replaces verbose full-text output dumps between steps with auto-extracted key-value summaries, achieving 80-90% token savings while preserving essential context. The scratchpad is backed by Redis for performance and falls back to in-memory storage when Redis is unavailable.

This document covers the scratchpad's architecture, key layout, auto-extraction strategies, and integration with recipe execution. For information about recipe execution flow, see [Recipe Execution](#4.2). For details on S3 log storage, see [Execution Configuration](#4.3).

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:1-20]()

---

## Problem Statement

Before the scratchpad system, recipe steps received context from previous steps as verbose text dumps—full agent responses, complete tool call results, and raw output strings. This approach had critical problems:

| Problem | Impact | Example |
|---------|--------|---------|
| **Token Bloat** | Steps 5+ consumed 10K+ tokens for context alone | 5-step recipe: ~50K context tokens |
| **LLM Confusion** | Agents overwhelmed by irrelevant details | "Read all 47 PRs, then create 1 new PR" → agent reads PRs again |
| **Cost Escalation** | Linear token growth per step: O(n²) | 10-step recipe: $2-5 in context tokens alone |
| **Context Window Limits** | Long recipes hit 128K limits | Execution failures at step 8-10 |

The scratchpad addresses these by extracting only the essential structured data from each step—URLs, key-value pairs, tool call summaries—and injecting it into subsequent steps as compact, queryable context.

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:1-16]()

---

## Architecture Overview

```mermaid
graph TB
    RecipeExecutor["RecipeExecutor<br/>(recipe_executor.py)"]
    Scratchpad["RecipeScratchpad<br/>(recipe_scratchpad.py)"]
    RedisClient["Redis Client<br/>(core/redis/client.py)"]
    FallbackDict["In-Memory Dict<br/>(fallback)"]
    
    RecipeExecutor -->|"initialize(execution_id)"| Scratchpad
    RecipeExecutor -->|"write_inputs()"| Scratchpad
    RecipeExecutor -->|"write_step_results()"| Scratchpad
    RecipeExecutor -->|"format_context_for_step()"| Scratchpad
    
    Scratchpad -->|"try Redis first"| RedisClient
    Scratchpad -->|"fallback if unavailable"| FallbackDict
    
    RedisClient -.->|"HSET/HGET/HGETALL"| RedisHash["Redis Hash<br/>recipe_exec:{execution_id}"]
    FallbackDict -.->|"dict operations"| MemoryStore["Python Dict<br/>{field: value}"]
    
    subgraph "Storage Backends"
        RedisHash
        MemoryStore
    end
    
    subgraph "Lifecycle"
        Init["1. Initialize<br/>write_inputs()"]
        Execute["2. Execute Steps<br/>write_step_results()"]
        Format["3. Format Context<br/>format_context_for_step()"]
        Cleanup["4. Ephemeral<br/>(no explicit cleanup)"]
        
        Init --> Execute
        Execute --> Format
        Format --> Cleanup
    end
```

The `RecipeScratchpad` class provides a key-value abstraction over Redis hashes, with automatic fallback to in-memory dictionaries. It's initialized once per recipe execution and accessed by all steps in sequence.

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:33-63](), [orchestrator/api/recipe_executor.py:632-636]()

---

## Redis Key Layout

The scratchpad stores all execution context in a single Redis hash with structured field names:

```mermaid
graph LR
    subgraph "Redis Hash: recipe_exec:abc-123"
        InputFields["<b>_input: Fields</b><br/>_input:issue_key<br/>_input:project<br/>_input:content"]
        MetaFields["<b>_meta: Fields</b><br/>_meta:recipe_id<br/>_meta:total_steps"]
        Step1["<b>step_1: Fields</b><br/>step_1:tool_results<br/>step_1:output_summary<br/>step_1:exports"]
        Step2["<b>step_2: Fields</b><br/>step_2:tool_results<br/>step_2:output_summary<br/>step_2:exports"]
        StepN["<b>step_N: Fields</b><br/>step_N:tool_results<br/>step_N:output_summary<br/>step_N:exports"]
    end
    
    InputFields -.->|"written once at start"| WriteInputs["write_inputs()"]
    MetaFields -.->|"written once at start"| WriteMeta["write_meta()"]
    Step1 -.->|"written after step completes"| WriteResults["write_step_results()"]
    Step2 -.->|"written after step completes"| WriteResults
    StepN -.->|"written after step completes"| WriteResults
```

### Field Types

| Field Pattern | Type | Purpose | Example Value |
|---------------|------|---------|---------------|
| `_input:{key}` | String | Original trigger/input data | `_input:issue_key` → `"PILOT-123"` |
| `_meta:recipe_id` | String | Recipe PK | `"42"` |
| `_meta:total_steps` | String | Step count | `"5"` |
| `step_{N}:tool_results` | JSON Array | Auto-extracted tool summaries | `[{"action": "JIRA_GET_ISSUE", "key_data": {"issue_key": "PILOT-123"}}]` |
| `step_{N}:output_summary` | String | First 500 chars of agent output | `"Created PR #456 successfully..."` |
| `step_{N}:exports` | JSON Object | Explicit agent-written data | `{"pr_number": "456", "branch": "fix/pilot-123"}` |

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:8-16](), [orchestrator/core/services/recipe_scratchpad.py:99-143]()

---

## Data Flow Through Recipe Execution

```mermaid
sequenceDiagram
    participant Executor as RecipeExecutor
    participant Scratchpad as RecipeScratchpad
    participant Redis as Redis Hash
    participant Agent as AgentRuntime
    participant S3 as S3 Bucket

    Note over Executor: execute_recipe_direct()
    Executor->>Scratchpad: initialize(execution_id)
    Scratchpad->>Redis: create hash recipe_exec:abc-123
    
    Executor->>Scratchpad: write_inputs({"issue_key": "PILOT-123"})
    Scratchpad->>Redis: HSET _input:issue_key "PILOT-123"
    
    Executor->>Scratchpad: write_meta(recipe_id=42, total_steps=3)
    Scratchpad->>Redis: HSET _meta:recipe_id "42"
    
    Note over Executor: Step 1 execution
    Executor->>Scratchpad: format_context_for_step(1)
    Scratchpad-->>Executor: None (no prior steps)
    
    Executor->>Agent: execute_step(system_prompt, input_data)
    Agent-->>Executor: {"output": "...", "tool_calls": [...]}
    
    Executor->>Scratchpad: write_step_results(step_order=1, tool_calls, output)
    Scratchpad->>Scratchpad: auto-extract key data
    Scratchpad->>Redis: HSET step_1:tool_results "[{...}]"
    Scratchpad->>Redis: HSET step_1:output_summary "Created..."
    
    Executor->>S3: upload full logs (step_1.json)
    
    Note over Executor: Step 2 execution
    Executor->>Scratchpad: format_context_for_step(2)
    Scratchpad->>Redis: HGETALL recipe_exec:abc-123
    Redis-->>Scratchpad: {_input:issue_key, step_1:tool_results, ...}
    Scratchpad-->>Executor: "=== CONTEXT ===\nStep 1: JIRA_GET_ISSUE -> issue_key: PILOT-123\n..."
    
    Executor->>Agent: execute_step(system_prompt + context, input_data)
    Agent-->>Executor: {"output": "...", "tool_calls": [...]}
    
    Executor->>Scratchpad: write_step_results(step_order=2, tool_calls, output)
    Scratchpad->>Redis: HSET step_2:tool_results "[{...}]"
    
    Note over Executor: Step 3 execution (same pattern)
```

### Key Integration Points

1. **Initialization** ([recipe_executor.py:632-636]()): Create scratchpad, write inputs and metadata
2. **Context Injection** ([recipe_executor.py:179-182]()): Format compact context before each step
3. **Auto-Extraction** ([recipe_executor.py:753-759]()): Extract and store results after each step
4. **S3 Upload** ([recipe_executor.py:466-509]()): Store full verbose logs separately

**Sources:** [orchestrator/api/recipe_executor.py:559-777](), [orchestrator/core/services/recipe_scratchpad.py:174-252]()

---

## Auto-Extraction Strategy

The scratchpad uses zero-LLM regex-based extraction to identify key data from tool calls and agent output. This ensures consistent, fast summarization without additional API costs.

### Extraction Targets

```mermaid
graph TB
    ToolCalls["Tool Call Results<br/>(raw JSON/text)"]
    AgentOutput["Agent Output<br/>(raw text)"]
    
    ToolCalls --> ExtractURLs["Extract URLs<br/>https?://[^\\s]+"]
    ToolCalls --> ExtractKV["Extract Key-Value Pairs<br/>^([A-Z][A-Za-z_ ]{1,30}):\\s+(.+)$"]
    ToolCalls --> ExtractJSON["Parse JSON Objects<br/>json.loads()"]
    
    ExtractURLs --> Summary1["urls: [...]"]
    ExtractKV --> Summary2["key_data: {Branch: fix/..., PR: #456}"]
    ExtractJSON --> Summary3["structured: {...}"]
    
    Summary1 --> Scratchpad["step_N:tool_results<br/>JSON Array"]
    Summary2 --> Scratchpad
    Summary3 --> Scratchpad
    
    AgentOutput --> Truncate["Truncate to 500 chars"]
    Truncate --> OutputField["step_N:output_summary<br/>String"]
```

### Extraction Examples

| Input | Extraction | Output Field |
|-------|------------|--------------|
| Tool result: `{"data": {"html_url": "https://github.com/org/repo/pull/456"}}` | `urls: ["https://github.com/org/repo/pull/456"]` | `step_1:tool_results` → `[{"action": "GITHUB_CREATE_PULL_REQUEST", "key_data": {"urls": [...]}}]` |
| Tool result: `"Branch: fix/PILOT-123\nPR: #456"` | `key_data: {Branch: "fix/PILOT-123", PR: "#456"}` | `step_1:tool_results` → `[{"action": "...", "key_data": {...}}]` |
| Agent output: 2000 chars | Truncate to 500 chars + "..." | `step_1:output_summary` → `"Created PR #456 successfully. The branch fix/PILOT-123 has been..."` |

### Implementation

The extraction logic is in `_extract_tool_summaries()` and `_extract_key_data()`:

```python
# Auto-extract from tool call result
def _extract_key_data(result_obj: Any) -> Dict[str, Any]:
    extracted = {}
    
    # 1. Extract URLs
    urls = _extract_urls(str(result_obj))
    if urls:
        extracted["urls"] = urls
    
    # 2. Extract key-value pairs (e.g. "Branch: fix/issue")
    kv_pairs = _extract_kv_pairs(str(result_obj))
    if kv_pairs:
        extracted.update(kv_pairs)
    
    # 3. Parse as JSON if possible
    if isinstance(result_obj, dict):
        extracted["structured"] = result_obj
    
    return extracted
```

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:254-362]()

---

## Context Formatting

When a step requests context via `format_context_for_step(N)`, the scratchpad builds a compact, structured summary of all prior steps:

```mermaid
graph TB
    FormatRequest["format_context_for_step(3)"]
    
    FormatRequest --> ScanFields["Scan Redis fields<br/>for step_1:*, step_2:*"]
    
    ScanFields --> BuildHeader["Build header:<br/>'=== EXECUTION CONTEXT (Steps 1-2 completed) ==='"]
    
    BuildHeader --> InputSection["## Inputs<br/>- issue_key: PILOT-123<br/>- project: PILOT"]
    
    InputSection --> Step1Section["## Step 1<br/>- JIRA_GET_ISSUE -> issue_key: PILOT-123, summary: 'Fix auth bug'<br/>- Agent: 'Retrieved issue details...'"]
    
    Step1Section --> Step2Section["## Step 2<br/>- GITHUB_CREATE_PULL_REQUEST -> urls: ['https://github.com/.../pull/456']<br/>- Agent: 'Created PR #456...'<br/>- Exports: {pr_number: '456', branch: 'fix/pilot-123'}"]
    
    Step2Section --> Result["Formatted Context String<br/>(~500 tokens vs 5000 raw)"]
```

### Example Formatted Context

```
========================================
EXECUTION CONTEXT (Steps 1-2 completed)
========================================

## Inputs
- issue_key: PILOT-123
- project: PILOT
- content: Fix authentication timeout bug

## Step 1
- JIRA_GET_ISSUE -> issue_key: PILOT-123, summary: Fix auth timeout bug
- Agent: "Retrieved issue PILOT-123. Summary: Fix authentication timeout bug. Priority: High. Status: In Progress."

## Step 2
- GITHUB_CREATE_PULL_REQUEST -> urls: https://github.com/org/repo/pull/456
- Agent: "Created PR #456 successfully. Branch: fix/pilot-123. Title: Fix auth timeout bug (PILOT-123)."
- Exports: {pr_number: 456, branch: fix/pilot-123}
```

This formatted context replaces a 5000-token raw dump with ~500 tokens of essential information.

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:174-252]()

---

## Storage Tiers

The recipe execution system uses a three-tier storage strategy to balance performance, cost, and queryability:

```mermaid
graph LR
    subgraph "Tier 1: Ephemeral (Redis)"
        Redis["RecipeScratchpad<br/>recipe_exec:{id}<br/><br/>TTL: session lifetime<br/>Size: ~10-50 KB<br/>Purpose: inter-step context"]
    end
    
    subgraph "Tier 2: Compact (PostgreSQL)"
        CompactDB["RecipeExecution.step_results<br/><br/>Size: ~200 KB<br/>Storage: JSONB column<br/>Purpose: UI display, queries"]
    end
    
    subgraph "Tier 3: Cold (S3)"
        FullLogs["s3://bucket/workspaces/{ws}/logs/<br/>executions/{id}/step_N.json<br/><br/>Size: ~2-10 MB<br/>Purpose: debugging, audit"]
    end
    
    RecipeExecutor["RecipeExecutor"] -->|"write during execution"| Redis
    RecipeExecutor -->|"write after each step"| CompactDB
    RecipeExecutor -->|"upload after each step"| FullLogs
    
    Frontend["Frontend UI"] -->|"read for progress display"| CompactDB
    Frontend -->|"lazy-load on demand"| FullLogs
    
    Redis -.->|"ephemeral, not persisted"| Cleanup["Auto-cleanup after<br/>execution completes"]
```

### Tier Comparison

| Tier | Storage | Retention | Size | Access Pattern | Use Case |
|------|---------|-----------|------|----------------|----------|
| **Redis** | `RecipeScratchpad` hash | Session lifetime | 10-50 KB | Read/write during execution | Inter-step context passing |
| **PostgreSQL** | `RecipeExecution.step_results` JSONB | Permanent | 200 KB | Read for UI rendering | Progress display, quick queries |
| **S3** | `step_N.json` files | Permanent | 2-10 MB | Lazy-load on demand | Full debugging, audit trail |

### Compact Summary Format

The compact summary stored in PostgreSQL omits full output and tool results, replacing them with previews:

```json
{
  "step_id": "step-1",
  "order": 1,
  "agent_id": 42,
  "agent_name": "JIRA Agent",
  "status": "success",
  "duration_ms": 3200,
  "tokens_used": 450,
  "tool_calls_summary": ["JIRA_GET_ISSUE (success)", "JIRA_ADD_COMMENT (success)"],
  "output_preview": "Retrieved issue PILOT-123 successfully. Priority: High, Status: In Progress...",
  "log_url": "s3://bucket/workspaces/abc/logs/executions/xyz/step_1.json"
}
```

**Sources:** [orchestrator/api/recipe_executor.py:466-553](), [orchestrator/core/services/recipe_scratchpad.py:1-16]()

---

## Frontend Integration

The frontend displays recipe progress using compact summaries from PostgreSQL and lazy-loads full logs from S3 when users expand step details.

```mermaid
graph TB
    ExecutionKitchen["ExecutionKitchen<br/>(execution-kitchen.tsx)"]
    StepProgress["RecipeStepProgress<br/>(recipe-step-progress.tsx)"]
    API["API Client<br/>(api-client.ts)"]
    
    ExecutionKitchen -->|"render"| StepProgress
    
    StepProgress -->|"initial render"| CompactData["Display compact summaries<br/>tool_calls_summary<br/>output_preview<br/>duration_ms, tokens_used"]
    
    StepProgress -->|"user clicks step"| Expand["toggleStep(order)"]
    
    Expand -->|"if no full logs cached"| LoadLogs["loadFullLogs(stepOrder)"]
    
    LoadLogs -->|"GET /api/workflow-recipes/{id}/executions/{exec_id}/steps/{N}/logs"| API
    
    API -->|"fetch from S3, return JSON"| FullLogData["Full log data:<br/>messages, tool_calls,<br/>raw output, metadata"]
    
    FullLogData --> Render["Render expanded view:<br/>tool parameters<br/>full output<br/>timing breakdown"]
```

### API Endpoint for Full Logs

The frontend lazy-loads full logs via:

```
GET /api/workflow-recipes/{recipe_id}/executions/{execution_id}/steps/{step_order}/logs
```

This endpoint:
1. Looks up the `log_url` from `RecipeExecution.step_results[N]`
2. Downloads the S3 object (e.g. `step_3.json`)
3. Returns the full log JSON to the frontend

**Sources:** [frontend/components/workflows/recipe-step-progress.tsx:77-92](), [frontend/components/workflows/execution-kitchen.tsx:1-20]()

---

## Fallback Behavior

The scratchpad gracefully degrades to in-memory storage when Redis is unavailable:

```mermaid
graph TB
    Init["RecipeScratchpad.__init__(execution_id)"]
    
    Init --> TryRedis["Try get_redis_client()"]
    
    TryRedis -->|"success"| UseRedis["self._redis = client<br/>self._fallback = {}"]
    TryRedis -->|"exception"| UseFallback["self._redis = None<br/>self._fallback = {}"]
    
    UseRedis --> Operations1["Redis Operations<br/>HSET, HGET, HGETALL"]
    UseFallback --> Operations2["Dict Operations<br/>dict[field] = value"]
    
    Operations1 -.->|"on Redis error"| FallbackRecovery["Catch exception<br/>Fall back to _fallback dict"]
    
    FallbackRecovery --> Operations2
```

### Fallback Guarantees

| Operation | Redis Available | Redis Unavailable | Behavior Change |
|-----------|-----------------|-------------------|-----------------|
| `write_inputs()` | `HSET` to Redis | Write to `_fallback` dict | None (transparent) |
| `write_step_results()` | `HSET` to Redis | Write to `_fallback` dict | None (transparent) |
| `format_context_for_step()` | `HGETALL` from Redis | Read from `_fallback` dict | None (transparent) |
| **Persistence** | Ephemeral (session) | **Ephemeral (process)** | ⚠️ Lost if process crashes |
| **Concurrency** | Safe (Redis atomicity) | **Not safe** | ⚠️ Single-process only |

The fallback ensures recipe execution never fails due to Redis unavailability, but loses durability guarantees. In production, Redis should always be available.

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:40-94]()

---

## Performance Impact

The scratchpad's token savings compound across recipe steps:

### Token Savings Example (5-Step Recipe)

| Approach | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Total |
|----------|--------|--------|--------|--------|--------|-------|
| **Verbose (old)** | 2K | 5K (1×2K context) | 10K (2×2K context) | 17K (3×2K context) | 26K (4×2K context) | **60K tokens** |
| **Scratchpad (new)** | 2K | 2.5K (500 context) | 3K (1K context) | 3.5K (1.5K context) | 4K (2K context) | **15K tokens** |
| **Savings** | 0% | 50% | 70% | 79% | 85% | **75% total** |

### Cost Impact

With GPT-4 Turbo pricing ($0.01/1K input tokens):
- **Old approach**: 60K tokens × $0.01/1K = **$0.60 per execution**
- **New approach**: 15K tokens × $0.01/1K = **$0.15 per execution**
- **Savings**: $0.45 per execution (75% cost reduction)

For workspaces running 1000 recipes/month: **$450/month savings**.

**Sources:** [orchestrator/core/services/recipe_scratchpad.py:1-20]()

---

## Agent Export Tool

Agents can explicitly write key data to the scratchpad using the `scratchpad_write` tool, which is injected into every recipe step:

```mermaid
graph LR
    Agent["Agent Execution"]
    
    Agent -->|"LLM generates tool call"| ToolCall["scratchpad_write({<br/>  key: 'pr_number',<br/>  value: '456'<br/>})"]
    
    ToolCall --> Handler["handle_scratchpad_write()"]
    
    Handler --> Validate["Validate key format<br/>(alphanumeric + underscore)"]
    
    Validate --> Write["scratchpad.write_export(<br/>  step_order=2,<br/>  key='pr_number',<br/>  value='456'<br/>)"]
    
    Write --> Redis["Redis HSET<br/>step_2:exports<br/>{pr_number: '456'}"]
    
    Redis --> NextStep["Step 3 receives context:<br/>'- Exports: {pr_number: 456}'"]
```

### Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "scratchpad_write",
    "description": "Write a key-value pair to the recipe scratchpad for downstream steps to access. Use this to explicitly export important data (IDs, URLs, status codes) that later steps need.",
    "parameters": {
      "type": "object",
      "properties": {
        "key": {
          "type": "string",
          "description": "Key name (alphanumeric + underscore only). Example: pr_number, issue_id, branch_name"
        },
        "value": {
          "type": "string",
          "description": "Value to store. Keep concise—this will be injected into downstream step contexts."
        }
      },
      "required": ["key", "value"]
    }
  }
}
```

This allows agents to bypass auto-extraction for critical data they know downstream steps need.

**Sources:** [orchestrator/modules/tools/builtin/scratchpad_tool.py:1-60](), [orchestrator/api/recipe_executor.py:236-254]()

---

## Code Integration Points

### Initialization in Recipe Executor

[orchestrator/api/recipe_executor.py:632-636]()
```python
from core.services.recipe_scratchpad import RecipeScratchpad
scratchpad = RecipeScratchpad(recipe_execution_id)
scratchpad.write_inputs(input_data)
scratchpad.write_meta(recipe_id, total_steps)
```

### Context Injection Before Step Execution

[orchestrator/api/recipe_executor.py:179-182]()
```python
if scratchpad:
    ctx = scratchpad.format_context_for_step(step_order)
    if ctx:
        messages.append({"role": "system", "content": ctx})
```

### Auto-Extraction After Step Completion

[orchestrator/api/recipe_executor.py:753-759]()
```python
if scratchpad:
    scratchpad.write_step_results(
        step_order=step_order,
        tool_calls=all_tool_calls,
        agent_output=content,
        agent_exports={},
    )
```

### Manual Export During Execution

[orchestrator/api/recipe_executor.py:236-254]()
```python
if tool_name == SCRATCHPAD_TOOL_NAME and scratchpad:
    result_text = handle_scratchpad_write(
        key=tool_args.get("key", "unknown"),
        value=tool_args.get("value", ""),
        scratchpad=scratchpad,
        step_order=step_order,
    )
```

**Sources:** [orchestrator/api/recipe_executor.py:559-777](), [orchestrator/core/services/recipe_scratchpad.py:1-362]()

---