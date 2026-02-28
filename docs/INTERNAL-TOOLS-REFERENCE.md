# Internal Tools Reference & Test Plan

> Source of truth for all 35 internal tools on the Automatos AI platform.
> Use this document to understand tool behavior, write tests, and verify tool execution end-to-end.

**Last updated:** 2026-02-25
**Tool count:** 35 (19 core + 15 platform actions + 1 recipe built-in)

---

## Table of Contents

1. [Tool Inventory](#1-tool-inventory)
2. [Invocation Architecture](#2-invocation-architecture)
3. [Security Model](#3-security-model)
4. [Tool Reference — Research Tools](#4-research-tools-7)
5. [Tool Reference — Database Tools](#5-database-tools-2)
6. [Tool Reference — File Operations](#6-file-operations-5)
7. [Tool Reference — Shell](#7-shell-1)
8. [Tool Reference — HTTP](#8-http-1)
9. [Tool Reference — SSH](#9-ssh-1)
10. [Tool Reference — Composio](#10-composio-1)
11. [Tool Reference — Document Generation](#11-document-generation-1)
12. [Tool Reference — Platform Actions](#12-platform-actions-15)
13. [Tool Reference — Recipe Built-in](#13-recipe-built-in-1)
14. [Test Matrix](#14-test-matrix)

---

## 1. Tool Inventory

| # | Tool Name | Category | Security Level | Permission | Executor |
|---|-----------|----------|---------------|------------|----------|
| 1 | `search_knowledge` | research | safe | read | AgentPlatformTools |
| 2 | `semantic_search` | research | safe | read | AgentPlatformTools |
| 3 | `search_codebase` | research | safe | read | AgentPlatformTools |
| 4 | `search_tables` | research | safe | read | MultimodalKnowledgeTools |
| 5 | `search_images` | research | safe | read | MultimodalKnowledgeTools |
| 6 | `search_formulas` | research | safe | read | MultimodalKnowledgeTools |
| 7 | `search_multimodal` | research | safe | read | MultimodalKnowledgeTools |
| 8 | `query_database` | database | safe | read | UnifiedToolExecutor |
| 9 | `smart_query_database` | database | safe | read | UnifiedToolExecutor |
| 10 | `read_file` | file_ops | safe | read | ActionExecutor |
| 11 | `write_file` | file_ops | cautious | read, write | ActionExecutor |
| 12 | `delete_file` | file_ops | dangerous | read, write, delete | ActionExecutor |
| 13 | `list_directory` | file_ops | safe | read | ActionExecutor |
| 14 | `create_directory` | file_ops | cautious | read, write | ActionExecutor |
| 15 | `execute_command` | shell | dangerous | read, write, execute | ActionExecutor |
| 16 | `http_request` | api | cautious | read, execute | UnifiedToolExecutor |
| 17 | `ssh_execute` | ssh | dangerous | read, write, execute | UnifiedToolExecutor |
| 18 | `composio_execute` | api | cautious | read, execute | ComposioToolExecutor |
| 19 | `generate_document` | file_ops | cautious | read, write | AgentPlatformTools |
| 20 | `platform_list_agents` | agents | — | read | PlatformActionExecutor |
| 21 | `platform_get_agent` | agents | — | read | PlatformActionExecutor |
| 22 | `platform_list_recipes` | recipes | — | read | PlatformActionExecutor |
| 23 | `platform_get_recipe` | recipes | — | read | PlatformActionExecutor |
| 24 | `platform_get_llm_usage` | analytics | — | read | PlatformActionExecutor |
| 25 | `platform_get_cost_breakdown` | analytics | — | read | PlatformActionExecutor |
| 26 | `platform_list_documents` | documents | — | read | PlatformActionExecutor |
| 27 | `platform_get_workspace_info` | workspace | — | read | PlatformActionExecutor |
| 28 | `platform_get_memory_stats` | memory | — | read | PlatformActionExecutor |
| 29 | `platform_list_connected_apps` | integrations | — | read | PlatformActionExecutor |
| 30 | `platform_create_agent` | agents | — | write | PlatformActionExecutor |
| 31 | `platform_update_agent` | agents | — | write | PlatformActionExecutor |
| 32 | `platform_create_recipe` | recipes | — | write | PlatformActionExecutor |
| 33 | `platform_store_memory` | memory | — | write | PlatformActionExecutor |
| 34 | `platform_delete_agent` | agents | — | destructive | PlatformActionExecutor |
| 35 | `scratchpad_write` | recipe | — | write | handle_scratchpad_write |

---

## 2. Invocation Architecture

### Chat Pipeline Tool Flow

```
User Message
  │
  ├─ StreamingChatService.stream_response()
  │   │
  │   ├─ SmartToolRouter.route(query, available_tools)
  │   │   ├─ IntentClassifier.classify(query) → IntentResult
  │   │   ├─ If requires_tools=false → skip tools entirely
  │   │   ├─ If SEMANTIC_TOOL_ROUTING=true → rank by cosine similarity
  │   │   └─ Else → keyword-based category filter
  │   │
  │   ├─ Build OpenAI messages with filtered tools
  │   │
  │   ├─ LLM generates response (may include tool_calls)
  │   │
  │   └─ Tool Loop (max iterations with ToolExecutionTracker):
  │       │
  │       ├─ ToolExecutionTracker.should_skip_execution()
  │       │   ├─ Check per-tool retry limit
  │       │   ├─ Check exact duplicate (tool_name + args_hash)
  │       │   └─ Check semantic duplicate for search tools
  │       │
  │       ├─ UnifiedToolExecutor.execute_tool(tool_name, params)
  │       │   │
  │       │   ├─ If tool_name.startswith("platform_"):
  │       │   │   └─ PlatformActionExecutor.execute()
  │       │   │
  │       │   ├─ If tool_name == "composio_execute":
  │       │   │   └─ ComposioToolExecutor.execute()
  │       │   │
  │       │   └─ Else: self.tool_routes[tool_name]()
  │       │       ├─ _execute_platform_tool  → AgentPlatformTools
  │       │       ├─ _execute_database_tool  → Knowledge API + NL2SQL
  │       │       ├─ _execute_multimodal_tool → MultimodalKnowledgeTools
  │       │       ├─ _execute_file_op        → ActionExecutor
  │       │       ├─ _execute_shell          → ActionExecutor
  │       │       ├─ _execute_http_request   → httpx (domain-whitelisted)
  │       │       ├─ _execute_ssh            → paramiko
  │       │       └─ _execute_generate_document → DocumentGenerationService
  │       │
  │       └─ Append tool result → next LLM turn
  │
  └─ Final assistant message streamed to client
```

### Key Files

| Component | File | Purpose |
|-----------|------|---------|
| Tool Registry | `modules/tools/registry/tool_registry.py` | Core 19 tool definitions (ToolSpec) |
| Action Registry | `modules/tools/discovery/action_registry.py` | 15 platform action definitions (ActionDefinition) |
| Platform Actions | `modules/tools/discovery/platform_actions.py` | Platform action registration |
| Unified Executor | `modules/tools/execution/unified_executor.py` | Execution routing for all tools |
| Platform Executor | `modules/tools/discovery/platform_executor.py` | Database queries for platform actions |
| Smart Tool Router | `consumers/chatbot/smart_tool_router.py` | Intent-based tool filtering |
| Chat Service | `consumers/chatbot/service.py` | Tool loop orchestration |
| Scratchpad Tool | `modules/tools/builtin/scratchpad_tool.py` | Recipe-only scratchpad |

### Tool Loop Safety

The `ToolExecutionTracker` prevents infinite loops with:

| Mechanism | Details |
|-----------|---------|
| Per-tool retry limit | `search_knowledge`: 2, `composio_execute`: 2, `read_file`: 3, default: 3 |
| Exact dedup | Hash of (tool_name + sorted JSON args) |
| Semantic dedup | SequenceMatcher ratio >= 0.75 for search tools |
| Search tools tracked | `search_knowledge`, `semantic_search`, `search_codebase`, `search_tables`, `search_images`, `search_formulas`, `search_multimodal`, `smart_query_database`, `query_database` |

---

## 3. Security Model

### Security Levels

| Level | Meaning | Tools |
|-------|---------|-------|
| `safe` | Read-only, no side effects | All research tools, `read_file`, `list_directory`, `query_database`, `smart_query_database` |
| `cautious` | Writes data, non-destructive | `write_file`, `create_directory`, `http_request`, `composio_execute`, `generate_document` |
| `dangerous` | Destructive or high-risk | `delete_file`, `execute_command`, `ssh_execute` |

### Context Filtering

Tools are filtered by the agent's `active_context` setting:

| Context | Allowed Categories |
|---------|--------------------|
| `general` | communication, research, productivity, system, collaboration, developer, api, file_ops, database |
| `coding` | developer, github, git, code, file_ops, devtools, research |
| `ops` | cloud, k8s, aws, infrastructure, monitoring, database, shell |
| `communication` | communication, slack, email, chat, collaboration |
| `research` | research, data, search, rag, database |

**Exceptions:** `switch_context` and `search_knowledge` are always allowed regardless of context.

### Composio Gate

`composio_execute` requires:
1. Agent has EXTERNAL app assignments (`AgentAppAssignment.app_type == "EXTERNAL"`)
2. Assigned apps are connected for the workspace (`EntityManager.get_connected_apps()`)
3. Intersection of assigned and connected apps is non-empty

---

## 4. Research Tools (7)

### 4.1 `search_knowledge`

| Field | Value |
|-------|-------|
| **Description** | Search the Automatos knowledge base for documentation, guides, and platform info. Has a 2-attempt limit. |
| **Category** | research |
| **Security** | safe |
| **Executor** | `AgentPlatformTools.execute_tool` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Search query |
| `limit` | number | no | 5 | Max results |

**Example:**
```json
{"action": "search_knowledge", "params": {"query": "How to create an agent?", "limit": 5}}
```

**Response:** Array of knowledge base chunks with content, source, and similarity scores.

**Test Scenarios:**
- Happy path: query with known content returns relevant results
- Empty query: should return error or empty results
- `limit=0`: edge case, verify behavior
- `limit=100`: large limit, verify capping
- Non-existent content: returns empty results gracefully
- 2-attempt limit enforcement per turn

---

### 4.2 `semantic_search`

| Field | Value |
|-------|-------|
| **Description** | Find semantically similar content across all platform documents using vector embeddings. |
| **Category** | research |
| **Security** | safe |
| **Executor** | `AgentPlatformTools.execute_tool` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Concept or topic to find similar content for |
| `limit` | number | no | 5 | Max results |

**Example:**
```json
{"action": "semantic_search", "params": {"query": "authentication patterns", "limit": 5}}
```

**Test Scenarios:**
- Happy path: conceptual query returns semantically related content
- Very short query (single word): should still return results
- Query in non-English: verify behavior
- No documents in workspace: returns empty gracefully

---

### 4.3 `search_codebase`

| Field | Value |
|-------|-------|
| **Description** | Search indexed codebases for functions, classes, and implementations. |
| **Category** | research |
| **Security** | safe |
| **Executor** | `AgentPlatformTools.execute_tool` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Code pattern, function name, or concept |
| `file_type` | string | no | — | Filter by extension (e.g., `py`, `ts`) |
| `project_name` | string | no | `Automatos-ai` | Project name to search |

**Example:**
```json
{"action": "search_codebase", "params": {"query": "authenticate_user", "project_name": "Automatos-ai"}}
```

**Test Scenarios:**
- Happy path: known function name returns code results
- File type filter: `file_type=py` narrows results
- Non-existent project: returns error or empty
- Special characters in query: handles regex-like input safely

---

### 4.4 `search_tables`

| Field | Value |
|-------|-------|
| **Description** | Search for tables and structured data extracted from documents. Returns tables in Markdown, CSV, and JSON formats. |
| **Category** | research |
| **Security** | safe |
| **Executor** | `MultimodalKnowledgeTools.search_tables` |
| **Added in** | PRD-19 |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What kind of table or data to find |
| `limit` | number | no | 5 | Max tables to return |

**Example:**
```json
{"action": "search_tables", "params": {"query": "API response times", "limit": 3}}
```

**Test Scenarios:**
- Happy path: query matches extracted table content
- No tables in knowledge base: returns empty gracefully
- Table format verification: response includes markdown/CSV/JSON representations

---

### 4.5 `search_images`

| Field | Value |
|-------|-------|
| **Description** | Search for images, diagrams, and charts with AI-generated descriptions and OCR text. |
| **Category** | research |
| **Security** | safe |
| **Executor** | `MultimodalKnowledgeTools.search_images` |
| **Added in** | PRD-19 |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | What kind of image/diagram to find |
| `limit` | number | no | 5 | Max images to return |

**Example:**
```json
{"action": "search_images", "params": {"query": "database schema diagram", "limit": 3}}
```

**Test Scenarios:**
- Happy path: returns image metadata with descriptions
- No images indexed: returns empty
- Response includes OCR text and AI description fields

---

### 4.6 `search_formulas`

| Field | Value |
|-------|-------|
| **Description** | Search for mathematical formulas and equations. Returns LaTeX format with variable and operator extraction. |
| **Category** | research |
| **Security** | safe |
| **Executor** | `MultimodalKnowledgeTools.search_formulas` |
| **Added in** | PRD-19 |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Mathematical concept or formula type |
| `limit` | number | no | 5 | Max formulas to return |

**Example:**
```json
{"action": "search_formulas", "params": {"query": "Shannon entropy", "limit": 3}}
```

**Test Scenarios:**
- Happy path: returns LaTeX formulas
- No formulas indexed: returns empty
- Response includes variable/operator extraction

---

### 4.7 `search_multimodal`

| Field | Value |
|-------|-------|
| **Description** | Unified search across ALL knowledge types: documents, code, tables, images, formulas. |
| **Category** | research |
| **Security** | safe |
| **Executor** | `MultimodalKnowledgeTools.search_multimodal` |
| **Added in** | PRD-19 |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Research query spanning multiple content types |
| `kb_types` | array | no | `["document", "table", "image", "formula", "codegraph"]` | Knowledge types to search |
| `limit` | number | no | 10 | Max total results across all types |

**Example:**
```json
{"action": "search_multimodal", "params": {"query": "authentication system", "kb_types": ["document", "codegraph", "image"], "limit": 10}}
```

**Test Scenarios:**
- Happy path: returns mixed results from multiple types
- Single `kb_types` filter: only that type returned
- Empty `kb_types`: verify default behavior
- Invalid kb_type value: graceful error

---

## 5. Database Tools (2)

### 5.1 `query_database`

| Field | Value |
|-------|-------|
| **Description** | Query databases using natural language. Converts to SQL via NL-to-SQL. Supports PandasAI chart/insight generation. |
| **Category** | database |
| **Security** | safe |
| **Executor** | `UnifiedToolExecutor._execute_database_tool` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Natural language query |
| `database_name` | string | no | — | Specific database/knowledge source |
| `analysis_prompt` | string | no | — | PandasAI prompt for insights/charts |

**Example:**
```json
{"action": "query_database", "params": {"query": "Show failed workflows in the last 14 days"}}
```

**Response schema:**
```json
{
  "success": true,
  "database": "automatos_main",
  "sql": "SELECT ...",
  "row_count": 5,
  "data": [{"col1": "val1"}],
  "columns": ["col1"],
  "execution_time_ms": 120,
  "pandas_ai": {"insight": "..."}
}
```

**Execution flow:**
1. List knowledge database sources via Knowledge API
2. If `database_name` specified, select matching source
3. If no sources found, fallback to direct main DB NL-to-SQL
4. Generate SQL via LLM, execute, return results
5. Optionally run PandasAI insight generation

**Test Scenarios:**
- Happy path: NL query returns data with SQL
- No database sources: falls back to main DB
- Invalid SQL generated: returns error
- Only SELECT allowed: verify INSERT/UPDATE/DELETE rejected
- PandasAI insight: verify `analysis_prompt` triggers insight
- Large result set: verify 1000-row cap
- 2-attempt limit per turn

---

### 5.2 `smart_query_database`

| Field | Value |
|-------|-------|
| **Description** | Intelligent database query with clarification, rephrasing, explanation, and visualization suggestions. |
| **Category** | database |
| **Security** | safe |
| **Executor** | `UnifiedToolExecutor._execute_smart_database_tool` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | — | Natural language query |
| `database_name` | string | no | — | Specific database |
| `skip_clarification` | boolean | no | false | Skip clarification questions |
| `clarification_answers` | object | no | — | Answers to previous clarification questions |
| `include_visualization` | boolean | no | true | Include visualization suggestions |

**Example:**
```json
{"action": "smart_query_database", "params": {"query": "Show me sales trends"}}
```

**Response schema (success):**
```json
{
  "success": true,
  "database": "automatos_main",
  "sql": "SELECT ...",
  "row_count": 10,
  "data": [...],
  "columns": [...],
  "execution_time_ms": 0,
  "explanation": "This shows...",
  "rephrased_query": "...",
  "visualization": {"chart_type": "line", ...},
  "follow_up_questions": [...],
  "pandas_ai": {...}
}
```

**Response schema (needs clarification):**
```json
{
  "success": true,
  "status": "needs_clarification",
  "clarifications": [{"question": "...", "options": [...]}],
  "message": "Please provide more details...",
  "original_query": "..."
}
```

**Test Scenarios:**
- Happy path: returns data with explanation
- Ambiguous query: returns `needs_clarification` status
- Multi-turn: pass `clarification_answers` → query completes
- `skip_clarification=true`: bypasses clarification
- `include_visualization=false`: no visualization in response
- Schema error: returns `success: false` with error
- 2-attempt limit per turn

---

## 6. File Operations (5)

### 6.1 `read_file`

| Field | Value |
|-------|-------|
| **Description** | Read contents of a file from the workspace. |
| **Category** | file_ops |
| **Security** | safe |
| **Executor** | `ActionExecutor.read_file` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `file_path` | string | yes | — | Path relative to workspace |
| `encoding` | string | no | `utf-8` | File encoding |

**Example:**
```json
{"action": "read_file", "params": {"file_path": "config.json"}}
```

**Response:**
```json
{
  "success": true,
  "action": "read_file",
  "params": {...},
  "requested_path": "config.json",
  "workspace": "/tmp/automatos_workspace",
  "result": "file contents here"
}
```

**Test Scenarios:**
- Happy path: read existing file
- File not found: `success: false` with error
- Path traversal attempt (`../../etc/passwd`): must be blocked
- Binary file: verify handling
- Large file: verify no OOM
- Different encoding: `encoding=latin-1`

---

### 6.2 `write_file`

| Field | Value |
|-------|-------|
| **Description** | Write content to a file in the workspace (creates if doesn't exist). |
| **Category** | file_ops |
| **Security** | cautious |
| **Executor** | `ActionExecutor.write_file` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `file_path` | string | yes | — | Path relative to workspace |
| `content` | string | yes | — | Content to write |
| `encoding` | string | no | `utf-8` | File encoding |

**Example:**
```json
{"action": "write_file", "params": {"file_path": "output.txt", "content": "Hello World"}}
```

**Test Scenarios:**
- Happy path: write new file, verify content
- Overwrite existing file
- Path traversal: blocked
- Empty content: creates empty file
- Nested path (`a/b/c.txt`): creates parent dirs or errors
- Very large content: verify handling

---

### 6.3 `delete_file`

| Field | Value |
|-------|-------|
| **Description** | Delete a file or directory from the workspace. |
| **Category** | file_ops |
| **Security** | dangerous |
| **Permissions** | read, write, delete |
| **Executor** | `ActionExecutor.delete_file` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `file_path` | string | yes | — | Path to file/directory to delete |

**Test Scenarios:**
- Happy path: delete existing file
- Delete directory
- File not found: error
- Path traversal: blocked
- Delete workspace root: must be blocked

---

### 6.4 `list_directory`

| Field | Value |
|-------|-------|
| **Description** | List contents of a directory in the workspace. |
| **Category** | file_ops |
| **Security** | safe |
| **Executor** | `ActionExecutor.list_directory` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `dir_path` | string | no | `.` | Directory path |

**Test Scenarios:**
- Happy path: list files in directory
- Empty directory: returns empty list
- Non-existent directory: error
- Default (no `dir_path`): lists workspace root
- Path traversal: blocked

---

### 6.5 `create_directory`

| Field | Value |
|-------|-------|
| **Description** | Create a new directory in the workspace. |
| **Category** | file_ops |
| **Security** | cautious |
| **Executor** | `ActionExecutor.create_directory` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `dir_path` | string | yes | — | Path to directory to create |

**Test Scenarios:**
- Happy path: create new directory
- Nested path: `a/b/c` — verify behavior (recursive or error)
- Already exists: no error or specific error
- Path traversal: blocked

---

## 7. Shell (1)

### 7.1 `execute_command`

| Field | Value |
|-------|-------|
| **Description** | Execute a shell command in the sandboxed workspace (whitelisted commands only). |
| **Category** | shell |
| **Security** | dangerous |
| **Permissions** | read, write, execute |
| **Executor** | `ActionExecutor.execute_command` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `command` | string | yes | — | Shell command (must be whitelisted) |
| `timeout` | number | no | 30 | Timeout in seconds |

**Whitelisted commands:** `ls`, `cat`, `grep`, `find`, `git`, `python`, `npm`, `docker`

**Example:**
```json
{"action": "execute_command", "params": {"command": "ls -la", "timeout": 10}}
```

**Test Scenarios:**
- Happy path: `ls -la` returns directory listing
- Non-whitelisted command (`rm -rf /`): must be rejected
- Timeout: command exceeding timeout returns error
- Command injection attempt (`ls; rm -rf /`): blocked
- Empty command: error
- `timeout=0`: verify behavior

---

## 8. HTTP (1)

### 8.1 `http_request`

| Field | Value |
|-------|-------|
| **Description** | Make HTTP requests to whitelisted internal/platform URLs. Domain-restricted to prevent SSRF. |
| **Category** | api |
| **Security** | cautious |
| **Permissions** | read, execute |
| **Executor** | `UnifiedToolExecutor._execute_http_request` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `url` | string | yes | — | Full URL (must be whitelisted domain) |
| `method` | string | no | `GET` | HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS |
| `headers` | object | no | `{}` | Request headers |
| `body` | object | no | `{}` | Request body (JSON, for POST/PUT/PATCH) |
| `timeout` | number | no | 30 | Timeout in seconds (max: 120) |

**Allowed domains:**
- `automatos-ai.railway.internal`
- `automatos-ai-frontend.railway.internal`
- `api.automatos.app`
- `ui.automatos.app`
- `localhost`
- `127.0.0.1`

**Response schema:**
```json
{
  "success": true,
  "tool": "http_request",
  "status_code": 200,
  "response_headers": {...},
  "body": {...},
  "duration_ms": 45,
  "url": "...",
  "method": "GET"
}
```

**Test Scenarios:**
- Happy path: GET to `/health` returns 200
- Non-whitelisted domain (`google.com`): rejected with error
- POST with JSON body: body sent correctly
- Timeout: returns timeout error
- Connection refused: returns connection error
- Invalid URL (no hostname): rejected
- Invalid HTTP method: rejected
- Max response body: capped at 1MB
- `timeout` > 120: clamped to 120
- HEAD request: no body in response

---

## 9. SSH (1)

### 9.1 `ssh_execute`

| Field | Value |
|-------|-------|
| **Description** | Execute commands on a remote server via SSH. Supports password, private key, and stored credential auth. |
| **Category** | ssh |
| **Security** | dangerous |
| **Permissions** | read, write, execute |
| **Executor** | `UnifiedToolExecutor._execute_ssh` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `host` | string | yes | — | SSH hostname or IP |
| `command` | string | yes | — | Shell command to execute remotely |
| `username` | string | no | `automatos` | SSH username |
| `port` | number | no | 22 | SSH port |
| `password` | string | no | — | SSH password |
| `private_key` | string | no | — | PEM-format private key |
| `credential_id` | string | no | — | Stored credential ID (preferred) |
| `timeout` | number | no | 60 | Command timeout (max: 300) |

**Response schema:**
```json
{
  "success": true,
  "tool": "ssh_execute",
  "exit_code": 0,
  "stdout": "...",
  "stderr": "...",
  "host": "...",
  "command": "...",
  "duration_ms": 1200
}
```

**Auth resolution order:**
1. If `credential_id` provided → look up `StoredCredential` (workspace-scoped)
2. If `private_key` provided → parse PEM (RSA or Ed25519)
3. If `password` provided → password auth
4. If none → error

**Test Scenarios:**
- Happy path: execute command on reachable server
- Missing host: error
- Missing command: error
- No auth provided: error with clear message
- Invalid credential_id: error
- Auth failure: returns `SSH authentication failed`
- Timeout: returns timeout error
- Large stdout: truncated at 500KB
- `timeout` > 300: clamped to 300
- Paramiko not installed: returns specific error

---

## 10. Composio (1)

### 10.1 `composio_execute`

| Field | Value |
|-------|-------|
| **Description** | Execute an external app action via Composio (connected third-party apps). Action-specific params go inside the `params` object. |
| **Category** | api |
| **Security** | cautious |
| **Permissions** | read, execute |
| **Executor** | `ComposioToolExecutor.execute` |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `app_name` | string | no | — | App name (e.g., `GMAIL`, `SLACK`, `GITHUB`) |
| `action` | string | yes | — | Action name (e.g., `GMAIL_LIST_EMAILS`) |
| `params` | object | no | `{}` | Action-specific parameters |

**Example:**
```json
{
  "action": "composio_execute",
  "params": {
    "action": "SLACK_SEND_MESSAGE",
    "params": {"channel": "#general", "text": "Hello"}
  }
}
```

**Access control:**
- Requires workspace context (`workspace_id`)
- Agent must have active EXTERNAL app assignments
- Assigned apps must be connected for the workspace
- Action name auto-uppercased for consistency
- Stray top-level params remapped into `params` (defensive LLM handling)

**Test Scenarios:**
- Happy path: execute connected app action
- Missing `action` field: error
- No workspace_id: error
- Agent has no external app assignments: access denied
- App not connected: access denied
- Stray params at top level: remapped into `params`
- Case-insensitive action name: `slack_send_message` → `SLACK_SEND_MESSAGE`
- Composio executor not available: error
- PRD-37 capability validation: action blocked if it doesn't match intent

---

## 11. Document Generation (1)

### 11.1 `generate_document`

| Field | Value |
|-------|-------|
| **Description** | Generate a polished PDF, DOCX, or XLSX document from structured data. Returns a download URL. |
| **Category** | file_ops |
| **Security** | cautious |
| **Permissions** | read, write |
| **Executor** | `AgentPlatformTools.execute_tool` |
| **Added in** | PRD-63 |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `title` | string | yes | — | Document title |
| `format` | string | yes | — | Output format: `pdf`, `docx`, or `xlsx` |
| `data` | object | yes | — | Data to populate (sections for reports, rows/columns for tables) |
| `template_name` | string | no | — | Template name (auto-selected if omitted) |

**Data formats:**

For reports (PDF/DOCX):
```json
{
  "sections": [
    {"title": "Section Name", "content": "Full paragraph text..."}
  ],
  "author": "...",
  "date": "..."
}
```

For tables (XLSX):
```json
{
  "columns": ["col1", "col2"],
  "rows": [["val1", "val2"]]
}
```

**Response schema:**
```json
{
  "success": true,
  "filename": "...",
  "format": "pdf",
  "download_url": "...",
  "size_kb": 42
}
```

**Test Scenarios:**
- Happy path: generate PDF with sections
- Happy path: generate XLSX with tabular data
- Missing `title`: error
- Missing `format`: error
- Invalid format (`pptx`): error
- Missing `data`: error
- Empty sections: verify behavior
- Template not found: auto-selects default

---

## 12. Platform Actions (15)

Platform actions are routed by the `platform_` prefix in tool name. They execute via `PlatformActionExecutor` with direct database queries scoped to the workspace.

### Permission Levels

| Level | Actions | Confirmation |
|-------|---------|-------------|
| **read** | list_agents, get_agent, list_recipes, get_recipe, get_llm_usage, get_cost_breakdown, list_documents, get_workspace_info, get_memory_stats, list_connected_apps | No |
| **write** | create_agent, update_agent, create_recipe, store_memory | No |
| **destructive** | delete_agent | Yes (requires_confirmation=true) |

---

### 12.1 `platform_list_agents`

| Field | Value |
|-------|-------|
| **Category** | agents |
| **Permission** | read |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `status_filter` | string (enum: active, inactive, all) | no | Filter by status. Default: `all` |

**Response:** `{ success, agents: [{id, name, type, status, description, created_at}], count }`

**Test Scenarios:**
- Returns all agents in workspace
- `status_filter=active`: only active agents
- Empty workspace: returns `count: 0`

---

### 12.2 `platform_get_agent`

| Field | Value |
|-------|-------|
| **Category** | agents |
| **Permission** | read |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `agent_name` | string | no | Name of agent (fuzzy match via ILIKE) |
| `agent_id` | integer | no | Agent ID |

**Response:** `{ success, agent: {id, name, type, status, description, model, provider, assigned_tools, tags, created_at, updated_at} }`

**Test Scenarios:**
- By name: partial match works
- By ID: exact match
- Neither provided: error
- Agent not found: error
- Cross-workspace: cannot access other workspace's agents

---

### 12.3 `platform_list_recipes`

| Field | Value |
|-------|-------|
| **Category** | recipes |
| **Permission** | read |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `status_filter` | string (enum: active, inactive, all) | no | Default: `all` |

**Response:** `{ success, recipes: [{id, name, template_id, description, tags, created_at}], count }`

---

### 12.4 `platform_get_recipe`

| Field | Value |
|-------|-------|
| **Category** | recipes |
| **Permission** | read |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `recipe_name` | string | no | Recipe name (fuzzy) |
| `recipe_id` | integer | no | Recipe ID |

**Response:** `{ success, recipe: {id, name, template_id, description, tags, step_count, steps: [{name, type}], total_executions} }`

---

### 12.5 `platform_get_llm_usage`

| Field | Value |
|-------|-------|
| **Category** | analytics |
| **Permission** | read |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `days` | integer | no | Lookback days. Default: 30 |

**Response:** `{ success, period_days, total_requests, total_tokens, by_model: [{model, provider, requests, input_tokens, output_tokens, total_tokens}] }`

---

### 12.6 `platform_get_cost_breakdown`

| Field | Value |
|-------|-------|
| **Category** | analytics |
| **Permission** | read |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `days` | integer | no | Lookback days. Default: 30 |
| `group_by` | string (enum: model, agent, day) | no | Grouping. Default: `model` |

**Response:** `{ success, period_days, group_by, total_cost, breakdown: [{<group_by>: key, total_cost, input_cost, output_cost, requests}] }`

---

### 12.7 `platform_list_documents`

| Field | Value |
|-------|-------|
| **Category** | documents |
| **Permission** | read |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `limit` | integer | no | Max documents. Default: 50, max: 200 |

**Response:** `{ success, documents: [{id, filename, file_type, file_size, status, chunk_count, uploaded_at}], count }`

---

### 12.8 `platform_get_workspace_info`

| Field | Value |
|-------|-------|
| **Category** | workspace |
| **Permission** | read |

**Parameters:** None

**Response:** `{ success, workspace: {id, name, plan, is_personal, agent_count, document_count, created_at} }`

---

### 12.9 `platform_get_memory_stats`

| Field | Value |
|-------|-------|
| **Category** | memory |
| **Permission** | read |

**Parameters:** None

**Response:** `{ success, total_memories, workspace_id }` or `{ success, total_memories: 0, message: "..." }` if mem0 unavailable.

---

### 12.10 `platform_list_connected_apps`

| Field | Value |
|-------|-------|
| **Category** | integrations |
| **Permission** | read |

**Parameters:** None

**Response:** `{ success, connected_apps: [{app_name, app_type, assigned_to_agents}], count }`

---

### 12.11 `platform_create_agent`

| Field | Value |
|-------|-------|
| **Category** | agents |
| **Permission** | write |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | Agent name |
| `agent_type` | string (enum: chatbot, worker, researcher, coder) | no | Default: `chatbot` |
| `description` | string | no | Agent description |
| `model` | string | no | LLM model (e.g., `gpt-4o`) |

**Response:** `{ success, agent: {id, name, type, status, description}, message }`

**Test Scenarios:**
- Happy path: create agent with name
- Missing name: error
- All optional params: agent created with full config
- Duplicate name: verify behavior (allowed or rejected)

---

### 12.12 `platform_update_agent`

| Field | Value |
|-------|-------|
| **Category** | agents |
| **Permission** | write |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `agent_id` | integer | no | Agent ID |
| `agent_name` | string | no | Current name (lookup) |
| `new_name` | string | no | Rename |
| `description` | string | no | New description |
| `status` | string (enum: active, inactive) | no | New status |

**Response:** `{ success, agent_id, changes: [...], message }`

**Test Scenarios:**
- Rename agent
- Deactivate agent
- No changes specified: returns "No changes specified"
- Agent not found: error

---

### 12.13 `platform_create_recipe`

| Field | Value |
|-------|-------|
| **Category** | recipes |
| **Permission** | write |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | Recipe name |
| `description` | string | yes | What the recipe does |
| `tags` | array of strings | no | Categorization tags |

**Response:** `{ success, recipe: {id, name, template_id, description}, message }`

---

### 12.14 `platform_store_memory`

| Field | Value |
|-------|-------|
| **Category** | memory |
| **Permission** | write |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `content` | string | yes | Information to remember |

**Response:** `{ success, message }` or error if mem0 unavailable.

**Test Scenarios:**
- Happy path: memory stored
- Missing content: error
- Mem0 not configured: error
- Mem0 API failure: error with status code

---

### 12.15 `platform_delete_agent`

| Field | Value |
|-------|-------|
| **Category** | agents |
| **Permission** | destructive |
| **Requires Confirmation** | YES |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `agent_id` | integer | no | Agent ID |
| `agent_name` | string | no | Agent name |

**Response (blocked):** `{ success: false, requires_confirmation: true, action, permission_level: "destructive", message, params }`

**Response (confirmed):** `{ success: true, deleted_agent: {id, name}, message }`

**Test Scenarios:**
- Confirmation required: first call returns `requires_confirmation: true`
- Agent not found: error
- Neither ID nor name: error
- Cross-workspace isolation: cannot delete another workspace's agent

---

## 13. Recipe Built-in (1)

### 13.1 `scratchpad_write`

| Field | Value |
|-------|-------|
| **Description** | Store a named value in the shared recipe scratchpad for downstream steps. Only available during recipe execution. |
| **Injected by** | Recipe step executor (not in ToolRegistry) |
| **Executor** | `handle_scratchpad_write()` in `modules/tools/builtin/scratchpad_tool.py` |

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `key` | string | yes | Short snake_case name (e.g., `pr_url`, `branch_name`) |
| `value` | string | yes | The value to store |

**OpenAI function schema:**
```json
{
  "type": "function",
  "function": {
    "name": "scratchpad_write",
    "description": "Store a named value in the shared recipe scratchpad for other steps to use.",
    "parameters": {
      "type": "object",
      "properties": {
        "key": {"type": "string", "description": "A short snake_case name"},
        "value": {"type": "string", "description": "The value to store"}
      },
      "required": ["key", "value"]
    }
  }
}
```

**Returns:** `"Stored '<key>' in scratchpad."`

**Test Scenarios:**
- Happy path: store key-value, verify in scratchpad
- Missing key: error
- Missing value: error
- Overwrite existing key: verify behavior
- Only available in recipe context: not in chat tools

---

## 14. Test Matrix

Use this checklist per tool. Mark each cell with PASS / FAIL / SKIP / N/A.

### Template

| Test Case | Description | Expected | Status |
|-----------|-------------|----------|--------|
| **Happy path** | Valid params, normal execution | Success response with data | |
| **Missing required params** | Omit each required param | Error with clear message | |
| **Invalid param type** | Wrong type (string instead of number) | Error or graceful handling | |
| **Empty string params** | Pass `""` for required strings | Error or handled | |
| **Boundary values** | `limit=0`, `limit=999999`, `timeout=0` | Capped or error | |
| **Permission denied** | Agent without tool access | Blocked by ToolRegistry | |
| **Context filtering** | Tool not in agent's active_context | Filtered out by SmartToolRouter | |
| **Security boundary** | Path traversal, SSRF, injection | Blocked | |
| **Retry limit** | Call same tool > limit times | Blocked by ToolExecutionTracker | |
| **Semantic dedup** | Similar query on search tool | Blocked by ToolExecutionTracker | |
| **Error handling** | Executor throws exception | `{success: false, error: "..."}` | |
| **Workspace isolation** | Access data from another workspace | Blocked (data not returned) | |

### Per-Tool Quick Reference

| Tool | Happy | Missing | Invalid | Security | Retry | Isolation |
|------|-------|---------|---------|----------|-------|-----------|
| search_knowledge | | | | | 2 max | |
| semantic_search | | | | | 2 max | |
| search_codebase | | | | | 2 max | |
| search_tables | | | | | 2 max | |
| search_images | | | | | 2 max | |
| search_formulas | | | | | 2 max | |
| search_multimodal | | | | | 2 max | |
| query_database | | | | | 2 max | |
| smart_query_database | | | | | 2 max | |
| read_file | | | | path traversal | 3 max | |
| write_file | | | | path traversal | 2 max | |
| delete_file | | | | path traversal | 3 max | |
| list_directory | | | | path traversal | 2 max | |
| create_directory | | | | path traversal | 3 max | |
| execute_command | | | | whitelist, injection | 3 max | |
| http_request | | | | domain whitelist, SSRF | 3 max | |
| ssh_execute | | | | auth required | 3 max | |
| composio_execute | | | | app gate, capability | 2 max | workspace |
| generate_document | | | | | 3 max | |
| platform_list_agents | | | | | — | workspace |
| platform_get_agent | | | | | — | workspace |
| platform_list_recipes | | | | | — | workspace |
| platform_get_recipe | | | | | — | workspace |
| platform_get_llm_usage | | | | | — | workspace |
| platform_get_cost_breakdown | | | | | — | workspace |
| platform_list_documents | | | | | — | workspace |
| platform_get_workspace_info | | | | | — | workspace |
| platform_get_memory_stats | | | | | — | workspace |
| platform_list_connected_apps | | | | | — | workspace |
| platform_create_agent | | | | | — | workspace |
| platform_update_agent | | | | | — | workspace |
| platform_create_recipe | | | | | — | workspace |
| platform_store_memory | | | | | — | workspace |
| platform_delete_agent | | | | confirmation | — | workspace |
| scratchpad_write | | | | recipe-only | — | recipe |
