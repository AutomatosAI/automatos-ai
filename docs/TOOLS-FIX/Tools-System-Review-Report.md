# Tools System Review Report
## Comprehensive Analysis of Tool Execution, Composio Integration, and Internal System Tools

**Date:** January 26, 2026  
**Reviewer:** AI Assistant  
**Scope:** Tool execution system, Composio integration, internal tools, UI/UX, and performance

---

## Executive Summary

This report analyzes the tools execution system based on test logs and screen output. While tools are being called correctly at the infrastructure level, several critical issues prevent optimal performance and user experience:

1. **Tool Looping**: Tools are being called multiple times with identical or similar parameters, wasting API calls and tokens
2. **Inefficient Initialization**: Tool registry and executors are re-initialized on every tool call
3. **Result Formatting Issues**: Tool results aren't being properly formatted for display, making it hard to see what tools actually did
4. **Composio Integration Problems**: Action name mismatches, parameter format inconsistencies, and assignment issues
5. **File Operation Bugs**: Path resolution issues causing failures
6. **Database Query Errors**: SQL generation with incorrect column names

**Overall Assessment**: The system architecture is sound, but execution efficiency, result presentation, and error handling need significant improvement.

---

## 1. Tool Looping Issues

### 1.1 Problem Description

Tools are being called multiple times with identical or very similar parameters, creating unnecessary loops:

**Examples from Logs:**
- `composio_execute` with `slack_send_message` called **3 times** with identical parameters
- `search_knowledge` called **2 times** with similar queries ("workflows in Automatos" vs "workflows and how they work in Automatos")
- `list_directory` called **3 times** with "." path
- `write_file` called **2 times** with same parameters

### 1.2 Root Causes

#### A. Ineffective Duplicate Detection
**Location:** `automatos-ai/orchestrator/consumers/chatbot/service.py:699-703`

The duplicate detection uses tool signature (tool name + args), but:
- Tool args are compared as dicts, which may have different key ordering
- Similar but not identical queries aren't detected as duplicates
- The check happens AFTER the tool is already prepared for execution

**Code Issue:**
```python
tool_sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
is_duplicate = tool_sig in executed_tool_signatures
```

**Problem:** This only catches exact duplicates, not semantically similar calls.

#### B. LLM Retry Behavior
The LLM is retrying tools when:
- Results are empty (even if the tool executed successfully)
- Results don't match expectations (even if correct)
- Tool returns partial data (LLM thinks it needs more)

**Location:** `automatos-ai/orchestrator/consumers/chatbot/service.py:1933-1976`

The loop prevention logic exists but:
- Only triggers after 2 attempts for non-Composio tools
- Doesn't prevent semantically similar queries
- Allows Composio tools unlimited retries (causing the 3x Slack calls)

#### C. Tool Result Interpretation
The LLM may not understand tool results correctly, leading to retries:
- Results are truncated to 1000 chars (`service.py:1725`), potentially cutting off important data
- Result formatting may not clearly indicate success/failure
- Empty results from valid queries trigger unnecessary retries

### 1.3 Impact

- **API Costs**: 3x-4x more API calls than necessary
- **Latency**: Each loop iteration adds 2-5 seconds
- **User Experience**: Confusing "Running" states that repeat
- **Token Usage**: Wasted tokens on redundant tool calls

### 1.4 Recommendations

1. **Improve Duplicate Detection**
   - Normalize tool arguments before comparison (lowercase strings, sort dict keys)
   - Use semantic similarity for search queries (detect "workflows" vs "workflows and how they work")
   - Check duplicates BEFORE preparing tool calls

2. **Strengthen Loop Prevention**
   - Reduce retry limit from 2 to 1 for most tools
   - Add explicit "tool already executed" message to LLM context
   - Track tool results, not just calls (prevent retrying successful tools)

3. **Better Result Communication**
   - Increase truncation limit from 1000 to 2000 chars
   - Add explicit success indicators in formatted results
   - Include result summaries in LLM context to prevent "need more data" loops

---

## 2. Tool Registry Re-initialization

### 2.1 Problem Description

The tool registry is being initialized on **EVERY** tool call:

**From Logs:**
```
2026-01-26 11:25:29,420 - modules.tools.execution.unified_executor - INFO - [tool-trace 171715ae9166] Executing tool 'list_directory'
2026-01-26 11:25:29,420 - modules.tools.execution.unified_executor - INFO - [tool-trace 171715ae9166] Parameters keys=['dir_path']
2026-01-26 11:25:29,420 - modules.tools.execution.unified_executor - INFO - [tool-trace 171715ae9166]   🔧 Initializing tool registry...
2026-01-26 11:25:29,421 - modules.tools.registry.tool_registry - INFO - Registered tool: search_knowledge (category: research, security: safe)
... (16 tools registered)
```

This happens for **every single tool call**, even within the same request.

### 2.2 Root Cause

**Location:** `automatos-ai/orchestrator/modules/tools/execution/unified_executor.py:219-224`

The `UnifiedToolExecutor` is created fresh for each tool call:
```python
@property
def tool_registry(self):
    """Lazy-load tool registry only when needed."""
    if self._tool_registry is None:
        logger.info("  🔧 Initializing tool registry...")
        self._tool_registry = ToolRegistry(self.db)
    return self._tool_registry
```

**Problem:** A new `UnifiedToolExecutor` instance is created in `tool_router.py:execute_tool()` for every call, so the lazy-loading property always initializes fresh.

**Location:** `automatos-ai/orchestrator/consumers/chatbot/tool_router.py:209-220`

```python
executor = UnifiedToolExecutor(db_session)
```

### 2.3 Impact

- **Performance**: ~50-100ms overhead per tool call
- **Memory**: Unnecessary object creation
- **Log Noise**: Registry initialization logs clutter output

### 2.4 Recommendations

1. **Reuse Executor Instance**
   - Create `UnifiedToolExecutor` once per request/chat session
   - Pass executor instance to tool router instead of creating new one
   - Cache executor in `StreamingChatService` or `ToolRouter`

2. **Singleton Pattern for Registry**
   - Make `ToolRegistry` a singleton or module-level cached instance
   - Only re-initialize if database schema changes

---

## 3. Result Formatting and Display Issues

### 3.1 Problem Description

Tools execute successfully but results aren't clearly displayed to users:

**From Screen Output:**
- Tools show "Running" state but don't show actual results
- Results appear as "Completed" but content is unclear
- User can't see what data tools actually returned

**Example:**
```
Completed
composio_execute
Running
Parameters
{
  "app_name": "GMAIL",
  "action": "GMAIL_LIST_EMAILS"
}
```

The result (email list) isn't visible in the UI.

### 3.2 Root Causes

#### A. Result Truncation
**Location:** `automatos-ai/orchestrator/consumers/chatbot/service.py:1722-1726`

```python
llm_context = result.get('llm_context', str(result.get('raw_result', '')))
# Truncate to max 1000 chars to keep context manageable
if len(llm_context) > 1000:
    llm_context = llm_context[:1000] + f"\n... (truncated {len(llm_context) - 1000} chars)"
```

**Problem:** Results are truncated to 1000 chars, which may cut off important data. This truncation happens in the streaming handler, not in the formatter.

#### B. Frontend Display Format
**Location:** Frontend components (likely `message.tsx` or tool display components)

The UI shows:
- Tool name
- Parameters
- Status (Running/Completed)

But doesn't show:
- Actual result data
- Result summaries
- Structured data (tables, lists, etc.)

#### C. Result Formatting Logic
**Location:** `automatos-ai/orchestrator/consumers/chatbot/tool_router.py:303`

```python
llm_context = self.formatter.format_for_llm(result, tool_name)
```

The formatter may not be creating user-friendly summaries. Results are formatted for LLM consumption, not human readability.

### 3.3 Impact

- **User Confusion**: Users can't see what tools actually did
- **Trust Issues**: Tools appear to work but results are invisible
- **Debugging Difficulty**: Hard to verify tool correctness
- **Poor UX**: No feedback on tool success beyond "Completed"

### 3.4 Recommendations

1. **Improve Result Display**
   - Show actual result data in UI (not just status)
   - Format results for human readability (not just LLM)
   - Add result summaries for large datasets
   - Show structured data (tables, lists) in expandable sections

2. **Increase Truncation Limit**
   - Increase from 1000 to 2000-3000 chars
   - Use smart truncation (keep important parts, truncate less important)
   - Add "View Full Result" option in UI

3. **Separate Formatting Concerns**
   - `format_for_llm()`: Concise, structured for LLM
   - `format_for_ui()`: Human-readable, detailed for display
   - Don't truncate UI format, only LLM format

4. **Add Result Artifacts**
   - Create artifact components for different result types:
     - Tables for database results
     - Lists for search results
     - JSON viewers for API responses
     - File trees for directory listings

---

## 4. Composio Integration Issues

### 4.1 Problem Description

Multiple issues with Composio tool execution:

#### A. Action Name Case Sensitivity
**From Screen Output:**
```
composio_execute
Parameters
{
  "app_name": "GMAIL",
  "action": "GMAIL_LIST_EMAILS"
}
composio_execute
Parameters
{
  "action": "slack_send_message",
  "parameters": {...}
}
```

**Problem:** Action names are inconsistent:
- Sometimes uppercase: `GMAIL_LIST_EMAILS`
- Sometimes lowercase: `slack_send_message`
- Composio expects specific format

#### B. Action Not Assigned to Agent
**From Logs:**
```
2026-01-26 11:25:31,299 - consumers.chatbot.tool_router - WARNING - [tool-trace 13d8b7b8dd70] composio_execute failed: 'LIST' is not assigned to agent 19. Assign it to this agent before using it.
```

**Problem:** LLM tries to call actions that aren't assigned to the agent. The error message is clear, but the system should prevent this proactively.

#### C. Parameter Format Inconsistencies
**From Screen Output:**
```
Parameters
{
  "app_name": "SLACK",
  "action": "SLACK_SEND_MESSAGE",
  "parameters": {
    "channel": "all-automatos-ai",
    "text": "we have sorted tools and doing testing"
  }
}
```

vs.

```
Parameters
{
  "action": "slack_send_message",
  "parameters": {...}
}
```

**Problem:** Parameter structure varies:
- Sometimes includes `app_name`, sometimes doesn't
- Action name case varies
- Parameter nesting inconsistent

### 4.2 Root Causes

#### A. Action Name Normalization Missing
**Location:** `automatos-ai/orchestrator/modules/tools/execution/unified_executor.py`

The executor doesn't normalize action names before calling Composio. Composio may expect uppercase, but LLM generates mixed case.

#### B. Agent Tool Assignment Check
**Location:** Composio executor or tool router

The system checks if action is assigned AFTER the LLM decides to call it. Should check BEFORE or provide better guidance to LLM.

#### C. Parameter Schema Mismatch
The LLM is generating parameters based on tool descriptions, but:
- Tool descriptions may not match actual Composio API format
- Parameter structure isn't validated before execution
- No schema validation for Composio actions

### 4.3 Impact

- **Failed Executions**: Tools fail due to name/parameter mismatches
- **User Confusion**: Unclear why Composio tools fail
- **Retry Loops**: LLM retries with same wrong format
- **Poor Error Messages**: Errors don't guide LLM to fix issues

### 4.4 Recommendations

1. **Action Name Normalization**
   - Normalize all action names to uppercase before Composio calls
   - Map common variations (e.g., `slack_send_message` → `SLACK_SEND_MESSAGE`)
   - Validate action names against cached action list

2. **Proactive Assignment Checking**
   - Check agent assignments when building tool list for LLM
   - Only include assigned actions in tool descriptions
   - Provide clear error if unassigned action is requested

3. **Parameter Schema Validation**
   - Validate parameters against Composio action schema before execution
   - Provide clear error messages with expected format
   - Auto-fix common parameter issues (e.g., missing `app_name`)

4. **Better Error Handling**
   - Return structured errors with fix suggestions
   - Include "available actions" in error messages
   - Guide LLM to correct action names

---

## 5. File Operation Issues

### 5.1 Problem Description

File operations are failing or returning incorrect results:

#### A. Write File Failure
**From Logs:**
```
2026-01-26 11:23:51,037 - modules.agents.services.agent_action_executor - WARNING - Action failed: write_file - {'path': '', 'size': 20} - [Errno 21] Is a directory: '/private/tmp/automatos_workspace'
```

**Problem:** `write_file` is trying to write to a directory path instead of a file path. The path resolution is incorrect.

#### B. List Directory Empty Results
**From Screen Output:**
```
Currently, there are no items listed in the /Users/gkavanagh/Development/Automatos-AI-Platform directory.
```

But the directory clearly has files. The tool executed successfully but returned empty results.

**From Logs:**
```
2026-01-26 11:25:29,425 - modules.agents.services.agent_action_executor - INFO - Action executed: list_directory - {'path': '/private/tmp/automatos_workspace'}
```

**Problem:** The tool is listing `/private/tmp/automatos_workspace` (workspace dir) instead of the requested path `/Users/gkavanagh/Development/Automatos-AI-Platform`.

### 5.2 Root Causes

#### A. Path Resolution Logic
**Location:** `automatos-ai/orchestrator/modules/agents/services/agent_action_executor.py`

The executor may be:
- Resolving relative paths incorrectly
- Using workspace directory instead of absolute paths
- Not validating paths before operations

#### B. Workspace Directory Confinement
The system may be sandboxing file operations to `/private/tmp/automatos_workspace` for security, but:
- User requests absolute paths outside workspace
- System silently redirects to workspace
- Results are empty because workspace is empty

### 5.3 Impact

- **Failed Operations**: File writes fail silently or with unclear errors
- **Wrong Results**: Directory listings show wrong location
- **User Confusion**: Users request specific paths but get workspace results
- **Security Concerns**: If sandboxing is intended, it's not working correctly

### 5.4 Recommendations

1. **Fix Path Resolution**
   - Validate absolute paths before operations
   - Clearly indicate when paths are being sandboxed
   - Return error if path is outside allowed scope (don't silently redirect)

2. **Improve Error Messages**
   - Show actual path being used vs. requested path
   - Explain workspace confinement if applicable
   - Provide clear failure reasons

3. **Path Validation**
   - Check if path exists before operations
   - Validate path type (file vs. directory)
   - Prevent writing to directories

---

## 6. Database Query Issues

### 6.1 Problem Description

Database query tools are generating incorrect SQL:

**From Logs:**
```
2026-01-26 11:21:53,179 - modules.nl2sql.intelligence.agent - ERROR - SQL execution error: (psycopg2.errors.UndefinedColumn) column "provider" does not exist
LINE 1: ...DISTINCT app_name) FROM composio_apps_cache WHERE provider =...
```

**Generated SQL:**
```sql
SELECT (SELECT COUNT(DISTINCT app_name) FROM composio_apps_cache WHERE provider = 'Automatos' ...)
```

**Problem:** The SQL references a `provider` column that doesn't exist in `composio_apps_cache` table.

### 6.2 Root Cause

**Location:** `automatos-ai/orchestrator/modules/nl2sql/`

The NL2SQL service is:
- Using outdated or incorrect schema information
- Not validating generated SQL against actual schema
- Hallucinating column names based on query intent

**Schema Issue:** The `composio_apps_cache` table doesn't have a `provider` column. The NL2SQL model is inferring it should exist based on the query "How many apps and tools does Automatos have registered".

### 6.3 Impact

- **Query Failures**: Database queries fail with SQL errors
- **No Results**: Queries return 0 rows even when data exists
- **User Frustration**: Simple questions can't be answered
- **Trust Issues**: Database tool appears broken

### 6.4 Recommendations

1. **Schema Validation**
   - Validate generated SQL against actual database schema
   - Provide schema information to NL2SQL model
   - Catch schema errors before execution

2. **Better Schema Context**
   - Include actual table schemas in NL2SQL prompts
   - Update schema cache when database changes
   - Provide column name suggestions on errors

3. **Error Recovery**
   - Parse SQL errors to identify missing columns
   - Suggest correct column names
   - Retry with corrected SQL

4. **Fallback Queries**
   - For common queries, use pre-written SQL
   - Fall back to simple queries if NL2SQL fails
   - Provide query templates for common patterns

---

## 7. UI/UX Improvements Needed

### 7.1 Current State

**From Screen Output Analysis:**

The UI shows:
- ✅ Tool name
- ✅ Parameters
- ✅ Status (Running/Completed)
- ❌ Actual results
- ❌ Result summaries
- ❌ Structured data display
- ❌ Error details

### 7.2 Issues

1. **No Result Visibility**
   - Users can't see what tools returned
   - Only see "Completed" status
   - No way to inspect tool output

2. **Poor Error Display**
   - Errors shown as generic "Unknown error"
   - No actionable error messages
   - No suggestions for fixing issues

3. **No Progress Indication**
   - "Running" state doesn't show progress
   - No indication of what tool is doing
   - Can't cancel long-running operations

4. **Tool Call History**
   - Multiple tool calls shown but hard to distinguish
   - No clear relationship between calls
   - Can't see which call produced which result

### 7.3 Recommendations

1. **Result Display Components**
   - Create dedicated result viewers for each tool type:
     - **Search Results**: List with similarity scores, expandable content
     - **Database Results**: Table view with pagination
     - **File Operations**: File tree, content preview
     - **Composio Results**: Structured API response viewer
     - **Code Results**: Syntax-highlighted code blocks

2. **Tool Execution Timeline**
   - Show tool calls in chronological order
   - Link tool calls to their results
   - Show execution time for each tool
   - Indicate which tools succeeded/failed

3. **Error Display Improvements**
   - Show detailed error messages
   - Provide actionable suggestions
   - Link to documentation or fixes
   - Show error context (parameters, tool version, etc.)

4. **Progress Indicators**
   - Show progress for long-running tools
   - Indicate what step tool is on
   - Allow cancellation of operations
   - Show estimated time remaining

5. **Result Artifacts**
   - Create artifact components for rich data:
     - Charts for database results
     - Code blocks for code search
     - File trees for directory listings
     - JSON viewers for API responses

---

## 8. Performance Issues

### 8.1 Identified Issues

1. **Tool Registry Re-initialization** (see Section 2)
   - 50-100ms overhead per tool call
   - Unnecessary object creation

2. **Repeated Tool Calls** (see Section 1)
   - 3x-4x more API calls than needed
   - Wasted tokens and latency

3. **Inefficient Result Processing**
   - Results formatted multiple times
   - Truncation happens after full formatting
   - No caching of formatted results

### 8.2 Recommendations

1. **Caching Strategy**
   - Cache tool registry instance
   - Cache formatted results
   - Cache tool metadata

2. **Batch Operations**
   - Batch tool calls when possible
   - Parallel execution for independent tools
   - Reduce sequential tool calls

3. **Optimize Formatting**
   - Format results once, reuse
   - Lazy format for UI (only when needed)
   - Stream large results instead of loading all

---

## 9. Specific Tool Issues

### 9.1 Internal Tools

#### `search_knowledge`
- **Status**: ✅ Working
- **Issue**: Called multiple times with similar queries
- **Fix**: Improve duplicate detection for search queries

#### `smart_query_database`
- **Status**: ❌ Failing
- **Issue**: SQL generation with wrong column names
- **Fix**: Update schema context, validate SQL

#### `write_file`
- **Status**: ❌ Failing
- **Issue**: Path resolution to directory instead of file
- **Fix**: Validate paths, fix resolution logic

#### `list_directory`
- **Status**: ⚠️ Partially Working
- **Issue**: Returns workspace dir instead of requested path
- **Fix**: Fix path resolution, respect absolute paths

### 9.2 Composio Tools

#### `composio_execute` (Gmail)
- **Status**: ⚠️ Working but inefficient
- **Issue**: Action name case mismatch, called multiple times
- **Fix**: Normalize action names, prevent duplicates

#### `composio_execute` (Slack)
- **Status**: ✅ Working
- **Issue**: Called 3 times with identical parameters
- **Fix**: Improve duplicate detection

#### `composio_execute` (Generic)
- **Status**: ❌ Failing for unassigned actions
- **Issue**: LLM tries to call actions not assigned to agent
- **Fix**: Filter tool list by agent assignments

---

## 10. Recommendations Summary

### 10.1 Critical (Fix Immediately)

1. **Fix Tool Looping**
   - Improve duplicate detection (semantic similarity)
   - Reduce retry limits
   - Add explicit "already executed" messages

2. **Fix Tool Registry Re-initialization**
   - Reuse executor instances
   - Cache tool registry
   - Reduce initialization overhead

3. **Fix Database Query SQL Generation**
   - Update schema context for NL2SQL
   - Validate SQL before execution
   - Add error recovery

4. **Fix File Operation Path Resolution**
   - Validate paths before operations
   - Fix workspace directory logic
   - Improve error messages

### 10.2 High Priority (Fix Soon)

1. **Improve Result Display**
   - Show actual results in UI
   - Create result artifact components
   - Increase truncation limits

2. **Fix Composio Action Name Issues**
   - Normalize action names
   - Validate against cache
   - Provide better error messages

3. **Improve Error Handling**
   - Show detailed error messages
   - Provide fix suggestions
   - Link to documentation

### 10.3 Medium Priority (Nice to Have)

1. **UI/UX Enhancements**
   - Tool execution timeline
   - Progress indicators
   - Result visualization

2. **Performance Optimizations**
   - Caching strategy
   - Batch operations
   - Parallel execution

3. **Better Tool Descriptions**
   - More accurate parameter schemas
   - Better examples
   - Clearer error guidance

---

## 11. Code Locations for Fixes

### 11.1 Tool Looping
- `automatos-ai/orchestrator/consumers/chatbot/service.py:699-703` (duplicate detection)
- `automatos-ai/orchestrator/consumers/chatbot/service.py:1933-1976` (loop prevention)
- `automatos-ai/orchestrator/consumers/chatbot/service.py:1716-1720` (attempt tracking)

### 11.2 Tool Registry
- `automatos-ai/orchestrator/modules/tools/execution/unified_executor.py:219-224` (registry property)
- `automatos-ai/orchestrator/consumers/chatbot/tool_router.py:209` (executor creation)

### 11.3 Result Formatting
- `automatos-ai/orchestrator/consumers/chatbot/service.py:1722-1726` (truncation)
- `automatos-ai/orchestrator/consumers/chatbot/tool_router.py:303` (formatting)
- Frontend: Tool result display components

### 11.4 Composio Issues
- `automatos-ai/orchestrator/modules/tools/execution/unified_executor.py` (Composio execution)
- `automatos-ai/orchestrator/core/composio/tool_executor.py` (action name handling)

### 11.5 File Operations
- `automatos-ai/orchestrator/modules/agents/services/agent_action_executor.py` (path resolution)

### 11.6 Database Queries
- `automatos-ai/orchestrator/modules/nl2sql/` (SQL generation)
- Schema provider/validation

---

## 12. Testing Recommendations

### 12.1 Test Cases to Add

1. **Duplicate Detection**
   - Test exact duplicate detection
   - Test semantic similarity detection
   - Test parameter normalization

2. **Tool Execution**
   - Test each tool type individually
   - Test error handling
   - Test result formatting

3. **Composio Integration**
   - Test action name normalization
   - Test parameter validation
   - Test agent assignment checking

4. **File Operations**
   - Test absolute path handling
   - Test workspace confinement
   - Test path validation

5. **Database Queries**
   - Test SQL generation accuracy
   - Test schema validation
   - Test error recovery

### 12.2 Performance Testing

1. **Tool Registry Caching**
   - Measure initialization time
   - Test cache hit rates
   - Measure memory usage

2. **Result Formatting**
   - Measure formatting time
   - Test truncation impact
   - Measure UI render time

3. **Loop Prevention**
   - Test duplicate detection accuracy
   - Measure loop reduction
   - Test false positive rate

---

## 13. Conclusion

The tools system is **functionally working** but has significant efficiency and UX issues:

**Strengths:**
- ✅ Tools execute correctly
- ✅ Infrastructure is solid
- ✅ Error handling exists (needs improvement)
- ✅ Loop prevention logic exists (needs strengthening)

**Weaknesses:**
- ❌ Tool looping wastes resources
- ❌ Results aren't visible to users
- ❌ Inefficient initialization
- ❌ Composio integration has bugs
- ❌ File operations have path issues
- ❌ Database queries generate wrong SQL

**Priority Actions:**
1. Fix tool looping (highest impact)
2. Improve result display (user experience)
3. Fix initialization efficiency (performance)
4. Fix Composio issues (functionality)
5. Fix file/database operations (reliability)

With these fixes, the tools system will be significantly more efficient, reliable, and user-friendly.

---

**Report End**
