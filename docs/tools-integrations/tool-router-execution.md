# Tool Router & Execution

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/widgets/CodingCanvasWidget/CodeEditor.tsx](frontend/components/widgets/CodingCanvasWidget/CodeEditor.tsx)
- [frontend/components/widgets/CodingCanvasWidget/EditorTabs.tsx](frontend/components/widgets/CodingCanvasWidget/EditorTabs.tsx)
- [frontend/components/widgets/CodingCanvasWidget/FileExplorer.tsx](frontend/components/widgets/CodingCanvasWidget/FileExplorer.tsx)
- [frontend/components/widgets/CodingCanvasWidget/index.tsx](frontend/components/widgets/CodingCanvasWidget/index.tsx)
- [frontend/components/widgets/CodingCanvasWidget/useWorkspaceFiles.ts](frontend/components/widgets/CodingCanvasWidget/useWorkspaceFiles.ts)
- [frontend/components/widgets/FileWidget/FilePreview.tsx](frontend/components/widgets/FileWidget/FilePreview.tsx)
- [frontend/components/widgets/FileWidget/index.tsx](frontend/components/widgets/FileWidget/index.tsx)
- [frontend/components/widgets/ToolApprovalWidget/index.tsx](frontend/components/widgets/ToolApprovalWidget/index.tsx)
- [frontend/components/widgets/__tests__/tool-approval-widget.test.tsx](frontend/components/widgets/__tests__/tool-approval-widget.test.tsx)
- [frontend/components/widgets/index.ts](frontend/components/widgets/index.ts)
- [frontend/components/widgets/router.ts](frontend/components/widgets/router.ts)
- [frontend/components/widgets/types.ts](frontend/components/widgets/types.ts)
- [frontend/components/workspace/WorkspaceExplorer.tsx](frontend/components/workspace/WorkspaceExplorer.tsx)
- [frontend/components/workspace/gallery-view/deliverable-preview.tsx](frontend/components/workspace/gallery-view/deliverable-preview.tsx)
- [orchestrator/consumers/chatbot/tool_router.py](orchestrator/consumers/chatbot/tool_router.py)
- [orchestrator/modules/context/sections/platform_actions.py](orchestrator/modules/context/sections/platform_actions.py)
- [orchestrator/modules/context/sections/tools.py](orchestrator/modules/context/sections/tools.py)
- [orchestrator/modules/tools/discovery/action_registry.py](orchestrator/modules/tools/discovery/action_registry.py)
- [orchestrator/modules/tools/discovery/action_semantic_index.py](orchestrator/modules/tools/discovery/action_semantic_index.py)
- [orchestrator/modules/tools/execution/unified_executor.py](orchestrator/modules/tools/execution/unified_executor.py)
- [orchestrator/modules/tools/formatting/result_formatter.py](orchestrator/modules/tools/formatting/result_formatter.py)
- [orchestrator/modules/tools/registry/tool_registry.py](orchestrator/modules/tools/registry/tool_registry.py)
- [orchestrator/modules/tools/services/composio_hint_service.py](orchestrator/modules/tools/services/composio_hint_service.py)
- [orchestrator/modules/tools/services/composio_tool_service.py](orchestrator/modules/tools/services/composio_tool_service.py)
- [orchestrator/modules/tools/tool_router.py](orchestrator/modules/tools/tool_router.py)
- [orchestrator/tests/test_action_registry_filtered.py](orchestrator/tests/test_action_registry_filtered.py)
- [orchestrator/tests/test_action_semantic_index.py](orchestrator/tests/test_action_semantic_index.py)
- [orchestrator/tests/test_p2w2_tool_approval_card.py](orchestrator/tests/test_p2w2_tool_approval_card.py)
- [orchestrator/tests/test_platform_actions_section.py](orchestrator/tests/test_platform_actions_section.py)
- [orchestrator/tests/test_tool_router_semantic.py](orchestrator/tests/test_tool_router_semantic.py)

</details>



## Purpose & Scope
This page documents the unified tool execution and routing architecture of the Automatos AI platform. It covers how natural language requests or agent decisions are translated into concrete tool calls via `UnifiedToolExecutor` [orchestrator/modules/tools/execution/unified_executor.py:58](), dispatched across specialized execution modules (`exec_*`), managed through the `ToolRegistry` [orchestrator/modules/tools/registry/tool_registry.py:158]() and `ActionRegistry` [orchestrator/modules/tools/discovery/action_registry.py:59](), semantically indexed via `ActionSemanticIndex` [orchestrator/modules/tools/discovery/action_semantic_index.py:113](), and formatted into standard responses using `ToolResultFormatter` [orchestrator/modules/tools/formatting/result_formatter.py:18]().

---

## 1. UnifiedToolExecutor & Routing Logic

The `UnifiedToolExecutor` class acts as the single entry point for all tool executions across the platform [orchestrator/modules/tools/execution/unified_executor.py:5-64](). Rather than scattering tool invocation code across consumers (chatbot, recipes, workers), request handlers instantiate a per-request executor via `_get_executor_for_request()` [orchestrator/modules/tools/tool_router.py:54-62]() using a shared singleton `ToolRegistry`.

The executor relies on a declarative routing dictionary (`self.tool_routes`) that maps tool names (such as `search_knowledge`, `read_file`, `execute_command`, or `composio_execute`) to their corresponding handler methods or sub-executors [orchestrator/modules/tools/execution/unified_executor.py:96-146]().

### Natural Language Space to Code Entity Space: Tool Routing Flow
The diagram below maps abstract execution tasks from user intent to specific Python classes and methods in the codebase.

```mermaid
graph TD
    UserQuery[""NaturalLanguageQuery<br/>'Search knowledge base and run Jira tool'""] --> ToolRouter[""ToolRouter.get_tools_for_agent<br/>(orchestrator/modules/tools/tool_router.py)""]
    
    subgraph "Routing & Registry Space"
        ToolRouter --> Registry[""ToolRegistry<br/>(orchestrator/modules/tools/registry/tool_registry.py)""]
        ToolRouter --> ActionReg[""ActionRegistry<br/>(orchestrator/modules/tools/discovery/action_registry.py)""]
        Registry --> ToolSpec[""ToolSpec<br/>(name, category, executor_class)""]
    end
    
    subgraph "Execution Space (UnifiedToolExecutor)"
        ToolRouter --> UTE[""UnifiedToolExecutor<br/>(orchestrator/modules/tools/execution/unified_executor.py)""]
        UTE --> RouteMap[""self.tool_routes Map""]
        
        RouteMap -->|research| PlatformTools[""AgentPlatformTools<br/>(exec_platform / exec_research)""]
        RouteMap -->|file_ops| FileExec[""ActionExecutor<br/>(exec_file_ops)""]
        RouteMap -->|composio| CompExec[""ComposioToolExecutor<br/>(exec_composio)""]
        RouteMap -->|platform| PlatformExec[""PlatformActionExecutor<br/>(exec_platform)""]
    end
    
    ToolSpec --> UTE
```

**Sources:**
- [orchestrator/modules/tools/execution/unified_executor.py:58-146]()
- [orchestrator/modules/tools/tool_router.py:54-62]()
- [orchestrator/modules/tools/registry/tool_registry.py:91-168]()
- [orchestrator/modules/tools/discovery/action_registry.py:59-93]()

---

## 2. Specialized Execution Modules

Tool execution logic is modularized under `modules/tools/execution/` to maintain clean separation of concerns [orchestrator/modules/tools/execution/unified_executor.py:13-15](). The core modules include:

*   **`exec_platform` / `exec_research`**: Handles research and platform introspection tools such as `search_knowledge`, `semantic_search`, and `search_codebase` [orchestrator/modules/tools/execution/unified_executor.py:98-102]().
*   **`exec_workspace` / `exec_file_ops`**: Manages sandboxed file operations (`read_file`, `write_file`, `list_directory`, `delete_file`) within workspace directories [orchestrator/modules/tools/execution/unified_executor.py:114-120]().
*   **`exec_shell`**: Executes isolated shell commands via `execute_command` [orchestrator/modules/tools/execution/unified_executor.py:122]().
*   **`exec_composio`**: Integrates external SaaS applications through `composio_execute` utilizing database caching and the Composio SDK [orchestrator/modules/tools/execution/unified_executor.py:131]().
*   **`exec_multimodal`**: Handles multimodal retrievals such as `search_multimodal`, `search_tables`, and `search_images` [orchestrator/modules/tools/execution/unified_executor.py:108-112]().

**Sources:**
- [orchestrator/modules/tools/execution/unified_executor.py:27-36]()
- [orchestrator/modules/tools/execution/unified_executor.py:96-146]()

---

## 3. Tool Registry & Action Discovery

All platform capabilities are cataloged in centralized registries that generate OpenAI-compatible function schemas and documentation.

### `ToolRegistry`
The `ToolRegistry` class maintains a single source of truth for built-in platform tools, categorized by `ToolCategory` (e.g., `RESEARCH`, `FILE_OPERATIONS`, `SHELL_COMMANDS`, `WIDGET`) and assigned a `SecurityLevel` (`SAFE`, `CAUTIOUS`, `DANGEROUS`, `CRITICAL`) [orchestrator/modules/tools/registry/tool_registry.py:38-58, 158-168](). Each tool is wrapped in a `ToolSpec` dataclass which defines its parameters (`ToolParameter`) and execution mappings [orchestrator/modules/tools/registry/tool_registry.py:61-129]().

### `ActionRegistry` & `ActionSemanticIndex`
Platform actions invoked via `platform_execute` are registered in `ActionRegistry` [orchestrator/modules/tools/discovery/action_registry.py:59-65](). To prevent prompt bloat when hundreds of actions exist, `ActionSemanticIndex` embeds action definitions and parameter enums, ranking them via cosine similarity against user prompts [orchestrator/modules/tools/discovery/action_semantic_index.py:113-166]().

### Natural Language Space to Code Entity Space: Semantic Action Indexing
```mermaid
graph TD
    Query[""User Prompt / Task Description<br/>'Find all open Jira issues'""] --> Index[""ActionSemanticIndex<br/>(orchestrator/modules/tools/discovery/action_semantic_index.py)""]
    
    subgraph "Semantic Indexing Space"
        Index --> EmbedMgr[""EmbeddingManager<br/>(core.llm.create_embedding_manager)""]
        Index --> Cache[""CacheService<br/>(Redis Text Cache)""]
        Index --> Reg[""ActionRegistry<br/>(action_registry.py)""]
        Reg --> ActionDef[""ActionDefinition<br/>(name, category, parameters)""]
    end
    
    subgraph "Filtering & Ranking Space"
        Index --> Floor[""_apply_relevance_floor<br/>(SEMANTIC_TOOL_ROUTING_FLOOR)""]
        Floor --> Scored[""Ranked Action List<br/>(List[Tuple[str, float]])""]
        Scored --> Section[""PlatformActionsSection<br/>(orchestrator/modules/context/sections/platform_actions.py)""]
    end
    
    Section --> Prompt[""Injected Prompt Action Catalog""]
```

**Sources:**
- [orchestrator/modules/tools/registry/tool_registry.py:38-168]()
- [orchestrator/modules/tools/discovery/action_registry.py:27-131]()
- [orchestrator/modules/tools/discovery/action_semantic_index.py:57-111, 113-166]()
- [orchestrator/modules/context/sections/platform_actions.py:30-102]()

---

## 4. Composio Integration & Hinting

External SaaS applications are integrated via Composio with a multi-tier resolution and degradation strategy.

### `ComposioToolService` & `ComposioHintService`
*   **`ComposioToolService`**: Resolves requested Composio actions into per-action OpenAI function-calling tools, performing exact name extraction, hint-scoped searches, and semantic SDK lookups [orchestrator/modules/tools/services/composio_tool_service.py:63-113]().
*   **`ComposioHintService`**: Generates context-aware system message hints using a 3-tier strategy (Capability-based taxonomy matching, token filtering with mandatory capability gating, and top-N fallback) [orchestrator/modules/tools/services/composio_hint_service.py:89-164]().

### Degradation Seam
When Composio API keys or dependencies are absent, `_offerable_candidates()` and `_integrations_unavailable_result()` cleanly filter out Composio tools and return explicit refusal payloads rather than failing open or throwing silent exceptions [orchestrator/modules/tools/tool_router.py:121-174]().

**Sources:**
- [orchestrator/modules/tools/services/composio_tool_service.py:63-166]()
- [orchestrator/modules/tools/services/composio_hint_service.py:89-164]()
- [orchestrator/modules/tools/tool_router.py:121-174]()

---

## 5. Result Formatting & Failure Handling

The `ToolResultFormatter` class serves as the single source of truth for normalizing tool outputs across chatbot interfaces, background workers, and workflows [orchestrator/modules/tools/formatting/result_formatter.py:18-22](). It cleans document filenames [orchestrator/modules/tools/formatting/result_formatter.py:25-42](), extracts useful content excerpts from text chunks [orchestrator/modules/tools/formatting/result_formatter.py:45-67](), and fetches full document content from S3 or database chunk reassembly [orchestrator/modules/tools/formatting/result_formatter.py:112-165]().

### Failure and Stop Marker Handling
When tool executions encounter errors or deliberate programmatic halts, `select_failure_message()` inspects result payloads against `STOP_MARKERS` (`requires_confirmation`, `onboarding_restricted`, `over_quota`) to ensure user-facing copy is accurately surfaced instead of defaulting to generic error strings [orchestrator/modules/tools/tool_router.py:80-110]().

**Sources:**
- [orchestrator/modules/tools/formatting/result_formatter.py:18-165]()
- [orchestrator/modules/tools/tool_router.py:80-110]()

---