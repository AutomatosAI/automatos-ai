# Code Graph

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/knowledge/CodeGraphPanel.tsx](frontend/components/knowledge/CodeGraphPanel.tsx)
- [frontend/components/knowledge/CodeGraphVisualization.tsx](frontend/components/knowledge/CodeGraphVisualization.tsx)
- [orchestrator/api/codegraph.py](orchestrator/api/codegraph.py)
- [orchestrator/modules/agents/services/agent_platform_tools.py](orchestrator/modules/agents/services/agent_platform_tools.py)
- [orchestrator/modules/codegraph/codegraph_service.py](orchestrator/modules/codegraph/codegraph_service.py)
- [orchestrator/modules/codegraph/github_auth.py](orchestrator/modules/codegraph/github_auth.py)
- [orchestrator/modules/tools/discovery/actions_codegraph.py](orchestrator/modules/tools/discovery/actions_codegraph.py)
- [orchestrator/modules/tools/discovery/handlers_codegraph.py](orchestrator/modules/tools/discovery/handlers_codegraph.py)
- [orchestrator/tests/test_prd154_s8_codegraph.py](orchestrator/tests/test_prd154_s8_codegraph.py)
- [orchestrator/tests/test_prd165_s4_codegraph.py](orchestrator/tests/test_prd165_s4_codegraph.py)
- [orchestrator/tests/test_prd183_s4_codegraph_reindex.py](orchestrator/tests/test_prd183_s4_codegraph_reindex.py)

</details>



The Code Graph system provides capabilities for indexing code repositories, extracting symbols and relationships, and enabling intelligent code search and analysis. It bridges the gap between natural language queries and code entities, allowing agents and users to understand codebase structure, trace dependencies, and analyze architectural patterns.

## CodeGraph Service Indexing

The core of the Code Graph system is the `CodeGraphService` [orchestrator/modules/codegraph/codegraph_service.py:66-69](). This service is responsible for cloning GitHub repositories, parsing code files, extracting symbols and relationships, generating semantic embeddings, and storing this information in the database.

### Data Structures

The `CodeGraphService` operates on several key data structures:
*   `CodeSymbol`: Represents a code entity like a function, class, interface, or variable. It stores `symbol_type`, `name`, `qualified_name`, `file_path`, `line_number`, `signature`, `docstring`, `code_snippet`, and `metadata` [orchestrator/modules/codegraph/codegraph_service.py:34-45]().
*   `CodeRelationship`: Captures connections between `CodeSymbol` instances, such as `calls`, `imports`, `extends`, `implements`, or `references`. It includes `from_symbol`, `to_symbol`, `relationship_type`, and `metadata` [orchestrator/modules/codegraph/codegraph_service.py:48-54]().
*   `ParseResult`: The output of parsing a single file, containing lists of `CodeSymbol` and `CodeRelationship` objects, along with `file_hash`, `lines_of_code`, and `language` [orchestrator/modules/codegraph/codegraph_service.py:57-63]().

### Indexing Process

The primary entry point for indexing is `index_github_project` [orchestrator/modules/codegraph/codegraph_service.py:121-129](). This asynchronous method performs the following steps:
1.  **Authentication**: Resolves a GitHub authentication token, prioritizing a GitHub App installation token if configured, otherwise falling back to a Personal Access Token (PAT) [orchestrator/modules/codegraph/codegraph_service.py:147-150]().
2.  **Project Status Check**: Checks if the project already exists and if it's currently in an 'indexing' state. If so, it resets the status to 'pending' to allow re-indexing [orchestrator/modules/codegraph/codegraph_service.py:153-167]().
3.  **Project Record Management**: Creates or updates a `codegraph_projects` database record, setting its status to 'indexing' [orchestrator/api/codegraph.py:165-181]().
4.  **Repository Cloning**: Clones the specified GitHub repository into a temporary directory. It supports shallow clones for speed and handles authentication [orchestrator/modules/codegraph/codegraph_service.py:200-209]().
5.  **File Traversal and Filtering**: Iterates through the cloned repository's files, applying exclude patterns (e.g., `node_modules`, `__pycache__`) and filtering by supported language extensions [orchestrator/modules/codegraph/codegraph_service.py:220-230]().
6.  **Incremental Indexing**: For existing projects, it compares file hashes to identify changed, added, or deleted files, only re-parsing what's necessary [orchestrator/modules/codegraph/codegraph_service.py:173-179]().
7.  **Code Parsing**: Utilizes a `TreeSitterParser` for multi-language support (14+ languages) if available, otherwise falls back to legacy parsers for Python, TypeScript, and JavaScript [orchestrator/modules/codegraph/codegraph_service.py:87-112](). The parser extracts `CodeSymbol` and `CodeRelationship` objects.
8.  **Embedding Generation**: For each extracted symbol, semantic embeddings are generated using the configured embedding manager [orchestrator/modules/codegraph/codegraph_service.py:75-76]().
9.  **Database Storage**: Stores the extracted symbols, relationships, and embeddings in the `codegraph_symbols`, `codegraph_relationships`, and `codegraph_files` tables [orchestrator/modules/codegraph/codegraph_service.py:390-400]().
10. **Status Update**: Updates the project status to 'ready' or 'failed' upon completion [orchestrator/modules/codegraph/codegraph_service.py:402-409]().

### Reindexing
The system supports reindexing projects. The `platform_codegraph_reindex` tool [orchestrator/modules/tools/discovery/actions_codegraph.py:32-57]() allows agents to trigger a reindex of an existing project. The `set_auto_reindex` method [orchestrator/modules/codegraph/codegraph_service.py:1600-1615]() in `CodeGraphService` enables or disables automatic reindexing, which is crucial for GitHub push webhooks to function [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:8-11]().

```mermaid
graph TD
    A[User/Agent Request] --> B{Index GitHub Repo?};
    B -- Yes --> C[API Endpoint: /api/code-graph/index/github];
    C --> D[Rate Limit Check];
    D -- Pass --> E[Update codegraph_projects status to 'indexing'];
    E --> F[Background Task: CodeGraphService.index_github_project];
    F --> G[Resolve GitHub Auth Token];
    G --> H[Clone Repo (temp dir)];
    H --> I{Iterate Files};
    I -- For each file --> J[Compare File Hash (Incremental Indexing)];
    J -- Changed/New --> K[Parse File (Tree-Sitter/Legacy)];
    K --> L[Extract CodeSymbols & CodeRelationships];
    L --> M[Generate Embeddings for Symbols];
    M --> N[Store in DB: codegraph_symbols, codegraph_relationships, codegraph_files];
    N --> O[Update codegraph_projects status to 'ready'/'failed'];
    O --> P[Return IndexResponse];

    B -- No / Reindex --> Q[API Endpoint: /api/code-graph/reindex];
    Q --> F;
    Q --> R[API Endpoint: /api/code-graph/set-auto-reindex];
    R --> S[Update codegraph_projects.auto_reindex in DB];
```
Title: CodeGraph Indexing and Reindexing Flow
Sources:
- [orchestrator/modules/codegraph/codegraph_service.py:121-129]()
- [orchestrator/modules/codegraph/codegraph_service.py:147-150]()
- [orchestrator/modules/codegraph/codegraph_service.py:153-167]()
- [orchestrator/api/codegraph.py:165-181]()
- [orchestrator/modules/codegraph/codegraph_service.py:200-209]()
- [orchestrator/modules/codegraph/codegraph_service.py:220-230]()
- [orchestrator/modules/codegraph/codegraph_service.py:173-179]()
- [orchestrator/modules/codegraph/codegraph_service.py:87-112]()
- [orchestrator/modules/codegraph/codegraph_service.py:75-76]()
- [orchestrator/modules/codegraph/codegraph_service.py:390-400]()
- [orchestrator/modules/codegraph/codegraph_service.py:402-409]()
- [orchestrator/modules/tools/discovery/actions_codegraph.py:32-57]()
- [orchestrator/modules/codegraph/codegraph_service.py:1600-1615]()
- [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:8-11]()

## GitHub Authentication

The `github_auth.py` module [orchestrator/modules/codegraph/github_auth.py:1-11]() handles resolving GitHub authentication tokens for accessing private repositories. The `resolve_github_token` function [orchestrator/modules/codegraph/github_auth.py:30-47]() attempts to:
1.  Use a GitHub App installation token if `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, and `GITHUB_APP_INSTALLATION_ID` are configured. These tokens are short-lived (~1 hour) and cached for efficiency [orchestrator/modules/codegraph/github_auth.py:20-22]().
2.  Fall back to a Personal Access Token (PAT) provided by `Config().GITHUB_PAT` if the GitHub App is not configured or token minting fails [orchestrator/modules/codegraph/github_auth.py:25-28]().

This design ensures that CodeGraph indexing can proceed even if the GitHub App configuration is incomplete or fails, by gracefully degrading to the PAT [orchestrator/modules/codegraph/github_auth.py:42-47]().

```mermaid
graph TD
    A[CodeGraphService Needs GitHub Token] --> B{Is GitHub App Configured?};
    B -- No --> C[Use GITHUB_PAT];
    B -- Yes --> D[Check Token Cache for Installation ID];
    D -- Cached & Valid --> E[Return Cached Token];
    D -- Not Cached / Expired --> F[Mint New Installation Token];
    F --> G[Generate JWT with GITHUB_APP_ID & Private Key];
    G --> H[POST to GitHub API for Access Token];
    H -- Success --> I[Cache Token & Expiry];
    I --> E;
    H -- Failure --> C;
    C --> J[Return Token (PAT or App Token)];
```
Title: GitHub Token Resolution Flow
Sources:
- [orchestrator/modules/codegraph/github_auth.py:1-11]()
- [orchestrator/modules/codegraph/github_auth.py:30-47]()
- [orchestrator/modules/codegraph/github_auth.py:20-22]()
- [orchestrator/modules/codegraph/github_auth.py:25-28]()
- [orchestrator/modules/codegraph/github_auth.py:42-47]()

## CodeGraph API and Reindexing

The CodeGraph API is exposed via FastAPI endpoints in `orchestrator/api/codegraph.py` [orchestrator/api/codegraph.py:1-6]().

Key API endpoints include:
*   `POST /api/code-graph/index/github`: Triggers the indexing of a GitHub repository. It immediately creates a project record with 'indexing' status and offloads the actual indexing to a background task [orchestrator/api/codegraph.py:116-141]().
*   `POST /api/code-graph/reindex/{project_id}`: Reindexes an existing project.
*   `GET /api/code-graph/projects`: Lists all indexed projects for the current workspace [orchestrator/api/codegraph.py:200-214]().
*   `GET /api/code-graph/search/symbols`: Performs a fuzzy search for code symbols by name [orchestrator/api/codegraph.py:240-256]().
*   `GET /api/code-graph/search/semantic`: Performs a semantic search for code symbols based on natural language queries [orchestrator/api/codegraph.py:270-286]().
*   `GET /api/code-graph/symbol/{project_id}/{qualified_name}`: Retrieves details for a specific symbol [orchestrator/api/codegraph.py:300-315]().
*   `GET /api/code-graph/call-graph/{project_id}/{qualified_name}`: Fetches the call graph for a given symbol [orchestrator/api/codegraph.py:329-345]().
*   `GET /api/code-graph/dependencies/{project_id}/{qualified_name}`: Analyzes dependencies for a symbol [orchestrator/api/codegraph.py:359-375]().
*   `GET /api/code-graph/architecture/{project_id}`: Provides a high-level architectural overview [orchestrator/api/codegraph.py:389-405]().
*   `POST /api/code-graph/ask/{project_id}`: Allows natural language questions about the codebase [orchestrator/api/codegraph.py:419-435]().
*   `POST /api/code-graph/webhook/github`: Handles GitHub webhooks for automatic reindexing on push events [orchestrator/api/codegraph.py:449-450]().
*   `PUT /api/code-graph/projects/{project_id}/auto-reindex`: Sets the `auto_reindex` flag for a project [orchestrator/api/codegraph.py:683-697]().

The `get_codegraph_service` dependency [orchestrator/api/codegraph.py:105-112]() ensures that each API request gets a `CodeGraphService` instance with the correct database session and embedding manager.

## CodeGraph Tools/Handlers

CodeGraph functionalities are exposed to agents as platform tools, defined in `orchestrator/modules/tools/discovery/actions_codegraph.py` [orchestrator/modules/tools/discovery/actions_codegraph.py:1-6]() and implemented by handlers in `orchestrator/modules/tools/discovery/handlers_codegraph.py` [orchestrator/modules/tools/discovery/handlers_codegraph.py:1-11](). These tools allow agents to programmatically interact with the code graph.

Registered actions include:
*   `platform_codegraph_list_projects`: Lists indexed repositories [orchestrator/modules/tools/discovery/actions_codegraph.py:14-30]().
*   `platform_codegraph_search`: Searches for symbols semantically or by name [orchestrator/modules/tools/discovery/actions_codegraph.py:32-57]().
*   `platform_codegraph_get_symbol`: Retrieves details of a specific symbol [orchestrator/modules/tools/discovery/actions_codegraph.py:59-83]().
*   `platform_codegraph_call_graph`: Traces the call graph for a symbol [orchestrator/modules/tools/discovery/actions_codegraph.py:87-109]().
*   `platform_codegraph_dependencies`: Performs change-impact analysis [orchestrator/modules/tools/discovery/actions_codegraph.py:111-134]().
*   `platform_codegraph_architecture`: Provides a high-level architecture overview [orchestrator/modules/tools/discovery/actions_codegraph.py:136-158]().
*   `platform_codegraph_index`: Onboards or refreshes a repository [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:79-107]().
*   `platform_codegraph_reindex`: Reindexes an existing project [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:114-123]().
*   `platform_codegraph_set_auto_reindex`: Enables/disables automatic reindexing [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:139-149]().

These tools are integrated into the `AgentPlatformTools` class [orchestrator/modules/agents/services/agent_platform_tools.py:26-30](), making them available for agents to use during execution. The `search_codebase` tool [orchestrator/modules/agents/services/agent_platform_tools.py:97-134]() specifically routes to either fuzzy or semantic search based on the `search_type` parameter.

```mermaid
graph TD
    A[Agent] --> B[AgentPlatformTools];
    B --> C{Tool Call: search_codebase};
    C -- query, project_name, search_type="fuzzy" --> D[CodeGraphService.search_symbols];
    C -- query, project_name, search_type="semantic" --> E[CodeGraphService.semantic_search];
    D --> F[DB: codegraph_symbols (fuzzy match)];
    E --> G[DB: codegraph_symbols (vector search)];
    F --> H[Results];
    G --> H;
    B --> I{Tool Call: get_call_graph};
    I -- symbol, project_name, depth, direction --> J[CodeGraphService.get_call_graph];
    J --> K[DB: codegraph_relationships];
    K --> L[Call Graph Data];
    B --> M{Tool Call: platform_codegraph_index};
    M -- project_name, github_url, branch --> N[CodeGraphService.index_github_project];
    N --> O[Indexing Process];
    O --> P[Indexing Status];
```
Title: Agent Interaction with CodeGraph Tools
Sources:
- [orchestrator/modules/tools/discovery/actions_codegraph.py:1-6]()
- [orchestrator/modules/tools/discovery/handlers_codegraph.py:1-11]()
- [orchestrator/modules/tools/discovery/actions_codegraph.py:14-30]()
- [orchestrator/modules/tools/discovery/actions_codegraph.py:32-57]()
- [orchestrator/modules/tools/discovery/actions_codegraph.py:59-83]()
- [orchestrator/modules/tools/discovery/actions_codegraph.py:87-109]()
- [orchestrator/modules/tools/discovery/actions_codegraph.py:111-134]()
- [orchestrator/modules/tools/discovery/actions_codegraph.py:136-158]()
- [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:79-107]()
- [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:114-123]()
- [orchestrator/tests/test_prd183_s4_codegraph_reindex.py:139-149]()
- [orchestrator/modules/agents/services/agent_platform_tools.py:26-30]()
- [orchestrator/modules/agents/services/agent_platform_tools.py:97-134]()

## CodeGraphPanel and CodeGraphVisualization UI

The frontend provides a user interface for managing and visualizing code graphs.
*   `CodeGraphPanel` [frontend/components/knowledge/CodeGraphPanel.tsx:49-50]() is the main component for interacting with CodeGraph projects. It allows users to:
    *   List and manage indexed projects [frontend/components/knowledge/CodeGraphPanel.tsx:100-117]().
    *   Add new GitHub repositories for indexing [frontend/components/knowledge/CodeGraphPanel.tsx:119-160]().
    *   Search for symbols (fuzzy or semantic) within a selected project [frontend/components/knowledge/CodeGraphPanel.tsx:162-197]().
    *   Ask natural language questions about the code [frontend/components/knowledge/CodeGraphPanel.tsx:199-215]().
    *   Trigger reindexing and delete projects.
    *   Polls for indexing progress to update project statuses [frontend/components/knowledge/CodeGraphPanel.tsx:83-98]().

*   `CodeGraphVisualization` [frontend/components/knowledge/CodeGraphVisualization.tsx:87-88]() is a ReactFlow-based component for rendering interactive code graphs. It supports:
    *   Displaying call graphs, import graphs, and inheritance graphs [frontend/components/knowledge/CodeGraphVisualization.tsx:92-93]().
    *   Adjusting graph depth and direction (incoming/outgoing/both) [frontend/components/knowledge/CodeGraphVisualization.tsx:93-94]().
    *   Visualizing architectural overlays like communities (clusters) and hotspots [frontend/components/knowledge/CodeGraphVisualization.tsx:98-99]().
    *   Highlighting nodes based on search queries.
    *   Displaying code snippets, signatures, and docstrings for selected nodes [frontend/components/knowledge/CodeGraphVisualization.tsx:102-104]().
    *   A file tree sidebar for navigation and filtering [frontend/components/knowledge/CodeGraphVisualization.tsx:109-114]().
    *   Node coloring based on symbol type and cluster membership [frontend/components/knowledge/CodeGraphVisualization.tsx:64-77]().
    *   Heatmap visualization for metrics like fan-out and betweenness [frontend/components/knowledge/CodeGraphVisualization.tsx:79-84]().

The UI components interact with the backend API via `apiClient` [frontend/components/knowledge/CodeGraphVisualization.tsx:4-4]().

```mermaid
graph TD
    A[User] --> B[CodeGraphPanel (UI)];
    B --> C{Tab Selection: Projects, Search, Visualize};

    C -- Projects --> D[List Projects];
    D --> E[Add Project Modal];
    E -- Submit --> F[apiClient.codegraphIndexGithub];
    F --> G[Backend API: /api/code-graph/index/github];
    G --> H[Update Project Status (Polling)];

    C -- Search --> I[Search Input (Query, Project, Mode)];
    I -- Submit --> J[apiClient.codegraphSearchSymbols / apiClient.codegraphSearchSemantic];
    J --> K[Backend API: /api/code-graph/search/*];
    K --> L[Display Search Results];

    C -- Visualize --> M[CodeGraphVisualization];
    M --> N[Symbol Search Input];
    N -- Select Symbol --> O[apiClient.codegraphGetCallGraph];
    O --> P[Backend API: /api/code-graph/call-graph];
    P --> Q[Render ReactFlow Graph (Nodes, Edges)];
    Q --> R[Architecture Overlay (Clusters, Hotspots)];
    Q --> S[Code Snippet Panel (Selected Node)];
    Q --> T[File Tree Sidebar];
```
Title: CodeGraph Frontend UI Interaction Flow
Sources:
- [frontend/components/knowledge/CodeGraphPanel.tsx:49-50]()
- [frontend/components/knowledge/CodeGraphPanel.tsx:100-117]()
- [frontend/components/knowledge/CodeGraphPanel.tsx:119-160]()
- [frontend/components/knowledge/CodeGraphPanel.tsx:162-197]()
- [frontend/components/knowledge/CodeGraphPanel.tsx:199-215]()
- [frontend/components/knowledge/CodeGraphPanel.tsx:83-98]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:87-88]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:92-93]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:93-94]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:98-99]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:102-104]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:109-114]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:64-77]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:79-84]()
- [frontend/components/knowledge/CodeGraphVisualization.tsx:4-4]()

---