# 25.3. Graph Agent Tools

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/knowledge/BusinessGraphPanel.tsx](frontend/components/knowledge/BusinessGraphPanel.tsx)
- [frontend/components/knowledge/BusinessGraphVisualization.tsx](frontend/components/knowledge/BusinessGraphVisualization.tsx)
- [frontend/components/knowledge/KnowledgeGraphExplorer.tsx](frontend/components/knowledge/KnowledgeGraphExplorer.tsx)
- [orchestrator/api/harness.py](orchestrator/api/harness.py)
- [orchestrator/api/knowledge_graph.py](orchestrator/api/knowledge_graph.py)
- [orchestrator/api/shopify.py](orchestrator/api/shopify.py)
- [orchestrator/api/widget_workflows.py](orchestrator/api/widget_workflows.py)
- [orchestrator/modules/context/sections/graph_context.py](orchestrator/modules/context/sections/graph_context.py)
- [orchestrator/modules/knowledge/community_reports.py](orchestrator/modules/knowledge/community_reports.py)
- [orchestrator/modules/knowledge/graph_extraction.py](orchestrator/modules/knowledge/graph_extraction.py)
- [orchestrator/modules/knowledge/graph_service.py](orchestrator/modules/knowledge/graph_service.py)
- [orchestrator/modules/knowledge/primitive_heartbeat.py](orchestrator/modules/knowledge/primitive_heartbeat.py)
- [orchestrator/modules/memory/operations/__init__.py](orchestrator/modules/memory/operations/__init__.py)
- [orchestrator/modules/tools/discovery/actions_graph.py](orchestrator/modules/tools/discovery/actions_graph.py)
- [orchestrator/modules/tools/discovery/handlers_graph.py](orchestrator/modules/tools/discovery/handlers_graph.py)
- [orchestrator/tests/test_graph_relation_vocab.py](orchestrator/tests/test_graph_relation_vocab.py)
- [orchestrator/tests/test_graph_single_store.py](orchestrator/tests/test_graph_single_store.py)
- [orchestrator/tests/test_harness_api.py](orchestrator/tests/test_harness_api.py)
- [orchestrator/tests/test_p2w1_relics_deleted.py](orchestrator/tests/test_p2w1_relics_deleted.py)
- [orchestrator/tests/test_prd165_s2_graph.py](orchestrator/tests/test_prd165_s2_graph.py)
- [orchestrator/tests/test_prd165_s3_community.py](orchestrator/tests/test_prd165_s3_community.py)
- [orchestrator/tests/test_prd183_s1_catalog_webhook.py](orchestrator/tests/test_prd183_s1_catalog_webhook.py)
- [orchestrator/tests/test_prd189_s3_webhook_debounce.py](orchestrator/tests/test_prd189_s3_webhook_debounce.py)

</details>



This page details the platform tools available for interacting with the knowledge graph, specifically `query_graph`, `graph_neighbors`, `graph_impact`, and `graph_path`. It also covers the `GraphContext` prompt section and the single-store invariant for the graph. These tools enable agents to programmatically access and analyze the structured knowledge stored in the business knowledge graph.

## Graph Agent Tools Overview

Agents interact with the knowledge graph through a set of platform actions defined in `orchestrator/modules/tools/discovery/actions_graph.py` and implemented in `orchestrator/modules/tools/discovery/handlers_graph.py`. These tools allow agents to query the graph, find neighboring nodes, analyze impact, and discover paths between concepts.

### `query_graph`

The `query_graph` tool allows agents to query the knowledge graph using a natural language question. It leverages the `GraphifyService` to score nodes based on the query terms and then traverses the graph (BFS or DFS) from the highest-scoring node to generate a text summary within a specified token budget.

- **Purpose**: Answer natural language questions by summarizing relevant parts of the knowledge graph.
- **Implementation**:
    - `handle_query_graph` [orchestrator/modules/tools/discovery/handlers_graph.py:98-177]() is the handler for this tool.
    - It calls `_get_service().load_graph()` [orchestrator/modules/tools/discovery/handlers_graph.py:123]() to load the workspace graph.
    - `_resolve_agent_team()` [orchestrator/modules/tools/discovery/handlers_graph.py:45-62]() and `_get_filtered_graph()` [orchestrator/modules/tools/discovery/handlers_graph.py:64-67]() apply team-based filtering (PRD-124) to ensure agents only see relevant nodes.
    - `svc.score_nodes()` [orchestrator/modules/tools/discovery/handlers_graph.py:136]() scores graph nodes against the question terms.
    - `svc.dfs()` [orchestrator/modules/tools/discovery/handlers_graph.py:154]() or `svc.bfs()` [orchestrator/modules/tools/discovery/handlers_graph.py:156]() performs graph traversal.
    - `svc.subgraph_to_text()` [orchestrator/modules/tools/discovery/handlers_graph.py:161]() converts the traversed subgraph into a text summary.
- **Parameters**:
    - `question` (str): The natural language query.
    - `mode` (str, optional): Traversal mode, "bfs" (default) or "dfs".
    - `depth` (int, optional): Traversal depth (default 2).
    - `token_budget` (int, optional): Maximum tokens for the text summary (default 1500).

### `graph_neighbors`

The `graph_neighbors` tool retrieves all direct neighbors of a specified node in the knowledge graph. It can be filtered by relation type.

- **Purpose**: Explore immediate connections of a concept.
- **Implementation**:
    - `handle_graph_neighbors` [orchestrator/modules/tools/discovery/handlers_graph.py:184-240]() is the handler.
    - It loads the graph using `_get_service().load_graph()` [orchestrator/modules/tools/discovery/handlers_graph.py:201]().
    - `_find_node_by_label()` [orchestrator/modules/tools/discovery/handlers_graph.py:70-90]() resolves the concept label to a node ID.
    - It iterates through the node's neighbors and filters them based on `relation_filter`.
- **Parameters**:
    - `concept` (str): The label or ID of the node.
    - `relation_filter` (str, optional): Filters neighbors by a specific relation type.

### `graph_impact`

The `graph_impact` tool identifies nodes that are impacted by or impact a given concept, traversing the graph up to a specified depth. It considers both directional and bidirectional relations.

- **Purpose**: Understand upstream and downstream dependencies or influences of a concept.
- **Implementation**:
    - `handle_graph_impact` [orchestrator/modules/tools/discovery/handlers_graph.py:247-350]() is the handler.
    - It uses `_DIRECTIONAL_RELATIONS` [orchestrator/modules/tools/discovery/handlers_graph.py:19-25]() and `_BIDIRECTIONAL_RELATIONS` [orchestrator/modules/tools/discovery/handlers_graph.py:26-29]() to determine traversal direction.
    - It performs a BFS-like traversal to find impacted and impacting nodes.
- **Parameters**:
    - `concept` (str): The label or ID of the node.
    - `depth` (int, optional): Traversal depth (default 2).
    - `direction` (str, optional): "upstream", "downstream", or "both" (default).

### `graph_path`

The `graph_path` tool finds the shortest path between two specified nodes in the knowledge graph.

- **Purpose**: Discover connections and relationships between two distinct concepts.
- **Implementation**:
    - `handle_graph_path` [orchestrator/modules/tools/discovery/handlers_graph.py:357-420]() is the handler.
    - It uses `_find_node_by_label()` [orchestrator/modules/tools/discovery/handlers_graph.py:70-90]() to resolve both start and end concepts.
    - It calls `_get_service().find_path()` [orchestrator/modules/tools/discovery/handlers_graph.py:394]() to compute the shortest path.
- **Parameters**:
    - `start_concept` (str): The label or ID of the starting node.
    - `end_concept` (str): The label or ID of the ending node.

### Graph Agent Tools Data Flow

The following diagram illustrates the data flow when an agent uses a graph tool:

```mermaid
graph TD
    A[Agent Request] --> B{PlatformActionExecutor};
    B --> C{handlers_graph.py};
    C -- "Calls _get_service()" --> D[GraphifyService];
    D -- "Loads graph from DbWorkspaceClient" --> E[PostgreSQL: workspace_graphs table];
    D -- "Applies team_filtered_view" --> F{Team Filtering (PRD-124)};
    F --> G{Graph Traversal / Analysis (e.g., BFS, DFS, shortest_path)};
    G -- "Results (nodes, edges)" --> H[GraphifyService];
    H -- "Converts to text summary" --> I[LLM (for summarization)];
    I --> J[Formatted Tool Output];
    J --> B;
    B --> K[Agent Response];

    subgraph handlers_graph.py
        C_query[handle_query_graph]
        C_neighbors[handle_graph_neighbors]
        C_impact[handle_graph_impact]
        C_path[handle_graph_path]
    end

    C --> C_query;
    C --> C_neighbors;
    C --> C_impact;
    C --> C_path;

    style A fill:#ace,stroke:#333,stroke-width:2px
    style K fill:#ace,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#f9f,stroke:#333,stroke-width:2px
```
Sources:
- [orchestrator/modules/tools/discovery/handlers_graph.py:98-177]()
- [orchestrator/modules/tools/discovery/handlers_graph.py:184-240]()
- [orchestrator/modules/tools/discovery/handlers_graph.py:247-350]()
- [orchestrator/modules/tools/discovery/handlers_graph.py:357-420]()
- [orchestrator/modules/tools/discovery/handlers_graph.py:45-62]()
- [orchestrator/modules/tools/discovery/handlers_graph.py:64-67]()
- [orchestrator/modules/tools/discovery/handlers_graph.py:70-90]()
- [orchestrator/modules/knowledge/graph_service.py]()

## `GraphContext` Prompt Section

The `GraphContext` is a crucial section injected into the agent's prompt when graph-related tasks are anticipated. It provides the agent with a high-level overview of the knowledge graph, including its size, last build time, and available communities. This context helps the agent understand the scope and freshness of the graph data it can query.

- **Purpose**: Inform the agent about the existence and basic characteristics of the knowledge graph, enabling it to decide whether to use graph tools.
- **Implementation**:
    - The `GraphContext` class [orchestrator/modules/context/sections/graph_context.py]() is responsible for assembling this section.
    - It fetches graph metadata using `apiClient.getBusinessGraphMeta()` [frontend/components/knowledge/BusinessGraphPanel.tsx:202-206](), which reads from `graph/meta.json` stored in `DbWorkspaceClient`.
    - The metadata includes `node_count`, `edge_count`, `community_count`, and `last_built`.
    - This information is then formatted into a prompt section.

```mermaid
graph TD
    A[Agent Context Assembly] --> B[ContextService];
    B --> C[GraphContext Section];
    C -- "Fetches graph metadata" --> D[GraphifyService.get_graph_meta()];
    D -- "Reads from DbWorkspaceClient" --> E[PostgreSQL: workspace_graphs table (graph/meta.json)];
    E --> F[Graph Metadata (node_count, edge_count, last_built, etc.)];
    F --> C;
    C -- "Formats into prompt text" --> G[Agent Prompt];
```
Sources:
- [orchestrator/modules/context/sections/graph_context.py]()
- [frontend/components/knowledge/BusinessGraphPanel.tsx:202-206]()
- [orchestrator/modules/knowledge/graph_service.py:68-69]()

## Graph Single-Store Invariant

A critical architectural principle for the knowledge graph is the "single-store invariant." This means that business entities (products, orders, customers, etc.) are stored exclusively in the `workspace_graphs` table via the `GraphifyService`. The `knowledge_nodes` and `knowledge_edges` tables, while still existing for historical reasons (e.g., learning-tile metrics), are strictly read-only and are not used for storing core business entities. This prevents data duplication and ensures a single source of truth for graph data.

- **Purpose**: Maintain data integrity and a clear separation of concerns by ensuring business entities are stored in one canonical location.
- **Implementation Details**:
    - The `GraphifyService` [orchestrator/modules/knowledge/graph_service.py]() is the sole writer to `workspace_graphs`.
    - `DbWorkspaceClient` [orchestrator/modules/knowledge/graph_service.py:57]() handles the actual read/write operations to `workspace_graphs`.
    - The `test_graph_single_store.py` [orchestrator/tests/test_graph_single_store.py]() suite explicitly tests this invariant, ensuring that no business-entity writes occur to `knowledge_nodes` or `knowledge_edges` from `modules/knowledge/` or `api/shopify.py`.
    - The `_BUSINESS_WRITE_PATTERNS` [orchestrator/tests/test_graph_single_store.py:130-139]() in the test suite define patterns that would violate this invariant.
    - The `api/shopify.py` [orchestrator/api/shopify.py]() module, which handles Shopify integration, is specifically audited to ensure it writes to `workspace_graphs` and not `knowledge_nodes`/`knowledge_edges`.
    - The `_emit_graph_primitive` [orchestrator/modules/knowledge/primitive_heartbeat.py]() function, called by `GraphifyService`, emits heartbeat signals for graph build status, but this does not involve writing business entities to `knowledge_nodes`/`knowledge_edges`.

```mermaid
graph TD
    subgraph "Canonical Business Graph Store"
        A[GraphifyService] --> B[DbWorkspaceClient];
        B --> C[PostgreSQL: workspace_graphs table];
    end

    subgraph "Learning Substrate (Read-Only for Business Entities)"
        D[Analytics Engine] --> E[PostgreSQL: knowledge_nodes/knowledge_edges tables];
        F[API Endpoints (e.g., /api/system, /api/execution_history)] --> E;
    end

    C -- "Contains business entities (products, orders, etc.)" --> G[Business Entities];
    E -- "Contains learning data (read-only for business entities)" --> H[Learning Data];

    style C fill:#ace,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ace,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
    style A fill:#fff,stroke:#333,stroke-width:2px
    style D fill:#fff,stroke:#333,stroke-width:2px
    style F fill:#fff,stroke:#333,stroke-width:2px
```
Sources:
- [orchestrator/modules/knowledge/graph_service.py]()
- [orchestrator/tests/test_graph_single_store.py:1-75]()
- [orchestrator/tests/test_graph_single_store.py:130-139]()
- [orchestrator/api/shopify.py]()
- [orchestrator/modules/knowledge/primitive_heartbeat.py]()
- [orchestrator/tests/test_prd183_s1_catalog_webhook.py:1-12]()

---