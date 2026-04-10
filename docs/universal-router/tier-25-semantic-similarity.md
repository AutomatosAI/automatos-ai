# Tier 2.5: Semantic Similarity

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/reviews/COMPOSIO-TOOL-REGRESSION-REVIEW.md](docs/reviews/COMPOSIO-TOOL-REGRESSION-REVIEW.md)
- [orchestrator/alembic/versions/20260224_add_semantic_routing_columns.py](orchestrator/alembic/versions/20260224_add_semantic_routing_columns.py)
- [orchestrator/alembic/versions/20260225_create_marketplace_widgets.py](orchestrator/alembic/versions/20260225_create_marketplace_widgets.py)
- [orchestrator/alembic/versions/20260225_create_sdk_api_keys.py](orchestrator/alembic/versions/20260225_create_sdk_api_keys.py)
- [orchestrator/alembic/versions/20260225_create_widget_installs_reviews.py](orchestrator/alembic/versions/20260225_create_widget_installs_reviews.py)
- [orchestrator/alembic/versions/20260225_create_workspace_shares.py](orchestrator/alembic/versions/20260225_create_workspace_shares.py)
- [orchestrator/api/chat.py](orchestrator/api/chat.py)
- [orchestrator/api/chat_voice.py](orchestrator/api/chat_voice.py)
- [orchestrator/consumers/chatbot/auto.py](orchestrator/consumers/chatbot/auto.py)
- [orchestrator/consumers/chatbot/intent_classifier.py](orchestrator/consumers/chatbot/intent_classifier.py)
- [orchestrator/consumers/chatbot/personality.py](orchestrator/consumers/chatbot/personality.py)
- [orchestrator/consumers/chatbot/service.py](orchestrator/consumers/chatbot/service.py)
- [orchestrator/consumers/chatbot/smart_tool_router.py](orchestrator/consumers/chatbot/smart_tool_router.py)
- [orchestrator/core/llm/manager.py](orchestrator/core/llm/manager.py)
- [orchestrator/core/routing/engine.py](orchestrator/core/routing/engine.py)
- [orchestrator/core/routing/semantic_indexer.py](orchestrator/core/routing/semantic_indexer.py)
- [orchestrator/core/seeds/seed_cto_agent.py](orchestrator/core/seeds/seed_cto_agent.py)
- [orchestrator/modules/orchestrator/service.py](orchestrator/modules/orchestrator/service.py)
- [orchestrator/modules/tools/discovery/actions_analytics_enhanced.py](orchestrator/modules/tools/discovery/actions_analytics_enhanced.py)
- [orchestrator/modules/tools/discovery/handlers_analytics_enhanced.py](orchestrator/modules/tools/discovery/handlers_analytics_enhanced.py)
- [orchestrator/modules/tools/discovery/handlers_search.py](orchestrator/modules/tools/discovery/handlers_search.py)
- [orchestrator/modules/tools/discovery/platform_actions.py](orchestrator/modules/tools/discovery/platform_actions.py)
- [orchestrator/modules/tools/discovery/platform_executor.py](orchestrator/modules/tools/discovery/platform_executor.py)
- [orchestrator/modules/tools/tool_router.py](orchestrator/modules/tools/tool_router.py)

</details>



## Purpose and Scope

Tier 2.5 of the Universal Router implements **agent embedding-based cosine similarity** to intelligently route requests to the most relevant agent. This tier sits between rule-based routing (Tier 2a/2b) and keyword-based intent classification (Tier 2c), providing a balance between speed and reasoning.

Unlike keyword matching, which relies on exact string patterns, semantic similarity understands the **meaning** of the user's request and compares it against pre-computed vector representations of each agent's capabilities. This enables routing based on conceptual overlap rather than lexical overlap, which is critical for complex multi-agent environments.

**Sources:** [orchestrator/core/routing/engine.py:11-16](), [orchestrator/core/routing/engine.py:123-126]()

---

## Routing Tier Sequence

Tier 2.5 executes **after** pattern-based tiers (2a, 2b) but **before** keyword matching (2c) and LLM classification (Tier 3). This positioning is intentional: semantic matching is more nuanced than keyword matching but significantly faster and cheaper than full LLM calls.

### UniversalRouter Execution Flow

Title: UniversalRouter Tiered Logic
```mermaid
flowchart TD
    Start["RequestEnvelope"] --> T0["Tier 0: User Override"]
    T0 -->|"override_agent_id set"| Decision1["Return RoutingDecision"]
    T0 -->|"no override"| T1["Tier 1: Cache Lookup"]
    
    T1 -->|"cache hit"| Decision2["Return RoutingDecision"]
    T1 -->|"cache miss"| T2a["Tier 2a: Routing Rules<br/>(source_pattern match)"]
    
    T2a -->|"rule matched"| Decision3["Return RoutingDecision"]
    T2a -->|"no match"| T2b["Tier 2b: TriggerSubscription<br/>(jira_trigger)"]
    
    T2b -->|"trigger matched"| Decision4["Return RoutingDecision"]
    T2b -->|"no match"| T25["Tier 2.5: Semantic Similarity<br/>(_tier2_5_semantic)"]
    
    T25 -->|"confidence >= DIRECT_ROUTE"| CachePut["Cache decision"]
    CachePut --> Decision5["Return RoutingDecision"]
    
    T25 -->|"confidence < DIRECT_ROUTE"| Candidates["Build candidate list"]
    Candidates --> T2c["Tier 2c: Intent Classifier<br/>(keyword match)"]
    
    T2c -->|"intent matched"| Decision6["Return RoutingDecision"]
    T2c -->|"no match"| T3["Tier 3: LLM Classification<br/>(_classify_with_llm)"]
    
    T3 --> Decision7["Return RoutingDecision or None"]
    
    style T25 fill:#ffffff,stroke:#000000,stroke-width:3px
```

**Key design decision:** Tier 2.5 runs *before* Tier 2c because semantic matching understands agent capabilities, while keyword matching is coarse-grained and can be hijacked by overly broad rules. If semantic candidates are found, the system can prioritize them in the Tier 3 LLM fallback.

**Sources:** [orchestrator/core/routing/engine.py:123-147]()

---

## Agent Embedding Generation

Each agent's semantic embedding is a **vector representation** of its capabilities. The system aggregates data from multiple internal models to create a comprehensive profile for indexing.

### Embedding Input Sources
- **Core Identity:** Agent `name`, `description`, and `slug` from the `Agent` model.
- **Persona:** Instructions and personality traits defined in the agent's configuration.
- **Tools & Skills:** The descriptions of tools associated with the agent via `AgentAppAssignment` or `SkillLoader`.
- **Platform Actions:** If the agent has platform management capabilities, those actions (e.g., `platform_list_agents`) are included in the semantic text.

Title: Natural Language to Code Entity Mapping (Agent Profile)
```mermaid
flowchart LR
    subgraph "Natural Language Space"
        UserQuery["'How much have I spent on LLMs?'"]
    end

    subgraph "Code Entity Space"
        Agent["core.models.core.Agent"]
        Assignment["core.models.composio_cache.AgentAppAssignment"]
        Action["modules.tools.discovery.platform_executor.PlatformActionExecutor"]
        
        Agent --- Assignment
        Agent --- Action
    end

    UserQuery -.->|"Cosine Similarity"| Agent
    Agent -->|"Aggregated by"| Indexer["core.routing.semantic_indexer"]
```

**Sources:** [orchestrator/core/routing/engine.py:33-40](), [orchestrator/modules/tools/discovery/platform_executor.py:164-172](), [orchestrator/consumers/chatbot/intent_classifier.py:78-107]()

---

## Similarity Calculation and Thresholds

When a request arrives, Tier 2.5 computes cosine similarity between the **query embedding** and each active agent's embedding in the current workspace. The system utilizes the `LLMManager` to interface with embedding providers (typically OpenAI or OpenRouter).

### Confidence Thresholds

| Threshold | Type | Behavior |
|-----------|----------|----------|
| **High (e.g. 0.90+)** | `DIRECT_ROUTE` | Direct routing — return `RoutingDecision` immediately and cache the result for Tier 1. |
| **Medium (e.g. 0.50+)** | `CANDIDATE` | Included in the candidate list passed to Tier 3 (LLM) to narrow the search space. |
| **Low** | `REJECT` | Ignored; the agent is considered irrelevant to the user's intent. |

Title: Semantic Matching Process
```mermaid
flowchart TB
    Query["RequestEnvelope.content"] --> Embed["llm_manager.generate_embeddings"]
    Embed --> QueryVec["Query Vector"]
    
    subgraph "Workspace Agents"
        A1["Agent A (Vector)"]
        A2["Agent B (Vector)"]
    end
    
    QueryVec --> Sim["Cosine Similarity"]
    A1 --> Sim
    A2 --> Sim
    
    Sim --> Scores["Ranked List"]
    
    Scores --> Direct{"Top Score >= Threshold?"}
    Direct -->|"Yes"| Route["Direct RoutingDecision"]
    Direct -->|"No"| Candidates["Pass candidates to Tier 3"]
```

**Sources:** [orchestrator/core/routing/engine.py:129-136](), [orchestrator/core/llm/manager.py:30-41]()

---

## Implementation Details

### Intent Integration
Tier 2.5 is closely coupled with the `SmartIntentClassifier`. If the classifier detects a specific intent (e.g., `DATA_QUERY`), the semantic router prioritizes agents whose toolsets match that intent (e.g., agents with `platform_get_llm_usage` or `query_database`).

### Tool Looping & Semantic Deduplication
Semantic similarity is also used in the `ToolExecutionTracker` within the `ChatService`. It prevents infinite tool loops by checking if a new tool query is semantically similar to a previous one in the same conversation turn using `_queries_are_similar`.

### Candidate Shortlisting
If no agent meets the direct routing threshold, Tier 2.5 returns a list of candidates. This list is then used in Tier 3 (LLM Classification) to provide the LLM with "hints," significantly improving the accuracy of the final routing decision compared to a "blind" classification.

**Sources:** [orchestrator/consumers/chatbot/service.py:57-67](), [orchestrator/consumers/chatbot/service.py:78-85](), [orchestrator/consumers/chatbot/intent_classifier.py:23-34](), [orchestrator/core/routing/engine.py:149-158]()

---

## Integration with Chat API

The routing decision (including semantic confidence) is processed within the `api/chat.py` flow. Before streaming a response, the `UniversalRouter` is called to resolve the target agent.

Title: Chat to Routing Integration
```mermaid
sequenceDiagram
    participant UI as "Chat UI"
    participant API as "orchestrator/api/chat.py"
    participant Router as "core.routing.engine.UniversalRouter"
    participant LLM as "core.llm.manager.LLMManager"
    
    UI->>API: POST /api/chat
    API->>Router: route(RequestEnvelope)
    Router->>LLM: generate_embeddings(content)
    LLM-->>Router: vector
    Router-->>API: RoutingDecision (Tier 2.5)
    API-->>UI: StreamingResponse with Agent Context
```

**Sources:** [orchestrator/api/chat.py:22-24](), [orchestrator/api/chat.py:103-107](), [orchestrator/core/llm/manager.py:86-117]()

---