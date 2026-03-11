# Tier 2.5: Semantic Similarity

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/api/chat.py](orchestrator/api/chat.py)
- [orchestrator/api/routing.py](orchestrator/api/routing.py)
- [orchestrator/consumers/chatbot/auto.py](orchestrator/consumers/chatbot/auto.py)
- [orchestrator/consumers/chatbot/intent_classifier.py](orchestrator/consumers/chatbot/intent_classifier.py)
- [orchestrator/consumers/chatbot/personality.py](orchestrator/consumers/chatbot/personality.py)
- [orchestrator/consumers/chatbot/smart_orchestrator.py](orchestrator/consumers/chatbot/smart_orchestrator.py)
- [orchestrator/consumers/chatbot/smart_tool_router.py](orchestrator/consumers/chatbot/smart_tool_router.py)
- [orchestrator/core/models/system_settings.py](orchestrator/core/models/system_settings.py)
- [orchestrator/core/routing/engine.py](orchestrator/core/routing/engine.py)
- [orchestrator/modules/tools/discovery/__init__.py](orchestrator/modules/tools/discovery/__init__.py)
- [orchestrator/scripts/setup_jira_trigger.py](orchestrator/scripts/setup_jira_trigger.py)

</details>



## Purpose and Scope

Tier 2.5 of the Universal Router uses **agent embedding-based cosine similarity** to intelligently route requests to the most relevant agent. This tier sits between rule-based routing (Tier 2a/2b) and keyword-based intent classification (Tier 2c), providing a balance between speed and accuracy.

Unlike keyword matching, which relies on exact string patterns, semantic similarity understands the **meaning** of the user's request and compares it against pre-computed vector representations of each agent's capabilities. This enables routing based on conceptual overlap rather than lexical overlap.

For information about the overall routing architecture and how all tiers work together, see [Routing Architecture](#8.1). For details on the LLM-based fallback when semantic routing is inconclusive, see [Tier 3: LLM Classification](#8.6).

**Sources:** [orchestrator/core/routing/engine.py:121-134](), [orchestrator/core/routing/engine.py:360-446]()

---

## Routing Tier Sequence

Tier 2.5 executes **after** pattern-based tiers (2a, 2b) but **before** keyword matching (2c) and LLM classification (Tier 3). This positioning is intentional: semantic matching is more nuanced than keyword matching but faster than LLM calls.

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
    T2b -->|"no match"| T25["Tier 2.5: Semantic Similarity<br/>(find_similar_agents)"]
    
    T25 -->|"confidence >= 0.85"| CachePut["Cache decision"]
    CachePut --> Decision5["Return RoutingDecision"]
    
    T25 -->|"confidence < 0.85"| Candidates["Build candidate list"]
    Candidates --> T2c["Tier 2c: Intent Classifier<br/>(keyword match)"]
    
    T2c -->|"intent matched"| Decision6["Return RoutingDecision"]
    T2c -->|"no match"| T3["Tier 3: LLM Classification<br/>(_classify_with_llm)"]
    
    T3 --> Decision7["Return RoutingDecision or None"]
    
    style T25 fill:#e6f3ff,stroke:#333,stroke-width:3px
```

**Key design decision:** Tier 2.5 runs *before* Tier 2c because semantic similarity understands agent capabilities, while keyword matching is coarse-grained and can be hijacked by overly broad rules. When semantic matching finds strong candidates, those are passed directly to Tier 3 (LLM), bypassing keyword matching entirely.

**Sources:** [orchestrator/core/routing/engine.py:121-144](), [orchestrator/core/routing/engine.py:308-354]()

---

## Agent Embedding Generation

Each agent's semantic embedding is a **vector representation** of its capabilities, computed from:
- Agent name
- Agent description
- Tags
- Assigned app names (from `AgentAppAssignment`)

The embedding is stored in the `Agent.semantic_embedding` column (PostgreSQL vector type) and cached to avoid repeated API calls. A `semantic_text_hash` column tracks the input text; embeddings are only regenerated when the hash changes.

```mermaid
flowchart LR
    Agent["Agent Record"] --> Inputs["Aggregated Text Input"]
    
    Inputs --> Hash["SHA-256 Hash"]
    Hash --> Compare{"Hash Changed?"}
    
    Compare -->|"unchanged"| Skip["Skip embedding<br/>(use cached)"]
    Compare -->|"changed"| Embed["embedding_manager<br/>.generate_embedding"]
    
    Embed --> Store["Update Agent:<br/>- semantic_embedding<br/>- semantic_text_hash"]
    
    Skip --> Return["Return agent"]
    Store --> Return
    
    subgraph "Input Components"
        Inputs
        Name["agent.name"]
        Desc["agent.description"]
        Tags["agent.tags"]
        Apps["Assigned app names<br/>(AgentAppAssignment)"]
        
        Name --> Inputs
        Desc --> Inputs
        Tags --> Inputs
        Apps --> Inputs
    end
```

### Embedding Model

Embeddings are generated via `EmbeddingManager` (typically OpenAI `text-embedding-3-small` or configurable alternatives) using the same embedding infrastructure as RAG. The model produces 1536-dimensional vectors optimized for semantic similarity comparisons.

**Sources:** [orchestrator/core/routing/semantic_indexer.py:1-100]() (referenced but not provided), [orchestrator/core/routing/engine.py:371-379]()

---

## Similarity Calculation

When a request arrives, Tier 2.5 computes cosine similarity between the **query embedding** and each active agent's **semantic_embedding**.

```mermaid
flowchart TB
    Query["User Query:<br/>'Send email to team about<br/>the bug fix deployment'"] --> QueryEmbed["embedding_manager<br/>.generate_embedding"]
    
    QueryEmbed --> QueryVec["Query Embedding<br/>[1536 dims]"]
    
    subgraph "Agent Embeddings (Pre-computed)"
        A1["Email Agent<br/>semantic_embedding"]
        A2["Code Review Agent<br/>semantic_embedding"]
        A3["Data Analyst Agent<br/>semantic_embedding"]
    end
    
    QueryVec --> Score1["VectorOperations<br/>.cosine_similarity"]
    A1 --> Score1
    Score1 --> S1["Score: 0.89"]
    
    QueryVec --> Score2["VectorOperations<br/>.cosine_similarity"]
    A2 --> Score2
    Score2 --> S2["Score: 0.52"]
    
    QueryVec --> Score3["VectorOperations<br/>.cosine_similarity"]
    A3 --> Score3
    Score3 --> S3["Score: 0.31"]
    
    S1 --> Ranked["Ranked Results:<br/>1. Email Agent (0.89)<br/>2. Code Review (0.52)<br/>3. Data Analyst (0.31)"]
    S2 --> Ranked
    S3 --> Ranked
    
    Ranked --> Decision{"Top Score >= 0.85?"}
    Decision -->|"yes (0.89)"| DirectRoute["Direct Route:<br/>Return RoutingDecision<br/>agent_id=Email Agent"]
    Decision -->|"no"| PassCandidates["Pass top N candidates<br/>to Tier 3 (LLM)"]
```

### Confidence Thresholds

| Threshold | Constant | Behavior |
|-----------|----------|----------|
| **0.85** | `SIMILARITY_DIRECT_ROUTE` | Direct routing — return `RoutingDecision` immediately, cache result |
| **< 0.85** | Below threshold | Pass top candidates to Tier 3 (LLM classification) |
| **N/A** | `MAX_LLM_CANDIDATES` (typically 5) | Maximum candidates to pass to Tier 3 |

**Sources:** [orchestrator/core/routing/engine.py:360-446](), [orchestrator/core/routing/semantic_indexer.py:50-100]() (referenced)

---

## Direct Routing vs Candidate Passing

Tier 2.5 operates in **two modes** depending on the top similarity score:

### Mode 1: Direct Routing (Confidence ≥ 0.85)

When the top agent's similarity score meets or exceeds the direct routing threshold (0.85), Tier 2.5:

1. Creates a `RoutingDecision` with `route_type="agent"`
2. Sets `confidence` to the similarity score
3. Stores the decision in `RoutingCache` for future Tier 1 hits
4. Returns immediately (bypasses Tier 2c and Tier 3)

```python
# orchestrator/core/routing/engine.py:417-431
if top_score >= SIMILARITY_DIRECT_ROUTE:
    decision = RoutingDecision(
        route_type="agent",
        agent_id=top_agent.id,
        confidence=top_score,
        reasoning=f"Semantic match '{top_agent.name}' (score={top_score:.2f})",
    )
    if self._cache is not None:
        self._cache.put(
            envelope.workspace_id,
            envelope.content,
            envelope.source,
            decision,
        )
    return decision, []
```

**Sources:** [orchestrator/core/routing/engine.py:417-431]()

### Mode 2: Candidate Passing (Confidence < 0.85)

When the top score is below the threshold, Tier 2.5:

1. Returns `None` as the decision (does not route directly)
2. Returns a list of top candidates (up to `MAX_LLM_CANDIDATES`)
3. Tier 2c (keyword matching) is **skipped** if candidates exist
4. Candidates are passed to Tier 3 (LLM) to narrow the selection

This mode provides the LLM with a **shortlist** of likely agents, reducing hallucination and improving accuracy. The LLM still sees all active agents but receives semantic hints.

```python
# orchestrator/core/routing/engine.py:434-442
# Below direct threshold — return top candidates for Tier 3
candidates = [
    agent for agent, _score in scored[:MAX_LLM_CANDIDATES]
]
logger.info(
    "[router] Tier 2.5: no direct match (top=%.2f), "
    "passing %d candidates to Tier 3",
    top_score, len(candidates),
)
return None, candidates
```

**Sources:** [orchestrator/core/routing/engine.py:434-446](), [orchestrator/core/routing/engine.py:147-156]()

---

## Integration with Tier 3 (LLM)

When Tier 2.5 passes candidates to Tier 3, the LLM classification prompt includes a **semantic hint** section to guide the LLM:

```python
# orchestrator/core/routing/engine.py:686-696
semantic_hint = ""
if semantic_candidates:
    names = [
        f"'{c.name}' (ID {c.id})"
        for c in semantic_candidates[:3]
    ]
    semantic_hint = (
        f"\nSemantic analysis suggests: {', '.join(names)}. "
        "Consider these first, but use your judgment based on the "
        "full agent list.\n"
    )
```

The LLM always sees **all active agents** in the workspace (not just the candidates) but uses the semantic hints as a starting point. This prevents the pre-filter from removing the correct agent due to embedding noise.

**Sources:** [orchestrator/core/routing/engine.py:452-612](), [orchestrator/core/routing/engine.py:686-696]()

---

## Cache Integration

Successful Tier 2.5 routing decisions are stored in `RoutingCache` (Redis-backed) for instant Tier 1 hits on subsequent identical requests. The cache key is derived from:
- `workspace_id`
- Normalized content (lowercased, whitespace-collapsed)
- `ChannelSource` (e.g., `chatbot`, `jira_trigger`)

```mermaid
flowchart LR
    T25["Tier 2.5:<br/>Semantic Match"] --> Check{"Score >= 0.85?"}
    
    Check -->|"yes"| CreateDecision["RoutingDecision<br/>route_type=agent<br/>confidence=0.89"]
    Check -->|"no"| NoCacheSkip["Skip cache<br/>(ambiguous result)"]
    
    CreateDecision --> CachePut["RoutingCache.put(<br/>workspace_id,<br/>content,<br/>source,<br/>decision)"]
    
    CachePut --> RedisSet["Redis SET:<br/>routing:ws:{workspace_id}:<br/>{content_hash}"]
    
    RedisSet --> FutureHit["Future requests:<br/>Tier 1 cache hit<br/>(< 1ms)"]
    
    NoCacheSkip --> T2cOrT3["Continue to<br/>Tier 2c or Tier 3"]
```

**Cache TTL:** Routing cache entries use a configurable TTL (default 24 hours) to balance hit rate with freshness when agent configurations change.

**Sources:** [orchestrator/core/routing/engine.py:424-430](), [orchestrator/core/routing/cache.py:1-150]() (referenced)

---

## Fallback Behavior

Tier 2.5 gracefully degrades when semantic embeddings are unavailable:

### No Embeddings Available

If no agents in the workspace have embeddings (e.g., fresh workspace, indexer not run), Tier 2.5:

1. Logs a warning with counts (active agents vs agents with embeddings)
2. Returns `(None, [])` — no decision, no candidates
3. Routing falls through to Tier 2c (keyword matching) or Tier 3 (LLM)

```python
# orchestrator/core/routing/engine.py:381-405
if not scored:
    # Check why — are there agents but none with embeddings?
    agent_count = (
        self._db.query(Agent)
        .filter(
            Agent.workspace_id == envelope.workspace_id,
            Agent.status == "active",
        )
        .count()
    )
    embedded_count = (
        self._db.query(Agent)
        .filter(
            Agent.workspace_id == envelope.workspace_id,
            Agent.status == "active",
            Agent.semantic_embedding.isnot(None),
        )
        .count()
    )
    logger.warning(
        "[router] Tier 2.5: no scored agents — "
        "%d active agents, %d with embeddings in workspace %s",
        agent_count, embedded_count, envelope.workspace_id,
    )
    return None, []
```

### Exception Handling

If embedding generation fails (API error, network timeout), Tier 2.5 logs the exception and falls through:

```python
# orchestrator/core/routing/engine.py:444-446
except Exception:
    logger.exception("[router] Tier 2.5 failed — falling through")
    return None, []
```

This ensures routing always completes even when semantic matching is unavailable.

**Sources:** [orchestrator/core/routing/engine.py:381-446]()

---

## Management API

Administrators can manage semantic embeddings via two REST endpoints:

### POST /api/routing/semantic/reindex

Force regenerate embeddings for all active agents in the workspace. Supports a `force` query parameter to re-embed even if the text hash is unchanged (useful after model changes).

```mermaid
sequenceDiagram
    participant Admin
    participant API as "POST /api/routing/semantic/reindex"
    participant Indexer as "embed_workspace_agents()"
    participant DB as "PostgreSQL Agent Table"
    participant Embedding as "EmbeddingManager"
    
    Admin->>API: POST ?force=false
    API->>Indexer: embed_workspace_agents(workspace_id, db, force=False)
    
    loop For each active agent
        Indexer->>DB: Query agent metadata
        DB-->>Indexer: name, description, tags, apps
        
        Indexer->>Indexer: Build semantic text
        Indexer->>Indexer: Compute SHA-256 hash
        
        alt Hash changed OR force=true
            Indexer->>Embedding: generate_embedding(semantic_text)
            Embedding-->>Indexer: embedding vector [1536 dims]
            Indexer->>DB: UPDATE Agent SET semantic_embedding=..., semantic_text_hash=...
        else Hash unchanged
            Indexer->>Indexer: Skip (cached embedding valid)
        end
    end
    
    Indexer-->>API: reindexed_count
    API->>DB: Count total agents
    API->>DB: Count agents with embeddings
    API-->>Admin: {reindexed: N, total_agents: M, agents_with_embeddings: K}
```

**Endpoint Schema:**
```json
{
  "status": "ok",
  "reindexed": 5,
  "total_agents": 14,
  "agents_with_embeddings": 14,
  "force": false
}
```

**Sources:** [orchestrator/api/routing.py:454-497]()

### GET /api/routing/semantic/status

Inspect embedding status for all active agents in the workspace. Returns per-agent metadata including whether embeddings exist and their dimensionality.

**Endpoint Schema:**
```json
{
  "total_agents": 14,
  "agents_with_embeddings": 14,
  "agents": [
    {
      "id": 1,
      "name": "Email Agent",
      "has_embedding": true,
      "embedding_dims": 1536,
      "text_hash": "abc123def456..."
    },
    {
      "id": 2,
      "name": "Data Analyst",
      "has_embedding": false,
      "embedding_dims": 0,
      "text_hash": null
    }
  ]
}
```

Use this endpoint to identify agents that need reindexing (e.g., after description updates).

**Sources:** [orchestrator/api/routing.py:504-541]()

---

## Performance Characteristics

| Operation | Latency | Cost | Cacheable |
|-----------|---------|------|-----------|
| **Query embedding** | ~50-100ms | $0.0001 per request (OpenAI API) | No (per-request) |
| **Cosine similarity** (×N agents) | <1ms (in-memory numpy) | Free | N/A |
| **Cache hit** (Tier 1) | <1ms (Redis GET) | Free | Yes |
| **Direct route** (confidence ≥ 0.85) | ~50-100ms total | ~$0.0001 | Yes (future requests) |
| **Candidate passing** (confidence < 0.85) | ~50-100ms + Tier 3 | ~$0.0001 + Tier 3 cost | Depends on Tier 3 |
| **Reindex (14 agents)** | ~2-3 seconds | ~$0.002 | N/A (one-time) |

### Optimization Notes

1. **Agent embeddings are pre-computed** and stored in PostgreSQL. Only the query embedding is generated per request.
2. **Cosine similarity is computed in-memory** using `VectorOperations.cosine_similarity` (numpy-based), not via database queries.
3. **Cache hits (Tier 1)** bypass embedding generation entirely, returning decisions in <1ms.
4. **Batch reindexing** uses `embedding_manager.generate_embeddings_batch()` to reduce API round-trips.

**Sources:** [orchestrator/core/routing/engine.py:360-446](), [orchestrator/core/math/vector_operations.py:1-50]() (referenced)

---

## Example Flow: High Confidence

```mermaid
sequenceDiagram
    participant User
    participant ChatAPI as "/api/chat"
    participant Router as "UniversalRouter"
    participant T25 as "Tier 2.5 (Semantic)"
    participant Embed as "EmbeddingManager"
    participant DB as "PostgreSQL"
    participant Cache as "RoutingCache (Redis)"
    
    User->>ChatAPI: "Send email to team about sprint results"
    ChatAPI->>Router: route(envelope)
    
    Note over Router: Tier 0, 1, 2a, 2b<br/>all return None
    
    Router->>T25: _tier2_5_semantic(envelope)
    T25->>Embed: generate_embedding(query)
    Embed-->>T25: query_embedding [1536 dims]
    
    T25->>DB: Query active agents with semantic_embedding
    DB-->>T25: [Email Agent, Code Agent, Data Agent]
    
    loop For each agent
        T25->>T25: cosine_similarity(query_embedding, agent.semantic_embedding)
    end
    
    Note over T25: Email Agent: 0.91<br/>Code Agent: 0.52<br/>Data Agent: 0.38
    
    T25->>T25: Top score 0.91 >= 0.85 (DIRECT ROUTE)
    
    T25->>Cache: put(workspace_id, content, source, decision)
    Cache-->>T25: OK
    
    T25-->>Router: RoutingDecision(agent_id=Email Agent, confidence=0.91)
    Router-->>ChatAPI: RoutingDecision
    
    ChatAPI->>ChatAPI: Activate Email Agent
    ChatAPI-->>User: Streaming response with Email Agent
```

**Sources:** [orchestrator/core/routing/engine.py:121-134](), [orchestrator/core/routing/engine.py:360-446]()

---

## Example Flow: Low Confidence (Candidate Passing)

```mermaid
sequenceDiagram
    participant User
    participant ChatAPI as "/api/chat"
    participant Router as "UniversalRouter"
    participant T25 as "Tier 2.5 (Semantic)"
    participant Embed as "EmbeddingManager"
    participant DB as "PostgreSQL"
    participant T3 as "Tier 3 (LLM)"
    participant LLM as "LLM Manager"
    
    User->>ChatAPI: "Help me analyze customer feedback trends"
    ChatAPI->>Router: route(envelope)
    
    Note over Router: Tier 0, 1, 2a, 2b<br/>all return None
    
    Router->>T25: _tier2_5_semantic(envelope)
    T25->>Embed: generate_embedding(query)
    Embed-->>T25: query_embedding [1536 dims]
    
    T25->>DB: Query active agents with semantic_embedding
    DB-->>T25: [Data Analyst, Feedback Agent, Report Agent, ...]
    
    loop For each agent
        T25->>T25: cosine_similarity(query_embedding, agent.semantic_embedding)
    end
    
    Note over T25: Data Analyst: 0.76<br/>Feedback Agent: 0.72<br/>Report Agent: 0.69
    
    T25->>T25: Top score 0.76 < 0.85 (AMBIGUOUS)
    T25-->>Router: (None, [Data Analyst, Feedback Agent, Report Agent])
    
    Note over Router: Skip Tier 2c<br/>(candidates provided)
    
    Router->>T3: _classify_with_llm(envelope, semantic_candidates)
    T3->>DB: Query ALL active agents
    DB-->>T3: [All 14 agents]
    
    T3->>T3: Build prompt with semantic hints:<br/>"Semantic analysis suggests: Data Analyst, Feedback Agent..."
    T3->>LLM: generate_response(classification_prompt)
    LLM-->>T3: {"agent_id": 2, "confidence": 0.90}
    
    T3-->>Router: RoutingDecision(agent_id=2, confidence=0.90)
    Router-->>ChatAPI: RoutingDecision
    
    ChatAPI->>ChatAPI: Activate Data Analyst Agent
    ChatAPI-->>User: Streaming response with Data Analyst
```

**Sources:** [orchestrator/core/routing/engine.py:434-446](), [orchestrator/core/routing/engine.py:452-612]()

---

## Database Schema

The `Agent` model includes semantic routing columns:

| Column | Type | Description |
|--------|------|-------------|
| `semantic_embedding` | `VECTOR(1536)` | Pre-computed embedding vector (pgvector type) |
| `semantic_text_hash` | `VARCHAR(64)` | SHA-256 hash of input text (name + description + tags + apps) |

The hash enables **incremental reindexing**: embeddings are only regenerated when the hash changes, avoiding unnecessary API calls.

**Indexing:** The `semantic_embedding` column supports efficient similarity searches via pgvector's IVFFlat or HNSW indexes, though the current implementation fetches all agents in-memory and computes similarity in Python for simplicity.

**Sources:** [orchestrator/core/models/core.py:1-200]() (referenced, Agent model), [orchestrator/core/routing/semantic_indexer.py:50-150]() (referenced)

---

## Configuration

Tier 2.5 behavior is controlled by constants in the semantic indexer module:

```python
# Confidence threshold for direct routing
SIMILARITY_DIRECT_ROUTE = 0.85

# Maximum candidates to pass to Tier 3
MAX_LLM_CANDIDATES = 5
```

These constants are **not exposed as environment variables** in the current implementation. To adjust thresholds, modify the indexer module and restart the orchestrator.

**Future enhancement:** PRD-64 may expose these as system settings in the `system_settings` table for runtime configuration.

**Sources:** [orchestrator/core/routing/semantic_indexer.py:1-50]() (referenced)

---

## Comparison with Other Tiers

| Tier | Method | Latency | Accuracy | Use Case |
|------|--------|---------|----------|----------|
| **2a (Rules)** | Source pattern match | <1ms | 100% (when rule exists) | Explicit routing rules |
| **2b (Triggers)** | TriggerSubscription lookup | <1ms | 100% (when subscription exists) | Event-driven routing (Jira, webhooks) |
| **2.5 (Semantic)** | Cosine similarity | ~50-100ms | High (when embeddings accurate) | General-purpose intelligent routing |
| **2c (Keywords)** | Intent classifier regex | <1ms | Medium (broad categories) | Fallback pattern matching |
| **3 (LLM)** | LLM classification | ~200-500ms | Very high (contextual) | Ambiguous requests, multi-agent decisions |

**Key insight:** Tier 2.5 provides a **sweet spot** between speed (faster than LLM) and intelligence (smarter than keywords). It handles the majority of routine requests with high accuracy while deferring edge cases to the LLM.

**Sources:** [orchestrator/core/routing/engine.py:1-850]()

---

## Limitations and Edge Cases

### Cold Start Problem

Newly created agents have no embeddings until the first reindex runs. During this window, Tier 2.5 skips them (as if they don't exist) and routing falls through to lower tiers.

**Mitigation:** The workspace creation flow should trigger an async reindex job, or the agent creation API should embed new agents immediately.

### Embedding Staleness

Agent descriptions, tags, or app assignments can change without triggering automatic reindexing. The `semantic_text_hash` prevents unnecessary re-embedding but doesn't force updates.

**Mitigation:** Admins should run `POST /api/routing/semantic/reindex` after bulk agent updates. A cron job or background worker could also monitor for stale embeddings.

### Ambiguous Queries

When multiple agents have similar embeddings (e.g., "Email Agent" and "Gmail Agent"), the top score may be high but the wrong agent is selected. The `0.85` threshold reduces false positives but doesn't eliminate them.

**Mitigation:** Tier 3 (LLM) provides a second opinion when confidence is borderline. Semantic hints guide the LLM toward the correct agent.

### Model Dependency

Semantic routing quality depends on the embedding model. Changing from `text-embedding-3-small` to `text-embedding-3-large` or a different provider requires full reindexing with `force=true`.

**Mitigation:** Store the embedding model name in the `Agent` table and validate consistency during similarity calculations.

**Sources:** [orchestrator/core/routing/engine.py:381-405](), [orchestrator/api/routing.py:454-497]()

---