# Tier 1: Cache Lookup

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/api/routing.py](orchestrator/api/routing.py)
- [orchestrator/core/routing/engine.py](orchestrator/core/routing/engine.py)
- [orchestrator/modules/context/sections/tools.py](orchestrator/modules/context/sections/tools.py)
- [orchestrator/modules/tools/discovery/action_registry.py](orchestrator/modules/tools/discovery/action_registry.py)
- [orchestrator/modules/tools/discovery/handlers_search.py](orchestrator/modules/tools/discovery/handlers_search.py)
- [orchestrator/modules/tools/tool_router.py](orchestrator/modules/tools/tool_router.py)
- [orchestrator/scripts/setup_jira_trigger.py](orchestrator/scripts/setup_jira_trigger.py)
- [orchestrator/tests/test_action_registry_filtered.py](orchestrator/tests/test_action_registry_filtered.py)
- [orchestrator/tests/test_tool_router_semantic.py](orchestrator/tests/test_tool_router_semantic.py)

</details>



## Purpose and Scope

Tier 1: Cache Lookup is the second tier in the Universal Router's decision-making pipeline, executing immediately after **Tier 0: User Overrides** [orchestrator/core/routing/engine.py:95-101](). When a request has no explicit override, the router checks a Redis-backed routing cache to see if an identical request has been routed recently. This tier provides sub-5ms routing decisions for repeated requests, dramatically reducing LLM API costs and latency compared to **Tier 3: LLM Classification** [orchestrator/core/routing/engine.py:149-158]().

The cache stores complete `RoutingDecision` objects keyed by workspace, normalized content hash, and source [orchestrator/core/routing/cache.py:43](). Cache hits return immediately with high confidence; cache misses fall through to **Tier 2: Rule-Based Routing** or **Tier 3: LLM Classification**, which then populate the cache for future requests [orchestrator/core/routing/engine.py:103-158]().

Sources: [orchestrator/core/routing/engine.py:103-108](), [orchestrator/core/routing/cache.py:43](), [orchestrator/core/routing/engine.py:149-158]()

---

## Cache Lookup Flow

The `UniversalRouter` [orchestrator/core/routing/engine.py:58]() calls `_tier1_cache` as the first automated step in its `route` method [orchestrator/core/routing/engine.py:103-108]().

### Routing Decision Logic
Title: UniversalRouter Tier 1 Decision Flow
```mermaid
graph TB
    Envelope["RequestEnvelope [core/models/routing.py]"]
    Tier1["UniversalRouter._tier1_cache()<br/>[orchestrator/core/routing/engine.py:103]"]
    CacheCheck{"RoutingCache.get() [orchestrator/core/routing/cache.py]"}
    CacheHit["Cache Hit"]
    CacheMiss["Cache Miss"]
    ReturnDecision["Return RoutingDecision<br/>(confidence=original)"]
    Tier2["Fall through to Tier 2a/2b/2.5/2c"]
    
    Envelope --> Tier1
    Tier1 --> CacheCheck
    CacheCheck -->|"Key exists"| CacheHit
    CacheCheck -->|"No key"| CacheMiss
    CacheHit --> ReturnDecision
    CacheMiss --> Tier2
```
Sources: [orchestrator/core/routing/engine.py:103-108](), [orchestrator/core/routing/cache.py:43]()

The implementation in `UniversalRouter` is a thin wrapper around the `RoutingCache` service [orchestrator/core/routing/engine.py:70-73]():

```python
def _tier1_cache(self, envelope: RequestEnvelope) -> Optional[RoutingDecision]:
    if self._cache is None:
        return None
    return self._cache.get(
        envelope.workspace_id, envelope.content, envelope.source
    )
```
Sources: [orchestrator/core/routing/engine.py:103-108]()

---

## Cache Key Generation & Normalization

To ensure high hit rates, the content is normalized before hashing. This prevents minor variations (whitespace, casing, punctuation) from causing cache misses.

### Content to Hash Mapping
Title: Natural Language Normalization to RoutingDecisionRecord
```mermaid
graph LR
    subgraph "Natural Language Space"
        Input1["'List my agents'"]
        Input2["'list my agents. '"]
        Input3["'LIST MY AGENTS'"]
    end

    subgraph "Code Entity Space"
        Normalize["_normalize_content()<br/>[orchestrator/core/routing/cache.py]"]
        Hash["hashlib.sha256()"]
        Decision["RoutingDecisionRecord [core/models/routing.py]"]
    end

    Input1 --> Normalize
    Input2 --> Normalize
    Input3 --> Normalize
    Normalize -->|"list my agents"| Hash
    Hash -->|"env_hash"| Decision
```
Sources: [orchestrator/core/routing/cache.py:43](), [orchestrator/core/routing/engine.py:52-55]()

The `_normalize_content` function [orchestrator/core/routing/cache.py:43]() processes the string, while `_envelope_hash` in the engine creates the unique identifier used for logging and deduplication [orchestrator/core/routing/engine.py:52-55]().

| Component | Logic | File Reference |
|-----------|-------|----------------|
| **Normalization** | Strips whitespace, lowercases, removes specific punctuation | [orchestrator/core/routing/cache.py:43]() |
| **Hashing** | `sha256(normalized_content + "\|" + source.value)` | [orchestrator/core/routing/engine.py:54-55]() |
| **Workspace Scope** | Redis keys are prefixed with `routing:{workspace_id}:` | [orchestrator/core/routing/cache.py:43]() |

---

## RoutingCache Implementation

The `RoutingCache` class manages the lifecycle of routing data in Redis. It is used by the `UniversalRouter` [orchestrator/core/routing/engine.py:70-73]() and is initialized as part of the routing stack.

### Data Structure & TTL
Cached decisions are stored as JSON strings in Redis.

Title: Redis Storage Schema for RoutingCache
```mermaid
graph TB
    subgraph "Redis Storage [orchestrator/core/routing/cache.py]"
        Key["routing:{workspace_id}:{hash}"]
        Value["JSON Object"]
    end
    
    subgraph "RoutingDecision Fields [core/models/routing.py]"
        RT["route_type: 'agent' | 'workflow'"]
        AID["agent_id: int"]
        WID["workflow_id: int"]
        CONF["confidence: float"]
        REAS["reasoning: str"]
        C_FLAG["cached: True"]
    end
    
    Value --- RT
    Value --- AID
    Value --- WID
    Value --- CONF
    Value --- REAS
    Value --- C_FLAG
```
Sources: [orchestrator/core/routing/cache.py:43](), [orchestrator/core/models/routing.py:35-40]()

---

## Cache Population & Learning Loop

The cache is populated after a successful routing decision is made by lower tiers, particularly after LLM classification.

### Sequence: Learning from LLM
Title: Cache Population Sequence
```mermaid
sequenceDiagram
    participant R as UniversalRouter
    participant C as RoutingCache
    participant L as Tier 3: LLM Classification
    participant DB as RoutingDecisionRecord

    R->>C: get(envelope)
    C-->>R: None (Miss)
    R->>L: _classify_with_llm(envelope)
    L-->>R: RoutingDecision(agent_id=10, confidence=0.95)
    R->>C: put(envelope, decision)
    Note over C: SETEX routing:{ws}:{hash} TTL
    R->>DB: _log_decision(envelope, decision)
```
Sources: [orchestrator/core/routing/engine.py:149-158](), [orchestrator/core/routing/engine.py:157-158]()

### Population Scenarios
1.  **High Confidence Hits:** When the LLM returns a confidence above the `ROUTING_LLM_CONFIDENCE_THRESHOLD` [orchestrator/core/routing/engine.py:47](), the result is cached to avoid future API costs.
2.  **Auto-Brain Integration:** The complexity assessment layer also utilizes Redis cache lookups to determine task complexity before any heavy processing occurs, mirroring the Tier 1 routing pattern.

---

## Configuration

The behavior of Tier 1 is governed by several environment variables defined in the system configuration.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROUTING_LLM_CONFIDENCE_THRESHOLD` | 0.5 (via config) | Minimum confidence required to cache an LLM decision |
| `REDIS_URL` | (Standard Env) | Connection string for the cache backend |

Sources: [orchestrator/core/routing/engine.py:47](), [orchestrator/config.py]()

---

## Monitoring & Corrective Actions

Routing decisions, including whether they were served from the cache, are persisted in the `RoutingDecisionRecord` table [orchestrator/core/models/routing.py:39]().

*   **Decision Logging**: The `_log_decision` method in `UniversalRouter` records the outcome of every routing attempt, including the tier that matched and whether the result was retrieved from cache [orchestrator/core/routing/engine.py:99-100]().
*   **Correction API**: The `/api/routing/decisions` endpoint allows admins to inspect recent decisions and see which were served from cache via the `cached` boolean field [orchestrator/api/routing.py:110-155]().
*   **Decision Listing**: The `list_decisions` function in the routing API supports filtering by `agent_id`, `source`, and `was_corrected` status [orchestrator/api/routing.py:110-120]().

Sources: [orchestrator/core/routing/engine.py:99-100](), [orchestrator/api/routing.py:110-155](), [orchestrator/core/models/routing.py:39]()

---