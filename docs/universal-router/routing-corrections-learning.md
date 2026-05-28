# Routing Corrections & Learning

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



This document describes the feedback loop system that enables the `UniversalRouter` to learn from user corrections and improve routing accuracy over time. When the router selects an incorrect agent or workflow, users can submit corrections that feed back into the routing cache and decision history, creating a continuous learning mechanism that eventually updates the Tier 1 cache automatically.

For information about the core routing engine and tier strategy, see **10.1 Routing Architecture**. For cache lookup implementation details, see **10.3 Tier 1: Cache Lookup**.

Sources: [orchestrator/core/routing/engine.py:1-16]() | [orchestrator/api/routing.py:1-33]()

---

## System Overview

The routing corrections system provides three key capabilities:

1.  **Decision Tracking**: Every routing decision is logged to the `routing_decisions` table (represented by the `RoutingDecisionRecord` model) with full context, including request content hash, source, selected agent/workflow, confidence, and reasoning [orchestrator/core/routing/engine.py:857-881]().
2.  **User Corrections**: Admins can flag incorrect routing decisions and specify the correct `agent_id` via the `POST /api/routing/corrections` endpoint [orchestrator/api/routing.py:291-343]().
3.  **Cache Learning**: Corrections automatically update the `RoutingCache` after 2+ repeated corrections for the same content pattern, allowing the system to "self-heal" without manual rule intervention [orchestrator/api/routing.py:320-331]().

This creates a feedback loop where routing accuracy improves dynamically based on real-world usage.

Sources: [orchestrator/api/routing.py:291-343]() | [orchestrator/core/routing/engine.py:58-163]()

---

## Correction Workflow

### High-Level Flow

The following diagram illustrates the lifecycle of a message from initial routing to user correction and subsequent cache update.

**Diagram: Correction Feedback Loop**

```mermaid
sequenceDiagram
    participant User as "User"
    participant Frontend as "Next.js Frontend"
    participant ChatAPI as "POST /api/chat"
    participant Router as "UniversalRouter"
    participant Cache as "RoutingCache"
    participant DB as "PostgreSQL (RoutingDecisionRecord)"
    participant AdminUI as "Admin UI"
    participant CorrectionAPI as "POST /api/routing/corrections"

    User->>ChatAPI: "Send message"
    ChatAPI->>Router: "route(envelope)" [orchestrator/core/routing/engine.py:79]
    Router->>Cache: "get(workspace_id, content, source)" [orchestrator/core/routing/engine.py:103]
    Cache-->>Router: "None (miss)"
    Router->>Router: "_classify_with_llm()" [orchestrator/core/routing/engine.py:149]
    Router-->>ChatAPI: "RoutingDecision(agent_id=5, confidence=0.72)"
    Router->>DB: "_log_decision()" [orchestrator/core/routing/engine.py:157]
    
    ChatAPI-->>Frontend: "Response + Headers (x-routing-agent-id=5)"
    Frontend-->>User: "Message from wrong agent"
    
    User->>AdminUI: "Flag incorrect routing"
    AdminUI->>CorrectionAPI: "POST {request_id, correct_agent_id=12}" [orchestrator/api/routing.py:291]
    CorrectionAPI->>DB: "UPDATE was_corrected=true, corrected_agent_id=12" [orchestrator/api/routing.py:313]
    CorrectionAPI->>Cache: "record_correction(workspace_id, content, 12)" [orchestrator/api/routing.py:324]
    
    alt "correction_count >= 2"
        Cache->>Cache: "Auto-update cached decision to agent_id=12"
    end
    
    CorrectionAPI-->>AdminUI: "{"status": "corrected"}"
```

Sources: [orchestrator/api/routing.py:291-343]() | [orchestrator/core/routing/engine.py:79-163]()

---

## Decision Tracking

### RoutingDecisionRecord Schema

Every routing decision is persisted to the database via the `_log_decision` method in `UniversalRouter`. The `RoutingDecisionRecord` model tracks the following key attributes:

| Column | Type | Purpose |
| :--- | :--- | :--- |
| `request_id` | UUID | Unique identifier linking to the `RequestEnvelope` [orchestrator/api/routing.py:65]() |
| `envelope_hash` | String | SHA256 hash of normalized content used for cache keys [orchestrator/core/routing/engine.py:52-55]() |
| `route_type` | String | Type of target: "agent", "workflow", or "orchestrate" [orchestrator/api/routing.py:68]() |
| `agent_id` | Integer | Selected `Agent.id` (nullable if workflow) [orchestrator/api/routing.py:69]() |
| `confidence` | Float | Router confidence score (0.0-1.0) [orchestrator/api/routing.py:71]() |
| `was_corrected` | Boolean | Flag set to True after a user correction [orchestrator/api/routing.py:73]() |
| `corrected_agent_id` | Integer | The `Agent.id` specified by the user as correct [orchestrator/api/routing.py:74]() |

Sources: [orchestrator/api/routing.py:63-79]() | [orchestrator/core/routing/engine.py:857-881]()

---

### Decision Logging Implementation

The `UniversalRouter` class logs every decision via `_log_decision` [orchestrator/core/routing/engine.py:857](). This function captures the `RequestEnvelope` and the resulting `RoutingDecision` before committing them to the database. This data is primarily used by the `list_decisions` endpoint [orchestrator/api/routing.py:110-156]() to populate the Admin UI for review.

Sources: [orchestrator/core/routing/engine.py:857-881]() | [orchestrator/api/routing.py:110-156]()

---

## Correction Submission API

### POST /api/routing/corrections

Records a user correction for a specific routing decision. This endpoint is defined in `orchestrator/api/routing.py` and follows this logic:

1.  **Lookup**: It fetches the original decision using the `request_id` provided in the `CorrectionRequest` [orchestrator/api/routing.py:302-306]().
2.  **Update DB**: It updates the `RoutingDecisionRecord` to reflect the correction, setting `was_corrected=True` and storing the `corrected_agent_id` [orchestrator/api/routing.py:313-315]().
3.  **Update Cache**: It invokes `RoutingCache.record_correction` to increment the internal correction counter for that specific content hash [orchestrator/api/routing.py:324-329]().

Sources: [orchestrator/api/routing.py:291-343]() | [orchestrator/api/routing.py:81-84]()

---

## Cache Learning Mechanism

The `RoutingCache` implements a learning mechanism based on correction frequency:

1.  **Content Normalization**: Content is lowercased and whitespace is collapsed via `_normalize_content` to ensure consistent hashing [orchestrator/core/routing/cache.py:43]().
2.  **Auto-Update Threshold**: When `record_correction` is called, it increments a counter in Redis. If the counter for a specific content/agent pair reaches a threshold (typically 2), the cache entry for that content is updated to the corrected agent [orchestrator/api/routing.py:320-331]().
3.  **Immediate Effect**: Subsequent requests with the same content hash will hit Tier 1 (Cache) in `UniversalRouter.route` and return the corrected agent immediately, bypassing semantic similarity and LLM classification [orchestrator/core/routing/engine.py:103-107]().

Sources: [orchestrator/core/routing/cache.py:43]() | [orchestrator/api/routing.py:320-331]() | [orchestrator/core/routing/engine.py:102-107]()

---

## Unrouted Events

When all tiers (Rules, Semantic, and LLM) fail to route a request, the system stores an `UnroutedEvent` [orchestrator/core/routing/engine.py:161-163]().

```python
# [orchestrator/core/routing/engine.py:161-163]
logger.info("[router] No route found for request %s — storing unrouted event", env_hash)
self._store_unrouted_event(envelope, reason="All routing tiers exhausted (including LLM)")
```

The `_store_unrouted_event` method persists the raw content and metadata to PostgreSQL, allowing administrators to identify gaps in the routing logic or missing agent capabilities [orchestrator/core/routing/engine.py:883-900]().

Sources: [orchestrator/core/routing/engine.py:161-163]() | [orchestrator/core/routing/engine.py:883-900]()

---

## Database Schema Relationships

The following diagram bridges the Natural Language concepts to the Code Entity space by associating system names with specific code identifiers used in the routing and learning logic.

**Diagram: Routing & Learning Entities**

```mermaid
erDiagram
    "UniversalRouter (engine.py)" ||--o{ "RoutingDecisionRecord (routing.py)" : "logs_via_log_decision"
    "RoutingDecisionRecord (routing.py)" }|--|| "Agent (core.py)" : "points_to_target_agent_id"
    "RoutingDecisionRecord (routing.py)" ||--o{ "RoutingCache (cache.py)" : "triggers_record_correction"
    "RoutingRule (routing.py)" }|--|| "Agent (core.py)" : "routes_to_target_agent_id"
    "RequestEnvelope (routing.py)" ||--|| "RoutingDecisionRecord (routing.py)" : "associated_via_request_id"
    
    "RoutingDecisionRecord (routing.py)" {
        uuid request_id
        string envelope_hash
        boolean was_corrected
        int corrected_agent_id
        float confidence
    }
    
    "Agent (core.py)" {
        int id
        string name
        string description
    }

    "RoutingRule (routing.py)" {
        int id
        string source_pattern
        string[] intent_keywords
        int priority
    }
```

Sources: [orchestrator/core/routing/engine.py:58]() | [orchestrator/api/routing.py:63-79]() | [orchestrator/core/models/core.py:34-101]() | [orchestrator/api/routing.py:162-185]()

---