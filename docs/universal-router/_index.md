# Universal Router

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



The Universal Router is Automatos AI's intelligent message routing system that determines which agent or workflow should handle an incoming request. It implements a 7-tier routing strategy that progressively escalates from fast, deterministic rules to semantic similarity and LLM-based classification, ensuring optimal routing accuracy while minimizing latency and cost.

For information about complexity assessment (which precedes routing), see [Complexity Assessment (AutoBrain)](#9.2). For context assembly after routing, see [Context Service](#4).

---

## Routing Architecture

The Universal Router operates on a normalized input (`RequestEnvelope`) and produces a routing decision (`RoutingDecision`) by evaluating the request through a series of tiers until a match is found.

### High-Level Flow

```mermaid
graph TB
    Input["RequestEnvelope [core.models.routing]"]
    Router["UniversalRouter.route() [orchestrator/core/routing/engine.py]"]
    
    Input --> Router
    
    Router --> T0["Tier 0: User Override<br/>override_agent_id/override_workflow_id"]
    T0 -->|Hit| Decision
    T0 -->|Miss| T1
    
    T1["Tier 1: Cache Lookup<br/>RoutingCache (Redis)"]
    T1 -->|Hit| Decision
    T1 -->|Miss| T2a
    
    T2a["Tier 2a: Routing Rules<br/>source_pattern match"]
    T2a -->|Hit| Decision
    T2a -->|Miss| T2b
    
    T2b["Tier 2b: Trigger Subscription<br/>TriggerSubscription (Jira)"]
    T2b -->|Hit| Decision
    T2b -->|Miss| T2_5
    
    T2_5["Tier 2.5: Semantic Similarity<br/>Cosine on agent embeddings"]
    T2_5 -->|High Score| Decision
    T2_5 -->|Ambiguous| Candidates
    
    Candidates["Semantic Candidates"] --> T3
    T2_5 -->|No Match| T2c
    
    T2c["Tier 2c: Intent Classifier<br/>Keyword matching"]
    T2c -->|Hit| Decision
    T2c -->|Miss| T3
    
    T3["Tier 3: LLM Classification<br/>_classify_with_llm()"]
    T3 -->|Classified| Decision
    T3 -->|Failed| Unrouted
    
    Decision["RoutingDecision [core.models.routing]"]
    Unrouted["UnroutedEvent [core.models.routing]"]
    
    Decision --> Log["RoutingDecisionRecord [routing_decisions table]"]
    Unrouted --> UnroutedTable["UnroutedEvent [unrouted_events table]"]
```

**Sources:** [orchestrator/core/routing/engine.py:79-163](), [orchestrator/core/models/routing.py:35-42]()

---

### RequestEnvelope

The `RequestEnvelope` is the normalized input to the router, containing:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `UUID` | Unique request identifier |
| `workspace_id` | `UUID` | Workspace context |
| `source` | `ChannelSource` | Origin channel (chatbot, slack, telegram, jira_trigger, etc.) |
| `content` | `str` | Normalized message text |
| `override_agent_id` | `Optional[int]` | Explicit agent selection (Tier 0) |
| `override_workflow_id` | `Optional[int]` | Explicit workflow selection (Tier 0) |
| `metadata` | `Dict` | Channel-specific metadata (e.g., trigger_name for Jira) |

Envelopes are created by **ingestors** that normalize messages from different sources. For details, see [Routing Architecture](#10.1).

**Sources:** [orchestrator/core/models/routing.py:37](), [orchestrator/core/routing/engine.py:37]()

---

### RoutingDecision

The `RoutingDecision` is the output, specifying how to handle the request:

| Field | Type | Description |
|-------|------|-------------|
| `route_type` | `str` | "agent", "workflow", or "orchestrate" |
| `agent_id` | `Optional[int]` | Target agent ID (if route_type=agent) |
| `workflow_id` | `Optional[int]` | Target workflow ID (if route_type=workflow) |
| `confidence` | `float` | Routing confidence (0.0–1.0) |
| `reasoning` | `str` | Human-readable explanation |

**Route Types:**

- **`agent`** — Route to a specific agent (most common).
- **`workflow`** — Route to a workflow/recipe (Tier 2b triggers, or explicit rules).
- **`orchestrate`** — No clear match; use orchestrator LLM (fallback when confidence < threshold).

**Sources:** [orchestrator/core/models/routing.py:38-42](), [orchestrator/core/routing/engine.py:38-42]()

---

### Decision Logging

All routing decisions are logged to the `routing_decisions` table for analytics and learning via `RoutingDecisionRecord`.

```mermaid
erDiagram
    "RoutingDecisionRecord [orchestrator/core/models/routing.py]" {
        int id PK
        uuid request_id
        uuid workspace_id
        string envelope_hash
        string source
        string route_type
        int agent_id
        int workflow_id
        float confidence
        bool cached
        bool was_corrected
        int corrected_agent_id
        timestamp created_at
    }
    
    "UnroutedEvent [orchestrator/core/models/routing.py]" {
        int id PK
        uuid request_id
        uuid workspace_id
        string source
        string content
        string reason
        timestamp created_at
    }
```

User corrections update `was_corrected` and `corrected_agent_id`, which feed into the cache learning loop.

**Sources:** [orchestrator/core/models/routing.py:39-41](), [orchestrator/core/routing/engine.py:162-163](), [orchestrator/api/routing.py:63-79]()

---

## Tier 0: User Overrides

When the user explicitly selects an agent or workflow, routing bypasses all other tiers. This is handled by `_tier0_override` in the routing engine.

**Confidence:** Always 1.0 (user decision is authoritative). For details, see [Tier 0: User Overrides](#10.2).

**Sources:** [orchestrator/core/routing/engine.py:169-184]()

---

## Tier 1: Cache Lookup

The `RoutingCache` stores recent routing decisions in Redis, keyed by `(workspace_id, content_hash, source)`.

**Normalization:** Content is lowercased and whitespace-stripped before hashing to improve hit rate via `_normalize_content`.

**Hit Rate:** Cache hits are ~5ms; misses proceed to Tier 2a. For details, see [Tier 1: Cache Lookup](#10.3).

**Sources:** [orchestrator/core/routing/cache.py:43](), [orchestrator/core/routing/engine.py:103-107]()

---

## Tier 2: Rule-Based Routing

Workspace admins can define routing rules in the `routing_rules` table. This tier includes:
- **Tier 2a**: Direct source pattern matching via `_tier2a_rules`.
- **Tier 2b**: `TriggerSubscription` for external events like Jira webhooks via `_tier2b_trigger_subscription`.
- **Tier 2c**: Keyword matching via `IntentClassifier` against `RoutingRule.intent_keywords`.

For details, see [Tier 2: Rule-Based Routing](#10.4).

**Sources:** [orchestrator/core/routing/engine.py:109-122](), [orchestrator/api/routing.py:23-27](), [orchestrator/core/models/composio.py:32](), [orchestrator/scripts/setup_jira_trigger.py:123-136]()

---

## Tier 2.5: Semantic Similarity

This tier uses agent embeddings to find the best match via cosine similarity. It embeds agent capabilities (description, skills, tools) into a vector space. 

```mermaid
graph TB
    "Agent [orchestrator/core/models/core.py]" --> Text["build_agent_semantic_text()"]
    "AgentSkillAssignment" --> Text
    "AgentAppAssignment [orchestrator/core/models/composio_cache.py]" --> Text
    
    Text --> Embed["EmbeddingManager<br/>generate_embedding()"]
    
    Embed --> Store["agent.semantic_embedding"]
```

**Thresholds:**
- `SIMILARITY_DIRECT_ROUTE = 0.95`
- `SIMILARITY_CANDIDATE_MIN = 0.40`

For details, see [Tier 2.5: Semantic Similarity](#10.5).

**Sources:** [orchestrator/core/routing/engine.py:123-136](), [orchestrator/core/models/composio_cache.py:33]()

---

## Tier 3: LLM Classification

When all previous tiers fail, the router uses an LLM to classify the request. It utilizes the `ROUTER` context mode to assemble a prompt containing available agents and semantic hints from Tier 2.5.

For details, see [Tier 3: LLM Classification](#10.6).

**Sources:** [orchestrator/core/routing/engine.py:148-158](), [orchestrator/modules/context/sections/tools.py:41]()

---

## Routing Corrections & Learning

Users can correct routing decisions via the `POST /api/routing/corrections` endpoint. This records the correction in `RoutingDecisionRecord` and updates the `RoutingCache`.

**Threshold:** The system auto-updates the cache after **2+ corrections** for the same content hash. For details, see [Routing Corrections & Learning](#10.7).

**Sources:** [orchestrator/api/routing.py:290-343](), [orchestrator/core/routing/cache.py:112-152]()

---

## Management & Child Pages

This is a high-level overview. For deep dives into specific components, refer to the following child pages:

- **[Routing Architecture](#10.1)** — Detailed schema of `RequestEnvelope` and `RoutingDecision`.
- **[Tier 0: User Overrides](#10.2)** — How explicit UI selections bypass the engine.
- **[Tier 1: Cache Lookup](#10.3)** — Implementation of the Redis-based `RoutingCache`.
- **[Tier 2: Rule-Based Routing](#10.4)** — Managing `RoutingRule` and `TriggerSubscription`.
- **[Tier 2.5: Semantic Similarity](#10.5)** — Cosine similarity logic and agent embeddings.
- **[Tier 3: LLM Classification](#10.6)** — The `ROUTER` context mode and classification prompts.
- **[Routing Corrections & Learning](#10.7)** — The feedback loop that improves routing over time.

---