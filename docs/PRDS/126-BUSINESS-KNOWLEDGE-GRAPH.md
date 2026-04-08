# PRD-126 — Business Knowledge Graph (Graphify Integration)

**Version:** 1.1
**Type:** Implementation
**Status:** Draft (updated: native viz tab, hyperedges, confidence scoring, graph_diff pulled to v1)
**Priority:** P1
**Research Base:** PRDs 08 (RAG), 11 (CodeGraph), 21 (NL2SQL), 05 (Memory), 76 (Reports), 80 (Context Service), 82A (Missions), 121 (HARNESS), 124 (Team Document Scoping), Mission Zero
**Open Source:** [safishamsi/graphify](https://github.com/safishamsi/graphify) — MIT License, 3.7k stars
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-04-08

---

## 1. Goal

**User-facing:** Your agents understand how your business fits together — not just individual documents, but how concepts, processes, people, and systems connect. Ask "what's the impact if we change our refund window?" and get an answer that traces through policy, Shopify config, email templates, budget rules, and customer FAQ — automatically.

**Technical:** Add a 5th knowledge source alongside RAG, NL2SQL, CodeGraph, and Memory. Integrate the open-source Graphify library (`pip install graphifyy[leiden]`) to build per-workspace relational knowledge graphs from documents, agent reports, database schemas, and connected app metadata. Expose graph traversal via 5 new platform tools. Wire into ContextService so agents receive relevant subgraph context alongside their existing memory and RAG injections.

The result: agents reason over *relationships between concepts* instead of flat document chunks. AUTO (CTO) gets a navigable map of the entire business. HARNESS uses `graph_diff()` to detect organizational drift. Mission Zero derives agent roster from graph communities instead of LLM guessing.

---

## 2. Background — Why a Knowledge Graph

### 2.1 The Gap in Current Knowledge Sources

| Source | Answers | Cannot Answer |
|--------|---------|---------------|
| RAG (PRD-08) | "What does this document say about X?" | "How does X relate to Y across different documents?" |
| NL2SQL (PRD-21) | "What are the numbers for X?" | "Why do these numbers matter to our process?" |
| CodeGraph (PRD-11) | "What function handles X?" | "What business process does this function serve?" |
| Memory (PRD-05) | "What have I learned about X?" | "What does the organization collectively know about X?" |

All four sources return **isolated results** — document chunks, database rows, code symbols, or agent memories. None returns the *relationships between concepts across sources*.

When a CEO asks "Are we ready to expand to the UK?" the answer requires traversing:
- SCOUT's research reports on UK competitors (documents)
- LEDGER's international shipping cost analysis (database + documents)
- COMMS' UK-localized content drafts (documents)
- Legal compliance nodes: GDPR, UK consumer protection (documents)
- Shopify UK marketplace integration status (connected app metadata)
- Gaps: what's missing that nobody has researched yet

No single knowledge source can produce this. A knowledge graph can.

### 2.2 Why Graphify

Graphify is a Python library (MIT licensed, 3.7k GitHub stars) that transforms documents and code into queryable knowledge graphs. Key properties:

1. **Two-pass extraction:** Deterministic AST parsing (tree-sitter, no LLM) for code + LLM-assisted semantic extraction for documents/images. Code never leaves the container.
2. **No embeddings required:** Leiden community detection uses graph topology (edge density), not vector embeddings. No vector DB dependency.
3. **71x token reduction:** Graph traversal returns structured relationships within a token budget, not raw document text.
4. **Pure Python, pip-installable:** `pip install graphifyy[leiden]` — NetworkX + tree-sitter + graspologic. No external services, no GPU.
5. **Incremental updates:** SHA256-based file cache. Only re-processes changed content.
6. **Library, not just CLI:** All functions importable — `extract()`, `build_from_json()`, `cluster()`, `_bfs()`, `_score_nodes()`, `_subgraph_to_text()`.

### 2.3 Relationship to Phase 3 (Neural Field Orchestration)

PRD-100 describes Phase 3 as "shared semantic landscapes" where agents operate on a common understanding. The Business Knowledge Graph is the concrete implementation of this concept — a workspace-scoped relational graph that all agents read from and contribute to. It bridges Phase 2 (Mission Mode) and Phase 3 by giving agents shared relational context without requiring the full Neural Field infrastructure.

---

## 3. What Ships

| Component | Description |
|-----------|-------------|
| `GraphifyService` | Orchestrator service. Builds, updates, and queries per-workspace knowledge graphs. Registered at startup. |
| Graph build pipeline | 4-source extraction: documents (LLM), code repos (tree-sitter), agent reports (LLM), DB schemas (introspection). Incremental via SHA256 cache. |
| 5 platform tools | `platform_query_graph`, `platform_graph_neighbors`, `platform_graph_communities`, `platform_graph_impact`, `platform_graph_stats` |
| `GraphSection` context section | New ContextService section. Injects relevant subgraph into agent prompts based on current task/question. Token-budgeted. |
| Workspace file storage | `/graph/graph.json`, `/graph/cache/`, `/graph/reports/`, `/graph/communities.json` |
| Post-ingest hook | Triggers incremental graph rebuild when documents are uploaded, agent reports submitted, or cloud docs synced. |
| Interactive HTML export | `graph.html` served as a workspace file — fallback human-viewable business map. |
| **Business Graph tab** | Native D3 visualization in Knowledge Bases page. Fork of `KnowledgeGraphVisualizer.tsx` with community coloring, god node highlighting, confidence filtering, search, and node detail panel. See §5.4. |
| **Graph diff** | `graph_diff()` on every rebuild — compares current graph to previous snapshot. Diff banner in UI shows what changed. |
| Hyperedge support | Group relationships (3+ nodes) extracted alongside nodes/edges. Feed into community detection. |

## 4. What Does NOT Ship (Deferred)

| Deferred | Target | Why |
|----------|--------|-----|
| Neo4j backend | v2 | NetworkX in-memory + JSON persistence is sufficient for v1 workspace sizes |
| Frontend graph visualization component | **v1 Phase 5** | Reuses existing `KnowledgeGraphVisualizer.tsx` (D3) — new "Business Graph" tab in Knowledge Bases page. See §5.4 and Phase 5. |
| Mission Zero v2 (roster from communities) | v2 | Requires graph to be stable and tested before deriving agent assignments |
| HARNESS `graph_diff` integration | **v1 Phase 5** | `graphify.analyze.graph_diff()` ships out of the box. 10 lines to wire into historical snapshots. See Phase 5. |
| Cross-workspace graph federation | v3 | Marketplace-level pattern sharing across tenants |
| Real-time graph streaming (WebSocket) | v3 | Batch rebuild is sufficient for business documents that change hourly, not secondly |
| Graph-aware mission decomposition | v2 | MissionPlanner could use graph to identify task dependencies, but v1 uses LLM-only decomposition |
| Shopify product graph auto-sync | v2 | V1 builds graph from documents about Shopify; v2 could pull product catalog directly via Composio |
| Code repo indexing via tree-sitter | v2 | Focus v1 on business documents; code repos add complexity with workspace worker coordination |
| Image/diagram extraction | v2 | Requires multimodal LLM calls; focus v1 on text documents |

---

## 5. Architecture — How It Fits

### 5.1 Knowledge Source Hierarchy

```
                         ┌──────────────────────────────┐
                         │      ContextService          │
                         │  (modules/context/service.py) │
                         └──────────────┬───────────────┘
                                        │
          ┌─────────┬──────────┬────────┼────────┬───────────┐
          │         │          │        │        │           │
       ┌──▼──┐  ┌──▼───┐  ┌──▼──┐  ┌──▼──┐  ┌──▼────┐  ┌──▼──┐
       │ RAG │  │NL2SQL │  │Code │  │Mem0 │  │GRAPH  │  │Tools│
       │     │  │       │  │Graph│  │     │  │(NEW)  │  │     │
       └─────┘  └───────┘  └─────┘  └─────┘  └───────┘  └─────┘
       chunks    rows       symbols  learned  relations   schemas
       by sim    by query   by code  per-agent cross-all  per-agent
```

### 5.2 When Each Source Is Used

The SmartIntentClassifier (9 intents) and SmartToolRouter already route by intent. Graph queries are triggered by **relational intent** — questions about connections, dependencies, impact, or cross-domain understanding.

| Signal in User Query | Primary Source | Graph Role |
|---------------------|---------------|------------|
| "What does [doc] say about X?" | RAG | Not used |
| "How many / what's the count / revenue" | NL2SQL | Not used |
| "What function / class / import" | CodeGraph | Not used |
| "Remember when / last time" | Memory | Not used |
| "How does X connect to Y?" | **Graph** | Primary |
| "What's the impact of changing X?" | **Graph** + NL2SQL | Graph traces dependencies, NL2SQL quantifies |
| "Who owns X? What depends on it?" | **Graph** | Primary |
| "Are we ready for X?" (multi-domain) | **Graph** + RAG + NL2SQL | Graph maps coverage, RAG fills details, NL2SQL adds numbers |
| "What's our end-to-end process for X?" | **Graph** | Primary — traverses the full chain |
| "What are we missing for X?" | **Graph** | Gap analysis — identifies missing nodes/edges |

### 5.3 Graph Tool Routing

Added to `auto.py` Tier 2 keywords and `SmartToolRouter`:

```python
_GRAPH_KEYWORDS = {
    "platform_query_graph": [
        "how does", "connected to", "relationship between", "relate to",
        "end to end", "flow", "process for", "overview of", "map of",
        "what connects", "trace from", "path between"
    ],
    "platform_graph_impact": [
        "what if we change", "impact of", "what breaks", "affects",
        "downstream", "upstream", "dependencies of", "depends on",
        "consequences of", "ripple effect"
    ],
    "platform_graph_communities": [
        "what areas", "departments", "clusters", "domains",
        "groups of", "categories of", "themes in"
    ],
    "platform_graph_neighbors": [
        "what's related to", "connected to", "associated with",
        "linked to", "touches", "involves"
    ],
    "platform_graph_stats": [
        "graph health", "knowledge coverage", "how complete",
        "graph size", "how connected"
    ]
}
```

### 5.4 Frontend Visualization Architecture

The Business Graph tab is a **native D3 component** in the Knowledge Bases page, not an iframe'd HTML file. This decision is based on existing infrastructure:

| Existing Component | What It Does | What We Reuse |
|-------------------|--------------|---------------|
| `KnowledgeGraphVisualizer.tsx` | D3 force-directed entity graph | Force layout, node coloring by type, search, zoom |
| `CodeGraphVisualization.tsx` | ReactFlow code symbol graph | Cluster grouping, heatmap coloring, mini-map |
| `document-management.tsx` | Knowledge Bases page with tabs | Tab pattern: `Documents \| Database \| Templates \| CodeGraph` |

**New tab:** `Business Graph` — added alongside existing tabs.

**Data flow:**

```
graph.json (workspace file)
  → GET /api/workspace/files/graph/graph.json
  → BusinessGraphPanel.tsx (fetches + parses)
  → BusinessGraphVisualization.tsx (D3 force-directed render)
```

**Component features:**
- **Community coloring** — each Leiden cluster gets a distinct color from the existing warm palette
- **God node highlighting** — node radius scaled by degree (most-connected concepts are visually larger)
- **Confidence slider** — filter edges below a threshold (hide INFERRED < 0.6, show only EXTRACTED, etc.)
- **Search** — type a concept name, matching nodes highlight and camera pans to focus
- **Click node** — detail panel shows: label, file_type, source_file, all edges with relation/confidence, community membership
- **Click node → open source** — if source_file is a workspace document, clicking opens it in the document viewer
- **Community sidebar** — lists all clusters by label, node count, cohesion score. Click to focus view on that cluster.
- **Diff banner** — if `/graph/latest_diff.json` exists, shows "3 new nodes, 1 removed since last build" with link to build report

**Design system compliance:**
- Glass-morphism card containers (existing pattern)
- Orange brand accent (#FF6B35) for selected/highlighted nodes
- Lucide React icons exclusively
- Dark theme primary
- Responsive — collapses sidebar on mobile

---

## 6. Design Decisions

### 6.1 Workspace Files, NOT New DB Tables

Same decision as HARNESS (PRD-121 §6.3). Graphs are too large and structurally complex for JSONB columns. Workspace files are:
- Agent-readable via `workspace_read_file`
- Human-browsable in the UI
- Naturally versioned by date
- No schema migration required

The graph lives at `/graph/graph.json` (NetworkX node-link format). Metadata (node count, edge count, last rebuild, community labels) is stored in `/graph/meta.json` for fast reads without loading the full graph.

### 6.2 LLM Extraction via Existing Pipeline, NOT Graphify's Skill Prompts

Graphify's semantic extraction (Pass 2) is implemented as Claude Code skill prompts, not Python functions. The Python library only does AST extraction. For Automatos:

- We reuse our existing LLM pipeline (`LLMManager` → OpenRouter) for semantic extraction
- We write our own extraction prompts that output Graphify's node/edge JSON schema
- This lets us use any model (not just Claude) and control costs via `config.LLM_MODEL`
- Code extraction (Pass 1) uses Graphify's tree-sitter pipeline directly — it's pure Python

### 6.3 Team-Scoped Graph Access (PRD-124 Integration)

The graph stores **all** workspace knowledge in a single `graph.json`, but queries are filtered by the requesting agent's `team` field (PRD-124). This avoids building separate per-team graphs (expensive, duplicative) while enforcing access control at query time.

**How it works:**

1. **Extraction tags nodes with `team_access`:** When a document is extracted, the resulting nodes inherit the document's `team_access` array from PRD-124. Nodes from public documents (`team_access = []`) are visible to all agents.

2. **Graph queries filter by team:** All 5 platform tools and `GraphSection` receive `agent_team` from the request context. Traversal skips nodes where `team_access` is non-empty and doesn't contain the agent's team. The filtering rule mirrors PRD-124's SQL: `team_access == [] OR agent_team IS NULL OR agent_team IN team_access`.

3. **Team-filtered traversal:** BFS/DFS treat team-blocked nodes as non-existent — they're never visited, never returned, and never appear in context injection. An agent cannot discover the existence of team-restricted nodes even through edge relationships.

4. **AUTO (team=NULL) sees everything:** The CTO agent has no team restriction, consistent with PRD-124's "agent with no team sees all documents" rule.

**Node schema addition:**

```json
{
  "id": "refund_policy",
  "label": "Refund Policy",
  "team_access": ["Support", "Operations"],
  "source_file": "/documents/policies/refund-policy.md"
}
```

**Why one graph, not per-team graphs:**
- Cross-team edges are valuable for AUTO's organizational awareness
- Leiden clustering on the full graph produces better communities than fragmented subgraphs
- `graph_diff()` is meaningful at workspace level, not team level
- Single build pipeline, single cache, single rebuild on document change

### 6.4 Async Wrapper Around Synchronous Library

Graphify is entirely synchronous. Our backend is async (FastAPI). All Graphify calls run in `asyncio.get_event_loop().run_in_executor(None, ...)` to avoid blocking the event loop. Graph builds run as background tasks, not inline with requests.

### 6.4 Incremental Builds, NOT Full Rebuilds

Graphify's SHA256 cache (`graphify.cache`) tracks file content hashes. When a document changes:
1. Post-ingest hook fires
2. `GraphifyService.incremental_update()` detects changed files via hash comparison
3. Only changed files re-extracted (LLM cost proportional to change, not total size)
4. Graph rebuilt from cached + new extractions
5. Leiden re-clustered (fast — O(n log n) on the graph, not the documents)

Full rebuild only on: first build, manual trigger, or cache corruption.

### 6.5 Graph Loaded On-Demand, NOT At Startup

Graphs can be 10-50MB for large workspaces. Loading all workspace graphs at server startup would consume too much memory. Instead:

- `GraphifyService` maintains an LRU cache of loaded graphs (max 20 workspaces)
- First query for a workspace loads its `graph.json` from workspace files
- Cache eviction on LRU basis
- Graph rebuild invalidates cache entry

### 6.6 Platform Tools, NOT Core Tools

Graph tools follow the 3-file platform tool pattern (like HARNESS, governance, workspace tools). They are NOT core tools because:
- They depend on workspace context (workspace_id scoping)
- They're dispatched through `PlatformActionExecutor`
- They follow the existing `platform_*` naming convention
- Agent access is controlled via existing tool assignment, not hardcoded

---

## 7. Graph Schema

### 7.1 Node Types

| `file_type` | Source | Examples |
|-------------|--------|----------|
| `document` | RAG pipeline (uploaded docs, synced cloud docs) | "Refund Policy", "Brand Guidelines", "Q2 Marketing Plan" |
| `report` | Agent reports (PRD-76) | "SCOUT Weekly Competitor Analysis", "LEDGER Monthly Budget Review" |
| `concept` | LLM extraction from documents | "Customer Retention", "UK Market Expansion", "Free Shipping Threshold" |
| `entity` | LLM extraction | "Shopify", "Stripe", "GDPR", "Product X" |
| `process` | LLM extraction | "Order Fulfillment", "Refund Flow", "Onboarding Sequence" |
| `metric` | NL2SQL schema introspection | "Monthly Revenue", "Refund Rate", "Customer Acquisition Cost" |
| `rule` | Blueprints + governance | "Max Budget Per Agent: $50/week", "Require System Prompt" |
| `agent` | Agent roster | "AUTO", "SCOUT", "LEDGER" (with role, team, skills) |
| `integration` | Connected apps via Composio | "Shopify Store", "Gmail", "Slack #general" |
| `table` | NL2SQL schema | "orders", "products", "customers" |
| `code` | Tree-sitter AST (v2, deferred) | Functions, classes, modules |

### 7.2 Node Schema

```json
{
  "id": "refund_policy",
  "label": "Refund Policy",
  "file_type": "document",
  "source_file": "/documents/policies/refund-policy.md",
  "source_location": null,
  "workspace_id": "ws_42",
  "team_access": ["Support", "Operations"],
  "source_type": "rag",
  "last_updated": "2026-04-01T10:30:00Z",
  "confidence": "EXTRACTED",
  "metadata": {
    "document_id": 156,
    "chunk_count": 8,
    "word_count": 2400
  }
}
```

### 7.3 Edge Types

| `relation` | Direction | Example |
|------------|-----------|---------|
| `contains` | parent → child | "Refund Policy" → "30-Day Window Rule" |
| `implements` | process → rule | "Refund Flow" → "Refund Policy" |
| `references` | doc → concept | "Q2 Plan" → "UK Market Expansion" |
| `depends_on` | process → process | "Refund Flow" → "Payment Processing" |
| `constrained_by` | entity → rule | "Marketing Spend" → "Q2 Budget: $15K" |
| `owned_by` | process → agent | "Competitor Monitoring" → "SCOUT" |
| `measures` | metric → process | "Refund Rate" → "Refund Flow" |
| `integrates_with` | process → integration | "Order Fulfillment" → "Shopify Store" |
| `stored_in` | metric → table | "Monthly Revenue" → "orders" |
| `reports_to` | agent → agent | "SCOUT" → "AUTO" |
| `semantically_similar_to` | concept → concept | "Customer Churn" ↔ "Retention Rate" |
| `triggers` | event → process | "New Order" → "Fulfillment Flow" |
| `conflicts_with` | rule → rule | "Free Shipping > $50" ↔ "Minimum AOV Target: $35" |

### 7.4 Edge Schema

```json
{
  "source": "refund_flow",
  "target": "refund_policy",
  "relation": "implements",
  "confidence": "EXTRACTED",
  "confidence_score": 1.0,
  "source_file": "/documents/processes/refund-flow.md",
  "source_location": null,
  "weight": 1.0,
  "_src": "refund_flow",
  "_tgt": "refund_policy"
}
```

### 7.4b Hyperedge Schema

Hyperedges capture **group relationships** where 3+ nodes participate in a shared concept, flow, or pattern that pairwise edges alone cannot express. Examples: all agents in an authentication flow, all components of a deployment pipeline, all policies governing a business process.

```json
{
  "id": "order_fulfillment_pipeline",
  "label": "Order Fulfillment Pipeline",
  "nodes": ["shopify_store", "payment_processing", "shipping_service", "email_notification"],
  "relation": "participate_in",
  "confidence": "EXTRACTED",
  "confidence_score": 0.95,
  "source_file": "/documents/processes/order-flow.md"
}
```

Hyperedges are extracted by the LLM alongside nodes and edges (see §8.2). Maximum 3 per document to avoid noise. They feed directly into community detection — Leiden uses hyperedge membership as a strong clustering signal.
```

### 7.5 Confidence Levels

| Level | Score Range | Meaning | Source |
|-------|-------------|---------|--------|
| `EXTRACTED` | 1.0 | Directly stated in source material | AST parsing, explicit document references |
| `INFERRED` | 0.4–0.9 (per-edge) | Derived by LLM from context. Score is **per-edge, not flat** — the LLM must reason about each edge individually. Direct structural evidence (shared data, clear dependency): 0.8–0.9. Reasonable inference: 0.6–0.7. Weak/speculative: 0.4–0.5. | Semantic extraction, similarity detection |
| `AMBIGUOUS` | 0.1–0.3 | Uncertain relationship — flagged for review | Cross-document inference, low-confidence LLM output |

> **Note:** Never use 0.5 as a default. The extraction prompt (§8.2) requires the LLM to justify each score. Flat scores lose the signal needed for confidence-based filtering in `platform_graph_impact` and `GraphSection`.

---

## 8. Graph Build Pipeline

### 8.1 Sources and Extraction Methods

| Source | Extraction | LLM Required | Trigger |
|--------|-----------|-------------|---------|
| **Documents** (uploaded, cloud-synced) | LLM prompt: extract concepts, entities, processes, relationships | Yes | Post-ingest hook in document pipeline |
| **Agent Reports** (PRD-76) | LLM prompt: extract findings, recommendations, entities mentioned | Yes | Post-report hook in `platform_submit_report` |
| **DB Schemas** (NL2SQL connections) | Introspection: tables, columns, FK relationships → nodes + edges | No | On database knowledge source connection |
| **Agent Roster** | Direct mapping: agent records → nodes, `reports_to` → edges | No | On agent create/update |
| **Blueprints & Guardrails** | Direct mapping: rules → nodes, agent assignments → edges | No | On blueprint create/update |
| **Connected Apps** (Composio) | Metadata mapping: app name, actions → nodes | No | On app connection |

### 8.2 Document Extraction Prompt

The LLM extraction prompt for documents outputs Graphify-compatible JSON:

```
You are extracting a knowledge graph from a business document.

Given the document below, extract:
1. CONCEPTS — abstract ideas, strategies, goals mentioned
2. ENTITIES — named things: products, companies, people, tools, services
3. PROCESSES — workflows, procedures, sequences described
4. METRICS — measurable quantities referenced
5. RULES — constraints, policies, thresholds stated
6. RELATIONSHIPS — how the above connect to each other

Output JSON:
{
  "nodes": [
    {"id": "snake_case_id", "label": "Human Name", "file_type": "concept|entity|process|metric|rule", "source_file": "<doc_path>"}
  ],
  "edges": [
    {"source": "node_id_a", "target": "node_id_b", "relation": "<relation_type>", "confidence": "EXTRACTED|INFERRED|AMBIGUOUS", "confidence_score": 0.85}
  ],
  "hyperedges": [
    {"id": "snake_case_id", "label": "Human Label", "nodes": ["node_id_1", "node_id_2", "node_id_3"], "relation": "participate_in|implement|form", "confidence": "EXTRACTED|INFERRED", "confidence_score": 0.9, "source_file": "<doc_path>"}
  ]
}

Rules:
- Use snake_case IDs derived from the label
- Only create edges where the relationship is clearly stated or strongly implied
- Mark directly stated relationships as EXTRACTED (confidence_score: 1.0)
- Mark implied relationships as INFERRED with a per-edge confidence_score:
  - 0.8–0.9: strong structural evidence (shared data, clear dependency)
  - 0.6–0.7: reasonable inference with some uncertainty
  - 0.4–0.5: weak or speculative. Never default to 0.5 — reason about each edge.
- Mark uncertain relationships as AMBIGUOUS (confidence_score: 0.1–0.3)
- Do not hallucinate entities not present in the document
- Prefer specific labels over generic ones ("30-Day Refund Window" not "Time Limit")
- Add hyperedges when 3+ nodes participate in a shared concept/flow/pattern that pairwise edges cannot capture. Maximum 3 per document. Use sparingly.
```

### 8.3 Build Sequence

```
1. Collect sources
   ├── List workspace documents (S3 key listing)
   ├── List agent reports (platform_get_latest_report per agent)
   ├── List DB schemas (database_knowledge table)
   ├── List agent roster (agents table)
   ├── List blueprints (agent_blueprints table)
   └── List connected apps (composio_connections table)

2. Check cache (SHA256 per source)
   ├── Unchanged sources → load cached extraction JSON
   └── Changed/new sources → queue for extraction

3. Extract (parallel where possible)
   ├── Documents → LLM extraction (asyncio.gather, batched by 5)
   ├── Reports → LLM extraction (asyncio.gather, batched by 5)
   ├── DB schemas → deterministic introspection
   ├── Roster → deterministic mapping
   ├── Blueprints → deterministic mapping
   └── Connected apps → deterministic mapping

4. Merge extractions
   └── Combine all node/edge JSON arrays, deduplicate by node ID

5. Build graph
   └── graphify.build_from_json(merged_extraction) → NetworkX graph

6. Cluster
   └── graphify.cluster(G) → communities dict

7. Analyze
   ├── graphify.god_nodes(G, top_n=20) → most-connected concepts
   └── graphify.surprising_connections(G, communities) → cross-cluster edges

8. Export
   ├── graphify.to_json(G, communities, "/graph/graph.json")
   ├── graphify.to_html(G, communities, "/graph/graph.html")
   └── Write /graph/meta.json (node_count, edge_count, community_count, last_built, god_nodes)

9. Cache
   └── Save SHA256 + extraction JSON per source to /graph/cache/
```

### 8.4 Token Cost Estimate

| Source Type | Avg Tokens per Extraction | Count (typical workspace) | Total |
|-------------|--------------------------|---------------------------|-------|
| Document (LLM) | ~2,000 input + ~500 output | 50 docs | ~125K tokens |
| Agent Report (LLM) | ~1,000 input + ~300 output | 14 agents × 4 reports | ~73K tokens |
| DB Schema (no LLM) | 0 | N/A | 0 |
| Roster/Blueprints (no LLM) | 0 | N/A | 0 |
| **Total first build** | | | **~400–500K tokens** |
| **Incremental update** (5 changed docs) | | | **~25K tokens** |

At OpenRouter rates (~$0.50/M tokens for Haiku-class), first build costs ~$0.20–0.25. Incremental updates are negligible (~$0.01). Validated empirically: graphify extraction on 267 Automatos docs consumed ~1.2M tokens across 15 parallel agents — a 50-doc workspace scales proportionally to ~400–500K.

---

## 9. Platform Tools

### 9.1 `platform_query_graph` (read)

Natural language query over the business graph. Uses BFS/DFS traversal from keyword-matched start nodes.

```json
{
  "name": "platform_query_graph",
  "description": "Query the business knowledge graph with a natural language question. Returns relevant concepts, entities, and their relationships within a token budget.",
  "parameters": {
    "question": {"type": "string", "description": "Natural language question about business relationships"},
    "mode": {"type": "string", "enum": ["bfs", "dfs"], "default": "bfs", "description": "Traversal mode: bfs for breadth-first (broad context), dfs for depth-first (deep chains)"},
    "depth": {"type": "integer", "default": 3, "description": "Maximum traversal depth from start nodes"},
    "token_budget": {"type": "integer", "default": 2000, "description": "Maximum tokens in response"}
  },
  "_internal": {
    "team": "Resolved from agent.team (PRD-124). Not user-facing. Filters traversal to team-accessible nodes only."
  }
}
```

**Implementation:** Wraps `_score_nodes()` → `_bfs()` / `_dfs()` → `_subgraph_to_text()` from `graphify.serve`.

### 9.2 `platform_graph_neighbors` (read)

Direct connections of a specific concept.

```json
{
  "name": "platform_graph_neighbors",
  "description": "Get all direct connections of a concept in the business knowledge graph. Shows what a concept relates to and how.",
  "parameters": {
    "concept": {"type": "string", "description": "Name or ID of the concept to explore"},
    "relation_filter": {"type": "string", "description": "Optional: filter by relationship type (e.g., 'depends_on', 'owned_by')"}
  }
}
```

### 9.3 `platform_graph_communities` (read)

Business domain clusters detected by Leiden algorithm.

```json
{
  "name": "platform_graph_communities",
  "description": "List the detected business domains (clusters) in the knowledge graph. Each community represents a group of closely related concepts.",
  "parameters": {
    "community_id": {"type": "integer", "description": "Optional: get details for a specific community. Omit to list all communities."}
  }
}
```

### 9.4 `platform_graph_impact` (read)

Impact analysis — what depends on a given concept.

```json
{
  "name": "platform_graph_impact",
  "description": "Analyze the impact of changing or removing a concept. Traces all downstream dependencies, affected processes, agents, and rules.",
  "parameters": {
    "concept": {"type": "string", "description": "The concept to analyze impact for"},
    "max_depth": {"type": "integer", "default": 4, "description": "Maximum depth of dependency traversal"}
  }
}
```

**Implementation:** BFS from concept node. Traverses directional edges forward (`depends_on`, `implements`, `constrained_by`, `triggers`, `measures`) AND bidirectional edges both ways (`semantically_similar_to`, `conflicts_with`). Conflict edges are critical for impact analysis — changing a rule that conflicts with another rule must surface both. Returns affected nodes grouped by depth level with edge relation context.

### 9.5 `platform_graph_stats` (read)

Graph health and coverage metrics.

```json
{
  "name": "platform_graph_stats",
  "description": "Get knowledge graph health metrics: node count, edge count, community count, god nodes (most connected concepts), orphan nodes, last rebuild time.",
  "parameters": {}
}
```

**Implementation:** Reads `/graph/meta.json` (fast path — no graph load required).

---

## 10. ContextService Integration

### 10.1 New Section: `GraphSection`

Registered in `SECTION_REGISTRY` alongside existing sections (identity, personality, skills, tools, memory, etc.).

```python
class GraphSection(ContextSection):
    """Injects relevant business graph context into agent prompts."""

    name = "business_graph"
    priority = 45  # After memory (40), before tools (50)
    max_tokens = 800  # Token budget for graph context

    async def render(self, context: ContextInput) -> str | None:
        if not context.current_message:
            return None

        graph = await self.graph_service.load_graph(context.workspace_id)
        if graph is None:
            return None

        # Use classified intent + extracted entities for scoring (not raw word splits).
        # SmartIntentClassifier already extracts entities — reuse them for graph relevance.
        # Fallback to message.split() only if classification unavailable.
        terms = context.extracted_entities or context.current_message.split()
        scored = _score_nodes(graph, terms)
        if not scored or scored[0][0] < 0.3:  # No relevant nodes
            return None

        # BFS from top-scoring nodes
        start_nodes = [nid for score, nid in scored[:3] if score >= 0.3]
        nodes, edges = _bfs(graph, start_nodes, depth=2)
        subgraph_text = _subgraph_to_text(graph, nodes, edges, token_budget=self.max_tokens)

        return f"## Business Context (Knowledge Graph)\n\n{subgraph_text}"
```

### 10.2 Context Modes That Include Graph

| ContextMode | Include Graph? | Rationale |
|-------------|---------------|-----------|
| `CHATBOT` | Yes | User questions may be relational |
| `TASK_EXECUTION` | Yes | Mission tasks benefit from business context |
| `HEARTBEAT` | No | Heartbeats have fixed prompts; graph context adds noise |
| `COORDINATOR` | No | Coordinator manages task lifecycle, not business reasoning |
| `VERIFIER` | Yes (read-only) | Verifier can check if output is consistent with known relationships |
| `WIDGET` | No | Widgets have narrow, pre-defined queries |

### 10.3 Token Budget

GraphSection gets 800 tokens max (configurable). This is small enough to not crowd out other context sections, but large enough to include 15-25 relevant nodes with relationships. The `_subgraph_to_text()` function handles truncation to budget.

---

## 11. Post-Ingest Hooks

### 11.1 Document Upload / Cloud Sync

After a document is successfully processed by the RAG pipeline (status = "completed" in `documents` table):

```python
# In document processing pipeline, after successful ingest
await graph_service.schedule_incremental_update(
    workspace_id=workspace_id,
    changed_sources=[{"type": "document", "path": doc_path, "id": doc_id}]
)
```

### 11.2 Agent Report Submission

After `platform_submit_report` writes the report:

```python
# In handlers_reporting.py, after report creation
await graph_service.schedule_incremental_update(
    workspace_id=workspace_id,
    changed_sources=[{"type": "report", "path": report_path, "agent_id": agent_id}]
)
```

### 11.3 Debouncing

Multiple rapid changes (e.g., batch document upload) are debounced. `schedule_incremental_update()` queues a rebuild job with a 60-second delay. If another change arrives within the window, the timer resets. Only one rebuild runs per debounce window.

---

## 12. Workspace File Layout

```
/graph/
  graph.json                    # NetworkX node-link format (full graph)
  graph.html                    # Interactive vis.js visualization
  meta.json                     # Quick stats: counts, god_nodes, last_built
  communities.json              # Community labels + member lists
  cache/
    {sha256_hash}.json          # Cached extraction per source file
  reports/
    {YYYY-MM-DD}_build.md       # Build report: what changed, new nodes/edges
  history/
    {YYYY-MM-DD}_graph.json     # Historical snapshots (for graph_diff in v2)
```

---

## 13. Agent Usage Scenarios

### 13.1 AUTO (CTO) — Organizational Awareness

```
User: "How's the business doing?"

AUTO's process:
1. platform_graph_stats → "142 nodes, 387 edges, 8 communities, 3 orphans"
2. platform_graph_communities → lists 8 business domains
3. platform_query_graph("key risks and dependencies") → traces critical paths
4. Synthesizes: "Your business knowledge covers 8 domains. Marketing and
   Sales are tightly connected (community 0, 34 nodes). Supply chain has
   3 orphan concepts — no processes connect to 'Warehouse SLA' or
   'Supplier Contract #7'. Recommend RADAR investigate."
```

### 13.2 ATLAS (BI Lead) — Impact Analysis

```
User: "What happens if we discontinue Product X?"

ATLAS's process:
1. platform_graph_impact("Product X", max_depth=4) →
   - Product X → Collection "Summer Essentials" (contains)
   - Product X → 3 email campaigns (references)
   - Product X → Blog post "Top 5 Summer Picks" (features)
   - Product X → 12% of CLOSER's demo scripts (uses)
   - Product X → Supplier Agreement #7 (constrained_by)
2. platform_query_data("SELECT SUM(total) FROM orders WHERE product_name = 'Product X' AND created_at > NOW() - INTERVAL '30 days'") → $45K/mo
3. search_knowledge("supplier agreement product X") → RAG pulls contract text
4. Synthesizes impact report with financial + operational + content effects
```

### 13.3 SCOUT (Market Research) — Knowledge Contribution

```
SCOUT's weekly heartbeat:
1. Web research → competitor analysis
2. platform_submit_report(type="research", title="Competitor Pricing Update")
3. Post-report hook → graph_service.schedule_incremental_update()
4. Graph auto-extracts: new competitor entity nodes, pricing relationship edges
5. Next time ANY agent queries "competitive landscape" → graph returns
   the full competitive map with connections to own products/pricing
```

### 13.4 ORACLE (Knowledge Ops) — Graph Curator

```
ORACLE's role with the graph:
1. platform_graph_stats → identifies orphan nodes, stale communities
2. platform_graph_communities → checks for oversized clusters (> 25% of graph)
3. Resolves conflicts: SCOUT says competitor price is X, CLOSER says Y
   → graph has both nodes, ORACLE merges/corrects
4. Submits curation report: "Pruned 5 stale nodes from Q1 docs,
   merged 3 duplicate competitor entries, flagged 2 conflicts for review"
```

### 13.5 Shopify Store — End-to-End Example

A workspace running a Shopify store. User has uploaded:
- Brand guidelines PDF
- Product pricing spreadsheet
- Customer service SOPs
- Refund policy
- Q2 marketing plan

Connected: Shopify (via Composio), Stripe, Gmail, Slack.

**Graph after first build:**

```
Community 0 — "Products & Catalog" (28 nodes)
  Product X ← collection "Summer" ← Shopify Store
  Product Y ← collection "Basics" ← Shopify Store
  Pricing Strategy → Free Shipping Threshold ($50)
  Pricing Strategy → Competitor Benchmark

Community 1 — "Customer Operations" (22 nodes)
  Refund Policy → 30-Day Window → Shopify Returns API
  Refund Flow → Email Notification → Gmail
  Customer FAQ → Refund Policy
  GUIDE (agent) ← owns → Customer FAQ

Community 2 — "Marketing & Brand" (19 nodes)
  Brand Guidelines → Tone of Voice → COMMS (agent)
  Q2 Plan → Email Campaign → Product X
  Q2 Plan → Social Campaign → Instagram
  SCOUT (agent) ← owns → Competitor Monitoring

Community 3 — "Finance & Operations" (15 nodes)
  Stripe → Payment Processing → Order Fulfillment
  LEDGER (agent) ← owns → Budget Tracking
  Q2 Budget: $15K → constrained_by → Marketing Spend

Community 4 — "Governance" (12 nodes)
  Blueprint: cost-tier → PULSE, PATCH, NEXUS, PROSPECT
  Blueprint: strict → LEDGER, SENTINEL
  Max Budget Per Agent: $50/week
```

**CEO asks:** "What if we reduce the refund window from 30 to 14 days?"

```
platform_graph_impact("30-Day Window"):
  Depth 1: Refund Policy (contains), Refund Flow (implements)
  Depth 2: Customer FAQ (references Refund Policy), Shopify Returns API
            (implements Refund Flow), Email Notification (triggers)
  Depth 3: GUIDE (owns Customer FAQ), COMMS (owns Email Notification),
            Gmail (sends Email Notification)
  Depth 4: Customer Satisfaction Metric (measures Refund Flow)

Result: "Changing the refund window affects:
  - Refund Policy (update doc)
  - Shopify Returns API config (update setting)
  - Customer FAQ (GUIDE needs to update 3 answers)
  - Email notification template (COMMS needs to update)
  - Customer satisfaction metric (ATLAS should monitor for impact)
  5 agents involved, 8 downstream items to update."
```

---

## 14. Implementation Phases

### Phase 1 — Core Service + Graph Build (3-4 days)

**New Files:**

| File | ~Lines | Purpose |
|------|--------|---------|
| `orchestrator/modules/knowledge/graph_service.py` | 350 | `GraphifyService`: build, rebuild, load, query. LRU cache. Debounced updates. |
| `orchestrator/modules/knowledge/graph_extraction.py` | 200 | LLM extraction prompts + response parsing. Schema-to-graph mappers for roster/blueprints/DB schemas. |

**Modified Files:**

| File | Change |
|------|--------|
| `requirements.txt` | Add `graphifyy[leiden]` |

**Acceptance Criteria:**
- [ ] `pip install graphifyy[leiden]` succeeds in container
- [ ] `GraphifyService.build_graph(workspace_id)` produces valid `graph.json`
- [ ] Incremental update only re-extracts changed sources
- [ ] SHA256 cache persists across restarts
- [ ] Graph loads in < 2s for 200-node graph

### Phase 2 — Platform Tools (3-4 days)

**New Files:**

| File | ~Lines | Purpose |
|------|--------|---------|
| `orchestrator/modules/tools/discovery/actions_graph.py` | 120 | 5 ActionDefinitions registered via `register_graph_actions()` |
| `orchestrator/modules/tools/discovery/handlers_graph.py` | 250 | Handler functions for 5 platform tools |

**Modified Files:**

| File | Change |
|------|--------|
| `orchestrator/modules/tools/discovery/platform_actions.py` | Add `from .actions_graph import register_graph_actions` + call in `register_all_actions()` |
| `orchestrator/modules/tools/execution/unified_executor.py` | Add 5 handler entries to `_handlers` dict |
| `orchestrator/consumers/chatbot/auto.py` | Add graph keywords to `_PLATFORM_KEYWORDS` |

**Acceptance Criteria:**
- [ ] All 5 tools registered in ActionRegistry
- [ ] `platform_query_graph` returns relevant subgraph text within token budget
- [ ] `platform_graph_impact` traces dependencies correctly
- [ ] `platform_graph_communities` lists detected clusters
- [ ] `platform_graph_stats` returns from meta.json without loading full graph
- [ ] Tools routed correctly via SmartToolRouter keywords

### Phase 3 — Post-Ingest Hooks (2 days)

**Modified Files:**

| File | Change |
|------|--------|
| `orchestrator/modules/rag/services/knowledge_multimodal.py` | Add post-ingest hook: `graph_service.schedule_incremental_update()` after document processing completes |
| `orchestrator/modules/tools/discovery/handlers_reporting.py` | Add post-report hook: `graph_service.schedule_incremental_update()` after `platform_submit_report` |
| `orchestrator/api/database_knowledge.py` | Add hook: rebuild graph when NL2SQL data source connected |
| `orchestrator/api/agent_endpoints.py` | Add hook: update graph when agent created/updated |

**Acceptance Criteria:**
- [ ] Document upload triggers incremental graph update
- [ ] Agent report triggers incremental graph update
- [ ] Multiple rapid changes debounce to single rebuild
- [ ] Graph stays current without manual intervention

### Phase 4 — ContextService Integration (2-3 days)

**New Files:**

| File | ~Lines | Purpose |
|------|--------|---------|
| `orchestrator/modules/context/sections/graph_context.py` | 80 | `GraphSection` — renders relevant subgraph into agent context |

**Modified Files:**

| File | Change |
|------|--------|
| `orchestrator/modules/context/service.py` | Register `GraphSection` in `SECTION_REGISTRY` |
| `orchestrator/modules/context/modes.py` | Add `business_graph` to CHATBOT, TASK_EXECUTION, VERIFIER modes |

**Acceptance Criteria:**
- [ ] Agent prompts include `## Business Context` section when graph has relevant nodes
- [ ] No graph context injected when question is unrelated (score < 0.3)
- [ ] Token budget respected (≤ 800 tokens for graph section)
- [ ] Graph context appears in CHATBOT and TASK_EXECUTION modes
- [ ] Graph context does NOT appear in HEARTBEAT or COORDINATOR modes

### Phase 5 — Native Graph Visualization Tab + HARNESS graph_diff (5-7 days)

**Why native, not iframe:** Automatos already has `KnowledgeGraphVisualizer.tsx` (D3 force-directed) and `CodeGraphVisualization.tsx` (ReactFlow) in the frontend. The Business Graph tab reuses the D3 component with graph.json data — it looks and feels native, uses the existing glass-morphism design system, and can interact with other Automatos UI components (click a node → open the source document, filter by agent ownership, etc.). An iframe'd HTML file would be a dead end.

**New tab in Knowledge Bases page:**

```
Documents | Database | Templates | CodeGraph | Business Graph
                                               ^^^^^^^^^^^^
```

**New Files:**

| File | ~Lines | Purpose |
|------|--------|---------|
| `frontend/components/knowledge/BusinessGraphVisualization.tsx` | 300 | D3 force-directed graph fed by `graph.json`. Fork of `KnowledgeGraphVisualizer.tsx` with: community coloring (Leiden clusters → color groups), god node highlighting (larger radius for high-degree nodes), confidence filtering slider, search with focus. |
| `frontend/components/knowledge/BusinessGraphPanel.tsx` | 200 | Tab container. Fetches `graph.json` + `meta.json` via workspace files API. Stats bar (node count, edge count, communities, last rebuilt). Community sidebar (list clusters, click to focus). Node detail panel (click node → edges, source file, confidence). |
| `frontend/components/knowledge/GraphDiffBanner.tsx` | 80 | Shows "3 new nodes, 1 removed since last build" when `graph_diff` data exists. Links to build report. |

**Modified Files:**

| File | Change |
|------|--------|
| `frontend/components/documents/document-management.tsx` | Add `Business Graph` tab alongside existing `Documents \| Database \| Templates \| CodeGraph` tabs |
| `orchestrator/modules/knowledge/graph_service.py` | Add `to_html()` call in build pipeline (fallback). Add `graph_diff()` call comparing current graph to previous snapshot in `/graph/history/`. Save diff to `/graph/latest_diff.json`. |

**HARNESS graph_diff (trivial — pulled from v2):**

```python
# In graph_service.py build pipeline, after step 8 (Export)
from graphify.analyze import graph_diff

if history_path.exists():
    old_graph = load_graph(history_path / f"{yesterday}_graph.json")
    diff = graph_diff(old_graph, new_graph)
    save_json(workspace_files / "graph/latest_diff.json", diff)
    # diff = {"new_nodes": [...], "removed_nodes": [...], "new_edges": [...], "summary": "..."}

# Save today's snapshot for next diff
save_json(history_path / f"{today}_graph.json", graph_data)
```

**Acceptance Criteria:**
- [ ] "Business Graph" tab visible in Knowledge Bases page
- [ ] D3 graph renders nodes colored by community with force-directed layout
- [ ] God nodes visually larger (radius scaled by degree)
- [ ] Click node → detail panel shows label, edges, source file, confidence
- [ ] Confidence slider filters edges below threshold
- [ ] Community sidebar lists clusters, click focuses view
- [ ] Search bar highlights matching nodes
- [ ] `graph.html` still generated as fallback (workspace file browser)
- [ ] `graph_diff()` runs on every build, diff saved to `/graph/latest_diff.json`
- [ ] `GraphDiffBanner` shows change summary when diff data exists
- [ ] Design follows existing glass-morphism + orange brand system

---

## 15. Build Plan Summary

| Phase | What | Builder | Files | Days | Dependencies |
|-------|------|---------|-------|------|-------------|
| 1 | Core service + graph build | Ralph | 2 new | 3-4 | None |
| 2 | Platform tools | Ralph | 2 new, 3 modified | 3-4 | Phase 1 |
| 3 | Post-ingest hooks | Ralph | 0 new, 4 modified | 2 | Phase 1 |
| 4 | ContextService integration | Ralph | 1 new, 2 modified | 2-3 | Phase 1 |
| 5 | Native graph visualization tab + graph_diff | Ralph | 3 new, 2 modified | 5-7 | Phase 1 |
| **Total** | | | **8 new, 11 modified** | **~16-20 days** | |

Phases 2-5 can run in parallel after Phase 1 completes.

### Phase dependency diagram

```
Phase 1 (Core Service)
  ├── Phase 2 (Platform Tools)
  ├── Phase 3 (Post-Ingest Hooks)
  ├── Phase 4 (ContextService)
  └── Phase 5 (Visualization Tab + graph_diff)
        └── Requires Phase 1 graph.json format stable
```

---

## 16. Key Integration Points

| System | How It's Used | What Does NOT Change |
|--------|--------------|---------------------|
| **Team Document Scoping** (PRD-124) | Nodes inherit `team_access` from source documents. All graph queries filter by `agent.team`. See §6.3. | PRD-124's document filtering, API key team locking, and widget scoping unchanged |
| **RAG pipeline** (PRD-08) | Post-ingest hook triggers graph update | RAG chunking, retrieval, S3 storage unchanged |
| **NL2SQL** (PRD-21) | Schema introspection feeds table/column nodes into graph | NL2SQL query execution unchanged |
| **Memory** (PRD-05, Mem0) | Graph is workspace-scoped, memory is agent-scoped — they don't overlap | Memory injection unchanged |
| **CodeGraph** (PRD-11) | Deferred to v2 — code nodes will complement CodeGraph symbols | CodeGraph unchanged |
| **ContextService** (PRD-80) | New `GraphSection` registered alongside existing sections | All other sections unchanged |
| **Tool Router** (PRD-122) | 5 new tools registered via ActionRegistry | Tool routing logic unchanged |
| **Reports** (PRD-76) | Post-report hook feeds agent outputs into graph | Report storage/display unchanged |
| **HARNESS** (PRD-121) | `graph_diff()` runs on every build. Diff saved to `/graph/latest_diff.json`. HARNESS reads via `workspace_read_file`. | HARNESS loop logic unchanged — diff is a new input file it can optionally consume |
| **Missions** (PRD-82A) | v2: graph-aware decomposition | v1: Mission system unchanged |
| **Agent Roster** | Roster agents become graph nodes with `reports_to` edges | Agent CRUD unchanged |
| **Blueprints** | Blueprint rules become graph nodes | Blueprint enforcement unchanged |

---

## 17. Sequence Diagram — Query Flow

```
User                    Chatbot Pipeline           GraphifyService         Graphify Library
  │                           │                          │                        │
  │  "What connects our       │                          │                        │
  │   refund flow to Shopify?"│                          │                        │
  │──────────────────────────>│                          │                        │
  │                           │                          │                        │
  │                    SmartIntentClassifier              │                        │
  │                    → intent: "graph_query"            │                        │
  │                           │                          │                        │
  │                    SmartToolRouter                    │                        │
  │                    → includes platform_query_graph    │                        │
  │                           │                          │                        │
  │                    ContextService.build()             │                        │
  │                           │──GraphSection.render()──>│                        │
  │                           │                          │──load_graph(ws_id)────>│
  │                           │                          │<──NetworkX graph───────│
  │                           │                          │──_score_nodes()───────>│
  │                           │                          │──_bfs(depth=2)────────>│
  │                           │                          │──_subgraph_to_text()──>│
  │                           │<──"## Business Context"──│                        │
  │                           │                          │                        │
  │                    LLM call (with graph context       │                        │
  │                    injected in system prompt)         │                        │
  │                           │                          │                        │
  │                    Agent also calls                   │                        │
  │                    platform_query_graph               │                        │
  │                           │──execute_tool()─────────>│                        │
  │                           │                          │──_score_nodes()───────>│
  │                           │                          │──_bfs(depth=3)────────>│
  │                           │<──subgraph text──────────│                        │
  │                           │                          │                        │
  │<──"The refund flow        │                          │                        │
  │    connects to Shopify    │                          │                        │
  │    through..."            │                          │                        │
```

---

## 18. Testing Strategy

### Unit Tests

| Test | What It Validates |
|------|-------------------|
| `test_graph_extraction_prompt` | LLM extraction prompt produces valid node/edge JSON |
| `test_graph_build_from_extraction` | `build_from_json()` creates correct NetworkX graph |
| `test_graph_clustering` | Leiden detects expected communities from known test graph |
| `test_graph_query_bfs` | BFS from scored nodes returns expected traversal |
| `test_graph_impact_analysis` | Impact traversal follows directional edges correctly |
| `test_graph_cache_hit` | SHA256 cache prevents re-extraction of unchanged docs |
| `test_graph_cache_miss` | Modified doc triggers re-extraction |
| `test_graph_debounce` | Multiple rapid changes produce single rebuild |
| `test_graph_token_budget` | `_subgraph_to_text()` respects token limit |
| `test_graph_lru_eviction` | Cache evicts oldest workspace graph at capacity |
| `test_graph_section_relevance` | GraphSection returns None when no nodes score > 0.3 |

### Integration Tests

| Test | What It Validates |
|------|-------------------|
| `test_document_upload_triggers_rebuild` | Post-ingest hook fires and graph updates |
| `test_report_triggers_rebuild` | `platform_submit_report` triggers graph update |
| `test_platform_query_graph_e2e` | Full tool call → graph load → traversal → response |
| `test_context_includes_graph` | ContextService renders GraphSection in CHATBOT mode |
| `test_context_excludes_graph_heartbeat` | GraphSection not rendered in HEARTBEAT mode |
| `test_graph_tools_in_action_registry` | All 5 tools registered and discoverable |
| `test_graph_html_export` | Build pipeline produces valid HTML file |
| `test_graph_diff_on_rebuild` | `graph_diff()` produces correct new/removed nodes on rebuild |
| `test_graph_diff_first_build` | First build (no history) skips diff gracefully |

### Frontend Tests

| Test | What It Validates |
|------|-------------------|
| `test_business_graph_tab_renders` | Tab visible in Knowledge Bases page, loads graph.json |
| `test_business_graph_node_click` | Click node → detail panel shows edges + source file |
| `test_business_graph_community_filter` | Click community in sidebar → view focuses on cluster |
| `test_business_graph_confidence_slider` | Moving slider hides/shows edges by confidence threshold |
| `test_business_graph_search` | Search highlights matching nodes and pans camera |
| `test_business_graph_empty_state` | No graph.json → shows "Build your knowledge graph" CTA |
| `test_graph_diff_banner` | Diff data exists → banner shows change summary |

---

## 19. Risks

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | LLM extraction produces inconsistent node IDs across documents | Normalize IDs via `_make_id()` (lowercase, non-alphanum → underscore). Dedup in merge step. |
| 2 | Graph grows unbounded as documents accumulate | Node limit per workspace (default 5000). Prune orphan nodes older than 90 days. |
| 3 | Graph rebuild blocks request handling | All rebuilds in background task via `run_in_executor`. Debounced to max 1 per minute. |
| 4 | Graphify library update breaks API | Pin version in requirements.txt. Wrap all graphify imports in `graph_service.py` — single point of change. |
| 5 | LLM hallucinated relationships pollute graph | INFERRED confidence (0.5) on all LLM-extracted edges. Graph queries can filter by confidence. Extraction prompt explicitly says "do not hallucinate." |
| 6 | Large workspace graph exceeds memory | NetworkX overhead is ~1KB per node. 5000 nodes = ~5MB. Well within container limits. Monitor via `/graph/meta.json` node count. |
| 7 | GraphSection adds latency to every chat message | Graph load from LRU cache is < 10ms. `_score_nodes` is O(n) string match, ~5ms for 5000 nodes. Total < 20ms. |
| 8 | Tree-sitter C extensions fail to build in container | Deferred code extraction to v2. V1 uses only LLM extraction + deterministic mapping. No tree-sitter dependency in v1. |
| 9 | Team-scoped graph leaks cross-team data via edges | Traversal treats team-blocked nodes as non-existent — BFS/DFS never visits them. Edge targets pointing to blocked nodes are pruned from results. Test with cross-team edge cases. |
| 10 | PRD-124 ships after PRD-126 Phase 1 — team_access not yet on documents | Phase 1 extraction checks for `team_access` column. If absent (PRD-124 not yet deployed), all nodes get `team_access = []` (public). No breakage — team filtering is a no-op until PRD-124 lands. |

---

## 20. Success Criteria

PRD-126 is done when:

- [ ] `graphifyy[leiden]` installed in backend container
- [ ] `GraphifyService` builds graph from workspace documents, reports, roster, blueprints, and DB schemas
- [ ] Incremental updates process only changed sources (SHA256 cache verified)
- [ ] 5 platform tools registered, routable, and functional
- [ ] `platform_query_graph` returns relevant subgraph within token budget
- [ ] `platform_graph_impact` traces multi-hop dependencies correctly
- [ ] Post-ingest hooks trigger graph updates automatically (documents + reports)
- [ ] ContextService injects relevant graph context in CHATBOT and TASK_EXECUTION modes
- [ ] Interactive HTML visualization generated as fallback (workspace files)
- [ ] **Business Graph tab** renders in Knowledge Bases page with D3 force-directed layout
- [ ] Community coloring, god node highlighting, confidence slider, search, and node detail panel working
- [ ] `graph_diff()` runs on every rebuild, diff saved to `/graph/latest_diff.json`
- [ ] Diff banner displays in Business Graph tab when changes detected
- [ ] Hyperedges extracted and stored in graph.json
- [ ] AUTO can answer "how does X connect to Y?" using graph tools
- [ ] Graph rebuild completes in < 60s for a 50-document workspace
- [ ] No new database migrations required
- [ ] Unit + integration tests passing with 80%+ coverage on new code

---

## 21. Future — v2 and Beyond

| Version | Feature | Unlocks |
|---------|---------|---------|
| ~~v2~~ **v1** | ~~HARNESS `graph_diff()`~~ **Shipped in Phase 5** | Diff runs on every build. UI shows change banner. HARNESS can read `/graph/latest_diff.json` directly. |
| v2 | Mission Zero v2 — derive agent roster from graph communities | Agent-to-domain assignment backed by data, not LLM guessing |
| v2 | Graph-aware mission decomposition | MissionPlanner uses graph to identify task dependencies and required agents |
| v2 | Code repo indexing via tree-sitter | Combine code structure with business concepts in one graph |
| v2 | Shopify product catalog auto-sync | Real-time product/collection/order nodes via Composio |
| v2 | Image/diagram extraction | Whiteboard photos, org charts, process diagrams → graph nodes |
| v3 | Cross-workspace pattern sharing | "Workspaces running Shopify stores typically have these communities..." |
| v3 | Graph-powered agent recommendations | "Your graph shows a Finance cluster with no assigned agent — create LEDGER?" |
| v3 | Neural Field bridge | Graph topology informs semantic field structure (PRD-100 Phase 3) |
