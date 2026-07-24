# Automatos — Architecture Diagrams

> Mermaid diagrams for the brain. Companion to `BRAIN-BLUEPRINT.md` (section refs below
> point there). Render in any Mermaid-aware viewer (GitHub, VS Code, Obsidian).
> **Status:** baseline — 2026-05-29, verified against branch `feat/widget-page-context-on-regular-chat`.

---

## 1. System context (who talks to Automatos)

```mermaid
flowchart TB
    user["End user / merchant"]
    store["Storefront visitor (widget)"]
    chans["Channel users (Slack, Telegram, ...)"]

    subgraph automatos["Automatos Platform"]
        fe["Next.js frontend"]
        api["FastAPI orchestrator"]
        workers["Background workers + scheduler"]
    end

    llm["LLM providers (BYOK / platform / OpenRouter)"]
    composio["Composio (100+ external APIs)"]
    shopify["Shopify (catalog + orders)"]
    mem0["Mem0 (long-term memory)"]
    stores["Postgres / S3 / S3 Vectors / Qdrant / Redis"]

    user --> fe --> api
    store -->|"widget SDK"| api
    chans -->|"adapter webhook/poll"| api
    api --> workers
    api --> llm
    api --> composio
    api --> shopify
    api --> mem0
    api --> stores
    workers --> stores
```

---

## 2. Layered container view (dependency direction is downward)

```mermaid
flowchart TB
    subgraph L7["L7 · Ops"]
        obs["Monitoring: Prometheus/Loki/Grafana"]
        voice["Voice service + pipeline"]
    end
    subgraph L6["L6 · SaaS surface"]
        mkt["Marketplace"]
        onb["Onboarding wizard (VOYAGER/BLUEPRINT/SCRIBE/FORGE)"]
        fec["Frontend (api-client.ts, hooks)"]
    end
    subgraph L5["L5 · Reach"]
        chadapt["11 channel adapters"]
        widg["Widget API"]
        plug["Vertical plugins (generic, shopify)"]
        comp["Composio gateway"]
    end
    subgraph L4["L4 · Orchestration"]
        coord["CoordinatorService (missions)"]
        pb["Playbook/recipe executor"]
    end
    subgraph L3["L3 · Knowledge"]
        mem["Memory stack (L1/L2/L3)"]
        rag["RAG"]
        kg["Knowledge graph"]
        nl["NL2SQL"]
    end
    subgraph L2["L2 · Cognitive core"]
        router["Universal Router"]
        af["AgentFactory"]
        ute["UnifiedToolExecutor"]
        ar["ActionRegistry"]
    end
    subgraph L1["L1 · Core services"]
        cfg["config.py"]
        auth["auth / RequestContext"]
        db["DB sessions"]
        llmm["LLM manager"]
    end
    subgraph L0["L0 · Substrate"]
        pg["Postgres"]
        s3["S3 + S3 Vectors"]
        qd["Qdrant"]
        rd["Redis"]
        m0["Mem0"]
    end

    L6 --> L5 --> L4 --> L3 --> L2 --> L1 --> L0
    L7 -.observes.-> L1
```

---

## 3. Sequence — one chat turn (the Arc, BRAIN §1)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant API as api/chat.py
    participant RC as RequestContext
    participant Auto as AutoBrain
    participant R as UniversalRouter
    participant AF as AgentFactory
    participant TL as Tool loop
    participant UTE as UnifiedToolExecutor
    participant Mem as Memory
    U->>API: POST /api/chat (message)
    API->>RC: inject (workspace_id, user, auth)
    API->>Mem: load history + context bundle
    API->>Auto: assess(message)
    Auto-->>API: verdict {complexity, RESPOND|DELEGATE|MISSION}
    alt DELEGATE or MISSION
        API->>R: route(envelope)
        R-->>API: agent_id (tier 0..3)
    end
    API->>AF: activate_agent(agent_id, workspace_id)
    AF-->>API: AgentRuntime (LLM key, tools, executor)
    loop until no more tool calls
        API->>TL: generate_response(tools)
        TL->>UTE: execute_tool(name, args)
        UTE-->>TL: result (prefix-dispatched)
    end
    API->>Mem: store_transcript() + graph update
    API-->>U: stream final text (SSE)
```

---

## 4. Sequence — mission lifecycle (BRAIN §3.6; DB-authoritative, restart-durable)

```mermaid
sequenceDiagram
    autonumber
    participant Caller as chat/api
    participant Co as CoordinatorService (5s tick)
    participant Pl as MissionPlanner
    participant Di as MissionDispatcher
    participant AF as AgentFactory
    participant Re as MissionReconciler
    participant DB as Postgres
    Caller->>Co: create_mission(goal)
    Co->>DB: OrchestrationRun = PENDING
    Co->>Pl: decompose(goal)
    Pl->>DB: tasks + dependency rows + BoardTasks
    Co->>DB: run = AWAITING_APPROVAL
    Caller->>Co: approve_plan()
    Co->>DB: run = RUNNING; ready tasks = QUEUED
    loop every 5s while RUNNING (re-read from DB)
        Co->>Di: dispatch_ready()
        Di->>AF: execute_with_prompt(task)
        AF-->>Di: output
        Di->>DB: record_task_completion (or re-queue)
        Co->>Re: reconcile()
        Re->>DB: COMPLETED -> VERIFIED; re-dispatch stalls
    end
    Co->>DB: run = VERIFYING -> AWAITING_HUMAN
    Caller->>Co: review_mission(accept)
    Co->>DB: run = COMPLETED; deliverable saved (.md -> S3)
```

---

## 5. Sequence — widget message with vertical plugin dispatch (BRAIN §4.7, §7)

```mermaid
sequenceDiagram
    autonumber
    participant SDK as Storefront SDK
    participant W as api/widgets/chat.py
    participant Auth as widget_auth
    participant Reg as PLUGIN_REGISTRY
    participant P as Plugin (generic|shopify)
    participant KG as Knowledge graph
    participant CS as ChatService
    SDK->>W: POST /api/widgets/chat (message, page_context, trigger_reason)
    W->>Auth: resolve API key/JWT
    Auth-->>W: WidgetAuthContext (workspace_id, perms)
    W->>W: vertical = workspace.settings.vertical
    W->>Reg: lookup(vertical)
    Reg-->>W: plugin
    W->>P: handle_widget_message(...)
    opt shopify
        P->>KG: FBT / collection / vendor edges
        KG-->>P: related products
    end
    P-->>W: WidgetPluginResult (message, system_preamble?)
    W->>CS: run agent (generic core)
    CS-->>SDK: SSE stream (text + tool events)
```

Note: generic core (`chat.py`) holds **zero** Shopify identifiers — CI gate
`check-no-shopify-in-generic.sh` enforces it.

---

## 6. Flow — memory write & read (BRAIN §4.2)

```mermaid
flowchart LR
    subgraph write["Write (after turn)"]
        t["transcript"] --> l2w["L2 Postgres memory_short_term (verbatim)"]
        t --> l3w["L3 Mem0 add() (fact extraction)"]
        t --> l1w["L1 Redis session summary"]
        l2w -.hourly.-> decay["Ebbinghaus decay + consolidation"]
    end
    subgraph read["Read (before turn)"]
        q["query"] --> cr["ContextRouter analyse"]
        cr --> l1r["L1 session"]
        cr --> l2r["L2 ILIKE/time search"]
        cr --> l3r["L3 Mem0 search (5min Redis cache)"]
        l1r --> bun["budget-capped ContextBundle"]
        l2r --> bun
        l3r --> bun
        bun --> sec["MemorySection -> system prompt"]
    end
```

---

## 7. Flow — RAG ingest & retrieve (BRAIN §3.3)

```mermaid
flowchart LR
    subgraph ingest["Ingest"]
        up["upload_document"] --> row["Postgres documents row"]
        row --> s3u["S3 raw upload"]
        s3u --> chunk["semantic chunk"]
        chunk --> emb["embed (2048-dim)"]
        emb --> idx["S3 Vectors add_documents (external_file_id = documents.id)"]
    end
    subgraph retrieve["Retrieve (agent tool)"]
        query["query"] --> enh["enhance (HyDE / decomp)"]
        enh --> cand["embed + S3 Vectors search"]
        cand --> rrf["RRF + rerank + parent-expand + knapsack"]
        rrf --> ctx["context into prompt"]
    end
```

---

## 8. Data substrate map (BRAIN §5)

```mermaid
flowchart TB
    subgraph pg["Postgres — system of record (116 tables)"]
        d1["agents / personas / blueprints"]
        d2["workspaces / members / users"]
        d3["orchestration_* / board_tasks"]
        d4["workflow_templates / recipe_executions"]
        d5["memory_short_term / knowledge_*"]
        d6["documents / document_chunks"]
        d7["tools / composio_* / routing_*"]
        d8["marketplace_* / skills / llm_models"]
        d9["channel_connections / widget_* / sites"]
        d10["credentials / audit_logs / sdk_api_keys"]
    end
    s3["S3 — markdown deliverables, raw docs, marketplace bundles, harness changelog"]
    s3v["S3 Vectors — documents-index (RAG)"]
    qd["Qdrant — field_memory (mission-scoped)"]
    rd["Redis — session, cache, queue, rate-limit, pub/sub"]
    m0["Mem0 — long-term facts (external service)"]
```

---

## 9. State machine — OrchestrationTask (BRAIN §3.6)

```mermaid
stateDiagram-v2
    [*] --> PENDING
    PENDING --> QUEUED: dependencies met
    QUEUED --> ASSIGNED: dispatcher picks agent
    ASSIGNED --> RUNNING: agent starts
    RUNNING --> COMPLETED: output recorded
    RUNNING --> QUEUED: error (re-queue)
    ASSIGNED --> QUEUED: stall > 60s (re-dispatch)
    RUNNING --> QUEUED: stall > 300s (re-dispatch)
    COMPLETED --> VERIFIED: reconciler (advisory verify)
    VERIFIED --> [*]
```

> Gap (BRAIN §8, G5/G10): re-queue on error stores only `failure_detail` — verifier
> critique is **not** fed back into the retry. Verification is advisory-only (PRD-103);
> the reconciler docstring still claims a verdict path — doc/code drift to fix.
