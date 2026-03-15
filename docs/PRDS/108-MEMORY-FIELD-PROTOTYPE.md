# PRD-108 — Memory Field Prototype

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P1
**Dependencies:** PRD-107 (Context Interface Abstraction), PRD-100 (Master Research)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-15

---

## 1. Problem Statement

### 1.1 The Hypothesis

PRD-100 Risk #6: *"Context Engineering theory doesn't translate to code — PRD-108 is the prototype gate. If field prototype doesn't outperform message passing, reassess Phase 3."*

**Hypothesis:** Agents sharing a continuous semantic field — where information resonates, decays, and forms attractors — produce higher-quality collaborative output than agents passing discrete messages.

### 1.2 The Telephone Game Problem

```
Message-Passing (Today):
Agent A → "Here are my research findings: ..." → Agent B
Agent B → "Based on your findings, I conclude..." → Agent C
Agent C sees A's work through B's interpretation.
If A's finding #7 is relevant to C but B didn't mention it → lost.

Shared Field (Proposed):
┌──────────── SHARED FIELD ────────────┐
│  Agent A injects 20 findings         │
│  Agent B injects 15 analyses         │
│  Finding #7 resonates with Analysis #3│
│    → both amplified automatically    │
│  Finding #12 unreferenced            │
│    → decays naturally over time      │
│  Agent C queries field               │
│    → sees amplified #7+#3 first      │
│    → #12 still retrievable but faded │
└──────────────────────────────────────┘
```

No telephone game. Agent C accesses the full field, with relevance surfaced by resonance rather than filtered by intermediate agents.

### 1.3 What This PRD Delivers

A controlled experiment: same task, same agents, same models — one run with message-passing (PRD-107 `RedisSharedContext`), one with a shared vector field (`VectorFieldSharedContext`). Measured comparison on context quality, task accuracy, token efficiency, and latency. Results determine whether Phase 3 (PRDs 110-116) proceeds.

---

## 2. Prior Art Analysis

### 2.1 Vector Store Backend Evaluation

| Backend | Key Strength | Key Weakness | Verdict |
|---------|-------------|-------------|---------|
| **Qdrant** | Native payload filtering (datetime, numeric, keyword); Recommendations API = resonance discovery; Docker single-command deploy; `:memory:` mode for tests | No built-in TTL on points; docs sparse for recommend API | **PRIMARY — deploy as Railway Docker service** |
| **FAISS** | Fastest CPU search at small scale; trivial persistence (`write_index`/`read_index`); zero infrastructure | No metadata filtering (external table needed); no thread-safe writes; no payload storage | **BENCHMARK ONLY — in-process comparison baseline** |
| **Redis Vector Search** | Would have been zero-infra (Redis already deployed); native TTL = automatic decay | **NOT AVAILABLE** — Railway deploys vanilla Redis, not Redis Stack. No RediSearch, no RedisJSON modules | **ELIMINATED** |
| **Pinecone** | Managed service, serverless | External dependency; network latency; cost per query | **REJECTED — adds dependency for a self-contained prototype** |
| **S3 Vectors** (existing) | Already configured at `automatos-vector-index`, 2048-dim cosine | Document-oriented (designed for RAG, not live fields); no TTL; no real-time metadata queries | **WRONG ABSTRACTION — keep for documents, not fields** |

### 2.2 Temporal Decay Research

| Approach | Source | What We Adopt | What We Reject |
|----------|--------|---------------|----------------|
| **Exponential decay** `S(t) = S₀ × e^(-λt)` | Ebbinghaus (1885), standard IR | Core decay formula with λ=0.1 (7h half-life). Already implemented at `memory_types.py:65` | — (adopted as-is) |
| **Elasticsearch decay functions** | ES `function_score` | Score-time application (no deletion) — patterns persist, decay computed at query time | Three decay profiles (linear, exp, gauss) — exponential is sufficient for prototype |
| **LRU/LFU/ARC cache eviction** | Standard CS literature | Access count as reinforcement signal — frequently accessed patterns resist decay | ARC's adaptive tuning — over-complex for prototype |
| **Hebbian reinforcement** | Hebb (1949) | "Neurons that fire together wire together" — co-accessed patterns boost each other. +5% per access, cap at 2× | Continuous Hebbian learning rates — fixed increment is simpler |
| **Spaced repetition** | Kornell & Bjork (2008) | Re-access resets decay clock (uses `last_accessed`, not `created_at`) | SRS scheduling algorithm — agents access on-demand, not on schedule |

### 2.3 Context Engineering Theory (Chapters 08-11)

| Concept | What We Adopt for Prototype | What's Deferred to Phase 3 |
|---------|---------------------------|---------------------------|
| **8 core operations** (Ch. 08) | 5 of 8: inject, query (≈resonate), decay, reinforce (≈amplify), measure_stability | 3 deferred: attenuate, tune, collapse |
| **Boundary permeability** (Ch. 08) | `effective_strength = strength × boundary_permeability` — configurable per field | Dynamic boundary adjustment — fixed permeability per field in prototype |
| **Resonance formula** (Ch. 09) | `R(A,B) = cos(θ)² × |A| × |B|` — squared cosine amplifies high-similarity pairs | Learned semantic relatedness function — cosine is the prototype baseline |
| **Attractor protection** (Ch. 09) | `effective_decay = decay_rate × (1 - attractor_protection)` where protection = Σ(resonance × 0.5), cap 0.5 | Full attractor dynamics (formation, classification, basin mapping) |
| **Multi-field operations** (Ch. 10) | Not in prototype — single field per mission | Superposition, interference, coupling — Phase 3 (PRD-110) |
| **Attractor detection** (Ch. 11) | Simple stability metric: `avg_strength × 0.6 + organization × 0.4` | Gradient convergence, bifurcation detection — Phase 3 (PRD-112) |

### 2.4 Existing Infrastructure Reuse

| Component | Location | How We Reuse It |
|-----------|----------|-----------------|
| `EmbeddingManager` | `core/llm/embedding_manager.py` | `generate_embeddings_batch()` with qwen3-embedding-8b (2048-dim) via OpenRouter. Zero new embedding infrastructure |
| `SharedContextManager` | `inter_agent.py:400-649` | The Phase 2 baseline. Wrapped by `RedisSharedContext` (PRD-107) as the control condition |
| `MEMORY_DECAY_RATE` | `config.py` | λ=0.1 — same decay rate for field patterns. Consistent with L2 memory behavior |
| `MEMORY_DECAY_ARCHIVE_THRESHOLD` | `config.py` | 0.3 — filter threshold for decayed patterns. We use 0.05 (stricter) for field queries |
| `MemoryNamespace` pattern | `unified_memory_service.py:39-117` | Extend: `mem:{workspace_id}:field:{field_id}` for field-scoped Qdrant collections |
| `ContextProvider` / `SharedContextPort` | PRD-107 `core/ports/context.py` | The interface the field adapter implements. Validates PRD-107's design |

---

## 3. Architecture

### 3.1 Qdrant Deployment

```
Railway Project
├── automatos-ai-api (existing)
├── automotas-ai-frontend (existing)
├── agent-workspace-worker (existing)
├── redis (existing — vanilla, no Vector Search)
├── postgres (existing)
└── qdrant (NEW — Docker service)
    Image: qdrant/qdrant:latest
    Port: 6333 (HTTP), 6334 (gRPC)
    Volume: /qdrant/storage (persistent)
    Env: QDRANT__SERVICE__GRPC_PORT=6334
    Internal URL: qdrant.railway.internal:6333
```

Config extension:

```python
# In config.py
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # Optional for Railway internal
FIELD_EMBEDDING_DIM = 2048  # Match qwen3-embedding-8b
FIELD_DECAY_RATE = float(os.getenv("FIELD_DECAY_RATE", "0.1"))  # λ
FIELD_REINFORCE_BONUS = float(os.getenv("FIELD_REINFORCE_BONUS", "0.05"))  # +5% per access
FIELD_REINFORCE_CAP = float(os.getenv("FIELD_REINFORCE_CAP", "2.0"))  # Max 2× original
FIELD_ARCHIVAL_THRESHOLD = float(os.getenv("FIELD_ARCHIVAL_THRESHOLD", "0.05"))
FIELD_BOUNDARY_PERMEABILITY = float(os.getenv("FIELD_BOUNDARY_PERMEABILITY", "1.0"))
```

### 3.2 Field Data Model

Each mission field is a Qdrant collection. Each pattern is a point:

```python
# Qdrant collection per field
collection_name = f"field_{context_id}"

# Point schema
{
    "id": "uuid-string",           # Qdrant point ID
    "vector": [float × 2048],      # qwen3-embedding-8b output
    "payload": {
        "agent_id": 12,            # Who injected this pattern
        "key": "finding_3",         # Human-readable label
        "value": "EU AI Act Article 6 requires...",  # Content (for retrieval)
        "strength": 0.85,           # Initial injection strength (after boundary permeability)
        "created_at": "2026-03-15T10:23:45Z",
        "last_accessed": "2026-03-15T10:23:45Z",
        "access_count": 0,          # Hebbian reinforcement counter
        "content_hash": "sha256...", # Dedup key
    }
}
```

### 3.3 System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    COORDINATOR (PRD-102)                     │
│  create_context() → field_id                                │
│  assign_task(agent, context_id=field_id)                    │
│  query(field_id, "mission summary") → assess completion     │
│  destroy_context(field_id) → cleanup                        │
└──────────┬──────────────────┬──────────────────┬────────────┘
           │                  │                  │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Agent A    │   │  Agent B    │   │  Agent C    │
    │ Researcher  │   │  Analyst    │   │   Writer    │
    │             │   │             │   │             │
    │ inject(20   │   │ query() →   │   │ query() →   │
    │  findings)  │   │ read A's    │   │ resonant    │
    │             │   │ inject(15   │   │ patterns    │
    │             │   │  analyses)  │   │             │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                  │                  │
    ┌──────▼──────────────────▼──────────────────▼────────────┐
    │            VectorFieldSharedContext                      │
    │            (implements SharedContextPort)                │
    │                                                         │
    │  inject() → embed → upsert to Qdrant                   │
    │  query()  → embed → search Qdrant → decay score → rank │
    │  reinforce() → increment access_count + boost strength  │
    ├─────────────────────────────────────────────────────────┤
    │                      Qdrant                             │
    │  Collection: field_{context_id}                         │
    │  Vectors: 2048-dim cosine similarity                    │
    │  Payload: agent_id, strength, timestamps, access_count  │
    └─────────────────────────────────────────────────────────┘
```

---

## 4. Core Operations

### 4.1 Operation 1: `inject(pattern, strength)`

Add an embedding to the shared field with metadata.

```python
async def inject(self, context_id: str, key: str, value: str,
                 agent_id: int, strength: float = 1.0) -> None:
    """Inject a pattern into the shared field.

    1. Generate embedding from key+value
    2. Apply boundary permeability
    3. Deduplicate by content hash
    4. Upsert to Qdrant
    """
    content = f"{key}: {value}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Dedup: if same content already in field, reinforce instead of duplicating
    existing = await self._find_by_hash(context_id, content_hash)
    if existing:
        await self._reinforce(context_id, existing.id)
        return

    embedding = await self._embedder.generate_embedding(content)
    effective_strength = strength * self._boundary_permeability

    point_id = str(uuid.uuid4())
    await self._client.upsert(
        collection_name=f"field_{context_id}",
        points=[PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "agent_id": agent_id,
                "key": key,
                "value": value,
                "strength": effective_strength,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "access_count": 0,
                "content_hash": content_hash,
            },
        )],
    )
```

### 4.2 Operation 2: `query(embedding, top_k)`

Retrieve resonant patterns by cosine similarity, weighted by decay + reinforcement.

```python
async def query(self, context_id: str, query: str,
                agent_id: int, top_k: int = 10) -> list[dict]:
    """Query the field for resonant patterns.

    1. Embed the query
    2. Search Qdrant for similar vectors (over-fetch for post-decay filtering)
    3. Apply temporal decay scoring
    4. Filter by archival threshold
    5. Reinforce accessed patterns (Hebbian)
    6. Return ranked results
    """
    query_embedding = await self._embedder.generate_embedding(query)

    raw_results = await self._client.search(
        collection_name=f"field_{context_id}",
        query_vector=query_embedding,
        limit=top_k * 3,  # Over-fetch for post-decay filtering
    )

    # Apply decay + reinforcement scoring
    now = datetime.utcnow()
    scored = []
    for hit in raw_results:
        payload = hit.payload
        age_hours = (now - datetime.fromisoformat(payload["last_accessed"])).total_seconds() / 3600
        decayed_strength = self._compute_decayed_strength(
            initial_strength=payload["strength"],
            age_hours=age_hours,
            access_count=payload["access_count"],
        )

        # Filter out archived patterns
        if decayed_strength < self._archival_threshold:
            continue

        # Resonance score = cosine_similarity² × decayed_strength
        resonance = (hit.score ** 2) * decayed_strength

        scored.append({
            "id": hit.id,
            "key": payload["key"],
            "value": payload["value"],
            "score": resonance,
            "agent_id": payload["agent_id"],
            "decayed_strength": decayed_strength,
            "cosine_similarity": hit.score,
        })

    # Sort by resonance score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored[:top_k]

    # Hebbian reinforcement: accessed patterns get a boost
    accessed_ids = [r["id"] for r in top_results]
    if accessed_ids:
        await self._reinforce_batch(context_id, accessed_ids)

    return top_results
```

### 4.3 Operation 3: `decay()` — Score-Time

Decay is NOT a periodic job. It's computed at query time within `_compute_decayed_strength()`:

```python
def _compute_decayed_strength(
    self,
    initial_strength: float,
    age_hours: float,
    access_count: int,
) -> float:
    """S(t) = S₀ × e^(-λt) × access_boost

    λ = 0.1 → half-life ≈ 6.93 hours
    access_boost = 1 + (access_count × 0.05), capped at 2.0
    """
    decay = math.exp(-self._decay_rate * age_hours)
    access_boost = min(1.0 + (access_count * self._reinforce_bonus), self._reinforce_cap)
    return initial_strength * decay * access_boost
```

**Decay calibration:**

| λ Value | Half-Life | Use Case |
|---------|-----------|----------|
| 0.05 | ~14 hours | Long-running missions (multi-day research) |
| **0.1** | **~7 hours** | **Default — standard mission duration** |
| 0.2 | ~3.5 hours | Fast-turnaround tasks |

Start with λ=0.1. Run sensitivity analysis across {0.05, 0.1, 0.2} during the experiment.

### 4.4 Operation 4: `reinforce(pattern_id)` — Hebbian

When a pattern is accessed via `query()`:

```python
async def _reinforce_batch(self, context_id: str, point_ids: list[str]) -> None:
    """Hebbian reinforcement: accessed patterns resist decay.

    1. Increment access_count
    2. Update last_accessed timestamp (resets decay clock)
    3. Co-access bonus: patterns accessed together get mutual strength boost
    """
    collection = f"field_{context_id}"
    now = datetime.utcnow().isoformat()

    for pid in point_ids:
        # Get current payload
        points = await self._client.retrieve(collection, ids=[pid])
        if not points:
            continue
        payload = points[0].payload

        # Update access metadata
        new_count = payload["access_count"] + 1
        await self._client.set_payload(
            collection_name=collection,
            payload={
                "access_count": new_count,
                "last_accessed": now,
            },
            points=[pid],
        )

    # Co-access bonus: if multiple patterns accessed together, boost all
    if len(point_ids) > 1:
        for pid in point_ids:
            points = await self._client.retrieve(collection, ids=[pid])
            if not points:
                continue
            current_strength = points[0].payload["strength"]
            boosted = min(current_strength * 1.02, current_strength * self._reinforce_cap / points[0].payload.get("strength", 1.0))
            await self._client.set_payload(
                collection_name=collection,
                payload={"strength": boosted},
                points=[pid],
            )
```

### 4.5 Operation 5: `measure_stability()`

Quantify field convergence — used for telemetry and experiment analysis.

```python
async def measure_stability(self, context_id: str) -> dict:
    """Measure field stability — how converged the field is.

    stability = (avg_strength × 0.6) + (organization × 0.4)

    organization = 1 - (stddev_strength / mean_strength) if mean > 0 else 0
    """
    collection = f"field_{context_id}"

    # Get all points
    points, _ = await self._client.scroll(collection, limit=10000)
    if not points:
        return {"stability": 0.0, "pattern_count": 0, "avg_strength": 0.0}

    now = datetime.utcnow()
    strengths = []
    for p in points:
        age_hours = (now - datetime.fromisoformat(p.payload["last_accessed"])).total_seconds() / 3600
        ds = self._compute_decayed_strength(
            p.payload["strength"], age_hours, p.payload["access_count"]
        )
        strengths.append(ds)

    avg_strength = sum(strengths) / len(strengths)
    if avg_strength > 0:
        stddev = (sum((s - avg_strength) ** 2 for s in strengths) / len(strengths)) ** 0.5
        organization = max(0.0, 1.0 - (stddev / avg_strength))
    else:
        organization = 0.0

    stability = (avg_strength * 0.6) + (organization * 0.4)

    return {
        "stability": round(stability, 4),
        "pattern_count": len(points),
        "avg_strength": round(avg_strength, 4),
        "organization": round(organization, 4),
        "active_patterns": sum(1 for s in strengths if s >= self._archival_threshold),
        "decayed_patterns": sum(1 for s in strengths if s < self._archival_threshold),
    }
```

---

## 5. `VectorFieldSharedContext` — Full Adapter

Implements PRD-107's `SharedContextPort` interface:

```python
# orchestrator/modules/context/adapters/vector_field.py

import hashlib
import math
import uuid
from datetime import datetime
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from core.ports.context import SharedContextPort
from core.llm.embedding_manager import EmbeddingManager
from config import config


class VectorFieldSharedContext(SharedContextPort):
    """PRD-108 prototype: vector field backend for SharedContextPort.

    Creates a Qdrant collection per mission field. Patterns are points
    with 2048-dim embeddings and metadata payloads. Resonance is
    computed as cosine_similarity² × decayed_strength at query time.
    """

    def __init__(self):
        self._client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        self._embedder = EmbeddingManager()
        self._decay_rate = config.FIELD_DECAY_RATE
        self._reinforce_bonus = config.FIELD_REINFORCE_BONUS
        self._reinforce_cap = config.FIELD_REINFORCE_CAP
        self._archival_threshold = config.FIELD_ARCHIVAL_THRESHOLD
        self._boundary_permeability = config.FIELD_BOUNDARY_PERMEABILITY
        self._dimension = config.FIELD_EMBEDDING_DIM

    async def create_context(self, team_agent_ids: list[int],
                             initial_data: Optional[dict] = None) -> str:
        field_id = str(uuid.uuid4())
        collection_name = f"field_{field_id}"

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self._dimension,
                distance=Distance.COSINE,
            ),
        )

        # Inject initial data if provided
        if initial_data:
            for key, value in initial_data.items():
                await self.inject(field_id, key, str(value),
                                  agent_id=0, strength=1.0)

        return field_id

    async def inject(self, context_id, key, value, agent_id, strength=1.0):
        # ... (Section 4.1 implementation)
        ...

    async def query(self, context_id, query, agent_id, top_k=10):
        # ... (Section 4.2 implementation)
        ...

    async def destroy_context(self, context_id: str) -> None:
        collection_name = f"field_{context_id}"
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass  # Collection may already be cleaned up
```

---

## 6. Experiment Design

### 6.1 Task Selection

The experiment task must be:
- **Multi-agent** — requires at least 2 agents to collaborate
- **Context-dependent** — later agents benefit from earlier agents' full context
- **Measurable** — output quality can be objectively scored
- **Repeatable** — same task can run multiple times for statistical significance

**Task: "Research a topic and produce an analysis report"**

| Role | Agent | Actions |
|------|-------|---------|
| Researcher | Agent A | Web search + document analysis → inject findings into shared context |
| Analyst | Agent B | Query context for findings → produce structured analysis → inject analysis |
| Writer | Agent C | Query context for resonant patterns → produce final report |

### 6.2 Topic Selection (5 Topics)

| # | Topic | Why This Topic |
|---|-------|---------------|
| 1 | EU AI Act compliance requirements for SaaS platforms | Complex regulation, multi-faceted, requires synthesis |
| 2 | Comparison of vector database architectures for production ML | Technical depth, multiple dimensions to compare |
| 3 | Impact of remote work policies on software team productivity | Mix of qualitative and quantitative data |
| 4 | State of AI agent frameworks: LangGraph vs CrewAI vs AutoGen | Directly relevant, verifiable claims |
| 5 | Carbon footprint reduction strategies for cloud infrastructure | Cross-domain (engineering + sustainability) |

### 6.3 Experimental Conditions

| Variable | Control (Message-Passing) | Treatment (Shared Field) |
|----------|--------------------------|-------------------------|
| **Context mechanism** | `RedisSharedContext` (PRD-107) — key-value dict, no semantic ranking | `VectorFieldSharedContext` — Qdrant vectors, resonance scoring, decay |
| **LLM model** | Sonnet 4.6 for all roles | Sonnet 4.6 for all roles |
| **Task description** | Identical per topic | Identical per topic |
| **Tool access** | Same per role | Same per role |
| **Token budget** | Same per role | Same per role |
| **Agent system prompts** | Same per role | Same per role |

### 6.4 Metrics

| Metric | How Measured | Primary/Secondary |
|--------|-------------|-------------------|
| **Information Retention** | Count of Agent A's findings that appear in final output (manual + LLM-assisted) | **Primary** — core hypothesis test |
| **Context Quality** | Blind human eval (1-5 Likert): "Does the report reflect all relevant source findings?" | **Primary** |
| **Task Accuracy** | LLM-as-judge (PRD-103 rubric-scored) against reference answer | **Primary** |
| **Token Efficiency** | Total tokens consumed across all agents (from `llm_usage`) | **Secondary** |
| **Latency** | Wall-clock time from mission start to final output | **Secondary** |
| **Cross-Agent Resonance** | Count of field patterns accessed by >1 agent | **Secondary** (field condition only) |
| **Field Stability** | Convergence score at mission end via `measure_stability()` | **Secondary** (field condition only) |
| **Embedding Cost** | Number of `inject()` + `query()` API calls × embedding cost | **Secondary** |

### 6.5 Success Criteria (Phase 3 Gate)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Information Retention** | Field retains ≥20% more of Agent A's findings | Core hypothesis: field fixes the telephone game |
| **Context Quality** | Human eval ≥0.5 points higher (on 5-point scale) | Perceptible quality improvement |
| **Token Efficiency** | Field uses ≤120% of message-passing tokens | Small overhead acceptable; >20% overhead = too expensive |
| **Latency** | Field completes in ≤150% of message-passing time | Embedding overhead must be bounded |

**Decision rules:**
- **ALL four pass:** Phase 3 validated. Proceed to PRDs 110-116.
- **Information retention fails:** Core hypothesis is wrong. Reassess Phase 3 entirely.
- **Only token/latency fail:** Field works but costs too much. Optimize embeddings before proceeding.
- **Only quality fails:** Field preserves more context but doesn't improve output. Investigate prompt engineering.

### 6.6 Statistical Methodology

- **5 topics × 2 conditions × 3 repetitions = 30 total runs**
- **Paired comparison:** Same topic in both conditions reduces topic-variance noise
- **Wilcoxon signed-rank test** for non-parametric paired comparison (small sample)
- **Blind evaluation:** Human raters don't know which condition produced which output
- **LLM-as-judge as secondary metric** (PRD-103 rubric format) — cross-validated against human eval

---

## 7. Resonance Scoring

### 7.1 Core Formula

From Context Engineering Chapter 09:

```
R(A, B) = cos(θ)² × |A| × |B|
```

Where:
- `cos(θ)` = cosine similarity between embedding vectors A and B
- `|A|`, `|B|` = decayed strength values of patterns A and B

**Why squared cosine:** Amplifies high-similarity pairs and suppresses noise.

| Cosine Similarity | Resonance Factor (cos²) | Effect |
|-------------------|------------------------|--------|
| 0.95 | 0.90 | Strong resonance — these patterns amplify each other |
| 0.80 | 0.64 | Moderate resonance |
| 0.60 | 0.36 | Weak resonance — barely above noise |
| 0.40 | 0.16 | Negligible — effectively filtered |

### 7.2 Query-Time Resonance Scoring

For a query Q against field pattern P:

```python
resonance_score = cosine_similarity(Q, P)² × decayed_strength(P)
```

This is what `query()` uses to rank results. The resonance formula between two field patterns (for attractor detection) is:

```python
mutual_resonance = cosine_similarity(P1, P2)² × decayed_strength(P1) × decayed_strength(P2)
```

### 7.3 Anti-Domination Safeguard

One strong pattern could dominate the field, drowning out everything else.

**Mitigation:** Cap resonance amplification:
- Maximum strength after reinforcement: `initial_strength × reinforce_cap` (default 2.0×)
- Co-access bonus capped at +2% per co-access event
- Monitor `max_strength / min_strength` ratio per field — if >10×, log a warning

---

## 8. Telemetry & Experiment Data

### 8.1 Per-Experiment Telemetry

Every experiment run produces structured data via PRD-106 `mission_events`:

| Event Type | Data Captured |
|------------|--------------|
| `field.created` | `{field_id, team_size, initial_data_count}` |
| `field.injected` | `{field_id, agent_id, key, strength, content_hash}` |
| `field.queried` | `{field_id, agent_id, query_preview, results_count, top_score}` |
| `field.reinforced` | `{field_id, pattern_ids, access_counts}` |
| `field.stability` | `{field_id, stability, pattern_count, active, decayed}` |
| `field.destroyed` | `{field_id, final_pattern_count, total_queries, total_injects}` |

### 8.2 Experiment Results Table

```sql
CREATE TABLE experiment_results (
    id          SERIAL PRIMARY KEY,
    experiment  VARCHAR(50) NOT NULL,    -- 'prd108_field_v1'
    topic       VARCHAR(255) NOT NULL,
    condition   VARCHAR(20) NOT NULL,    -- 'message_passing' | 'shared_field'
    run_number  SMALLINT NOT NULL,       -- 1, 2, 3

    -- Primary metrics
    info_retention_count  INTEGER,       -- Findings from Agent A in final output
    info_retention_total  INTEGER,       -- Total findings from Agent A
    context_quality_score REAL,          -- Human eval 1-5
    task_accuracy_score   REAL,          -- LLM-as-judge 0-1

    -- Secondary metrics
    total_tokens          INTEGER,
    total_cost_usd        NUMERIC(12,8),
    duration_ms           INTEGER,
    embedding_calls       INTEGER,

    -- Field-specific (NULL for message_passing)
    field_patterns_count  INTEGER,
    cross_agent_accesses  INTEGER,
    field_stability       REAL,

    created_at  TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(experiment, topic, condition, run_number)
);
```

### 8.3 Analysis Queries

```sql
-- Paired comparison: information retention by condition
SELECT
    topic,
    MAX(CASE WHEN condition = 'message_passing' THEN info_retention_count END) AS mp_retention,
    MAX(CASE WHEN condition = 'shared_field' THEN info_retention_count END) AS field_retention,
    MAX(CASE WHEN condition = 'shared_field' THEN info_retention_count END) -
    MAX(CASE WHEN condition = 'message_passing' THEN info_retention_count END) AS retention_delta
FROM experiment_results
WHERE experiment = 'prd108_field_v1'
GROUP BY topic, run_number
ORDER BY topic, run_number;

-- Aggregate: mean metrics by condition
SELECT
    condition,
    AVG(info_retention_count::float / NULLIF(info_retention_total, 0)) AS avg_retention_rate,
    AVG(context_quality_score) AS avg_quality,
    AVG(task_accuracy_score) AS avg_accuracy,
    AVG(total_tokens) AS avg_tokens,
    AVG(duration_ms) AS avg_duration
FROM experiment_results
WHERE experiment = 'prd108_field_v1'
GROUP BY condition;
```

---

## 9. Cost Analysis

### 9.1 Embedding Costs

| Operation | Calls per Mission | Cost per Call | Total |
|-----------|------------------|---------------|-------|
| `inject()` — Researcher (20 findings) | 20 | ~$0.001 | $0.020 |
| `inject()` — Analyst (15 analyses) | 15 | ~$0.001 | $0.015 |
| `query()` — Analyst (5 queries) | 5 | ~$0.001 | $0.005 |
| `query()` — Writer (5 queries) | 5 | ~$0.001 | $0.005 |
| `query()` — Coordinator (3 status checks) | 3 | ~$0.001 | $0.003 |
| **Total embedding overhead per mission** | **48** | | **$0.048** |

### 9.2 Comparison with LLM Costs

A typical 3-agent mission with Sonnet 4.6 costs ~$0.15-0.50 in LLM calls. The field's embedding overhead ($0.048) is 10-30% on top — within the ≤120% token efficiency threshold.

### 9.3 Qdrant Storage

At prototype scale (30 experiment runs × 50 patterns × 2048 floats × 4 bytes = ~12MB), storage is negligible. Qdrant's free tier handles millions of vectors.

---

## 10. Cross-PRD Integration

| PRD | Integration | Notes |
|-----|-------------|-------|
| **PRD-107** | `VectorFieldSharedContext` implements `SharedContextPort`. Validates PRD-107's interface design | If the interface doesn't feel right during implementation, update PRD-107 before Phase 3 |
| **PRD-102** | Coordinator manages field lifecycle: `create_context()` at mission start, `destroy_context()` at end, `query()` for progress assessment | Coordinator code is identical for both conditions — only adapter differs |
| **PRD-103** | Experiment's LLM-as-judge scoring uses PRD-103's rubric format for `task_accuracy_score` | Consistent quality measurement methodology |
| **PRD-106** | Field events (`field.created`, `field.injected`, etc.) flow into `mission_events` telemetry | First real data for the telemetry pipeline |
| **PRD-105** | Embedding costs counted against mission budget | `inject()` and `query()` embedding calls tracked in `llm_usage` with `request_type='embedding'` |
| **Phase 3** (110-116) | Experiment results are the go/no-go gate. Pass/fail criteria determine entire Phase 3 roadmap | This is the most important deliverable of PRD-108 |

---

## 11. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Prototype is just "RAG with extra steps" — resonance adds no value beyond cosine similarity | High | Medium | Include decay + reinforcement. If results match plain RAG, the unique mechanisms aren't contributing — honest negative result |
| 2 | Over-engineering — building attractor dynamics, bifurcation, multi-field coupling | Medium | Medium | Hard scope: 5 operations only. No attractors, no coupling, no emergence. Test the minimum hypothesis |
| 3 | Wrong experiment task — doesn't exercise field's advantages | High | Medium | Choose tasks where context preservation is critical. Validate with dry run. 5 topics provide diversity |
| 4 | Confirmation bias — desire for Phase 3 biases evaluation | Medium | High | Blind human evaluation. LLM-as-judge as secondary cross-validation. Pre-registered pass/fail thresholds |
| 5 | Embedding quality bottleneck — qwen3 produces poor domain embeddings | High | Low | Sanity check first: known-similar texts should have similarity >0.8. Switch to OpenAI text-embedding-3-large if needed |
| 6 | Qdrant deployment complexity on Railway | Medium | Low | Docker single-command deploy. Persistent volume for data. `:memory:` mode for local dev. Fallback to FAISS if Railway deployment fails |
| 7 | Decay rate miscalibration — λ=0.1 too fast or too slow | Medium | Medium | Sensitivity analysis across λ ∈ {0.05, 0.1, 0.2}. Pick the λ that maximizes information retention |
| 8 | Uncontrolled resonance amplification — one pattern dominates | Medium | Medium | Reinforcement cap (2.0×), co-access bonus cap (+2%), strength ratio monitoring |
| 9 | "Neural field resonance" is just cosine similarity rebranded | High | Medium | The novelty is: (a) decay removes stale info, (b) reinforcement amplifies co-accessed patterns. If (a)+(b) don't improve results, accept the result honestly |
| 10 | Small sample size (30 runs) lacks statistical power | Medium | Medium | Wilcoxon signed-rank is designed for small paired samples. Effect size (d>0.5) more important than p-value. If results are ambiguous, run 30 more |

---

## 12. Acceptance Criteria

### Must Have

- [x] **Qdrant backend selected and justified** — comparison table, Railway deployment spec (Section 2.1, 3.1)
- [x] **Field data model defined** — point schema with vector, payload structure, collection naming (Section 3.2)
- [x] **5 core operations implemented** — inject, query, decay, reinforce, measure_stability with code (Section 4)
- [x] **Resonance scoring formula** — `R = cos²(θ) × decayed_strength` with anti-domination safeguard (Section 7)
- [x] **Decay calibration** — λ=0.1 default, sensitivity analysis plan, score-time implementation (Section 4.3)
- [x] **`VectorFieldSharedContext` adapter** — implements PRD-107's `SharedContextPort` completely (Section 5)
- [x] **Experiment protocol** — 5 topics, 2 conditions, 3 reps, 30 total runs (Section 6)
- [x] **Pass/fail criteria for Phase 3** — 4 explicit thresholds with decision rules (Section 6.5)
- [x] **Telemetry capture** — field events, experiment_results table, analysis queries (Section 8)
- [x] **Cost analysis** — embedding overhead per mission, comparison with LLM costs (Section 9)

### Should Have

- [x] **Statistical methodology** — Wilcoxon signed-rank, paired design, blind evaluation (Section 6.6)
- [x] **Hebbian reinforcement** — co-access bonus, access count tracking (Section 4.4)
- [x] **Anti-domination safeguard** — reinforcement cap, strength ratio monitoring (Section 7.3)
- [x] **System diagram** — coordinator → agents → field → Qdrant (Section 3.3)

### Nice to Have

- [x] **Experiment results table DDL** — `experiment_results` for structured analysis (Section 8.2)
- [x] **FAISS benchmark plan** — local-only comparison baseline (Section 2.1)
- [x] **Config extension** — all field parameters configurable via env vars (Section 3.1)

---

## Appendix A: Research Sources

| Source | What It Informed |
|--------|-----------------|
| Qdrant (`qdrant/qdrant`) | Payload filtering, Recommendations API, Docker deploy, `:memory:` testing mode |
| FAISS (`facebookresearch/faiss`) | `IndexFlatL2` exact search baseline, thread-safety limitations |
| Redis Vector Search (Redis Stack) | Eliminated — Railway Redis is vanilla, no Stack modules |
| Ebbinghaus Forgetting Curve (1885) | Exponential decay formula `S(t) = S₀ × e^(-λt)` |
| Hebb, *Organization of Behavior* (1949) | Co-access reinforcement pattern — "fire together, wire together" |
| Elasticsearch decay functions | Score-time decay application (no deletion) |
| Kornell & Bjork (2008) | Spaced repetition — re-access resets decay clock |
| Context Engineering, Ch. 08 | 8 core field operations, boundary permeability |
| Context Engineering, Ch. 09 | Resonance formula, decay formula, attractor protection |
| Context Engineering, Ch. 10 | Multi-field operations (deferred to Phase 3) |
| Context Engineering, Ch. 11 | Stability measurement, gradient convergence (simplified for prototype) |
| Automatos `memory_types.py:65` | Existing exponential decay with access_count boost |
| Automatos `inter_agent.py:400-649` | SharedContextManager — Phase 2 baseline |
| Automatos `embedding_manager.py` | qwen3-embedding-8b, 2048-dim, batch support |
| Automatos `config.py` | `MEMORY_DECAY_RATE=0.1`, consistent decay parameters |
| Railway community (station.railway.com) | Confirmed vanilla Redis — no Redis Stack support |
