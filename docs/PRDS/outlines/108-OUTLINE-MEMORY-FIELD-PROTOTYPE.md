# PRD-108 Outline: Memory Field Prototype

**Type:** Research + Design Outline
**Status:** Outline
**Depends On:** PRD-107 (Context Interface Abstraction), PRD-100 (Master Research), Context Engineering repo (chapters 08-11)
**Feeds Into:** Phase 3 PRDs (110-116), PRD-82B (Sequential Mission Coordinator)

---

## 1. Problem Statement

PRD-100 Risk #6 states: *"Context Engineering theory doesn't translate to code — PRD-108 is the prototype gate. If field prototype doesn't outperform message passing, reassess Phase 3."* This PRD defines that experiment.

### The Hypothesis

Agents sharing a continuous semantic field — where information resonates, decays, and forms attractors — produce higher-quality collaborative output than agents passing discrete messages.

### What Message-Passing Looks Like Today

```
Agent A → "Here are my research findings: ..." → Agent B
Agent B → "Based on your findings, I conclude..." → Agent C
Agent C reads A's summary (lossy) + B's summary (lossier)
```

Context degrades at every hop. Agent C sees Agent A's work through Agent B's interpretation. If Agent A's finding #7 is relevant to Agent C but Agent B didn't mention it, it's lost.

### What a Shared Field Would Look Like

```
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

### Why This Matters

If the field approach demonstrably outperforms message-passing — even modestly — it validates the entire Phase 3 roadmap (PRDs 110-116). If it doesn't, we save months of engineering by staying with message-passing and focus Phase 3 investment elsewhere.

### What This PRD Delivers

A controlled experiment: same task, same agents, same models — one run with message-passing, one with a shared vector field. Measured comparison on context quality, task accuracy, token efficiency, and latency.

---

## 2. Prior Art Research Targets

### Vector Store Backends to Evaluate

| Backend | Key Strength | Key Weakness | Prototype Suitability |
|---------|-------------|-------------|----------------------|
| **FAISS** (facebookresearch/faiss) | Fastest CPU search at small scale; trivial persistence via `write_index()`/`read_index()`; `IndexFlatL2` exact for <10K vectors | No metadata filtering (external table required); no thread-safe writes (needs mutex); no TTL/decay; no built-in payload storage | **Good for benchmarking** — pure vector math, minimal overhead, ideal for controlled experiment |
| **Qdrant** (qdrant/qdrant) | Payload filtering as first-class (datetime, numeric, keyword); Recommendations API (positive/negative examples = resonance discovery); Docker single-command deploy; Python client with `:memory:` local mode | No built-in TTL on points; ACORN search adds latency; documentation sparse for recommend API | **Best for production prototype** — native metadata, scoped queries, real-time upsert |
| **Redis Vector Search** (Redis Stack) | Already deployed in Automatos; native TTL on keys = automatic decay; hybrid KNN + metadata filtering in single command; sub-ms latency at small scale | Max 32,768 dimensions (fine for 2048); index rebuild on schema change; 10 max attributes per index | **Best for zero-infrastructure prototype** — extends existing Redis, TTL maps to decay |
| **Pinecone** | Managed service, serverless; namespace isolation per mission; metadata filtering | External dependency; cost per query; latency overhead from network hop | **Not recommended** — adds dependency for a prototype that should be self-contained |
| **S3 Vectors** (existing in Automatos) | Already configured (`automatos-vector-index`); workspace-scoped; 2048-dim cosine; batch put/search | Document-oriented (not designed for ephemeral field patterns); no TTL; no real-time metadata queries; designed for RAG, not live fields | **Not recommended for field** — wrong abstraction level, but useful reference for embedding pipeline |

### Temporal Decay Research Targets

| Approach | Source | Key Insight |
|----------|--------|-------------|
| **Exponential decay** `S(t) = S₀ × e^(-λt)` | Standard IR, Ebbinghaus (1885) | Smooth, configurable via λ; half-life = ln(2)/λ. Already implemented in `memory_types.py:65` |
| **Elasticsearch decay functions** | ES `function_score` query | Three profiles: `linear`, `exp`, `gauss` — each with origin, scale, offset, decay params. Score-time application, no deletion |
| **Cache eviction as decay analogy** | LRU/LFU/ARC literature | LRU = recency (last access time); LFU = frequency (access count); ARC = adaptive hybrid. Maps to field reinforcement |
| **Hebbian reinforcement** | Hebb (1949), Touchette & Lloyd (2004) | "Neurons that fire together wire together" — co-accessed vectors boost each other. Key mechanism for resonance |
| **Spaced repetition** | Kornell & Bjork (2008) | Re-access resets decay clock; repeated retrieval increases retention strength. Maps to agent re-queries |

### Context Engineering Theory (Chapters 08-11)

| Chapter | Document | Key Extraction for Prototype |
|---------|----------|------------------------------|
| **08** | `docs/context-engineering/00_foundations/08_neural_fields_foundations.md` | 8 core operations (inject, decay, resonate, amplify, attenuate, tune, collapse, measure). Boundary permeability controls injection strength: `effective_strength = strength × boundary_permeability` |
| **09** | `docs/context-engineering/00_foundations/09_persistence_and_resonance.md` | Decay formula `S(t) = S₀ × e^(-λt)`. Resonance formula `R(A,B) = cos(θ) × |A| × |B| × S(A,B)`. Attractor protection: `effective_decay = decay_rate × (1 - attractor_protection)` where protection = Σ(resonance × 0.5), capped at 0.5 |
| **10** | `docs/context-engineering/00_foundations/10_field_orchestration.md` | Multi-field operations: superposition (vector addition), constructive/destructive interference, field coupling (weak/strong/directional). Sequential vs parallel processing. Feedback loops for self-refinement |
| **11** | `docs/context-engineering/00_foundations/11_emergence_and_attractor_dynamics.md` | Attractor detection via gradient convergence: `‖gradient_field(x,y)‖ < threshold`. Stability formula: `stability = (avg_attractor_strength × 0.6) + (organization_score × 0.4)`. Bifurcation detection for phase transitions |

### Key Patterns from Existing Automatos Infrastructure

| Component | Location | Reuse Opportunity |
|-----------|----------|-------------------|
| **EmbeddingManager** | `orchestrator/core/llm/embedding_manager.py` | Ready-to-use: `generate_embeddings_batch()` with qwen3-embedding-8b (2048-dim) via OpenRouter |
| **SharedContextManager** | `orchestrator/modules/agents/communication/inter_agent.py:400-649` | Merge strategies (override/append/consensus), team scoping, Redis + in-memory dual store. Prototype's Phase 2 baseline |
| **Memory decay config** | `orchestrator/config.py:100-111` | `MEMORY_DECAY_RATE=0.1`, `MEMORY_DECAY_ARCHIVE_THRESHOLD=0.3` — same λ for field decay |
| **MemoryNamespace** | `orchestrator/modules/memory/unified_memory_service.py:39-117` | Extend pattern: `mem:{workspace_id}:field:{field_id}` for field-scoped storage |
| **Redis infrastructure** | `orchestrator/core/redis/` | Connection pooling, pub/sub channels, TTL management. Zero new infrastructure for Redis-backed prototype |
| **S3VectorsBackend** | `orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py` | Reference for vector operations. Not directly reusable (wrong abstraction) but demonstrates the embedding→store→search pipeline |
| **L2 Short-term Memory** | `orchestrator/modules/memory/operations/` | Ebbinghaus decay + importance scoring + access tracking + promotion. Structural template for field decay |
| **ContextProvider port** | PRD-107 `orchestrator/core/ports/context.py` (planned) | `SharedContextPort.inject()/query()` — the interface the field adapter must implement |

---

## 3. Prototype Scope

### What This Is

A **minimum viable shared field** where 2-3 agents inject findings as embeddings and query for resonant patterns. NOT full neural field orchestration — just enough to test whether shared fields outperform message passing.

### Minimum Viable Operations (5 core)

| # | Operation | What It Does | Formula |
|---|-----------|-------------|---------|
| 1 | **inject(pattern, strength)** | Add embedding to shared field with metadata | `effective_strength = strength × boundary_permeability` |
| 2 | **query(embedding, top_k)** | Retrieve resonant patterns by cosine similarity, weighted by decay + reinforcement | Ranked by `similarity × decay × access_boost` |
| 3 | **decay()** | Apply exponential decay to all field patterns | `S(t) = S₀ × e^(-λt)` |
| 4 | **reinforce(pattern_id)** | Counter decay when a pattern is accessed (Hebbian reinforcement) | `access_count += 1; last_accessed = now()` |
| 5 | **measure_stability()** | Quantify field convergence | `stability = (avg_attractor_strength × 0.6) + (organization_score × 0.4)` |

### What's Explicitly Out of Scope

- Full attractor dynamics (form, classify, basin mapping) — too complex for prototype
- Bifurcation detection — Phase 3 (PRD-112)
- Multi-field coupling (superposition, interference) — Phase 3 (PRD-110)
- K8s micro-agent infrastructure — Phase 3 (PRD-113)
- Symbolic mechanism integration — Phase 3 (PRD-114)
- Emergence detection & safety — Phase 3 (PRD-115)

### Agent Setup

```
┌──────────────────────────────────────────┐
│              SHARED FIELD                │
│  (Redis Vector Search or Qdrant)         │
│                                          │
│  Vectors: 2048-dim (qwen3-embedding-8b)  │
│  Metadata: agent_id, strength, timestamp │
│  Decay: λ=0.1 (7h half-life)            │
│  Reinforcement: +5% per access, cap 2×   │
├──────────────┬───────────────────────────┤
│              │                           │
▼              ▼                           ▼
Agent A      Agent B                   Agent C
(Researcher) (Analyst)                (Writer)
inject()     inject()                  query()
query()      query()                   query()
```

- **Agent A (Researcher):** Injects raw findings from web search / document analysis
- **Agent B (Analyst):** Reads A's findings via field query, injects analysis/synthesis
- **Agent C (Writer):** Queries field for resonant patterns, produces final output

### Implements PRD-107 Interface

The field prototype implements `SharedContextPort` from PRD-107:

```python
class VectorFieldSharedContext(SharedContextPort):
    """PRD-108 prototype: vector field backend for SharedContextPort."""

    async def inject(self, context_id, key, value, agent_id, strength=1.0):
        embedding = await self._embedder.generate_embedding(f"{key}: {value}")
        effective_strength = strength * self.boundary_permeability
        await self._store.upsert(
            field_id=context_id,
            vector=embedding,
            payload={
                "agent_id": agent_id,
                "key": key,
                "value": value,
                "strength": effective_strength,
                "created_at": datetime.utcnow().isoformat(),
                "access_count": 0,
            }
        )

    async def query(self, context_id, query, agent_id, top_k=10):
        query_embedding = await self._embedder.generate_embedding(query)
        raw_results = await self._store.search(
            field_id=context_id,
            vector=query_embedding,
            top_k=top_k * 2,  # Over-fetch for post-decay filtering
        )
        # Apply decay + reinforcement scoring
        scored = self._decay_scorer.score_with_decay(raw_results)
        # Record access (Hebbian reinforcement)
        for result in scored[:top_k]:
            await self._store.increment_access(result.id)
        return scored[:top_k]

    async def create_context(self, team_agent_ids, initial_data=None):
        field_id = str(uuid.uuid4())
        await self._store.create_field(field_id, dimension=2048)
        return field_id

    async def destroy_context(self, context_id):
        await self._store.delete_field(context_id)
```

---

## 4. Resonance Implementation

### 4.1 Core Formula (from Context Engineering Chapter 09)

```
R(A, B) = cos(θ) × |A| × |B| × S(A, B)
```

Where:
- `cos(θ)` = cosine similarity between embedding vectors A and B
- `|A|`, `|B|` = strength values of patterns A and B
- `S(A, B)` = semantic relatedness function (for prototype: cosine similarity itself, making `S(A,B) = cos(θ)`)

**Simplified prototype formula:**
```
resonance_score = cosine_similarity² × strength_A × strength_B
```

The squared cosine amplifies high-similarity pairs and suppresses noise — vectors at 0.9 similarity get 0.81 resonance while vectors at 0.5 get only 0.25.

### 4.2 Temporal Decay

```python
def compute_decayed_strength(initial_strength, age_hours, access_count, lambda_=0.1):
    """
    S(t) = S₀ × e^(-λt) × access_boost

    λ = 0.1 → half-life ≈ 6.93 hours
    access_boost = 1 + (access_count × 0.05), capped at 2.0
    """
    decay = math.exp(-lambda_ * age_hours)
    access_boost = min(1.0 + (access_count * 0.05), 2.0)
    return initial_strength * decay * access_boost
```

**Decay calibration:**

| λ Value | Half-Life | Use Case |
|---------|-----------|----------|
| 0.05 | ~14 hours | Long-running missions (multi-day research) |
| 0.1 | ~7 hours | Default — standard mission duration |
| 0.2 | ~3.5 hours | Fast-turnaround tasks |

**Start with λ=0.1.** Measure in-field access patterns for 1 week, then calibrate via median lifetime of accessed patterns.

### 4.3 Reinforcement Mechanism

When Agent B queries the field and accesses a pattern originally injected by Agent A:

1. **Access count incremented** — pattern's `access_count += 1`
2. **Last-accessed timestamp updated** — enables recency-based boosting
3. **Decay effectively reset** — next decay calculation uses new `last_accessed`, not `created_at`
4. **Co-accessed patterns linked** (Hebbian) — patterns accessed in the same query get a mutual boost

```python
async def reinforce_on_access(self, accessed_pattern_ids: list[str]):
    """Hebbian reinforcement: co-accessed patterns boost each other."""
    for pid in accessed_pattern_ids:
        await self._store.increment_access(pid)
    # Co-access bonus: if Agent B accessed patterns 3 and 7 together,
    # both get a small mutual strength boost
    if len(accessed_pattern_ids) > 1:
        for pid in accessed_pattern_ids:
            await self._store.boost_strength(pid, bonus=0.02)
```

### 4.4 Score-Time Decay vs Deletion

**Decision: Score-time decay (no deletion).**

Patterns are never deleted from the field. Decay is applied at query time as a scoring modifier. This means:
- Decayed patterns can be resurrected if re-accessed (Hebbian reinforcement)
- Field history is preserved for telemetry analysis (PRD-106)
- No cleanup jobs needed during experiment
- Archival threshold (`strength × decay < 0.05`) filters them from query results

**Exception:** Redis TTL-based approach is an alternative for Redis Vector Search backend — keys auto-expire, but `touch()` on access resets TTL. Simpler but less nuanced.

---

## 5. Experiment Design

### 5.1 Task Selection

The experiment task must be:
- **Multi-agent** — requires at least 2 agents to collaborate
- **Context-dependent** — later agents benefit from earlier agents' full context (not just summaries)
- **Measurable** — output quality can be objectively scored
- **Repeatable** — same task can be run multiple times for statistical significance

**Proposed task:** "Research a topic and produce an analysis report"
- Agent A (Researcher): Search for information on topic X, inject findings
- Agent B (Analyst): Read findings, produce structured analysis
- Agent C (Writer): Produce final report combining research + analysis

**Run 5 different topics** through both conditions (message-passing and field) for 10 total runs.

### 5.2 Experimental Conditions

| Condition | Context Mechanism | Implementation |
|-----------|-------------------|----------------|
| **Control: Message-Passing** | Agent A's output summarized → passed as text to Agent B → B's output passed to Agent C | Existing SharedContextManager (`RedisSharedContext` adapter from PRD-107) |
| **Treatment: Shared Field** | All agents inject/query shared vector field. No explicit message passing. | `VectorFieldSharedContext` adapter from this PRD |

**Controlled variables (identical across conditions):**
- Same LLM model for each role (e.g., Sonnet 4.6 for all)
- Same task descriptions and success criteria
- Same tool access per agent
- Same token budgets

### 5.3 Metrics

| Metric | How Measured | Why It Matters |
|--------|-------------|----------------|
| **Context Quality** | Human eval (1-5 Likert): "Did the final report reflect all relevant findings from Agent A?" | Core hypothesis: field preserves more context |
| **Information Retention** | Count: how many of Agent A's findings appear in the final output | Quantifies the telephone game problem |
| **Task Completion Accuracy** | LLM-as-judge (rubric-scored) against gold-standard answer | Overall quality comparison |
| **Token Efficiency** | Total tokens consumed across all agents | Field should reduce redundant re-explanation |
| **Latency** | Wall-clock time from mission start to final output | Field adds embedding overhead — is it worth it? |
| **Cross-Agent Resonance** | Count of field patterns accessed by >1 agent | Measures whether shared field is actually used |
| **Field Stability** | Convergence iteration count (stability score > 0.75) | Validates attractor dynamics theory |

### 5.4 Success Criteria (Pass/Fail for Phase 3)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Information Retention** | Field retains ≥20% more findings than message-passing | Must demonstrate the telephone game fix |
| **Context Quality** | Human eval ≥0.5 points higher (on 5-point scale) | Perceptible quality improvement |
| **Token Efficiency** | Field uses ≤120% of message-passing tokens | Small overhead acceptable; >20% overhead = too expensive |
| **Latency** | Field completes in ≤150% of message-passing time | Embedding overhead must be bounded |

**If ALL four pass:** Phase 3 validated. Proceed to PRDs 110-116.
**If ANY fail:** Analyze which mechanism underperformed. Consider targeted fixes before abandoning Phase 3.
**If information retention fails:** The core hypothesis is wrong. Reassess Phase 3 entirely.

### 5.5 Statistical Validity

- **5 topics × 2 conditions = 10 runs minimum**
- **3 repetitions per topic-condition pair = 30 total runs** (for variance estimation)
- **Paired comparison:** Same topic in both conditions, reducing topic-variance noise
- **Wilcoxon signed-rank test** for non-parametric paired comparison (small sample)

---

## 6. Key Design Questions

### Q1: Which vector store backend?

**Options:**
- **Redis Vector Search** — Zero new infrastructure; native TTL = automatic decay; hybrid KNN + metadata filtering in single command; sub-ms latency
- **Qdrant** — Richer metadata filtering; Recommendations API maps to resonance discovery; `:memory:` mode for testing; better query expressiveness
- **FAISS** — Fastest raw vector operations; simplest API; requires external metadata table and mutex for writes

**Recommendation: Redis Vector Search for v1 prototype.** Zero infrastructure cost (already deployed), TTL maps naturally to decay, and the experiment doesn't need Qdrant's advanced features. If Redis limitations hit (10 attribute max, no recommendation API), migrate to Qdrant for v2.

**Fallback: Qdrant Docker container** if Redis Vector Search module isn't available in current Redis deployment (needs Redis Stack, not vanilla Redis).

### Q2: Embedding model and dimension?

**Decision: Use existing `qwen/qwen3-embedding-8b` (2048-dim) via OpenRouter.** Already configured in `EmbeddingManager`. No new dependencies. Consistent with S3 Vectors dimension.

**Cost consideration:** Each `inject()` call requires one embedding API call (~$0.001). Each `query()` requires one. For 50 inject + 30 query = 80 calls per mission = ~$0.08. Negligible.

### Q3: How to handle the `S(A, B)` semantic relatedness function?

**Options:**
- **Cosine similarity as `S(A,B)`** — simplest; resonance = cosine² × strengths
- **Learned function** — train on access patterns; which pairs do agents actually co-access?
- **Domain-specific rules** — e.g., code findings relate to code analyses

**Recommendation: Cosine similarity for prototype.** It's the obvious baseline. If it works, great. If resonance patterns look random, explore learned functions in Phase 3.

### Q4: Field update frequency — continuous or batched?

**Options:**
- **Continuous** — Every `inject()` immediately available to `query()`
- **Batched** — Injections accumulate; field "ticks" periodically (like heartbeat)

**Recommendation: Continuous.** The experiment measures real-time collaboration quality. Batching would add artificial latency and reduce the field's advantage over message-passing. Redis and Qdrant both support real-time upsert.

### Q5: Decay implementation — score-time or periodic cleanup?

**Recommendation: Score-time decay for the experiment** (compute decay at query time). This preserves all data for post-experiment analysis. Consider periodic cleanup for production if storage grows.

**Exception:** If using Redis TTL backend, decay IS periodic cleanup (key expiration). Use `PERSIST` on access to reset TTL (reinforcement).

### Q6: What constitutes a "resonant" pattern in practice?

**Concrete definition for prototype:**
- A pattern is "resonant" with a query if `cosine_similarity > 0.7` AND `decayed_strength > 0.1`
- Resonance between two patterns: `resonance = cosine_sim(A, B)² × strength_A × strength_B`
- Patterns with resonance > 0.5 are "strongly resonant" — flagged in telemetry

### Q7: How does the field connect to the coordinator (PRD-102)?

The coordinator creates a field at mission start via `SharedContextPort.create_context()`. Each task's agent receives the `context_id`. Agents inject their outputs into the field. The coordinator queries the field to assess mission progress. At mission end, the coordinator destroys the field via `destroy_context()`.

```
Coordinator → create_context([agent_a, agent_b, agent_c])
            → assign_task(agent_a, context_id=field_id)
            → assign_task(agent_b, context_id=field_id)
            → assign_task(agent_c, context_id=field_id)
            → [agents inject/query field during execution]
            → query(field_id, "mission summary") → assess completion
            → destroy_context(field_id)
```

---

## 7. Acceptance Criteria for Full PRD-108

### Must Have

- [ ] **Vector store backend selected and justified** — comparison table with benchmarks at prototype scale (100-5000 vectors, 2048 dimensions)
- [ ] **Field data model defined** — schema for field patterns: vector, metadata (agent_id, strength, timestamp, access_count, content_hash), storage layout
- [ ] **5 core operations implemented** — inject, query, decay, reinforce, measure_stability — with method signatures, type hints, and pseudocode
- [ ] **Resonance scoring formula** — concrete implementation of `R(A,B) = cos(θ)² × |A| × |B|` with decay and reinforcement modifiers
- [ ] **Decay calibration strategy** — how to set λ, how to measure half-life in practice, default values with rationale
- [ ] **`VectorFieldSharedContext` adapter** — implements PRD-107's `SharedContextPort` interface completely
- [ ] **Experiment protocol** — task selection, controlled variables, metrics, pass/fail thresholds, statistical methodology
- [ ] **Baseline implementation** — message-passing condition using `RedisSharedContext` from PRD-107
- [ ] **Telemetry capture** — per-experiment data collection: inject counts, query counts, resonance scores, access patterns, convergence timeline
- [ ] **Pass/fail criteria for Phase 3** — explicit thresholds that determine whether Phase 3 proceeds

### Should Have

- [ ] **Benchmark results** — latency measurements for inject/query at 100, 500, 1000, 5000 vectors
- [ ] **Decay visualization** — chart showing field state over time (pattern strengths, attractor formation, convergence)
- [ ] **Hebbian reinforcement validation** — demonstrate that co-accessed patterns actually gain strength over time
- [ ] **Cost analysis** — embedding API costs per mission run, storage costs, comparison with message-passing baseline
- [ ] **Integration test** — end-to-end test with 2 agents, 1 shared field, inject→query→reinforce cycle

### Nice to Have

- [ ] **Attractor detection** — identify stable convergent patterns in the field (gradient convergence method from Chapter 11)
- [ ] **Dual-backend comparison** — run same experiment on Redis Vector Search AND Qdrant, compare operational characteristics
- [ ] **Per-mission adapter override** — factory supports selecting field vs message-passing backend per mission (for A/B testing in production)
- [ ] **Field replay** — ability to replay field evolution from stored events (for debugging and analysis)

---

## 8. Risks & Dependencies

### Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | **Prototype too simple to be meaningful** — 5 operations on a cosine similarity index may just be "RAG with extra steps" | High | Medium | Include resonance scoring (cosine²), decay, and reinforcement. If results match plain RAG, the unique mechanisms aren't contributing — and that's a valid negative result |
| 2 | **Prototype too complex to finish** — over-engineering attractor dynamics, bifurcation detection, multi-field coupling | Medium | Medium | Hard scope: 5 operations only. No attractors, no coupling, no emergence. Build the minimum that tests the hypothesis |
| 3 | **Measuring the wrong thing** — experiment task doesn't exercise the field's advantages | High | Medium | Choose tasks where context preservation is critical (research synthesis, multi-step analysis). Validate task selection with dry run |
| 4 | **Confirmation bias in evaluation** — desire for Phase 3 to succeed biases human eval | Medium | High | Blind evaluation: human raters don't know which condition produced which output. LLM-as-judge scoring as secondary metric |
| 5 | **Embedding quality bottleneck** — if qwen3-embedding-8b produces poor embeddings for the domain, cosine similarity is meaningless | High | Low | Run embedding quality sanity check first: known-similar texts should have similarity > 0.8 |
| 6 | **Redis Vector Search not available** — current Redis deployment may be vanilla Redis without Stack modules | Medium | Medium | Check with `MODULE LIST` command. Fallback: Qdrant Docker container |
| 7 | **Decay rate miscalibration** — λ=0.1 may be too fast (useful info decays before Agent C reads it) or too slow (noise persists) | Medium | Medium | Run sensitivity analysis: test λ ∈ {0.05, 0.1, 0.2}. Pick the one that maximizes information retention metric |
| 8 | **Cost of embeddings scales poorly** — each inject/query costs an API call | Low | Low | At prototype scale (80 calls/mission × $0.001 = $0.08), negligible. Flag if scaling to 1000+ calls |
| 9 | **Neural field "resonance" is just cosine similarity rebranded** — no novel mechanism beyond standard vector search | High | Medium | The novelty is: (a) decay removes stale info automatically, (b) reinforcement amplifies co-accessed patterns, (c) same interface swaps backends. If (a)+(b) don't improve results, the rebranding concern is valid — accept the result honestly |
| 10 | **Uncontrolled resonance amplification** — one strong pattern dominates the field, drowning out everything else | Medium | Medium | Cap resonance effect with scale factor: `amplification = resonance × 0.2`. Monitor max/min strength ratio per field |

### Dependencies

| Dependency | PRD | Why |
|------------|-----|-----|
| **SharedContextPort interface** | PRD-107 | Field adapter implements this interface — must be defined first |
| **Coordinator creates/destroys fields** | PRD-102 | Coordinator manages field lifecycle per mission |
| **Success criteria for verification** | PRD-103 | Experiment's LLM-as-judge scoring follows verification patterns |
| **Telemetry capture** | PRD-106 | Experiment data feeds into telemetry pipeline |
| **EmbeddingManager** | Built | Generates 2048-dim embeddings via OpenRouter — already exists |
| **Redis infrastructure** | Built | Connection pooling, TTL management — already exists |
| **Context Engineering repo** | Available | Chapters 08-11 define theoretical operations — already written |

### Cross-PRD Notes

- **PRD-107 (Context Interface):** The field prototype IS the validation gate for PRD-107's interface design. If `SharedContextPort.inject()/query()` doesn't feel right during prototype implementation, update the interface before Phase 3.
- **PRD-102 (Coordinator):** Coordinator should be able to query the field for "mission progress" — add `query(context_id, "mission status summary")` use case.
- **PRD-103 (Verification):** Experiment's quality scoring methodology should align with PRD-103's verification patterns for consistency.
- **PRD-106 (Telemetry):** Every experiment run produces telemetry: inject/query counts, resonance distributions, convergence curves. This is the first real data for outcome telemetry.
- **Phase 3 PRDs (110-116):** Prototype results determine whether these proceed. The experiment's pass/fail criteria gate the entire Phase 3 roadmap.

---

## Appendix: Research Sources

| Source | What It Informed |
|--------|-----------------|
| FAISS (facebookresearch/faiss) | IndexFlatL2 for exact search at small scale; `add_with_ids()`/`search()`/`remove_ids()` API; thread-safety limitations (CPU reads safe, writes need mutex) |
| Qdrant (qdrant/qdrant) | Payload filtering + datetime indexing; Recommendations API for resonance discovery; ACORN search (v1.16.0+); `:memory:` local mode for testing |
| Redis Vector Search (Redis Stack) | FT.CREATE/FT.SEARCH with KNN; native TTL as automatic decay; hybrid vector + metadata queries; sub-ms latency at small scale |
| Ebbinghaus Forgetting Curve (1885) | Exponential decay formula `S(t) = S₀ × e^(-λt)`; spaced repetition as reinforcement mechanism |
| Hebb, *Organization of Behavior* (1949) | "Neurons that fire together wire together" — co-access reinforcement pattern |
| Elasticsearch decay functions | Score-time decay application (no deletion); configurable profiles (linear/exp/gauss) |
| Kornell & Bjork (2008) | Spaced repetition and distributed practice — re-access resets decay clock |
| Context Engineering, Ch. 08 (Neural Fields Foundations) | 8 core field operations; boundary permeability; field initialization parameters |
| Context Engineering, Ch. 09 (Persistence & Resonance) | Resonance formula; decay formula; attractor protection mechanics; field stability measurement |
| Context Engineering, Ch. 10 (Field Orchestration) | Multi-field operations (superposition, interference, coupling); sequential vs parallel processing; feedback loops |
| Context Engineering, Ch. 11 (Emergence & Attractors) | Gradient convergence detection; fixed point classification; basin mapping; bifurcation detection; convergence tolerance (0.01) |
| Automatos `memory_types.py:65` | Existing exponential decay: `retention = np.exp(-self.decay_factor * time_elapsed)` with access_count boost |
| Automatos `inter_agent.py:400-649` | SharedContextManager: merge strategies, team scoping, Redis + in-memory, 2h TTL |
| Automatos `embedding_manager.py` | qwen3-embedding-8b via OpenRouter, 2048 dimensions, batch support with configurable concurrency |
| Automatos `config.py:100-111` | L2 memory decay parameters: `MEMORY_DECAY_RATE=0.1`, archive threshold, promotion criteria |
| PRD-107 Outline | `SharedContextPort` interface definition: `inject()`, `query()`, `create_context()`, `destroy_context()` |
