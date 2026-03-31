# Prior Art Matrix

## Purpose

This document maps the closest known prior art against each claim element to help counsel identify which claims survive and which need narrowing. This is not an exhaustive search — counsel should conduct a formal prior-art search.

## Claim Elements Key

| Code | Element |
|------|---------|
| E1 | Mission-scoped shared vector field (created/destroyed per mission) |
| E2 | Resonance scoring: cos^2(theta) x S_0 x e^(-lambda*t) x B(n) |
| E3 | Temporal decay from last access time, computed at query time |
| E4 | Hebbian access-count reinforcement with cap |
| E5 | Co-access strength bonus (mutates stored strength) |
| E6 | Content-hash deduplication with reinforce-on-collision |
| E7 | Field stability metric (avg_strength x 0.6 + organization x 0.4) |
| E8 | Agent-callable platform tools (query/inject/stability) |
| E9 | Swappable backend interface (SharedContextPort ABC) |
| E10 | Multi-agent LLM coordination (not single-agent memory) |

## Comparator 1: Blackboard Architecture (Hayes-Roth, 1985)

**What it is:** Shared workspace where independent "knowledge sources" read and write. Control component decides which source acts next. Rule-based matching, not semantic retrieval.

**Patents found:** US6574653B1 (blackboard-centric layered architecture), US5506999A (event-driven blackboard processing). Both describe symbolic systems.

| Element | Present? | Notes |
|---------|----------|-------|
| E1 | Partial | Shared workspace exists, but not mission-scoped or vector-based |
| E2 | No | No resonance scoring, no cosine similarity |
| E3 | No | No temporal decay |
| E4 | No | No Hebbian reinforcement |
| E5 | No | No co-access dynamics |
| E6 | No | No content deduplication |
| E7 | No | No stability metric |
| E8 | No | Knowledge sources are rule-triggered, not tool-callable |
| E9 | No | No swappable backend |
| E10 | No | Expert systems, not LLM agents |

**Risk to claims:** Low. Conceptual ancestor only. The shared workspace concept is prior art, but no specific claim element overlaps.

**Safest angle:** Acknowledge the ancestry. Differentiate on semantic retrieval, dynamics, and LLM-specific integration.

## Comparator 2: Stigmergy (Grasse, 1959; Dorigo et al., 1996)

**What it is:** Indirect coordination through environmental modification. Ants deposit pheromones that decay over time and are reinforced by subsequent traversal.

| Element | Present? | Notes |
|---------|----------|-------|
| E1 | No | Spatial fields, not mission-scoped vector collections |
| E2 | No | No cosine similarity or resonance scoring |
| E3 | Partial | Pheromone evaporation is analogous to temporal decay |
| E4 | Partial | Pheromone deposit on traversal is analogous to access reinforcement |
| E5 | No | No co-access concept |
| E6 | No | No content deduplication |
| E7 | No | No convergence metric |
| E8 | No | Agents are reactive, not tool-calling LLMs |
| E9 | No | No swappable backend |
| E10 | No | Swarm robotics, not LLM coordination |

**Risk to claims:** Low for the combination. Individual elements (decay, reinforcement) have clear biological precedent. The combination applied to LLM coordination is different.

**Safest angle:** Cite as theoretical foundation, not competing implementation.

## Comparator 3: CrewAI Memory System (2024)

**What it is:** CrewAI provides per-crew shared memory with semantic retrieval and recency-aware ranking. Documentation describes shared memory, semantic search, and context injection around tasks.

| Element | Present? | Notes |
|---------|----------|-------|
| E1 | Partial | Crew-level shared memory exists, but unclear if mission-scoped with create/destroy lifecycle |
| E2 | No | No documented resonance formula; retrieval appears to be standard vector search |
| E3 | Partial | "Recency-aware" ranking is documented but specific mechanism is unclear |
| E4 | No | No documented Hebbian access reinforcement |
| E5 | No | No documented co-access bonus |
| E6 | Unknown | Deduplication behavior not documented in public sources |
| E7 | No | No stability metric |
| E8 | Partial | Memory is injected into agent context, but unclear if agents can directly query/inject via tools |
| E9 | No | No documented swappable backend interface |
| E10 | Yes | Multi-agent LLM coordination |

**Risk to claims:** Medium. This is the closest commercial competitor. The "shared memory with semantic retrieval for multi-agent LLMs" concept overlaps. The specific dynamics (resonance formula, Hebbian reinforcement, co-access, lifecycle binding, stability metric) are not documented in CrewAI's public materials.

**Safest angle:** Narrow claims to the specific resonance scoring formula, Hebbian dynamics, and lifecycle integration. Do not claim "shared memory for agents" broadly — CrewAI has a version of that.

**Action for counsel:** Investigate CrewAI's actual implementation (open source) to determine if any undocumented features overlap with E2-E7.

## Comparator 4: LangGraph Shared State / LangMem (2024)

**What it is:** LangGraph provides shared state dictionaries passed through graph nodes. LangMem adds semantic memory with vector search and store-backed persistence.

| Element | Present? | Notes |
|---------|----------|-------|
| E1 | Partial | Shared state exists per graph execution, but key-value based, not vector-based |
| E2 | No | No resonance scoring |
| E3 | No | No temporal decay on state entries |
| E4 | No | No access-based reinforcement |
| E5 | No | No co-access dynamics |
| E6 | No | State is overwritten, not deduplicated |
| E7 | No | No stability metric |
| E8 | Partial | Agents access state through graph edges, not semantic query tools |
| E9 | No | Tightly coupled to LangGraph execution model |
| E10 | Yes | Multi-agent LLM coordination |

**Risk to claims:** Low-Medium. LangGraph's shared state is structurally different (key-value vs. semantic vectors). LangMem adds semantic search but appears to be per-user/per-thread, not mission-scoped shared across agents.

**Safest angle:** Differentiate on semantic retrieval with dynamics (not static key-value state) and mission-scoped lifecycle.

## Comparator 5: Mitra — Field-Theoretic Memory (arXiv:2602.21220, Jan 2026)

**What it is:** Academic paper proposing treatment of agent memory as continuous fields governed by modified heat equations (PDEs with diffusion, decay, and source terms). Evaluated on single-agent conversation benchmarks. Reports +116% F1 on multi-session reasoning.

| Element | Present? | Notes |
|---------|----------|-------|
| E1 | No | No mission scoping; memory is per-agent, persistent |
| E2 | Partial | Uses field dynamics for retrieval, but PDE-based diffusion, not cos^2 x decay x boost |
| E3 | Yes | Temporal decay is core to the PDE formulation |
| E4 | No | No explicit Hebbian access reinforcement |
| E5 | No | No co-access bonus |
| E6 | No | No content-hash deduplication |
| E7 | No | No stability metric |
| E8 | No | No agent-callable tools; memory is automatic |
| E9 | No | No swappable backend |
| E10 | No | Single-agent memory enhancement, not multi-agent coordination |

**Risk to claims:** Medium-High for broad "field dynamics for agent memory" claims. Low for the specific combination. Mitra's work validates the theoretical direction but addresses a different problem (single-agent memory quality) with a different mechanism (PDE-based diffusion on 2D manifold).

**Safest angle:** Cite Mitra as the closest academic prior art. Differentiate clearly on: (1) multi-agent shared field vs. single-agent memory, (2) direct vector search with post-hoc scoring vs. PDE-based diffusion, (3) mission lifecycle binding, (4) agent-callable tools, (5) production deployment vs. benchmark evaluation.

**Action for counsel:** This paper was published January 31, 2026, approximately 7 weeks before our implementation. Counsel should assess whether our claims survive Mitra as prior art given the clear differences in scope and mechanism.

## Comparator 6: AutoGen GroupChat / OpenAI Swarms / Google A2A

**What they are:** Message-passing coordination for multi-agent LLMs. AutoGen uses conversational exchange. Swarms use conversation handoff. A2A uses agent-to-agent protocol messages.

| Element | Present? | Notes |
|---------|----------|-------|
| E1-E9 | No | All use message passing, not shared vector fields |
| E10 | Yes | Multi-agent LLM coordination |

**Risk to claims:** None for specific claims. These represent the baseline approach that our system is designed to improve upon.

## Summary Matrix

| Element | Blackboard | Stigmergy | CrewAI | LangGraph | Mitra | AutoGen/Swarms |
|---------|-----------|-----------|--------|-----------|-------|---------------|
| E1 Mission-scoped field | Partial | - | Partial | Partial | - | - |
| E2 Resonance scoring | - | - | - | - | Partial | - |
| E3 Temporal decay | - | Partial | Partial | - | Yes | - |
| E4 Hebbian reinforcement | - | Partial | - | - | - | - |
| E5 Co-access bonus | - | - | - | - | - | - |
| E6 Hash dedup + reinforce | - | - | ? | - | - | - |
| E7 Stability metric | - | - | - | - | - | - |
| E8 Agent-callable tools | - | - | Partial | Partial | - | - |
| E9 Swappable backend | - | - | - | - | - | - |
| E10 Multi-agent LLM | - | - | Yes | Yes | - | Yes |

**Elements with NO known prior art:** E5 (co-access bonus), E6 (hash dedup with reinforce-on-collision), E7 (stability metric), E9 (swappable backend for A/B).

**Elements with partial prior art:** E1, E2, E3, E4, E8 — these need the combination argument.

**Conclusion for counsel:** The strongest claim position is the specific combination of E1-E10 applied to multi-agent LLM coordination. Individual elements have precedent. The combination does not appear in any known system. Elements E5, E6, E7 may be independently novel but are stronger as dependent claims under the combination.
