# Claim Boundary

## Strongest Independent System Claim

A system for coordinating multiple LLM agents within a mission, comprising:

- a mission-scoped shared vector collection created on mission start and destroyed on mission completion
- an injection mechanism that converts agent task outputs into high-dimensional embedding vectors with metadata (agent ID, strength, timestamp, content hash) and stores them in the shared collection
- a query mechanism that retrieves patterns by semantic similarity and ranks them by a resonance score combining squared cosine similarity, exponential temporal decay from last access, and access-count-based reinforcement
- agent-callable platform tools enabling agents to query, inject, and measure the shared field during task execution

**Why this is the strongest system claim:** It captures the full architecture — the shared field, the scoring function, the lifecycle binding, and the agent interface — as a single coordinated system. Each element alone exists in prior art. The combination applied to multi-agent LLM coordination does not appear in any known system.

## Strongest Independent Method Claim

A method for reducing information loss in sequential multi-agent LLM pipelines, comprising:

- creating a shared embedding space at mission start
- automatically injecting each agent's completed task output into the shared space
- enabling any downstream agent to query the shared space by semantic meaning, bypassing intermediate agents
- ranking results by resonance (squared cosine similarity x temporally-decayed strength x access reinforcement)
- reinforcing patterns that are retrieved (Hebbian access boost) and co-retrieved (co-access strength bonus)
- destroying the shared space when the mission terminates

**Why this is the strongest method claim:** It focuses on the process — what happens and in what order — rather than the system components. This is harder to design around because competitors would need to change the workflow, not just swap out a component.

## Dependent Claim Ideas

| # | Depends On | Element |
|---|-----------|---------|
| D1 | System | Content deduplication via SHA-256 hash with reinforce-on-collision |
| D2 | System | Field stability metric (weighted mean strength + organization score) |
| D3 | System | Boundary permeability coefficient controlling injection strength |
| D4 | System | Over-fetch factor (3x top_k) compensating for archival filtering |
| D5 | Method | Seeding the shared space with the mission goal as the first pattern |
| D6 | Method | Archival threshold filtering at query time (non-destructive) |
| D7 | System | Abstract interface (SharedContextPort) enabling backend substitution for A/B comparison |
| D8 | Method | Co-access bonus mutating stored strength (not just query-time scoring) |

## What We Are NOT Claiming

We do not claim invention of:

- Vector embeddings or cosine similarity
- Temporal decay or exponential decay functions
- Hebbian learning as a concept
- Blackboard architectures or shared workspaces in general
- Semantic search or approximate nearest neighbor retrieval
- Vector databases (Qdrant, Pinecone, etc.)
- Multi-agent systems or agent orchestration in general
- The concept of shared memory for software agents

We do not claim:

- Universal superiority over all multi-agent coordination approaches
- That message passing is obsolete or broken in all contexts
- Academic proof of effectiveness across diverse workloads
- Exhaustive prior-art search (counsel should conduct formal search)
- That any single element of this system is novel in isolation

## Claim Scope Assessment

**Broadest defensible scope:** The specific combination of mission-scoped shared vector field + resonance scoring (cos^2 x decay x reinforcement) + mission lifecycle binding + agent-callable tools, applied to multi-agent LLM coordination.

**Narrowest fallback scope:** The resonance scoring formula (cos^2 x temporally-decayed strength x access boost with co-access bonding) applied to a shared multi-agent knowledge field, with content-hash deduplication using reinforce-on-collision.

**Counsel should advise on:** Whether the broadest scope survives a formal prior-art search, and which dependent claims add the most defensive value.
