# Invention Statement

## Plain-English Summary

When multiple AI agents work together on a task, they currently pass messages to each other in a chain — like the telephone game. Each agent in the chain decides what to forward and what to drop. Information gets lost at every step. We built a shared memory space where every agent puts its work and any agent can search it by meaning. Instead of asking "what did the last agent tell me," an agent asks "what does anyone know about X?" and gets answers ranked by relevance, freshness, and how often that knowledge has been useful.

## Technical Summary

The system creates a mission-scoped vector database collection for each multi-agent task. Agents inject knowledge as high-dimensional embedding vectors and query by semantic similarity. Results are ranked by a resonance function: squared cosine similarity multiplied by exponentially-decayed strength and access-based reinforcement. The field is created when a mission starts, seeded with the mission goal, populated as tasks complete, and destroyed when the mission ends. A swappable backend interface allows the same orchestration code to run against either the vector field or a conventional message-passing baseline.

## Invention Elements

1. **Mission-scoped shared vector field** — one collection per mission, not per agent or per session, bound to mission lifecycle
2. **Resonance scoring** — R = cos^2(theta) x S_0 x e^(-lambda*t) x min(1 + 0.05n, 2.0), combining semantic similarity, temporal decay, and usage reinforcement in a single ranking function
3. **Temporal decay** — exponential decay from last access time (not creation time), computed at query time without mutating stored data
4. **Hebbian reinforcement** — access count boosts effective strength; co-retrieved patterns receive mutual strengthening ("fire together, wire together")
5. **Content deduplication with reinforce-on-collision** — SHA-256 hash check before injection; duplicates strengthen the existing pattern instead of creating clutter
6. **Field stability metric** — weighted combination of mean decayed strength and strength distribution uniformity, indicating convergence or fragmentation
7. **Agent-callable platform tools** — agents choose when to query, inject, and measure the field during execution (not forced by the system)
8. **Swappable backend interface** — SharedContextPort ABC with identical method signatures for vector field and message-passing implementations, enabling controlled A/B comparison

## Commercial Relevance

Multi-agent AI systems are a growing market segment (Microsoft AutoGen, CrewAI, LangGraph, OpenAI Swarms, Google A2A). All current platforms use some form of message passing for inter-agent coordination. This architecture provides a measurably different approach that addresses a known weakness (information loss in sequential pipelines). The implementation is integrated into a working commercial platform (Automatos AI) and has been deployed to production. A provisional patent would establish a priority date for the specific combination of elements described above.
