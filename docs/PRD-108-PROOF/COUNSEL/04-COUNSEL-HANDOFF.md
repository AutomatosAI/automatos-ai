# Counsel Handoff

## What to File Around

A provisional patent application for the specific combination of:

1. A mission-scoped shared semantic vector field for multi-agent LLM coordination
2. Resonance-based retrieval scoring (cos^2 x temporally-decayed strength x access reinforcement)
3. Hebbian co-access reinforcement between co-retrieved patterns
4. Content-hash deduplication with reinforce-on-collision
5. Field stability metric for convergence monitoring
6. Mission lifecycle integration (create/seed/inject/destroy)
7. Agent-callable platform tools for direct field interaction
8. Swappable backend interface enabling A/B comparison against message-passing baseline

The claim is originality in the combination, not in any individual primitive.

## What to Avoid Saying

| Do Not Say | Say Instead |
|-----------|-------------|
| "First ever shared memory for AI agents" | "Novel combination of elements for multi-agent LLM coordination" |
| "Eliminates information loss" | "Reduces information loss in controlled testing" |
| "Proven superior to message passing" | "Shows directional advantage over message passing in controlled scenarios" |
| "Nobody else does this" | "No known system combines these specific elements" |
| "Telephone game is dead" | "Addresses the information-loss problem in sequential agent pipelines" |
| "Patented technology" | "Patent-pending" (only after filing) |

## Top Risks

### Risk 1: CrewAI Shared Memory (Medium)

CrewAI documents crew-level shared memory with semantic retrieval. Their public documentation does not describe resonance scoring, Hebbian reinforcement, co-access dynamics, lifecycle binding, or stability metrics. However, their implementation is open source and may contain undocumented features.

**Mitigation:** Counsel should review CrewAI's source code (especially memory module) to confirm no overlap with claim elements E2-E7.

### Risk 2: Mitra Field-Theoretic Memory (Medium-High for broad claims)

ArXiv paper 2602.21220 (January 2026) proposes field-based memory with decay for AI agents. Published ~7 weeks before our implementation. Key differences: single-agent (not multi-agent shared), PDE-based diffusion (not direct vector search with resonance), no mission lifecycle, no agent-callable tools, no production deployment.

**Mitigation:** Narrow claims to multi-agent shared coordination (not single-agent memory). Cite Mitra as prior art and differentiate explicitly.

### Risk 3: Blackboard Architecture Ancestry (Low)

The shared workspace concept has 40+ years of prior art. Our claims must be specific enough that "shared workspace" alone doesn't anticipate them.

**Mitigation:** Claims focus on the semantic vector field with specific dynamics, not the general concept of a shared workspace.

### Risk 4: Deterministic Test Embeddings (Presentation Risk)

Local A/B tests use bag-of-words deterministic embeddings, not real neural network embeddings. This is standard practice for algorithmic verification but could be mischaracterized as "fake results" if not properly disclosed.

**Mitigation:** Already addressed in evidence chain (03-EVIDENCE-CHAIN.md). Tests are clearly labeled as mechanism verification, not production benchmarks. Production evidence uses real Qwen embeddings.

### Risk 5: AI Co-Development (Low, if properly disclosed)

Implementation was developed with assistance from Claude (Anthropic). Per USPTO guidance (February 2024, "Inventorship Guidance for AI-Assisted Inventions"), AI-assisted inventions are patentable if a natural person made significant contributions to conception.

**Mitigation:** Inventor declaration includes AI assistance disclosure. All inventive concepts (architecture, formula, combination) were conceived by the named inventor.

## Realistic Claim Scope Assessment

**Most likely to survive examination:** The combination claim — mission-scoped shared vector field + resonance scoring + Hebbian dynamics + lifecycle integration, applied to multi-agent LLM coordination. No known prior art combines all of these.

**Likely to need narrowing:** Any claim that could be read as "shared memory for AI agents" broadly — CrewAI and others have versions of shared memory.

**Strongest independent element:** The resonance scoring formula (cos^2 x decay x access boost) applied to a shared multi-agent field. No known system uses this specific formulation.

**Strongest dependent element:** Co-access strength bonus with stored strength mutation. No known prior art for this specific mechanism.

## Questions for Counsel

1. Does the combination of elements E1-E10 survive a formal prior-art search as a system claim?
2. Should the resonance formula (E2) be filed as a separate independent claim or only as part of the system?
3. Does Mitra (2602.21220) create any issue for multi-agent claims, given it's single-agent only?
4. Is CrewAI's shared memory implementation close enough to require narrowing E1?
5. Should co-access bonus (E5) and content-hash dedup (E6) be independent claims? They appear to have no prior art.
6. Is the production mission evidence sufficient for reduction to practice, or do we need more runs?
7. Should we file in the US only, or consider PCT for international coverage?

## Reading Order for Counsel

Estimated reading time: 25 minutes.

| Order | Document | Time | Purpose |
|-------|----------|------|---------|
| 1 | This document (04-COUNSEL-HANDOFF.md) | 5 min | Orient to the filing |
| 2 | 00-INVENTION-STATEMENT.md | 3 min | What the invention is |
| 3 | 01-CLAIM-BOUNDARY.md | 5 min | What to claim and not claim |
| 4 | 02-PRIOR-ART-MATRIX.md | 7 min | What's out there and what survives |
| 5 | 03-EVIDENCE-CHAIN.md | 5 min | What we can prove and what we can't |

If counsel wants deeper technical detail:
- `docs/PRD-108-TECHNICAL-DISCLOSURE.md` — full formal disclosure
- `docs/PRD-108-ALGORITHMS.md` — all 15 algorithms with math

## Inventor Information

| Field | Value |
|-------|-------|
| Name | Gerard Kavanagh |
| Residence | Ireland |
| Entity status | Micro entity |
| Repository | Private GitHub (AutomatosAI/automatos-ai) |
| Platform | Automatos AI (automatos.app) |
| Date of conception | March 15, 2026 |
| Date of first reduction to practice | March 21, 2026 |
| First public disclosure | None prior to filing |
