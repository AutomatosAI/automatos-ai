# PRD-108 Claim Memo

## Purpose

This memo defines the strongest investor-safe claim that can be supported today by the PRD-108 materials and referenced implementation.

## Core Claim

PRD-108 is a differentiated orchestration approach for multi-agent LLM systems:

- each mission gets a shared semantic field
- agents inject and query by meaning rather than relying on forwarded summaries
- retrieval is ranked by resonance, not just raw similarity
- relevance changes over time through decay and reinforcement
- the field is tied to mission lifecycle events
- the backend is swappable through a shared interface, enabling direct A/B comparison with a simpler message-passing baseline

## Best Defensible Statement

PRD-108 appears to be a novel combination for multi-agent LLM coordination: a mission-scoped shared semantic field with resonance-based retrieval, reinforcement dynamics, lifecycle integration, and direct comparability against a message-passing baseline through a common context interface.

## What We Are Claiming

We are claiming originality at the level of **system design and implementation combination**, specifically:

1. A **mission-scoped shared field** rather than per-agent or generic session memory.
2. A **shared abstraction layer** via `SharedContextPort`, allowing the same orchestration code to run against:
   - `RedisSharedContext`
   - `VectorFieldSharedContext`
3. A specific retrieval/ranking design:
   - cosine-similarity squared
   - query-time temporal decay
   - access-based reinforcement
   - co-access strengthening
   - archival filtering
4. **Mission lifecycle integration**:
   - create on mission start
   - seed with mission goal
   - inject task outputs during execution
   - destroy and garbage-collect on terminal runs
5. **Agent-callable field tools** exposed during execution:
   - `platform_field_query`
   - `platform_field_inject`
   - `platform_field_stability`

## What We Are Not Claiming

We are not claiming to have invented:

- vector embeddings
- cosine similarity
- blackboard systems
- temporal decay
- Hebbian learning
- stigmergic reinforcement
- semantic search in general

We are also not claiming:

- universal superiority over all multi-agent architectures
- academic proof across diverse workloads
- definitive patentability
- exhaustive prior-art closure

## Why This Matters

The differentiation story is not "we invented memory." It is:

- we built a specific coordination substrate for multi-agent LLM work
- we integrated it into an existing orchestration stack
- we made it measurable against a baseline
- we produced early evidence that it reduces information loss in a controlled scenario

That combination is materially stronger than a generic "shared vector DB" claim.

## Evidence Already Available

### 1. Working architecture

- The interface exists in `automatos-ai/orchestrator/core/ports/context.py`.
- The vector-field implementation exists in `automatos-ai/orchestrator/modules/context/adapters/vector_field.py`.
- The simpler baseline exists in `automatos-ai/orchestrator/modules/context/adapters/redis_context.py`.

### 2. Coordinator integration

`automatos-ai/orchestrator/services/coordinator_service.py` shows:

- field creation per mission
- seeding with mission goal
- task-output injection
- terminal cleanup

### 3. Agent tools

`automatos-ai/orchestrator/modules/tools/discovery/actions_field.py` defines:

- `platform_field_query`
- `platform_field_inject`
- `platform_field_stability`

### 4. Tests and demos

The PRD-108 documents cite:

- 57 unit tests
- 16 stress-test assertions
- a controlled A/B demo script

Those references align with real files under `automatos-ai/orchestrator/tests/`.

## Current Boundaries

The evidence today most strongly supports:

- mechanism works
- implementation is real
- differentiation is concrete
- the controlled A/B result is promising

The evidence does not yet support:

- broad claims of production-wide superiority
- generalizable performance claims across mission types
- market-wide uniqueness without more prior-art review

## Recommended External Positioning

Use language like:

> PRD-108 is a working, mission-scoped coordination architecture for multi-agent LLM systems. It combines shared semantic retrieval, temporal relevance dynamics, and lifecycle-aware orchestration in a way that is directly testable against message passing. Our claim is originality in the combination and implementation, not invention of each underlying primitive.

Avoid language like:

- "first ever"
- "nobody else does this"
- "proved the telephone game is dead"
- "this conclusively beats all shared-memory systems"
