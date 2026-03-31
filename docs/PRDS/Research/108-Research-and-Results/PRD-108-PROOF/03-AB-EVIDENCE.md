# PRD-108 A/B Evidence

## Purpose

This document repackages the current PRD-108 evidence into an investor-safe empirical summary.

It is intentionally narrower than a paper. It should be read as:

- evidence that the mechanism is real
- evidence that the mechanism improves retrieval in a controlled scenario
- not proof of universal superiority across all missions

## Evidence Sources

Primary sources:

- `automatos-ai/docs/PRD-108-ALGORITHMS.md`
- `automatos-ai/docs/PRD-108-IMPLEMENTATION.md`
- `automatos-ai/docs/PRD-108-TECHNICAL-DISCLOSURE.md`

Executable references:

- `automatos-ai/orchestrator/tests/test_vector_field.py`
- `automatos-ai/orchestrator/tests/demo_field_stress.py`
- `automatos-ai/orchestrator/tests/demo_ab_comparison.py`

## Evidence Layer 1: Implementation Verification

The core mechanism is documented as implemented and traceable to code:

- `SharedContextPort` defines the common interface
- `VectorFieldSharedContext` implements the PRD-108 field
- `RedisSharedContext` provides the message-passing baseline
- the mission coordinator creates, seeds, uses, and destroys the field
- field tools are exposed to agents during execution

This matters because the A/B comparison is not just conceptual. It is tied to a real baseline/backend swap.

## Evidence Layer 2: Unit and Stress Results

### Unit tests

The source docs report 57 passing unit tests covering:

- decay math
- create/destroy lifecycle
- inject with deduplication
- query ranking
- Hebbian reinforcement
- stability measurement
- helper behavior

### Stress and integration-style assertions

The source docs report 16 passing assertions, including:

- resonance ranking
- 24h decay from `1.0000` to `0.0907`
- reinforcement advantage for frequently accessed patterns
- archival threshold behavior
- stability changes under mixed ages
- 150-pattern / 50-agent scenario
- controlled cross-agent visibility scenario

## Evidence Layer 3: Controlled A/B Demonstration

The strongest current comparative evidence is the script:

- `automatos-ai/orchestrator/tests/demo_ab_comparison.py`

The script runs the same scenario against two backends:

1. `run_vector_field(...)`
2. `run_redis_baseline(...)`

### Experimental setup

- 3-agent mission scenario
- 10 research findings
- 3 analyses
- 7 downstream queries from Agent C
- same inputs for both backends

Important limitation:

- the script uses `fake_embed(...)` with `DIM = 128` and word-overlap semantics for the demo
- this means the A/B is a controlled mechanism test, not a production-grade benchmark with full embedding stack realism

That is acceptable for early proof, but it should be disclosed clearly.

## Reported Comparative Results

From `PRD-108-ALGORITHMS.md` as documented in the March 21, 2026 PRD-108 evidence set:

| Metric | Vector Field | Message Passing |
|--------|-------------|-----------------|
| Context coverage | 86% (6/7) | 43% (3/7) |
| Information loss | 1 finding | 4 findings |
| Patterns visible to Agent C | 13 | 6 |

Interpretation:

- the vector field preserved broader downstream visibility
- the baseline exposed only what Agent B forwarded
- the vector field answered more of Agent C's downstream needs

## What This Evidence Supports

The current A/B evidence supports these claims:

1. A shared semantic field can reduce information loss caused by sequential forwarding.
2. Backend swapping through a common interface enables direct comparison with a control.
3. The field design improves downstream retrieval in the tested scenario.

## What This Evidence Does Not Yet Prove

The current evidence does not yet prove:

1. superiority across all mission types
2. superiority across all embedding models
3. production-grade latency/performance in realistic networked deployment
4. end-to-end business outcome improvements across many workflows

## Investor-Safe Summary

The right way to describe the A/B result is:

> In a controlled three-agent comparison using the same scenario and a shared interface, the PRD-108 vector-field backend recovered materially more downstream-relevant information than the message-passing baseline.

The wrong way to describe it is:

> PRD-108 has conclusively proven that message passing is obsolete.

## Strongest Current Evidence Sentence

PRD-108 already has early comparative evidence that its coordination model reduces downstream information loss relative to a simpler message-passing baseline in a controlled scenario.

## Next Validation Step

The next credibility upgrade should be a broader A/B package with:

- more mission types
- real embedding model configuration
- explicit scoring rubric
- raw outputs
- reproducible command log

That would turn this from promising internal evidence into advisor-grade comparative validation.
