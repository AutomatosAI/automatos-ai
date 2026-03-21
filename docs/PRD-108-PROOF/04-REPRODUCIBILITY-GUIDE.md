# PRD-108 Reproducibility Guide

## Purpose

This guide shows a technical advisor where to look and what to run to verify the PRD-108 claims.

It is not a full benchmark kit. It is a practical inspection and rerun path.

## Repository Areas

Core implementation:

- `automatos-ai/orchestrator/core/ports/context.py`
- `automatos-ai/orchestrator/modules/context/adapters/vector_field.py`
- `automatos-ai/orchestrator/modules/context/adapters/redis_context.py`
- `automatos-ai/orchestrator/modules/context/factory.py`
- `automatos-ai/orchestrator/modules/context/instrumentation.py`

Agent tool surface:

- `automatos-ai/orchestrator/modules/tools/discovery/actions_field.py`
- `automatos-ai/orchestrator/modules/tools/discovery/platform_executor.py`
- `automatos-ai/orchestrator/consumers/chatbot/auto.py`

Coordinator integration:

- `automatos-ai/orchestrator/services/coordinator_service.py`

Tests and demos:

- `automatos-ai/orchestrator/tests/test_vector_field.py`
- `automatos-ai/orchestrator/tests/demo_field_stress.py`
- `automatos-ai/orchestrator/tests/demo_ab_comparison.py`

Primary docs:

- `automatos-ai/docs/PRD-108-ALGORITHMS.md`
- `automatos-ai/docs/PRD-108-IMPLEMENTATION.md`
- `automatos-ai/docs/PRD-108-TECHNICAL-DISCLOSURE.md`

## Quick Inspection Checklist

### 1. Confirm the common interface exists

Open:

- `automatos-ai/orchestrator/core/ports/context.py`

Verify the four required methods:

- `create_context`
- `inject`
- `query`
- `destroy_context`

This is the basis for the "same orchestration, different backend" claim.

### 2. Confirm the vector-field backend exists

Open:

- `automatos-ai/orchestrator/modules/context/adapters/vector_field.py`

Verify these implementation features:

- Qdrant-backed collection per mission field
- payload indexes
- SHA-256 deduplication
- query-time resonance scoring
- decay and access boost
- co-access reinforcement

### 3. Confirm the baseline exists

Open:

- `automatos-ai/orchestrator/modules/context/adapters/redis_context.py`

Verify that it implements the same port with a simpler keyword/message-passing baseline.

### 4. Confirm orchestration integration exists

Open:

- `automatos-ai/orchestrator/services/coordinator_service.py`

Inspect:

- `_create_mission_field(...)`
- `_inject_task_output_into_field(...)`
- `_destroy_mission_field(...)`
- `_cleanup_terminal_fields(...)`

This is the evidence for mission lifecycle ownership.

### 5. Confirm agents have direct field tools

Open:

- `automatos-ai/orchestrator/modules/tools/discovery/actions_field.py`

Verify these actions:

- `platform_field_query`
- `platform_field_inject`
- `platform_field_stability`

## Suggested Verification Commands

Run from:

```bash
cd <repo-root>/automatos-ai/orchestrator
```

### Install dependencies if needed

```bash
pip install -r requirements.txt
```

Important dependency called out in the docs:

- `qdrant-client>=1.12.0`

### Run the unit tests for PRD-108

```bash
python -m pytest tests/test_vector_field.py -q
```

What this should verify:

- decay math
- inject/dedup behavior
- resonance ranking logic
- reinforcement behavior
- stability computation

### Run the A/B demonstration

```bash
python tests/demo_ab_comparison.py
```

What this should print:

- vector field vs redis baseline header
- context coverage comparison
- information loss comparison
- verdict section

### Run the stress / demo script

```bash
python tests/demo_field_stress.py
```

What this is intended to show:

- resonance and ranking behavior
- decay and reinforcement behavior
- archival behavior
- multi-agent pattern counts

## Evidence Chain

The PRD-108 source docs claim this evidence chain:

- specification completed
- implementation committed
- unit tests passing
- stress assertions passing
- technical disclosure written

For external review, the most credible check is:

1. inspect the code paths above
2. rerun the tests
3. rerun the A/B script
4. compare the outputs to the documented claims

## What a Technical Advisor Should Conclude

After following the steps above, a reasonable reviewer should be able to conclude:

- there is a real implementation
- it is integrated into the orchestration layer
- it is directly comparable to a simpler baseline
- the claimed mechanics are supported by executable tests and demos

## What Still Needs More Work

This repro path does not yet establish:

- broad production benchmark coverage
- third-party replication
- exhaustive novelty proof
- externally audited performance claims

Those are follow-on validation steps, not prerequisites for establishing that the architecture is real and differentiated.
