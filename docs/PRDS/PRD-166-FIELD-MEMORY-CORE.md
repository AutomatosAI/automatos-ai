# PRD-166 — Field Memory Core (WS-11)

**Chain:** Block B, branch `ralph/prd-166-field-memory-core` from main after PRD-165 (uses the GraphView shell). Size **M**. **Feeds PRD-164** (distillation + dispatch digest).
**Source:** report §2.4; PRD-154 S7 landed the viz swap, archive-don't-destroy, `_agent_id` fix.

## Overview

D7 BINDING: the field becomes workspace-persistent memory that compounds across missions — scored honestly, budgeted honestly, visualized on the shared shell, with the zombie surfaces deleted.

## Binding amendments

D7, D11 (field query token budgets are config, not hardcoded caps), Q17 (c) workspace-persistent, Q18 (react-force-graph on React 18 — landed PRD-154), Q19 default: graph primary + retrieval-trace inspector secondary (Letta-style) in the detail panel, Q20: field tools mission-gated (fail with a clear message outside missions until workspace fields exist — then workspace-scoped), Q29-redux default: redis backend decision deferred — Qdrant stays, measured first.

## User Stories

### S1: Workspace-persistent field lifecycle
Mission fields merge into a per-workspace field at terminal (instead of only archiving): episode payloads carry provenance (mission_id, task_id, agent_id, timestamps); `expired_at` soft-archive for mission-scoped views; retention/compaction job (decay + consolidation keeps Qdrant bounded).
**Acceptance:**
- [ ] Two sequential seeded missions → workspace field contains both, provenance intact (test)
- [ ] Compaction test: low-stability patterns pruned, high-stability survive
- [ ] Completed-mission Field tab renders the frozen mission view (PRD-154 base) AND links to the workspace field — dev-browser verify

### S2: Honest scoring + budgeted query
Three-factor scoring (similarity × stability × recency) with adaptive half-life; token-budgeted query API returning `{patterns, truncated: bool}` — no silent caps (D11: budget from config); real field health check (replace hardcoded 'healthy').
**Acceptance:**
- [ ] Scoring unit suite with golden rankings
- [ ] Budget test: oversize result sets flag `truncated` and chunk correctly
- [ ] Health check reflects an induced Qdrant outage (test)

### S3: Dispatch integration
Pinned field digest in task dispatch prompts (replaces nothing yet — PRD-164 swaps the 8K upstream stuffing); budget-gate checkpoint warning when field context is dropped for budget.
**Acceptance:**
- [ ] Dispatched task prompt contains the digest within budget (test)
- [ ] Warning event emitted when dropped (test)

### S4: Viz on the shared shell + inspector
Field viz consumes the PRD-165 GraphView shell (node+edge data — edges were missing); detail panel = retrieval-trace inspector (which patterns fired for which query, Letta-style); delete the `/field-theory` zombie page tree + its api-client methods (PRD-154 S7 began; finish: zero references).
**Acceptance:**
- [ ] Live mission + workspace field render via the shell — dev-browser verify
- [ ] Inspector shows the trace for a test query — dev-browser verify
- [ ] `/field-theory` route + `api-client.ts:2466-2496` methods gone; contract green

## Non-Goals

Semantic agent matching (164), mission planner changes (163/164), redis backend swap.

## Success Metrics

- Workspace field demonstrably influences a second mission's task context (golden e2e).
- Field tab works for live AND completed missions; zero blank-tab states.
- Qdrant size bounded across a 20-mission soak (compaction works).

## Testing

Scoring/budget unit suites, lifecycle integration tests, soak script; vitest for inspector. Updated: PRD-154 S7 archive tests extended to merge semantics. Full suite + contract green.
