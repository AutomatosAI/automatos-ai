# PRD-164 — Planning Intelligence & Integration Seams (WS-9)

**Chain:** Payoff PRD. Branch `ralph/prd-164-planning-intelligence` from main only after **157, 159, 163, 166** are merged. Size **L**.
**Source:** report §3.1 (seams), §4 WS-9. This is where the platform starts to FLOW — everything before it built the read-side it consumes.

## Overview

The five highest-leverage integrations: every planner consults what the platform knows (RAG+memory+graph+roster); agent selection becomes semantic; agent outputs feed back into knowledge (D6 flywheel); chat renders platform actions as real widgets; recurring agents learn.

## Binding amendments

D5/D6/D7/D11; Q58: flywheel ON by default with per-workspace opt-out, `source_type='agent_output'`, agent-outputs surfaced as a filterable team-like scope, Q60: heartbeat agents GET memory (statelessness ends — scoped recall + write hooks), Q61: planners converge on one ContextService planning mode (AutoBrain, MissionPlanner, board `plan_task`), Q62: `TOOL_WIDGET_MAP` dead router deleted, replaced by live widget routing, Q21 default: agent selection = blend (embeddings of capability cards + mission-memory history + live field at dispatch; one embedding call per dispatch accepted), Q22: upstream stuffing replaced by field digest with per-task budget.

## User Stories

### S1: Planning Context Pack
One ContextService "planning" mode: RAG retrieval on the goal (157 path), top-k mission summaries + task failures (159 recall), KG subgraph (165 endpoint), roster + performance; consumed by MissionPlanner (`planner.py:699`), board `plan_task` (`board_tasks.py:824`), and AutoBrain. Token-budgeted per D11.
**Acceptance:**
- [ ] Golden test: seeded prior-mission failure visibly changes a new plan (the learning demo)
- [ ] All three planners consume the same pack (no parallel assemblies — grep gate)
- [ ] Pack stays within budget on oversized fixtures (test)

### S2: Semantic agent matching
AgentMatcher: embedding capability cards (Qdrant) + historical task performance from mission memory + live field signal (166) → ranked agents with reasons; honors explicit `agent_overrides` (163 S4).
**Acceptance:**
- [ ] Golden matrix: 10 task fixtures → expected top-agent (allowing ties)
- [ ] Override always wins (test)
- [ ] Match reasons logged on the task for the approval card

### S3: Output flywheel (D6)
Mission syntheses, generated documents, submitted reports route through the existing ingestion manager (chunked, searchable, graph-extracted); KG incremental build learns the three source types it drops today; deliverables get list/get agent tools + a mission-page tab; mission completion can emit `generate_document` (template path from 167 when merged — optional dependency, feature-flag if 167 unmerged).
**Acceptance:**
- [ ] Completed mission's synthesis retrievable via RAG in the next chat turn (e2e test)
- [ ] KG contains entities from a seeded report (test)
- [ ] Workspace opt-out respected (test)
- [ ] Auto lists its own deliverables via tool (reachability + integration test)

### S4: Field-digest dispatch + replanning guards
Replace the 8K-char upstream-output stuffing with the 166 field digest (per-task budget, Q22); bounded replanning (LLMCompiler-style joiner) + progress ledger/stall counter (Magentic-One pattern) on mission execution.
**Acceptance:**
- [ ] Dispatch prompt size drops ≥60% on the multi-task fixture while the golden task still passes
- [ ] Stall test: induced loop triggers replan-or-halt within bounds, with audit trail

### S5: Chat renders the platform
Delete dead `TOOL_WIDGET_MAP` (`frontend/components/widgets/router.ts`); widget routing driven by live tool names for: mission cards (163 S4 base), board tasks, deliverables, documents, schedule, memory writes; heartbeat agents get scoped memory recall + write hooks (Q60).
**Acceptance:**
- [ ] Each listed tool result renders its widget in chat — dev-browser verify
- [ ] Stale-name drift impossible: widget keys validated against the registry in the PRD-155 reachability test
- [ ] Heartbeat run recalls a memory written by its previous run (test)

## Non-Goals

New planner algorithms beyond joiner/ledger; cross-workspace intelligence; template editor internals (167).

## Success Metrics

- The demo: ask Auto to plan a mission in a workspace with relevant docs + a prior failed mission → the plan cites the doc, avoids the failed approach, picks the right agent, and the approval card explains why.
- Dispatch token cost per task down ≥60% with unchanged golden-task pass rate (D11's "capped" half).
- Flywheel: agent outputs appear in retrieval within one ingestion cycle.

## Testing

Golden planning suite (fixtures with expected plan deltas), matcher matrix, flywheel e2e, widget dev-browser pass. **Protected: recipe (20) + hint (25) suites are explicit gates** — this PRD touches their neighborhoods. Full suite + contract green.
