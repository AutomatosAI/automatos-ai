# PRD-159 — Memory Quality & Lifecycle (WS-4)

**Chain:** Block A, branch `ralph/prd-159-memory-quality` from main after Night-1. Size **M**. **Blocks PRD-164.**
**Source:** report §2.2, §3; PRD-154 S3 landed `infer:false`, bulk delete, decay retune, touch-on-recall.

## Overview

D5 BINDING: the orchestrator owns extraction; mem0 is storage/search/dedup only. This PRD rewrites what gets remembered (operational taxonomy, the owner's core ask: *volume of real, operational memories*), how it consolidates, and how it's recalled — ending the "user said hello" era.

## Binding amendments

D5, D11 (distiller on a cheap model; ~1 LLM call/turn target vs 3 today), Q9 priority order: tool-execution outcomes → mission/task failure learnings → playbook patterns → user/business facts, Q10: the PRD-131d operational-exclusion is DELETED (typed taxonomy replaces the blanket ban), Q11 default: L2 lifespan weeks + mission/task memories don't decay until consolidated, Q14 default: widget-customer memories isolated from operator Explorer, Q16: cheap-model distiller.

## User Stories

### S1: Operational distill prompt + typed facts
Rewrite the distill prompt (`smart_memory.py:578-613`): DELETE the operational-exclusion lines; emit typed `{fact, type, importance}` with taxonomy `{tool_outcome, task_learning, playbook_pattern, user_fact, business_fact, preference, procedure}` (Zep ontology incl. `procedure`). Distiller routed to the cheap model via `create_llm_manager` config. Raw-exchange fallback (`:456-462`) is deleted — on distill failure, store nothing and log (no junk path).
**Acceptance:**
- [ ] Prompt-eval test set: 12 transcript fixtures → expected typed facts (golden test, allows fuzzy match on wording, strict on type + presence)
- [ ] 'user said hello' class fixtures produce ZERO writes
- [ ] Distiller model assertion test (cheap tier)
- [ ] Existing distill tests updated; `pytest -q` green

### S2: Tool-execution outcome capture (Q9 #1)
Post-execution hook in the unified executor: failures and notable successes (new channel IDs, auth quirks, rate limits, schema surprises) become typed `tool_outcome` memories under the workspace namespace — written direct (`infer:false`), deduped by content-hash.
**Acceptance:**
- [ ] Failed Composio call produces a tool_outcome memory with app+action+error class (test)
- [ ] Hash-dedup: identical outcome twice → one row (test)
- [ ] No write on trivial successes (noise gate test)
- [ ] Protected hint-service suite (25 tests) untouched and green

### S3: Recall that actually fires
Default recall includes L2 + recipe/workflow namespaces (`memory_stats.py:66-101`, `smart_memory.py:213-230`, `context_router.py:489-509`): semantic search over operational types without the temporal-regex gate; relevance floor applied (server threshold, Q-default); ordered+paginated daily logs (`mem0_client.py:466-496` sends size/sort).
**Acceptance:**
- [ ] 'why did the deploy mission fail?' recalls a seeded task_failure memory (integration test)
- [ ] Playbook learnings visible in Explorer and recalled outside step-1 of the same playbook
- [ ] Low-relevance junk below floor never injected (test)
- [ ] Protected recipe suite (20 tests) green

### S4: Consolidation replaces decay-deletion
Sleep-time consolidation (message-count/idle-triggered job replacing the dead cron + dead `end_session` path): merge near-duplicates, resolve contradictions by recency+confidence (contradiction-based invalidation instead of the 15h time-decay as the primary lifecycle), promote stable L2 → L3, archive losers. Delete the dead L1 session machinery (`end_session`, `decisions/action_items` columns) — no shims.
**Acceptance:**
- [ ] Consolidation test: 5 near-dup memories → 1 canonical with provenance
- [ ] Contradiction test: newer fact supersedes, older archived with reason
- [ ] Promotion test: accessed L2 reaches L3 via the job
- [ ] Dead session paths removed; suite green

### S5: Honest events + Explorer truth
`memory_stored` SSE fires only after persistence with the real tier (`service.py:1099-1107` + `smart_orchestrator.py:384-434` race); Explorer reflects UPDATE/DELETE events (openmemory `memories.py:432-433` ADD-only sync fixed in the fork); vector-store metadata carries workspace/tier/category (fork `:418-421`); 'both'-tier double-write default removed (single namespace + explicit agent-tier triggers).
**Acceptance:**
- [ ] SSE test: no event when distill yields zero facts; tier matches what persisted
- [ ] Fork sync test: UPDATE/DELETE reflected in SQL view
- [ ] Metadata filterable by tier in semantic search (test)
- [ ] Explorer duplicate count drops to zero for new writes (test)

### S6: mem0 V3 fork upgrade (structural follow-up)
Upgrade the fork to mem0 V3 baseline so S4/S5 patches ride a current base; document the patch set (auth dep from PRD-156, sync fixes, metadata) as fork-maintenance notes.
**Acceptance:**
- [ ] Fork tests green on V3; orchestrator integration suite green against the new image
- [ ] Patch-set doc in automatos-mem0 README

## Non-Goals

Field memory (166), planning context packs + heartbeat-agent memory (164), widget memory UX.

## Success Metrics

- 7-day pilot window: ≥50 typed operational memories/workspace/week (volume), with ≥80% rated useful in a 20-sample manual audit (quality) — measured via memory_stats.
- Memory LLM cost/turn ≤ 1 distill call on the cheap tier (vs 3 calls today).
- Zero 'transient interaction event' memories in the audit sample.

## Testing

New golden distill suite, consolidation suite, recall integration tests; fork test additions. Updated: decay/lifecycle tests from PRD-154 S3, SSE tests. Protected: recipe (20) + hint (25) suites explicitly in the acceptance gate. Full suite + contract green.
