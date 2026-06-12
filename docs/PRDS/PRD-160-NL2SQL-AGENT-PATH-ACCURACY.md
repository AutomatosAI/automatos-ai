# PRD-160 — NL2SQL Agent Path & Accuracy Stack (WS-5)

**Chain:** Block B, branch `ralph/prd-160-nl2sql-accuracy` from main after Night-1 (needs PRD-156 S3's scoping). Size **M**.
**Source:** report §2.3; PRD-154 S12 revived the broken tabs; PRD-156 S3 disabled the unsafe paths.

## Overview

NL2SQL becomes a first-class, workspace-safe Auto tool with a real accuracy stack: scoped in-process execution, few-shot retrieval over verified pairs, AST validation, dry-run + bounded self-correction, and a regression harness — the Vanna/WrenAI patterns adapted to the existing `DatabaseKnowledgeService`.

## Binding amendments

Q59: yes — first-class Auto tool once safe (this PRD is "safe"), Q-defaults from §5: semantic layer starts as admin-instructions v1 (MDL-lite later), Query Templates tab gets implemented (not deleted) on the existing tables, training loop = verified-pair few-shot (no fine-tuning).

## User Stories

### S1: In-process scoped agent path
`smart_query_database` calls `DatabaseKnowledgeService.smart_query` in-process (DELETE the HTTP self-call), threading workspace + connection scoping (PRD-156 S3 base); honors `database_name`; re-enable in intent-classifier suggested_tools.
**Acceptance:**
- [ ] No `httpx` self-call remains (grep gate); cross-workspace matrix test green
- [ ] Auto answers a seeded NL question via the tool in integration test

### S2: Accuracy stack
sqlglot AST validator (SELECT-only, table allowlist from connection scope); `EXPLAIN` dry-run + `statement_timeout`; bounded self-correction retries=2 feeding the error back; low-cardinality value sampling into prompt context.
**Acceptance:**
- [ ] Validator suite: UPDATE/DELETE/DDL/multi-statement → rejected; cross-schema reference → rejected
- [ ] Self-correction test: seeded bad-column question recovers within 2 retries
- [ ] Timeout test: runaway query bounded

### S3: Training loop that trains
Persist question-SQL pair embeddings to pgvector (today computed then discarded — `example_store.py:90-101`); few-shot retrieval of top-k verified pairs into generation; thumbs-up in SQL Explorer marks a pair verified; Training tab manages the pair library.
**Acceptance:**
- [ ] Golden test: question similar to a verified pair includes it few-shot and improves the eval (S5 harness)
- [ ] Verified-pair CRUD via Training tab (API + dev-browser)

### S4: Semantic layer v1 + audit truth
Admin-instructions semantic layer (per-connection business definitions injected into generation) on the existing tables; NL-path writes audit rows (fills the Audit History tab that PRD-154 revived visually); shared source selector across all six tabs; implement Query Templates execute path (the button that could never succeed).
**Acceptance:**
- [ ] Definitions measurably steer generation (golden test)
- [ ] Every NL query lands one audit row with workspace, agent, SQL, outcome (test)
- [ ] Template execute round-trips — dev-browser verify

### S5: Eval harness
sql-eval-style regression suite over the verified-pair library against a seeded schema; runs in CI as a non-required job (test-net precedent); accuracy reported in PR.
**Acceptance:**
- [ ] `pytest tests/nl2sql_eval -q` produces an accuracy score; baseline recorded; gate = no regression vs baseline

## Non-Goals

MDL/full semantic-layer modeling language, write-query support, cross-database joins, fine-tuning.

## Success Metrics

- Eval accuracy ≥ baseline+15pts after S2+S3 on the 30-question seeded set.
- 100% of NL queries audited; zero unscoped executions (matrix).
- All six Databases tabs fully functional against real backends.

## Testing

New validator/eval/training suites; updated tab API tests from PRD-154 S12. Full suite + contract green.
