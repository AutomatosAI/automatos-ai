# PRD-156 — Security & Tenancy Hardening (WS-1)

**Chain:** Night 1, stacked on `ralph/prd-155-route-contract` tip. Branch `ralph/prd-156-security-tenancy`. Size **S-M**. **Blocks the PRD-150 open-core cut.**
**Source:** report §2.1/§2.2/§2.3/§2.10 security findings, §4 WS-1.

## Overview

Closes every cross-tenant and injection hole the review confirmed, before open-core widens exposure. Constraint: PRD-09 — board SDK-key reads stay on the narrow scope-gated dependency; do NOT touch the shared hybrid auth. All identity threading stays auth-provider-agnostic (PRD-150).

## Binding amendments

Q1 fail-closed already landed (PRD-154 S2). Q6 default: multimodal `knowledge_items` gets **workspace-scoping only** here (freeze; fold-vs-invest decided later). Q15: mem0 gets token auth NOW (one FastAPI dependency in the fork, provider-agnostic). Q13 default: the mock `/api/v1/memory` surface is deleted here.

## User Stories

### S1: Workspace+team filters on multimodal search tools
All four tools (`search_tables/search_images/search_formulas/search_multimodal`, `modules/rag/services/multimodal_knowledge_tools.py:65-401`) gain mandatory `workspace_id` clauses + team filter via the shared filter builder (from PRD-154 S2 / extended in PRD-157). Also: persist the `team_access` form field that upload currently drops (`api/knowledge_multimodal.py:258`), and fix the broken similarity subquery (`WHERE content = :query` exact-match) to embed the query text properly.
**Acceptance:**
- [ ] Cross-workspace fixture test: workspace B's items never returned for workspace A (all four tools)
- [ ] Upload persists team_access; retrieval respects it
- [ ] Similarity test: novel query returns ranked results (no NULL-similarity arbitrary ordering)
- [ ] `pytest -q` green

### S2: mem0/OpenMemory token auth
The server has zero auth; the orchestrator already sends `Authorization: Token MEM0_API_KEY` (`mem0_client.py:160`) that's ignored. Add one FastAPI dependency in the automatos-mem0 fork validating the token on every router; key from env; orchestrator side unchanged. Provider-agnostic (no Clerk).
**Acceptance:**
- [ ] Fork test: requests without/with-wrong token → 401; with token → 200
- [ ] Orchestrator integration test against the patched server image stays green
- [ ] Railway env documented in the PR (chain checklist: env BEFORE deploy)

### S3: NL2SQL tenancy
Workspace scoping threaded through NL2SQL query + audit + analytics endpoints; DISABLE the `query_main_database` fallback and the unauthenticated HTTP self-call (the proper in-process path is PRD-160; until then the unsafe path is OFF, not shimmed). Remove NL2SQL from intent-classifier `suggested_tools` until PRD-160 re-enables it scoped (`smart_tool_router.py:80-84`, `intent_classifier.py:275`).
**Acceptance:**
- [ ] Cross-workspace NL2SQL test fails closed
- [ ] Grep gate: no `query_main_database` reachable from chat tool surface
- [ ] Audit/analytics endpoints scoped (test)

### S4: Document template injection + IDOR
Template CRUD gets workspace ownership checks (cross-workspace IDOR confirmed); Jinja2 rendering moves to `SandboxedEnvironment`; WeasyPrint gets a `url_fetcher` allowlist (no file:// or internal-network fetch from user templates).
**Acceptance:**
- [ ] IDOR test: workspace B cannot read/update/delete A's template
- [ ] SSTI test: `{{ cycler.__init__.__globals__ }}`-class payloads render inert
- [ ] url_fetcher test: file:// and 169.254/10.x URLs refused
- [ ] `pytest -q` green (update any template tests asserting unsandboxed behavior)

### S5: Remaining closures
Widget memory delete ownership check (`widget_memory.py:277-280`, mirror `memory_stats.py:462-470`); auth on `GET /api/documents/content` path-read (`api/documents.py:411-452`); scope RAG analytics + `document_usage` writes by workspace (`modules/rag/service.py:1007-1168`, `api/documents.py:1080,1362`); delete the mock `/api/v1/memory` router + `AdvancedMemoryManager` (`api/memory.py` fake CRUD/random stats — Q13); entity-KG endpoints stay gated (PRD-154) pending PRD-165.
**Acceptance:**
- [ ] Ownership/auth tests for each closure
- [ ] Mock memory surface gone; route-contract test (PRD-155) updated manifest proves no frontend caller remained
- [ ] `pytest -q` green

## Non-Goals

RAG retrieval quality (157), teams UX (158), NL2SQL rebuild (160), actor-identity threading everywhere (168 carries `created_by='system'` cleanup).

## Success Metrics

- Zero cross-workspace reads in the tenancy test matrix (all search surfaces).
- security-reviewer agent pass on the diff reports no CRITICAL/HIGH.
- Open-core cut (PRD-150 program) unblocked.

## Testing

New `orchestrator/tests/security/test_tenancy_matrix.py` (parametrized across surfaces), SSTI/url_fetcher tests, fork auth tests. Full suite + contract tests green. security-reviewer agent run is part of acceptance.
