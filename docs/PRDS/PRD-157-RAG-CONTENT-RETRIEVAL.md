# PRD-157 — RAG Content & Retrieval Quality (WS-2)

**Chain:** Block A, branch `ralph/prd-157-rag-retrieval` from main after Night-1 merge. Size **M**. **Blocks PRD-164.**
**Source:** report §2.1, §3; PRD-154 S1/S2 landed the hydration + fail-closed base.

## Overview

Agents must actually read what the platform knows. This PRD finishes the retrieval path: one scoping choke point for every search surface, document-reading tools, citation-grade context assembly with token budgets instead of char-cuts, and deletion of the placebo/dead RAG surfaces.

## Binding amendments

D1 (Teams enforcement at retrieval), D11 (token-budgeted assembly; full-doc reads allowed within budget — Q8 default: capped excerpts with paged continuation, map-reduce summarize above cap), Q-defaults: Q4 suppress human doc-widgets outside agent scope, Q7 rebuild widget/SDK docs on the real schema + vector path after prod-schema verification.

## User Stories

### S1: Centralized retrieval filter builder
`build_retrieval_filters(agent|context) → {workspace_id, team_terms}` in one module; routed through ALL search paths: RAG retrieve, UI semantic search (`api/documents.py:935`), RAG-test endpoint, multimodal tools (from PRD-156 S1), KG retrieval, future NL2SQL. Fail-closed everywhere. Optionally mirror team ACL into S3 filterable metadata (`__PUBLIC__` sentinel) as pre-filter defense-in-depth.
**Acceptance:**
- [ ] Single grep-able choke point; tenancy matrix test (PRD-156) extended to every path through the builder
- [ ] Team normalization (lowercase) applied at the builder so 'Support'/'support' match

### S2: `read_document` + `grep_documents` agent tools
Paged full-content reading (Letta pattern): `read_document(id, page|offset)` over the existing `/api/documents/{id}/content` service path, `grep_documents(pattern, team?)` over chunk text. Registered via the 3-file platform-tool pattern; results carry `source_id`, page markers, staleness timestamp. Respect S1 filters + D11 token budget per call.
**Acceptance:**
- [ ] Agent in chat reads past char 500 of a seeded 10-page doc (integration test)
- [ ] Team-scoped agent cannot read another team's doc (matrix test)
- [ ] Tool registered + reachable (PRD-155 reachability test green)

### S3: Token-budgeted context assembly + citations
Replace the 6000-char hard cut (`consumers/chatbot/service.py:1269-1271`) and the five independent char-truncation sites with one model-aware token budgeter (whole-chunk accumulation, RAGFlow-style numbered sources `[1]..[n]` + source map injected for citation answers).
**Acceptance:**
- [ ] Budgeter unit tests (model limits, whole-chunk boundaries, ordering by score)
- [ ] Chat answers cite numbered sources mapping to real doc ids (integration test)
- [ ] No `[:6000]`/char-slice truncations remain on the context path (grep gate)

### S4: Retrieval perf pass
Cache `RAGConfig` (today: up to 7 `SessionLocal`s per instantiation, `service.py:47-95`); reuse the S3 backend client (`:798-803` builds one per query); make access tracking async fire-and-forget without a blocking session (`:321-349`); drop the pure-Python knapsack DP (`:674-732`) for the budgeter from S3.
**Acceptance:**
- [ ] p50 retrieval latency benchmark halves on the seeded corpus (pytest-benchmark, before/after in PR)
- [ ] No per-query backend/config construction (assert via instrumentation test)

### S5: Document pinning + honest doc widgets
Pin a document to a conversation/agent (context always includes it within budget). Human-facing doc widgets/links suppressed for docs outside the answering agent's scope (Q4).
**Acceptance:**
- [ ] Pinned doc present in context across turns (test)
- [ ] Scoped-agent chat shows no out-of-scope `[View Document]` links — dev-browser verify

### S6: Delete placebo + dead surfaces
`rag_configurations` settings the backend never reads (`enable_graph_retrieval`, `graph_max_hops` — `configure-rag-modal.tsx:47-114`) either get implemented in PRD-164's graph-RAG or deleted here (default: delete the placebo UI now, PRD-164 reintroduces real ones); unmounted duplicate router `modules/rag/services/knowledge_multimodal.py`; `use-rag-feedback.ts` orphan; `FALLBACK_DATA` fake documents (`use-document-api.ts:29-43`).
**Acceptance:**
- [ ] Placebo controls gone from UI; dead files deleted; contract tests green
- [ ] API failure shows error state, not 'Example Document 1.pdf' — dev-browser verify

## Non-Goals

Team dropdown/management UX (PRD-158), graph-assisted RAG + flywheel ingestion (PRD-164), widget/SDK docs rebuild beyond schema verification (track as PRD-158 follow-up if prod drift confirmed).

## Success Metrics

- Seeded eval: 20-question doc-QA set answered with citations, ≥90% containing the gold passage (vs near-0 today on >500-char-deep answers).
- Zero char-based truncation on the agent context path.
- Retrieval p50 ≤ half of baseline on the benchmark corpus.

## Testing

Extend `test_rag_hydration.py`; new `test_retrieval_filters.py` (matrix), `test_read_document_tool.py`, budgeter unit suite, pytest-benchmark p50 gate. Update formatter/chat tests asserting the old 6000-char behavior. Full suite + contract gates green.
