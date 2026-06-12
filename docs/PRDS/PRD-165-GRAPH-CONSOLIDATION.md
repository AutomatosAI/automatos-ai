# PRD-165 — Graph Consolidation: Knowledge Graph UX + CodeGraph Agent Surface (WS-10)

**Chain:** Block B, branch `ralph/prd-165-graph-consolidation` from main after Night-1. Size **M**. Feeds PRD-166 (shell), PRD-169.
**Source:** report §2.5, §2.6, §3.2 (one graph-view stack); PRD-154 S6/S8 landed export + search fixes.

## Overview

One GraphView shell for every graph surface, a Knowledge Graph users can actually drill into (the owner's complaint: filter chips eat the screen, colors are noise, drill-in is shallow), and CodeGraph promoted to a real agent capability. The legacy entity-KG surface (gated in PRD-154) is deleted.

## Binding amendments

D8/D12; Q23: consolidate on the PRD-126 workspace graph, DELETE the PRD-21 entity explorer + endpoints, Q25: chips collapsed by default, filter/color prefs persisted per user+workspace, Q26: type-colors deterministic palette-as-data (landed PRD-154, extended here), Q27: connectors declare styling as data in meta.json (no hardcoded Shopify branches), Q28 default: server-side cluster-first drill-in now (LightRAG pattern) — ends full-graph.json downloads, Q29: build-time LLM community titles/summaries ON (GraphRAG reports), labels user-editable, Q30: tab renamed "Knowledge Graph" (canonical terms), Q31: codegraph auto-indexes connected repos, Q32: full codegraph tool family for ALL agents (PRD-143), Q33 default: webhook reindex + staleness stamp in results, Q34: delete EnhancedVectorStore, standardize on the working pgvector path, Q36: GitHub App installation tokens (workspace integration) over PATs, Q37: result budget config-driven (D11), Q38 default: GitHub repos now; local-workspace indexing lands with PRD-170.

## User Stories

### S1: Shared GraphView shell
One component: toolbar, collapsible legend (chips hidden by default, Q25), palette-from-CSS-vars, error boundary, empty/loading states, node detail side-panel slot. Engines: react-force-graph (force) + ReactFlow (DAGs) — delete raw-D3 and any remaining three.js graph stacks from the bundle.
**Acceptance:**
- [ ] KG + codegraph viz + mission DAG consume the shell (field viz follows in PRD-166)
- [ ] d3-graph + dead engines removed from package.json; bundle-size delta in PR
- [ ] Legend collapse/expand + persisted prefs — dev-browser verify; vitest for pref store

### S2: KG drill-in that's useful
Server-side subgraph endpoint (cluster-first: communities → expand → neighbors; no full graph.json to client); node side-panel: properties, sources (provenance to documents), connected entities, "expand from here"; path-finding mode between two nodes (tool + UI); search-to-focus.
**Acceptance:**
- [ ] 50k-node fixture: initial paint < 3s via cluster view (benchmark)
- [ ] Drill: community → node → source document round-trip — dev-browser verify
- [ ] `platform_graph_path` tool registered + tested

### S3: Community reports
Build-time LLM titles/summaries/rank per community (GraphRAG pattern, cheap model, D11); sidebar lists named clusters; labels editable (persisted); reports exposed to agents via `platform_graph_communities` (rewired in PRD-154 S6).
**Acceptance:**
- [ ] Build produces titled communities on seeded corpus (test, LLM mocked for determinism)
- [ ] Editable label persists (API test)

### S4: CodeGraph as an agent capability
Register the three implemented-but-unregistered executors (call-graph, dependency/impact, architecture `/ask`) + `list_projects`/`get_symbol` via the 3-file pattern; semantic routing inside `search_codebase`; results carry `path:line`, signature, staleness timestamp (PRD-154 S8 base); auto-index on repo connect; webhook-triggered incremental reindex honoring `auto_reindex`; GitHub App installation tokens replace PAT handling; DELETE EnhancedVectorStore (Q34).
**Acceptance:**
- [ ] Agent answers "what calls X?" + "what breaks if I change Y?" in integration tests on a seeded repo
- [ ] Webhook reindex test; stale results stamped
- [ ] EnhancedVectorStore gone (no shim); reachability gate green

### S5: Legacy deletion
Delete the PRD-21 entity explorer UI + unauthenticated endpoints (gated since PRD-154), dead viz controls, "Business Graph" label → "Knowledge Graph" everywhere user-facing.
**Acceptance:**
- [ ] Endpoints + components deleted; contract green; grep gate on the banned label

## Non-Goals

Field viz internals (166), graph-assisted RAG + flywheel (164), codegraph→graphify convergence (Q35: stays separate for now), local-path indexing (170).

## Success Metrics

- One graph engine pair in the bundle; all graph surfaces on the shell.
- KG demo: find cluster by name → drill to entity → open source doc, in under 5 clicks.
- Agents use codegraph tools in ≥1 real debugging mission during pilot week (observable via tool audit).

## Testing

Shell vitest suite, subgraph endpoint tests + benchmark, tool integration tests, webhook tests. Updated: KG export tests from PRD-154 S6. Full suite + contract green.
