# PRD-168 — Platform Hygiene: Dead Code, Honest Surfaces, Actor Identity (WS-13b)

**Chain:** Early, anytime after PRD-155. Branch `ralph/prd-168-platform-hygiene` from main. Size **M**. Feeds PRD-169.
**Source:** report §2.12; the contract test (155) makes this deletion wave safe. Follows the PRD-142 S6 cut-loop discipline: **route verdicts need the api-client→hook→component indirection trace** before deleting.

## Overview

Delete the ~25 dead frontend files and dead/duplicate routers, split the god-file api-client, end fabricated data outside dev, and thread real actor identity through write paths.

## Binding amendments

Q13 (mock memory router — deleted in 156), Q62 (TOOL_WIDGET_MAP — deleted in 164), Q64 default: Assignments→Playbooks tab survives; `/playbooks` page redirects then deletes, Q65 default: UI migrates to canonical `/api/playbooks` (rename-in-place; legacy `/api/workflow-recipes` deleted after), Q66 default: `/chat/[id]` deleted (nothing links to it; shareable links are a future feature done right), Q67 (fixed in 154), Q68 default: `sites.py` dropped, Q69: api-client mock system removed (PRD-153 local stack covers dev), Q70 default: multi-agent coordination/execution-theater/benchmarking trees deleted (roadmap can resurrect from git), Q71-id: stale 2024 outage report archived, Q87: mount `rag_feedback` + chat thumbs (small, real feature — keep), Q89: cloud-upload visible block fixed or feature hidden.

## User Stories

### S1: Frontend dead-code wave
Delete the confirmed-dead clusters: dashboard/, workflow-management subtree, execution-theater/, composio/, 6 dead hooks (~1,041 lines), ~3,900 lines of dead `components/documents` (modern-file-manager, document-library, document-upload duplicates...), 3 debug pages (api-debug/api-diagnostics throwing in prod), orphan `/context` route (analytics folded into Knowledge Base per Q74), byte-identical duplicate stream route. **Each deletion verified via the indirection trace** (api-client→hook→component) per the Wave-5 lesson.
**Acceptance:**
- [ ] Trace evidence per cluster in PR description
- [ ] Typecheck, lint, contract suite, build all green; bundle-size delta reported

### S2: Backend dead-router wave
Delete: unmounted `database_knowledge_simple.py`; `api_playbooks.py` resolution per Q65 (canonical name claimed by the live implementation, loser deleted); `agents.py`/`agent_endpoints.py` route-shadowing resolved (one owner for `/api/agents`, `/active` 422 fixed); `sites.py`; the no-op `POST /agents/{id}/learn` either implements persistence or is removed (default: remove; PRD-159 owns real learning); mount `rag_feedback.py` + wire chat thumbs (Q87).
**Acceptance:**
- [ ] Mount-assertion manifest (155) updated; zero silently-unmounted routers remain
- [ ] `/api/agents/active` returns 200 (regression test for the shadowing 422)
- [ ] Feedback thumbs round-trip (test)

### S3: api-client split + mock removal
Split the 2,694-line `api-client.ts` by domain (board, documents, missions, graph, …) behind one transport with auth/baseUrl/error handling; DELETE the embedded mock system that hardcodes known-404s as acceptable (`api-client.ts:234-235`); prod logging strip.
**Acceptance:**
- [ ] No module > 400 lines (repo style rule); one transport; tree-shakeable domains
- [ ] Mock system gone; dev relies on the PRD-153 local stack
- [ ] Contract suite green (it now sees the real call sites per domain)

### S4: Actor identity
Write paths record the real actor: replace `created_by='system'` (`system.py:45,55,219`, `documents.py:217`) and `user_id='1'` (`database_knowledge.py:186`) with `RequestContext.user_id` threading — auth-provider-agnostic (PRD-150), agent actions attribute the agent id.
**Acceptance:**
- [ ] Audit test: human upload, agent write, system job each carry distinct correct actors
- [ ] Grep gate: no `created_by='system'` literals outside genuine system jobs

### S5: Honest metrics completion
Whatever fabricated data PRD-154 S10 didn't reach: fake validation checkmarks, remaining hardcoded stats, stale `reports/api-health-report.md` archived; every metric shown is served by a real endpoint or removed.
**Acceptance:**
- [ ] Grep gate for the catalogued literals (list from §2.12/§2.13)
- [ ] Analytics Overview shows real numbers or honest empty states — dev-browser verify

## Non-Goals

UX standardization (169), new features. This PRD only deletes, mounts, splits, and attributes.

## Success Metrics

- ≥ 5,000 net lines deleted; zero phantom endpoints reachable from the UI.
- Every router mounted explicitly; every metric real.

## Testing

Contract + mount + reachability suites are the safety net (155); indirection-trace evidence per deletion; updated tests for route-shadowing fix. Full suite green.
