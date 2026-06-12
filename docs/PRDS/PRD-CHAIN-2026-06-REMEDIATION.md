# PRD Chain — Platform Remediation (2026-06)

**Source of truth:** [`reports/PLATFORM_DEEP_REVIEW_2026-06.md`](../../reports/PLATFORM_DEEP_REVIEW_2026-06.md) (§2 evidence, §4 workstreams, §5 decisions Q1–Q97).
**Method:** Ralph chain — one PRD per workstream, stacked worktrees, verified story maps, build→accept→review per PRD, Gerard merges each morning.

## Locked decisions (BINDING across all PRDs)

| # | Decision |
|---|----------|
| D1 | RAG team scoping reuses the existing agent **Teams** concept: docs multi-tagged, agent team membership gates retrieval at query time |
| D2 | Document templates: **visual block editor** over JSON storage; render PDF + DOCX |
| D3 | Code Canvas: **Claude Agent SDK headless** per workspace; UI renders session/diffs/approvals/git |
| D4 | Execution: Ralph PRD chain (this document) |
| D5 | Memory: **orchestrator owns extraction** (single distill pass, operational taxonomy); mem0 = storage/search/dedup, `infer:false` everywhere |
| D6 | Output flywheel: mission/playbook/document outputs **auto-ingest by default** (source_type=`agent_output`, per-workspace opt-out) |
| D7 | Field memory: **workspace-persistent** target; archive-don't-destroy as the stepping stone |
| D8 | **Studio is the future** — fixes land in command-center; classic board/calendar/chat sunset after parity |
| D9 | Mission auto-approval: **workspace policy enum + dollar ceiling** + per-request chat override; countdown approval card |
| D10 | Studio is live to pilots → fake ticker/counts fixed in **Wave 0** |
| D11 | Cost posture: **quality first, capped** — accept 2–3× per-action spend; token-budgeted assembly, cheap-model distiller, mission $ ceilings |
| D12 | The ~90 §5 defaults are **accepted**; each PRD embeds its relevant ones as BINDING amendments — override by exception at PRD review |

Project rules apply to every PRD (repo `CLAUDE.md`): no backward-compat shims; delete what you replace in the same PR; no new tables/tools/hooks when an existing one fits; no `os.getenv` outside `config.py`; canonical terms (Playbook, Mission, Task, Deliverable, Knowledge Graph, Command Center, Auto).

## PRD index

| PRD | Name | WS | Size | Depends on | Wave |
|-----|------|----|------|------------|------|
| 154 | Wave-0 Quick Wins | §4 Wave 0 | S | — | Night 1 |
| 155 | Route Contract & Mount Honesty | WS-13a | S | — | Night 1 (stack on 154) |
| 156 | Security & Tenancy Hardening | WS-1 | S-M | — | Night 1 (stack on 155) |
| 157 | RAG Content & Retrieval Quality | WS-2 | M | 154 | Block A |
| 158 | Teams Model & Knowledge UX | WS-3 | M | 157 (filter builder) | Block B |
| 159 | Memory Quality & Lifecycle | WS-4 | M | 154 | Block A |
| 160 | NL2SQL Agent Path & Accuracy | WS-5 | M | 156 | Block B |
| 161 | Board Execution Engine | WS-6 | M | 154 | Block A |
| 162 | Calendar & Schedule Truth | WS-7 | S | — | Block A (lead) |
| 163 | Missions Lifecycle & Plan Mode | WS-8 | M | 154 | Block A |
| 164 | Planning Intelligence & Seams | WS-9 | L | 157, 159, 163, 166 | Payoff |
| 165 | Graph Consolidation (KG + CodeGraph) | WS-10 | M | 154 | Block B |
| 166 | Field Memory Core | WS-11 | M | 154 | Block B |
| 167 | Document Templates Block Editor | WS-12 | L | 156 | Net-new |
| 168 | Platform Hygiene & Dead Code | WS-13b | M | 155 | Early, anytime |
| 169 | UX Consistency & Design System | WS-14 | M | 168, 165 | Closing |
| 170 | Code Canvas — Agent SDK Embed | net-new | L | 156 | Net-new |

**Topology:** Night 1 is a stack (154 → 155 → 156, each branch cut from the previous tip, PRD-153 honesty rule: if the base is incomplete the acceptance FAILS, no shims). After Night-1 merges, Block A PRDs (162, 157, 159, 161, 163) branch **from main in parallel** — they touch disjoint modules; merge order within the block: 162 → 161 → 163 → 157 → 159 (tool-registry files overlap slightly; later merges rebase). Block B (158, 160, 165, 166) follows the same pattern. 164 only starts after 157/159/163/166 are merged. 167/170 are net-new and can run any night after 156. 168 early (its contract test from 155 protects it); 169 last.

**Branch/worktree convention:** `ralph/prd-NNN-<slug>`, worktree `../automatos-ai-prdNNN`.

## Testing policy (every PRD)

1. **Orchestrator suite green:** `cd orchestrator && pytest -q` against Postgres exactly as `.github/workflows/test.yml` runs it (130 test files; per-test timeout). A PRD that intentionally changes behavior **updates the affected tests in the same story** — never deletes or skips them to get green.
2. **Protected suites — do not regress:** recipe step-communication tests (20, `recipe_executor`) and Composio hint-service tests (25). PRDs 159/163/164 touch adjacent code; these suites are explicit acceptance gates there.
3. **New behavior = new tests first** (TDD): every story names its test file(s); target ≥80% coverage on new modules.
4. **Frontend:** `tsc --noEmit` + ESLint green; vitest for new pure logic (config exists, suite is thin — grow it); UI stories verify in browser via dev-browser before marking done.
5. **Route-contract test** (lands in PRD-155): frontend api paths ⊆ backend routes — becomes a required gate for every subsequent PRD in the chain.
6. **Acceptance gates:** each PRD ships `scripts/ralph/acceptance-prdNNN.sh` (PRD-153 pattern) — the per-story checks below are summarized there.

## Runner artifacts

`scripts/ralph/prd-NNN.json` story maps are authored for all 17 PRDs now ("run later"). `PROMPT_build_prdNNN.md`, `acceptance-prdNNN.sh`, and the overnight chain script are generated **at chain-launch per night** (they bake in worktree paths + verifier fixes, and story maps must be re-verified against the then-current main before each night — file:line evidence drifts as earlier PRDs merge).

**Pre-launch checklist (from PRD-150→153 chain):** Railway env vars set before any deploy-touching PRD; `alembic stamp` if migrations were squashed; never run a same-night PRD-142-S6 cut loop in parallel with this chain.

## Open research debt

WS-12/PRD-167: the OSS block-editor comparison (BlockNote vs Plate vs Puck) never ran (session limit) — PRD-167's S1 is that evaluation, time-boxed, before the editor stories.
