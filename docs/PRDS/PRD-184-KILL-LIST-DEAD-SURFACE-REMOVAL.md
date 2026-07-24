# PRD-184: Kill-list — remove dead & actively-dishonest surface (the never-written July deletions)

**Phase:** Phase 2 — Module Deep-Review remediation (the reserved kill-list slot)
**Branch:** `feat/prd-184-kill-list-dead-surface` · **Worktree:** `automatos-ai-prd184`
**Status:** DRAFT — spec only, no build yet. **Audit-first.** Every §5 item was re-checked against **`origin/main @ 9dd4c848a`** (not the working tree) via `git ls-tree` / `git grep`; Waves 0–3 shipped over the last three weeks and **already cut some of these targets** — this PRD re-lists only what is *still present*.
**Build size:** S–M (deletion + caller migration; the one M is the legacy workflow-engine retire). **Risk:** Low **iff** each cut is grep-proven zero-caller before it lands — that proof is the gate.
**Source:** `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` **§5** (the kill-list; "Author these as PRD-184"); per-item dossiers in `reports/dossiers/`.

---

## Overview

**North-Star:** decoy and fabricating surface misleads the humans *and the agents* that read this code. An agent platform whose tool cards, endpoints, and module names *lie* — `/execute` that executes nothing, a `modules/evaluation/` that evaluates nothing, an "RBAC pillar" with no `workspace_id` — degrades every agent's output, because the next agent (Auto, a code-reading subagent, a new engineer) treats the decoy as real and routes into it. **The codebase should stop lying.** This PRD deletes grep-proven-dead or actively-dishonest surface. No moat framing; no new capability — pure subtraction.

**PILOT lens (locked):** "cold-start empty" is **not** a kill target. Empty tables, synthetic seed, and forever-zero-*yet-wired* counters stay — driving usage fills them. Only *dead* (zero-caller, unreachable) or *fabricating* (returns invented data, marks failure green) surface is cut. See `feedback-pilot-usage-not-quality-signal`.

**Framing (CLAUDE.md §3):** Refactor / subtraction — "pick the canonical path, migrate the losers, **delete the losers**." (CLAUDE.md §5: "delete what you replace, in the same PR — no `_legacy` that lives forever.")

---

## Audit — done vs remaining (every §5 item, re-checked @ `main 9dd4c848a`)

| # | Item (path/symbol) | Verdict | Cut by / notes | Conf |
|---|---|---|---|---|
| 1 | `modules/learning/` + `modules/evaluation/` + `api/api_playbooks.py` | **REMAINING** | all present; `modules/evaluation` zero external callers; `learning` re-exported only by `modules/__init__.py` + `api_playbooks` | High |
| 2 | `api/permissions.py` + `AgentToolPermission`/`Tool`/`PermissionAuditLog` tables | **DONE** | file gone; symbols survive only in guard `tests/test_p2w2_authz_relics.py` → **PRD-195** authZ consolidation | High |
| 3 | Placebo agent endpoints: `/{id}/execute`, `/active`+`/health`, `/{id}/performance` | **PARTIAL** | `/active`+`/health` now real workspace `COUNT` queries; `/performance` reads real `performance_metrics` — both **repaired** (agents-skills wave). **`POST /{id}/execute` still fabricates** (invented `execution_id`, hardcoded `"2025-08-01…"`, executes nothing) → **REMAINING** | High |
| 4 | `nl2sql/intelligence/` (1,687 LOC) + PRD-21 limb `service.py:680-883` + `SchemaLinker` | **DONE** | `intelligence/` source deleted; `_build_connection_string`/`SchemaLinker` survive only as tombstone comment + guard `tests/test_prd199_nl2sql.py` → **PRD-199 S5**. *(An untracked `__pycache__` bytecode dir is the only thing left on the working-tree disk — not tracked, not on `main`.)* | High |
| 5 | F079 trio `EnhancedVectorStore`/`SearchService`/`ContextRetrievalEngine` + `/api/v1/memory` FAISS leg | **DONE** | no class defs anywhere; only tombstones + guard `tests/test_prd197_substrate.py` → **PRD-197 S1**. FAISS `/v1/memory` router gone → **PRD-187 S5** | High |
| 6 | Memory relics: `AdvancedMemoryManager` + `api/memory.py` + `MemoryKnowledgeSystem` | **DONE** | dropped by `alembic/versions/prd187_s5_drop_memory_relics.py`; guard `tests/test_p2w1_relics_deleted.py` → **PRD-187 S5** | High |
| 7 | `llm-core` scaffolding: `function_executor`/`function_registry`/`response_parser`/`semantic_skill_matcher` + `api/anthropic_client.py` | **REMAINING** | all present; `api/anthropic_client.py` = **zero importers**; the four are kept alive **only** by the `core/llm/__init__.py` barrel (lines 40-43). Bonus dead sibling: `core/global_function_registry.py` (zero external callers) | High |
| 8 | `exec_planning.py` stub vertical (8 template-writer "planning" tools, 0 LLM) | **REMAINING** | present; wired into `unified_executor.py:36,969-975` (routed) but exposed to no agent — de-route required | Med-High |
| 9 | `tools/execution/concurrency.py` + `tools/service.py` `ToolService` + `composio_tool_router` delegate | **REMAINING** | `concurrency.py` = **zero callers** (clean); `ToolService` instantiated only by its own barrel `tools/__init__.py:10` (live code uses `ComposioToolService`); `composio_tool_router.py` file is live — only its crash-on-`db_session` delegate is dead | Med |
| 10 | 7 legacy channel adapters (teams/google_chat/signal/imessage/irc/matrix/line) + `_ping_platform_legacy` | **REMAINING** | all 7 present at `orchestrator/channels/*_adapter.py` (~1,570 LOC); `_ping_platform_legacy` present `api/channels.py:144` | High |
| 11 | Frontend relics: `/api-control`, `/styleguide`, `workspaceMeta` pill, `/chat/[id]` zombie route | **REMAINING** | `app/api-control/page.jsx`, `app/styleguide/page.tsx`, `app/chat/[id]/page.tsx` present; `workspaceMeta` in `components/layout/studio-sidebar.tsx`; `/chat/[id]` has 3 live `router.push` callers. *(The 2 redundant lockfiles → **PRD-209 S6**, not cut here.)* | High |
| 12 | Discarded KG output: `graph.html`, `surprising_connections`/`score_all`, hyperedge prompting, `knowledge_nodes`/`knowledge_edges` tables | **PARTIAL** | tables **dropped** (`prd187_s5_drop_memory_relics.py` → **PRD-187 S5**); `graph.html` not tracked (moot). **BUT** `surprising_connections`/`score_all` are **still actively called** in `modules/knowledge/graph_service.py:301,307,471` → REMAINING **only if** the output is grep-proven unconsumed | Med |
| 13 | Legacy workflow engine `api/workflows.py` (1,425 L) + `workflow_templates` + `jira_bug_triage` | **REMAINING** (retire) | `api/workflows.py`, `api/workflow_templates.py`, `workflow_templates` table, `modules/workflows/recipes/jira_bug_triage` all present — migrate then unmount | High |
| 14 | `PlaybookMiner` vertical: `api_playbooks` + `learning/playbooks/miner.py` + `PlaybooksPanel` | **REMAINING** (retire) | `miner.py`, `components/playbooks/PlaybooksPanel.tsx`, `app/playbooks/page.tsx` present; `api_playbooks` mounted `main.py:55` (explicitly *not* an execution router per `playbook_engine.py:41`). Overlaps item 1 | High |
| 15–18 | Decide-then-cut (RAG dark features; `TOOL_SIGNAL_RECORDER_ENABLED`; field-benchmark + `context/experiment.py`; the two caller-less `ContextMode`s + `tone.py`) | **PRESENT — Gerard's call** | all present on `main`; **not** unilaterally cut → see **Open questions (§12)** | — |

**Tally:** of the 14 delete/retire bullets — **4 DONE** (2, 4, 5, 6), **2 PARTIAL** (3, 12), **8 REMAINING** (1, 7, 8, 9, 10, 11, 13, 14). The 4 decide-then-cut bullets are surfaced, not cut.
**Estimated removal (REMAINING, tracked source, excluding the 2 generated lockfiles): ~6,975 LOC** — ~4,870 delete-now, ~2,105 retire.

> **Boundary:** mem0 residue (`mem0_openapi.json`, mem0 scripts, stale mem0 PRD docs) is **out of scope here** → owned by **PRD-211** (memory topology). Excluded to avoid duplication.

---

## Build size + Risk (per tier)

Deletions are **low-risk iff grep-proven zero-caller** — that proof is the gate, not an afterthought. Delete-now items are files/symbols with no live consumer (or one barrel re-export to trim in the same commit). Retire items touch a *reachable* surface (a mounted route, a Composio-webhook recipe) → they carry a **migration** step and are Medium risk until the caller moves. The one recurring failure mode on this repo is **source-grep guard drift** (`project-phase2-module-deep-review`): a guard test that pins "symbol X must not return" goes red when a *legitimate* refactor moves X — so every guard is **repointed/added in the same commit as the deletion**, never a follow-up.

---

## Stories (test-first; CI is the only gate — no local runs)

> Only **REMAINING** items get a story; DONE items are one audit-table row each. Each story = the path(s) · a **grep-prove-zero-callers** step · the deletion · a **source-grep guard test** (repointed/added in the same commit) so the surface can't silently regrow.

### Tier A — Delete now (dead-on-arrival / fabricating / zero real callers)

**S1 · Kill the "learning"/"evaluation" theatre — S · _evals-learning E; F069/F082/F080_**
Delete `modules/evaluation/` (30 L, zero external callers) and the `modules/learning/` package (`__init__`, `feedback/`, `patterns/`, `playbooks/`). Grep-prove: the only importers are `modules/__init__.py` (trim the re-export) and `api_playbooks` (goes in S9-retire). `miner.py` + `api_playbooks.py` migrate with the PlaybookMiner vertical (**S10**) — this story removes the two empty-theatre packages.
**Test:** `test_no_learning_evaluation_imports` (source-grep guard: no live `modules.learning`/`modules.evaluation` import outside the deleted tree). Pure.

**S2 · Delete the llm-core dead scaffolding — S · _llm-core C.10_**
Delete `core/llm/function_executor.py` (437) · `function_registry.py` (407) · `response_parser.py` (360) · `semantic_skill_matcher.py` (203) · `core/global_function_registry.py` (304, zero external callers) · `api/anthropic_client.py` (248, **zero importers**). Repoint the `core/llm/__init__.py` barrel (drop lines 40-43) **in the same commit**.
**Test:** `test_llm_core_no_dead_scaffolding` (grep guard: none of the six symbols importable / re-exported). Pure.

**S3 · Delete the `exec_planning` stub vertical — S · _planning-scheduling B5/§12_**
Delete `modules/tools/execution/exec_planning.py` (339, 8 hardcoded template-writers, 0 LLM). De-route in the same commit: remove the import at `unified_executor.py:36` and the three dispatch branches (`:969-975`), and drop the 8 tool names from any registry. **Grep-prove** no agent toolset references those 8 names before cutting.
**Test:** `test_exec_planning_deleted_and_unrouted` (grep guard: symbol gone + `unified_executor` has no `exec_planning` dispatch). Pure.

**S4 · Delete the dead tool-runtime scaffolding — S · _tool-runtime C.6_**
Delete `modules/tools/execution/concurrency.py` (166, **zero callers** — clean). Then grep-trace `ToolService` (`tools/service.py`, 145): live code instantiates `ComposioToolService`, not this; if the only `ToolService(` site is the `tools/__init__.py:10` barrel factory and *that* has no external caller, delete both and trim the barrel. Excise the specific crash-on-`db_session` delegate method in `composio_tool_router.py` (the **file stays** — it is live via `exec_composio`/`unified_executor`). If any `ToolService` caller is real, **surface it (Open-Q), don't flatten.**
**Test:** `test_no_tools_concurrency_import`; `test_toolservice_singular_unused` (grep guards). Pure.

**S5 · Delete the driverless legacy channel adapters — S · _channels F081_**
Delete the 7 adapters `orchestrator/channels/{teams,google_chat,signal,imessage,irc,matrix,line}_adapter.py` (~1,570 L, byte-near-identical, no driver) and `_ping_platform_legacy` (`api/channels.py:144`, 0 callers). Grep-prove no live channel dispatch imports them (the one active channel path does not).
**Test:** `test_no_legacy_channel_adapters` (grep guard). Pure.

**S6 · Delete the frontend relics — S · _deployability F084; auto-core/observability F036/F038_**
Delete `app/api-control/page.jsx` (98, placebo for the PRD-168-removed mock system) and `app/styleguide/page.tsx` (442, routed in prod). Remove the fabricated `workspaceMeta='pilot · 11 op'` pill from `components/layout/studio-sidebar.tsx`. *(The 2 redundant frontend lockfiles are owned by **PRD-209 S6** — a fresh-clone build-determinism concern — not cut here.)* `/chat/[id]` route deletion → **S10** (has live callers).
**Test:** `test_no_placebo_routes` (grep guard: no `/api-control` / `/styleguide` route; no `workspaceMeta` literal). Pure.

**S7 · Un-fabricate the agent `/execute` endpoint — S · _agents-skills C.4/C.5_**
`POST /api/agents/{id}/execute` (`api/agents.py:711`) returns an invented `execution_id`, a hardcoded `"started_at":"2025-08-01T12:57:03Z"`, and executes nothing. Either **wire it to the real Mission/Playbook launch path** or **delete the endpoint** (Open-Q — it now carries an authZ dependency, so a caller may expect it). `/active`+`/health`+`/performance` are already repaired — leave them.
**Test:** `test_agent_execute_not_fabricated` (asserts no hardcoded timestamp / invented id path). Pure.

### Tier B — Retire (migrate the real surface, then delete)

**S9 · Retire the legacy workflow engine — M · _missions C.8/J.10; F078_**
`api/workflows.py` (1,425) + `api/workflow_templates.py` (380) + the `workflow_templates` table are a fifth execution engine, Composio-webhook-reachable via `jira_bug_triage` (`modules/workflows/recipes/`), with none of the Mission/board hardening. **Migrate the `jira_bug_triage` recipe onto the Mission/Playbook path first**, repoint the webhook, then unmount the routers and drop `workflow_templates` (migration). Delete-in-same-PR once the recipe runs on the canonical engine.
**Test:** `test_jira_recipe_runs_on_mission_path`; `test_workflows_engine_unmounted` (grep guard: no `api.workflows` router mount). Pure/mocked. Migration self-applies on boot.

**S10 · Retire the PlaybookMiner scaffold + the `/chat/[id]` zombie route — S · _playbooks E; auto-core F036_**
Delete the demo miner vertical: `api/api_playbooks.py` (47) + `modules/learning/playbooks/miner.py` (89) + `components/playbooks/PlaybooksPanel.tsx` (92) + `app/playbooks/page.tsx` (37); unmount `playbooks_router` (`main.py:55`). *(Real step-sequence mining over `recipe_executions` is a fine future PRD — this removes the fabricating scaffold, not the idea.)* Separately, migrate the 3 live `router.push(\`/chat/${id}\`)` callers (`activity-feed.tsx:236`, `execution-detail.tsx:333`, `recent-activity-widget.tsx:71`) to the established `/chat?chatId=` query-param pattern, then delete `app/chat/[id]/page.tsx` (36).
**Test:** `test_playbook_miner_deleted`; `test_no_chat_id_route_pushes` (grep guard: no `router.push('/chat/…')` and no `app/chat/[id]`). Pure.

### Tier C — Prove-then-cut (live-called; delete **only** if the output is unconsumed)

**S8 · KG discarded-output computation — S · _knowledge-graphs E.5_**
`surprising_connections` / `score_all` are **actively called** in `modules/knowledge/graph_service.py:301,307,471` — the §5 "dead computation" claim does **not** hold at `main` unless the *output* is dropped downstream. **Grep-prove the results are unconsumed** (no reader of the scored/surprising output; no tile renders it) **before** cutting; if consumed, this is not dead — close the item. Also confirm the hyperedge-prompting parse-then-drop and the forever-zero analytics tiles.
**Test:** `test_kg_scored_output_has_no_reader` (guard, only if proven). Pure.

---

## Sequencing

- **Delete-now (S1–S7) are independent** and parallel-safe — disjoint file ownership; the only shared touch is a barrel `__init__.py` per story (trim in that story's commit, never inline elsewhere).
- **Retire before unmount:** S9 needs the `jira_bug_triage` migration onto Mission/Playbook *first*; S10 needs the 3 chat-route callers moved to `/chat?chatId=` *first*. Deletion is the commit *after* the last caller moves.
- **S8 is a trace + decision**, not a blind delete — it gates on proving the output is unconsumed.
- S1 and S10 overlap on the `learning/playbooks` tree — land S1 (packages) and S10 (miner scaffold + api_playbooks) as one coordinated pair so `modules/__init__` is repointed once.

---

## Verification (CI is the only gate — no local runs)

Per `feedback-no-local-servers`: no servers/builds/`pytest`/`tsc`/installs on the dev machine — write code + **pure** tests (no DB/network/Composio), commit, push, let PR CI verify. **Every deletion ships with (a) a grep-proof that callers are zero and (b) a source-grep guard test, both in the same commit** (the guard-drift lesson: repoint/add the guard where the symbol dies). No route left mounted with no handler; no import left dangling; new/removed backend route → hand-update `orchestrator/reports/route-manifest.json` sorted + count-bumped (`route-manifest-frontend-contract`). Migrations (drop `workflow_templates`) self-apply on boot.

---

## Conventions (non-negotiable — `automatos-ai/CLAUDE.md`)

- **No backward-compat shims** — the old path is deleted in the same PR, no `_legacy` suffix (§4/§5).
- **No `os.getenv()` outside `config.py`** — relevant to the S8 dark-flag decision.
- Canonical vocab: **Playbook** (not Recipe), **Mission** (not Workflow), **Deliverable**, **Knowledge Graph**, **Command Center**, **Auto**.
- **Never unilaterally descope a decide-then-cut item (§12)** — surface it (below), do not defer or delete on your own initiative.
- Branch `feat/prd-184-kill-list-dead-surface`; commit, push, open a PR; CI is the gate.

---

## Success metrics (the definition of "stops lying")

- **~6,975 LOC of tracked source removed** across the 8 REMAINING items, **0 dangling imports/routes/mounts** afterwards. *(Lockfiles are PRD-209 S6.)*
- **Every deletion is grep-proven zero-caller** and carries a guard test → the dead surface **cannot silently regrow**.
- The **fabricating** surfaces are gone or wired: `/execute` no longer invents an `execution_id`; no module named `evaluation`/`learning` signposts away from the real loops.
- The legacy **fifth execution engine** is unmounted; `jira_bug_triage` runs on the canonical Mission/Playbook path.
- **The codebase stops advertising dead lanes** to the agents and humans that read it — the audit table above goes all-green on the next re-run.

---

## Open questions — Gerard's call (§12; surfaced, not deferred)

1. **RAG dark features (§5 item 15):** ship the document-**pinning UI** (light up RAG S5) or **delete** it? And the **multimodal search tools** over a store no one feeds — keep dark, or cut? *(Distinct from the agent-pinning feature in `use-pinned-agents.ts`, which is live — not a target.)*
2. **`TOOL_SIGNAL_RECORDER_ENABLED` (item 16):** `config.py:977` defaults `false`; read by `modules/tools/discovery/signal_recorder.py`. **Default-true (use it) or delete (drop the seam)** — "dark forever" is the one wrong state.
3. **Field-benchmark + `context/experiment.py` (item 17):** the stale `tools/benchmark_results/benchmark_vector_field_*20260330*.json` (7 files) + `field-memory-benchmark-report.md` + the orphaned `modules/context/experiment.py` — archive the results and cut the orphan, or keep as a live harness?
4. **Two caller-less `ContextMode`s + `tone.py` (item 18):** `ContextMode.COORDINATOR` + `ORCHESTRATOR_STAGE` (defined in `modules/context/modes.py`/`budget.py`, never selected) and `modules/context/tone.py` — delete, or wire a selector?
5. **`/execute` endpoint (S7):** wire to the real launch path, or delete? It now carries an authZ dependency, so a client may expect it.
6. *(Lockfile survivor moved to **PRD-209 S6 / its Q5** — the fresh-clone-boot PRD owns the lockfile cull; the "which of `package-lock.json` / `pnpm-lock.yaml` / `yarn.lock` survives" question lives there.)*

---

*Traceability: `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` **§5** (the reserved PRD-184 kill-list) + per-item dossier sections — evals-learning E · auth-identity C.5 · agents-skills C.4/C.5 · nl2sql C.8 · vector-substrate C.5 / context-assembly F079 · memory E.4 · llm-core C.10 · planning-scheduling B5 · tool-runtime C.6 · channels F081 · deployability F084 · knowledge-graphs E.5 · missions C.8/J.10 · playbooks E. **Every verdict audited against `origin/main @ 9dd4c848a`** via `git ls-tree`/`git grep` (DONE items confirmed by their shipped guard tests: PRD-187 S5 memory/KG-table drop, PRD-195 authZ relics, PRD-197 S1 F079, PRD-199 S5 nl2sql). mem0 residue excluded → **PRD-211**. North-Star framed; PILOT lens; no moat framing; §12 decide-then-cut items surfaced, not cut.*
