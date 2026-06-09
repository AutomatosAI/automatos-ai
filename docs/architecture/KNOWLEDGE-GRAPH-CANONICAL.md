# Knowledge Graph — Canonical Store & Boundary

> The design spec for the **one design decision** in `PRD-142-CORE-DESIGN-REVIEW.md` §7:
> name the moat's canonical store, draw the boundary so the five different "graphs" in the
> codebase never collide, cut the dead twin, and name the scalability path.
>
> **Status:** Design decision — recommendation accepted (PRD-142 §7, Gerard 2026-05-29).
> **Author:** Design review session 2026-05-29.
> **Resolves gaps:** `BRAIN-BLUEPRINT.md` §8 **G11** (downgraded) and **G13** (cut).
> **Verified against:** branch `feat/widget-page-context-on-regular-chat`, code reads 2026-05-29.

---

## 1. The concern, and why it downgrades

The gap analysis flagged **G11 — "two graph stores, no single source of truth"** as a P2 corruption
risk. Investigation **downgrades it to a documentation task**, because the moat is *already*
single-sourced and there is *no active dual-write*. The risk is **latent**, not live: it only
materialises if someone *starts* dual-writing business entities into the learning store later. The
fix is therefore a **documented, enforced boundary** — not a migration, not a rewrite.

This document draws that boundary so it can't be crossed by accident.

---

## 2. There are five "graphs" — name them apart

"Knowledge graph" has been used loosely for five distinct concerns with five distinct stores. They
are not duplicates of each other; conflating them is what produced the G11/G13 confusion.

| # | Concern | Store | Module / writer | Verdict |
|---|---|---|---|---|
| 1 | **Business Knowledge Graph — THE MOAT** (products, orders, FBT, business memory) | `workspace_graphs` table | `modules/knowledge/graph_service.py` (`GraphifyService`) via `core/graph_storage.py` (`DbWorkspaceClient`) | **CANONICAL** |
| 2 | **Agent / config learning** (HARNESS diagnoses, prescriptions, outcomes) | `knowledge_nodes` / `knowledge_edges` / `learning_outcomes` | `modules/memory/storage/knowledge_system.py` | **KEEP, hard boundary** — learning-only; **reshaped into the HARNESS store (Role 2) in Wave 4** (§2A); the dead triple-store API is removed (W4-S13) |
| 3 | **Mission coordination field** (mission-scoped working memory) | Qdrant `field_memory` | `vector_field.py` | **KEEP** (orthogonal) |
| 4 | **Dead twin of #3** | `core/neural_field/` (6 files, 112 nodes) | only importer = dead workflow-era `AgentExecutionManager` | **CUT** as one unit with `execution_manager` (§5) |
| 5 | **Code intelligence** (symbol graph of this codebase) | `codegraph_symbols` | `codegraph_service.py` | **KEEP** (orthogonal — about code, not the business) |
| 6 | **Tool-selection learning** (which tool for which intent, from usage) | `tool_routing_edges` / `tool_routing_affinities` | `signal_recorder.py` → `edge_builder.py` → `graph_router.py` (PRD-138/139) | **KEEP, canonical (Role 1)** — learning-only, never business entities (§2A) |

#1 and #2 are the only two that *look* like the same thing. They are not: #1 is the merchant's
business; #2 is the agent's learning. §4 makes the line concrete. #6 is a third, separate kind of
learning — about *tools*, not the business and not agent config — added in §2A.

---

## 2A. The tool-routing learning graph (PRD-138/139) — canonical tool-selection learning

A **sixth** graph: the platform's live **tool-selection** learning loop. It is distinct from #1 (the
moat) and from #2 (agent/config learning), and — like every learning store — it must never hold a
business entity.

**What it is, and that it is already wired end-to-end.**
`modules/tools/discovery/signal_recorder.py` (a batched recorder, invoked after **every** tool
execution at `modules/tools/tool_router.py:614,659`) → the `tool_routing_edges` /
`tool_routing_affinities` tables (`core/models/tool_routing.py:39, 123`) → `core/services/edge_builder.py`
(nightly recompute from `tool_execution_logs`, Wilson-bounded confidence) →
`modules/tools/discovery/graph_router.py` (ranks tool chains at selection time; `fails_for_intent`
affinities **penalise** bad paths). It is `SEMANTIC_TOOL_ROUTING`-gated (default on) and bounds the
LLM's tool set so the model never sees the whole catalogue.

**The two learning roles (PRD-142 Wave 4 §7).** The platform's learning splits cleanly into two
canonical stores — do not conflate or merge them:

| Role | Learns | Canonical store |
|---|---|---|
| **Role 1 — tool-selection** | which tool for which intent, from real usage | the tool-routing graph (this section) |
| **Role 2 — config / diagnosis** | which config change improves the org | the **HARNESS structured store** — reshaped `learning_outcomes` + the new `harness_prescriptions` table (W4-S11/S12), replacing the dead `KnowledgeGraph`/`LearningEngine` API on `knowledge_nodes/edges` (#2; removed W4-S13) |

**The cross-link (read-only).** HARNESS's DIAGNOSE phase **reads** the tool-routing `fails_for_intent`
affinities as an inefficiency signal (W4-S10) — a sustained tool failure for an intent can surface a
`tool_assignment_remove` prescription. This is a one-way read from Role 1 into Role 2: HARNESS never
**writes** the tool-routing tables, and neither store ever holds a business entity.

**Boundary.** The §4 boundary contract applies to `tool_routing_*` verbatim: they store tool/intent
affinities and edges only — never products, orders, customers, or FBT. A business-entity write to a
tool-routing table is as forbidden as one to `knowledge_nodes/edges`.

---

## 3. The canonical store — `workspace_graphs`

**What it is.** The business graph is persisted as **path-keyed JSON artefacts** in the
`workspace_graphs` table — columns `(workspace_id, path, content, updated_at)`, composite key
`(workspace_id, path)` (`core/graph_storage.py`). `DbWorkspaceClient` is a drop-in filesystem client:
`GraphifyService` writes the graph via `_write_json` / reads via `_read_json`, so `content` is a
serialised graph artefact (the graph JSON), keyed per workspace.

**Why Postgres, not the workspace worker.** The worker container is only provisioned on demand;
wizard-created workspaces have none, so every graph write used to 404 and the in-memory graph
evaporated (`graph_storage.py` docstring). Persisting to Postgres keyed to the tenant row made the
write path deterministic for every workspace — and incidentally put the moat in the system of record
(`GUARDRAILS.md` A3).

**Single-writer, single-reader, proven.** Shopify's `frequently_bought_with` (FBT) edges — the
clearest piece of accumulated business value — are written and read **only** through
`GraphifyService` into `workspace_graphs` (`api/shopify.py:484, 582, 801`; the sync at `:785` merges
FBT via `import_graph(merge=True)`, strips-and-re-adds stale FBT at `:882–891`; the comment at
`:733` is explicit that "nodes ever land in workspace_graphs"). There is **zero** write of business
entities to `knowledge_nodes/edges`. The moat is single-sourced today.

**Verdict:** `workspace_graphs` is **CANONICAL** for all business graph data. Everything else derives
or stays out.

---

## 4. The boundary contract (the enforceable rule)

> **Business entities live in `workspace_graphs` (via `GraphifyService`) and nowhere else.
> `knowledge_nodes` / `knowledge_edges` is the agent-learning substrate and stores learning only —
> never products, orders, customers, or FBT relations.**

Why the two must never merge:

- **Different lifecycles.** Business data is imported/synced from the merchant's systems (Shopify,
  uploads) and is *authoritative*. Learning data is *derived* from agent experience and feedback
  (`LearningEngine.learn_from_feedback`, `knowledge_system.py:1003`) and is *advisory*.
- **Different blast radius on corruption.** A wrong business edge mis-sells to a customer; a wrong
  learning edge mis-routes an agent. Mixing them lets an advisory write corrupt authoritative data —
  exactly the latent G11 risk.
- **Different module ownership.** #1 lives in `modules/knowledge/`; #2 in `modules/memory/storage/`.
  The module boundary already reflects the concern boundary — keep it.

**How to enforce (proposed):**
1. A CI grep gate: no writes to `knowledge_nodes`/`knowledge_edges` (or the tool-routing learning
   tables `tool_routing_*`, §2A) from `modules/knowledge/`, `api/shopify.py`, or any vertical
   integration; no business-entity writes (`product`, `order`, `frequently_bought_with`) into any
   learning store.
2. A one-line docstring on both stores pointing at this contract.
3. The `knowledge_nodes/edges` store folds into HARNESS (its natural home — agent self-learning,
   PRD-142 §8). **Decided 2026-05-29 (Gerard): fix, not cut.** It is the self-learning capability the
   platform is committed to — a *dead loop* today (`add_knowledge`/`learn_from_feedback` have zero
   callers, the classes are never instantiated), not junk. **Wave 4 wires the loop**; keep the tables
   and schema as the starting point and reshape only if HARNESS's real requirements demand it. Until
   then the boundary (§4) holds so it never collides with the moat.

---

## 5. The dead twin — `core/neural_field/` is CUT (resolves G13)

`core/neural_field/` (6 files, 112 nodes) is a dead twin of the live mission-coordination field
(`vector_field.py` → Qdrant `field_memory`, #3). Its **only** importer is the workflow-era
`AgentExecutionManager` (`execution_manager.py:184`, neural-field write at `:348–356`), and that
class is itself dead: it is **never instantiated** (the `AgentService.execution_manager` property
that would build it is never read) and its main method `execute_workflow_subtasks` has **zero
callers** (code-verified 2026-05-29). It was superseded by `AgentFactory.execute_with_prompt`, which
Missions / Playbooks / Chat all use. So neural_field is reachable **only through dead code** — the
honest framing is "its one importer is dead," **not** "0 importers" (that earlier claim came off the
stale graph and was wrong). On the PRD-142 §11 **CUT** list.

> **Do not confuse with `vector_field`.** `vector_field.py` → Qdrant `field_memory` (#3, PRD-108) is
> the **live** Mission semantic field — 100% alive, **KEEP / FIX, never CUT**. `neural_field` is the
> *Redis*-backed PRD-59 predecessor. Different store, different era. If anything in the Mission field
> is unwired, that is a FIX, never a cut.

**Action:** cut `core/neural_field/` **together with** `AgentExecutionManager` and the dead
`AgentService.execution_manager` property — they are one workflow-era unit. Verify the
`channels/__init__.py` re-export is a no-op import first. This collapses the "two coordination
fields" confusion (G13) to one.

---

## 6. Code intelligence — `codegraph_symbols` stays separate

`codegraph_service.py` (`codegraph_symbols`, 55 nodes) graphs **this codebase's symbols**, not the
merchant's business. It is orthogonal to the moat and to learning — a developer/agent tool for
navigating code. **KEEP, unchanged.** It is named here only so it is never folded into the moat by
mistake.

---

## 7. Enterprise-scale flag — the storage-format path

**The honest limitation.** The moat serialises the *whole graph* as a JSON blob in
`workspace_graphs.content` and loads it into **NetworkX in memory, per request** (the
`DbWorkspaceClient` filesystem-in-a-table pattern, §3). For a small merchant this is fine and fast.
At enterprise scale — tens of thousands of products, deep order history, dense FBT — loading and
parsing the entire graph on every query will not hold.

**The path (named HARDEN, not a blocker):**

| Stage | Storage | When |
|---|---|---|
| **Today** | JSON blob in `content`, NetworkX in memory per request | works for current merchants |
| **Next** | Queryable **edge tables** (`graph_nodes` / `graph_edges` keyed by workspace) — load subgraphs, not the whole blob; index by relation/type | when a single workspace's graph stops fitting comfortably in a request |
| **If needed** | A dedicated graph engine (e.g. Postgres + `pgRouting`/recursive CTEs, or a graph DB) | only if traversal patterns demand it — earn it with evidence |

This evolves the *storage*, not the *contract* — `GraphifyService` stays the single writer/reader, so
the boundary (§4) holds across the migration. **Not a blocker for production**; it is a named item
on the moat's HARDEN list (PRD-142 §7) with a measurable trigger (graph load time per request).

---

## 8. Definition of Done

- [ ] **One canonical store named** — `workspace_graphs` documented as the sole business-graph store (this doc).
- [ ] **Boundary enforced** — CI gate rejects business-entity writes to `knowledge_nodes/edges` (§4).
- [ ] **Dead twin cut** — `core/neural_field/` deleted, `channels/__init__.py` re-export removed (§5).
- [ ] **Learning store has a home + fix commitment** — folded into HARNESS; Wave 4 wires the dead loop (§4). The 3 `COUNT(*)` dashboard tiles are hidden/labelled until the loop is live, so the Wave-0 dashboard doesn't imply a running system.
- [ ] **Scalability trigger instrumented** — graph load time per request emitted to the dashboard; the edge-table migration is specced (not built) when it crosses threshold (§7).
- [ ] **Tenant isolation** — graph reads/writes keyed by `workspace_id`; cross-workspace test proves no leak.

---

## 9. Open questions (for discussion before the build PRD)

1. **`knowledge_nodes/edges` fate — RESOLVED 2026-05-29 (Gerard): fix, not cut.** Wave 4 (HARNESS) wires the dead learning loop; keep the tables + schema as the starting point, reshape only if HARNESS's requirements demand. Cut-trigger removed.
2. **Edge-table migration timing** — proactively spec it now, or wait for the load-time trigger to fire? (Leaning: spec on trigger, instrument now.)
3. **Boundary enforcement mechanism** — CI grep gate (cheap, proposed) vs a typed repository layer that makes a business-entity write to the learning store impossible by construction (stronger, more work).

---

**This is a design decision and a boundary, not a migration.** The moat is already single-sourced;
this document keeps it that way and names the scale path. No code until the build PRD is approved.
