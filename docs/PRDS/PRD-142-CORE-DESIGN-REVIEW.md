# PRD-142: Automatos Core Design Review — Rewrite vs Fix

**Status:** Draft for decision (design review, not yet a build PRD)
**Author:** Claude (design review session 2026-05-29), commissioned by Gerard Kavanagh
**Priority:** P0 — this is the gate the production roadmap runs through
**Scope:** The entire platform surface — every primitive, the connective tissue, the data layer, the frontend — across **all seven repositories** (§4A), not just `automatos-ai`
**Method:** Knowledge-graph mining (`graphify-out/graph.json`, 20,323 nodes / 24,103 edges) + every PRD (1–142) + vision/design docs + code-health sweep + six fresh subsystem maps + a survey of the six companion repos. Claims are graph-verified where a number appears.
**Design spec it ratifies:** `docs/architecture/BRAIN-BLUEPRINT.md`, `DIAGRAMS.md`, `GUARDRAILS.md`, `TEST-PLAN.md`
**Companion design docs:** `docs/architecture/PLAYBOOK-ENGINE-DESIGN.md`, `docs/architecture/KNOWLEDGE-GRAPH-CANONICAL.md`
**Supersedes:** the "Rock-Solid" working title in memory (`prd142-rock-solid-plan`). The six waves there become this PRD's execution roadmap (§12).
**Build branch:** TBD on approval (this document is docs-only)
**Depends on:** both PRD-141s merged to main (Platform Reliability + Widget Vertical-Agnostic Refactor)

---

## 1. The question this PRD answers

Automatos works. It is at PRD-141+, has run real missions, real chat, real widget conversations, a real Shopify integration. But it was built fast, largely by one developer, and it carries the debt of that velocity: 116 tables, 103 routers, ~945 backend files, ~670 frontend files, two tool loops, three playbook engines, ~1,970 broad exception handlers, and one frontend test.

So the question is **not** "does it work" — it does. The question is:

> **Treating today's Automatos as a working MVP, and designing the production target as if we were building an enterprise platform from scratch — which subsystems do we rewrite, and which do we fix in place?**

This document gives the answer, per subsystem, with the evidence. The design target itself is specified in `BRAIN-BLUEPRINT.md`; this PRD is the **decision layer** on top of it: what is sound, what is salvageable, what is rotten, what is dead. No functionality is removed — only dead code and duplicate paths are cut. New features are explicitly out of scope.

---

## 2. How the verdict was reached

This is not an opinion. Each subsystem was assessed against four evidence streams:

1. **Contract** — is the boundary/interface right? (from `BRAIN-BLUEPRINT.md` §3 primitive contracts)
2. **Coupling** — graph-mined inbound/outbound edges. Pathological coupling is a rewrite signal; a clean blast radius is a harden signal.
3. **Implementation health** — file size, test ratio, duplication, fire-and-forget, fail-open paths (code-health sweep).
4. **Intent lineage** — what the PRDs *meant* the subsystem to be vs what it became (PRD archaeology 1–142).

The graph was queried directly (the CLI is on a stale format; the JSON was mined with Python) to verify the three highest-stakes claims: the Playbook consolidation surface, dead-code cut safety, and the moat's store dependencies. Those numbers appear inline below and are flagged **(graph-verified)**.

---

## 3. The decision framework

Every subsystem gets exactly one verdict:

| Verdict | Meaning | Trigger |
|---|---|---|
| **KEEP** | Contract right **and** implementation healthy | Sound design, low debt. Hygiene only (file splits). |
| **HARDEN** | Contract right, implementation has fixable debt | Good boundary, but god-files / missing tests / swallowed errors / scattered config. |
| **REWRITE** | The contract itself is wrong, or coupling is pathological | One concept with multiple implementations; user-visible work that can't survive restart. |
| **CUT** | Dead weight | No real dependents; superseded; duplicate. |

**The prior I opened the review with, and the evidence confirmed:** a platform at PRD-141 with real functionality is almost never a big-bang rewrite. You harden most, rewrite the two or three rotten cores, cut the dead weight. The evidence did not move me off that.

---

## 4. The verdict

> **FIX, don't rewrite.** The platform's *contracts* are mostly right — the Arc (one loop, many front doors), DB-as-system-of-record, vertical-agnostic core, markdown-deliverables-in-S3. The rot is implementation debt and duplication, not architecture.

The whole platform reduces to:

- **0 rewrites** — the biggest single piece is the Playbook engine **consolidate + harden** (§6): one scattered-but-live engine (recipe = playbook) unified into one clean durable flow, not rebuilt.
- **1 design decision** — the knowledge-graph canonical store, already 90% settled (§7).
- **1 test-net rebuild** — the frontend, at enterprise bar (§9).
- **3 P0 fixes that gate everything** — session leak, migration safety, fail-open authz (§10).
- **Harden-in-place** — chat, memory, RAG, NL2SQL, tools, config, data layer.
- **Keep** — missions, routing, channels, widgets, agents/factory.
- **Cut** — the dead-weight inventory (§11).

This is a strangler-fig program, not a teardown.

---

## 4A. Platform topology — seven repos, one platform

Automatos is **not a monorepo**. The design target spans **seven repositories**, and the platform's production-readiness is gated by the *weakest* of them. The core review (§5) is `automatos-ai`; the satellites carry their own verdicts and their own debt.

| Repo | Role | Stack | Size | Test / CI posture | Verdict |
|---|---|---|---|---|---|
| **automatos-ai** | Core: the Arc, primitives, orchestrator, frontend | Python + Next.js | ~945 py / ~670 ts | 1 FE test; backend thin | per §5 (HARDEN + 1 consolidation) |
| **automatos-mem0** | L3 long-term memory — **self-hosted fork of `mem0ai/mem0`** | Python | 869 files / 154 tests | upstream tests; on `fix/pool-exhaustion` | **KEEP (fork) + HARDEN** — pool health + a fork-maintenance strategy |
| **automatos-widget-sdk** | Embeddable storefront widget (Shadow DOM, ~9KB) — **defines the widget API contract** | TS monorepo (pnpm/turbo) | 76 files / 16 tests | tests + **pre-merge CI** (newest) | **KEEP** — most mature satellite; add cross-repo contract tests |
| **automatos-shopify** | Merchant-facing Shopify app + extensions (the distribution wedge) | Remix/React-Router + Prisma | 17 files | **0 tests** | **KEEP + HARDEN** — needs tests; vertical front-end only (core logic stays in `automatos-ai`) |
| **automatos-skills** | Skill library (PRD-120) — markdown/content, loaded by the platform | content (0 code) | ~18 domains | n/a | **KEEP** — confirm load path is DB-backed, not a runtime file-hack (GUARDRAILS D3) |
| **automatos-voice** | Voice services + pipeline (L7) | Python + Railway | 16 files | **0 tests** | **KEEP (peripheral) + light HARDEN** — low priority |
| **automatos-testing** | Central test harness — **black-box, prod-hitting, 5-phase smoke** | Python + n8n | 45 files / 34 tests | **stale: master last 2025-08-31** | **HARVEST then RETIRE** — see §9 |

**Three cross-repo truths this surfaces:**

1. **Test posture is wildly inconsistent.** widget-sdk has CI; shopify and voice have zero tests; the central testing repo is 9 months stale and tests a *renamed* surface (`/api/workflows/*`, pre-Mission/Playbook) including the now-dead `neural_field`. "Production-ready" can't be claimed per-repo — it needs one bar across all seven.
2. **Real cross-repo contracts have no contract tests.** SDK ↔ widget API (the `page_context` change on the current branch is exactly this seam), shopify app ↔ core widget API, skills ↔ platform loader, voice ↔ core. These integration seams are where drift hides.
3. **The L3 memory is a maintained fork.** `automatos-mem0` forks `mem0ai/mem0` and self-hosts on Railway — an upstream-drift cost plus its own reliability surface (it's on a pool-exhaustion fix branch, the same connection-health theme as P0 **G1**). The Memory verdict (§5) is two halves: the async *client* in `automatos-ai` (PRD-141 Phase 1) and the forked *server* here.

---

## 5. Per-subsystem verdict

| Subsystem | Grade | Verdict | Primary evidence |
|---|---|---|---|
| **Missions** (Sequential Coordinator 82A) | A− | **KEEP** | DB-authoritative, 5s tick, restart-durable. The reference implementation for the whole platform. Debt: `coordinator_service.py` 3,020 lines (split); retry doesn't feed verifier critique (G5). |
| **Routing** (UniversalRouter) | B | **KEEP** | Tiers 0–3 clean. Minor: negative routing signals (already in PRD-141-reliability Phase 4). |
| **Channels** (11 adapters) | B | **KEEP** | Consistent adapter pattern, no structural debt. |
| **Widgets** | B | **KEEP** | Vertical-agnostic refactor fully landed (PRD-141-widget). Debt: hardcoded `users.id=1` at `api/widgets/chat.py:89` (G14). |
| **Agents / Factory** | B− | **KEEP** | Already rewritten clean (2,716→1,507 lines). Agents = DB rows. Healthy. |
| **Chat** | C | **HARDEN** | Contract right (the Arc). `consumers/chatbot/service.py` 2,276-line god service; **two tool loops** vs AgentFactory (G6); 0.00 test ratio. |
| **Memory** | C | **HARDEN** | 3 real layers, sound. `unified_memory_service.py` 2,202 lines; dual L3 write paths (G12); Mem0 sync→async mid-flight (PRD-141-reliability Phase 1). L3 *server* is a self-hosted fork (`automatos-mem0` — see §4A); this verdict covers the async **client** in `automatos-ai`, the fork carries its own. |
| **RAG / KB** | C+ | **HARDEN** | Genuinely strong retrieval (RRF + rerank + parent-expand + knapsack). `documents.py` 1,924 lines; 0.05 test ratio. |
| **NL2SQL** | C | **HARDEN** | Works. Credential-handling discipline (D4); test-thin. Lower priority. |
| **Tools** | C+ | **HARDEN** | 3-file pattern + ActionRegistry + prefix dispatch is the right contract. `tool_registry.py` 1,528 lines. |
| **Config / Secrets / Authz** | C− | **HARDEN (P0 slice)** | `os.getenv` outside config.py (G7); **fail-open authz** (G3). Sweep + flip to fail-closed. |
| **Data / Sessions / Migrations** | D+ | **HARDEN (P0 slice)** | `get_db()` never commits/rolls back → 9hr idle-in-tx blocks DDL (G1); migrations one giant txn, no lock_timeout (G2). Contract fine; plumbing leaks. |
| **Knowledge Graph** (the moat) | C | **DESIGN DECISION** | Canonical store already single-sourced (`graph_service.py` + `graph_storage.py`, graph-verified). Needs a documented boundary + storage-format scalability path. See §7. |
| **Frontend** | F (tests) | **HARDEN code / REWRITE test posture** | 1 test / 670 files. `api-client.ts` 2,687 lines. Keep every surface; rebuild the test net at enterprise bar (§9). |
| **Playbooks / Workflows / Recipes** | F | **CONSOLIDATE + HARDEN** | One live concept (recipe = playbook) scattered across ~5,400 LOC + a dead `modules/workflows/` twin. Fire-and-forget `asyncio.create_task` dies on restart (G4). Not a rewrite — unify + make durable. See §6. |

Detailed evidence and per-subsystem Definition of Done live in `BRAIN-BLUEPRINT.md` §3 and `GUARDRAILS.md` §H.

---

## 6. The one consolidation — Playbook engine (consolidate + harden)

**Why this is consolidate-and-harden, not a rewrite.** "Recipe" and "playbook" are the **same thing** (recipe = the pre-rename name; the FE now says playbook). The execution logic is **live and working**, just scattered: the loop in `api/recipe_executor.py` (1,997 lines), the launch front door in `api/workflow_recipes.py` (1,872 lines, despite its name), with a dead `modules/workflows/` twin alongside. The *contract is right.* Two fixable defects: (1) it's **scattered** across files that diverge in dedup/retry/streaming — consolidate to one clean flow; (2) the launch path is **not restart-durable** — `asyncio.create_task` fire-and-forget means an in-flight playbook silently dies on a process restart (G4) — harden by porting the Mission durability model. No new engine, no wrong contract → **not a rewrite**. (`PLAYBOOK-ENGINE-DESIGN.md` §1.)

**Why it's feasible (code-verified).** The consolidation surface is large internally — **506 nodes across ~30 files** — but the **execution-launch blast radius is just six call sites** behind two entry points (`launch_recipe_task`, `execute_recipe_direct`): the workflow API (`workflow_recipes.py:905`), the Composio trigger (`composio.py:886`), the workspace webhook (`webhooks.py:682`), the platform-tool executor (`handlers_playbooks.py:487`), the playbook scheduler (`playbook_scheduler.py:208`), and the task reconciler (`task_reconciler.py:273`). The broader ~36-edge / ~21-file graph coupling is mostly shared utils and type imports — `channels/__init__.py` and `action_registry.py` import **no** execution path at all. The engine consolidates behind a stable interface without rippling across the platform.

**The design.** Collapse the triplet into one durable, DB-backed engine that **borrows the Mission coordinator's durability model** (DB tick, restart recovery, state in Postgres not a process dict). Missions already solved this exact problem (KEEP, grade A−); the playbook engine should reuse that pattern rather than invent a third.

Full design — interface, state model, migration order, deletion plan — in **`docs/architecture/PLAYBOOK-ENGINE-DESIGN.md`**.

**Gerard's decision (2026-05-29):** it is **not a rewrite** — "playbooks used to be called recipes before we renamed [on] the front end; workflows was old and mostly should have been removed... is it just scattered and we need to clean up to one clean flow?" Yes. Approved as a **consolidate-and-harden**, *as design* — no code until the build is green-lit.

---

## 7. The moat — knowledge-graph canonical store

The knowledge graph is the strategic moat (the merchant's products, orders, FBT relationships, business memory — the thing that compounds and can't be exported to a competitor). The gap analysis flagged "two graph stores, no single source" (G11). Investigation **downgraded** that concern:

**The moat is already single-sourced.** Business facts live in `workspace_graphs` (the Graphify pipeline → `graph_service.py` + `graph_storage.py`, graph-verified as the canonical writer/reader). Shopify FBT edges are written and read *only* here. There is **zero sync** to the `knowledge_nodes/edges` tables — so there is no active corruption, only a *latent* risk if someone starts dual-writing later. Fix = a documented boundary, not a migration.

**Five graphs, named apart so they never collide again:**

| Concern | Store | Verdict |
|---|---|---|
| Business Knowledge Graph (THE MOAT) | `workspace_graphs` (Graphify) | **CANONICAL** |
| Agent learning / inference | `knowledge_nodes` / `knowledge_edges` | KEEP with hard boundary (learning-only, never business entities); fold into HARNESS; **Wave 4 wires the dead loop — fix, not cut (2026-05-29)** |
| Mission coordination field | `vector_field.py` → Qdrant `field_memory` | KEEP |
| (dead twin of above) | `neural_field/` (112 nodes; **only consumer is the dead workflow-era `AgentExecutionManager`** — see §11) | **CUT** (resolves G13) |
| Code intelligence | `codegraph_symbols` | KEEP (orthogonal) |

**Enterprise-bar flag:** the moat currently serialises the whole graph as a JSON blob loaded into NetworkX *in memory per request*. Fine for a small merchant; won't hold at enterprise scale. Storage-format evolution (blob → queryable edge tables or a graph DB) is a named HARDEN item — not a blocker.

Full taxonomy, the boundary contract, and the scalability path in **`docs/architecture/KNOWLEDGE-GRAPH-CANONICAL.md`**.

**Gerard's decision (2026-05-29):** recommend the canonical store (done above); HARNESS stays (see §8).

---

## 8. HARNESS — kept, gated, sequenced last

HARNESS (PRD-121/140 — the self-learning / self-management loop) **stays**. It is a named differentiator. The earlier "cut-or-quarantine" flag was about the *current implementation*, not the capability: an immature loop that can self-modify (reassign tools, change power modes, auto-rollback) is precisely the thing that destabilises the cores we are hardening.

**Verdict: KEEP, behind `HARNESS_SELF_MANAGEMENT_ENABLED` (default false), hardened last** — after the cores soak. This is already how PRD-141-reliability Phase 5 and the Wave 4 sequencing read. The `knowledge_nodes/edges` learning graph (§7) is its natural substrate — a dead loop today, **fixed (wired) in Wave 4, not cut** (decided 2026-05-29).

**Gerard's decision (2026-05-29):** keep HARNESS, full scope, gated and last.

---

## 9. Frontend — enterprise test bar

The frontend is grade F on **test debt alone** (1 test / 670 files). The components are sound and **every surface stays** (Command Center, Activity, kanban, analytics, widgets — per the standing rule). This is a *test-net rebuild*, not a feature rewrite, plus splitting the 2,687-line `api-client.ts`.

**Standard: production-ready enterprise.** Full Playwright coverage across all ten golden journeys (J1–J10 in `TEST-PLAN.md`), not a critical-path subset. Vitest (present, unused) is wired for unit/component; Playwright for E2E.

**Gerard's decision (2026-05-29):** full Playwright, enterprise bar — locked.

### 9A. Cross-repo test posture — one bar across seven repos

"Production-ready" is a property of the *platform*, not of `automatos-ai` alone (§4A). Three rules close the cross-repo test gap:

1. **Tests co-locate with the code they cover.** Each repo owns its own unit/integration suite next to its source — `automatos-shopify` (0 tests) and `automatos-voice` (0 tests) get suites; `automatos-widget-sdk` (already has CI) is the template; the core suites land in Wave 2.
2. **Contract tests guard the real seams.** The integration boundaries between repos have *no* tests today and that is where drift hides: SDK ↔ widget API (the `page_context` change on the current branch is exactly this seam), Shopify app ↔ core widget API, skills ↔ platform loader, voice ↔ core. These get versioned contract tests in the consuming repo's CI.
3. **`automatos-testing` is harvested, then retired.** It is a black-box, prod-hitting, 5-phase smoke harness that is **9 months stale** (master last 2025-08-31), tests a *renamed* surface (`/api/workflows/*`, pre-Mission/Playbook) and the now-**dead** `neural_field`. It does **not** rescue the test gap. Action: **harvest** its five journey definitions as golden-journey seeds (they map onto J1–J10), re-home live smoke as a **post-deploy CI check**, then **retire the repo**. Do not revive it as a standalone stale mirror.

---

## 10. The three P0s that gate everything

These are not subsystem rewrites. They are urgent and they block the rest of the program:

| ID | Defect | Location | Why it gates |
|---|---|---|---|
| **G1** | `get_db()` never commits/rolls back → 9hr idle-in-tx | `core/database/database.py:105` | Blocks DDL/deploys. Every migration in the hardening plan stalls behind it. |
| **G2** | Migrations wrapped in one transaction, no `lock_timeout` | `alembic/env.py:31` | Combined with G1, deploys hard-block. |
| **G3** | Fail-open authz (`_check_agent_permission`, `validate_composio_action` return True on error) | per `GUARDRAILS.md` E3 | Latent privilege escalation. Must fail closed before anything ships. |

Fix these first or the rest of the work fights them.

---

## 11. CUT list (dead weight — graph-verified)

No functionality is removed; these are dead code, superseded scaffolding, and duplicate paths. Per Gerard's standing rule, **verify every match before deletion**.

| Item | Evidence | Action |
|---|---|---|
| `core/neural_field/` (6 files, 112 nodes) **+ `AgentExecutionManager`** (`execution_manager.py`) | Workflow-era unit: neural_field's **only** importer is `AgentExecutionManager.execute_workflow_subtasks`, which is **never instantiated** (the `AgentService.execution_manager` property that builds it is never read) and **has zero callers** (code-verified 2026-05-29). Superseded by `AgentFactory.execute_with_prompt`. | **CUT as one unit** (resolves G13) |
| `api/chatbot_llm.py` (547 lines) | Mounted "legacy" `main.py:1052`; **6 inbound edges from 5 files** (graph-verified) | **CUT after migrating** `recipe_executor`, `board_tasks`, `pandas_ai_service` + 2 others |
| `_stream_workflow_bridge` (`chat.py:37`) | Zero call sites (Explore-confirmed; graph predates it) | **CUT** (re-verify on current branch) |
| `chatbot_router` | Still mounted in `main.py` | **CUT** with `chatbot_llm` |
| `seed_recipes_marketplace_v2.py` | Superseded seed | **CUT** |
| ~53 / 200 dead tables | Graph-mining (2026-04-10 snapshot) | **CUT after re-verify** (graph stale; confirm on live schema) |
| ~198 unreferenced routes | Graph-mining | **CUT after re-verify** |
| Context Forge remnants (PRD-33/34/35) | Superseded by Composio (PRD-36) | **CUT** |
| PRD-20 400+ MCP scaffolding, PRD-12 stubbed Playbook miner | Superseded / never wired | **CUT** |

The cut list is sequenced **after** the test net (Wave 5), so tests guard the survivors before deletions land.

---

## 12. Execution roadmap

Maps onto the six waves from the `prd142-rock-solid-plan` memory, now anchored to verified verdicts. Each wave ends with a code-reviewer gate; risky waves add a canary soak.

| Wave | Theme | Contents | Gate |
|---|---|---|---|
| **0** | Measurement first (~1wk) | Extend PRD-141 Phase 0 telemetry into one "Is it working?" dashboard: activation, mission success, per-primitive health, error rate by subsystem, widget engagement. | Dashboard answers the founding question with a number. |
| **1** | Stop the bleeding (~2wk) | **P0s: G1 sessions, G2 migrations, G3 authz** (§10). Unblock deploys, close the security hole. | Deploys unblocked; authz fails closed; canary soak. |
| **2** | Test net (~2–3wk) | Golden journeys J1–J10 (`TEST-PLAN.md`); contract tests on hot routers; **full Playwright frontend** (§9). | 100% of golden journeys covered. |
| **3** | Primitive hardening (~3–4wk) | Each HARDEN subsystem against its `GUARDRAILS.md` §H Definition of Done: split god-files (chat, memory, RAG, tools), unify the tool loop (G6), sweep `os.getenv` (G7), moat boundary + storage path (§7). | Per-primitive dashboard tile green. |
| **3R** | The consolidation (within Wave 3) | Playbook engine consolidate + harden (§6, `PLAYBOOK-ENGINE-DESIGN.md`). Build behind interface, migrate the 6 launch call sites, consolidate the scattered engine, delete the dead `modules/workflows/` twin. | Restart-durability test passes; dead paths deleted. |
| **4** | Self-learning / HARNESS (gated, last) | PRD-141-reliability Phase 5; `knowledge_nodes/edges` learning graph wired. Flag-gated, after soak. | Canary on one workspace; rollback verified. |
| **5** | Execute the cut list (~2wk) | Execute the §11 cut list (tests now guarding survivors); remove dead tables/routes/modules. **Shopify hero workflows + one-click onboarding are UNBUNDLED (2026-05-29) → separate `automatos-shopify` vertical PRD; out of scope for this core review.** | Cut greps return zero; survivors green under the test net. |

~12 weeks, zero new features, a number on "working" from week one.

---

## 13. Definition of Done (per subsystem)

A subsystem is "production-ready" when it meets its `BRAIN-BLUEPRINT.md` §3 contract **and** the `GUARDRAILS.md` §H checklist:

- [ ] Golden-journey test exists and passes (`TEST-PLAN.md`).
- [ ] Failure path tested — degrades or errors visibly, never silently.
- [ ] Restart-safe — no user-visible work lost on process restart.
- [ ] Observable — emits the telemetry the dashboard needs.
- [ ] Tenant-isolated — proven by a cross-workspace test.
- [ ] One source of truth — no dual write paths.
- [ ] Dashboard tile — a number answering "is this primitive working right now?"

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| Playbook consolidation regresses a live execution path | Build behind a stable interface; strangler-fig migration of the 6 launch call sites; restart-durability test before any delete; dead paths deleted only after parity proven. |
| Graph snapshot is stale (2026-04-10, pre-PRD-141) | Cut-list deletions (dead tables/routes) re-verified against the live schema before execution. Recommend `/graphify --update` before Wave 5. |
| `os.getenv`/bare-except sweeps introduce regressions | Opportunistic, not big-bang (per PRD-141-reliability precedent); each change is a pure widening, code-reviewer gate. |
| HARNESS destabilises hardened cores | Flag-gated, default false, Wave 4 only, canary + rollback (§8). |
| Frontend rebuild scope creep | Scope is tests + `api-client.ts` split only — no component rewrites; surfaces preserved. |
| Cross-repo contract drift (SDK ↔ widget API, Shopify ↔ core, skills ↔ loader, voice ↔ core) | Versioned contract tests on each seam in the consumer's CI (§9A); the stale central harness is retired, not relied on. |
| Mem0 fork (`automatos-mem0`) drifts from upstream / carries its own reliability surface | Pin an upstream baseline + document a fork-maintenance cadence; the pool-exhaustion fix it's mid-flight on is the same connection-health theme as P0 **G1** — treat as P0-adjacent. |

---

## 15. Success metrics

| Metric | Current | Target | How measured |
|---|---|---|---|
| Idle-in-tx events blocking DDL | recurring | 0 | connection-leak test + deploy logs |
| Fail-open authz branches | ≥2 | 0 | grep gate (every authz else-branch denies) |
| Playbook engines | 3 | 1 | grep for the triplet routers after 3R |
| Playbook restart-durability | dies on restart | recovers from DB | restart test (US in build PRD) |
| Golden journeys covered | ~0 | 10 / 10 | `TEST-PLAN.md` J1–J10 |
| Frontend tests | 1 / 670 | enterprise Playwright suite | CI |
| Business-graph stores (moat) | 1 canonical + 1 latent | 1 canonical + bounded learning graph | boundary doc + grep |
| Dead code (neural_field + execution_manager, chatbot_llm, bridge) | present | deleted | cut greps return zero |
| "Is it working?" dashboard | none | live from Wave 0 | Grafana tile per primitive |

---

## 16. What's locked vs open

**Locked (Gerard, 2026-05-29):**
- Overall: fix-not-rewrite — **zero rewrites**; the Playbook consolidate+harden is the biggest single piece (§6), as design only.
- HARNESS kept, gated, last.
- Frontend at full-Playwright enterprise bar.
- Moat canonical = `workspace_graphs`; recommendation accepted.

**Open for discussion before the build PRD:**
- Playbook engine interface shape (`PLAYBOOK-ENGINE-DESIGN.md` is the proposal).
- `knowledge_nodes/edges` fate — **RESOLVED 2026-05-29: fix, not cut.** Wave 4 (HARNESS) wires the dead learning loop; keep the tables/schema as the starting point, reshape only if HARNESS requirements demand.
- Whether to run `/graphify --update` before Wave 5 to refresh the cut-list evidence.
- Cross-repo test strategy: co-locate per-repo + contract tests on the seams, harvest-then-retire `automatos-testing` (recommended, §9A) vs reviving the central harness.
- Mem0 fork strategy: how long to carry `automatos-mem0` as a fork — pin-and-maintain, upstream our fixes, or track a cadence.

---

**This PRD is the decision. The build PRDs (per wave) come after it's approved.**
