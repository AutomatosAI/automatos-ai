# PRD-142 Wave 5 — Execute the Cut List (Delete the Dead Weight, Under Green)

> **Parent:** `PRD-142-CORE-DESIGN-REVIEW.md` §11 (the graph-verified CUT list), §12 (the Wave 5 row — *"Execute the §11 cut list; remove dead tables/routes/modules; cut greps return zero; survivors green under the test net"*), §10 (success metric: dead code deleted), and the standing rule at §11: **verify every match before deletion.**
> **Design companions:** `CLAUDE.md` §5 (replace cleanly — delete what's superseded; remove orphan imports, unused files, dead routes) and `GUARDRAILS.md` B2 (delete what you replace). The refreshed evidence base: `graphify-out/` (regenerated 2026-06-09).
> **Status:** Build-PRD — drafted 2026-06-09. **Gate satisfied:** Wave 5 is sequenced **after** the test net (Wave 2, merged) and the primitive hardening (Wave 3, merged) precisely so the tests guard the survivors before deletions land (parent §11). Waves 0–4 are on `main`.
> **Type:** **Delete, under green, verify-before-cut. Zero functionality removed** — only dead code, superseded scaffolding, and duplicate paths. No refactors of survivors, no new features, no new primitives.
> **Verified against:** worktree `automatos-ai-wave4` @ `main` (post-Wave-4). Graph refreshed 2026-06-09 (scoped to `orchestrator/`: 16,361 nodes / 57,019 edges, in `graphify-out/`). **The graph is the starting point, not the verdict** — degree (what a node calls) ≠ liveness (whether anything live reaches it); every cut is confirmed by **inbound-reachability grep** on the live branch.
> **Depends on:** Wave 2 (the test net — the guardrail that makes deletion safe) + Wave 3 (the hardened survivors). The Wave 4 S13 cut (KnowledgeGraph/LearningEngine/HierarchicalMemorySystem) is the proving precedent — it showed a stale "0 importers" claim was wrong and that an `api/` symbol was a false-positive, so live re-verify is mandatory, not optional.
> **Reuse-first:** N/A by inversion — this is the **deletion** wave. The only "build" is migrating the handful of live callers of an otherwise-dead shim (`chatbot_llm`) onto the canonical path before it's cut.
> **Ralph config:** authored on approval (`scripts/ralph/prd-142-wave5.*`, the Wave 0 three-file pattern). **Human-gated, excluded from the autonomous loop:** the DB table DROPs (prod migrations, per Gerard's standing rule), and any cut whose caller-migration needs a sign-off.

---

## 1. The founding question for Wave 5

- **Wave 0** — *can we measure it?* Yes (Command Center vitals).
- **Wave 1** — *can we stop the bleeding?* Yes.
- **Wave 2** — *can we prove it stays working?* Yes (J1–J10 + a required CI gate).
- **Wave 3** — *is each primitive rock solid?* Yes (hardened under the net).
- **Wave 4** — *can Auto safely manage and improve itself?* Yes (HARNESS wired + governed, behind flags).
- **Wave 5** answers *"can we now safely delete the dead weight?"* — the workflow-era scaffolding, the legacy shims, the superseded modules and duplicate paths the design review inventoried — and **prove**, by green tests and zero cut-greps, that nothing live depended on them.

The cut list is **graph-verified but on a 2026-04-10 snapshot — stale by five PRDs**. So the wave's discipline is: **refresh the graph (done 2026-06-09), then verify every match by inbound reachability before each delete.** One item is already confirmed: `core/neural_field/` + `AgentExecutionManager` are genuinely dead on the live branch — the lazy `AgentService.execution_manager` property that builds AEM is **never accessed anywhere** (cross-`orchestrator` grep empty), `execute_workflow_subtasks` has **zero callers**, and `neural_field`'s only importer is inside that dead unit (`execution_manager.py:184`).

## 2. What Wave 5 **is** — and is **not**

**IS:** execute the §11 cut list; per-item **inbound-reachability** verification on the live branch; **migrate-then-cut** the few items with live callers; delete dead modules, routes, and tables; remove orphaned imports/exports/mounts; keep the test net green throughout; drive every cut to a **zero cut-grep**.

**IS NOT:** removing any live functionality; refactoring or "improving" the survivors; the Shopify hero-workflow / one-click-onboarding work (**unbundled by the parent → separate `automatos-shopify` vertical PRD; out of scope**); new features, primitives, or providers; trusting graph **degree** as a kill/keep verdict; running prod **DB DROPs** autonomously (human-gated).

## 3. The deletion contract — what "done" means (per item)

A cut is "done" only when **all** hold:
1. **Inbound reachability is zero on the live branch** — no instantiation sites, call sites, route mounts, or live-schema reads — proven by grep, **not** by graph degree or "looks orphaned."
2. **External callers checked** — for routes, that no webhook / SDK / external consumer hits them (edges the internal graph can't see).
3. **The few live callers are migrated first** — to the canonical path, each tested green, **before** the shim is unmounted.
4. **The delete is clean** — the file/module/route/table goes, and so do its orphaned imports, exports, mounts, and seeds (`CLAUDE.md` §5).
5. **The test net stays green** (`orchestrator-tests` CI) and the **cut-grep for the symbol returns zero**.
6. **DB DROPs additionally:** a reversible, reviewed migration, **applied by a human** (never by the loop).

## 4. Current-state map — the §11 cut list with refreshed evidence (2026-06-09)

| # | Item | §11 evidence | Refreshed status | Phase |
|---|---|---|---|---|
| 1 | `core/neural_field/` + `AgentExecutionManager` | only importer is AEM; never instantiated; zero callers | **VERIFIED DEAD** — `execution_manager` @property never accessed; `execute_workflow_subtasks` 0 callers; neural_field imported only at `execution_manager.py:184` | 1 |
| 2 | `api/chatbot_llm.py` (547 ln) + `chatbot_router` | legacy mount; 6 inbound edges / 5 files | chatbot_llm = legacy shim (graph rationale edge flags it); `chatbot_router` **not found** in orchestrator-scoped graph → grep-confirm | 2 |
| 3 | `_stream_workflow_bridge` (`chat.py`) | zero call sites | graph showed **outbound** edges only (degree); **inbound not yet verified** | 1 (if grep confirms 0 inbound) |
| 4 | `seed_recipes_marketplace_v2.py` | superseded seed | `_v2` not found; non-v2 seed present | grep-confirm `_v2` → cut |
| 5 | Context Forge remnants (PRD-33/34/35) | superseded by Composio | **not found** (orchestrator scope) | 3 (grep full tree) |
| 6 | PRD-20 MCP scaffolding + PRD-12 Playbook miner | superseded / never wired | **not found** | 3 (grep full tree) |
| 7 | ~198 unreferenced routes | graph-mined, stale | needs live re-mine + external-caller check | 4 |
| 8 | ~53 dead tables | graph-mined, stale | needs live-schema read-verify | 5 (human-gated) |

> Counts (7, 8) are **estimates from a stale snapshot** — they are re-derived on the live branch in Phase 0, not taken as given.

## 5. Verification map (read before deleting a line)

This is a deletion wave, so the "map" is the **method**, not reuse:
- **The refreshed graph** (`graphify-out/graph.json`, `GRAPH_REPORT.md`) is the candidate-finder — it surfaces orphans and edges, nothing more.
- **Inbound-reachability grep is the verdict.** For each symbol: instantiation sites (`X(`), call sites, property/attribute reads, route mounts, model reads. For modules: importers outside themselves. The Wave-4 S13 pattern.
- **Degree ≠ liveness.** A node with high degree can be dead (it calls common functions / has internal edges but nothing live reaches it). The 2026-06-09 Haiku graph pass mislabeled neural_field/AEM/the bridge "active" on degree alone — every such call must be re-checked by reachability.
- **Routes need external-caller checks** (webhooks, SDK, frontend) that the internal graph can't see.
- **Tables need live-schema read/write verification** (the analytics learning-tile reads `knowledge_nodes` — proof that "graph-mined dead" can be wrong).

## 6. Workstreams & user stories

### WS-V — Phase 0: refresh + per-item verification *(do FIRST — produces the kill/keep list)*
- **US-V1** — Graph refreshed (done 2026-06-09, `graphify-out/`). *(complete)*
- **US-V2** — For each of the 8 items, run the inbound-reachability greps on the live branch; produce a per-item **kill / migrate-first / keep / already-gone** verdict with the grep evidence. No deletions in this WS.
- **AC:** every item has a recorded verdict + evidence; items marked "keep" are removed from the wave with a one-line reason.

### WS-W — Clean deletes *(verified zero-caller; no migration needed)*
- **US-W1** — Cut `core/neural_field/` + `AgentExecutionManager` (+ the dead `execution_manager` property + `execute_workflow_subtasks`) as **one unit** (resolves G13).
- **US-W2** — Cut `_stream_workflow_bridge` (only if US-V2 confirms zero inbound).
- **US-W3** — Cut `seed_recipes_marketplace_v2.py` (if confirmed superseded).
- **AC per story:** cut-grep returns zero; `orchestrator-tests` green; orphan imports removed.

### WS-X — Migrate-then-cut: the legacy LLM shim
- **US-X1** — Migrate the live callers of `api/chatbot_llm.py` (recipe_executor, board_tasks, pandas_ai_service + the rest US-V2 finds) to the canonical LLM service; test each.
- **US-X2** — Unmount + delete `chatbot_llm.py` and `chatbot_router`; cut-grep zero.
- **AC:** no caller references the shim; chat + recipe + board-task paths green.

### WS-Y — Superseded scaffolding
- **US-Y1** — Grep the **full tree** (not just orchestrator) for Context Forge remnants (PRD-33/34/35); delete if superseded by Composio.
- **US-Y2** — Same for PRD-20 MCP scaffolding + PRD-12 stubbed Playbook miner.
- **AC:** cut-greps zero; Composio paths unaffected.

### WS-Z — Dead routes
- **US-Z1** — Re-mine unreferenced routes on the live branch; for each, confirm zero inbound **and** no external/webhook/SDK consumer; unmount.
- **AC:** route greps zero; the J1–J10 journeys + API smoke stay green.

### WS-AA — Dead tables *(human-gated)*
- **US-AA1** — Re-verify zero reads/writes on the **live schema** for each candidate table; author reversible DROP migrations.
- **US-AA2 (human)** — Apply the migrations on prod (Gerard); the loop never runs them.
- **AC:** migration verified against the live head chain; survivors green; no read references remain.

## 7. Sequencing & gates

Phase 0 (WS-V, verify) **first** — nothing is deleted until its verdict is recorded. Then, easiest→riskiest: **WS-W** (clean deletes) → **WS-X** (migrate-then-cut) → **WS-Y** (scaffolding) → **WS-Z** (routes) → **WS-AA** (tables, last + human-gated). Each story ends with **CI green + cut-grep zero**; each WS ends with a **code-reviewer** gate. Table DROPs depend on their routes being cut first (WS-Z before WS-AA). **Exit gate (parent §12):** *cut greps return zero; survivors green under the test net.*

## 8. Deletions / cleanups (the wave itself)

Every WS is a deletion. The only "replace then delete" is WS-X (migrate the `chatbot_llm` callers, then delete the shim + router — `CLAUDE.md` §5). Remove orphaned imports, exports, route mounts, and seed entries alongside each cut.

## 9. Out of scope

- **Shopify hero workflows + one-click onboarding** — unbundled by the parent (2026-05-29) → separate `automatos-shopify` vertical PRD.
- Refactoring or "tidying" the **survivors** (Wave 5 deletes dead code; it does not touch live code beyond the WS-X caller migration).
- The `jobs/` and `integrations/` modules the refreshed graph flagged as low-connectivity — **not on the §11 list**; auditing them is a separate exercise, not this wave (added only on explicit decision — §12).
- Any new feature, primitive, provider, or moat/business-graph change.

## 10. Success metrics

- **Cut-greps return zero** for every executed item (the parent's literal exit gate).
- **Survivors green** under the test net on every cut (`orchestrator-tests` + `check-shopify-isolation`).
- `neural_field` + `execution_manager` + `chatbot_llm` + the bridge greps return zero (parent §10 dead-code row).
- Dead-route count and dead-table count **measurably reduced** against re-verified live numbers (not the stale estimates).
- **Zero prod incidents** attributable to a cut (the verify-before-delete discipline holding).

## 11. Risks

| Risk | Mitigation |
|---|---|
| **Deleting live code** (stale graph / degree-not-liveness) | Phase 0 inbound-reachability verify per item + CI gate. The S13 false-positive proves this is real. |
| **DB DROPs are irreversible in prod** | Human-gated, reversible migrations, live-schema read-verify first; tables cut **after** their routes (WS-AA last). |
| **`chatbot_llm` migration regresses its callers** | Migrate + test each caller before unmounting; chat/recipe/board-task journeys green. |
| **A route cut breaks an external/webhook/SDK consumer** invisible to the internal graph | Explicit external-caller check in WS-Z, not just internal edges. |
| **"Not found" false confidence** (the 2026-06-09 graph was scoped to `orchestrator/`) | Grep the **full monorepo** for the "not found" items (Context Forge, MCP, chatbot_router) before declaring them gone. |
| **Cut-grep false-zero** from a renamed symbol | Grep the symbol AND its known aliases / string references (the S13 `prompt_analyzer` string-ref lesson). |

## 12. Open decisions (for Gerard — settle before the relevant WS)

1. **DB DROPs human-gated?** Recommend **yes** — I prep + verify the migrations, you apply on prod (matches your standing rule). *(WS-AA)*
2. **`jobs/` + `integrations/` low-connectivity modules** — in scope or out? They're **not** on the §11 list; recommend **out** (separate audit) unless you want them folded in. *(affects scope)*
3. **`chatbot_llm` canonical target** — confirm the migration target is the current LLM service (`create_llm_manager`) so WS-X migrates callers to the right path. *(WS-X)*
4. **Full-tree re-graph?** The 2026-06-09 refresh was `orchestrator/`-scoped; re-run graphify across the full monorepo for the "not found" items, or settle them by grep? Recommend **grep** (cheaper, sufficient for absence checks). *(WS-Y)*
