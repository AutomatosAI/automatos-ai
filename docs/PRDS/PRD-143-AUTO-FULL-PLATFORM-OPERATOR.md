# PRD-143: Auto as Full Platform Operator (Phase 1 — The Enabler)

> **Status:** Implemented on `ralph/prd-143-operator-obs-lock` — pending Gerard review + obs-manifest sign-off (Open Q1/Q2). 2026-06-10.
> **Revision 2 (2026-06-09):** Security model **inverted** per the all-in autonomy pilot decision (memory `pilot-all-in-autonomy`). Auto now gets **full access to every platform API** — including platform-administrative ones. The **only** hard-excluded tier is **observability/oversight**, locked to `system_role="super_admin"` (Gerard, the sole super admin). Rationale: in a low-blast-radius pilot, exclusions teach nothing — but the **watchtower stays human**. Auto must never read, write, or game its own oversight channel, and no other principal needs platform internals.
> **Builds on (does NOT rebuild):** PRD-142 Wave 4 (HARNESS / autonomy / structured-store foundation), PRD-138/139 (tool-routing graph + semantic selection), PRD-140 (hierarchy permissions), PRD-120 (skills). The autonomy dial, HARNESS loop, audit, rollback, and the tool-routing graph already exist — this PRD **extends** them.
> **Type:** Net-new capability **on existing rails**. Phase 1 = the **enabler**; the full ~200-tool catalogue, the full skills library, and additional concierge journeys are **follow-on PRDs**.
> **Verified against:** `main`, code reads 2026-06-09 (graphify AST graph rebuilt: 19,996 nodes / 63,575 edges).
> **Pilot context:** mostly-test workspaces, low blast radius — open it up to learn. Gerard merges the gate-changing PR himself; the §12.3 release gate + security review remain (memory `pilot-all-in-autonomy`).

---

## 1. Introduction / the problem

Automatos is deep — 116 SQLAlchemy tables, 103 routers, agents/missions/playbooks/channels/widgets — and that depth reads as **"too hard to use."** The fix is not a simpler platform; it's a smarter front door: **Auto** becomes the **operator/concierge** that removes the difficulty. A user says *"set up my Shopify store,"* *"manage my social,"* *"be my assistant,"* and Auto — knowing the whole system — **calls the platform's own APIs to actually do it**, then recommends the channels/tools to keep it running.

That requires three things the platform doesn't fully have yet: (1) **enough tools** — Auto can only operate the parts of the platform exposed as tools (~120 today, target ~200, wrapping *almost every platform API*); (2) **the right tools per request** — you cannot put 200 tool definitions in an LLM context, so semantic + graph **tool selection** is mandatory, not optional; (3) a **fully-autonomous concierge loop** that plans and executes end-to-end.

Phase 1 delivers the **enabler** for all three, plus the **one hard safety boundary** of the revised model: **observability and the oversight control plane are super-admin only** (Gerard). Everything else — agent CRUD, workspace management, channels, billing, user/role management, the lot — is operator-tier and Auto-reachable under `autonomy=full`, protected by gates-and-logs (PRD-140 hierarchy checks, audit, HARNESS rollback, kill-switch) rather than exclusion.

---

## 2. Goals

- **Full operator surface for Auto:** Auto can reach **every platform API** exposed as a tool, at full autonomy — including the formerly admin-gated ones. No platform capability is exclusion-gated from Auto **except** the observability tier.
- **The observability lock (the safety rail):** introduce a **super-admin tier** covering obs tools + obs HTTP routers, hard-locked to `system_role="super_admin"`, fail-closed, **independent of the autonomy dial**. Auto cannot see its own watchtower; users cannot see platform internals.
- **Tool breadth:** a repeatable mechanism to wrap the platform's own APIs as platform tools (the 3-file pattern, scaffolded systematically), scaling ~120 → ~200, plus a first meaningful batch of operator tools.
- **Selection at scale (the enabler):** semantic selection is already on (`SEMANTIC_TOOL_ROUTING=true`, top-K 15); make the **graph learning layer genuinely useful** by **seeding it from telemetry**, and prove selection stays correct at ~200 tools.
- **Concierge MVP:** one real journey executed **fully autonomously** end-to-end — plan → call platform APIs → set up → recommend.
- **On the Wave 4 foundation:** reuse autonomy=full, HARNESS, audit, and rollback — extend, never rebuild.

---

## 3. What this PRD **is** — and is **not**

**Is** (Phase 1 enabler):
- The **observability/super-admin lock** (tools + routers) and the **opening of everything else** to Auto at full autonomy.
- A **tool-scaffolding mechanism** + a first batch of new operator tools toward ~200.
- **Graph cold-start by telemetry seeding** + selection robustness/observability at scale.
- **One fully-autonomous concierge journey** (MVP), end-to-end, audited.

**Is not:**
- **The full ~200-tool catalogue** — Phase 1 builds the mechanism + a batch; the long tail is follow-on PRDs.
- **The full skills library** — Phase 1 skills Auto only enough for the MVP journey (PRD-120 expansion is separate).
- **Composio / external-app work** — platform tools wrap **Automatos's own APIs**; Composio (856 external apps) is a different, existing surface and out of scope here.
- **Rebuilding HARNESS, the autonomy dial, or the tool-routing graph** — PRD-142 Wave 4 / PRD-138/139 own those; this PRD consumes them.
- **Exposing the observability tier** to Auto or non-super-admin users — explicitly forbidden (FR-3).
- **Removing the gates-and-logs safety net** — PRD-140 hierarchy checks, the destructive backstop, audit, and HARNESS rollback all still run on every operator-tier call. Open ≠ unguarded.
- **A full concierge frontend** beyond what the MVP journey needs.

---

## 4. Current-state map (verified 2026-06-09)

| Piece | State | Implication for Phase 1 |
|---|---|---|
| Semantic selection | `SEMANTIC_TOOL_ROUTING=true`, `TOP_K=15` (`config.py`) — **already on** | LLM already gets top-15, not all tools. Keep; tune for ~200. |
| Graph learning layer | `signal_recorder` → `tool_routing_edges/affinities` → `edge_builder` (nightly) → `graph_router`; **under-seeded** = cold-start | Seed from telemetry (WS-C). |
| Platform tools | **~120** actions, 3-file pattern (`modules/tools/discovery/`) | Scaffold the gap to ~200 (WS-B). |
| Tool gating | `ActionDefinition.admin_only: bool` (`action_registry.py:38`) + `exclude_admin` filters (`:125-339`). **Only 7 actions are `admin_only=True` today** — 6 are obs (`platform_query_loki_logs`, `platform_query_prometheus`, `platform_get_alerts`, `platform_get_logs`, `platform_list_services`, `platform_get_system_health`) + `platform_set_autonomy_level` | The admin tier is **already ~= the obs tier**. WS-A formalizes that: obs → `super_admin_only`, everything else → operator. |
| Executor gate | `platform_executor.py:506-523` — admin gate; **`full_autonomy → is_admin=True` bypass at `:510`** | Under the new model this bypass is **correct** for operator tools and **must not** apply to `super_admin_only` (WS-A). |
| Tool surface | `tool_router.py:398/405/439` — `exclude_admin=not is_admin`; PRD-122 workspace-owner fallback (`:358-372`) auto-grants admin surface | Operator tools flow to Auto already; `super_admin_only` must be excluded here regardless of `is_admin`/autonomy. |
| RBAC | `system_role` incl. `"super_admin"`; `is_admin = system_role in ("admin","super_admin")` (`core/auth/hybrid.py:694`); **API keys map to `system_role="admin"`** (`:783`) | The super-admin gate must check `== "super_admin"` exactly — `is_admin` and API keys must NOT pass. No `require_super_admin` helper exists yet — build one canonical dependency. |
| Obs HTTP routers | `heartbeat`, `analytics`, `analytics_api`, `analytics_real`, `analytics_charts`, `llm_analytics`, `memory_stats`, `statistics`, `composio_analytics`, `database_analytics`, `execution_history`, `kpi_api`, `reports` — today gated only by `get_request_context_hybrid` (**any authenticated workspace user can read them**) | The router-level obs lock is **new work** (US-002b) — the old draft had none. |
| Telemetry | `tool_execution_logs` feed `edge_builder` | The seed source for cold-start. |
| Wave 4 | autonomy dial (full/standard), HARNESS, audit, rollback | The concierge's execution + safety substrate; the kill-switch. |

---

## 5. Reuse map (read before building)

| Need | Reuse this |
|---|---|
| New platform tools | The **3-file pattern** — `actions_*.py` + `handlers_*.py` + register in `platform_actions.py` (the only sanctioned extension point). |
| Tool tiering | `ActionDefinition.admin_only` + the `exclude_admin` filters — **extend** with `super_admin_only`, don't replace. |
| Selection | `ActionSemanticIndex` (semantic), `graph_router`, `smart_tool_router` (US-014 delegation), `signal_recorder`/`edge_builder` (PRD-138/139). |
| Cold-start seed | `tool_execution_logs` + the existing `edge_builder` recompute — add a backfill/seed path. |
| Concierge execution | The Arc — `AutoBrain` / `tool_router` planning + execution; the autonomy dial (full) + HARNESS audit/rollback (Wave 4). |
| Authz | `system_role="super_admin"` (`hybrid.py`) + `get_request_context_hybrid` — add ONE canonical `require_super_admin` FastAPI dependency in `core/auth` and reuse it on every obs router. |
| Per-call safety on the now-open surface | PRD-140 hierarchy permission check (`platform_executor.py:569+`), destructive backstop (`:676`), Wave 4 audit + rollback. |

---

## 6. User stories

### WS-A — The observability lock + full operator surface *(do FIRST — it's the safety rail)*

**US-001: Add a `super_admin_only` tier to the registry.**
*As the Super admin, I want observability tools hard-locked to me, so Auto and users can never read or game the oversight channel.*
- [x] Add `super_admin_only: bool = False` to `ActionDefinition` (`action_registry.py`), alongside the existing `admin_only`.
- [x] Registry selection/listing paths (`to_first_class_schemas`, `to_dispatcher_schema`, `build_prompt_summary`, semantic + graph selection) exclude `super_admin_only` actions unless the caller's `system_role == "super_admin"` — fail-closed (unknown/absent role → excluded).
- [x] `PlatformActionExecutor` refuses a `super_admin_only` action **before execution** unless `caller_context.system_role == "super_admin"`. The check runs **before and independent of** the `full_autonomy → is_admin=True` bypass (`platform_executor.py:510`) and the PRD-122 workspace-owner fallback (`tool_router.py:358-372`) — neither may satisfy it. **API-key principals (`system_role="admin"`, `hybrid.py:783`) must NOT pass.**
- [x] When Gerard himself is the driving principal (chat `caller_context` carries `system_role="super_admin"`), Auto MAY invoke obs tools on his behalf — the boundary is the principal, not the channel.
- [x] Tests: `test_super_admin_tool_excluded_from_auto_surface_at_full_autonomy`, `test_super_admin_tool_refused_for_admin_and_api_key`, `test_super_admin_principal_can_invoke`, `test_full_autonomy_bypass_does_not_cross_su_gate`. Typecheck passes.

**US-002: Reclassify the catalogue — obs locked, everything else open.**
*As the Super admin, I want the obs tier explicit and everything else operator-reachable, so Auto has the full platform and I keep the watchtower.*
- [x] Mark `super_admin_only=True`: `platform_query_loki_logs`, `platform_query_prometheus`, `platform_get_alerts`, `platform_get_logs`, `platform_list_services`, `platform_get_system_health`, and `platform_set_autonomy_level` (the kill-switch dial stays human — see Open Q2). Drop their now-redundant `admin_only` flags.
- [x] **Every other action — including formerly admin-gated and all platform-administrative capabilities (workspace deletion, billing, user/role management, system settings) — defaults to the operator tier and is Auto-reachable at `autonomy=full`.** This is the deliberate inversion of the 2026-06-07 draft: gates-and-logs (PRD-140 checks, destructive backstop, audit, rollback), not exclusion.
- [x] Produce a one-page **obs-tier manifest** listing every `super_admin_only` action and why; Gerard signs it off (Open Q1).
- [x] Tests assert manifest ⇆ registry parity (no obs action silently operator-reachable; no operator action silently su-locked). Typecheck passes.

**US-002b: Lock the observability HTTP routers (new in Rev 2).**
*As the Super admin, I want the obs/analytics REST surface readable by me only, so platform internals never leak to workspace users or Auto-driven calls.*
- [x] Add one canonical `require_super_admin` dependency to `core/auth` (checks `ctx.user.system_role == "super_admin"` exactly; 403 otherwise; fail-closed on missing context). No ad-hoc copies.
- [x] Apply it router-wide to the obs surface: `heartbeat`, `analytics`, `analytics_api`, `analytics_real`, `analytics_charts`, `llm_analytics`, `memory_stats`, `statistics`, `composio_analytics`, `database_analytics`, `execution_history`, `kpi_api`, `reports` (candidate list — Gerard prunes/extends at sign-off, Open Q1).
- [x] **Known consequence:** non-super-admin users lose the analytics/Command-Center dashboards these routers back. Acceptable in the pilot (Gerard is the only real human); revisit before GA. Frontend may hide those nav entries for non-super-admins as a courtesy, not as security.
- [x] The `automatos-monitoring` stack (Prometheus/Grafana/Loki) is network-level infra — confirm it is not exposed through any operator-reachable proxy endpoint other than the now-locked obs tools/routers.
- [x] Tests: `test_obs_router_403_for_member_admin_and_api_key`, `test_obs_router_200_for_super_admin`, one per locked router (parametrized). Typecheck passes.

### WS-B — Tool breadth mechanism + first batch

**US-003: Scaffold platform tools from existing routers/registry.**
*As a developer, I want a repeatable way to wrap a platform API as a 3-file tool, so reaching ~200 isn't 84 hand-written files.*
- [x] A scaffolding script/codegen that, given a platform API/router endpoint, emits the `actions_*`/`handlers_*` skeleton (name, description, params, workspace-scoping, default tier) for human curation — it does NOT auto-register without review.
- [x] Output obeys conventions: workspace-scoped, no `os.getenv`, canonical naming `platform_*`, default `super_admin_only=False` — but endpoints under the obs routers (US-002b list) are flagged `super_admin_only=True` by default for review.
- [x] Tests for the generator on 2-3 sample endpoints. Typecheck passes.

**US-004: First batch of operator tools (the high-value setup + admin surface).**
*As a user, I want Auto able to create agents, manage workspaces, configure channels/widgets, manage members, so it can actually run the platform.*
- [x] A prioritized batch (~15-25) of operator tools covering the **setup and administration** surface: agent CRUD, model/power config, channel connect, widget config, playbook/mission launch, knowledge upload, workspace member/role management, system settings — reusing existing platform actions where they already exist (do NOT duplicate). Administrative tools are **operator-tier** (Rev 2), with correct PRD-140 `permission_level` (`write`/`destructive`) so the hierarchy gate + audit cover them.
- [x] Each tool: 3-file pattern, workspace-scoped, correct tier + permission_level, a handler test. Typecheck passes.

### WS-C — Graph selection at scale (the enabler)

**US-005: Seed the graph from telemetry (cold-start).**
*As Auto, I want the routing graph to recommend sensible tools before it has months of usage, by learning from existing logs.*
- [x] A backfill that computes `tool_routing_edges`/`affinities` from existing `tool_execution_logs` (reuse `edge_builder`'s recompute; add a one-shot seed/backfill entry point) — workspace-scoped, idempotent.
- [x] After seeding, `graph_router` returns non-empty graph signal for common intents; semantic index remains the floor for unseeded intents.
- [x] Tests: `test_seed_backfills_edges_from_logs`, `test_seeded_graph_routes_common_intent`. Typecheck passes. (NO prod run — seed is human-applied like a migration.)

**US-006: Selection robustness + observability at ~200 tools.**
*As the Super admin, I want confidence the LLM gets the right handful out of 200, and I can see it working.*
- [x] Verify/tune top-K so adding tools doesn't starve or mis-rank (a test with a 200-tool fixture asserting the relevant tool is in the selected set for representative intents). The fixture includes `super_admin_only` actions and asserts they never appear in a non-super-admin selection.
- [x] Emit a tool-selection-health metric (selection hit-rate, fallback rate) via the existing telemetry/heartbeat mechanism — surfaced on the **super-admin-locked** dashboard (US-002b).
- [x] Tests: `test_relevant_tool_selected_at_scale`, `test_su_tools_never_selected_for_operator`, `test_selection_metric_emitted`. Typecheck passes.

### WS-D — Concierge MVP (fully autonomous)

**US-007: One fully-autonomous concierge journey.**
*As a user, I say "set up X" and Auto plans and does it end-to-end, no confirmation.*
- [x] Pick ONE journey (recommend "set up my workspace/agents"; Shopify as journey #2) and make Auto execute it fully autonomously under `autonomy=full`: plan → select tools (semantic+graph) → call the platform APIs → set up → recommend next channels/tools. With Rev 2 the journey may legitimately use administrative operator tools (e.g. member invites, workspace config).
- [x] Every step is **audited** (who/what/when via the Wave 4 audit trail); the journey respects the obs boundary (never calls a `super_admin_only` tool); HARNESS rollback + the autonomy/HARNESS flags remain the kill-switch.
- [x] A golden-journey test drives the flow end-to-end with mocked external calls. Typecheck passes.

### WS-E — Governance & guardrails

**US-008: Boundary + kill-switch tests + audit surface.**
*As the Super admin, I want proof the (inverted) boundary holds and a single switch to stop Auto.*
- [x] Cross-cutting **positive** tests: at `autonomy=full`, Auto's surface includes formerly admin-gated operator tools and can execute a representative administrative action end-to-end (audited, hierarchy-checked, rolled back in test).
- [x] Cross-cutting **negative** tests: no operator path (surface, dispatcher enum, semantic ranking, graph ranking, direct executor call, API-key call) reaches a `super_admin_only` tool or obs router; `autonomy=full` cannot cross the obs boundary; `platform_set_autonomy_level` is not Auto-invocable (the dial is human-held).
- [x] The kill-switch test: flipping autonomy/HARNESS flags halts autonomous execution mid-journey.
- [x] The audit trail records autonomous concierge actions distinctly (queryable), including every invocation of an administrative operator tool. Typecheck passes.

---

## 7. Functional requirements

- **FR-1:** Platform tools wrap **Automatos's own platform APIs** via the 3-file pattern. (NOT Composio/external — that is a separate surface.)
- **FR-2:** Tools carry a **tier**: `operator` (default — **everything**, including platform-administrative capabilities) or `super_admin_only` (**observability + the oversight control plane only**: obs query tools, system health, the autonomy dial).
- **FR-3:** **Auto's tool surface includes every operator tool at `autonomy=full` — and MUST NEVER include a `super_admin_only` tool**, at any autonomy level. The su gate checks `system_role == "super_admin"` exactly: `is_admin`, the full-autonomy bypass, the workspace-owner fallback, and API-key principals do **not** satisfy it. Fail-closed. Exception by design: when the driving principal IS the super admin, Auto may invoke obs tools on his behalf.
- **FR-4:** The **observability HTTP routers** (US-002b list) are locked behind one canonical `require_super_admin` dependency — 403 for every other principal, including admins and API keys.
- **FR-5:** Semantic + graph selection (`SEMANTIC_TOOL_ROUTING`) returns a bounded top-K (default 15) — the LLM never receives the full catalogue, and never a `super_admin_only` tool for a non-su principal. The graph is **seeded from `tool_execution_logs`**; the semantic index is the floor.
- **FR-6:** Under `autonomy=full`, the concierge journey executes end-to-end with no confirmation; every action is audited and reversible (HARNESS rollback); PRD-140 hierarchy checks + the destructive backstop still run on every call; the autonomy/HARNESS flags are the kill-switch and the dial stays human-held.
- **FR-7:** Adding tools is a **scaffold-then-curate** flow — generated tools are reviewed and tiered by a human before registration (no auto-registration).
- **FR-8:** Everything is workspace-scoped and tenant-isolated; no `os.getenv` outside `config.py`; canonical `platform_*` naming.

---

## 8. Non-goals (out of scope)

- The full ~200-tool catalogue (Phase 1 = mechanism + first batch).
- The full skills/knowledge library for every domain (Phase 1 skills only the MVP journey).
- Composio / external-app integrations.
- Rebuilding HARNESS / the autonomy dial / the tool-routing graph (PRD-142 Wave 4, PRD-138/139).
- Exposing the observability tier to Auto or non-super-admin users.
- Removing the per-call safety net (PRD-140 checks, audit, rollback) from the now-open operator surface.
- A multi-role obs story (viewer roles, per-workspace analytics opt-in) — pilot is Gerard-only; GA revisits.
- Multiple concierge journeys / a full concierge frontend (Phase 1 = one journey).
- Cross-tenant operation (every tool is workspace-scoped).

---

## 9. Technical considerations

- **The su tier** is the smallest safe extension of the existing mechanism — one new field + registry/executor gates + one FastAPI dependency + a classification pass. Do not invent a new permission system; reuse `system_role`. The subtle parts, all verified in code: (1) the `full_autonomy → is_admin=True` bypass (`platform_executor.py:510`) must sit **below** the su gate; (2) the PRD-122 workspace-owner fallback (`tool_router.py:358-372`) must not widen the su surface; (3) API keys are admin-equivalent (`hybrid.py:783`) and must still 403 on obs.
- **Risk posture (Rev 2, explicit):** opening workspace deletion, billing, and user/role management to Auto is an accepted pilot risk, mitigated by per-call hierarchy checks (PRD-140), the destructive backstop, full audit, HARNESS rollback, the human-held kill-switch, and the pilot's low blast radius. The harness rule stands: the AI cannot merge the gate-changing PR — Gerard does (memory `pilot-all-in-autonomy`).
- **Scaffolding** reads the existing routers/`ActionRegistry` to emit consistent 3-file skeletons; humans curate names/descriptions/params/tier. This is the only practical path to ~200 without drift.
- **Seeding** reuses `edge_builder`'s recompute from `tool_execution_logs` — a one-shot backfill entry point, applied like a migration (human-gated against prod).
- **Concierge execution** runs through the existing Arc (`AutoBrain`/`tool_router`) + Wave 4 autonomy/HARNESS — Phase 1 wires a journey, it does not build a new planner.
- **Selection at scale**: validate top-K and ranking with a synthetic 200-tool fixture (su actions included as negative cases) before the real catalogue grows.

---

## 10. Success metrics

| Metric | Current | Target |
|---|---|---|
| Platform tools (operator tier) | ~113 of ~120 | ~200 (mechanism + first batch in Phase 1) |
| Operator tools reachable by Auto at autonomy=full | partial (admin gate) | **100%** (the full operator tier) |
| Obs tools / routers reachable by non-super-admin | 5 tools admin-gated; **13 routers open to any workspace user** | **0** — su-locked, fail-closed, API keys excluded |
| Graph routing seeded | cold (empty edges) | non-empty for common intents (telemetry seed) |
| Tool-selection correctness at scale | unproven at 200 | relevant tool in selected set; su tools never selected for operators |
| Concierge journeys (autonomous) | 0 | 1 end-to-end, audited |
| Selection-health observability | none | a dashboard metric (hit-rate / fallback-rate) on the su-locked dashboard |

---

## 11. Open questions

1. **Obs manifest sign-off** — the exact `super_admin_only` action list (US-002) and obs-router list (US-002b) need Gerard's sign-off before the locks land. Candidate router list above errs broad; prune there.
2. **`platform_set_autonomy_level` placement** — this PRD puts the dial in the su tier (the kill-switch stays human; Auto must not raise its own autonomy). Confirm — it is the one capability deliberately withheld from "full access."
3. **Concierge journey for the MVP** — "set up my workspace/agents" (pure-platform) vs "set up my Shopify store." Recommend **workspace/agents** first, Shopify as journey #2.
4. **Dashboard breakage** — US-002b knowingly 403s the analytics/Command-Center data for non-super-admins. Fine for the pilot; decide the GA story (role-scoped analytics? workspace-local subset?) before opening to real users.
5. **Tool-breadth target precision** — is ~200 the real target, or "every platform API"? The scaffolding pass (US-003) will produce the actual number.
6. **Skills depth for the MVP** — how much curated knowledge does Auto need to drive the journey well (ties to PRD-120)?

---

**This PRD is the Phase-1 plan: the enabler for Auto-as-operator, Rev 2. One boundary instead of many: Auto gets the whole platform; Gerard keeps the watchtower. It builds on PRD-142 Wave 4, PRD-138/139, and PRD-140 — extends, never rebuilds — and enforces the observability lock as a hard, fail-closed invariant that no autonomy level, admin role, or API key can cross. The full catalogue, skills library, and additional concierge journeys follow once the enabler is proven.**
