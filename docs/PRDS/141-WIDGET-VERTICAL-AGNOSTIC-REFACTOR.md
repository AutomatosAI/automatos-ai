# PRD-141 — Widget Vertical-Agnostic Refactor (Shopify PoC Hardening)

**Status:** Draft v1 — pending approval
**Type:** Architecture / Refactor (with PoC delivery)
**Priority:** P1 — blocks deeper Shopify PoC features and unlocks vertical #2
**Owner:** Platform
**Author:** Gerard Kavanagh + Claude
**Reviewer:** Auto (CTO)
**Date:** 2026-05-28
**Related PRDs:**
- PRD-007 (Proactive opener mode)
- PRD-008-B (Cart-idle graph-grounded popup)
- PRD-009 (Product Knowledge Graph — Shopify catalog sync, FBT edges)
- PRD-071 (Unified skills / tools registry)
- PRD-120 (Skills marketplace)
- PRD-138 (Tool-routing graphs)

**Related repos:**
- `automatos-ai` (Python orchestrator + Next.js admin)
- `automatos-widget-sdk` (TypeScript widget SDK)
- `automatos-shopify` (Remix Shopify storefront app — install/auth/embed)
- `automatos-skills` (skill definitions)

---

## Operating Principle

> **The generic widget chatbot, the graph subsystem, and the RAG/memory subsystem must remain vertical-agnostic. Vertical-specific code (Shopify product handles, cart logic, FBT graph walks, etc.) lives in clearly-marked integration folders. Partnership coupling is permitted but folder-isolated; it may never be referenced from generic code by name.**

This is the rule the rest of the PRD enforces. It allows pragmatic partnership coupling (Shopify gets deep integration code because it's our first deep vertical) while keeping the generic platform reusable for any business — barbershop, SaaS dashboard, content publisher, etc.

---

## 1. Purpose

Automatos' widget chat endpoint and SDK currently have Shopify-shaped code inlined in their generic surfaces:

- `orchestrator/api/widgets/chat.py` reads `productHandle`, `productTitle`, `cartItems`, and other Shopify keys directly. It also embeds Shopify-shaped resolvers (`_resolve_graph_related_products`, `_resolve_cart_recommendations`) and message builders (`_build_proactive_opener_message`, `_build_cart_idle_opener_message`).
- `automatos-widget-sdk/packages/loader/src/{proactive,cart-idle}/` scrapes Shopify DOM directly.

This was acceptable while Shopify was the only vertical. With INBUILD UK as the first PoC partner and a second vertical inevitable, the inlined Shopify keys block the platform from working on any non-Shopify embed without code edits to generic surfaces.

This PRD:

1. Moves all Shopify-specific dispatch logic out of generic `chat.py` into a folder-isolated integration (`orchestrator/integrations/shopify/`).
2. Generalises the widget chat endpoint to accept an opaque `page_context` dict and route to a per-workspace vertical plugin (default: generic pass-through).
3. Updates the widget SDK so regular user messages carry `page_context` (today only proactive triggers do).
4. Preserves PRD-007 and PRD-008-B proactive popups byte-for-byte at every PR boundary using snapshot tests.
5. Unblocks PR #383 (image/URL surfacing for product cards) on a clean foundation.

It is **scoped for the Shopify PoC** — the abstraction is designed for the next vertical but only one vertical is implemented. The Out-of-Scope section names the deferred problems.

---

## 2. Background

### 2.1 What's working today (must not break)

- **PRD-007 product-page proactive opener.** Widget detects `time_on_page` on a product page → POSTs `{message: "", trigger_reason: "proactive_opener", page_context: {productHandle, productTitle, ...}}` → backend rewrites the message to a `[PROACTIVE_OPENER]`-prefixed directive that includes FBT / collection / vendor siblings from the workspace knowledge graph → agent generates a short opener line. ~7% widget engagement uplift in pilot.
- **PRD-008-B cart-idle proactive popup.** Widget detects cart-idle trigger → backend walks FBT edges across every cart line item → builds a graph-grounded cross-sell directive → agent generates a cross-sell line. Live on INBUILD.
- **Shopify catalog sync.** `orchestrator/api/shopify.py` runs SHOPIFY_BULK_QUERY_OPERATION via Composio, maps to graph nodes via `map_shopify_catalog`, persists via GraphifyService. Already isolated; not touched by this PRD.
- **Workspace settings UI.** Admin has a Shopify sync card + Widget SDK tab (per current production screenshot). Already vertical-flagged; not touched by this PRD.

### 2.2 What's broken / blocked

- **Mid-conversation messages drop page context.** Widget SDK sends `page_context` only on proactive triggers (see `automatos-widget-sdk/packages/core/src/client.ts:69-74` — `ChatRequest` body omits `page_context` on regular `sendMessage`). When a shopper types "Tell me more about this product" without a prior proactive opener in history, the agent has no idea which product they're on and falls back to a generic catalog list. Observed today on INBUILD.
- **Generic chat endpoint has Shopify keys hardcoded.** Adding vertical #2 requires editing `chat.py`. Future verticals will pile up `elif productHandle ... elif stylistId ... elif accountTier ...` style branches.

### 2.3 Why now

INBUILD UK is the first paying PoC partner. Their feedback is shaping the product. The next vertical conversation has already started (service bookings, B2B SaaS). The cost of generic-ifying scales with the number of inlined Shopify keys; cheaper to do now (one vertical, one workspace) than later (many).

---

## 3. Goals

- **G1.** `orchestrator/api/widgets/chat.py` contains zero vertical-specific key reads. All Shopify-shaped logic moves to `orchestrator/integrations/shopify/widget_proactive.py`.
- **G2.** Widget chat endpoint dispatches to a per-workspace vertical plugin based on `workspace.settings["vertical"]` (default: `"generic"` pass-through).
- **G3.** `automatos-widget-sdk` core sends `page_context` (opaque dict, no key knowledge) on every user message — proactive AND regular.
- **G4.** PRD-007 product-page opener and PRD-008-B cart-idle opener produce **byte-identical messages** before and after the refactor (snapshot test enforced).
- **G5.** A new workspace with no vertical setting falls back to a documented generic pass-through behaviour — the agent receives the context dict as a JSON block prepended to the message and can decide what to do.
- **G6.** INBUILD UK proactive popups continue working through every PR boundary; no "broken until refactor done" window.
- **G7.** PR #383 (image/URL + node_attrs surfacing) is unblocked and lands on the clean foundation.

---

## 4. Non-Goals (explicitly deferred)

- **Multi-vertical workspaces.** A single workspace running Shopify + bookings + email is real and will arrive — but this PRD assumes one vertical per workspace. Future PRD: `workspace.integrations = ["shopify", "calendly", "stripe"]` with intent-based routing.
- **Generic trigger taxonomy.** `time_on_page` and `cart_idle` stay as the trigger names. A future PRD generalises the trigger taxonomy when vertical #2 has its own triggers (e.g., `billing_page_opened`, `appointment_slot_view`).
- **Moving the Shopify-aware widget SDK loader** (`packages/loader/src/{proactive,cart-idle}/`) into a separate `@automatos/widget-sdk-shopify-loader` package. The loader stays where it is for this PRD; only the SDK core gets the `page_context` field added to regular messages. Loader split tracked as Out of Scope item OS-2.
- **Moving the Shopify catalog importer** (`map_shopify_catalog`, `api/shopify.py`) into the integrations folder. It's already isolated to one file and one router; cosmetic move can wait.
- **Vertical-plugin admin UI**. The `workspace.settings["vertical"]` field is set via API/migration, not a new UI control. Admin UI work tracked as Out of Scope item OS-3.
- **Non-Shopify vertical implementation.** Only Shopify gets a real plugin in this PRD. Generic pass-through is the only other dispatch target. Vertical #2's plugin is a separate PRD.

---

## 5. Concepts

### 5.1 Vertical
A label on a workspace indicating which integration plugin handles its widget chat dispatch. Stored at `workspace.settings["vertical"]`. Initial values: `"shopify"`, `"generic"` (default).

### 5.2 Vertical plugin
A Python module under `orchestrator/integrations/<vertical>/widget_proactive.py` implementing a single function:

```python
def handle_widget_message(
    *,
    message: str,
    page_context: dict | None,
    trigger_reason: str | None,
    workspace_id: UUID,
    db: Session,
) -> WidgetPluginResult:
    """Return the (possibly rewritten) message to feed the agent.

    Vertical-specific logic — graph walks, context-aware message
    building — lives here. The generic chat endpoint never reads
    vertical-specific keys from page_context.
    """
```

`WidgetPluginResult` is a small dataclass: `{message: str, context_note: str | None, telemetry: dict}`.

### 5.3 Generic pass-through plugin
The default plugin for any workspace without an explicit vertical. It JSON-formats `page_context` into a `Context: {...}` block and prepends it to the message. No vertical key reads. Agent skill is responsible for interpreting whatever shape the context has.

### 5.4 Plugin registry
A small dictionary mapping `vertical → plugin module`. Registered at startup. Generic dispatcher in `chat.py` does `plugin = PLUGIN_REGISTRY[workspace.vertical or "generic"]; plugin.handle_widget_message(...)`.

---

## 6. Architecture

### 6.1 Target call graph

```
Host site (Shopify theme via automatos-shopify, or any embed)
  └─ widget-sdk core: POST /api/widgets/chat
       {message, conversation_id, page_context, trigger_reason?, agent_id, model_id}
           │
           ▼
  orchestrator/api/widgets/chat.py     ← GENERIC. no vertical key reads.
    ├─ resolve workspace from auth
    ├─ vertical = workspace.settings.get("vertical", "generic")
    ├─ plugin = PLUGIN_REGISTRY[vertical]
    ├─ result = plugin.handle_widget_message(
    │              message=body.message,
    │              page_context=body.page_context,
    │              trigger_reason=body.trigger_reason,
    │              workspace_id=workspace_id,
    │              db=db,
    │           )
    ├─ body.message = result.message
    └─ stream to agent via StreamingChatService (unchanged)
           │
           ▼
  Agent (skill from automatos-skills, e.g. shopify-support v1.3.1)
           │
           ▼
  Tools (platform_query_graph, platform_graph_neighbors, etc.) — all generic
```

### 6.2 File layout after refactor

**automatos-ai (orchestrator):**

```
orchestrator/
├── api/
│   ├── widgets/
│   │   └── chat.py             ← THIN. dispatcher only. no productHandle.
│   └── shopify.py              ← unchanged (catalog sync — already isolated)
├── integrations/
│   ├── __init__.py             ← PLUGIN_REGISTRY + base protocol
│   ├── generic/
│   │   └── widget_proactive.py ← pass-through, JSON-format context block
│   └── shopify/
│       ├── __init__.py
│       ├── widget_proactive.py ← all PRD-007/008-B logic moved here
│       ├── context_fields.py   ← PROACTIVE_OPENER_FIELDS constant moved here
│       └── tests/
│           ├── fixtures/
│           │   ├── product_page_context.json
│           │   ├── cart_idle_context.json
│           │   └── inbuild_graph_snapshot.pkl
│           └── test_widget_proactive.py  ← snapshot equivalence tests
└── modules/
    ├── knowledge/
    │   └── graph_extraction.py ← unchanged (map_shopify_catalog stays put)
    └── ... (all generic)
```

**automatos-widget-sdk (TypeScript, minor change only):**

```
packages/
├── core/
│   └── src/client.ts           ← sendMessage now sends page_context too
└── loader/
    └── src/{proactive,cart-idle}/  ← unchanged this PRD (see OS-2)
```

**automatos-shopify:** unchanged.

**automatos-skills:** Phase 4 updates `shopify-support/SKILL.md` (v1.3.2) to document the generic-context contract for the agent. New `default-widget-support/SKILL.md` added for generic-vertical workspaces.

### 6.3 Database & migration

One-line workspace settings migration:

```sql
UPDATE workspaces
SET settings = jsonb_set(settings, '{vertical}', '"shopify"')
WHERE settings ? 'shopify_domain'
  AND NOT settings ? 'vertical';
```

Idempotent. Runs once. Existing Shopify workspaces (including INBUILD) get `vertical = "shopify"`. Migration committed alongside Phase 1.

---

## 7. Phased Plan

Each phase = one PR. Proactive popups remain functional at every PR boundary. Snapshot tests pin equivalence.

### Phase 0 — Plugin scaffolding (foundation, no behaviour change)
**Files (automatos-ai):**
- New `orchestrator/integrations/__init__.py` — `PLUGIN_REGISTRY`, `WidgetPluginResult` dataclass, `WidgetPlugin` protocol
- New `orchestrator/integrations/generic/widget_proactive.py` — generic pass-through (JSON-format context block)
- New `orchestrator/integrations/shopify/widget_proactive.py` — empty shim that delegates back to `chat.py`'s existing inline functions (no behaviour change)
- New tests covering the registry contract

**Risk:** zero. Scaffolding only. Existing code paths unchanged.

**Acceptance:** `pytest orchestrator/integrations/` green; production deploy is a no-op.

---

### Phase 1 — Lift Shopify logic into the plugin (the risky one)
**Files (automatos-ai):**
- `orchestrator/integrations/shopify/widget_proactive.py` — receives the moved code (currently in `chat.py`): `_resolve_graph_related_products`, `_resolve_cart_recommendations`, `_build_proactive_opener_message`, `_build_cart_idle_opener_message`, `PROACTIVE_OPENER_FIELDS` (latter moves to `integrations/shopify/context_fields.py`)
- `orchestrator/api/widgets/chat.py` — inline functions DELETED. Endpoint now calls `plugin.handle_widget_message(...)`. Reads `workspace.settings["vertical"]` to pick plugin.
- New migration `migrations/<timestamp>_backfill_workspace_vertical.sql` — sets `vertical = "shopify"` on all existing Shopify workspaces
- New `orchestrator/integrations/shopify/tests/test_widget_proactive.py` — snapshot tests asserting byte-equality of opener + cart-idle output against captured fixtures

**Risk:** medium. Same code, new home, same behaviour. Snapshot tests are the safety net. INBUILD is the canary — first deploy validated against live traffic.

**Acceptance:**
- All snapshot tests pass
- Manual smoke test on INBUILD: proactive opener fires on product page, cart-idle fires on cart-with-items, both produce expected text
- No `productHandle`, `productTitle`, `cartItems`, `cartItemCount`, `shopify_*` strings anywhere in `orchestrator/api/widgets/chat.py` (grep gate)

---

### Phase 2 — SDK sends page_context on regular messages
**Files (automatos-widget-sdk):**
- `packages/core/src/types.ts` — `ChatRequest` already has `page_context?: PageContext`; widen to `Record<string, unknown>` (generic dict)
- `packages/core/src/client.ts` — `sendMessage(content, pageContext?)` accepts optional pageContext, includes in request body. Caller (host integration) passes it.
- `packages/loader/src/proactive/` — already passes pageContext on proactive path; now also pipes it through to regular sendMessage via the chat UI integration
- New unit test asserting the body shape on both regular and proactive sends

**Risk:** low. Additive field. Backend already accepts `page_context` on regular messages (just ignores it pre-Phase-1; uses it post-Phase-1 via generic plugin pass-through for non-Shopify workspaces, or via Shopify plugin for Shopify workspaces).

**Acceptance:**
- Unit tests assert `page_context` present in body for regular sendMessage when host provides it
- SDK semver bump (minor, additive)
- Shopify theme picks up new SDK version → existing test on INBUILD confirms behaviour unchanged

---

### Phase 3 — Skill update + generic context contract
**Files (automatos-skills):**
- `shopify/shopify-support/SKILL.md` → v1.3.2: add a "Page context" section documenting that the agent may receive a `Context:` block prefixed to a user message (when not on the proactive path), and how to use it (call `platform_query_graph` for the named entities)
- New `generic/default-widget-support/SKILL.md` v1.0 — minimal generic widget skill for the `generic` vertical: instructs the agent to read the prepended context dict, find what looks like an identifier or entity, query the workspace KB/graph generically

**Risk:** low. Skill changes are versioned and non-breaking.

**Acceptance:**
- Skill version bumped, PR opened and reviewed
- Smoke test on INBUILD: mid-conversation "Tell me more about this product" message now anchors correctly because Phase 2 SDK sends context AND Phase 1 plugin formats it AND Phase 3 skill knows what to do with it

---

### Phase 4 — Cleanup, docs, and the second-vertical exercise
**Files (automatos-ai):**
- `docs/integrations/README.md` — how to add a new vertical: copy `integrations/generic/`, implement `handle_widget_message`, register in `PLUGIN_REGISTRY`
- "Partnership coupling rules" section added to this PRD (or split into a separate `docs/architecture/integration-coupling-rules.md`)
- "Hypothetical barbershop vertical" walkthrough as a markdown doc — proves the abstraction works for vertical #2 without writing the code

**Risk:** zero. Documentation only.

**Acceptance:**
- Docs land
- A second engineer (or Auto) can describe how to add a non-Shopify vertical from the docs alone

---

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Phase 1 snapshot tests miss a subtle format difference, proactive opener regresses on INBUILD | Capture fixtures from PRODUCTION traffic (one product page, one cart-idle event) before Phase 1 starts. Tests assert exact string match. Canary deploy on INBUILD's workspace only for 24h before broader release. |
| R2 | Migration backfills wrong workspaces | Migration uses `settings ? 'shopify_domain'` filter — only workspaces with a Shopify domain get `vertical = "shopify"`. Dry-run query first, log affected workspace IDs, sanity-check count vs known Shopify workspaces. |
| R3 | A second engineer adds a new "elif productHandle" to `chat.py` later, drift sets in | Phase 1 PR includes a CI grep check: `grep -rE "productHandle|productTitle|cartItems|shopify_" orchestrator/api/widgets/ orchestrator/modules/context/` MUST return zero. Fails CI if violated. |
| R4 | Generic plugin's JSON context block confuses the agent (vs the structured PROACTIVE_OPENER_FIELDS approach) | Phase 3 skill update gives explicit examples. Start with `generic-default-widget-support` skill, iterate with real usage on a non-Shopify pilot. |
| R5 | SDK `page_context` on regular messages bloats request payloads | `page_context` already exists in the request type, just not currently sent on regular messages. Cap document size in client-side validation (e.g. 4KB). Backend rejects > 64KB. |
| R6 | INBUILD's proactive popups stop working between PR merges | Deploys are atomic — Phase 1 lands with the snapshot tests AND the migration in the same PR. No partial-state window. Canary on INBUILD for 24h before announcing to other Shopify partners (currently none). |
| R7 | Future verticals demand changes to the plugin protocol, breaking Shopify plugin | Protocol is small (one function, one return type). Versioned via Python type checker. Breaking change = explicit PR, not silent drift. |

---

## 9. Success Criteria

The PRD ships successfully when ALL of the following are true:

1. ✅ `grep -rE "productHandle|productTitle|cartItems|shopify_" orchestrator/api/widgets/chat.py orchestrator/modules/context/sections/` returns ZERO results
2. ✅ INBUILD UK proactive opener fires on a product page and produces the same opener text as before the refactor (manual + snapshot test verified)
3. ✅ INBUILD UK cart-idle popup fires on a cart with items and produces the same cross-sell text as before
4. ✅ Mid-conversation message "Tell me more about this product" on an INBUILD product page now correctly identifies the product (was broken pre-refactor; Phase 2+3 fix it)
5. ✅ A test workspace with `vertical = "generic"` and a non-Shopify page_context receives a `Context: {...}` prefixed message and the agent responds sensibly
6. ✅ All existing widget tests pass; new snapshot tests pass; CI grep gate enforces no Shopify keys in generic surfaces
7. ✅ `automatos-shopify` repo is unchanged
8. ✅ Migration applied; all existing Shopify workspaces have `settings.vertical = "shopify"`
9. ✅ Documentation in `docs/integrations/README.md` describes how to add a new vertical

---

## 10. Build Order (Approved Sequence)

| Order | PR | Branch | Repo | Reviewer | Gate |
|---|---|---|---|---|---|
| 1 | Capture fixtures from prod INBUILD traffic | `chore/capture-proactive-fixtures` | automatos-ai | — | Manual, one-off |
| 2 | Phase 0 — plugin scaffolding | `feat/widget-plugin-scaffolding` | automatos-ai | code-reviewer + architect | Tests + grep gate |
| 3 | Phase 1 — lift Shopify into plugin + migration + snapshot tests | `feat/widget-shopify-plugin-lift` | automatos-ai | code-reviewer + migration-reviewer + architect | Snapshot tests byte-equal + canary on INBUILD 24h |
| 4 | PR #383 — image/URL + node_attrs surfacing (parallel, independent) | `feat/shopify-graph-reland` (already open) | automatos-ai | code-reviewer | Existing PR; merge whenever ready |
| 5 | Phase 2 — SDK sends page_context on regular messages | `feat/sdk-pagecontext-on-regular` | automatos-widget-sdk | code-reviewer | Unit tests; SDK semver minor bump |
| 6 | Phase 3 — Skill v1.3.2 + new generic skill | `feat/skill-pagecontext-and-generic` | automatos-skills | code-reviewer | Manual smoke test on INBUILD |
| 7 | Phase 4 — docs + barbershop walkthrough | `docs/integration-vertical-howto` | automatos-ai | architect | Reviewed by a second engineer |

PR #383 (the image/URL feature for product cards) is independent and lands in parallel — it doesn't depend on the refactor and the refactor doesn't depend on it.

---

## 11. Decision Required Before Build

All decisions captured. The following are RESOLVED in this draft:

- **Plugin location:** `orchestrator/integrations/<vertical>/` inside `automatos-ai` (folder isolation, partnership coupling honoured). Not a separate Python package.
- **Backward-compat window:** Hard-cut at Phase 1 — old inline Shopify functions deleted in the same PR that introduces the plugin. Snapshot tests enforce equivalence. Matches the no-backward-compat-shim project rule.
- **Test harness:** pytest with frozen JSON fixtures + golden text files for byte-equality assertions. No new infra.
- **Scope of PR #383:** Independent track. Lands on its own. The refactor doesn't depend on it; it doesn't depend on the refactor.
- **First Shopify PoC partner:** INBUILD UK is the canary. 24h soak before declaring done.

**Open for reviewer (Auto) to confirm:**
- Are we comfortable hard-cutting at Phase 1 with snapshot tests as the only safety net, or do we want a feature flag fallback for 1 release cycle?
- Is the `vertical` field name correct, or do we want something more durable (`integration_profile`, `template_key`)?
- Does the "second engineer can add a vertical from docs alone" acceptance criterion need a concrete artifact (a written runbook + dry-run PR) or is it OK as a qualitative bar?

---

## 12. Integration Coupling Rules (the principle, written down)

A partnership integration (Shopify, future Stripe, future HubSpot, etc.) MAY:

- Add files under `orchestrator/integrations/<partner>/`
- Add admin UI surfaces (gated by workspace setting visibility)
- Reuse generic platform tools (`platform_query_graph`, `platform_search_memory`, etc.)
- Add new platform tools that are generic-but-tagged-with-the-partner

A partnership integration MAY NOT:

- Add fields with partner-specific names to generic models (no `Workspace.shopify_*` columns; use `settings` JSON)
- Be referenced by name (string match, import) from any file outside `orchestrator/integrations/<partner>/`
- Modify generic dispatch logic in `chat.py`, `streaming.py`, `intent_classifier.py`, etc.
- Add partner-specific keys to generic skill prompts (skills are per-vertical files; generic skill stays generic)

CI gate: `grep -rE "<partner-key-patterns>" --exclude-dir=integrations` MUST return zero for each partner.

These rules are enforced from PR #2 onward. Drift becomes visible at code-review time, not in production.

---

## 13. Appendix — Hypothetical Barbershop Walkthrough (sanity check)

To prove the abstraction works for vertical #2 before we have a vertical #2:

A barbershop chain installs the Automatos widget on their booking site. Their workspace `settings = {vertical: "barbershop", booking_provider: "calendly", ...}`. A new `orchestrator/integrations/barbershop/widget_proactive.py` is added with a `handle_widget_message` that:

- Reads `page_context.stylistId`, `page_context.serviceType`, `page_context.nextSlotUtc`
- Walks the workspace KB for that stylist's reviews + the next available slot
- Builds a proactive opener: `"Looking at booking with Sarah for a fade — her next opening is Tuesday at 2pm. Want me to hold it?"`

`PLUGIN_REGISTRY["barbershop"] = barbershop.widget_proactive`. Zero changes to `chat.py`. The barbershop's skill (`automatos-skills/barbershop/booking-host/SKILL.md`) knows what to do with the context. The barbershop's host site sends its own `page_context` shape. Nothing in the generic surfaces is touched.

If this walkthrough doesn't survive review (e.g., we discover the plugin protocol can't express barbershop's needs), the abstraction is wrong and we revise before Phase 1 ships.

---

## 14. References

- `orchestrator/api/widgets/chat.py` — current generic endpoint with inline Shopify logic
- `orchestrator/api/shopify.py` — catalog sync (out of scope for this PRD)
- `orchestrator/modules/knowledge/graph_extraction.py` — `map_shopify_catalog` (out of scope for this PRD)
- `automatos-widget-sdk/packages/core/src/client.ts` — `sendMessage` to extend with `page_context`
- `automatos-skills/shopify/shopify-support/SKILL.md` — v1.3.1 currently; v1.3.2 in Phase 3
- Memory: `feedback-no-backward-compat.md` — informs Phase 1 hard-cut decision
- Memory: `feedback-no-hardcoded-values.md` — informs the entire PRD

---

**End of PRD-141.**
