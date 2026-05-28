# Walkthrough — Adding a Barbershop Vertical (Architecture Validation)

> Hypothetical exercise from **[PRD-141 §13](../PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md#13-appendix--hypothetical-barbershop-walkthrough-sanity-check)**. Proves the plugin protocol introduced in Phase 0 (`orchestrator/integrations/__init__.py`) is not Shopify-shaped in disguise — and flags the rough edges a real vertical #2 would hit.
>
> Nothing in this document is built. There is no `orchestrator/integrations/barbershop/`. The point is to walk the protocol end-to-end against a vertical the team hasn't shipped, find the friction, and write it down before INBUILD goes live on the Phase 1 refactor.
>
> Companion docs:
>
> - **[PRD-141 §12 — Integration Coupling Rules](../PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md#12-integration-coupling-rules-the-principle-written-down)** — the boundary.
> - **[docs/integrations/README.md](../integrations/README.md)** — the runbook for actually adding a vertical.
> - **Working reference: `orchestrator/integrations/shopify/widget_proactive.py`** — every pattern below mirrors one in there.

---

## 1. The scenario

ClipShop UK runs 47 high-street barbershops. They install the Automatos widget on `clipshop.co.uk`. The site is a hand-rolled booking app — not Shopify, not Calendly. Their pages look like this:

- `/stylists/sarah-chen` — a stylist's profile page (bio, reviews, available services)
- `/book?stylist=sarah-chen&service=fade-cut` — booking widget for a specific stylist + service
- `/locations/manchester-piccadilly` — branch landing page (which stylists work there, opening hours)

The host theme is configured to embed page metadata into the Automatos widget:

```js
window.AutomatosWidget.init({
  workspaceId: "a8f7...",
  pageContext: {
    pageType: "stylist_profile",
    stylistId: "sarah-chen",
    stylistName: "Sarah Chen",
    serviceType: "fade-cut",
    nextSlotUtc: "2026-06-02T14:00:00Z",
    currentBookings: 12,
    locationSlug: "manchester-piccadilly",
    avgReviewScore: 4.8,
    reviewCount: 137,
  },
});
```

Workspace settings in the database:

```sql
SELECT settings FROM workspaces WHERE id = 'a8f7...';
-- {
--   "vertical": "barbershop",
--   "booking_provider": "internal",
--   "locations": ["manchester-piccadilly", "leeds-vicar-lane", ...]
-- }
```

Goal: when a visitor lands on `/stylists/sarah-chen` and idles ~7 seconds, the widget fires a `proactive_opener` request. The backend rewrites the message into a directive the agent uses to produce:

> "Looking at booking with Sarah for a fade — her next opening is Tuesday at 2pm. Want me to hold it?"

Identical pattern to PRD-007 for Shopify; vertical-specific data; zero changes to the generic dispatcher.

---

## 2. What gets added

Three new pieces, mirroring the Shopify shape:

| Where | What |
|---|---|
| `orchestrator/integrations/barbershop/` | The vertical plugin (this walkthrough) |
| `automatos-skills/barbershop/booking-host/SKILL.md` | The agent skill that consumes the rewritten directive |
| `scripts/ci/check-no-barbershop-in-generic.sh` | The CI grep gate (mirrors `check-no-shopify-in-generic.sh`) |

What gets **modified** in generic surfaces: **one line**, in `orchestrator/integrations/__init__.py`:

```python
from . import generic     # noqa: E402,F401  (registers "generic")
from . import shopify     # noqa: E402,F401  (registers "shopify")
from . import barbershop  # noqa: E402,F401  (registers "barbershop")  ← added
```

That is the entire generic-surface diff. No other file outside `orchestrator/integrations/` mentions `barbershop`. If the walkthrough needed more than that line, the abstraction would be broken.

---

## 3. The plugin

### 3.1 Package layout

Mirror Shopify's:

```
orchestrator/integrations/barbershop/
├── __init__.py             # self-registers as PLUGIN_REGISTRY["barbershop"]
├── widget_proactive.py     # handle_widget_message + private helpers
├── context_fields.py       # _OPENER_CONTEXT_FIELDS + value formatter (barbershop-shaped)
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── stylist_profile_context.json
    │   ├── booking_idle_context.json
    │   ├── clipshop_graph_snapshot.json
    │   ├── expected_stylist_opener.txt
    │   └── README.md
    └── test_widget_proactive.py
```

Folder-isolated. Nothing imports anything from `orchestrator/integrations/shopify/` — the two verticals do not share code (yet — see [Gap 4](#gap-4--_format_opener_context_value-is-duplicated)).

### 3.2 Self-registration

`__init__.py` is a copy-paste of Shopify's with one slug change:

```python
"""Barbershop widget plugin package — see widget_proactive.py."""

from __future__ import annotations

from integrations import PLUGIN_REGISTRY

from . import widget_proactive

PLUGIN_REGISTRY["barbershop"] = widget_proactive

__all__ = ["widget_proactive"]
```

### 3.3 Page-context field mapping

`context_fields.py` is the barbershop equivalent of Shopify's. Same shape, different keys:

```python
_OPENER_CONTEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("pageType",        "page_type"),
    ("stylistName",     "stylist"),
    ("serviceType",     "service"),
    ("nextSlotUtc",     "next_slot"),
    ("avgReviewScore",  "avg_review"),
    ("reviewCount",     "review_count"),
    ("locationSlug",    "location"),
    ("currentBookings", "current_bookings"),
)


def _format_opener_context_value(key: str, value) -> Optional[str]:
    if value is None or value == "" or value == 0 or value is False:
        return None
    if isinstance(value, str):
        return f'{key}="{value}"' if " " in value or '"' in value else f"{key}={value}"
    return f"{key}={value}"
```

The formatter is byte-identical to Shopify's. We tolerate the duplication for now (see [Gap 4](#gap-4--_format_opener_context_value-is-duplicated)).

### 3.4 `handle_widget_message` — proactive_opener path

The plugin handles two triggers: `proactive_opener` (on stylist profile pages) and `booking_idle` (on the booking page after N seconds without selecting a slot). Pseudocode mirrors Shopify's `_resolve_graph_related_products` + `_build_proactive_opener_message`:

```python
async def handle_widget_message(
    *,
    message: str,
    page_context: Optional[dict],
    trigger_reason: Optional[str],
    workspace_id: UUID,
    db: Session,
) -> WidgetPluginResult:
    if page_context is None or trigger_reason not in ("proactive_opener", "booking_idle"):
        return WidgetPluginResult(message=message)

    workspace_str = str(workspace_id)

    if trigger_reason == "booking_idle":
        rewritten = _build_booking_idle_message(page_context)
        return WidgetPluginResult(
            message=rewritten,
            context_note="barbershop: booking_idle rewrite",
            telemetry={
                "trigger_reason": trigger_reason,
                "stylist_id": page_context.get("stylistId"),
                "service_type": page_context.get("serviceType"),
            },
        )

    stylist_facts = await _resolve_stylist_facts(workspace_str, page_context)
    rewritten = _build_stylist_opener_message(
        page_context, stylist_facts=stylist_facts,
    )
    return WidgetPluginResult(
        message=rewritten,
        context_note="barbershop: proactive_opener rewrite",
        telemetry={
            "trigger_reason": trigger_reason,
            "stylist_id": page_context.get("stylistId"),
            "fact_count": len(stylist_facts),
        },
    )


async def _resolve_stylist_facts(
    workspace_id: str,
    page_context: dict,
) -> list[dict]:
    """Walk the workspace KB for one stylist's reviews + speciality + branch.

    Looks up the seed node by ``stylistId``, then walks 1-hop edges by relation:
      - ``has_review``      — top reviews to cite
      - ``has_speciality``  — "specialises in fades", etc.
      - ``works_at``        — the branch the stylist anchors to
    """
    stylist_id = (page_context or {}).get("stylistId")
    if not stylist_id:
        return []
    try:
        from modules.knowledge.graph_service import GraphifyService

        gs = GraphifyService()
        graph = await gs.load_graph(workspace_id)
        if graph is None:
            return []

        seed_id = None
        for node_id, attrs in graph.nodes(data=True):
            node_attrs = attrs.get("attrs") or {}
            if attrs.get("file_type") == "stylist_profile" and node_attrs.get("slug") == stylist_id:
                seed_id = node_id
                break
        if seed_id is None:
            return []

        facts: list[dict] = []
        for u, v, edata in graph.edges(seed_id, data=True):
            rel = (edata.get("relation") or "").lower()
            if rel not in ("has_review", "has_speciality", "works_at"):
                continue
            other = v if u == seed_id else u
            other_attrs = graph.nodes[other]
            facts.append({
                "relation": rel,
                "label": other_attrs.get("label") or other,
                "type": other_attrs.get("file_type", ""),
                "weight": edata.get("weight", 0),
            })
        facts.sort(key=lambda f: (
            {"has_review": 0, "has_speciality": 1, "works_at": 2}.get(f["relation"], 99),
            -f["weight"],
        ))
        return facts[:6]

    except Exception as e:  # noqa: BLE001 — opener falls back gracefully
        logger.warning("_resolve_stylist_facts failed: %s", e)
        return []


def _build_stylist_opener_message(
    page_context: dict, stylist_facts: list[dict],
) -> str:
    parts: list[str] = []
    for src_key, label in _OPENER_CONTEXT_FIELDS:
        rendered = _format_opener_context_value(label, page_context.get(src_key))
        if rendered is not None:
            parts.append(rendered)
    summary = ", ".join(parts) if parts else "no context"

    facts_block = ""
    if stylist_facts:
        rendered_facts = []
        for f in stylist_facts:
            label = f.get("label", "?")
            rel = f.get("relation", "")
            if rel == "has_review":
                rendered_facts.append(f'review: "{label}"')
            elif rel == "has_speciality":
                rendered_facts.append(f"specialises in {label}")
            elif rel == "works_at":
                rendered_facts.append(f"branch: {label}")
            else:
                rendered_facts.append(label)
        facts_block = (
            " Stylist facts from workspace KB (cite naturally — prefer "
            "specialities and reviews, mention next slot only if the "
            "visitor leans booking-ward): " + "; ".join(rendered_facts)
        )

    return (
        "[PROACTIVE_OPENER] Generate a contextual one-sentence opener "
        "(≤140 chars). RETURN PLAIN TEXT ONLY — no tool calls, no JSON, "
        "no markdown, no greetings. Use the facts below as your source of "
        "truth — do NOT invent specialities, slot times, or review content "
        "the context doesn't include. If a fact you'd want isn't here, ask "
        "a question instead of fabricating. "
        f"Context: {summary}.{facts_block}"
    )
```

Notice what isn't there:

- **No external API call.** `nextSlotUtc` arrives in `page_context` — the host site embeds it at page-render time. The plugin doesn't call ClipShop's booking API inside the rewrite (see [Gap 1](#gap-1--no-convention-for-partner-api-calls-inside-the-rewrite)).
- **No imports from generic surfaces.** All the plumbing lives inside `integrations/barbershop/`.
- **No new dispatch logic in `chat.py`.** The call site is unchanged: `plugin = PLUGIN_REGISTRY[vertical]; result = await plugin.handle_widget_message(...)`.

### 3.5 `booking_idle` trigger

When a visitor opens the booking page and leaves it idle for N seconds, the SDK fires `trigger_reason="booking_idle"`. The plugin synthesizes a nudge referencing the actual stylist + slot the page is configured for:

```python
def _build_booking_idle_message(page_context: dict) -> str:
    stylist = page_context.get("stylistName") or "the stylist"
    service = page_context.get("serviceType") or "an appointment"
    slot = page_context.get("nextSlotUtc")
    slot_part = f" next_slot={slot}." if slot else ""
    return (
        "[PROACTIVE_OPENER] [BOOKING_IDLE] The visitor is on the booking "
        f"page for {stylist}, service={service}.{slot_part} Generate one "
        "sentence (≤140 chars) nudging them to hold the slot OR asking "
        "what's blocking them. RETURN PLAIN TEXT ONLY — no tool calls, no "
        "markdown. Do NOT invent alternative slot times the context "
        "doesn't include."
    )
```

Same shape as Shopify's `_build_cart_idle_opener_message` — directive boilerplate is reused; only the data shape changes.

---

## 4. The matching skill

`automatos-skills/barbershop/booking-host/SKILL.md v1.0` — outline (frontmatter shape is illustrative; the real skill repo's frontmatter rules apply):

```markdown
---
name: booking-host
description: Conversational booking assistant for barbershop, salon, and
  appointment-based service businesses using the Automatos widget.
version: 1.0
verticals: [barbershop]
tools:
  - platform_query_graph
  - platform_search_memory
  - platform_calendar_lookup     # hypothetical — see Gap 1
  - widget_open_callback_form
---

## When to use

This skill runs when ``workspace.settings.vertical == "barbershop"`` and a
visitor sends a widget message.

## Proactive openers

Mid-conversation messages from the widget may arrive with a
``[PROACTIVE_OPENER]`` prefix. Example payload the agent sees:

    [PROACTIVE_OPENER] Generate a contextual one-sentence opener (≤140 chars).
    RETURN PLAIN TEXT ONLY ... Context: page_type=stylist_profile,
    stylist="Sarah Chen", service=fade-cut, next_slot=2026-06-02T14:00:00Z,
    avg_review=4.8, review_count=137. Stylist facts from workspace KB:
    specialises in fades; review: "Sarah's fade was the cleanest I've had";
    branch: Manchester Piccadilly.

When you see ``[PROACTIVE_OPENER]``:

1. Produce one sentence ≤140 chars, no tool calls.
2. Lead with the strongest grounded fact — a specific review, a speciality,
   or the actual slot time.
3. Never invent slot times, reviews, or specialities the directive doesn't
   name.
4. End with a soft question that invites a reply ("Want me to hold it?").

## Reading mid-conversation context

On regular messages (no ``[PROACTIVE_OPENER]``), the widget may prepend a
``(Context: {...})`` block. Treat any ``stylistId``, ``serviceType``, or
``locationSlug`` in that block as a hard anchor — call
``platform_query_graph`` with that identifier before falling back to
keyword search.

## Tools

- ``platform_query_graph`` — workspace KB (stylists, services, reviews,
  branches)
- ``platform_search_memory`` — past conversations the visitor may have had
- ``platform_calendar_lookup`` — *hypothetical* — would let the agent
  confirm a slot is still free at answer-time. Does not exist today. See
  Gap 1.
- ``widget_open_callback_form`` — human handoff for booking changes the
  agent can't make
```

What this skill is **not**:

- It does not reference Shopify, products, or carts.
- It does not pretend `platform_calendar_lookup` exists today — the description marks it hypothetical, and Gap 1 catalogues the missing convention.
- It is published independently of any Phase 1 Shopify skill change. The vertical's skill is in its own folder.

---

## 5. Worked example — end-to-end trace

A visitor lands on `/stylists/sarah-chen` and idles 7 seconds.

1. **SDK fires** `POST /api/widgets/chat`:

   ```json
   {
     "message": "Hi",
     "trigger_reason": "proactive_opener",
     "page_context": {
       "pageType": "stylist_profile",
       "stylistId": "sarah-chen",
       "stylistName": "Sarah Chen",
       "serviceType": "fade-cut",
       "nextSlotUtc": "2026-06-02T14:00:00Z",
       "avgReviewScore": 4.8,
       "reviewCount": 137,
       "locationSlug": "manchester-piccadilly",
       "currentBookings": 12
     }
   }
   ```

2. **Dispatcher** (`chat.py`) reads `workspace.settings.vertical → "barbershop"`. Calls `PLUGIN_REGISTRY["barbershop"].handle_widget_message(...)`.

3. **Plugin** walks the KB graph (finds Sarah Chen's node, walks 1-hop to `has_review`/`has_speciality`/`works_at`), returns:

   ```text
   message:
     "[PROACTIVE_OPENER] Generate a contextual one-sentence opener (≤140 chars).
      RETURN PLAIN TEXT ONLY ... Context: page_type=stylist_profile,
      stylist=\"Sarah Chen\", service=fade-cut, next_slot=2026-06-02T14:00:00Z,
      avg_review=4.8, review_count=137, location=manchester-piccadilly,
      current_bookings=12. Stylist facts from workspace KB: specialises in
      fades; review: \"Sarah's fade was the cleanest I've had\"; branch:
      Manchester Piccadilly."

   telemetry:
     {"trigger_reason": "proactive_opener",
      "stylist_id": "sarah-chen",
      "fact_count": 3}
   ```

4. **Dispatcher logs**:

   ```text
   PROACTIVE_REWRITE: vertical=barbershop trigger=proactive_opener
   original_msg_len=2 new_msg_len=... telemetry={...}
   ```

5. **Streaming agent** runs the `booking-host` skill and produces:

   > "Looking at booking with Sarah for a fade — her next opening is Tuesday at 2pm. Want me to hold it?"

6. **Visitor receives** the opener.

Nothing in steps 2–6 named `barbershop` outside `integrations/barbershop/` and the skill file. The dispatcher's behaviour is identical to the Shopify path — only the registry key differs.

---

## 6. What didn't need to change

For sanity, here is everything the Phase 1 abstraction lets stay put when vertical #2 arrives:

- `orchestrator/api/widgets/chat.py` — unchanged. The dispatcher reads `workspace.settings.vertical` and looks up the registry. No new branch.
- `orchestrator/api/widgets/auth.py`, `orchestrator/api/widgets/schemas.py` — unchanged. `page_context` is already typed as `Optional[dict[str, Any]]`.
- `orchestrator/modules/context/sections/*` — unchanged.
- `orchestrator/consumers/chatbot/*` — unchanged.
- `orchestrator/modules/knowledge/graph_service.py` — unchanged.
- The SDK contract — designed in US-013/014/015 (Phase 2 SDK work, pending in the SDK repo) to widen to `Record<string, unknown>`. Once that ships, the barbershop's `page_context` shape flows through without further SDK change.
- The Alembic column — already there (`workspaces.settings JSONB`). The barbershop backfill is an analogous `UPDATE` per workspace, modelled on the Shopify migration US-009 added.

The Phase 1 abstraction holds.

---

## 7. Known abstraction gaps

The walkthrough completes — but here are the rough edges that surfaced. None of these block Phase 1 (Shopify) shipping. They shape the PRD that adds vertical #2.

### Gap 1 — no convention for partner-API calls inside the rewrite

The barbershop plugin **assumes** the host site embeds `nextSlotUtc` into `page_context`. That works when the slot is computed at page-render time. It breaks for "is this slot still free *right now*?" — the canonical anti-stale check before saying "want me to hold it?".

Today there is no pattern for `handle_widget_message` to call a partner API (ClipShop's booking endpoint) inside the rewrite:

- No latency budget — `handle_widget_message` blocks the visitor's chat response.
- No failure convention — a 502 from the partner API would silently degrade to "pass message through unchanged" (per the Shopify pattern), which loses the slot data the visitor came for.
- No cache layer.

The skill outline above lists a hypothetical `platform_calendar_lookup` tool — that's the agent-side answer (let the agent fetch fresh slot data when asked). The plugin-side answer (rewrite-time freshness) is a separate question.

**Path forward:** when vertical #2 needs real-time partner data, add a `PluginPartnerClient` mixin with a timeout-bounded caching wrapper. Defer until needed; do not pre-build it for Phase 1.

### Gap 2 — `PROACTIVE_TRIGGER_REASONS` is centralised in `chat.py`

`PROACTIVE_TRIGGER_REASONS` (the frozenset that flips `is_proactive` for the agent's LLM-call shape — text-only, no Composio) lives in `orchestrator/api/widgets/chat.py`. The barbershop plugin's `booking_idle` trigger would need adding to that frozenset — a generic-surface edit.

Today's workaround: barbershop reuses the existing `cart_idle` string. The plugin sees the trigger name; the dispatcher only checks set membership. Ugly, works.

**Path forward:** verticals declare their trigger reasons via a module-level `TRIGGER_REASONS: frozenset[str]` constant on the plugin module, and the dispatcher unions all plugin-declared trigger reasons at import time. Mild lift; do it in the same PR that adds vertical #2.

### Gap 3 — no missing-skill detection at dispatch time

`workspace.settings.vertical = "barbershop"` activates the plugin. It does **not** activate the `barbershop/booking-host` skill. If the skill is missing or stale, the plugin's rewrite is meaningless — the agent receives a `[PROACTIVE_OPENER]` directive and has no instructions for what to do with it.

The Shopify path is fine today because `shopify-support` is installed everywhere we care about. The vertical-#2 path has no such guarantee.

**Path forward:** at workspace setup time, validate that `vertical == "X"` AND `skill X-support is installed`. Two options:

- a `bootstrap_check()` on the plugin protocol returning the skill name(s) it requires, or
- a static lookup table in `orchestrator/integrations/__init__.py` mapping vertical → required skill slugs.

The latter is simpler; the former couples better to plugin evolution.

### Gap 4 — `_format_opener_context_value` is duplicated

The barbershop `context_fields.py` formatter is byte-identical to Shopify's:

```python
f'{key}="{value}"' if " " in value or '"' in value else f"{key}={value}"
```

Two verticals, same code. That suggests a shared utility — e.g. `integrations/_common/context_fields.py` exporting the formatter.

**Path forward:** harvest only when vertical #3 lands (rule of three). Two copies is fine; three copies is a smell.

### Gap 5 — no `page_context` schema versioning

A host-site rename (`stylistId → professionalId` in v2 of their theme) silently breaks the rewrite. The plugin tolerates missing keys, so the fallback is "drop to a less-grounded opener" — the visitor sees a worse reply without any error logged or alert raised.

Shopify has the same hazard today (a theme dev renaming `productHandle` to `product_handle_v2` would silently degrade INBUILD).

**Path forward:** require a `page_context["_v"]: int` per vertical. The plugin asserts the major version it understands and logs a `KEY_DRIFT` warning when it sees a newer version. Telemetry only; not a blocker.

### Gap 6 — multi-location / per-row overrides

A 47-branch chain runs as one workspace with `vertical: "barbershop"` and a `locations: [...]` list. The plugin currently has no place to read per-location overrides (each branch has its own opening hours, its own stylist roster, its own pricing).

The protocol's `db: Session` argument means the plugin *can* read arbitrary state — but there is no shape convention for "per-location settings". A naive plugin would JSON-pack everything into `workspace.settings`; an opinionated one would want a `workspace_locations` table.

**Path forward:** out of scope for the plugin protocol. This is a product question — solve at the workspace-template level when the partner brief arrives.

### Gap 7 — `db: Session` is sync; the protocol is async

The protocol declares `db: Session` (SQLAlchemy sync session) but `handle_widget_message` is `async`. Today both Shopify functions use the session indirectly via `GraphifyService` (which loads its own state). If a vertical needed to run a `SELECT` inside the rewrite, it would be doing sync DB work inside an async function — fine in practice (FastAPI's sync session is threadlocal-backed) but a foot-gun.

**Path forward:** when a vertical first needs in-rewrite DB reads, either pass `AsyncSession` (PRD-XXX migration) or have the plugin offload via `asyncio.to_thread`. Document the pattern; do not change the protocol until forced.

---

## 8. Verdict

The Phase 1 plugin protocol is **vertical-neutral enough** to ship production-grade vertical #2 without revising the protocol. Every gap above is incremental — additive constants, additional helpers, additional contracts that bolt on without rewriting `WidgetPlugin`.

The walkthrough produces the target opener:

> "Looking at booking with Sarah for a fade — her next opening is Tuesday at 2pm. Want me to hold it?"

— and routes through the dispatcher the same way Shopify does. The abstraction is not Shopify-shaped in disguise.

**Recommendation:**

1. Ship Phase 1 as designed. The abstraction holds.
2. When vertical #2 lands, address Gaps 1, 2, and 3 in the same PR — they have direct partner-brief implications.
3. Reassess Gaps 4–7 only if and when a third vertical materialises.

---

**Last updated:** 2026-05-28
**Status:** Architecture validation complete. No blocking gaps identified.
**Reviewer:** Auto (peer review pending).
