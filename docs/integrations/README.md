# Vertical Integrations — How to Add a New One

> Authoritative reference for the coupling rules: **[PRD-141 §12 — Integration Coupling Rules](../PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md#12-integration-coupling-rules-the-principle-written-down)**. This README is the runbook; §12 is the law.

This guide is for the engineer who needs to ship vertical #2 (and #3, and #N) without touching a single line of generic widget code. If you find yourself editing `orchestrator/api/widgets/chat.py` to make your vertical work, you have drifted off the path — stop and re-read §12.

---

## (a) What is a vertical integration?

A **vertical** is a label on a workspace that tells the widget chat dispatcher which Python plugin should rewrite the user's message before it reaches the streaming agent. It is stored at `workspace.settings["vertical"]` and read by `orchestrator/api/widgets/chat.py` on every widget message.

- `"shopify"` — the working reference. See `orchestrator/integrations/shopify/widget_proactive.py`.
- `"generic"` — the default. No vertical-specific key reads; JSON-formats whatever `page_context` arrives and prepends it to the message. See `orchestrator/integrations/generic/widget_proactive.py`.
- `"<your-new-vertical>"` — what this doc helps you build.

A vertical plugin is a Python module implementing the `WidgetPlugin` protocol from `orchestrator/integrations/__init__.py`:

```python
async def handle_widget_message(
    *,
    message: str,
    page_context: Optional[dict],
    trigger_reason: Optional[str],
    workspace_id: UUID,
    db: Session,
) -> WidgetPluginResult: ...
```

`WidgetPluginResult` is a small dataclass — `message` (the possibly-rewritten user message), `context_note` (optional short string for logs), `telemetry` (dict the dispatcher attaches to its structured log line).

The dispatcher in `chat.py` looks up your plugin by the workspace's vertical string and calls `handle_widget_message`. Whatever you return as `result.message` is what the agent sees. Everything else about the request stays generic.

---

## (b) When to add a new vertical

Add a new vertical when **all** of the following are true:

1. **There is a real partner integration behind it.** Shopify, Stripe, HubSpot, Calendly — something with its own catalog, page shapes, or proactive triggers. Do not add a vertical to express a tone-of-voice difference or a single skill swap; those go in the skill catalog.
2. **The host site sends vertical-specific `page_context` shapes** that the generic JSON pass-through can't usefully exploit. If the agent skill can read whatever generic dict arrives and do the right thing, you don't need a plugin — you need a better skill.
3. **You need to rewrite the message before the agent sees it** — typically because a proactive trigger fires (`proactive_opener`, `cart_idle`, or your vertical's new trigger) and the directive needs to embed graph-resolved entities the agent can't fetch on its own.
4. **The behaviour is one-vertical-per-workspace.** PRD-141 explicitly defers multi-vertical workspaces. If your use case is "the same workspace runs Shopify AND bookings", stop and design the multi-vertical PRD — don't smuggle two verticals into one plugin.

If only (1) and (2) hold, write a generic-but-tagged skill and let the generic pass-through plugin do its job. If (3) holds, you need a plugin.

---

## (c) Step-by-step: add a new vertical

The mechanical recipe. Replace `<new>` with your vertical's slug (lower-case, no hyphens — e.g. `barbershop`, `stripe`, `calendly`).

### Step 1 — Copy the generic plugin as your starting point

```bash
cp -r orchestrator/integrations/generic orchestrator/integrations/<new>
```

You now have:

```
orchestrator/integrations/<new>/
├── __init__.py          # registers PLUGIN_REGISTRY["<new>"]
├── widget_proactive.py  # your handle_widget_message lives here
└── tests/
    └── test_widget_proactive.py
```

### Step 2 — Update `__init__.py` to self-register your slug

```python
"""<New> widget plugin package — see widget_proactive.py."""

from __future__ import annotations

from integrations import PLUGIN_REGISTRY

from . import widget_proactive

PLUGIN_REGISTRY["<new>"] = widget_proactive

__all__ = ["widget_proactive"]
```

### Step 3 — Wire your package into the top-level registry

Edit `orchestrator/integrations/__init__.py` and add one line at the bottom alongside the existing `generic` and `shopify` imports:

```python
from . import generic    # noqa: E402,F401  (registers "generic")
from . import shopify    # noqa: E402,F401  (registers "shopify")
from . import <new>      # noqa: E402,F401  (registers "<new>")
```

That's the only edit to a shared file in the whole flow.

### Step 4 — Implement `handle_widget_message` in `widget_proactive.py`

Strip the generic JSON-prefix logic and write what your vertical actually needs. The minimal plugin below is ~30 lines and is complete enough to copy-paste-modify:

```python
"""<New> vertical plugin — proactive opener for <new> page shapes.

Registered as PLUGIN_REGISTRY["<new>"]. See docs/integrations/README.md
for the recipe; see integrations/shopify/widget_proactive.py for a
working multi-trigger reference.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from integrations import WidgetPluginResult

logger = logging.getLogger(__name__)


async def handle_widget_message(
    *,
    message: str,
    page_context: Optional[dict],
    trigger_reason: Optional[str],
    workspace_id: UUID,
    db: Session,
) -> WidgetPluginResult:
    if page_context is None or trigger_reason != "proactive_opener":
        return WidgetPluginResult(message=message)

    entity = page_context.get("yourPageKey")
    if not entity:
        return WidgetPluginResult(message=message)

    rewritten = (
        "[PROACTIVE_OPENER] The visitor is looking at "
        f"{entity}. Generate one helpful sentence (≤140 chars). "
        "RETURN PLAIN TEXT ONLY — no tool calls, no markdown."
    )
    return WidgetPluginResult(
        message=rewritten,
        context_note="<new>: proactive_opener rewrite",
        telemetry={"trigger_reason": trigger_reason},
    )
```

Key shape rules — copy them, don't bend them:

- **Return the input message unchanged** when `trigger_reason` is `None`, when `page_context` is `None`, or when your vertical can't produce a useful rewrite for this request. Mid-conversation messages must pass through verbatim.
- **Never `raise`** out of `handle_widget_message`. Catch your own errors, log them, and return the unchanged message. The dispatcher logs your `telemetry`; it does not catch your exceptions.
- **Keep `telemetry` small and JSON-serialisable** — short string keys, primitive values, no full graph nodes.

### Step 5 — Turn on the vertical for a workspace

The vertical label is set in the database, not in a UI control (PRD-141 OS-3). Either backfill via an Alembic migration (see `orchestrator/alembic/versions/` for the Shopify backfill US-009 added) or set the row by hand for the first canary workspace:

```sql
UPDATE workspaces
SET settings = jsonb_set(coalesce(settings, '{}'::jsonb), '{vertical}', '"<new>"')
WHERE id = '<workspace-uuid>';
```

The next widget message that workspace sends will be dispatched through your plugin.

---

## (d) What you MAY NOT do

The boundary, restated from [PRD-141 §12](../PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md#12-integration-coupling-rules-the-principle-written-down). These are the rules CI enforces; review will revert anything that breaks them.

- **Do not modify `orchestrator/api/widgets/chat.py`** to special-case your vertical. The dispatcher is generic — it reads `workspace.settings["vertical"]`, looks up the registry, and calls the plugin. If you find yourself adding an `if vertical == "<new>"` branch in `chat.py`, you are doing it wrong.
- **Do not modify `orchestrator/modules/context/`, `orchestrator/consumers/chatbot/`, or `orchestrator/modules/knowledge/graph_service.py`** to add vertical-specific behaviour. These are gated by the same CI grep gate as `chat.py`.
- **Do not add columns or fields with vertical-specific names to generic models** (no `Workspace.barbershop_*`, no `Workspace.<new>_*`). Use `settings` JSON for everything vertical-specific the workspace needs.
- **Do not let your vertical's name appear (as a string match OR an import) in any file outside `orchestrator/integrations/<new>/`.** A barbershop opener file referenced from `chat.py` defeats the whole abstraction. The only file outside the integration folder that may name your vertical is the one-line `from . import <new>` in `orchestrator/integrations/__init__.py` (which is itself inside the integrations tree — generic surfaces remain clean).
- **Do not add partner-specific keys to the generic skill prompt.** Skills are per-vertical files; the generic skill stays generic. See section (g).
- **Do not raise out of `handle_widget_message`.** A plugin that crashes blocks the user's chat. Catch, log, return the unchanged message.

---

## (e) Test requirements

Two things must land alongside your plugin code.

### Snapshot fixtures and equivalence tests

Mirror the layout under `orchestrator/integrations/shopify/tests/fixtures/`:

```
orchestrator/integrations/<new>/tests/
├── conftest.py
├── fixtures/
│   ├── <new>_page_context.json        # realistic page_context for one trigger
│   ├── <new>_<trigger>_context.json   # one file per trigger your plugin handles
│   ├── expected_<new>_<trigger>_opener.txt   # verbatim string handle_widget_message returns
│   └── README.md                      # how the fixtures were generated (synthetic vs captured)
└── test_widget_proactive.py
```

Each test loads a fixture context, calls `handle_widget_message`, and asserts `result.message == expected_opener_text`. Byte-equality is the bar — a whitespace change or a reordered dict iteration in your rewrite path is a regression. See `orchestrator/integrations/shopify/tests/test_widget_proactive.py` for the working pattern.

If your rewrite reads a knowledge graph or other workspace state, capture (or hand-craft) a small graph snapshot fixture and load it from the test the same way the Shopify tests load `inbuild_graph_snapshot.json`.

### Grep gate update

The CI gate `scripts/ci/check-no-shopify-in-generic.sh` enforces the §12 coupling rule for Shopify. When you ship a new vertical with its own forbidden-key set, **extend the gate**. Add a sibling script (or a generalised gate that takes the vertical as an argument — author's discretion, but keep the existing Shopify gate working). The CI workflow lives at `.github/workflows/check-shopify-isolation.yml` — wire your new check in the same job.

The gated paths are the same as the Shopify gate's: `orchestrator/api/widgets/`, `orchestrator/modules/context/`, `orchestrator/modules/knowledge/graph_service.py`, `orchestrator/consumers/chatbot/`. Forbid your vertical's distinctive `page_context` keys (the equivalents of `productHandle`, `cartItems`) AND the `<new>_` prefix on identifiers in those paths.

### Local validation

Before opening the PR:

```bash
cd orchestrator && python -m pytest integrations/ -x --timeout=30
bash scripts/ci/check-no-shopify-in-generic.sh   # must still pass
bash scripts/ci/check-no-<new>-in-generic.sh     # the gate you added
```

All three must be green.

---

## (f) Where vertical-specific admin UI surfaces go

This PRD set `workspace.settings["vertical"]` via API + migration, not a UI control (PRD-141 deferred admin UI as OS-3). When you do need vertical-specific admin UI — a settings panel for the partner's API keys, a status widget for the catalog sync, a connection wizard — keep it folder-isolated the same way the backend plugin is.

- **Backend admin endpoints** for `<new>` live under a vertical-namespaced router, e.g. `orchestrator/api/<new>.py` for partner-specific CRUD (catalog sync status, OAuth callbacks, webhook registration). Mirror what `orchestrator/api/shopify.py` does for its sync path. The router is opt-in: not registered in the generic dispatch chain, only mounted when relevant.
- **Frontend admin pages** for `<new>` live under a vertical-namespaced route, e.g. `frontend/src/app/integrations/<new>/`. Visibility is gated by `workspace.settings["vertical"] === "<new>"` — generic users never see the panel.
- **Shared admin shell** (the page chrome, the workspace selector, the settings nav) stays generic and discovers integration panels via the existing registry pattern. Do not hard-code `<new>` into the shared shell.

The CI grep gate does **not** scan `orchestrator/api/<new>.py` or `frontend/src/app/integrations/<new>/` — those paths are vertical-namespaced and allowed to contain your vertical's identifiers. The gate scans only the generic surfaces.

---

## (g) Skill registration in `automatos-skills`

Plugins rewrite the user message; **skills** tell the agent what to do with it. Both must ship together for a vertical to behave end-to-end.

Add your skill to the `automatos-skills` repo under a vertical-namespaced folder:

```
automatos-skills/<new>/<new>-support/SKILL.md      # v1.0
```

The skill prompt must:

- **Document the page-context shape your plugin emits.** When `chat.py` prepends or your plugin rewrites a `Context: {...}` block, the agent reads it from there — the skill explains what keys to expect and which tools to call against them.
- **Reference only generic platform tools** (`platform_query_graph`, `platform_search_memory`, the rest of the PRD-71 tool catalog). If you find yourself wanting a `<new>_*` tool, ask whether it can be expressed as a generic-but-tagged platform tool first.
- **Stay out of the generic skill.** The `generic/default-widget-support/SKILL.md` file (added in PRD-141 US-017) is the fallback for workspaces without a vertical. Do not add `<new>`-specific instructions to it.

Once the skill is published, attach it to your vertical's workspace template (or to the canary workspace directly) so the agent loads it on every widget message.

See `automatos-skills/shopify/shopify-support/SKILL.md` for a working reference — that file is the model the PRD-141 Phase 3 stories use for `<new>` too.

---

## Working reference

Whenever this doc says "and X" without spelling out the detail, the answer is in **`orchestrator/integrations/shopify/widget_proactive.py`** — it is the production implementation of every pattern this README describes:

- Two-trigger dispatch (`proactive_opener`, `cart_idle`) with a single `handle_widget_message`
- Graph-resolved entities (single-seed and multi-seed walks via `GraphifyService`)
- Telemetry passed back to the dispatcher's log line without crossing the generic boundary
- Folder-isolated helpers (`context_fields.py`) referenced only from within the package
- Snapshot tests asserting byte-equality against captured fixtures

Read it, then come back here and ship your vertical.
