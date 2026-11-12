# Tools module

`orchestrator/modules/tools/` is where an agent's tool surface is assembled and
executed: which tools an agent is offered, how a call is validated and routed,
and how results come back. It serves both editions; the only edition-sensitive
piece is Composio availability (below).

## Layout

```
modules/tools/
├── tool_router.py            ToolRouter — get_tools_for_agent(), execute_tool(), validation,
│                             the capability filter; the ONE place Composio availability is consulted
├── composio_tool_router.py   the Composio leg of routing
├── registry/                 ToolRegistry, ToolSpec, ToolCategory, SecurityLevel — register_tool(ToolSpec)
├── execution/                UnifiedToolExecutor
├── discovery/                the platform actions Auto executes: ActionRegistry plus
│                             actions_<domain>.py / handlers_<domain>.py pairs
├── builtin/                  built-in tools (scratchpad)
├── partner/                  partner tools: read-only tools on a partner's own API, from a pack (PRD-275)
├── capabilities/ services/ sync/   capability filtering, supporting services, the Composio catalogue sync
├── formatting/               result_formatter — the standard result shape for LLMs and the UI
├── widget_callback.py
└── tests/
```

## Where an agent's tools come from

`get_tools_for_agent(agent_id, workspace_id)` (exported from `modules.tools`)
is the single source of truth for an agent's tool schemas. Four sources feed
it:

1. **Platform actions** — `platform_*` operations Auto runs against the
   platform itself (agents, Playbooks, Deliverables, documents, Missions,
   analytics, …), defined as `ActionDefinition`s in
   `discovery/actions_<domain>.py` and executed by
   `discovery/handlers_<domain>.py`. Each definition carries its OpenAI
   function schema, a permission level (`read` / `write` / `destructive`),
   and the confirmation / admin / super-admin flags.
2. **Registry tools** — `ToolSpec`s registered on the `ToolRegistry`
   (`registry/`), including the built-ins.
3. **Composio tools** — third-party app actions from the Composio catalogue
   (`composio_apps_cache`), bound per agent — when Composio is available.
4. **Partner tools** — the tools of the partner packs the agent's
   `configuration.partner_packs` lists and whose install names the agent,
   when `PARTNER_TOOLS_ENABLED` is on (below).

## Composio availability — the degrade seam (PRD-233 S2)

Composio is a hosted service that needs a key. The platform key is a
hosted-edition concern; the local edition is bring-your-own
(`COMPOSIO_API_KEY` in `.env`, env-only). The module handles absence
explicitly instead of failing open:

- **One predicate.** `core/composio/client.py: composio_available()` is true
  when `config.COMPOSIO_API_KEY` is set *and* the SDK imports;
  `composio_unavailable_reason()` says which is missing. Evaluated once per
  process.
- **One exclusion point.** `tool_router.py` consults the predicate at a single
  seam: unavailable ⇒ Composio tools are excluded from discovery and schemas,
  so the model is never offered a tool that cannot run. Platform actions and
  registry tools route as before.
- **One refusal shape.** A direct call to a Composio tool while unavailable
  returns `{"success": false, "error_code": "integrations_unavailable", …}`
  with the reason and the fix — never a silent success or a pass-through.
- **The UI agrees.** `GET /api/tools/integrations/status` (`api/tools.py`)
  returns the same predicate plus `key_configured`, `apps_cached`,
  `last_sync` and `sync_status`; the Tools page renders *"Integrations are
  disabled — no Composio API key is configured."* from it.
- **Boot.** `core/composio/bootstrap.py` runs as a boot stage: key + empty
  catalogue ⇒ a full catalogue sync on a background thread, then the seeded
  marketplace agents are re-bound to their apps; key + populated catalogue ⇒
  rebind only; no key ⇒ a log line saying why integrations are off. It never
  blocks or fails the boot.

Regression rule: the router's access gate stays fail-closed. Do not add a
second place that decides whether Composio is on.

## Partner tools (PRD-275)

`partner/` gives an agent read-only tools on a partner's own HTTP API. The one
exception to "integrations go through Composio" (AGENTS.md): for a partner's
API, write a pack; never add a client.

- **A pack is data.** A JSON document (`partner/pack.py`) declares tools, each
  a few GET operations: a path template, scalar parameters, the fields of the
  answer to keep and two caps. No host, no key, no method.
- **Installed per workspace.** `PUT /api/partner-packs/{name}` stores the pack
  as a `workspace_tool_config` row (`tool_id` `partner:<name>`) beside one of
  the workspace's `generic_api` credentials: the base URL, the key and how the
  key is sent.
- **Granted per agent, by a manager.** The install names the agents that may
  use the pack (`agents`), and each lists it in `configuration.partner_packs`.
  Only an agent with both is offered its tools (`partner/offer.py`, on
  `_get_tools_for_agent_core` and on a lean turn's grants), and both are read
  again when a call runs. An editor can edit an agent but not install a pack,
  so listing a pack on an agent the install does not name gives it nothing.
- **Run on the executor's seam.** `partner/runner.py` wraps
  `UnifiedToolExecutor.execute_tool`: the executor's own gates, then the
  arguments are checked, the key is read through the credential store, one GET
  goes through `core/security/web_access.py` (pinned address, no redirects,
  bytes capped), the answer is cut to the pack's fields, and one `partner_call`
  row is written to the credential's audit log.
- **The key never leaves the server.** It is not in the tool's schema, its
  result, a log line, the telemetry row or the audit row.
- **A partner's answer is untrusted input.** It reaches the model as the
  partner wrote it, so text in a record can try to steer the agent's other
  tools. Grant a pack to a lean agent (`runtime_profile: lean`) whose other
  grants are read-only; a lean agent is offered only what it is granted.

Regression rule: a pack has no method field and the client sends only GET. A
change that lets a pack, an argument or a redirect choose the method, the host
or the header needs the maintainer's approval first.

## Adding a platform action

Two files in `discovery/`, one for the definition and one for the handler —
follow an existing pair such as `actions_blog.py` / `handlers_blog.py`:

1. In `actions_<domain>.py`, a `register_<domain>_actions(registry)` function
   that calls `registry.register(ActionDefinition(name="platform_…",
   description=…, category=…, parameters={…OpenAI JSON schema…},
   permission_level=…, requires_confirmation=…))`. The description is
   embedded for semantic selection, so write it for the model.
2. In `handlers_<domain>.py`, the handler that performs the operation and
   returns the standard result shape.
3. A test under `modules/tools/tests/`.

Prefer extending an existing action over adding a near-duplicate, and a
platform action over a new Composio dependency for anything the platform can
do itself.

## Result shape

Handlers and tools return a dict the formatter
(`formatting/result_formatter.py`) can standardise:

```python
{"success": True, "data": ..., "metadata": {...}}                       # ok
{"success": False, "error": "what went wrong", "error_code": "..."}    # expected failure
```

## Conventions

- No `os.getenv` here — read `config`.
- Errors are explicit: a tool that cannot run says so in the result; the
  router never turns "unavailable" into "allowed".
- `super_admin_only` definitions are excluded from every listing and
  selection path unless explicitly included (fail-closed); keep destructive
  actions behind `requires_confirmation=True`.
