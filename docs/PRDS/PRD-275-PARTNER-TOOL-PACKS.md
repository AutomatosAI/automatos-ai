# PRD-275: Partner tool packs — read-only tools on a partner's own API

**Status:** Built, behind `PARTNER_TOOLS_ENABLED` (off by default). Written 10 Nov 2026.
**Maintainer approval:** Gerard, 10 Nov 2026, for a new tool source and for the exception to the Composio rule in §6. Dave reviews.
**The PRD number is provisional** until the maintainer confirms it.
**Grounded on:** `main @ 1372f6260`. Labels: **[T]** read in code at that commit; **[I]** inferred.

## 1. Why

A self-hosted Automatos sits inside another company's infrastructure, next to that company's own product. Its agents need to read that product's records (orders, shipments, tickets, accounts) to answer the questions the company's own staff ask.

Today no tool can do that safely:

| What exists | Why it does not fit [T] |
|---|---|
| Composio apps | A hosted third party. An install that exists to keep data inside the company's network cannot route its own API through it. |
| `http_request` | Reaches Automatos's own hosts only, and the model writes the headers, so a key would pass through the model (`modules/tools/execution/exec_shell.py`). |
| `platform_web_fetch` | Any public page, no key (`modules/tools/discovery/handlers_web.py`). |

**The gap:** a tool that calls a partner's API with a key the server holds, on addresses and paths the operator fixed in advance.

## 2. Goals

- An operator installs a **pack**: a JSON file that declares read-only tools on one partner's API.
- The partner's base URL and key live in the workspace's credential store. **The server adds the key**; the model never sees it.
- Only an agent the workspace's managers name at install is offered a pack's tools.
- Every call is a GET to the partner's own host, with every guardrail in §5 enforced in code and tested.
- Every call that reads the key leaves an audit row: the tool, the partner's status code, how long it took.

## 3. Non-goals

- **No writes.** A pack has no method field. Approvals for writes are a later design.
- **No partner-specific code in core.** A partner's pack, and its agent's instructions, live in that partner's own repository.
- **No new table, credential type or dependency** (§7).
- **No UI.** Three routes and the existing credential and agent routes are the surface.

## 4. Design

### 4.1 The pack

```json
{
  "schema": "automatos-partner-pack/v1",
  "name": "acme",
  "title": "Acme Hub",
  "version": "0.1.0",
  "tools": [{
    "name": "find_orders",
    "description": "Search and look up orders.",
    "scope": "orders:read",
    "operations": [
      {
        "name": "search",
        "description": "List orders, newest first.",
        "path": "/api/v1/orders",
        "parameters": {"type": "object", "properties": {
          "status": {"type": "string", "enum": ["open", "shipped"]},
          "limit": {"type": "integer", "minimum": 1, "maximum": 50}}},
        "query": {"sort": "-createdAt"},
        "response": {"root": "data", "keep": ["id", "status", "customer.name"], "max_items": 25}
      },
      {
        "name": "get",
        "description": "One order by its id.",
        "path": "/api/v1/orders/{order_id}",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
        "response": {"root": "data", "keep": ["id", "status", "total"]}
      }
    ]
  }]
}
```

- **A tool is a few GET operations on one kind of record.** The model is offered `acme_find_orders` and picks one with `operation`. A tool with one operation takes that operation's parameters directly.
- **A pack's name is at most 20 characters** and a tool's at most 41, so `<pack>_<tool>` fits the 64 characters model providers allow a tool name.
- **A pack carries no host and no key.** One pack serves a development and a production install.
- **Parameters are scalars** (`string`, `integer`, `number`, `boolean`) with `enum`, `minimum`, `maximum`, `minLength`, `maxLength`.
- **`query`** is a set of fixed values every call of the operation sends.
- **`response`** says where the records are (`root`), which fields reach the model (`keep`, dotted paths) and two caps: `max_items` (25 by default, 200 at most) and `max_chars` (4,000 by default, 6,000 at most). The chat keeps the first 2,000 tokens of any tool's answer, so the caps sit below that and a record is never cut in half.
- **`scope`** names the permission the partner's key needs. It is said back when the partner refuses the key.
- **An unknown field is refused**, so a pack that asks for anything else does not install.

The format and its checks are in `modules/tools/partner/pack.py`.

### 4.2 Installing a pack and granting it

1. The operator creates a credential of the existing type `generic_api`: base URL, API key, and how the key is sent (`bearer`, or `header` with a header name that ends in `Key` or `Token`). `POST /api/credentials/`.
2. `PUT /api/partner-packs/{name}` with the pack, the credential's id and the agents that may use it (`agents: [12, 31]`). For workspace managers (`workspace:manage`), as creating the credential is.
3. The pack's name is added to each of those agents: `configuration.partner_packs: ["acme"]`, through `PUT /api/agents/{id}`.

An agent is offered a pack's tools only when **both** hold: the install names the agent, and the agent lists the pack. Installing a pack for no agents offers it to no one. Auto and every other agent keep the tools they had.

**Who may grant a pack:** the workspace's managers, at install (§10-5, decided). Editing an agent needs only `agents:create` or `agents:update`, which an editor holds [T]; listing a pack on an agent the install does not name gives it nothing. An editor cannot reach a partner's records a manager did not allow.

| Route | What it does |
|---|---|
| `GET /api/partner-packs` | The workspace's packs and the tools each gives. |
| `PUT /api/partner-packs/{name}` | Install, or replace the pack of that name. 422 says what to change. |
| `DELETE /api/partner-packs/{name}` | Remove it. |

All three answer 404 while `PARTNER_TOOLS_ENABLED` is off.

### 4.3 One call

1. **Offered.** `tool_router._get_tools_for_agent_core`, the one source of an agent's tool schemas, adds the tools of the packs the agent lists. A lean turn (PRD-274) names them in its grants.
2. **Looked up.** `UnifiedToolExecutor.execute_tool` gains one innermost wrapper. A name shaped `<pack>_<tool>` that is not one of the executor's own tools is looked up: the workspace has the pack, and the calling agent lists it. Offering a tool is not what lets it run.
3. **Gated.** The executor's own gates judge the call first: a widget turn's scopes, a lean turn's grants, the policy plane.
4. **Checked.** The arguments are validated against the operation's schema. A bad call is refused here, before the key is read.
5. **Sent.** The key is read through the credential store. The server builds the URL from the credential's base URL and the pack's path, and sends one GET through the outbound seam (`core/security/web_access.py`).
6. **Shaped.** Only the fields the pack keeps reach the model, within the caps.
7. **Recorded.** One `partner_call` row on the credential's audit log. A call whose row cannot be written gives no answer.

The executor's telemetry row (`tool_execution_logs`) is written for every call, as for any tool.

## 5. Guardrails

Each is enforced in code and has a test in `orchestrator/tests/test_prd275_partner_tools.py`.

| Guardrail | How it is enforced |
|---|---|
| **GET only** | A pack has no method field and an unknown field is refused. The client builds no other request. A `method` argument from the model is an undeclared parameter. |
| **The partner's own host only** | The host comes from the credential, the path from the pack. A path value is one plain segment, percent-encoded. The built URL must have the base URL's origin. |
| **https, to a checked address** | The base URL must be https. The request goes to the address the outbound seam checked, with TLS verified against the name. Private, loopback, link-local and metadata ranges are refused. The environment's proxy settings are not read. |
| **No redirects** | A 3xx is refused and its `Location` is never fetched. |
| **Valid parameters only** | Checked against the operation's schema with `additionalProperties: false`. A value is at most 1,000 characters. |
| **A capped answer** | The body is read to 1 MB and no further. A list is cut to `max_items`, and what is kept never exceeds `max_chars`. |
| **One deadline** | 30 seconds for the whole call, however slowly the partner answers. |
| **The key stays hidden** | Sent in a header the server writes. Never in the URL, a result, a log line, the telemetry row or the audit row. |
| **Only a key header** | The credential may name `bearer`, or a header whose name ends in `Key` or `Token`. It cannot name a header that changes the request: a method override, a forwarded address, `Host`. |
| **Per-agent grant** | The install names the agents, and each agent lists the pack; both are checked when a tool is offered and again when it runs. |
| **Workspace isolation** | A pack, its credential and its agent must share one workspace. Another workspace's credential is "not found". |

## 6. The exception to "Integrations go through Composio"

AGENTS.md says integrations go through Composio and that no new third-party API client is added. PRD-274 §3 repeats it. **This PRD is one named exception**, approved by the maintainer:

- **What it allows:** read-only calls to a partner's own API from a pack, under every guardrail in §5.
- **What it does not allow:** a client for a public SaaS app (that stays Composio), a write, or a key anywhere but the credential store.
- **Nothing in Composio changes.** The deny list is untouched.

AGENTS.md carries the exception in the same pull request.

## 7. What is reused

| Need | Reused [T] |
|---|---|
| Where an installed pack lives | `workspace_tool_config`: workspace, `tool_id` (`partner:<name>`), `credential_id`, `configuration` JSONB. No migration. |
| The base URL and key | The `generic_api` credential type, encrypted, created through the existing routes. |
| An audited key read | `CredentialStore.get_decrypted_credential`, with the workspace check `ssh_execute` uses. |
| The audit row | `credential_audit_logs`, a new action `partner_call`. |
| A safe outbound request | `core/security/web_access.py`: resolve, check, pin; the pattern `platform_web_fetch` and run callbacks use. |
| Argument validation | `jsonschema`, already a dependency. |
| Wiring into long functions | A wrapper on each seam, as PRD-274's lean turns do. No long function's body is touched. |

**Why a new tool source and not a platform action:** a platform action is one fixed schema for every workspace. A pack is workspace data, and the model does better with one real schema per tool.

## 8. Rollout

- `PARTNER_TOOLS_ENABLED` is off by default. Until it is on, no pack installs, none is offered and none runs: the three seams return at once.
- Both editions run the same code. The first use is a self-hosted install.

## 9. Limits, stated plainly

- **A partner API on a private address is refused.** The outbound seam always refuses private ranges. A self-hosted install whose partner API is only reachable privately needs a decision (§10-2).
- **Session-mode (CLI) agents are not offered partner tools.** Their tools are a fixed catalogue. Chat turns, lean turns and agent runs are covered.
- **A turn Auto answers on its lightweight lane loads no tools but the platform dispatcher** (`consumers/chatbot/service.py`, the ATOM branch) [T]. An Auto that lists a pack is not offered it on such a turn. A turn of an agent the owner chose, or one Auto hands a ticket to, always loads its tools [T].
- **No list parameters.** Scalars only.
- **No rate limit and no cap on calls at once.** A model may call a partner tool as often as its turn allows. Each call writes two rows to `credential_audit_logs` (the store's own `accessed` row and `partner_call`), and nothing prunes them.
- **Under the policy plane, a partner tool is classed `internal_write`,** not `read`: the classifier knows read-only platform actions by their registry entry. The plane is off by default.
- **A partner's answer is not kept as a tool-outcome memory,** and the telemetry row holds no answer [T]. It is part of the conversation, as any tool's answer is: the transcript keeps it, and memory distillation may keep facts from it [I].
- **A pack does not move with a workspace** (PRD-273): `workspace_tool_config` is a connection, re-made at the destination.
- **Packs are not signed.** A workspace manager installs what they choose, as with a credential.

## 10. Open questions for the maintainers

1. **The AGENTS.md wording** of the exception (§6): as written in this pull request, or tighter?
2. **Private partner addresses on a self-hosted install:** keep refusing, or add an operator allow-list in `config.py` (local edition only)? *Recommendation:* decide when a real install needs it.
3. **A recent sign-in (step-up) to install a pack,** as for minting an API key? *Recommendation:* no; creating the credential, the more sensitive step, has none.
4. **The PRD number.**
5. **Who may grant a pack to an agent:** decided, workspace managers only. The install names the agents that may use the pack, so editing an agent (`agents:update`, which an editor holds) cannot give it a partner's records.
6. **A rate limit on partner calls,** per workspace or per credential, before the flag goes on for hosted workspaces? *Recommendation:* yes before hosted use; not needed for a single self-hosted install.
