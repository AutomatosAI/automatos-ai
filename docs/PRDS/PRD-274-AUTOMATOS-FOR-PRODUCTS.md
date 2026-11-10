# PRD-274: Automatos for products — the platform layer other products run on

**Status:** Approved, 9 Nov 2026 (Gerard): numbered PRD-274, with every §8 recommendation decided as recommended.
Written by Dave, 8 Nov 2026. Waves 0, 2 and 4 are built as draft PRs stacked on this one (§6); Waves 1, 3 and 5
start next.

**Why now:** external tools built to run *on* Automatos have independently hit the same missing layer.
- **One uses Automatos through its API as a machine customer.** Its platform requirements include a server key that
  can start a Playbook, file a task and write documents, and a "machine turn": no retrieval prefetch, a system and
  schema field, honoured model choice.
- **Another was rebuilt as a spike on a local Automatos:** its agent as an Automatos agent in its own workspace, and
  its memory in Automatos memory. Its measurements are in §4.6.

**The general need:** Automatos today is built around one owner talking to Auto. Every gap below appears when
someone else's product sits between Automatos and that product's own users: a shop, a support desk, a tutor, a
monitoring service, an incident room, a client's own app. Shopify is already one such product (the vertical setup flow, PRD-183 S5).
This PRD makes that the general case.

**Scope:** automatos-ai (core, API, memory, the chat pipeline). The web app only for key and workspace management.
Automatos Go and the Design Kit need nothing.

**Depends on:** the vertical setup flow (PRD-183 S5, `integrations/provisioning.py`, `api/verticals.py`), SDK keys
(`core/services/api_key_service.py`, PRD-195), the memory plane (PRD-187, `modules/memory/`), fact distillation
(PRD-159, `consumers/chatbot/smart_memory.py`), the widget memory rules (F154, F155), the page-context allow-list
(`services/page_context.py`).

**Touches what AGENTS.md says to ask a maintainer about first:** authentication and permissions (keys on new routes),
tenancy (workspaces created by a product, end-user scopes), and the removal of user data (end-user erasure). This PRD
is that ask.

## 1. Introduction

A **product** (a vertical: a shop app, a monitoring service, a client's own app, any external tool) runs on Automatos through a
server key. It creates a workspace for each of its own customers, projects or cases, and drives agents in them from
its backend. Its agents run lean, on the model it chose, with the product's live state handed in each turn and never
stored as memory. Automatos remembers the people the product serves, keeps facts in the shape the product's domain
needs, can say "where were we" after a break, and shows each agent only the facts it is allowed to know.

None of this is specific to one product. Each capability below names the other users it serves.

## 2. Goals

- A product, with its own key, creates, lists and purges workspaces for its customers (sub-tenants), each seeded with
  its agent roster, without code changes in automatos-ai for each new product.
- A product's server key, with scopes, reaches the agent runtime: chat turns, memory, Playbooks, tasks and documents,
  not only the widget and the board.
- A product's agent can run a **lean turn**: only the tools and context it was given, on the model it was given, with
  a time to first token close to the model's own.
- A product hands in per-turn **app context** (its live state) that the agent sees and memory never stores.
- Automatos remembers **the people a product serves** (end users), per end user, with consent, retention and erasure.
- What counts as a durable fact follows the product's domain (**memory profiles**). A product's per-turn framing and
  app context are never stored as facts; the owner's own standing instructions are kept, as today (PRD-256). A
  restated fact may sit beside the original until the hourly consolidation folds it.
- "Catch me up" or "where were we" after a break recalls from the stored facts.
- In a workspace with several agents, a fact can be limited to the agents allowed to know it (**need-to-know**).
- Auto, for the owner, behaves exactly as today unless a wave says otherwise.

## 3. Non-goals

- **A second chat stack.** Lean turns and machine turns are profiles of the existing pipeline
  (`consumers/chatbot/service.py`), not a parallel one.
- **A new third-party integration path.** Integrations still go through Composio; nothing here weakens the deny list.
- **Product-specific logic in core.** A product's rules (its domain logic, a monitoring service's matching, a shop's catalogue)
  stay in the product. Automatos proposes; the product disposes, and stays the authority over its own state.
- **Billing design.** Who pays for a product's sub-tenant workspaces is decided (§8-2: the product's owner workspace); the metering design is Wave 5's.

## 4. What exists today

### 4.1 Workspaces and verticals

- There is no route that creates a workspace. Workspaces are created on sign-up (hosted) or are the one default
  workspace (local); `scripts/create_test_workspace.py` and `scripts/purge_test_workspace.py` create and purge
  isolated ones from the console (the nightly suite and the PRD-247 simulation harness use them).
- **Verticals** already do find-or-create: `POST /api/verticals/{v}/provision` (`api/verticals.py`) runs
  `provision_vertical` (`integrations/provisioning.py`): find or create the workspace for an external id, seed the
  agent roster, mint a key. Each vertical registers a `VerticalProvisioner` in code and has its own internal key in
  config (`SHOPIFY_INTERNAL_API_KEY`). Two are registered: Shopify, and BudStacks (`integrations/budstacks/`), whose
  re-provision keeps the existing key (`reuse_existing_key`) because its app stores the key it was given.

### 4.2 Keys

- SDK keys (`ak_pub_*`, `ak_srv_*`) authenticate on the widget routes and on a narrow board path
  (`core/auth/hybrid.py` from line 625, which says why: changing the shared dependency "would make every endpoint
  accept SDK keys"). A server key cannot call `POST /api/chat`, the memory routes, Playbooks or documents.
- The `X-API-Key` lane is the instance's own `ORCHESTRATOR_API_KEY` only (admin-equivalent, `_api_key_context`).

### 4.3 The chat turn

- `POST /api/chat` takes no model: PRD-180 S3 removed the per-message selector, so the model comes from the agent's
  `model_config` or Auto's tier. `POST /api/widgets/chat` declares `model_id` (`api/widgets/chat.py:61`) but nothing
  reads it.
- There is no per-turn system or context field. `context` is a page context: allow-listed keys, strings capped at
  128 characters (`services/page_context.py`). It says where the user is, not what the app knows.
- Every turn assembles the platform's sections and offers the platform tools. In the spike (§4.6) an agent created
  with no tools and a 3,900-token context (`GET /api/agents/{id}/assembled-context`) was sent 34,000–45,000 prompt
  tokens and 26 platform tools per turn.

### 4.4 Memory

- `POST /api/memory` stores text as given into the **workspace** namespace. `GET /api/memory/search` searches by
  meaning, with no filter by tag or metadata (`api/widget_memory.py`). A failed store still returns 201 (the store's
  error dict is not checked).
- After an owner chat turn, an LLM pass distils durable facts (`_distill_durable_facts`) and stores them in the
  workspace tier by default; the agent tier only on phrases such as "for this agent" (`_classify_memory_tier`,
  PRD-159 S5). `platform_store_memory` files to the workspace even when the model names an agent (F189).
- Widget turns store nothing (F154, the early return in `consumers/chatbot/service.py`) and recall only the agent's
  own memories (F155). That was right for anonymous shop visitors; it also means no product can remember its users.
- Recall goes through the Context Router (`modules/memory/context_router.py`): a query it reads as temporal is routed
  to time-based layers.

### 4.5 What the external tools needed, side by side

| Capability (§5) | The machine-customer tool | The spike (§4.6) |
|---|---|---|
| 5.1 Sub-tenants | One workspace, with higher concurrency | A workspace per customer project |
| 5.2 Headless access | Playbooks, tasks and documents with a key; empty permissions deny; idempotency; per-key limits and ceiling | Chat and memory with a server key |
| 5.3 Lean turns | A machine turn: no prefetch, honoured model | 34–45k tokens and 26 tools per turn, 9.9 s to first word |
| 5.4 App context | A system and schema field | The tool's live state and its own results each turn |
| 5.5 End-user memory | — (no accounts at launch) | Each end user's history |
| 5.6 Memory profiles | Results tied to runs | Domain facts (people, places, commitments) |
| 5.7 Continuity | — | "Catch me up" at a new session |
| 5.8 Need-to-know | — | What one agent may know that another may not |

Other users of each capability are listed in §5.

### 4.6 The spike's measurements

A local install on the `edge` image of 1 Nov 2026, measured on 8 Nov 2026. Main has moved since 1 Nov; Wave 0
re-measures on main.

**Recall quality**, 24 user turns, right fact in the top 4 (in brackets: right fact first):

| Facts in memory | Keyword match | Keyword + Postgres full-text | Automatos memory |
|---|---|---|---|
| 40 | 13 (10) | 21 (12) | 22 (18) |
| 400 (half of them look-alikes) | 13 (10) | 16 (10) | 20 (15) |

At 400 facts, turns that describe a person rather than name them: Automatos 7/10, full-text 3/10. Search took a
median of 0.7–1.7 s, almost all of it the embedding call. Memory search is already the strongest part.

**The tool's agent as an Automatos agent**, 16 scripted turns, the tool's own code applying every structured command
the agent emitted:
- **The model followed the host app's contract:** 6 structured commands, all valid, none of the app's own results
  invented.
- **Turn cost:** 34,000–45,000 prompt tokens per turn; 9.9 s median to the first word; about $0.004 per turn by
  Automatos's own per-agent accounting (176 model calls for 20 chats).
- **Memory stored 46 facts:** about 30 repeat the per-turn app context; four turned the harness's per-turn framing
  ("The AI assistant should end each reply by asking…") into standing instructions, which is what distillation is
  meant to do with an owner's instructions (PRD-256), so a product's framing needs the app-context block (§5.4);
  about eight are the domain facts that mattered. Restated facts had not been folded yet: consolidation
  (`run_sleep_time_consolidation`) runs hourly and the run lasted about 20 minutes.
- **A new-session "catch me up" recalled nothing:** the memory section injected ~0 tokens and the agent recapped a
  different, earlier chat. The cause (found in Wave 0, #1434): a chat turn's long-term recall read only the turn's
  agent namespace, while distillation files facts in the workspace's by default (PRD-159 S5). Every chat turn has an
  agent, Auto included, so no chat turn's router recall reached a workspace-tier fact; the few distilled to the
  agent's own tier were reachable.

## 5. Design

Each capability names its other users. None needs a new third-party dependency.

### 5.1 Sub-tenants: a product creates workspaces for its customers

**Who else:** shop apps (per shop, today's Shopify path), agencies and MSPs (per client), incident rooms (per
incident), client PoCs (per environment), tutoring (per class).

Generalise `provision_vertical` from "registered in code" to "registered as data": a **product** record (name,
owner workspace, its internal key's hash, the agent roster template, key scopes to mint, retention) created by an
admin. The product's key calls find-or-create by external id, lists its workspaces, and purges one (reusing
`purge_test_workspace.py`'s guarded purge as a service). Sub-tenant workspaces carry `settings.product_id` and are
invisible to other products. Shopify becomes the first product record; its provisioner keeps its hooks.

### 5.2 Headless access: scoped server keys on named routes

**Who else:** any SaaS embedding Automatos, partner backends, machine customers, client integrations.

Add the server-key lane to a named set of routes, never the shared dependency (the reason in `hybrid.py` stands):
chat turns (§5.3), memory read and write (§5.5–5.8), starting a Playbook, filing a task, writing a Deliverable. Each
route declares the scope it needs. Empty permissions deny; an `Idempotency-Key` makes a retry safe; per-key spend
and rate limits are enforced; server keys get a per-key ceiling instead of the per-IP one.

### 5.3 Lean turns: a runtime profile for product agents

**Who else:** voice and Auto Live (time to first word), the Go app, widgets, anything real-time, and machine work.

An agent's `runtime_profile`: `full` (today, the default for Auto) or `lean`. A lean turn assembles only the agent's
own context, the memory recall it asks for (§5.5–5.8) and the app context (§5.4); offers only the tools granted to the
agent; and uses the agent's `model_config` or a per-request model from an allow-list the product's key carries.
Machine turns are a lean turn with no retrieval prefetch and a response schema.

**Target:** time to first token within 1.5 s of the model's own, and prompt overhead within 2,000 tokens of the
agent's assembled context.

### 5.4 App context: what the host app knows, this turn only

**Who else:** shop carts, support-ticket metadata, the document being edited, a dashboard's filters, an incident's
live state.

A `context` block on lean turns, separate from `page_context`: structured JSON and/or text, size-capped by config,
placed in the system prompt for this turn only. It is never written to the transcript that distillation reads, never
stored as memory, and not returned in history.

### 5.5 End-user memory: remembering the people a product serves

**Who else:** support that remembers a customer across visits, tutors that remember a student, coaches, companion
apps, a returning shopper.

A memory **subject** for end users: `MemoryNamespace` gains `end_user(product_id, end_user_id)`, written only when a
server key asserts the end user (a widget visitor without that assertion stays as F154 has it). Consent is the
product's responsibility and recorded per subject; retention is set per product; one call erases an end user's
memories (GDPR). Recall for an end user's turn reads that subject plus the workspace tier, never another end user's.

### 5.6 Memory profiles: what counts as a fact here

**Who else:** incident timelines (on-call and operations), case facts (legal), patient history (health), project decisions,
customer accounts, a machine customer's per-run results.

A workspace or agent picks a **memory profile**: the distillation prompt's fact types and examples. `general` is
today's. The others are named by domain (`customer`, `incident`, `case`, …) and are data, seeded like personas. For every
profile, distillation:
- ignores the app context block (§5.4), so a product's per-turn framing never becomes a standing instruction;
- keeps the owner's own standing instructions as today (PRD-256);
- leaves restated facts to consolidation, which already folds near-duplicates (`run_sleep_time_consolidation`);
  the Wave 0 eval measures whether they survive it.

Alongside distillation, an explicit write: `POST /api/memory` gains optional `subject`, `type`, `name` and `audience`
(§5.8). A product that knows its facts (a product's own structured facts, an incident tool's timeline) writes them directly, and a
failed store returns an error, not 201.

### 5.7 Continuity: "where were we"

**Who else:** anyone coming back after a break: Missions, an on-call shift handover, a returning user in any product.

A `resume` intent ("catch me up", "where were we"): recall the subject's L3 facts ranked by recency and
importance, plus the last session summary (`modules/memory/resume_context.py` and `GET /api/memory/resume` exist and
are the starting point). The spike's empty recall was not the temporal reading but the namespace: a chat turn read
only its agent's facts. Fixed in Wave 0 (#1434); the eval checks the `resume` intent on top of it.

### 5.8 Need-to-know: facts limited to the agents allowed to know them

**Who else:** a customer-facing agent beside internal notes (F155 patched this once, for widgets), HR or finance
agents in a shared workspace, an operations agent that should not see another team's notes, a product's
end-user-facing agent that should not see its operator's notes.

An optional `audience` label on a fact (a short list of names, e.g. `["operator"]` or `["finance"]`) and on an agent.
Recall returns a labelled fact only to an agent whose labels intersect it. Unlabelled facts behave as today. F155's
widget lane becomes one use of the general rule.

## 6. User stories

Each wave is its own branch and PR, opened as a draft and merged after this PRD. The order is §8-1's.

### Wave 0: measure on main, and fix what Auto already suffers from (built)

- `POST /api/memory` returns an error when the store fails, and nothing is kept in process memory (#1433).
- A chat turn's long-term recall reads the workspace's facts as well as its agent's (#1434).
- `evals/memory_pipeline.py`: `evals.memory_recall`'s corpus and gold set through the real chat-turn path, offline in
  CI or live against Qdrant and real embeddings, with look-alikes up to 414 facts; the acceptance test for Waves 3–5
  (#1435).
- `evals/memory_consolidation.py`: which restatements consolidation folds, and which distinct facts it loses (#1437).
  It found that consolidation deleted true facts about different subjects phrased alike: #1438, fixed on `main` in
  #1439.

### Wave 1: headless access (built)

- A real `POST /api/agents/{id}/execute`: the run is a board ticket assigned to the agent, answered at once with
  its id (#1461), in place of the fixed response (§8-8).
- Signed completion callbacks with the run id: an https `callback_url`, checked against private addresses when the
  run is asked for and before every send, an HMAC-SHA256 signature with the workspace's rotatable signing secret,
  and three attempts (#1462).
- `Idempotency-Key` on `/execute`: a repeat answers with the run it started, the same key on a different request
  is a 409, and a race files one ticket (#1463).
- Server keys on the named routes of §5.2, each with a declared scope, through one gate the shared auth dependency
  never learns about; empty permissions deny, a publishable key is refused (#1468).
- Per-key limits and ceiling: a key's requests and tokens a minute, under `SERVER_KEY_REQUESTS_PER_MINUTE`; on the
  key routes a valid key meets its own limits instead of the per-IP one (#1470).
- Memory read and write on keys (`memory:read`, `memory:write`); a key reads only facts no audience limits and none
  private to a person, and what it stores names the key (#1471).
- Starting a Playbook (`playbooks:execute`) and writing a Deliverable (`documents:write`) on keys (#1475).
- The headless chat turn: `POST /api/chat` with a key listing `agents:chat`, to a lean agent, with `model` and app
  context (§8-10); the model allow-list on the key (§8-5) (#1476).
- Alongside: the caller's address read one way, from the header the proxy sets (#1474); stacked PRs get every
  required check (#1473); private memories kept to their owner on list and search (#1472).

### Wave 2: lean turns and app context (built)

- `runtime_profile: lean` on agents: the agent's own identity, memory and the owner's standing rules, only the tools
  its tool groups grant (the executor refuses the rest), no Composio apps (§8-9) (#1446).
- A per-request `model` from the agent's allow-list, and an `appContext` block in that turn's prompt only, never
  stored (#1451).
- Acceptance, met in `evals/lean_turn.py` with live memory recall: 346 tokens of platform overhead (target 2,000),
  and the first token 430 ms after the model's own (target 1.5 s).

### Wave 3: end-user memory and the memory write API (next)

- The end-user subject, asserted by the product's server key (§8-6), consent record, retention and erasure; the
  extended `POST /api/memory`.

### Wave 4: memory profiles and need-to-know (built)

- Profiles as seeded data (`general`, `customer`, `incident`, `case`) in `memory_profiles`, picked per agent or
  workspace (#1452). In `evals/memory_profiles.py` the incident profile keeps all of an incident's timeline where
  `general` kept 70%, and the case profile stops general legal background being kept as a case fact.
- Workspace-defined profiles (§8-7): create, change and delete a workspace's own, on the table #1452 already
  has (#1460).
- `audience` labels on facts and agents, held at the durable store by the run's reader; Auto and the owner's own
  pages read everything (§8-11); consolidation never crosses audiences (#1453). F155's stricter widget rule stays,
  with the labels on top of it.

### Wave 5: sub-tenants (next)

- The product record (§8-3), find-or-create by external id with roster seeding, list, guarded purge; billed to the
  product's owner workspace (§8-2). Both registered verticals move onto it last, each keeping its behaviour: Shopify's
  re-provision mints a key, BudStacks' keeps the existing one.

### Follow-ups from Wave 1 (open)

| Follow-up | Waiting on |
| --- | --- |
| Set `CLIENT_IP_HEADER=X-Real-IP` on the hosted API service, then check Railway's edge replaces a client-supplied value | Gerard (an ops change on Railway) |
| Enforce a key's IP allow-list instead of refusing one (#1467) | The setting above |
| The app-wide per-IP limit is off in hosted (`AUTOMATOS_POLICY_PLANE` unset); the key routes are already exempt from it | A decision to turn the policy plane on |
| Re-send a callback a restart left `pending` (#1462 sends in-process) | Nothing: a small PR |
| A key's service user outlives the key and its workspace (no personal data in it) | Nothing: clean up on revocation and workspace erase |

## 7. Rollout

- Wave 0 fixes behaviour the platform already promises (a store that says when it failed, recall that reads the
  workspace's facts), so it ships unflagged.
- Waves 1–5 add capabilities. Each ships behind a flag in `orchestrator/config.py`, off by default, and changes
  nothing for the owner's Auto until enabled: `LEAN_TURNS_ENABLED` (Wave 2), `MEMORY_PROFILES_ENABLED` and
  `MEMORY_AUDIENCE_ENABLED` (Wave 4).
- Both editions: the local edition needs Waves 1–4 for self-hosted products; Wave 5 matters most to the hosted
  edition.
- The external tools above are the first products and the reference tests; Shopify moves onto Wave 5's product
  record last.

## 8. Decisions (decided 9 Nov 2026, Gerard: each as recommended)

1. **Order.** Wave 0 first: it improves Auto for every owner and builds the eval the rest is measured by. (Not:
   headless access first.)
2. **Who pays for a product's sub-tenant workspaces** (hosted): the product's owner workspace, rolled up. (Not: each
   sub-tenant on its own plan, or credits drawn from the product.)
3. **Products as data.** A product record created by an admin. (Not: a `VerticalProvisioner` in code per product.)
4. **The AI disclosure on lean turns.** Kept on every turn a person reads; the product may render it in its own words.
5. **Per-request model choice.** From an allow-list on the product's key. (Not: the agent's `model_config` only, or
   free choice.)
6. **End-user identity.** Asserted by the product's server key; the product is responsible for consent. (Not:
   Automatos-issued end-user tokens.)
7. **Memory profiles.** A small seeded set (`general`, `customer`, `incident`, `case`) plus workspace-defined ones.
8. **Signed completion callbacks with the run id, and the fake `POST /api/agents/{id}/execute`** (a route that
   returns a fixed response): in Wave 1, as machine-customer basics.
9. **Composio apps on lean turns.** Not offered until a product needs them: they need the per-turn app search, the
   main latency cost.
10. **`model` and app context on the widget chat route.** Not added there; they arrive with Wave 1's headless routes.
11. **Need-to-know and Auto.** Auto (and every system agent) reads every audience, as the owner's own pages do.

### Decisions made while building Wave 1 (10 Nov 2026, Dave)

12. **A key's chat turn runs as the key's own service user:** a non-login users row per key, never a workspace member.
    (Not: the workspace owner, whom a caller with no person used to default to.)
13. **Headless chat is its own scope, `agents:chat`.** (Not: `chat`, the widget's, which would grant it by accident.)
14. **A model a key names must be on the key's allow-list and the agent's.** (Not: the key's alone.)
15. **What a key starts is outside input.** A Playbook run a key starts is read as a payload, never the owner's
    words; a ticket a key files gets no operator consent and meets the workspace's approval policy.
16. **Memory delete and resume stay with people until Wave 3,** when a key's end users have their own scope.

### Open question for Wave 3

17. **The end-user subject's scope.** §5.5 names it `end_user(product_id, end_user_id)`, but the product record
    arrives in Wave 5. *Recommendation:* scope it by workspace, `(workspace, end_user_id)`, asserted by a server key
    of that workspace, so Wave 3 does not wait for Wave 5; each product customer becomes its own workspace in Wave 5,
    so no product id is needed.

## 9. Later

Proposals, for Gerard to place:

- **Field memory for products' multi-agent work** (Missions' shared field, `field_memory`) under the same subject and
  audience rules.
- **A product console** in the web app: a product's workspaces, usage and spend in one view.
- **Per-product Knowledge Graphs:** entities and relations from a product's facts (people, places, organisations;
  services and dependencies).
