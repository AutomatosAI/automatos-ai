# PRD-257: Enterprise: agents that watch the estate

**Status:** DRAFT for owner review · **Owner:** Gerard · **Written:** 10 Oct 2026
**Type:** Net-new feature, assembled mostly from what exists. The new parts are one shared brain per workspace outside Missions, access to the bank's systems, and the estate dashboard. Agents, heartbeats, the board, approvals, the field and OTel already exist and are reused.

**Owner's words (10 Oct 2026):** "The OS is its own engine now, Studio is how you manage via tasks and boards… let's build enterprise.automatos.app, an observability dashboard. It's in kubes and agents all skilled in DB, Kong, Finacle etc… they all share the same neural field, a shared brain, and like I do in Missions semantic search, but we have visuals that show the system from an agent's view… we are hoping to PoC for my bank… agents running in pods with heartbeats to monitor logs, systems, each agent has a persona and skill like Kong or Finacle expert, that's their role… db goes down, they search the brain for network issues or other possible solutions, they can then raise tickets on the board for other agents to investigate… the enterprise dashboard is the visual of all the agents in the clusters, VMs, and you can zone into an agent and talk with it."

**Decisions taken (10 Oct 2026):**

| | Decision |
|---|---|
| D1 Where agents run | **Registered estate runners** (revised 10 Oct 2026; was "in the cluster is enough"). The core runs from the existing Helm chart. Agents' tools run in **estate runners**: a pod per namespace or a service per VM, registered in Studio, holding scoped permissions, revocable, and authenticated by OIDC workload identity. Owner: "agents are registered on the platform via the Studio so we can control access and permissions, OIDC etc." |
| D5 Identity | **OIDC.** Runners authenticate with the bank's workload identity (a k8s service-account token or the bank's OIDC issuer), checked against a configured JWKS. People sign in through the bank's OIDC provider. |
| D2 Reaching bank systems | **Per-workspace monitoring endpoints.** The existing Loki/Prometheus tools stop being super-admin-only and global, and new read tools (Kong admin, k8s) use the 3-file pattern. This is a sanctioned exception to "integrations go through Composio", because Composio can't reach an internal bank network. The Composio deny list is untouched. |
| D3 Autonomy | **Act with approval.** Agents observe, write to the brain, search it, raise tickets and propose remediation. Any change to a bank system runs only after a human grants it, through the existing `approval_grants` (PRD-224–229). |
| D4 Model | **Bank-hosted.** Completions *and embeddings* stay inside the bank's network. No call to OpenRouter or a vendor API from the PoC install. |

---

## 1. The idea in one paragraph

Each system the bank runs (Kong, Finacle, the databases, the network, the k8s clusters and VMs) gets an Automatos agent whose persona and skill are that system. On every heartbeat the agent reads its system's logs, metrics and state, and writes what it notices into the workspace's **shared brain**: the PRD-108 memory field, lifted out of a single Mission. When something breaks, the agent searches the brain semantically ("network latency in the last hour", "connection pool exhaustion"), so it uses what the other agents noticed rather than only its own view. When the cause is outside its system, it raises a ticket on the board for the agent that owns that system. When it knows the fix, it proposes a remediation that waits for human approval. **enterprise.automatos.app** shows all of this: the estate as zones (clusters, VMs, systems), the agents in them, their health, the tickets passing between them and the brain they share. You can zoom into any agent and talk to it.

## 2. What exists today (verified 10 Oct 2026)

| Need | Today | Verdict |
|---|---|---|
| Persona and skill per agent | `personas`, `skills`, `agent_skills`, `agents.persona_id`/`custom_persona_prompt` (`core/models/core.py:247, 290`); seed pattern in `core/seeds/seed_shopify_agents.py`; `devops-sre` persona and a monitoring skill already seeded | **Reuse.** Kong, Finacle, DB, network and k8s agents are seed data. |
| Heartbeats | `agent.configuration["heartbeat"]` (interval, prompt, active hours, `auto_act`); `services/heartbeat_service.py` `_agent_tick` runs `ContextMode.HEARTBEAT_AGENT` with the full toolset and the field_memory section; results in `heartbeat_results` | **Reuse.** |
| Agent → agent tickets | `platform_create_task` with `assigned_agent_name`, `platform_assign_task` (`modules/tools/discovery/actions_board_tasks.py:58, 319`); `services/board_dispatcher.py` claims and runs them | **Reuse.** |
| Semantic search of a shared brain | `platform_field_query` falls back to workspace scope without a Mission (`handlers_field.py:74-79`); `query_workspace` filters on `workspace_id` (`modules/context/adapters/vector_field.py:416`); score = similarity × stability × recency | **Reuse.** |
| Writing to the shared brain | `platform_field_inject` refuses outside a Mission (`handlers_field.py:135-139`) and passes no provenance, so points carry no `workspace_id` (`:147`) | **Extend (W1).** |
| Approval before acting | `approval_grants` (`subject_type=tool_call`, `risk_tier`, `agent_id`), the policy gate (`modules/policy/gate.py`), Command Center governance | **Reuse.** Remediation tools declare a risk tier that requires a grant. |
| Reading logs and metrics | `platform_query_loki_logs`, `platform_query_prometheus`, `platform_get_alerts` (`actions_monitoring.py`): `super_admin_only=True`, one global `LOKI_URL`/`PROMETHEUS_URL` (`config.py:850-857`) | **Extend (W2b).** |
| Kong, k8s, Finacle | Nothing. `workspace_exec` blocks `kubectl` on purpose; `ssh_execute` uses `AutoAddPolicy`, which a bank won't accept | **Build (W2b).** |
| Where an agent sits in the estate | Fleet state (`services/fleet_state.py`) has runtime, current work, queue, cost; no cluster, VM or system | **Extend (W3).** |
| Chat with one agent | `/chat` agent picker; backend honours `agent_id` (`api/chat.py:139`); no `?agent=` deep link (`app/chat/page.tsx:68-76`) | **Extend (W4).** |
| Visualisation | `react-force-graph-2d/3d`, `reactflow`, three; `mission-field-viz.tsx`; `org-chart-canvas.tsx` | **Reuse libraries; build the estate view (W4).** |
| Registering a machine with scoped, revocable access | Three patterns, none complete. **CLI host pairing** (`core/models/cli_hosts.py`, `api/cli_hosts.py`): pairing code → one-time token → heartbeat → claim → result, per-ticket tokens (`mint_session_token`); but no scopes, `revoke_host` has no route or UI, and it is local-edition only (`config.py:2206`). **SDK API keys** (`core/models/sdk_api_keys.py`, widgets): scopes, allowed IPs/domains, expiry, revoke in Studio (`ApiKeyManager.tsx`); but exchanged JWTs outlive a revoked key (`api/widgets/auth.py:138-171`). **Shopify**: one shared secret from an external app server, no per-install credential | **Extend (W2a).** CLI host lifecycle plus SDK-key scopes. |
| OIDC / SSO | None. Users sign in with Clerk JWTs on the hosted edition (`core/auth/clerk.py`, JWKS); the local edition has no accounts. No OIDC, SAML or workload identity anywhere | **Build (W2a, W5).** |
| Running in k8s | Helm chart `charts/automatos/` (api, worker, frontend, migrate); `QDRANT_URL` is a value, Qdrant itself isn't shipped | **Extend (W5).** |
| Traces | PRD-256 OTel, complete; GenAI spans carry `agent_id`; collector config in `deploy/otel/` | **Reuse.** Exported to the bank's collector. |
| Bank-hosted model | Provider registry (`core/llm/providers.py`) has fixed OpenAI-compatible specs (openrouter, nvidia, deepseek); embeddings go via OpenRouter | **Extend (W5).** |

## 3. Non-negotiables

1. **Nothing changes a bank system without a grant.** Every write or remediation tool declares a risk tier that requires an `approval_grants` record. Read tools don't. There is no "auto-act" path for remediation, whatever `heartbeat.auto_act` says.
2. **No data leaves the bank.** With the PoC install, completions, embeddings, traces and logs go only to endpoints inside the bank's network. There are no outbound vendor calls; a test asserts which hosts the configured install talks to.
3. **Credentials stay in the existing credential store.** Monitoring-endpoint secrets are encrypted per workspace using the store integrations already use, never in `agent.configuration`, never in a prompt, never on a span.
4. **Tenant isolation.** The brain, endpoints, estate zones and tools are all scoped to the caller's workspace.
5. **Both editions keep working.** Everything here is additive and inert without endpoints configured. The local edition without Qdrant degrades to "brain off", as the field does today.
6. House rules: settings through `config.py`; no new table where an existing one fits; delete what a wave replaces (the global monitoring URLs, once per-workspace endpoints exist); new routes in the route manifest; new settings in `config-surface.json`.

## 4. Waves (one PR or a small series each, each with tests)

### W1: The shared brain outside Missions

- One stable field per workspace (the **estate field**): a deterministic `field_id` derived from the workspace, created on first write. Same Qdrant collection (`field_memory`), same scoring and decay.
- `platform_field_inject` writes to the estate field when no Mission is running, instead of refusing. Every point it writes carries `workspace_id`, `agent_id` and (new) `source: heartbeat | ticket | chat | mission`, so the dashboard and searches can filter by who noticed what.
- `platform_field_query` searches the estate field and archived Mission fields together (it already reads workspace scope; this makes it intentional and tested).
- Read-only REST for the dashboard: `GET /api/estate/brain?query=&agent_id=&since=` (workspace-scoped, authenticated, in the route manifest).
- **Proof:** a heartbeat-mode agent injects a finding with no Mission; a second agent's query returns it with its provenance; a different workspace's query does not.

### W2a: Estate runners, registered in Studio

- **One registry, not a second one.** `cli_hosts` generalises into a runner registry with a `kind` (`cli` for today's session hosts, `estate` for the new runners); a rename to `runners` happens in the same PR if the name gets in the way, with no compatibility alias. The CLI host keeps working unchanged.
- **Register in Studio** (Settings → Runners, extending today's Session mode pairing UI): name the runner, choose its zone, the agents it may run, its tools and its scopes, then get a pairing code. The runner exchanges the code once for its credential.
- **Scopes:** a `runner:*` vocabulary in `core/auth/scopes.py` (`runner:heartbeat`, `runner:claim`, `runner:brain.write`, `runner:tickets.create`, `runner:tools.<tool>`), plus `allowed_agent_ids`, `allowed_ips` and `expires_at` from the SDK-key model. Every runner route checks scope, not only identity.
- **OIDC workload identity:** a runner may skip the pairing secret and present its k8s service-account token, or a token from the bank's OIDC issuer. The token is verified against a configured JWKS (the `PyJWKClient` pattern in `core/auth/clerk.py`) and mapped to the registered runner by issuer and subject. Pairing codes stay for VMs without a workload identity.
- **Short-lived job tokens:** each claimed job gets a token scoped to that job (the `mint_session_token` model). A revoked runner's jobs stop at once, avoiding the widget-JWT problem.
- **Revoke, rotate, audit:** routes and Studio buttons for revoke and rotate (today's `revoke_host` is never called); pair, revoke, rotate and scope changes are audited.
- **The runner:** a small process packaged as a container image and a systemd unit. It heartbeats (the fleet and dashboard health), claims its agents' tool calls and heartbeat jobs, runs the W2b tools locally against its zone's systems, and sends results back. It never holds an LLM key: reasoning stays in the core, tools run at the edge.
- **Editions:** the estate runner is allowed in both editions behind `ESTATE_ENABLED`; session-mode CLI runtime stays local-only as today.
- **Proof:** pair by code and by OIDC token (a test JWKS); a call outside the runner's scopes is refused; revoking kills an in-flight job token; a runner can't claim another workspace's work.

### W2b: Reaching the bank's systems

- **Per-workspace monitoring endpoints.** A workspace configures named endpoints (kind: `loki | prometheus | kong_admin | k8s_api | http_health`, base URL, an auth reference into the credential store, TLS CA bundle). Storage reuses an existing table if one fits (the credential store's own table is the first candidate); a new table only if none does, with the reason in the PR.
- `platform_query_loki_logs`, `platform_query_prometheus`, `platform_get_alerts`: they lose `super_admin_only` and read the workspace's endpoint. The global `LOKI_URL`/`PROMETHEUS_URL`/Grafana settings that served the platform's own infra are **deleted** along with their callers' fallback, or kept only as the platform workspace's endpoint, with no second code path.
- New **read** tools, 3-file pattern: `platform_kong_status` (services, routes, upstream health, recent errors from the Kong admin API), `platform_k8s_status` (pods, events, restarts, node pressure for a namespace through the k8s API with a read-only service account), `platform_http_health` (probe an internal health URL from the configured list only).
- New **remediation** tools, risk tier requiring a grant: `platform_k8s_restart_workload` (rollout restart of a named deployment), `platform_kong_set_upstream_target` (enable or disable a target). The list is short on purpose and every entry is chosen with the bank (§6 Q3).
- TLS verification on, with the bank's CA bundle. No `AutoAddPolicy`, and no generic shell.
- **Finacle:** the tool depends on what the bank exposes (§6 Q2). The wave delivers it once that's known; until then Finacle is watched through its logs (Loki) and its database (`platform_query_data`).
- **Proof:** each tool against a fake server (recorded responses), workspace isolation, a remediation call creating a pending grant and running only after the grant.

### W3: The estate model and the agents

- **Where an agent sits:** `agent.configuration["estate"] = {zone, zone_kind: cluster | vm | system, systems: [...], endpoints: [...]}`. A zone is a label the workspace defines, not a new entity, unless the dashboard needs zone metadata beyond a name (then the PR justifies a table).
- Fleet state gains the estate fields and a health roll-up per agent: last heartbeat result, open tickets, pending grants.
- Seeds (insert-if-absent, workspace-installable as a pack): personas and skills for **Kong**, **Finacle**, **Database (Postgres/Oracle)**, **Network**, **Kubernetes platform**, plus an **Incident lead** that reads the brain, groups related findings and keeps the board tidy. Each skill names its tools, what "healthy" looks like, what goes into the brain, and when to raise a ticket for whom.
- Heartbeat prompts follow one contract: *read your system → write notable findings to the brain → when something is wrong, search the brain first → raise a ticket for the owning agent, or propose a remediation (a pending grant) → report.*
- **Proof, end to end** (scenario test with fake endpoints): the DB agent sees connection timeouts → searches the brain → finds the network agent's earlier "packet loss on node-3" finding → raises a ticket for the network agent → the network agent investigates and proposes `platform_k8s_restart_workload` → a grant is pending → granting it runs the call → the outcome lands in the brain.

### W4: enterprise.automatos.app, the dashboard

- **Estate view:** zones as regions (clusters, VMs, systems), agents placed in them, coloured by health; ticket hand-offs drawn as edges between agents (recent window); pending grants badged. Built on `reactflow` or `react-force-graph-2d`, both already dependencies; no new chart library.
- **Agent panel** (zoom into an agent): persona and skill, its systems, the latest heartbeat results, what it wrote to the brain, its open tickets and pending grants, and a **Talk to this agent** button.
- **Chat deep link:** `/chat?agent=<id>` opens a chat pinned to that agent (extends `app/chat/page.tsx`); the panel can also host the chat inline.
- **Brain view:** the estate field, reusing `mission-field-viz` (3D with the 2D fallback), filterable by agent, system and time; search box over `GET /api/estate/brain`.
- Grants are approved from the dashboard by calling the existing governance endpoints, not new ones.
- **Proof:** vitest for the view models and the deep link; an e2e that loads the estate view from seeded fixtures and opens an agent chat.

### W5: The bank install

- **Bank-hosted model:** a `self_hosted` provider spec (OpenAI-compatible adapter, PRD-236 pattern) whose base URL and key come from operator settings, for both completions and **embeddings**; the field and semantic routing use it when set. Bank-hosted inference servers (vLLM, TGI, Ollama) all speak the OpenAI API.
- **Helm values for the PoC:** Qdrant (subchart or the bank's own instance), the `self_hosted` model settings, OTel export to the bank's collector, the bank's CA bundle, ingress host `enterprise.<bank domain>` (and `enterprise.automatos.app` for our demo install), and egress denied by default.
- **Outbound lockdown:** with the PoC values, Composio, OpenRouter, web search and the other outbound integrations are off; a test runs the configured app and asserts that no outbound host outside the configured endpoints is contacted.
- A demo runbook: install, configure endpoints, install the agent pack, run the W3 scenario live.

## 5. Settings (new; final names in each wave's PR and `config-surface.json`)

| Setting | Default | |
|---|---|---|
| `ESTATE_ENABLED` | `false` | Turns on the estate routes, tools and dashboard. |
| `SELF_HOSTED_LLM_BASE_URL` / `SELF_HOSTED_LLM_API_KEY` | empty | The bank's OpenAI-compatible model server. |
| `SELF_HOSTED_EMBEDDING_BASE_URL` / `SELF_HOSTED_EMBEDDING_MODEL` | empty | Embeddings for the field and routing, inside the bank. |
| `ESTATE_RUNNER_OIDC_ISSUERS` | empty | Trusted issuers and their JWKS URLs for runner workload identity. |
| `OUTBOUND_LOCKDOWN` | `false` | Refuses every outbound integration that isn't a configured endpoint. |

## 6. Open questions for Gerard

1. **What is enterprise.automatos.app, in code?** I recommend a route group (`/enterprise`) in the existing frontend, behind `ESTATE_ENABLED`, served on that host by ingress. One codebase, no third app. The alternative is a separate Next.js app.
2. **How does the bank expose Finacle?** REST/SOAP APIs, logs only, its database, or JMX/health endpoints? That decides the Finacle tool in W2b.
3. **Which remediations are allowed in the PoC?** I've proposed two (restart a workload, toggle a Kong upstream target). The bank's change-management rules decide the final list.
4. **User sign-in at the bank (D5):** OIDC login is new auth work. Is it a third sign-in path beside Clerk, in the edition the bank runs, or do we replace Clerk with generic OIDC on the hosted edition too (Clerk itself speaks OIDC)? AGENTS.md asks for an issue before auth changes; this PRD is that record once you decide.
7. **Copilot identity at the bank:** a named engineer's GitHub account or a bank service account, and GitHub Enterprise Server or Cloud?
5. **New dependency for k8s:** the official `kubernetes` Python client, or plain `httpx` against the k8s API with the service-account token. I recommend `httpx` (no new dependency); the client is the alternative if you'd rather.
6. **"Enterprise" as a plan:** `plan_tiers.py` already has a coming-soon `enterprise` tier. Should the estate feature key off that tier on the hosted edition, or stay a deployment flag only?

## 7. Granularity of runners

A runner serves a zone (a namespace or a VM) and may host several agents' tools. One runner per agent is a registration choice, not new code.
