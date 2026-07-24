# Composio Integration — module dossier

> Phase-2 component deep-review. Scope: the external-action fabric — OAuth connect,
> apps/actions caches, action-metadata classification (destructive gate), per-agent
> scoped execution, trigger-webhook ingress, the meta-tool router, and the email/widget
> proxy. Judged against the North Star: does it make Auto and the agents *more
> autonomously capable and higher-quality for real clients?* No security-hardening lens
> here (that is the separate Opus pass) — but availability/reliability and the
> correctness of the guardrails that autonomy depends on are in scope.
>
> Grounded in `orchestrator/**` @ main `e040d9b53`, the banked W1 real-data evidence, and
> cited competitor sources. Section F is intentionally omitted.

---

## A. What it is

Composio is the platform's **adopted external-tool vendor** — a hosted catalog of ~850+
SaaS apps (Slack, Gmail, GitHub, Jira, Shopify, LinkedIn, …) with managed OAuth and a
single execute API. This module is the ~5,300-line **in-house wrapper** around that
vendor: it maps each Automatos workspace to a Composio "entity", runs the hosted-auth
connect/callback flow, mirrors the app/action catalog into local Postgres caches, resolves
Composio actions into per-action OpenAI function-calling tools for the LLM, executes them
with agent-scope validation, ingests Composio trigger webhooks, and proxies a Gmail email
panel. It is the **largest single external boundary in the platform** and the mechanism by
which an agent "does something in the real world" for a client. Adopting Composio-the-vendor
was the right call (§E); this dossier is about the wrapper and the gaps in it.

Key files: `orchestrator/api/composio.py` (1,035 L — HTTP surface + webhook),
`orchestrator/core/composio/client.py` (1,425 L — SDK wrapper),
`orchestrator/core/composio/tool_executor.py` (972 L — scoped execution),
`orchestrator/services/metadata_sync_service.py` (594 L — catalog sync),
`orchestrator/modules/tools/sync/composio_action_sync.py` (415 L — classifier sync),
`orchestrator/services/composio_sync_scheduler.py`, `orchestrator/modules/tools/composio_tool_router.py`,
`orchestrator/modules/tools/execution/exec_composio.py`,
`orchestrator/modules/tools/services/composio_tool_service.py`, `orchestrator/api/widget_email.py`.

---

## B. What it does — real implementation and data path

**Connect / OAuth.** `POST /api/composio/connect/{app}` (`composio.py:300`) resolves an
enabled Composio auth-config for the app slug and returns a hosted-auth redirect URL
(`client.py:299-365`). The wrapper is genuinely careful here: it picks the auth *scheme*
(OAUTH2 / API_KEY / BASIC), pins the chosen `auth_config_id` + scheme onto the workspace
connection row so **reconnects are sticky** (`composio.py:339-374`), short-circuits
NO_AUTH apps to immediate-active (`composio.py:319-330`), and the callback upserts the
connection resolving the `connection_id` three ways — from the redirect, from the Composio
API, then by scanning entity connections (`composio.py:405-445`). Connections live in
`composio_connections` keyed by a per-workspace `ComposioEntity`.

**Catalog mirror.** `MetadataSyncService.run_full_sync` (`metadata_sync_service.py:42`)
pulls **all** apps (`client.get_available_apps`, cursor-paginated, `client.py:562-683`) and
**all** actions in bulk via the Composio REST `/tools` endpoint
(`get_all_actions_bulk`, `client.py:859-1001`), groups by app, upserts into
`composio_apps_cache` / `composio_actions_cache`, deletes orphans so the DB matches
Composio exactly, and backfills parameter schemas per-app because the v3 bulk API omits
them (`:453-548`). This `ComposioActionCache` is the catalog the executor validates against.

**Per-action tool resolution.** For a chat turn or recipe step, `ComposioToolService.get_tools_for_step`
(`composio_tool_service.py:97`) resolves the agent's *allowed* apps (assigned ∩ connected,
with auto-inherit of workspace-connected apps when an agent has no explicit assignment,
`:294-338`), then produces per-action OpenAI tool schemas by one of three strategies:
(1) explicit action names parsed from the prompt → exact schema lookup; (2) `tool_hints` →
scoped SDK semantic search; (3) broadened SDK search (`:141-247`). Each Composio action
becomes its own top-level function (`GMAIL_SEND_EMAIL(...)`) rather than a generic
`composio_execute(action=…)`.

**Execution (two live paths).**
- *Meta-tool path:* `composio_execute` / per-action tools dispatched through the unified
  executor → `exec_composio.execute_composio_execute` (`exec_composio.py:134`) →
  `ComposioToolExecutor.execute` (`tool_executor.py:344`). This path is the substantial one:
  it validates **assigned? → connected? → mapped-in-cache?** (`:419-648`), fuzzy auto-maps a
  mis-named action when similarity > 0.7 while excluding destructive verbs (`:585-596`),
  resolves workspace file paths / URLs to Composio `FileUploadable` uploads for media actions
  (`:692-697`, `resolve_file_uploads:124`), off-loads the blocking SDK call to a worker thread
  so one slow tool can't stall the event loop (`:736`, PRD-161 S4), intercepts LinkedIn image
  posts to a direct-API workaround (`:704-729`), and extracts entities into Redis for
  follow-up suggestions (`:902-972`).
- *Chat per-action shortcut:* when Composio tools were injected, the chat loop routes the
  call **around** the unified executor to `_execute_composio_action`
  (`consumers/chatbot/service.py:1328-1335`) → `ComposioToolService.execute_action`
  (`composio_tool_service.py:257`) → raw `client.execute_action`. See §C — this is the
  designed-preferred path and it skips most of the protections above.

**Trigger ingress.** `POST /api/composio/webhook` (`composio.py:597`) normalises V2/V3
payloads, resolves the entity from `connected_account_id`, and routes by trigger-name
prefix. Only `JIRA_*` triggers have an ingestor; everything else is written to
`unrouted_events` and dropped (`:724-752`). Routed events dispatch an agent or a
recipe/workflow as a background task (`:772-786`).

**Metadata classification (destructive gate feeder).** A *separate* sync,
`ComposioActionSyncService` (`modules/tools/sync/composio_action_sync.py`), is supposed to
classify each action's capabilities / `destructive` flag into `composio_action_metadata`,
which backs the fail-closed destructive-action gate (F018). It is wired to a daily 04:00
cron (`composio_sync_scheduler.py`, `main.py:417-423`, default `COMPOSIO_SYNC_ENABLED=true`).
See §C — as coded it never writes a row.

**Real data path, as observed (W1).** `tool_execution_logs` = 2,341 rows, **100% synthetic**,
frozen 2026-05-05 (`data/tool-telemetry.md`). The only Composio rows are seeded fixtures
(`COMPOSIO / composio_execute`, "send an email to j***@acme.com"); the learned operating
graph's cluster 14 ("send an email…") likewise sits only in the all-zeros synthetic
workspace (`data/operating-graph-edges.md`). **No organically-recorded Composio execution
exists in the store** across two months of live operation.

---

## C. Honest quality — maturity **3 / 5**

The core is real, deep, and pilot-hardened — the Shopify vertical depends on this exact
code and the connect/execute happy path clearly works (scheme-sticky reconnects, file
uploads, thread off-loading, auto-mapping, the LinkedIn workaround are all evidence of
production scar tissue, not scaffolding). That earns it well above a 2. It does **not**
reach 4, because several *load-bearing* pieces that autonomy-with-guardrails depends on are
broken, bypassed, or absent. Concrete, evidence-grounded defects, worst first:

**C.1 — The trigger webhook accepts forged events (F028, NOT DONE — the sharpest edge).**
`composio.py:629-632` is byte-identical to the July finding: a V3 signature mismatch logs
*"V3 webhook signature mismatch — allowing through for debugging"* and falls through; the
verification-exception branch does the same. Worse, verification only runs when a signature
header is *present* (`:618,633`) — omit both headers and the unauthenticated `POST /webhook`
skips verification entirely. This is the platform's **autonomous inbound leg**; on it, a
forged Jira/agentic event dispatches a real agent or recipe (`:772-786`). Judged by the
North Star this is the single most important defect here: the ingress path that is supposed
to let Auto act on real-world events cannot trust what it receives. Industry-standard for
security-sensitive webhooks is HMAC signature verification with reject-on-mismatch
([codehooks](https://codehooks.io/blog/secure-zapier-make-n8n-webhooks-signature-verification),
[Stripe pattern](https://dev.to/whoffagents/stripe-webhook-security-signature-verification-idempotency-and-local-testing-1lk3)).

**C.2 — The destructive-gate metadata sync is a no-op; the gate runs blind.** The daily
classifier sync (built as the F018 fix) cannot populate its table as coded:
`ComposioActionSyncService._get_all_enabled_apps()` returns a **hardcoded 8-app placeholder**
(`"slack","github","gmail","google_calendar","notion","jira","linear","discord"`) with the
comment *"Placeholder - return common apps for now"* (`composio_action_sync.py:330-340`),
and `_fetch_app_actions` does `raw_actions = await self.composio_client.get_app_actions(app_id)`
(`:207`) — but `get_app_actions` is a **synchronous** method (`client.py:775`, `def`, returns
a list). `await`-ing a list raises `TypeError` for every app, so each is caught and appended
as an error `SyncResult`: **classified = 0, errors = 8, every run.** The sole writer of
`composio_action_metadata` is this path (`composio_action_sync.py:300`; grep-confirmed no
other constructor), and the manual `api/tools.py` sync writes the *different* table
(`composio_actions_cache`). So the destructive-gate metadata table has **no working writer**.
Net effect: F018's gate, marked FIXED in the residual map, in practice never sees a
per-action `destructive` classification and operates permanently in its fail-closed
cold-start mode — an 8-keyword *intent-text* heuristic (`taxonomy.intent_is_destructive`).
The residual map noted the cold-start caveat but not that the feeder is dead. A destructive
*action* under neutral wording ("issue a refund", "create a 10% discount") is exactly what
slips through, forever, not just until the first sync. *(Prod contents of
`composio_action_metadata` were not sampleable this run — see C.7 — so this is a
code-path finding; the code path is unambiguous.)*

**C.3 — The preferred chat path skips validation, the gate, telemetry, and file uploads.**
When Composio tools are injected, the chat tool-callback treats any call whose name is in
the injected `action_set` or starts with an allowed app prefix as a Composio action and
routes it to `_execute_composio_action` (`service.py:1322-1335`), which calls
`ComposioToolService.execute_action` — a **thin raw wrapper** over `client.execute_action`
with no agent per-action access check, no destructive gate, no telemetry hook, and no
file-upload resolution (`composio_tool_service.py:257-274`). All four of those protections
live only on the *meta-tool* path through `ComposioToolExecutor.execute` /
`tool_router.execute_and_format`. Because the hint service is *designed* to inject
per-action tools (so the LLM calls `SLACK_SEND_MESSAGE` directly), the shortcut is the
**common** path, not an edge case. Consequences that show up in the real data: (a) the
per-action `AgentAppFeature` enable/disable UI has no effect on chat; (b) the destructive
gate is bypassed for the path it most needs to cover; (c) **no `ToolExecutionLog` row is
written** — which is a direct contributor to the W1 finding that `tool_execution_logs` has
zero organic rows and the learned operating graph has never formed an edge for any of the 21
real workspaces (`data/tool-telemetry.md`, `data/operating-graph-edges.md`). The learning
loop the platform built (PRD-177/W7) is starved partly because its busiest producer doesn't
report.

**C.4 — Per-action Composio failures are reported as success.** The shortcut always returns
`{"success": True, …}` (`service.py:1335`) regardless of outcome; `_execute_composio_action`
bakes the error into the text `llm_context` ("Error executing …") but the tool-loop and the
running/error indicator see success. So a failed send-email in chat renders as a completed
tool call with no error chip — quietly undoing part of the F037 honesty fix (which is
correctly done for the `composio_execute` meta-tool path, `message.tsx:289-319`). For a
client-facing agent, silently-successful failures are a quality defect, not a cosmetic one.

**C.5 — The inbound autonomy leg barely exists (F066, NOT DONE).** Only `JIRA_*` triggers
route; Gmail and everything else dead-letter to `unrouted_events` (`composio.py:724-746`).
There is no inbound email channel, so the archetypal "customer emails → agent handles
refund" journey has no autonomous path — email only works pasted into chat or read via the
Gmail *panel*. For a platform whose North Star is autonomous client-facing operation, the
trigger fabric is wired for exactly one app family.

**C.6 — Architectural sprawl around a simple job.** There are **three** overlapping sync
services writing **two** metadata tables: `MetadataSyncService` → `composio_actions_cache`
(good REST-bulk path, manual/API-triggered only — not on any scheduler);
`ComposioActionSyncService` → `composio_action_metadata` (the broken daily classifier);
plus `composio_api_service` referenced as a third. The meta-tool `ComposioToolRouter`
(`composio_tool_router.py`) is wired end-to-end (`unified_executor.py:667` →
`exec_composio.py:236`) but its `search_tools` still calls `session.search_tools(...)` under
a *"This is a placeholder - adjust based on actual API"* comment (`:90-92`) — i.e. the
learned/hosted tool-router search leg is unproven. And `services/tool_manifest_service.py`
(141 L, S3 snapshotting) has **zero importers** (grep-confirmed) — a dead module squatting on
the "manifest" name. This is the reuse-over-build ground rule inverted: the same job coded
three times, one of them broken, one dead.

**C.7 — Real Composio state could not be inspected (a finding in itself).** W1 could reach
prod Postgres but did **not** sample `composio_connections`, `composio_actions_cache`,
`composio_action_metadata`, `agent_app_assignments`, or `trigger_subscriptions` — none
appear in `data/census.md`. So the count of real workspace connections, whether the action
cache is actually populated, and whether any destructive-metadata rows exist are **unknown
from banked evidence**. Combined with the 100%-synthetic telemetry, the honest position is:
*the execution machinery is well-built and the pilot exercises it, but there is no captured
signal proving what it does at runtime for the 21 real workspaces.* That gap is itself a
quality problem (§G).

**What's genuinely good (kept honest):** the auth-config scheme resolution + sticky
reconnect logic (`client.py:166-297`) is careful and solves a real Shopify pain; the
file-upload resolution shared across executor and recipe paths is correct and non-trivial
(`tool_executor.py:124-219`); the thread off-load (`:736`) is the right fix for a blocking
SDK; the auto-mapping suggestion engine turns a dead-end "action not mapped" into a ranked,
destructive-filtered suggestion list (`:507-637`); and the per-action tool presentation
(one function per action, params baked in) is the correct shape for LLM tool-calling.

---

## D. Competitive teardown

The relevant comparison is **not** "build our own tool platform" — it's how much of the
integration/auth/routing job to hand to a vendor vs. keep in the wrapper.

**Composio (the upstream vendor we already use).** Since this wrapper was written, Composio
shipped **Tool Router** as a hosted single MCP endpoint with dynamic tool discovery across
500–1000+ apps, managed OAuth with **automatic token refresh/rotation**, per-user /
per-environment scoped credentials, RBAC, sandboxed execution, and MCP API-key enforcement
(default since March 2026); it integrates directly with the Claude Agent SDK, OpenAI,
LangChain, etc. ([docs.composio.dev/tool-router](https://docs.composio.dev/tool-router/overview),
[composio.dev/toolkits/composio](https://composio.dev/toolkits/composio)). **Where it beats
the wrapper:** the bespoke `composio_actions_cache` mirror + orphan-delete + parameter
backfill, the alphabetical-search workaround (`client.py:102-106`, the wrapper's own comment
that the SDK's semantic search "returns alphabetical, not semantic"), and the placeholder
classifier are all jobs Tool Router / the hosted MCP endpoint now largely subsume. **Caveat
that cuts the other way:** Composio disclosed a **May 2026 security incident** — exfiltration
of thousands of customer API keys and GitHub OAuth tokens, with a multi-day partial shutdown
([V12 Labs](https://www.v12labs.io/blog/2026-06-16-ai-agent-tool-authentication-composio-arcade)).
That is a live **availability + supply-chain dependency risk** for a platform that stores its
Composio key and leans on Composio's managed credential store; it argues for keeping the
execution abstraction thin-but-swappable, not for a deeper lock-in without a fallback.

**Arcade.dev.** Auth-first agent-tool runtime: strong **per-user OAuth delegation**,
governance/permission boundaries, SIEM-exportable audit, and **self-hosted / air-gapped**
deployment — but ~112 first-party integrations, roughly 1/10th Composio's catalog, and no
public CVEs ([Merge](https://www.merge.dev/blog/composio-vs-arcade),
[Scalekit](https://www.scalekit.com/blog/arcade-alternatives)). **Where it beats us:**
exactly the governance/isolation surface this wrapper is weakest on (per-action scope
enforced everywhere, delegated auth, audit). **Where it loses:** breadth — and breadth is
what Auto needs to be broadly autonomous for varied clients.

**Zapier AI Actions / n8n (MCP-server automation layer).** Both now expose workflows as MCP
servers an agent can call ([gamut Zapier-MCP guide](https://www.gamut.so/blog/zapier-mcp-setup-guide),
[n8n](https://n8n.io/vs/zapier/)). Zapier's breadth (7,000+ apps) exceeds Composio's, but it
is a *workflow* abstraction (the agent calls a pre-built Zap), not fine-grained per-action
function-calling with managed per-user auth — a worse fit for an agent that must choose and
parameterise the exact action. Notably, the same webhook-security guidance that Automatos
fails on (verify HMAC, reject on mismatch for business/financial webhooks) is the shared
baseline across this whole class.

**Anthropic MCP connectors / remote MCP servers.** The MCP-native alternative: connect
agents to remote OAuth MCP servers directly. Composio's own Tool Router *is* delivered as an
MCP endpoint, so "adopt MCP" and "adopt Composio Tool Router" converge — the wrapper is on
the wrong side of that convergence, re-implementing discovery/auth the protocol now standardises.

**Where Automatos actually stands:** the catalog breadth and managed-auth are strong
*because they are Composio's*. The wrapper adds genuine value in scoped execution, file
handling, and pilot-specific auth ergonomics — but it also re-implements (three times, one
broken) sync/discovery that the vendor now ships, and it is materially **behind Arcade on
governance/scope enforcement** and **behind the whole field on webhook trust**.

---

## E. Build / extend / adopt / replace — verdict

**Keep the vendor (adopt Composio — already correct); EXTEND-and-consolidate the wrapper.**

- **Do not replace Composio.** Nothing external gives 850+ apps + managed OAuth at
  $29–$229/mo (§H). Arcade is auth-stronger but ~1/10th the catalog; breadth is load-bearing
  for autonomy. Replacing the vendor fails the North Star.
- **Adopt Composio Tool Router / the hosted MCP endpoint *more deeply*** to retire wrapper
  code the vendor now owns: the bespoke `composio_actions_cache` full-mirror + orphan-delete
  + parameter-backfill, the alphabetical-search workaround, and the placeholder classifier.
  Integration shape: route agent-scoped sessions through Tool Router's dynamic discovery
  instead of the local cache for *selection*, keeping a thin local cache only for the
  execution-validation contract and offline/degraded operation. Rough cost: a focused
  refactor, not a rebuild; it *removes* ~1,000+ lines net.
- **Consolidate to one sync service and one table.** Collapse `MetadataSyncService` +
  `ComposioActionSyncService` (+ the `composio_api_service` leg) into a single sync that
  fills one metadata table and actually classifies connected apps; delete the broken daily
  classifier and its 8-app placeholder. **Delete `services/tool_manifest_service.py`**
  (0 importers) and either wire or delete the placeholder `ComposioToolRouter.search_tools`.
- **Build (small, in-house, non-negotiable):** the three correctness/guardrail fixes that
  no vendor will do for us — reject-on-mismatch webhook verification (C.1), one canonical
  execution chokepoint so the chat per-action path inherits scope + gate + telemetry (C.3),
  and honest success/failure propagation (C.4).
- **Borrow from Arcade's model** for the governance gap: enforce per-action scope on *every*
  execution path, not just the meta-tool one. That's a wrapper change, not a vendor switch.

Net: the vendor decision was right and stays; the ~5,300-line wrapper should get **smaller
and more correct**, not bigger.

---

## G. Quality metric

Today there is **no quality number** for this module — the defining problem. Proposed,
cheap-first:

1. **Execution success rate & latency by app/action** from `tool_execution_logs` — *once
   C.3 is fixed so the chat path actually writes rows.* Today: unmeasurable (0 organic rows;
   100% synthetic per `data/tool-telemetry.md`). Target: p50/p95 latency and success% per
   top-20 action, tracked weekly.
2. **Destructive-gate coverage** = fraction of executed destructive actions that hit a
   populated `metadata.destructive` classification (not just the keyword heuristic). Today:
   effectively **0%** (C.2 — table has no working writer). Target: 100% of connected-app
   actions classified, sync verified non-empty.
3. **Webhook authenticity** = share of inbound trigger events that pass signature
   verification before dispatch. Today: **0** enforced (C.1). Target: 100% verified-or-rejected.
4. **Catalog freshness** = age of the newest `composio_actions_cache.last_synced_at` and %
   actions with non-empty parameter schemas. Measurable now via the sync-job table; not
   currently surfaced.
5. **Selection accuracy** (feeds T3): on a small gold set of "user intent → correct Composio
   action", measure top-1 / top-3 of `get_tools_for_step`. None exists today; the tool-routing
   eval set (`scripts/eval/tool_routing/`) is platform-tool-centric, not Composio-action-centric.

All five belong in the unified eval/telemetry harness (T3); #1–#4 are dashboard counters, #5
is an offline eval.

---

## H. Cost note (informational)

Two cost layers. **Composio platform fees:** Free 20K tool-calls/mo; $29/mo → 200K
(~$0.145/1K bundled, $0.299/1K overage); $229/mo → 2M (~$0.115/1K bundled, $0.249/1K
overage); premium tools ~3× ([composio.dev/pricing](https://composio.dev/pricing),
[usagepricing](https://www.usagepricing.com/blueprint/composio)). At pilot volume this is
negligible; it becomes a real line item only past ~200K actions/mo. **Compute inside the
wrapper:** each `get_tools_for_step` can trigger SDK per-app schema fetches (cached 1h,
`client.py:103-106`) and, on a cache miss, a re-fetch at limit=500; the daily `run_full_sync`
pulls *all* ~880 apps' actions in bulk (minutes, batched commits of 100) plus a capped 30-app
parameter backfill — heavy but off the request path. The tokens that matter are the **tool
schemas injected into the prompt**: capped at 30 tools × ~600 tokens ≈ 18K
(`composio_tool_service.py:40-41`), bounded further by `COMPOSIO_SECTION_MAX_TOKENS=1000`
for the hint text. No per-operation cost gate is applied on the Composio lane (the policy
plane's projected cost is 0 because no caller threads token estimates — cross-module finding,
not re-litigated here).

---

## I. UX / surface

Surfaces: **Tools** page (`frontend/app/tools/page.tsx` → `ToolsDashboard`;
`my-tools-dashboard.tsx` lists connected apps via `useConnectedApps`), the
`composio-apps-section` / `tool-actions-modal` / `tool-config-modal` for connect + per-agent
action enable/disable, the OAuth popup callback (`app/tools/callback`), the **email panel**
(`api/widget_email.py` proxying `GMAIL_FETCH_EMAILS`/send), and the in-chat running/error
tool chips (`message.tsx:289-319`). Concrete changes, North-Star-ranked:

1. **Make the per-action enable/disable UI actually bind to chat.** Today it gates only the
   meta-tool path (C.3), so operators toggling an agent's Slack actions see no effect in the
   agent's actual behaviour. Either enforce scope on the chat path (preferred) or remove the
   control — a setting that does nothing is worse than none.
2. **A connection-health surface that reflects reality.** Show per-app connection status,
   last successful execution, and last error — sourced from real telemetry once C.3 lands.
   Right now `my-tools-dashboard` shows connect status but nothing about whether actions
   *work*.
3. **Honest in-chat outcomes for Composio actions** (fix C.4) so a failed send-email shows an
   error chip, not a green tick.
4. **A trigger/automation surface** once F066/C.5 is addressed: which triggers are
   subscribed, what they route to, and a dead-letter view of `unrouted_events` so the "why
   didn't my Gmail trigger fire an agent" question is answerable in the UI.
5. **Admin sync visibility:** surface `composio_sync_jobs` (last run, apps/actions synced,
   errors) — it exists in the DB and nothing reads it; it would have made C.2's silent no-op
   visible.

---

## J. Upgrade path (impact × effort, judged by North-Star autonomy/quality)

**P0 — correctness/guardrails autonomy depends on (high impact, low effort):**
1. **Reject-on-mismatch webhook verification** (C.1): delete the "allow through for debugging"
   fall-throughs at `composio.py:629-632`, require a signature when a secret is configured,
   and 401 on missing/invalid. ~1 file. Unblocks a *trustable* inbound autonomy leg.
2. **Fix or replace the destructive-gate feeder** (C.2): make the classifier sync classify
   *connected* apps (drop the 8-app placeholder) and call the sync `get_app_actions` without
   `await`; or fold classification into the single consolidated sync (§E). ~2 files. Turns
   the F018 gate from blind to real.
3. **Honest success/failure** on the chat shortcut (C.4): return real success + emit an error
   chip. ~1 function.

**P1 — one execution chokepoint (high impact, medium effort):**
4. Route the chat per-action shortcut through the same `ComposioToolExecutor` /
   `execute_and_format` path as the meta-tool (C.3), so scope validation, the destructive
   gate, **telemetry**, and file-upload resolution apply everywhere. This is the highest-value
   structural fix: it simultaneously closes the gate bypass, makes the per-action enable UI
   real, and **feeds the learning loop** (starts filling `tool_execution_logs` for the 21 real
   workspaces, unblocking the whole W7 operating-graph investment). `consumers/chatbot/service.py`
   + a thin executor entry.

**P2 — consolidation & vendor-alignment (medium impact, medium effort):**
5. Collapse the three sync services into one; delete `tool_manifest_service.py` (0 importers)
   and the placeholder `ComposioToolRouter.search_tools` (or wire it). Removes the reinvention.
6. Evaluate **adopting Composio Tool Router / the hosted MCP endpoint** for *selection*
   (§E) to retire the bespoke mirror + alphabetical-search workaround; keep a thin local cache
   for the validation contract and degraded operation. Net line reduction.

**P3 — inbound autonomy (high impact, higher effort):**
7. Add an inbound email channel + generic trigger routing beyond `JIRA_*` (C.5/F066) so the
   refund-email-class journey has an autonomous path. This is where "agents do real client
   work end-to-end" is currently blocked.

**Cross-cutting hooks:** P0.2 + G#2 feed **T3** (a real destructive-gate coverage metric);
P1.4 is the specific fix that makes **T3's** per-module telemetry non-empty for this lane.
This module does not need the T1 graph substrate — its data is a vendor catalog + execution
log, not memory — but its telemetry, once flowing, is a primary *input* to the operating
graph that T1/tool-selection depend on.
