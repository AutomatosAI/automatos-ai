# Security & Defensive-Hardening Appendix — Automatos AI Phase-2 Review

> **Status:** Owner-authorised, analysis-only, **defensive** hardening review of the founder's own platform. No offensive action; every entry below is a *fix to apply to your own system to protect your own users*.
> **Model / pass:** Opus, the dedicated security pass deliberately separated from the Fable dossiers (per the brief §"Security & Defensive-Hardening Pass").
> **Source of truth:** pinned read-only worktree `origin/main @ 77bc9c6d5` (all 14 hardening waves merged). Every `file:line` below was read in that tree; the top-5 were spot-verified verbatim during this pass. Paths are relative to `orchestrator/` unless noted.
> **Inputs:** all 28 Phase-2 dossiers (`reports/dossiers/*.md`) + real-data evidence (`reports/dossiers/evidence/`, prod Postgres read-only 2026-07-04). The dossiers already surfaced the defects; this pass re-frames them as a prioritised defensive backlog with the specific hardening fix for each.
> **How to read the priority:** ranked by **real exposure** = (reachable by an untrusted party) × (blast radius) × (live-by-default). The three untrusted edges are the **live storefront widget**, the **channels/webhook ingress**, and the **Composio external-action lane**. Internal-only or pilot-dormant surfaces are ranked lower even when the defect is real.

---

## 0. Executive read

The platform's security posture is **"careful engineering, wired but not armed."** The hard, easy-to-get-wrong primitives are mostly done well — Fernet-encrypted credentials with BOLA checks, a server-minted actor identity that closes the agent-impersonation door, a fail-closed tenant-resolution spine with an advisory-lock provisioning race fix, a two-stage plugin security scanner, SSRF-blocked document rendering, a sanitised git-clone path, and a taint-gated memory-promotion job. Those are real and are credited in §7.

What is missing is **enforcement-by-default at the untrusted edges**, and it clusters into five themes:

1. **Webhook trust is broken on the two ingress lanes that dispatch autonomous work.** The Composio trigger webhook *logs and proceeds* on a signature mismatch (F028); the channels/workspace/playbook webhooks skip HMAC entirely when no signature header is present. A forged event dispatches a real agent or playbook.
2. **The governance/policy plane that would enforce budget, approval, and act-vs-ask is default-OFF and fails-open even when ON** — and four execution lanes bypass its chokepoint anyway. So on a real deployment there is effectively no runtime authorization boundary on external side-effects.
3. **Authorization forks five ways and enforces on ~7% of mutating routers.** Workspace `editor`/`viewer` roles are decorative; the empty-permission "god-key" is live on the widget plane; `super_admin` is locked out of admin routers — all because the unifying fix is behind the same OFF flag.
4. **The storefront widget — the one surface talking to the open internet — has an origin-check bypass, default-open CORS, an inert rate limiter, and shares one DB session across concurrent callback tasks.**
5. **Untrusted content flows into places that steer other agents:** cross-workspace skill attachment injects another tenant's prompt content into your agent; the memory section injects unfiltered stored content (including the model's own recorded lies) with no relevance floor; and a random-vector embedding fallback can silently poison retrieval.

None of these require a rebuild. The single highest-leverage action is a **staged rollout of `AUTOMATOS_POLICY_PLANE=on` with a fail-closed-for-destructive branch**, which closes the F040/F042/F043 authorization cluster at once. The webhook fixes are one-file deletions of "allow through" fall-throughs. The pilot's low real-traffic (dossiers confirm most surfaces are cold) means the exposure window is small *today* — but the storefront widget and channels are exactly the surfaces that go hot first when real customers arrive, so these should land before scale, not after.

---

## Top-3 risks (the ones to fix first)

1. **Forged webhook → autonomous agent/playbook execution (Composio + channels ingress).** `api/composio.py:630,633` logs `"...allowing through for debugging"` on a V3 signature mismatch *and* on the verification exception, then falls through; verification only runs when a signature header is present at all. The channels/workspace/playbook lanes have the same "HMAC optional when header absent" shape (`api/webhooks.py:59,67-69`). A forged Jira/agentic event dispatches a real agent or recipe (`api/composio.py:772-786`). **This is the sharpest edge on the platform: the inbound autonomy leg cannot trust what it receives.** Fix: require a signature when a secret is configured; **reject** (401) on mismatch/exception; return 401 when headers are absent. §1.1.

2. **The runtime authorization boundary is off by default, fails open, and is bypassed.** `AUTOMATOS_POLICY_PLANE` defaults false (`config.py:645`, verified) and is set nowhere in any deploy surface, so budget admission, act-vs-ask routing, the empty-permission-deny semantic (F042), and the `super_admin ⊇ admin` hierarchy (F043) are all dark; even ON, the gate *fails open* on any internal error (`modules/tools/execution/unified_executor.py:271-280`); and four lanes (chat Composio shortcut, playbook Composio steps, widget-email, `/api/tasks` direct-step) execute tools *around* the chokepoint entirely. Net: on every real deployment there is no enforced boundary on external side-effects, and the widget empty-permission key is a live god-key. Fix: staged `=on` rollout with a **fail-closed** branch for destructive/external risk classes, then close the four bypasses. §2.1, §2.2, §3.1.

3. **The live storefront widget — the only open-internet surface — has an origin bypass + default-open CORS + inert rate limiting.** A public `ak_pub_` key skips the domain check entirely when the `Origin`/`Referer` header is absent (`api/widgets/auth.py:182-183`, verified: `if origin and not check_domain(...)`); CORS allows *all* origins when `WIDGET_ORIGIN_ALLOWLIST` is unset (`api/widgets/cors.py:48-51`, verified, the default posture); and the per-process in-memory rate limiter is, by its own docstring, largely inert (`api/widgets/rate_limit.py:90-98`). Together these let a scraped public key be replayed from anywhere with weak throttling, burning the merchant's model budget and callback fan-out. Fix: deny public keys when origin is absent (fail-closed), require the allowlist in production, move the limiter to a shared store. §1.2.

---

## 1. The untrusted edges (highest exposure)

These three surfaces are reachable by parties you do not control — the open internet (widget), external platforms and any URL-poster (channels/webhooks), and third-party SaaS events (Composio). They get first priority regardless of current traffic.

### 1.1 Webhook authenticity — forged events dispatch autonomous work `[CRITICAL]`

**Risk.** Three inbound lanes turn an unauthenticated (or under-authenticated) HTTP POST into a real agent or playbook run:

- **Composio trigger webhook** (`api/composio.py:597`). Verified in the pinned tree: on a V3 signature mismatch the code executes `logger.warning("V3 webhook signature mismatch — allowing through for debugging")` and falls through (`:630`); the `except` branch does the same (`:633`). Worse, the whole verification block is gated on `if webhook_secret and v3_signature:` (`:618`) / `elif webhook_secret and x_composio_signature:` — **omit both signature headers and no verification runs at all.** A routed event then dispatches an agent or a recipe as a background task (`:772-786`). (The *legacy* header path does correctly `raise HTTPException(401)` on mismatch — so the fix is to make V3 behave like legacy, plus reject-on-absent.)
- **Workspace webhook** (`api/webhooks.py:308`) and **Playbook webhook** (`api/workflow_recipes.py:1752`). Auth is URL-as-secret plus an **optional** HMAC that is skipped when no signature header is present (`webhooks.py:59,67-69`). The URL secret is real defence-in-depth, but a leaked/logged webhook URL is then the entire boundary, and the HMAC that would re-verify is skippable.
- **GitHub PR webhook** (`api/github_webhooks.py:37`) is the one that does it right — **mandatory** HMAC, 500 without secret, 401 on missing/bad signature (`:60-70`). It also happens to be functionally dead (returns `workflow_execution_disabled`), but its *auth posture is the template* for the other two.

**Blast radius.** The Composio lane is the platform's designed autonomous-inbound leg; a forged event can drive an agent tool-loop (real external side-effects via Composio) or a playbook run in the target workspace. 13 live Composio-trigger dead-letters already exist in prod (`evidence/data/rag-feedback.md` / operating-graph evidence) — the lane receives real traffic.

**Defensive fix.**
- Composio (`api/composio.py:618-635`): delete both `"allowing through for debugging"` fall-throughs; on mismatch or verification exception, `raise HTTPException(401)`. When `COMPOSIO_WEBHOOK_SECRET` is configured, **require** a signature header and 401 if absent (treat missing-signature as untrusted, not as "no verification requested").
- Channels/playbook (`webhooks.py:59-69`): when a per-workspace/global webhook secret exists, make HMAC **mandatory** for that lane — reject on missing/invalid signature rather than skipping. Keep the URL secret as an additional factor, not the only one.
- Add a replay guard on all three: reject if `webhook-timestamp` is older than a small skew window, and dedup on `webhook-id`/`event_id` (this also fixes the retry→double-execution issue in §1.3).
- Slack specifically: the collected-but-unused `signing_secret` (`channels/drivers/slack.py:41`) should actually verify inbound `X-Slack-Signature`; today `_verify_webhook_signature` doesn't implement Slack's scheme at all (channels dossier C2-12).

**Effort:** S (delete fall-throughs + a required-signature branch per lane). **This is the single most important defensive change in the appendix.**

---

### 1.2 Storefront widget — origin bypass, open CORS, inert throttling `[HIGH]`

The widget (`api/widgets/*`, live on `inbuilduk`'s storefront) is the only surface exposed to arbitrary browsers. Four issues compound.

**1.2.a — Public-key origin-check bypass (F053, NOT DONE).** Verified: `api/widgets/auth.py:182-183` reads `origin = _extract_origin(request); if origin and not ApiKeyService.check_domain(...)` — when `Origin`/`Referer` is absent, `origin` is falsy and the domain check is **entirely skipped**, so a scraped `ak_pub_` key validates from any non-browser client (curl, server-side script) that omits the header. Public keys are meant to be origin-locked; this is the fail-*open* direction.
- **Fix:** for `ak_pub_*` keys, treat a missing origin as **deny** unless `allowed_domains` is explicitly empty-means-all *and the merchant opted into that*. Concretely: `if key_is_public and (not origin or not check_domain(...)): raise 403`. Fail closed on the key type whose entire security model is the origin lock.

**1.2.b — CORS default-open (F085-adjacent).** Verified: `api/widgets/cors.py:48-51` — `if not WIDGET_ORIGIN_ALLOWLIST: return True` allows *all* origins when the env var is unset, which is the default. The docstring says "in production the env var should always be set," but nothing enforces that.
- **Fix:** in `saas`/production edition, treat an empty allowlist as a **boot error** (fail-closed config, same pattern as `validate_auth_edition`), or default-deny with an explicit opt-in for dev. A permissive default on the internet-facing plane is the wrong resting state.

**1.2.c — Rate limiter is per-process and largely inert.** `WidgetRateLimitMiddleware` (`api/widgets/rate_limit.py:46-98`) uses an in-memory sliding-window dict that resets every deploy and is not shared across the 4 uvicorn workers; its own docstring notes the window check only activates once `api_key_id` is known, which happens *inside* the handler, not the middleware. The callback endpoint's DB-backed per-session/per-Site limits are real, but per-IP is explicitly deferred "until Redis" (`services/callback.py:44-46,128-129`).
- **Fix:** move to a Redis-backed shared sliding window keyed on `(api_key_id | client_ip)`; resolve `api_key_id` in the middleware (or a fast pre-handler) so the check actually gates the first request. Add a per-IP ceiling on `/api/widgets/chat` and `/api/widgets/callback` (both spend money — LLM turns and channel fan-out).

**1.2.d — Empty-permission god-key on the widget plane (F042, live).** With the policy plane OFF (default), `require_permission` on the widget path treats an empty permission list as **allow-all** (`api/widgets/auth.py:255-257`), while the board plane treats empty as deny. One minted no-permission key is unrestricted on widgets. (Root cause and platform-wide fix in §3.1.)
- **Fix:** the durable fix is the policy-plane rollout (§2.1); the local fix is to make empty-means-deny on the widget `require_permission` regardless of flag, so the internet-facing plane is never the permissive one.

**Blast radius.** A leaked public key (they ship in page HTML by design) replayed from a script with weak throttling → free LLM-turn burn on the merchant's budget, callback fan-out abuse, and (with an empty-perm key) unrestricted widget scopes. The GDPR-conscious callback design (phone never stored, salted per-Site hash — a genuine positive, §7) limits PII exposure here.

**Effort:** S–M (origin fail-closed + CORS boot-guard are small; Redis limiter is medium).

---

### 1.3 Channels ingress — sender identity, credential storage, retry-replay `[HIGH]`

Beyond the webhook-auth issue (§1.1), the channels lane has three data-integrity/trust gaps (channels dossier C2-12/14, D1):

**1.3.a — "Whoever messages the bot is the workspace."** Every inbound sender maps to `RequestUser(auth_type="webhook")` with no identity linkage (`ingestors/webhook.py:101`). There is no allowlist, no pairing/approval for unknown senders (contrast OpenClaw's one-time pairing-code model). Anyone who can reach the webhook URL acts with workspace authority.
- **Fix:** add an inbound-sender trust model — per-channel allowlist of known `chat_id`s, or a first-contact approval (a notification the owner must accept before an unknown sender's messages execute). At minimum, don't let an unknown sender trigger tool-using/destructive actions without an approval.

**1.3.b — Credentials stored plaintext while the schema claims encryption.** `core/models/channels.py:26` comments `config = Column(JSON …)  # encrypted credentials`, but the write path stores plaintext (`_json.dumps(config)`, `api/channels.py:362`). The code's claim about itself is false.
- **Fix:** encrypt `channel_connections.config` at rest with the existing `EncryptionService` (the same Fernet path used for Shopify tokens and BYOK keys — §7 shows it exists and is used correctly elsewhere), or delete the misleading comment and document the reliance on DB-at-rest encryption. Do not ship a schema that lies about protecting secrets.

**1.3.c — Retry → duplicate agent execution (no dedup).** The workspace webhook executes the agent synchronously inside the HTTP request (`webhooks.py:470-475`); nothing dedups `update_id`/`event_id`. Slack/Telegram redelivery on a slow response → the same event executes again, each burning tokens (and each firing any side-effects again).
- **Fix:** the same dedup index proposed in §1.1 (unique on `platform + event_id`) plus a fast-ack + background-execute pattern so a slow agent run can't trigger provider retries.

**Effort:** S–M. Note the channels plane is currently near-unused in prod (2 connections ever, the one active can't even reply due to F026), so exposure *today* is low — but these are the exact gaps that bite the day a channel goes live.

---

### 1.4 Composio external-action lane — the gate is blind and bypassed `[HIGH]`

The Composio wrapper is "the largest single external boundary in the platform" (composio dossier). Two guardrail defects (C.2, C.3) undermine the protections autonomy depends on:

**1.4.a — The destructive-action gate runs blind.** The daily classifier that populates `composio_action_metadata` (the table backing the fail-closed destructive gate, F018) **never writes a row**: `ComposioActionSyncService._get_all_enabled_apps()` returns a hardcoded 8-app placeholder (`modules/tools/sync/composio_action_sync.py:330-340`) and `_fetch_app_actions` `await`s a synchronous method (`:207` calling `client.py:775`, a plain `def` returning a list) → `TypeError` for every app, caught as an error `SyncResult`. So the gate operates permanently on an 8-keyword *intent-text* heuristic. A destructive action under neutral wording ("issue a refund", "create a 10% discount") is exactly what slips through — forever, not just at cold-start.
- **Fix:** classify *connected* apps (drop the placeholder) and call `get_app_actions` without `await` (or fold classification into the working `MetadataSyncService`). Add a CI/startup assertion that `composio_action_metadata` is non-empty for connected apps, so a silent no-op sync fails loudly (it went undetected for months).

**1.4.b — The preferred chat execution path skips scope, gate, and telemetry.** When Composio tools are injected (the *designed* common path), the chat loop routes calls to `_execute_composio_action` → raw `client.execute_action` (`consumers/chatbot/service.py:1321-1335` → `composio_tool_service.py:257-274`), bypassing the per-action agent-scope check, the destructive gate, file-upload resolution, and `ToolExecutionLog` telemetry — all of which live only on the *meta-tool* path. So the per-agent app-enable UI has no effect on chat, the gate is bypassed on its most-needed path, and the learning plane is starved (a direct contributor to `tool_execution_logs` being 100% synthetic).
- **Fix:** route the chat per-action shortcut through the same `ComposioToolExecutor.execute` / `execute_and_format` chokepoint as the meta-tool. This is the single structural fix that restores scope enforcement + the gate + telemetry on the highest-volume external lane (also flagged in §2.2 and the tool-runtime dossier J-2).

**1.4.c — Supply-chain dependency risk (informational, not a code fix).** Composio disclosed a May 2026 incident (exfiltration of customer API keys / GitHub OAuth tokens; multi-day partial outage — composio dossier D). You store a Composio key and lean on their managed credential store.
- **Fix / posture:** keep the execution abstraction thin-and-swappable (the wrapper already is), rotate the Composio key on a schedule, and treat Composio credential loss as a modelled DR scenario. Don't deepen lock-in without a fallback.

**Effort:** 1.4.a and 1.4.b are S–M and high-value; 1.4.c is an ops posture.

---

## 2. The enforcement plane (governance / policy / audit)

The theme that most amplifies every other risk: **the boundary that would enforce budget, approval, and act-vs-ask is off, fails open, and is bypassed.** (governance-policy dossier C.1–C.5, tool-runtime B.7/C.2.)

### 2.1 Policy plane default-OFF and fail-open `[CRITICAL, enabler]`

**Risk.** Verified: `POLICY_PLANE_ENABLED` defaults false (`config.py:645`) and the residual map confirms nothing in `envs/`, `docker-compose.yml`, or `railway.json` sets it. With the flag off, `_policy_gate_check` returns immediately (`unified_executor.py:227-228`) — byte-for-byte the legacy per-router gates. So on every real deployment: the rate limiter is inert (F040), the admin gate is the old workspace-membership auto-flip (F014), the widget empty-permission god-key stands (F042), and there is **zero** budget admission. Corroborated by data: `approval_grants` and policy-verdict `audit_logs` are 0 rows ever in prod.

**Worse, even ON it fails OPEN.** `unified_executor.py:271-280` treats any exception in the plane as "proceed." A malformed policy doc, DB blip, or registry miss silently waves the call through. A governance gate whose failure mode is *allow* is a monitoring aid, not an enforcement boundary — and the board gate (`api/board_tasks.py:989-994`) and budget/document reads have the same posture.

**Defensive fix.**
- **Stage a rollout of `AUTOMATOS_POLICY_PLANE=on`** (default-off → per-env → default-on). The plane is reversible by design, so risk is bounded. This one change activates the already-written F042 (empty=deny everywhere) and F043 (`super_admin ⊇ admin`) fixes.
- **Land a fail-closed branch first:** at the gate exception path (`unified_executor.py:275`), branch on the classified risk — `read`/`internal_write` may fail open (availability), but `destructive`/`external_side_effect`/`publish` must **fail closed** (deny with an errors-as-data remediation). A one-branch change; disproportionate correctness gain, because it's exactly the deletes/sends/refunds class the plane exists to gate.
- Thread real model+token estimates into `ToolCall` so budget admission actually prices the pending call (today `projected_cost` is always 0 because no caller passes estimates — `gate.py:163-168`), and pass a real `estimated_cost_usd` into the board gate so the dollar ceiling can bind (today defaults 0.0 → always under any ceiling).

**Effort:** Low for the flag + fail-closed branch (config + one branch); Medium for pricing. **This is the highest-leverage security action in the review** — it closes the F040/F042/F043 authorization cluster and turns "scaffolding" into "guardrails."

### 2.2 Four lanes bypass the single chokepoint `[HIGH]`

Even with the plane on, four execution lanes run tools *around* `_policy_gate_check` — and none fire telemetry or outcome-capture (tool-runtime B.8, governance C.4):
1. **Chat per-action Composio shortcut** (`consumers/chatbot/service.py:1321-1335`) — §1.4.b.
2. **Playbook Composio steps** (`api/recipe_executor.py:655`).
3. **Widget-email actions** (`api/widget_email.py:286,340,388,437`) — call `client.execute_action` directly.
4. **`/api/tasks` direct-step lane** (`api/tasks.py:62-124`) — see §4.1.

External side-effects — the highest-risk class — are the ones most able to slip the gate.
- **Fix:** route lanes 1–3 through the unified executor's chokepoint (or, for widget-email, through the same `ComposioToolExecutor`). For lane 4, see §4.1. This is what makes "one plane governs everything" true rather than aspirational.

**Effort:** M (per lane, mechanical).

### 2.3 Audit + GDPR are headless and unproven `[MEDIUM]`

- The Art.12 audit handler only registers when the flag is on (`main.py:521`), so no policy-verdict audit row has ever been written (0 in prod). Once §2.1 lands, add a **retention policy** for `audit_logs` (EU-AI-Act Art.12 mandates ≥6 months as a floor — governance H) and an operator surface to read it (there is none today — governance C.8).
- The GDPR erase cascade (`services/gdpr_service.py`) is real code across SQL + Qdrant field + mem0, but the field-memory/mem0 legs target stores that carry **no data-subject tag**, so `erase-subject` returns mostly `gaps` (governance C.9, memory B.4). A subject-erase that mostly returns gaps is not yet a defensible GDPR answer.
- **Fix:** add subject tags to field-memory/mem0 payloads at write time so `erase-subject` can actually delete; verify the cascade against live stores (currently unreachable from CI — flag as an ops task requiring Railway-internal access). Credit: the cascade honestly *reports* its gaps rather than lying (§7).

**Effort:** M.

---

## 3. Tenant isolation & access-control correctness

The tenant-*resolution* spine is genuinely strong (§7). The tenant-*authorization* layer is the unfinished half (auth-identity dossier C.1–C.6).

### 3.1 Authorization forks five ways; roles enforce on ~7% of routers `[HIGH]`

**Risk.** Five disjoint role vocabularies coexist and don't map to each other (`system_role`, workspace `owner/admin/editor/viewer`, frontend `SystemRole` incl. a `customer_manager` that exists in no backend set, SDK-key scopes, agent-tool matrix — auth-identity B). The consequences:
- **Editor/viewer roles are decorative on ~85 of ~92 mutating routers.** `@require_permission` (the only reader of the role matrix) appears on ~6 endpoints (team + widget). Every other mutating router takes the bare hybrid dependency, which resolves a workspace and a `system_role` but **never consults the workspace role** — so a `viewer` can create agents, launch missions, edit documents, and delete deliverables. The UI renders a "viewer" who is functionally an editor. (auth-identity C.2, measured ≈7% enforcement coverage.)
- **The empty-permission fork (F042)** is the live behaviour: empty perms = god-key on widgets, null-key on the board (§1.2.d).
- **`super_admin` is locked out of every admin router** on a default deployment (F043) — the hierarchy fix is behind the OFF flag.

**Defensive fix.**
- Flip `AUTOMATOS_POLICY_PLANE` on (§2.1) — closes F042/F043 immediately.
- Do the `@require_permission` sweep across the ~85 unguarded mutating routers so `editor`/`viewer` actually gate agent/mission/document/deliverable writes. This is the fix that turns per-tenant roles from decorative to real.
- Collapse the five vocabularies to one authority (`modules/policy/roles.py`) with typed adapters; delete the frontend `customer_manager` ghost.

**Effort:** M (mostly wiring + the router sweep). This is the "authorization is the module's unfinished work" headline (auth-identity).

### 3.2 Cross-workspace read/attach holes `[HIGH]`

Several concrete cross-tenant gaps, each a correctness hole independent of the role fork:

**3.2.a — Skill attach injects another tenant's prompt content (untrusted-content-steers-agent).** Verified: `api/agents.py:437-445` attaches skills by id filtering only `Skill.id.in_(ids), is_active == True` — **no global-or-own-workspace visibility check**. Same on bulk (`:367-375`) and `POST /{id}/skills` (`:785-789`). Another workspace's private skill content can be attached to your agent by id and thus **prompt-injected into your agent's every turn**. This is squarely the brief's "trust surface of inter-agent exchanges — one agent's output steering another." (agents-skills C.8.)
- **Fix:** apply the `_skill_visible_to` predicate (already used correctly in `api/skills.py:160`) to all three attach sites: a skill must be global or owned by the caller's workspace.

**3.2.b — Credential NULL-workspace hole.** Verified: `api/credentials.py:67` — `if hasattr(cred,'workspace_id') and cred.workspace_id and str(cred.workspace_id)!=str(ctx.workspace_id): raise 404`. A credential row with `workspace_id IS NULL` (legacy/globally-seeded) therefore passes for **every** tenant, and `resolve_credential`-by-name (`:599-608`) has no workspace filter and relies on this same null-permissive post-check. The by-id paths are otherwise sound. (auth-identity C.6.)
- **Fix:** treat null `workspace_id` as **deny** for non-admin callers, and scope `get_credential_by_name` to the workspace. Small, high-value.

**3.2.c — Cross-workspace file dedup returns another tenant's document id.** Verified: `modules/rag/ingestion/manager.py:737` — `SELECT id, status FROM documents WHERE file_hash = %s` with **no workspace predicate**, while the INSERT stamps `workspace_id` (`:774`). A byte-identical file uploaded by workspace B returns workspace A's document id. The fail-closed retrieval filters prevent B from *reading* A's content (so this is a broken-flywheel dead-end, not a content leak today), but it is a tenant-scoping violation one refactor away from a leak. Five callers use this path. (rag-retrieval C.7.)
- **Fix:** add `AND workspace_id = %s` to the dedup query.

**3.2.d — Cross-Mission agent-identity bleed on graph/document reads.** The `_agent_id` injection for graph/doc tools resolves via `.first()` on *any* running Mission in the workspace (`platform_executor.py:864-888`) — concurrent Missions can read the graph/documents under each other's team scope (too-wide or too-narrow). This is the same defect family F020 already fixed for *field* tools. (knowledge-graphs C.8.)
- **Fix:** thread the calling task's own run context into `_agent_id` resolution (reuse the F020 caller-context pattern), don't `.first()` a sibling Mission.

**3.2.e — The agent-tool RBAC plane is workspace-blind.** `AgentToolPermission`/`Tool`/`PermissionAuditLog` carry **no `workspace_id`** (`core/models/tools.py:103-140`), so `api/permissions.py` cannot be tenant-scoped even in principle and queries globally, ignoring `ctx`. It reads as an RBAC pillar; it governs nothing the runtime respects. (auth-identity C.5.)
- **Fix:** this is a delete, not a patch — kill the router + matrix + tables (after a writer audit). Tool authorization lives correctly in the Composio/tool-runtime lane.

**3.2.f — Chat and widget conversations all belong to user id=1.** Both `api/chat.py:82-89` (`get_user_id` ignores the authenticated principal) and the widget's `_get_widget_user_id` (`api/widgets/chat.py:89-104`) own conversations under a constant user row. In chat, the ownership check `chat.user_id != user_id` compares the constant to itself, so access control degenerates to workspace scoping, and PRD-163 approval attribution resolves to user 1's clerk id regardless of who is chatting. In any multi-user workspace this is shared history + misattributed approvals. (auto-core C.1, storefront-widget C-defect-5.)
- **Fix (chat):** thread `ctx.user` into `chats.user_id`, message saves, vote checks, and `_driving_clerk`. Cheap, high-trust. (Widget's user-id-1 is a separate design question — per-shopper identity — but the *chat* one is a straight bug.)

**Effort:** 3.2.a/b/c are S each; 3.2.d/e/f are S–M.

---

## 4. The compute/code-execution surface

### 4.1 `/api/tasks` direct-step lane runs raw shell/git with auth-only checks `[HIGH]`

**Risk.** Verified: `api/tasks.py` `submit_task` (`:62`) takes only `get_request_context_hybrid` + `get_db`, then enqueues **concrete shell/git/file steps** to the workspace worker via Redis — **no PolicyGate, no budget ceiling, no approval, no telemetry** (F060 residue; code-canvas C.7, governance C.4). This is a lane that *can* run unattended (automation/agents POST steps) and sits entirely outside the merged W11 policy plane. The steps are bounded by the worker's exec sandbox (allowlist + blocked patterns + non-root uid + `resolve_safe_path` — a genuine positive, §7), so it is not arbitrary RCE, but it is an ungoverned side-effecting ingress.
- **Fix:** route `/api/tasks/submit` through the policy gate (§2.1) so budget/approval/audit apply to the unattended-execution ingress; or, if the lane isn't needed, delete it (the missions/tool-runtime dossiers both flag it as a candidate). Cheapest safety win on the compute surface.

### 4.2 Code-canvas isolation is shared-process, not per-session `[MEDIUM, forward-looking]`

The workspace-worker runs one process (concurrency 3) on one shared volume; each canvas session is a `claude` subprocess of that shared worker fenced only by path-rebinding + a command-string regex, all as the same `worker` uid on one kernel (code-canvas C.4). The confinement code is pure-stdlib and unit-tested (§7), and the canvas cannot even be prompted yet (C.1, so exposure is nil *today*), but this is materially weaker than the per-task microVM isolation the peer coding agents ship.
- **Fix (when the lane goes live):** adopt a per-session sandbox runtime (E2B / Modal / Firecracker) so the canvas can run agentic code with kernel-level isolation. Not urgent while the loop is open, but a prerequisite before this lane executes untrusted or client code as a product feature. Persist an approval/decision audit record (today approvals live only in an in-memory registry — C.8 — so "zero unapproved writes" is unauditable).

### 4.3 Legacy workflow engine is Composio-webhook-reachable `[MEDIUM]`

The legacy `api/workflows.py` (1,424 lines) stays mounted with one live `execute-advanced` endpoint, and the coded `jira_bug_triage` recipe is reachable via Composio's `_dispatch_workflow` (`api/composio.py:828-850`) — a fifth execution engine with **none** of the missions/board hardening (F078; missions C.8, playbooks B). It rides the same forged-webhook risk as §1.1 into an ungoverned engine.
- **Fix:** retire it — migrate `jira_bug_triage` onto the Mission/Playbook path and unmount `workflows`/`workflow_templates`. Shrinks the ungoverned surface. (Also delete the dead `/api/playbooks` router, whose GET does `db.execute(raw_string)` without `text()` — playbooks B; harmless-because-crashes today but a squatter on the canonical route.)

**Effort:** 4.1 is S; 4.2 is L (adopt, deferred); 4.3 is M.

---

## 5. Data integrity & content-injection into the model

The brief specifically flags memory ("shared/promoted content across agents") and adversarial input. These are the paths where untrusted or low-quality *content* reaches the model and steers it.

### 5.1 Random-vector embedding fallback silently poisons retrieval `[HIGH]`

**Risk.** Verified: `core/llm/clients/base.py` `DeterministicEmbeddingProvider.generate_embedding` returns `np.random.default_rng(abs(hash(text)) % 2**32).standard_normal(dim)` — **random vectors** — and is selected on a missing/misconfigured embedding key (`embedding_manager.py:90-93,113-118`) with only a `logger.warning`. On the same OpenRouter key that has been 402-failing daily since mid-June (rag-retrieval C.2), a query embedding raises → in some paths swallowed to empty retrieval; in the config-fallback path it becomes *meaningless* retrieval. **A retrieval system whose emptiness is indistinguishable from its success cannot be trusted** — and worse, random-vector "grounding" would let an agent cite arbitrary documents as relevant.
- **Also note:** Python's built-in `hash()` is salted per-process (PYTHONHASHSEED) by default, so this "deterministic" provider isn't even deterministic across restarts — the same text yields different vectors, so the class is mislabelled *and* unstable.
- **Fix:** **remove `DeterministicEmbeddingProvider` from all production selection paths — fail loud instead of retrieving noise.** A missing embedding provider should be a hard error (the platform already uses fail-loud config elsewhere — §7), not a silent downgrade to random vectors. Keep it (if at all) strictly for offline unit tests, and if a deterministic test double is wanted, seed it from `hashlib.sha256`, not `hash()`.

**Effort:** S (delete the fallback selection; raise instead).

### 5.2 Memory section injects unfiltered content — including the model's own recorded lies `[MEDIUM]`

**Risk.** The memory section (`modules/context/sections/memory.py:261-267`) renders the top-8 stored memories with **no relevance floor and no content-type filter**. In prod, ~87% of `memory_short_term` is duplicated operational chatter (402-failure spam stored twice/run, heartbeat summaries), and the only "informative" rows are raw chat clippings **including the user telling Auto it is lying** ("auto you are lying again…") (memory C, context-assembly C.1). So the "What You Know About This User" block injected into every chat/task/recipe prompt can be 402-spam or Auto's own recorded overclaims — content that then steers the next turn. This is a content-integrity risk: low-trust stored content flows uncritically into the prompt.
- Related trust issue (auto-core C.3): Auto has been observed **overclaiming actions it didn't execute** ("I can't edit the live blog/CMS…"), and nothing compares claimed actions against `tool_execution_logs`.
- **Fix (assembly-side guarantee, independent of the write-side cleanup):** at `memory.py:261-267`, apply a relevance threshold and **exclude operational content types** (`playbook_summary`, `heartbeat_log`) from the user-memory block, so noise/lies cannot reach the prompt even when the store is dirty. Add the "overclaim guard" as a metric: a post-turn check that claimed actions have matching `tool_execution_logs` rows (feeds the trust story and requires §1.4.b/§2.2 telemetry to be real).

**5.2.a — Commerce KG polluted with conversation memory (shopper-facing traversal).** The live Shopify pilot's graph carries 510 `l2:` nodes — raw proactive-opener directive strings and chat clippings mixed into the same graph the widget openers traverse (shopify-vertical C.2). So the shopper-facing recommendation surface walks a graph ~2% conversational-noise by node count.
- **Fix:** partition catalog/orders extraction into its own graph partition, or filter `l2:`/content nodes out of `_resolve_graph_related_products` traversal. Keeps conversation content out of the customer-facing surface.

**Effort:** 5.2 is M; 5.2.a is M.

### 5.3 Untested extraction pipelines feed autonomous, provenance-cited output `[MEDIUM]`

The storefront opener says "bought together in X of Y orders" citing `co_count`/`total_orders` from `map_shopify_orders`/`map_shopify_catalog` (`modules/knowledge/graph_extraction.py:503,693`), which have **zero behavioral tests**, and both headline golden journeys are skipped (F091, widened by W13 auto-run on webhooks). A silent mapper regression makes the widget **confidently fabricate to real shoppers** with a provenance-styled sentence (storefront-widget C-defect-2, shopify C.6). This is an integrity risk on the one surface talking to end customers: autonomy that fabricates-with-citations is worse than a canned opener.
- **Fix:** add fixture-driven behavioral tests for both mappers and un-skip the golden journeys, gated in CI. Not a runtime hardening change but a defensive guarantee against silent corruption of client-facing claims. Also: catalog re-sync uses `merge=False` which **wipes** the FBT edges every catalog webhook (shopify C.1) — an integrity bug that silently destroys the cross-sell signal; fix to strip-then-remerge like the orders path.

**Effort:** S–M.

### 5.4 Graph promotion taint-gate — a positive to preserve `[credit]`

The field→durable memory promotion job evaluates a **taint gate first** — untrusted-provenance patterns never promote (`jobs/promote_field_memory.py:104-121`, `field_scoring.py:112-148`) — which is exactly the right defence for "shared/promoted content across agents" (memory C.7). **Keep this ordering.** When you consolidate memory (memory dossier E), carry the taint gate forward; it is the one place the platform already reasons about content trust at a promotion boundary.

---

## 6. Secrets, credentials & config hygiene

### 6.1 Tracked Clerk test artifact still in git history `[MEDIUM]`

`tests/e2e/.auth/user.json` is still tracked on live main (F012, PARTIAL) — `.gitignore` lists it but a gitignore is a no-op for an already-tracked file (no `git rm --cached` was done), so the gitleaks lane stays red pending a history purge (auth-identity C.7, deployability F012). JWTs were 60-second so live-credential risk is low, but the re-commit hazard is open.
- **Fix (human):** `git rm --cached` + history rewrite + rotate anything that was ever real. Greens the gitleaks lane.

### 6.2 Credential-resolution guessing cascade `[LOW-MEDIUM]`

`get_credential_data` (`core/llm/manager.py:135-363`) tries an explicit mapping, then **7+ name variations** ("production_openai_api", "openai", "Openai", …), then type-based lookup, then a **production→development environment fallback**, then env vars, all TTL-cached 300s (llm-core C.9). The prod→dev fallback is a footgun (a prod deploy can silently resolve a dev key), and the 5-minute cache means key rotation lags. Failure diagnosis is miserable.
- **Fix:** make the resolution order explicit and documented; drop the silent prod→dev fallback (or make it opt-in and logged loudly); shorten or invalidate the credential cache on rotation.

### 6.3 FutureAGI worker defaults to localhost → wasted calls / SSRF-shape `[LOW]`

`AGENT_OPT_WORKER_URL` defaults to `http://localhost:8080` (`config.py:809`), which makes `is_available` **always true** (`futureagi_service.py:64-67`), so the chat live-eval hook fires fire-and-forget HTTP at a dead localhost port on any deploy that hasn't stood the worker up (llm-core B.8). Fire-and-forget so it's wasted work + log noise, not user breakage — but pointing a server-side fetch at a default localhost port is the wrong resting state.
- **Fix:** default `AGENT_OPT_WORKER_URL` to empty so the service reports honestly unavailable and fires no calls unless configured.

### 6.4 CI gates don't bite → security lanes are advisory `[MEDIUM]`

Branch protection on `main` is `strict:false` with only `["orchestrator-tests","ioc-scan"]` required (F057, live-verified 2026-07-04 — deployability C-defect-4). So **none** of `frontend-ci`, `codeql`, `gitleaks`, `alembic-from-zero`, or the fresh-clone smoke lane can block a merge, and two lanes are green-while-broken (`continue-on-error` masks real failures). The supply-chain scanning (CodeQL + gitleaks + automated security fixes) *runs and is green* (§7) but doesn't gate.
- **Fix (30-second repo-admin action):** flip `strict:true` and add the green security/CI lanes as **required** contexts (the ready command is in `docs/runbooks/W12-BRANCH-PROTECTION.md`). This is what gives every security check teeth; the July review attributed real red-main incidents to exactly this gap.

**6.5 — Dead/misleading files (low-stakes hygiene).** `api/anthropic_client.py` still imports from a non-existent `api/base.py` (F083 leg 2, latent broken dead file); three tracked lockfiles → nondeterministic builds; `/api-control` and `/styleguide` are routed prod pages. None break a deploy; all erode the surface an evaluator/attacker sees. Fold into the never-authored PRD-184 kill-list.

**Effort:** 6.1 is XS (human); 6.4 is XS (repo-admin); the rest are S.

---

## 7. What is already handled well (honest credit)

The brief asks for honesty about what's done right. This is not a platform with a weak security *culture* — the hard primitives are careful. Specifically:

- **Tenant-resolution spine is genuinely strong.** `get_request_context_hybrid` does a real cross-tenant access check (an `X-Workspace-ID` a user can't reach is *not* honoured — `core/auth/hybrid.py:146-165`), audience-pinned JWT verification, existence-hiding 404s, and a `pg_advisory_xact_lock` provisioning race fix so a new user can't get duplicate workspaces (`:298-301`). A real W2 tenant-isolation test suite backs it. (auth-identity C.)
- **Actor identity is server-minted.** `exec_platform.execute_platform_action` strips any caller-supplied `_agent_id`/`_agent_name` and injects the trusted runtime identity (`modules/tools/execution/exec_platform.py:59-81`) — the agent-impersonation door is closed at the dispatch layer, not per-handler. (tool-runtime B.3.)
- **Credential store is real.** Fernet encryption, per-row BOLA checks on every by-id path, admin-gated decrypt/resolve, audit rows (`api/credentials.py`, PRD-18). Shopify Admin token is Fernet-encrypted at rest (F058), and the Shopify internal webhook key is fail-closed with `hmac.compare_digest` (F004) — the *right* posture, which §1.1 should propagate to the other lanes. (auth-identity C, shopify C.)
- **SSRF discipline in document rendering.** WeasyPrint's URL fetcher blocks non-public hosts and `file://` (`generation_service.py:31-56`); the DOCX image fetch refuses private hosts *and redirects* (`docx_renderer.py:88-120`); the legacy Jinja path renders through a `SandboxedEnvironment` (anti-SSTI, PRD-156 S4). (deliverables C.)
- **Git-clone is sanitised.** Skill/codegraph clones go through `core/security/git_sanitizer` (domain allowlist, branch validation, `--` separator) with a 42-pattern dangerous-content scan per SKILL.md (skill_loader, knowledge-graphs B3). The canvas git remote gets two injection gates and three-pass token redaction (code-canvas B6).
- **Plugin security scanner is real.** Two-stage: 42 static patterns (code/network/filesystem/prompt-injection groups, auto-block on critical) + an LLM risk scan (0–100 → safe/review/blocked), and it demonstrably ran on all 73 marketplace plugins with a plausible verdict spread (29/33/11). Launch posture ahead of ClawHub's (which shipped with no scanning). (agents-skills B4/D.)
- **The workspace-worker exec sandbox** — binary allowlist, blocked-pattern list, stripped env, `resolve_safe_path` rejecting absolute/`..`/null-byte, drop to non-root uid (`services/workspace-worker/executor.py:35-98`, `workspace_manager.py:242`) — is a careful containment for the shell/file lane. (code-canvas B1.)
- **Callback GDPR posture.** Shopper phone is **never stored** — only a SHA-256 salted-per-Site hash is persisted; plaintext is forwarded to merchant destinations and discarded (`services/callback.py:76-86`). (storefront-widget B.)
- **The taint-gated memory promotion** (§5.4) and the **honest GDPR gap-reporting** (the cascade reports what it *couldn't* delete rather than lying — `vector_field.py:231-247`) show the team reasons about content trust and doesn't fake compliance.
- **Fail-loud config and router mounting.** No hardcoded model/credential fallback (config raises with an actionable message); the router manifest raises `RouterMountError` on a silent import failure (`router_manifest.py:44-45`); config localisation is test-guarded (no `railway.internal` in defaults). The *instinct* is fail-closed even where the plane isn't armed.
- **The policy plane's design is security-aware:** an in-process typed bus that consciously **rejects shell-string hooks** ("RCE-by-configuration", `bus.py:4-6`), typed verdicts with `deny > ask > allow`, errors-as-data. It's off, not wrong (§2.1). (governance C positives.)
- **Supply-chain CI runs green:** CodeQL (python+js) + gitleaks full-history + automated security fixes are live and passing (deployability B/§7) — they just don't *gate* yet (§6.4).

The gap is consistently **arming and enforcing**, not **designing** — which is why the backlog below is mostly config flips, deletions of "allow through" fall-throughs, and wiring, rather than new systems.

---

## 8. Prioritised defensive-hardening backlog

Ranked by real exposure × blast radius × effort. Items marked **[human]** are owner/repo-admin actions, not code.

| # | Fix | Risk | Exposure | Effort | §ref |
|---|-----|------|----------|--------|------|
| 1 | **Reject-on-mismatch webhook verification** — delete Composio "allow through" fall-throughs (`api/composio.py:630,633`), 401; require signature when secret set on all three lanes; add timestamp/replay guard | CRITICAL | Composio + channels ingress (autonomous exec) | S | 1.1 |
| 2 | **Staged `AUTOMATOS_POLICY_PLANE=on` + fail-closed-for-destructive branch** (`unified_executor.py:275`) — activates F042/F043; price the budget gate | CRITICAL (enabler) | Every external side-effect lane | Low+M | 2.1, 3.1 |
| 3 | **Widget origin fail-closed** (`auth.py:182-183`: deny public key when origin absent) + **CORS boot-guard** (require allowlist in prod, `cors.py:48-51`) + **empty-perm=deny** on widget plane | HIGH | Open internet | S | 1.2 |
| 4 | **Route the 4 bypass lanes through the policy chokepoint** (chat-Composio, playbook-Composio, widget-email, `/api/tasks`) — restores scope + gate + telemetry | HIGH | External-action + compute lanes | M | 2.2, 4.1 |
| 5 | **Fix the destructive-gate feeder** (`composio_action_sync.py:330-340,207`) — classify connected apps, drop the `await`-list crash; assert non-empty at startup | HIGH | Composio destructive actions | S–M | 1.4.a |
| 6 | **Remove the random-vector embedding fallback from prod paths** — fail loud, not noise (`base.py` DeterministicEmbeddingProvider selection) | HIGH | Retrieval integrity | S | 5.1 |
| 7 | **Skill-attach visibility check** on all 3 sites (`api/agents.py:437,367,785`) — global-or-own-workspace only | HIGH | Cross-tenant prompt injection | S | 3.2.a |
| 8 | **Credential NULL-workspace = deny** + scope by-name lookup (`api/credentials.py:67,599`) | HIGH | Cross-tenant secret read | S | 3.2.b |
| 9 | **`@require_permission` sweep** across ~85 mutating routers so editor/viewer enforce; collapse 5 role vocabularies → 1 | HIGH | Intra-tenant privilege | M | 3.1 |
| 10 | **[human] Flip branch protection `strict:true` + require CodeQL/gitleaks/frontend/from-zero lanes** | MEDIUM (enabler) | Every future merge | XS | 6.4 |
| 11 | **`/api/tasks` under the policy gate (or delete)** — ungoverned shell/git ingress | HIGH | Unattended compute | S | 4.1 |
| 12 | **Cross-workspace file-hash dedup fix** (`manager.py:737` add `workspace_id`) + **`_agent_id` bleed fix** (`platform_executor.py:864`) | MEDIUM | Tenant scoping | S | 3.2.c/d |
| 13 | **Memory-injection quality floor** — exclude operational content-types + relevance threshold (`memory.py:261-267`); commerce-KG partition (`l2:` out of opener traversal) | MEDIUM | Content injection into model + shopper surface | M | 5.2, 5.2.a |
| 14 | **Encrypt `channel_connections.config`** (or delete the false "encrypted" comment) + inbound-sender trust model + webhook dedup | HIGH (when channels live) | Channels ingress | S–M | 1.3 |
| 15 | **Widget: Redis-backed rate limiter + per-IP ceilings + per-task DB sessions in callback fan-out** (`dispatcher.py:265-316`) | HIGH | Open internet abuse | M | 1.2.c |
| 16 | **[human] Purge `tests/e2e/.auth/user.json` from history + rotate** (F012) | MEDIUM | Secret re-commit | XS | 6.1 |
| 17 | **Behavioral tests for Shopify mappers + un-skip golden journeys** (F091); catalog re-sync strip-then-remerge (stop FBT wipe) | MEDIUM | Client-facing fabrication + integrity | S–M | 5.3 |
| 18 | **Retire legacy workflow engine** (Composio-reachable, ungoverned) + delete dead `/api/playbooks` raw-SQL router | MEDIUM | Ungoverned exec | M | 4.3 |
| 19 | **Chat user-identity** — thread `ctx.user` into `chats.user_id`/attribution (`api/chat.py:82-89`) | MEDIUM | Intra-tenant history/attribution | S | 3.2.f |
| 20 | **Drop credential prod→dev fallback + FutureAGI localhost default** (`manager.py:135-363`, `config.py:809`) | LOW-MED | Config footguns | S | 6.2, 6.3 |
| 21 | **Audit retention + operator surface + GDPR subject-tags** (after #2) | MEDIUM | Compliance | M | 2.3 |
| 22 | **Per-session sandbox for code-canvas** (E2B/Modal/Firecracker) + approval-audit persistence — before the lane runs client code | MEDIUM (forward) | Compute isolation | L | 4.2 |
| 23 | **Kill the workspace-blind agent-tool RBAC fossil** (`api/permissions.py` + tables) | LOW | Misleading surface | S | 3.2.e |

**Sequencing:** #1–#3 are the "close the untrusted edges" cluster and should land first and together. #2 (+#10) are enablers that make many other items real. #4–#8 are small, high-value hardening that can land in parallel. Everything below #12 is important-but-not-urgent given the pilot's current low real-traffic — but #14/#15 (channels + widget) must precede those surfaces going hot with real customers.

---

*Prepared as the dedicated Opus security & defensive-hardening pass. All findings are defensive hardening recommendations for the owner's own platform; every code claim cites `file:line` in the pinned tree `77bc9c6d5`; the top-5 (webhook fall-through, widget origin bypass, CORS default-open, credential NULL-workspace, random-vector fallback, policy flag) plus the skill-attach and `/api/tasks` findings were spot-verified verbatim in that tree during this pass. Cross-references to the source dossiers are named inline. This appendix appends to the Phase-2 final report per the brief.*
