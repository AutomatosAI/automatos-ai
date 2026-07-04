# Governance / Policy / Audit Plane — Module Dossier

**Module key:** `governance-policy` · **Tier:** deep · **Status:** partial (merged but dark-launched)
**Pinned tree:** origin/main @ `77bc9c6d5` (2026-07-03). All `file:line` refer to that tree.
**Baseline:** `reports/PLATFORM_OS_REVIEW_2026-07-01.md` (§12.4, F040/F059/F060/F068/F085/F086) + the residual map (`reports/dossiers/evidence/phase0-residual-map.md`).
**Real data:** `reports/dossiers/evidence/real-data-inventory.md`, `data/census.md`, `data/rag-feedback.md`.

> **Scope note:** this dossier covers sections A–E and G–J. Section F (enterprise-bar) and the defensive-hardening / adversarial-input / tenant-isolation lens are deliberately **out of scope** — they run as a separate Opus pass per the brief. Where a robustness fact is load-bearing for a *capability* judgement (e.g. "the gate fails open, so it isn't load-bearing"), it appears here as a functional finding, not a security recommendation.

---

## A. What it is

The governance-policy plane is the module that decides **what an agent is allowed to actually execute**, records that it happened, and provides the human-in-the-loop and compliance surfaces around it. Concretely it is three things stitched together:

1. **The W4 unified policy plane** (`orchestrator/modules/policy/`, PRD-174) — one typed `PolicyGate.check()` chokepoint that folds four guardrails (super-admin gate, caller-admin gate, model-aware budget admission, act-vs-ask routing under a per-workspace posture) into a single `Verdict` with `deny > ask > allow` precedence, plus an in-process typed event bus that fires an Art.12 audit handler on every verdict.
2. **The W11 durable-approval + compliance layer** (PRD-181) — a first-class `approval_grants` table (scoped / expiring / revocable) that gives *non-chat* agents (board tasks, playbook runs) a real approval workflow instead of a hard block, a GDPR export/erase cascade across SQL + Qdrant + mem0, and an EU-AI-Act Art.14 oversight-tier mapping.
3. **The older scattered gates it was meant to fold** — the mission approval-policy engine (`core/services/approval_policy.py`), the board approval gate (`services/board_approval.py`), the dollar-ceiling banding (`services/budget_ceiling.py`), the blueprint validator (`services/blueprint_validator.py`), and the per-router admin/super-admin checks.

The design intent is strong and correct: *"Auto's blueprint says what to attempt; this plane, and only this plane, decides what executes"* (`modules/policy/__init__.py:4-6`). The honest reality is that **almost none of it is switched on**: the entire W4 behavioural surface sits behind `AUTOMATOS_POLICY_PLANE`, which defaults OFF and is set nowhere in any deploy surface. What is live by default is exactly one gate (the board approval gate) — and its cost logic is vacuous.

---

## B. What it does — real implementation & data path

### B.1 The typed chokepoint (`PolicyGate`)

`PolicyGate.check(ToolCall)` (`modules/policy/gate.py:82-118`) evaluates in strict order, first-blocking-wins:

- **Super-admin gate** (`gate.py:89-102`) — an action flagged `super_admin_only` requires a literal `system_role == 'super_admin'` caller (via `roles.is_super_admin`, `roles.py:58-60`); otherwise `Verdict.deny`.
- **Admin gate** (`gate.py:106-160`) — an `admin_only` action requires the *caller's own* admin/owner role. This is the F014 fix: with the default-OFF `agents_inherit_admin` policy, there is **no** "the workspace has an admin, so the agent inherits it" flip. A no-identity caller (heartbeat / agent factory) is denied unless the workspace explicitly opts in.
- **Budget admission** (`gate.py:162-190` → `budget.check_budget`, `budget.py:159-216`) — reads `Workspace.plan_limits.budget` (a JSONB sub-key, no new column), sums spend-to-date from `llm_usage` over a rolling day/month/all window, adds the *projected* cost of the pending call, and denies if it crosses `max_cost_usd` or `max_total_tokens`. No budget key ⇒ inert allow.
- **Act-vs-ask routing** (`gate.py:192-225`) — the pure classifier `policy_document.classify_action` (`policy_document.py:215-249`) buckets the tool into `read / internal_write / external_side_effect / destructive / publish`; the per-workspace `PolicyDocument.route_for` (`policy_document.py:97-107`) maps that risk to `auto` or `ask` under the locked **Balanced** posture (`policy_document.py:50-56`). `ask` returns `Verdict.ask` with a `PolicyError` the model reads as tool content.

The verdict vocabulary (`types.py`) is the genuinely good part: `Verdict` mirrors Claude Code's hook verdict, re-keyed for tenancy, with `merge_verdicts` enforcing `deny > ask > allow` via an explicit rank table (`types.py:35-40`, `178-223`), and `PolicyError` is structured errors-as-data (`code / message_for_model / remediation / retryable`, `types.py:59-85`) so a denial is something Auto can *adapt to*, not an opaque failure.

### B.2 Where it attaches

One chokepoint, in the unified executor: `UnifiedToolExecutor._policy_gate_check` (`modules/tools/execution/unified_executor.py:204-280`), called at `unified_executor.py:395`. It resolves the `platform_execute` meta-tool's nested action (`:236-244`), runs `PolicyGate(self.db).check(...)` (`:245`), fires the policy bus for **every** verdict (`:256-263`), and returns `verdict_to_result` (errors-as-data, `errors.py:26-53`) on a block so the tool never runs. There is a second, lighter seam in the stdlib-only tool loop: `on_pre_tool` (`modules/tools/execution/tool_loop.py:396-420`) — a hook point beside dedup, currently the attach point the executor uses.

### B.3 The event bus + Art.12 audit

`PolicyBus` (`bus.py:55-87`) is an **in-process typed dispatcher** — deliberately *not* shell hooks (`bus.py:4-6`: "Bash-string hooks are RCE-by-configuration and don't fit a SaaS backend"). Handlers are `(event, ctx) -> Optional[Verdict]`; a raising handler is treated as no-opinion (`bus.py:79-86`). The one registered handler is `register_audit_handler` (`audit_handler.py:121-141`), which on every `PRE_TOOL_USE` writes one `AuditLog` row (`audit_handler.py:62-115`) via `AuditService(db).log(...)` recording tool, tenant, actor (`_resolve_actor` at `:41-59` distinguishes user/agent/system), verdict, reason, risk tier, and error code. Registration is **flag-gated**: `main.py:519-522` only calls `register_audit_handler()` when `policy_plane_enabled()`.

### B.4 Durable approval grants (W11 / the one live gate)

`ApprovalGrant` (`core/models/approval_grants.py`) is a scoped (`workspace + subject_type + subject_id + tool_name + risk_tier`), expiring (`expires_at`), revocable (`revoked_at`) row. The board path is wired **unconditionally** (not behind the flag): `_board_task_blocked_pending_approval` (`api/board_tasks.py:943-994`) runs before `_launch_task_execution` executes (`board_tasks.py:1017-1023`), calling `evaluate_board_task_approval` (`services/board_approval.py:63-147`), which reuses the mission `evaluate_approval` primitive (`core/services/approval_policy.py`), creates a durable grant on `ask`, and blocks the task. The grant API (`api/approval_grants.py`, mounted `router_manifest.py:78`) grants/denies/revokes and re-queues the blocked task via `notify_task_available` (`approval_grants.py:142-160`). The dispatcher routes through the same launcher.

### B.5 GDPR + EU-AI-Act

- **GDPR** (`api/gdpr.py` → `services/gdpr_service.py`): `/api/v1/gdpr/export|erase|erase-subject`, admin-gated (`gdpr.py:33-37`), whole-workspace erase requires a confirmation echo (`gdpr.py:64-70`). The cascade is real code across SQL tables, Qdrant field memory (`VectorFieldSharedContext.erase_workspace`), and mem0 per-namespace deletes (`gdpr_service.py:9-16, 219-241`), with honest `gaps` reporting where a store carries no subject tag (`gdpr_service.py:20-22, 170-181`).
- **EU-AI-Act** (`modules/policy/ai_act.py`): a pure mapping from risk class → `OversightTier` (MONITOR / HUMAN_ON_THE_LOOP / HUMAN_IN_THE_LOOP) with plain-language rationale for an approval card. Self-labelled a **scaffold** (`ai_act.py:10-12`); the formal Annex-IV file is a flagged fast-follow.

### B.6 The real-data path — what actually happened

This is where the module's story diverges hardest from the code's self-description:

| Surface | Row count (prod, 2026-07-04) | What it means |
|---|---|---|
| `approval_grants` | **0** (`census.md:43`, `rag-feedback.md:25`) | The one live gate has never created a grant in production. |
| `audit_logs` (policy verdicts) | **effectively 0** | The audit handler only registers under the flag (`main.py:521`); flag is OFF everywhere ⇒ the plane has never written a policy-verdict audit row. (`skill_audit_log` = 6 and `memory_access_log` = 6 are unrelated surfaces.) |
| `harness_prescriptions` | **0** (`census.md:58`) | The governed-actuation path (HARNESS `/approve` through the plane) has never run. |
| `llm_usage` | 31,081, fresh 07-03 (`census.md:41`) | The budget gate's *input* ledger is real and current — the gate would have data to price against, if it were on and if callers threaded estimates. |
| Frontend callers of `approval-grants` / `gdpr` / `policy_plane` | **0** (grep, source tree) | No human-in-the-loop or governance UI exists. The only "audit" UI is `CredentialAuditTab` (credentials, unrelated). |

The plane is, in the literal sense the data shows, **scaffolding that has never carried load in production.**

---

## C. Honest quality — how good is it *really*?

### Maturity score: **2 / 5** (architecture would earn 4; deployed reality drags it to 2)

**Justification.** The *design* is the best-thought-through governance layer in the platform — typed verdicts, one merge rule, errors-as-data, an in-process bus that consciously rejects shell-hook RCE, a DB-configured per-workspace posture, and a durable tool-agnostic grant model that correctly justifies its new table. If this were on and complete it would be a 4. But a governance plane is judged by what it *enforces*, and by that measure the deployed reality is a 2: the one gate that runs by default is cost-blind, the rest is dark, no caller feeds the budget gate, several execution lanes bypass the single chokepoint, and there is no operator surface. Governance you cannot see and does not fire is not yet governance.

### The concrete defects, with evidence

**C.1 — The entire W4 plane is off by default and set nowhere.** `POLICY_PLANE_ENABLED` defaults false (`config.py:645`); the residual map verified nothing in `envs/`, `docker-compose.yml`, `railway.json`, or the Dockerfile sets it, and `flag.py:24-32` fails safe to OFF. With the flag off, `_policy_gate_check` returns `None` immediately (`unified_executor.py:227-228`) — byte-for-byte the legacy per-router gates. So on every real deployment today: the rate limiter is inert (F040), the admin gate is the old workspace-membership flip (F014), the widget empty-permission god-key stands (F042), and there is **zero** budget admission. *(Data corroborates: 0 policy audit rows ever.)*

**C.2 — Even ON, the gate fails OPEN on any internal error.** `unified_executor.py:271-280`: any exception in the plane is logged and treated as "proceed". A governance gate whose failure mode is *allow* is a monitoring aid, not an enforcement boundary — a malformed policy doc, a DB blip, or a registry miss silently waves the call through. The board gate has the same posture (`board_tasks.py:989-994`), and so does the budget/document read (`budget.py:111-116`, `policy_document.py:139-144`). Fail-open is a defensible *availability* choice, but it means the plane cannot be relied on for anything that must not happen.

**C.3 — Budget admission prices nothing on the real paths.** The model-aware pricing source (`pricing.py:25-63`) reads the `llm_models` registry and never guesses — good. But `ToolCall.model_id / est_input_tokens / est_output_tokens` are optional (`gate.py:66-71`), and **no caller in the tree passes them** (the executor constructs `ToolCall` without them, `unified_executor.py:245-254`). So `projected_cost` is always 0 (`gate.py:163-168`); the budget gate can only catch a workspace that is *already* over its ceiling from prior spend, never the call in front of it. Meanwhile every governance dollar figure that *does* get computed — mission auto-approve (`coordinator_service.py:2424-2428`), mission bands (`dispatcher.py:452,457`), playbook ceiling (`recipe_executor.py:1002`) — uses the flat `COORDINATOR_COST_PER_1K_TOKENS = 0.003` rate (`config.py:721`), model-blind. F059's "one pricing source" landed as a *fifth* pricing source consumed only by the flag-gated gate. Pricing complexity went **up**, not down.

**C.4 — The "single chokepoint" is not single.** The residual map confirmed four lanes execute tools *around* the unified executor: the chat per-action Composio shortcut (`consumers/chatbot/service.py:1321-1334, 1550-1565`), Playbook Composio steps (`api/recipe_executor.py:655`), widget email actions (`api/widget_email.py:286,340,388,437`), and the `/api/tasks` direct-step lane which runs raw shell/git to the worker with auth-only checks (`api/tasks.py:62-139`). External side-effects — precisely the `ask` class the Balanced posture exists to gate — are the ones most likely to slip the gate.

**C.5 — The board dollar-ceiling is vacuous.** The one live gate never passes `estimated_cost_usd` (`board_tasks.py` calls `evaluate_board_task_approval` without it, so it defaults 0.0 at `board_approval.py:68`). Under `auto_below_budget` a 0.0-cost task is always at/below any positive ceiling ⇒ auto-approved. So the ceiling never fires; only an `always_ask` workspace ever produces a grant. Combined with 0 grants in prod, the enforcement that *is* wired is effectively cosmetic on default (`always_ask` is the default policy, so it would fire — but no workspace has exercised it).

**C.6 — Playbooks have a ceiling but no ask-gate; scheduled/webhook agents have neither.** `SUBJECT_PLAYBOOK_RUN` exists in the model (`approval_grants.py:54`) but has zero non-model references — no code ever creates a playbook approval grant. The playbook ceiling (`recipe_executor.py:1184-1208`) is opt-in (absent `cost_ceiling` ⇒ unlimited) and flat-priced. Scheduled/webhook agents are explicitly future work (`board_approval.py:12`; no approval/budget references in `scheduled_task_service.py`). So of the four non-chat surfaces the grant model was built to cover, one (board) is wired-but-cost-blind and three are ungoverned.

**C.7 — The rate limiter, even ON, is per-process and in-memory.** `main.py:827-838` constructs `Limiter(...)` with no `storage_uri`, so slowapi uses its default in-memory backend. Under the multi-worker Railway deploy the "60/minute" limit is per-worker, not per-tenant-cluster — a soft, evadable limit even after the flag flips. And there are no per-route `@limiter.limit` decorators anywhere, so only the global default applies. The F040 regression test grep-asserts the middleware string in source (`test_prd174_flag_gating.py:107-113`), not an actual rate-limit event.

**C.8 — No operator surface at all.** Zero frontend callers for approval grants, GDPR, oversight, or policy-plane config. A workspace admin cannot see the audit log, cannot see or action a pending approval, cannot set a budget or a posture, and cannot fire a GDPR export — all of it is API-only. For a plane whose entire Art.14 premise is *"a human can effectively oversee"*, the absence of the human surface is the single biggest gap against the North Star.

**C.9 — GDPR cascade is real but never exercised against the stores that matter.** The SQL leg is sound, but the Qdrant field-memory and mem0 legs (`gdpr_service.py:129-143`) target stores that were **unreachable from this environment** (mem0 host 404s, no Qdrant creds — `real-data-inventory.md:10-11`), and `erase-subject` depends on subject tags that field-memory/mem0 do not currently carry (honestly reported as `gaps`). It is correct-looking code with no production evidence it deletes what it claims, and a subject-level erase that mostly returns gaps is not yet a defensible GDPR answer.

### What is genuinely good (honest positives)

- The **verdict/error model** (`types.py`, `errors.py`) is clean, well-documented, and reusable; `deny > ask > allow` with a rank table is exactly right, and errors-as-data is a real capability lift for an autonomous agent.
- The **in-process bus rejecting shell hooks** (`bus.py:4-6`) is a mature, security-aware call that many "we ported Claude Code hooks" implementations get wrong.
- The **F014 admin fix** (caller's own role, no workspace-has-an-admin flip) and the **F042 single empty-permissions-deny semantic** (`roles.py:63-77`) are correct least-privilege closures — *when the flag is on*.
- The **board approval gate is the one place agents actually hit a durable, revocable, audited human-in-the-loop** — the right primitive, just cost-blind.
- **Fail-safe defaults are consistent and deliberate** (unknown risk ⇒ ask, corrupt policy ⇒ Balanced, unpriceable ⇒ don't guess). The *enforcement* fails open; the *classification* fails safe.

---

## D. Competitive teardown

The plane spans two markets that competitors keep separate: **runtime authorization/policy engines** (does this action execute?) and **AI-governance suites** (is the system compliant, documented, auditable?). Automatos tries to be both. Judged against best-in-class in each:

### D.1 Runtime policy / authorization engines

**Oso (in-process, Polar).** Oso is the closest architectural match: it embeds *in-process*, compiling Polar policies to bytecode for sub-millisecond decisions with zero serialization overhead, which is exactly the shape of `PolicyGate` — no sidecar, no network fan-out. Where Oso beats Automatos: (a) policy is a **declarative language** with tooling, testing, and a policy-authoring UI, versus Automatos's hardcoded Python routing tables (`policy_document.py:50-75`) and taxonomy sets that require a **deploy** to change; (b) Oso Cloud gives **centralized policy management, a decision log, and "explain" tooling**; (c) fine-grained relationship/ABAC modelling out of the box. Automatos's `route_overrides` per-workspace (`policy_document.py:83-85`) is a thin slice of what Polar expresses. Source: [osohq.com — Oso embeds in-process](https://www.osohq.com/learn/cerbos-alternatives-for-authorization), [johal.in — Oso Polar embeddable 2025](https://johal.in/oso-python-polar-authorization-policy-language-embeddable-2025/).

**Open Policy Agent (OPA) / Cerbos (sidecar PDPs).** OPA has become a standard "missing guardrail" for AI agents: a centralized decision point that reasons across tool-access + resource-access + command-authorization **in a single query** before every action, with Rego decoupling policy from enforcement. Cerbos runs as a stateless sidecar/service with sub-millisecond evaluation and SDKs for six languages. Where they beat Automatos: policy-as-code with full test/version/CI tooling, a decision-log stream, and a battle-tested Rego/Cerbos-YAML ecosystem. Where **Automatos is actually better**: it does *not* pay OPA/Cerbos's network-hop and state-sync cost, and its `PolicyGate` is already tenant-scoped and beside the tool loop — the sidecar model is a worse fit for a Python modular monolith than the embedded model. So the honest read is "adopt Oso's *model* (declarative + decision log + management UI), not OPA's *topology*." Source: [codilime — OPA as AI-agent guardrail](https://codilime.com/blog/why-use-open-policy-agent-for-your-ai-agents/), [cerbos.dev — Cerbos vs OPA](https://www.cerbos.dev/blog/cerbos-vs-opa), [openpolicyagent.org/docs](https://www.openpolicyagent.org/docs).

### D.2 LLM guardrails / observability planes

**Langfuse / LangSmith.** Langfuse is explicitly a **monitoring-and-evaluation** plane that traces the *effectiveness* of dedicated guardrail libraries (LLM Guard, NeMo Guardrails, Azure AI Content Safety, Prompt Armor) rather than enforcing itself. Where it beats Automatos: a real **decision/trace dashboard**, security-score trending, and incident investigation UI — the operator surface Automatos entirely lacks (C.8). Where Automatos is different (and arguably ahead of a pure-observability tool): its plane *enforces* at the execution boundary, not just observes. The lesson is that Automatos has the enforcement half Langfuse doesn't and lacks the visibility half Langfuse is built around. Source: [langfuse.com — security & guardrails](https://langfuse.com/docs/security-and-guardrails).

### D.3 AI-governance suites (the "governance-as-product" comparison)

**Credo AI / IBM watsonx.governance.** Credo AI (Forrester Wave leader Q3 2025) is **policy-first**: regulatory mapping to EU AI Act, ISO/IEC 42001, and NIST AI RMF via ready-made Policy Packs — but the analyst read is blunt: *Credo AI documents governance requirements but does not enforce them at the execution layer.* IBM watsonx.governance adds lifecycle monitoring, bias/explainability, model-behaviour tracking, and FedRAMP authorization. Where they beat Automatos: mature **compliance reporting, regulatory templates, model registry/lifecycle, and an evidence surface** for auditors. Where **Automatos is structurally the opposite and better-positioned**: it enforces *at the execution layer* (the exact thing Credo AI does not), and its `ai_act.py` Art.14 mapping + Art.12 audit handler are the enforcement-side hooks a Credo AI integration would sit *on top of*. Automatos's weakness vs these suites is precisely the reporting/documentation product surface — the Annex-IV file it flagged as fast-follow. Source: [slashdot — Credo AI vs IBM watsonx.governance 2025](https://slashdot.org/software/comparison/Credo-AI-vs-IBM-watsonx.governance/), [truefoundry — best AI governance tools 2026](https://www.truefoundry.com/blog/best-ai-governance-tools).

### D.4 Where Automatos actually stands

- **Enforcement architecture:** competitive-to-good. In-process, tenant-scoped, typed, errors-as-data — the right shape, matching Oso's embedded model and beating the sidecar planes on fit. But it is *inert by default*, so the real-world enforcement posture is behind everyone.
- **Policy expressiveness:** behind. Hardcoded Python tables + deploy-coupled taxonomy vs Oso/OPA/Cerbos declarative languages with tooling.
- **Operator/visibility surface:** far behind. Langfuse and the governance suites are *built around* the dashboard/report Automatos has none of.
- **Compliance product:** behind on documentation (Credo AI's Policy Packs / Annex-IV), ahead on execution-layer enforcement hooks.

Net: Automatos has the **enforcement seam** the observability and governance suites lack, and lacks the **policy language + operator surface + compliance reporting** they lead with. It is not behind because the architecture is wrong; it is behind because the plane is off, cost-blind, and headless.

---

## E. Build / extend / adopt / replace — the verdict

### Verdict: **EXTEND** (finish and turn on what's built) + **selectively ADOPT** a declarative policy engine for the routing/authorization core.

Do **not** replace the plane — the enforcement seam, the tenant-scoped chokepoint, the verdict/error model, and the durable-grant table are correct and hard-won, and no external product gives you an *in-tenant execution-boundary* gate for a Python monolith out of the box (the sidecar PDPs are a worse topological fit, and the governance suites explicitly don't enforce at execution). The dominant work is **extension**, not new construction, and it is squarely the CLAUDE.md "rehouse/finish, don't rebuild" case.

**Extend (the bulk — earns its keep because the scaffolding is already merged):**
1. **Turn the flag on, behind a staged rollout** — the plane is byte-for-byte reversible by design (`main.py:834-836`), so the risk is bounded. This is the single highest-leverage governance change and it is a config decision, not code.
2. **Thread token/model estimates into `ToolCall`** so budget admission actually prices the pending call (C.3) — the estimate already exists at the LLM-manager boundary.
3. **Close the chokepoint bypasses** (C.4) — route the four Composio/direct-step lanes through `_policy_gate_check` (or delete the direct-step lane, which the `tool-runtime` dossier also flags).
4. **Pass real `estimated_cost_usd` into the board gate** (C.5) and collapse the flat-rate dollar paths onto `pricing.py` so there is genuinely one pricing source (finish F059).
5. **Decide fail-open vs fail-closed per risk class** (C.2) — reads can fail open; destructive/external should fail closed. This is a one-branch change at `unified_executor.py:275` gated on the classified risk.
6. **Build the operator surface** (C.8) — an approvals queue, an audit-log view, a per-workspace posture/budget editor, a GDPR-request button. This is the North-Star-critical piece and is pure frontend on APIs that already exist.

**Adopt (the authorization/policy core, medium-term, per the §2 reuse bias):**
- **Oso (open-source, in-process Polar)** to replace the hardcoded routing tables and deploy-coupled taxonomy (`policy_document.py:50-75`, `taxonomy.py:230-280`). Oso's embedded model matches `PolicyGate` exactly (no sidecar), gives declarative per-tenant policy + a decision log + "explain", and lets a workspace change what's `ask` vs `auto` **without a deploy** — the F072 deploy-coupling closes structurally. Integration shape: keep `PolicyGate.check()` as the façade and the four-stage order; swap the *classification + routing* internals for a Polar evaluation over a `ToolCall`/`PolicyDocument` fact set. The verdict/error/bus/audit/grant machinery stays. Cost: open-source core is free; Oso Cloud (managed decision log + policy UI) is a paid tier — evaluate once the plane is on and carrying load. This is the one place an external engine genuinely beats the in-house table.

**Do not adopt:** OPA/Cerbos (sidecar topology is a worse fit than the already-embedded gate); a full governance suite like Credo AI / watsonx.governance (they don't enforce at the execution layer, which is Automatos's actual strength — at most integrate later for compliance *reporting*, not enforcement).

**Kill-list:** nothing to delete outright here, but two consolidations — (a) collapse the five dollar-pricing paths onto `pricing.py`; (b) once the plane is on, retire the legacy per-router admin checks the plane's `roles.py` supersedes (F043), rather than running both.

---

## G. Quality metric — how to measure this module over time

A governance plane's quality is **coverage × correctness × visibility**. Concrete, trackable numbers (feeding T3):

1. **Enforcement coverage %** — of all tool-executing lanes (unified executor + the four bypasses + board/mission/playbook/scheduled/webhook), the fraction that pass through the plane. **Today: with flag OFF ≈ 0% (only the board approval gate, cost-blind). With flag ON, ≈ 5 of 9 lanes.** Target: 100% of external-side-effect and destructive lanes.
2. **Priced-call %** — fraction of budget-gate evaluations that receive a real model+token estimate rather than 0. **Today: 0%.** Target: 100% on LLM-bearing calls.
3. **Audit completeness** — `audit_logs` policy-verdict rows ÷ tool calls executed, per tenant. **Today: ~0** (handler unregistered under the OFF flag). Target: 1.0 (every executed call has a verdict row).
4. **Approval-loop liveness** — grants created / granted / denied / expired per week, and median time-to-decision. **Today: 0 grants ever** (`census.md:43`). Any non-zero is progress; the metric only becomes meaningful once the operator surface exists.
5. **Fail-open rate** — count of gate evaluations that hit the exception path and proceeded (`unified_executor.py:275`). Should be ~0 in steady state; a spike is a policy/DB fault, not a safe default.
6. **GDPR erase completeness** — for an `erase-subject`, the ratio of stores actually purged to stores claimed, i.e. `1 − gaps/total`. **Today: unmeasurable/likely low** (field-memory + mem0 carry no subject tags; stores unreachable to verify). Target: no `gaps` for any store that holds subject data.

None of these are instrumented on a dashboard today — which is itself the finding. Metrics 1–3 are computable immediately from existing tables the moment the flag flips.

---

## H. Cost note (informational)

- **Per-call gate overhead (flag ON):** cheap. `PolicyGate.check` is one workspace-settings read (`policy_document`) + one action-registry lookup + a `func.sum` over `llm_usage` for the budget window (`budget.py:143-149`) + the pure classifier. The `llm_usage` aggregate is the only non-trivial cost — one indexed SUM per gated call; it should be cached per (workspace, window) rather than recomputed every call under load. No LLM tokens are spent by the gate itself.
- **Audit write:** one INSERT per tool call under the flag (`audit_handler.py:99-108`), on its own short-lived session (`audit_handler.py:154-168`). At the platform's tool-call volume this is modest but grows `audit_logs` unboundedly — needs a retention policy (EU-AI-Act Art.12 mandates ≥6 months, which is a *floor*, so retention is a compliance requirement, not just housekeeping).
- **GDPR export/erase:** heavy but rare — full SQL table dumps + async Qdrant/mem0 traversals (`gdpr_service.py:44-65`). Correctly off the hot path.
- **Net:** the plane adds negligible token cost and small, cacheable DB cost. Cost is not a reason to keep it off.

---

## I. UX / surface — Command Center and beyond

The plane is **100% headless today** (C.8) — the largest single gap against the North Star, because Art.14 "effective human oversight" is definitionally a UI claim. Concrete surfaces to build, all on APIs that already exist:

1. **Approvals queue (Command Center).** A live list of pending `ApprovalGrant`s (`GET /api/v1/approval-grants?status=pending`) with the risk tier, the Art.14 oversight rationale (`ai_act.oversight_for_risk`), the estimated cost, and grant/deny/revoke buttons. This is the human-in-the-loop the whole W11 slice was built for and it has no front door. It should share the PRD-163 in-chat approval-card component so chat and non-chat approvals tell one story.
2. **Audit log view (per workspace).** A filterable table over `audit_logs` (tool, actor, verdict, reason, risk) — the operator's window into what the plane decided. Today only `CredentialAuditTab` exists, for an unrelated surface. This is also the auditor-facing evidence surface the governance suites lead with (D.3).
3. **Policy posture + budget editor.** A settings panel writing `workspace.settings.policy_plane` (posture, `agents_inherit_admin`, `route_overrides`) via `set_policy_document`, and `plan_limits.budget` (ceiling, window). Today posture and budget are only editable by hand-writing JSON. Surface the Balanced/Strict/Permissive choice and the per-risk override toggles.
4. **GDPR self-service.** An admin button for export / erase-subject / erase-workspace (with the confirmation echo the API already enforces), plus a visible `gaps` report so the operator knows what was *not* deleted.
5. **A "governance is on/off" indicator.** Given the flag is the single biggest determinant of behaviour, the Command Center should plainly show whether the policy plane is enforcing, so an operator is never falsely reassured. (This ties to the `observability-slos` "is-it-working" strip.)

IA principle: governance should be **one Command Center pillar** (Approvals · Audit · Policy · Compliance), not scattered into settings tabs — it is the surface that makes autonomy-with-guardrails legible to the client, which is the North Star's second half ("agents deliver quality work for real clients" requires clients to trust and steer them).

---

## J. Upgrade path — prioritised (impact × effort), judged by North-Star impact

Ranked by *(does this make Auto safely more autonomous / higher-quality for clients?)* over effort.

| # | Change | Impact | Effort | Why it matters (North Star) |
|---|---|---|---|---|
| **1** | **Staged rollout of `AUTOMATOS_POLICY_PLANE=on`** (default-off → per-env → default-on), with the fail-open→fail-closed-for-destructive branch (J-4) landed first. | **Very high** | Low (config + one branch) | Nothing else in this dossier matters while the plane is off. Turning it on is what converts "scaffolding" into "guardrails," which is the precondition for trusting Auto with external side-effects. Reversible by design. |
| **2** | **Build the Approvals queue + Audit view + governance indicator** (I.1, I.2, I.5). | **Very high** | Medium (frontend on existing APIs) | Art.14 oversight is a UI claim; without a human surface the durable-grant machinery is unreachable. This is the single biggest client-trust lever and unblocks the whole W11 investment. |
| **3** | **Thread model+token estimates into `ToolCall`; collapse the 5 flat-rate dollar paths onto `pricing.py`** (C.3, finish F059). | High | Medium | Makes budget admission real (pre-call, model-aware) instead of only catching already-over-budget tenants; gives Auto a truthful cost ceiling so autonomy doesn't silently burn spend. |
| **4** | **Fail-closed for destructive/external risk classes** at the gate exception path (C.2). | High | Low (one risk-gated branch at `unified_executor.py:275`) | A governance gate that fails open cannot be relied on for the exact class of actions (deletes, sends, refunds) it exists to gate. Small change, disproportionate correctness gain. |
| **5** | **Close the four chokepoint bypasses** (C.4): route chat-Composio, playbook-Composio, widget-email, and `/api/tasks` through the gate (or delete the direct-step lane). | High | Medium | External side-effects are the highest-risk lane and the ones currently most able to skip the gate — closing them is what makes "one plane governs everything" true rather than aspirational. |
| **6** | **Wire the playbook + scheduled/webhook approval/budget gates** (C.6) using the existing `ApprovalGrant` model and `SUBJECT_PLAYBOOK_RUN`. | Medium-high | Medium | Extends the one working primitive (durable grants) to the three non-chat surfaces it was designed for, so *all* autonomous execution — not just board tasks — is governable. |
| **7** | **Pass real `estimated_cost_usd` to the board gate + make the ceiling bind** (C.5). | Medium | Low | Turns the one live gate from cost-blind to cost-aware; small fix, closes the "ceiling never fires" gap. |
| **8** | **Adopt Oso (Polar) for the routing/classification core** (E), keeping `PolicyGate` as the façade; retire the deploy-coupled taxonomy tables (F072). | Medium (high long-term) | High | Removes deploy-coupling so workspaces tune `ask`/`auto` policy live, adds a decision log + "explain," and replaces hand-rolled tables with a battle-tested engine — the one place an external engine genuinely beats the in-house build. Do after the plane is on and carrying load. |
| **9** | **Instrument metrics G.1–G.6 on a governance dashboard**; add a `retention` job for `audit_logs` (≥6-month EU-AI-Act floor). | Medium | Medium | Makes governance quality a tracked number (T3) rather than a vibe, and turns Art.12 record-keeping from "writes rows" into "retains and surfaces them" — the compliance floor. |
| **10** | **Verify + harden the GDPR subject-erase cascade** against live Qdrant/mem0; add subject tags so `erase-subject` stops returning mostly-`gaps` (C.9). | Medium | Medium | A GDPR answer that can't prove it deleted from the memory stores isn't a defensible one; needed before the vertical scales to real customer PII. |

**Dependency order:** #4 → #1 (fail-closed branch before flipping on) → #2 and #3 in parallel → #5/#6/#7 → #8 → #9/#10.

---

### One-line status
`governance-policy: 2/5, extend, best-architected governance plane in the platform but off by default, cost-blind, and headless — turn it on, price it, and give it a human surface.`
