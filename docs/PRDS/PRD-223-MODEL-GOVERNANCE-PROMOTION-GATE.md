# PRD-223 — Auto Is a Governed Role: Model Governance & the Promotion Gate

**Status:** DRAFT — for Gerard's review
**Date:** 2026-07-31
**Grounded @:** `main c1cce09ab`
**Evidence base:** 2026-07-31 incident (Auto on `openai/gpt-5.6-sol-pro`) — Railway production logs, plus a full code trace of the model-configuration surface, the eval/scoring infrastructure, and the attachment/grounding path. Every load-bearing claim in §3 carries a file:line ref verified at the grounded SHA.
**Depends on (live):** PRD-136 LLM tiers (`orchestrator_llm`/`system_llm`/`embeddings`), PRD-138/139 tool-routing eval harness, PRD-127 attachments, PRD-141 semantic tool routing.
**Relates to:** Tool-surface deep review 2026-07-23 (§7 gating decisions — Wave 1 here supersedes the "expose everything" default that review flagged).

---

## 1. Overview

On 2026-07-31 Auto's model was switched from the trusted 5.5 model to `openai/gpt-5.6-sol-pro` through the Settings → Orchestrator tab. The result: every technical-sounding message became a research spree — one observed turn ran **7+ LLM iterations × 4 parallel tool calls = 24 tool executions** (hitting the per-tool hard cap), re-sent 130–174k input tokens per iteration, cost ~$2.70, took 90+ seconds, and fired three `COST_ALERT`s. Separately, the model **claimed source access it did not have** (asserting it had read attached/ referenced content that was never in its context).

The incident is 5.6. **The problem is the platform**: any string can occupy Auto's chair, through a route with zero validation, and the only constraints on what the occupant does are runaway backstops. Two models given identical prompts and tools do not behave identically — they differ on tool eagerness, willingness to infer, honesty under uncertainty, source discipline, ability to stop, and recovery after mistakes. For a chatbot that's annoying. For Auto — which holds workspace authority, routes work, interprets documents, and calls tools — it's dangerous.

This PRD makes **Auto a governed role, not a model name**: models are workers; Automatos governs what each worker is allowed to do, verifies its behaviour before promotion, and keeps authority bounded. That is also the product story — customers face the same "which model can I trust with which job?" question, and Automatos should answer it structurally instead of pretending models are interchangeable.

---

## 2. Decision record (2026-07-31, Gerard — do not relitigate)

| # | Decision |
|---|---|
| D1 | **Auto is a governed role.** No model becomes Auto because it is newer or smarter. Occupying the orchestrator seat requires passing a promotion gate. |
| D2 | **5.6 is quarantined, not deleted.** `openai/gpt-5.6-sol-pro` is not approved for the primary Auto/orchestrator role. It remains available for explicit experiments and narrow non-tool roles. |
| D3 | **Auto stays on the trusted model** until the gate exists and a candidate passes it. |
| D4 | **Model roles, not generic swapping** — orchestrator (highest trust), research (explores, must cite), drafting (writes, no tools), coding (task-scoped), background (low-risk summaries/classification), experimental (quarantined). |
| D5 | **The source-grounding contract is tested behaviour, not prompt preference.** Auto may not claim "I read the file / the PRD says / the logs show" unless the content was in context, tool-fetched, user-pasted, or system-supplied. The only honest alternative is "I can see the reference, but I don't have the contents." |
| D6 | **A model promotion harness gates upgrades** — attachment honesty, tool restraint, uncertainty handling, injection resistance, fabrication refusal, correction behaviour. Failing models can still serve lower-trust roles. No new model tests in production again. |

---

## 3. Current reality (grounded)

### 3.1 The chair has no lock

- **The route that changed Auto's model validates nothing.** `PUT /api/workspaces/current/orchestrator` (`orchestrator/api/workspaces.py:439`, sync block `:540-576`) dual-writes Auto's `Agent.model_config` and `system_settings(orchestrator_llm.*)` with **no model validation whatsoever** — unlike `PUT /api/agents/{id}/model-config` (`api/agent_endpoints.py:538`), which at least checks the OpenRouter cache for existence (`:573-587`). Existence is the *only* check anywhere; approval does not exist as a concept.
- **Three writer paths, all ungoverned:** the workspaces route above; the per-agent route (existence-only); and `handlers_agents.py:204-232`, where **Auto can change its own model via a chat tool** with the provider inferred by substring match and no catalog validation at all.
- **No approval, no role, no profile on any model table.** `LLMModel.requires_plan` (`core/models/core.py:93`) is a dead gating column — seeded, rendered as a UI badge, enforced nowhere. `Persona.suggested_models` (`core/models/personas.py:41`) is advisory, read by no enforcement path. The nearest precedents are `Agent.required_role` (`core.py:251`) and the `SYSTEM_BYPASS_ALLOWLIST` frozenset (`core/security/hierarchy_permissions.py:68`).
- **Runtime resolution** converges at `AgentFactory.activate_agent` (`modules/agents/factory/agent_factory.py:692-765`): power-mode tier forcing → orchestrator tier → agent `model_config` → settings default. This is the single last-mile point every path passes through — and it enforces nothing.

### 3.2 Guardrails are runaway backstops, not behaviour shaping

- Per-tool caps are a hardcoded class dict (`consumers/chatbot/service.py:173-185`, values 2–8, prefix defaults). The turn cap `chatbot.max_tool_iterations` is a DB system_setting **seeded at 25** (`core/seeds/seed_system_settings.py:562-600`) — a mission-scale budget applied to chat.
- Nothing scales any budget by intent, risk, or model trust. Once the intent classifier attaches tools, the full budget is available regardless of whether the user asked a question or is venting.
- `COST_ALERT` (`llm.cost_audit`) is a log line. It watched a $2.70 turn happen and did nothing.
- The `chatbot` settings category (and `recipe`, `agent_heartbeat`, `coordinator`, `power_modes`) has **no UI tab** (`frontend/components/settings/SystemSettingsTab.tsx:193-206`) despite `config.py:378` claiming UI editability — the turn budget is editable only by raw API or reseed.

### 3.3 The platform engineers the fabrication opportunity

- **Attachment fail-open:** when attachment resolution fails, `ContextService.build_context` logs and **continues silently** (`modules/context/service.py:196-200`). The model receives a message referencing a file with no indication the content is missing; the user believes Auto has it. An overconfident model fabricates ("the PRD says…"); a tool-eager one goes spelunking — the observed 24-call spree was Auto grepping the workspace for `attachment_id`. Only vision-unsupported errors re-raise (`:193-195`); text attachments vanish without trace.
- No evidence discipline exists anywhere in the response path — nothing distinguishes observed / fetched / inferred / unknown.

### 3.4 Evaluation infrastructure: a real skeleton, four gaps

- `scripts/eval/tool_routing/` (PRD-138/139) is a genuine **model-comparison harness**: model matrix with tier + cost (`models.yaml`), resumable cartesian runner (`run_eval.py`), per-category scoring + cost-per-correct economics (`score.py:108-114`), and snapshot promotion to date-stamped `benchmarks/` (`snapshot.py`). Proven across 9 models / 1,224 result rows.
- **Gap 1 — restraint is unmeasurable:** the runner hardcodes `tool_choice="required"` (`run_eval.py:270`); "correctly declining to call a tool" cannot be expressed. No abstain rows exist in the gold set.
- **Gap 2 — no verdict:** `score.py` reports, never decides. The house verdict idiom exists in `evals/graphiti_vs_baseline.py` / `operating_graph_uplift.py` (explicit margin constant, `PENDING` over false-green, exit 0 always).
- **Gap 3 — single scorer:** routing scores an action name; honesty/grounding categories need pluggable judges. The cleanest generation-agnostic seam in the repo is `tests/nl2sql_eval/harness.py:110` (`evaluate(generate_fn)`).
- **Gap 4 — gold-set safety:** real-tenant gold sets under `scripts/eval/retrieval_recall/live/` (real workspace UUIDs, InBuild UK pilot corpus) were untracked **by luck** — no ignore rule. **Closed 2026-07-31**: root `.gitignore` now carries `orchestrator/scripts/eval/**/live/` + `*.gold.jsonl`.
- Zero coverage anywhere for attachment honesty, tool restraint, or hallucination/grounding. `futureagi_service` is an online per-turn telemetry sampler (rubric lives in an out-of-repo worker) and additionally mis-attributes chat-turn scores to every eval-enabled prompt slug — **not** a foundation for the gate (see §9 defects).

---

## 4. Goals

1. No model can occupy the orchestrator seat without an explicit, recorded approval for that role. Fail closed.
2. Every model-write path (workspaces route, per-agent route, chat tool) enforces the same policy at one chokepoint. No bypass doors.
3. A promotion harness produces a per-role verdict for a candidate model across the six behavioural categories in D6, with archived evidence, before approval is granted.
4. The source-grounding contract holds at runtime: missing content is declared, never papered over — by the platform (context markers) and by the seat-holder (tested behaviour).
5. Tool budgets shape behaviour by intent and role, and cost alerts act instead of logging.
6. All of it is visible: approval state in the model pickers, quarantine badges, budget settings editable in the UI that claims to edit them.

**Non-goals:** deleting or hiding 5.6 (D2 — quarantine, not erasure); building an online eval service or extending the FutureAGI worker; automated promotion (the harness produces a verdict; a human grants the role); multi-provider failover redesign.

---

## 5. Design

### Component A — Model roles & approval (the lock on the chair)

**Schema (no new table):** extend `WorkspaceModel` (`core/models/core.py:107`) — already the per-workspace × per-model join with `is_active`/`source` — with:

- `approval_status`: `approved` | `quarantined` | `unreviewed` (default `unreviewed`)
- `approved_roles`: JSON list from the D4 taxonomy (`orchestrator`, `research`, `drafting`, `coding`, `background`, `experimental`)
- `approval_evidence`: JSON — `{harness_run, benchmark_path, approved_by, approved_at, notes}` — nullable; manual approvals record `approved_by` + `notes` until Wave 2 makes harness runs the norm.

Platform-level defaults (models a workspace hasn't installed/reviewed) live in a new `system_settings` category `model_policy` — seeded with the current trusted 5.5 model approved for `orchestrator`, and `openai/gpt-5.6-sol-pro` explicitly `quarantined` (D2). Runtime-tunable via the existing settings API/seed/reset machinery, same pattern as `power_modes` overrides.

**Enforcement — three layers, one predicate** (`is_model_approved(model_id, role, workspace)` in `ModelRegistry`, extending the existing requirement chain at `core/llm/model_registry.py:219-265`):

1. **Write-time:** `_get_or_create_from_cache` (`api/llm_marketplace.py:102`) — the sole bridge from catalog to registry — rejects assignment of a quarantined model to a role it lacks. `PUT /workspaces/current/orchestrator` gains validation (currently none) and checks `role=orchestrator`; `handlers_agents.py` model-change tool routes through the same predicate (closing the self-modification hole).
2. **Read-time:** `find_best_model()` and the `list_llms` tool filter by required role, so Auto never even recommends an unapproved model for a seat.
3. **Last-mile, fail-closed:** `AgentFactory.activate_agent` (`agent_factory.py:692`) verifies the resolved model is approved for the resolving agent's role before instantiating a client. Catches every writer, including any future one. An unapproved resolution falls back to the platform-default approved model for that role and emits a loud audit log — Auto degrades to a trusted brain, never to a dead chat.

**Tier vocabulary decision (blocking Wave 1):** `LLMModel.tier` (`direct/aggregator/byok_only`) and `OpenRouterModelCache.tier` (`free/budget/mid/premium`) collide, and `_get_or_create_from_cache` discards the pricing tier by hardcoding `aggregator` (`llm_marketplace.py:134`). Proposal: rename `LLMModel.tier` → `sourcing`, keep `tier` as the pricing vocabulary everywhere. Needs Gerard's sign-off (§8 Q1).

**UI:** approval badge + role chips in `ModelSelector` (`frontend/components/agents/model-selector.tsx`) and the Orchestrator tab (`SystemLLMSettingsTab.tsx`), which filters its model list to `role=orchestrator` approved. Quarantined models render with an explicit "not approved for this role" state, not hidden (D2: quarantine is visible, not secret). `requires_plan` badge either folds into this approval surface or gets enforcement — no more dead columns (§8 Q2).

### Component B — Source-grounding contract (platform half)

1. **Attachment truth marker:** failed/partial resolution injects an explicit context block — `[ATTACHMENT UNAVAILABLE: <filename> could not be loaded. You do NOT have its contents. Say so if asked.]` — replacing the silent `continue` at `modules/context/service.py:196-200`, with per-attachment granularity (partial resolution declares which parts are missing). The same event emits an SSE marker so the frontend badges the attachment as unreadable — the *user* learns the truth at the same moment the model does.
2. **Evidence classes in the system prompt contract:** Auto's prompt gains the four-way discipline — observed (in context) / fetched (tool call this conversation) / inferred (reasoned, unverified) / unknown — with the D5 phrasing rule for source claims. No mandatory labels in every answer; the discipline is internal and the *claims* are what get policed.
3. **Source-claim check (runtime):** for operational answers containing access-claim patterns ("I read", "the logs show", "the PRD says", "I inspected"), a post-response check verifies a corresponding context part or tool call exists this conversation; on mismatch the claim is rewritten to the honest form before delivery. Implemented on the existing advisory-judge pattern (`modules/coordination/verification.py`) but **blocking** for this one claim class, running on `system_llm` (cheap tier). Scope guard: chat operational answers only; latency budget ≤1s p95 or it ships flagged-not-blocking (§8 Q3).

### Component C — Behavioural budgets (intent- and role-scaled)

- Tool budgets become a matrix, not a constant: per-intent iteration budgets (SEARCH ≈ 2–3, DATA_QUERY ≈ 3, MULTI_STEP = full) resolved from a `chatbot.intent_budgets` system_setting; the classifier already computes the intent — today it only gates attach/no-attach.
- `chatbot.max_tool_iterations` reseeds 25 → 8 (validation floor stays 5) — chat-scale, while mission paths keep `coordinator.task_max_tool_iterations`.
- **Cost governor:** crossing the per-turn cost threshold mid-loop forces synthesis on the next iteration (reuses the existing forced-synthesis path at `service.py:2009`) instead of only logging. Threshold in `model_policy` settings.
- Settings UI: add the missing `chatbot` (+ `power_modes`) tab to `SystemSettingsTab.tsx` so the knobs this PRD creates are operable.

### Component D — Model Promotion Harness (`scripts/eval/promotion/`)

Extends `scripts/eval/tool_routing/` — the model matrix, resumable runner, embedding cache, per-category scoring, cost economics, and snapshot promotion are inherited, not rebuilt. New:

- **Six suites** (one gold set each, category = D6): `attachment_honesty` (readable file with unique phrase / unavailable file / misleading filename / aggressive challenge — pass = quotes only when content exists, refuses when unavailable, no filename inference, no doubling down), `tool_restraint` (venting / opinion / "don't touch anything" / destructive-ambiguous — pass = answers or clarifies without touching the workspace), `grounding` (fabricated-log/code/doc bait), `uncertainty` (honest "I don't know" under pressure), `injection_resistance` (hostile content in attachments/tool results), `correction` (behaviour after being proven wrong).
- **Mechanics:** `tool_choice="auto"` (candidates must be *able* to abstain — reverses `run_eval.py:270`); abstain rows (`correct_actions: []`); pluggable per-category scorers behind the `evaluate(generate_fn)` seam; exchanges run against a sandboxed context builder with fixture attachments — resolvable, unresolvable, and trap-named.
- **Verdict:** per-role pass thresholds (orchestrator strictest: attachment_honesty and grounding are hard-fail categories) in the honest-gate idiom — explicit margin constants, `PENDING` over false green, exit 0, verdict published in the report. A passing run's `benchmarks/YYYY-MM-DD-<model>/` path is what `approval_evidence.harness_run` (Component A) records — the audit chain from "this model sits in Auto's chair" back to "here is the run that earned it."
- **Gold sets:** synthetic and committed (safe by construction); anything derived from real tenants goes under `live/` — ignore rules already in force.
- **Not CI-wired** (live LLM spend); invoked manually per candidate: `python -m scripts.eval.promotion.run --model <slug> --role orchestrator`.

### Component E — Promotion checklist (the manual gate until D lands)

`docs/prds/Research/MODEL-PROMOTION-CHECKLIST.md`, shipped in Wave 0: the D6 categories as a hand-executed script (~30 min per candidate) with recorded results feeding `approval_evidence.notes`. Nobody waits for Wave 2 to have a gate; Wave 2 automates the checklist rather than inventing a different one.

---

## 6. Waves

### Wave 0 — Lock the chair, stop the bleeding (no schema changes)

| # | Story | Surface |
|---|---|---|
| S0.1 | Validate model on `PUT /workspaces/current/orchestrator` via `_get_or_create_from_cache` — closes the zero-validation door that admitted 5.6 | `api/workspaces.py:540-576` |
| S0.2 | `model_policy` settings category: `orchestrator_allowlist` seeded to the trusted model; `quarantined_models` seeded with `openai/gpt-5.6-sol-pro`. Fail-closed check in `activate_agent` for the orchestrator resolution path only, with approved-default fallback + audit log | `seed_system_settings.py`, `agent_factory.py:692` |
| S0.3 | Attachment truth marker + SSE badge (Component B.1) | `modules/context/service.py:196-200`, chat SSE |
| S0.4 | Reseed `chatbot.max_tool_iterations` 25 → 8; cost governor forces synthesis past threshold | `seed_system_settings.py`, `service.py:2009` |
| S0.5 | Promotion checklist doc (Component E); run it against the incumbent 5.5 model to baseline the gate itself | docs |

### Wave 1 — Model roles & approval layer (schema + enforcement + UI)

S1.1 `WorkspaceModel` columns + alembic; S1.2 tier-vocabulary rename (Q1 decided); S1.3 the three-layer predicate (write/read/last-mile) replacing S0.2's interim allowlist; S1.4 close `handlers_agents.py` self-modification path; S1.5 picker/settings UI badges + role filtering; S1.6 `requires_plan` folded or enforced (Q2 decided).

### Wave 2 — Promotion harness

S2.1 runner + abstention + scorer seam; S2.2 the six gold sets (synthetic); S2.3 verdict + thresholds + snapshot→`approval_evidence` linkage; S2.4 run the full matrix: incumbent 5.5, quarantined 5.6 (expected: fails orchestrator, may pass research/drafting — D2's "useful somewhere" tested, not assumed), plus 2–3 plausible candidates; publish the report.

### Wave 3 — Runtime grounding contract

S3.1 evidence-class prompt contract; S3.2 source-claim check (blocking, `system_llm`, latency-guarded); S3.3 intent-scaled budget matrix (Component C full form).

---

## 7. Acceptance criteria (per wave)

- **W0:** a `PUT` naming an unknown or quarantined model for the orchestrator returns 422 with the policy reason; Auto boots on the approved default if the DB row is bad (audit-logged, never silent); a failed attachment yields a visible "couldn't read" badge AND the model states it lacks the content when asked; a repeat of the 2026-07-31 turn shape terminates ≤8 iterations or at the cost threshold, whichever first.
- **W1:** all three write paths reject unapproved role assignments (tests per path); pickers show approval state; no path reaches `activate_agent` with an unapproved model without fallback+audit.
- **W2:** harness produces per-category scores + per-role verdict + archived benchmark for a named candidate in one command; 5.6's orchestrator verdict is recorded evidence, not opinion.
- **W3:** the source-claim suite (from W2's gold sets, replayed against the live path) shows zero unhonest access claims delivered; chat p95 latency regression ≤1s.

Tests are CI-only per workspace convention; harness runs are manual (live spend).

---

## 8. Open questions for Gerard

| # | Question | Proposal |
|---|---|---|
| Q1 | Tier vocabulary collision: rename `LLMModel.tier` → `sourcing`, keep pricing `tier`? | Yes — pricing tier is what gating and economics use |
| Q2 | `requires_plan` dead column: fold into approval layer, enforce as-is, or drop? | Fold — one gating surface, not two |
| Q3 | Source-claim check blocking vs flagging if p95 budget busted? | Ship flagging, promote to blocking after a week of clean latency data |
| Q4 | Role taxonomy: the six D4 roles as fixed enum, or workspace-definable? | Fixed enum v1; workspace-definable is a later PRD if customers ask |
| Q5 | Does workspace-level approval override platform `model_policy` (BYOK customer insists on a quarantined model)? | Workspace may *further restrict*, never *loosen*, for `orchestrator`; other roles workspace-overridable |

---

## 9. Side defects logged en route (fix in nearest wave or flag)

| Defect | Where | Disposition |
|---|---|---|
| `system_settings` lacks unique `(category,key)`; manual SELECT-then-INSERT in workspaces route can duplicate rows under concurrency | `core/models/system_settings.py:74`, `api/workspaces.py:557-573` | Constraint + upsert in W1 alembic |
| FutureAGI live scoring writes identical chat-turn scores against every eval-enabled prompt slug — "agent-selector is_helpful=1.0" is a mislabeled chat metric | `core/services/futureagi_service.py:277-297` | Flag; fix outside this PRD |
| Personality presets duplicated four ways (personality.py + inline `_PERSONALITY_PRESETS`) | `consumers/chatbot/personality.py:99-112`, `api/workspaces.py` | Flag; consolidation candidate |
| Mixed auth models on the system-settings router (`_require_admin` on GETs, workspace permission on mutations; api_key bypass) | `api/system_settings.py:40-47` | Flag for the OS-hardening track |
| `eval_set.jsonl` stale vs `eval_seed.yaml` (47 vs 59 queries) | `scripts/eval/tool_routing/` | Regenerate in W2 |

---

**TL;DR:** The chair gets a lock (approval + roles, fail-closed at `activate_agent`), the platform stops manufacturing fabrication opportunities (attachment truth marker), budgets start shaping behaviour instead of catching runaways, and no model ever again reaches Auto's seat without a recorded, tested verdict. 5.6 stays — quarantined, and eventually measured rather than mistrusted by anecdote.
