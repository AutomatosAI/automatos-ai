# Phase 0 — Residual Map: July Review Findings vs Post-Wave Main

**What actually landed since the July 2026 platform OS review.** This is the definitive per-finding register for the Phase-2 dossier teams: read your module's cluster and capability rollup before reviewing, and treat the watch list as candidate leads.

| | |
|---|---|
| Baseline | `reports/PLATFORM_OS_REVIEW_2026-07-01.md` — 94-entry register (90 enumerable finding ids, see §5.1), reviewed `main @ 37fdecc4e` |
| Verified against | pinned read-only tree `p2-src` = `origin/main @ 77bc9c6d5` (commit dated 2026-07-03, checked out 2026-07-04) |
| Waves in scope | W1–W6 (PRD-171..176, PRs #462–465, #481–484) and W7–W14 (PRD-177..183 + PRD-170 Code Canvas, PRs #494–#502). PRD-184 (W14 dead-code kill list) was **never authored** — its intended deletions are all still open. |
| Method | Static analysis only: file reads, `rg`/grep, `git` read-only (log/show/blame), `gh` read-only (CI runs, branch protection), registry `curl` GET. No servers, builds, or test runs. Six parallel verifier passes, one per cluster group; surprising "fixed" claims re-spot-checked against the pinned tree during assembly. |
| Grading convention | Statuses reflect **behavior on a default-config deployment**. A complete fix parked behind a default-OFF flag grades **PARTIAL** (see §4.A). Verifier evidence supersedes the worklist's expected-fix mapping wherever they disagreed. |
| Statuses | **FIXED** — the cited defect no longer reproduces, on default config. **PARTIAL** — real code landed but the July behavior persists in part (usually flag-gated or half-scoped). **NOT DONE** — the defect reproduces as filed; nothing claims it. **UNVERIFIABLE** — no in-tree description or the fix lives outside the pinned tree. **REGRESSED** — worse than July (none this pass, but several fixes *widened adjacent exposure* — see §4.C). |

All file:line citations below are into the pinned tree unless marked "sibling repo" or "live GitHub state".

---

## 1. Headline stats

### 1.1 Overall (90 enumerable findings)

| Status | Count | Share |
|---|---:|---:|
| FIXED | **38** | 42% |
| PARTIAL | **22** | 24% |
| NOT DONE | **27** | 30% |
| UNVERIFIABLE | **3** | 3% |
| REGRESSED | **0** | 0% |
| **Total verified** | **90** | 100% |

The register's headline count was 94; only 90 finding ids were ever assigned in the published report (F061, F067, F073 do not exist in the text; there is no F094). The uncommitted 94-entry `findings[]` register is the only place the remaining ~4 entries could live — see §5.1.

### 1.2 The July criticals

Of the 14 title-flagged criticals (13 critical per the register plus F092 critical-adjusted): **9 FIXED, 4 PARTIAL, 1 UNVERIFIABLE, 0 open.**

- FIXED: F001 (execution-spine kwarg crash), F002/F003/F004/F005/F006/F007 (tenant isolation), F008 (Clerk open-core blocker), F013 (Shopify Remix app, cross-repo).
- PARTIAL: F009 (initdb mount fixed; fresh clone still cannot boot — second blocker), F010 (single alembic head; from-zero replay still fails), F012 (gitignore + scanning; artifact still tracked), F092 (supply-chain CI live; coverage ratchet unarmed; migration lane doesn't gate).
- UNVERIFIABLE: F011 (mem0 fork — cross-repo human runbook).

Every tenant-isolation and execution-spine critical is closed. The remaining critical residue is entirely in the deploy/CI leg.

### 1.3 Per-cluster breakdown

| Cluster | n | FIXED | PARTIAL | NOT DONE | UNVERIFIABLE |
|---|---:|---:|---:|---:|---:|
| execution-spine | 7 | 4 | 1 | 1 | 1 |
| policy-governance | 12 | 0 | 9 | 3 | 0 |
| tenancy-auth-secrets | 15 | 10 | 3 | 1 | 1 |
| operating-graph-tools-learning | 12 | 5 | 2 | 5 | 0 |
| memory-field-codegraph | 8 | 6 | 0 | 1 | 1 |
| channels-widget-shopify | 13 | 4 | 1 | 8 | 0 |
| deploy-ci-observability | 13 | 6 | 5 | 2 | 0 |
| frontend-truth-api-honesty | 10 | 3 | 1 | 6 | 0 |
| **Total** | **90** | **38** | **22** | **27** | **3** |

Shape of the outcome: tenancy/auth and memory/field/codegraph are largely closed; the execution spine's four defects are all genuinely fixed; **policy-governance produced zero unqualified fixes** because the entire W4 plane is behind a default-OFF flag; **channels is the largest fully-untouched cluster** (zero commits since the review); frontend-truth and the dead-code backlog were mostly deferred to the never-authored PRD-184.

---

## 2. Finding-by-finding register

### 2.1 Cluster: execution-spine

#### F001 — FIXED
**CRITICAL — renamed kwarg `content_truncate_chars` raises TypeError in the `ToolLoopExecutor` ctor; every non-chat execution (board / Missions / scheduled / webhooks / inter-agent) died on first turn, swallowed by retry, budget burned.** (July: `agent_factory.py:1070` vs `tool_loop.py:154`)
- Fix: `4f3bcce1f` — W1 PRD-171, PR #463 (merge `48258ce22`).
- Evidence: `orchestrator/modules/agents/factory/agent_factory.py:1079` now passes `content_truncate_tokens=0` to `ToolLoopExecutor`, whose ctor accepts `content_truncate_tokens` (`orchestrator/modules/tools/execution/tool_loop.py:192,198`). The construction at `agent_factory.py:1075` is unconditional after the first LLM response inside the retry loop, so the old kwarg would have crashed every non-chat turn; grep shows only 3 non-test construction sites (`agent_factory.py:1075`; `consumers/chatbot/service.py:1491` with `content_truncate_tokens=2000`; `tool_loop.py:170` internal). The through-the-factory test exists: `orchestrator/tests/test_prd171_execution_spine.py:89` builds `AgentFactory` via `__new__` and runs the REAL `execute_with_prompt` body (stubbing only LLM/monitoring/tool-loading), asserting a non-error result — plus a seam guard at `:141` asserting the removed kwarg raises TypeError.
- Notes: genuine one-line rename, no shim; the acceptance test closes the exact direct-construction gap that let the regression ship, and the guard test makes re-introduction fail loudly.

#### F023 — FIXED
**Board spine marks failed executions done with result=None — `_launch_task_execution` never inspects the returned `status:error` dict.** (July: `board_tasks.py:900-921`)
- Fix: `4f3bcce1f` (PR #463).
- Evidence: `orchestrator/api/board_tasks.py:1043` reads `exec_status = (exec_result or {}).get("status")`; the error branch at `:1056-1070` sets `task.status='failed'`, stores `exec_result['error']` into `error_message` (truncated to 500), sets `completed_at`, fires `_dispatch_task_failed`, and writes a failure report — mirroring the pre-existing crash handler at `:1100-1113`. Success path unchanged. Test drives the real `_run` coroutine with only the factory result stubbed: `test_prd171_execution_spine.py:314` asserts `status=='failed'` + error text; `:328` asserts success parity.
- Notes: honest failure — failed tasks surface through the same `_dispatch_task_failed` + report path as crashes; the board no longer shows green checkmarks over dead runs.

#### F024 — FIXED
**No lease renewal mid-run: any run over the 600s lease is swept back to assigned and double-executed.** (July: `board_dispatcher.py:176-205`)
- Fix: `4f3bcce1f` (PR #463).
- Evidence: `renew_lease` at `orchestrator/services/board_dispatcher.py:272-308` pushes `lease_until` forward ONLY WHERE `status='in_progress'` (never resurrects terminal rows). `_lease_heartbeat` at `orchestrator/api/board_tasks.py:905-940` renews every `BOARD_DISPATCH_LEASE_SECONDS/2` (300s for the 600s default) on its own short-lived session and stops when renewal returns False. Wired for the life of every run at `board_tasks.py:1011` (started even ahead of the approval gate) and cancelled in the finally block at `:1120-1128`. Both execution entry paths converge here: the dispatcher claim loop's `_launch_one` (`board_dispatcher.py:466-483`) delegates to the same `_launch_task_execution`. Crash semantics preserved: process death kills the heartbeat, the lease truly lapses, and `requeue_expired_leases` (`board_dispatcher.py:184`) sweeps it. Tests: `test_prd171_execution_spine.py:368,384`.
- Notes: shipped design is a wall-clock timer task rather than the suggested post-tool-hook renewal — strictly broader coverage (a single LLM call exceeding 600s is also protected). Renewal failure is logged-and-retried, never propagated into the run — appropriate best-effort semantics.

#### F025 — FIXED
**Kanban drag double-executes Mission-mirror tasks — PATCH status→in_progress fires launch for any non-Playbook source.** (July: `board_tasks.py:550-555`)
- Fix: `4f3bcce1f` (PR #463).
- Evidence: `_NON_EXECUTABLE_SOURCE_TYPES = frozenset({'recipe','orchestration','orchestration_task'})` at `orchestrator/api/board_tasks.py:43-46`, enforced on BOTH PATCH launch paths: `update_task`'s trigger_execution gate at `:561-568` and `update_task_status`'s drag gate at `:884-897`. Drift-guard tests match the literals the bridge actually stamps (`test_prd171_execution_spine.py:398,411` vs `orchestration_board_bridge.py:115,198`).
- Notes — adjacent residual (same double-execution class, different user action): the dispatch claim loop still filters only `source_type <> 'recipe'` (`board_dispatcher.py:92,117,146`), and both `run_task_now` (`board_tasks.py:828` — resets any non-running task to 'assigned') and `update_task`'s inbox→assigned auto-transition (`:543-544,:592-599`) can place a Mission-mirror row into 'assigned', where the loop will claim and board-execute it in parallel with the Mission engine. Bridge-created mirrors never land in 'assigned' by themselves (`orchestration_board_bridge.py:49-57`), so this requires an explicit Run-Now/assign on a mirror — narrower than the drag bug, but the exclusion should extend to `claim_tasks`/`run_task_now`. See §4.C.1.

#### F069 — NOT DONE
**Both `api_playbooks` endpoints dead-on-arrival: raw SQL fails under SQLAlchemy 2.0.23 (sole consumer of `modules/learning`).**
- Evidence: both endpoints remain dead and still mounted: `orchestrator/main.py:55` imports and `:968` mounts the router. GET `/api/playbooks` (`orchestrator/api/api_playbooks.py:36-44`) passes a raw SQL string to `db.execute` with no `text()` wrapper (no sqlalchemy text import anywhere in the file) — under pinned `sqlalchemy==2.0.23` (`orchestrator/requirements.txt:13`) this raises ArgumentError on every call. POST `/api/playbooks/mine` reaches `PlaybookMiner.persist_top` (`orchestrator/modules/learning/playbooks/miner.py:72-84`), which also executes a raw string AND carries the `:pattern::jsonb` bind-cast bug; with default `min_support=5` it never reaches the SQL because `_fetch_sequences` returns hardcoded demo data (`miner.py:27-34`, max support 3) and "succeeds" with `{"generated": []}` — a facade — while `min_support<=3` crashes. git log since 2026-06-25 on `api_playbooks.py` + `modules/learning` is empty.
- Notes: matches the "none planned" expectation — the adjacent W14/PRD-184 kill list that would have deleted this surface with F082 was never authored. `modules/learning` (whose sole consumer is this dead router) survives intact.

#### F078 — PARTIAL
**Legacy workflow engine still mounted: `api/workflows.py` with live execute endpoints, PRD-125 retirement never landed; `jira_bug_triage` (workflow_id=9000) reachable via Composio webhook dispatch.** (July: `workflows.py:1049`; `main.py:885-886`; `composio.py:839-846`)
- Fix (collateral, not a retirement): `19ea48825` — W2 PRD-172.
- Evidence: the retirement never landed. `orchestrator/api/workflows.py` is still 1,424 lines, imported at `main.py:38` and mounted at `main.py:947`; `workflow_templates` still mounted at `main.py:948` with the "Legacy - backward compatibility" comment. POST `/{workflow_id}/execute-advanced` remains a live execute endpoint dispatching real work through `get_task_runner` (`workflows.py:955-1000+`), and the legacy 9-stage pipeline `execute_workflow_with_progress` still exists (`workflows.py:1354`) and remains the Composio webhook fallback (`composio.py:898-920`). `jira_bug_triage` is still registered under workflow_id=9000 (`orchestrator/modules/workflows/recipes/__init__.py:79-81`) and reachable via `_dispatch_workflow` (`composio.py:829-850`). PRD-184 was never authored; no commit cites F078. What DID move: W2's `19ea48825` deleted three of the legacy execute endpoints — POST `/{workflow_id}/execute` (the July `:1049` cite), POST `/execute`, POST `/executions/` — as its F006 fix, shrinking the file 1543→1424 and cutting the execute surface from 3 endpoints to 1.
- Notes: the exact endpoint the July report cited is gone and the cross-tenant leg (F006) is closed, but the finding's substance — a fifth execution engine mounted alongside the Mission/board spine, with a webhook-reachable leg — remains fully true. The prescribed scoped migration (move the jira recipe onto the Mission path, then unmount both routers) has not started.

#### F093 — UNVERIFIABLE
**Dead-on-arrival/write-only endpoints cluster member — grouped with F069/F070, never individually described.** (July: report line 154, §4 medium themes)
- Evidence: the July report never describes F093 individually — its only substantive trace is the cluster line at `reports/PLATFORM_OS_REVIEW_2026-07-01.md:154` ("Dead-on-arrival or write-only endpoints (F069, F070, F093)"), where exemplars are given only for F069 and F070. Every neighbouring id gets an individual description elsewhere; F093 gets none, and the repo contains no fuller register (Appendix C.4 references a 94-entry `findings[]` register "as reported" but it was never committed). Grep-proven negatives: zero F093 mentions in any wave PRD and zero commits citing it — nothing even claims to have fixed it.
- Notes: what would verify it: the review workflow's authoritative `findings[]` register or the original transcripts (2026-07-01/02 runs). Sibling context cuts both ways: F069 is confirmed not-done while F070 was genuinely fixed by W9 — cluster membership predicts nothing.

### 2.2 Cluster: policy-governance

> Cluster-wide context: W4 (PRD-174, PR #484, commit `0710e4cae`) built a real policy plane — `orchestrator/modules/policy/` (gate/types/roles/pricing/budget/policy_document/bus/errors/flag) with one chokepoint in `unified_executor.py`. **Every behavioral change is behind `AUTOMATOS_POLICY_PLANE`, default OFF** (`config.py:645`; nothing in envs/, docker-compose.yml, railway.json or Dockerfile sets it; `modules/policy/flag.py:20-32` fails safe to OFF). With the flag OFF, default deployments keep July behavior byte-for-byte on F040/F014/F042/F043 and get no budget admission. Even when ON, the gate **fails OPEN on any internal error** (`unified_executor.py:275-280` — "any error here is logged and treated as proceed"). W11 (PRD-181 S2, PR #499, `5378424d3`) added the unconditional board approval-grant surface.

#### F085 — PARTIAL
**No unified policy plane: budgets/approvals/rate-limits/roles live in three partial mechanisms; approval engine binds Mission runtime only, API surface unaware of `approval_policy`.**
- Fix: `0710e4cae` (W4) + `5378424d3` (W11 S2).
- Evidence: the plane is real: `modules/policy/gate.py:74-118` (`PolicyGate.check` with super-admin → admin → budget → act-vs-ask) and one chokepoint in `orchestrator/modules/tools/execution/unified_executor.py:395-400` (`_policy_gate_check` at `:204-280`) evaluated before platform/workspace/registry/composio_execute routing. The API surface gained an approval-grant surface unconditionally: `api/approval_grants.py` mounted at `router_manifest.py:78`. BUT the entire gate is behind `AUTOMATOS_POLICY_PLANE` default OFF; flag OFF returns None at `unified_executor.py:227-228` ("byte-for-byte the legacy per-router gates"). Even flag ON, the Composio lane is NOT fully covered: the chat streaming loop's per-action shortcut executes Composio directly via `ComposioToolService.execute_action`, bypassing the chokepoint (`consumers/chatbot/service.py:1321-1334` → `_execute_composio_action:1550-1565`); Playbook steps do the same (`api/recipe_executor.py:655`); widget email actions call `client.execute_action` directly (`api/widget_email.py:286,340,388,437`); and `/api/tasks` direct-step never touches any tool executor (`api/tasks.py:62-124`).
- Notes: gate fails open on internal error (`unified_executor.py:275-280`); the Art.12 audit handler registration is also flag-gated (`main.py:519`). Mission approval (`approval_policy`) and the per-(workspace,agent) rate limiter deliberately stay separate mechanisms (`gate.py:26-29`) — "one plane" is a universal layer on top of the old three, not a replacement.

#### F086 — PARTIAL
**No pre-call budget admission on the hot path; no budget-exceeded exception class anywhere; LLM manager cost-logs only after the call.** (July: `tool_loop.py:326`)
- Fix: `0710e4cae` (W4).
- Evidence: a budget-exceeded class now exists (`modules/policy/budget.py:39` `BudgetExceeded`) and a genuine pre-call admission check exists: `check_budget` (`budget.py:159-216`) compares spend-to-date from `llm_usage` plus a projected-call estimate against `workspace.plan_limits.budget` ceilings; `PolicyGate._budget_gate` calls it on every tool call (`gate.py:162-190`) at the chokepoint. But (1) it only runs with the flag ON; (2) no production caller supplies `model_id`/`est_input_tokens`/`est_output_tokens` — the `ToolCall` built at `unified_executor.py:245-253` omits them, so projected cost is always 0 and the gate can only trip once the workspace is ALREADY over ceiling; (3) there is still no pre-LLM-call dollar admission: the LLM manager still only cost-logs after the call, by design (`core/llm/manager.py:702-733`); (4) the `on_pre_tool` seam exists (`tool_loop.py:396-405`) but zero call sites wire it; (5) the gate is inert unless a workspace configures `plan_limits.budget` (`budget.py:174-175` → allow).
- Notes: `BudgetExceeded` is defined but never raised anywhere — enforcement flows as errors-as-data (fine), but the July claim "no budget-exceeded exception class" is answered by a decorative class. A pure-generation loop with no tool calls is still never dollar-gated.

#### F059 — PARTIAL
**Four hardcoded price tables drive a model-blind Mission dollar gate (flat `COORDINATOR_COST_PER_1K_TOKENS`, up to 60x wrong vs the platform's own price map).** (July: `coordinator_service.py:2384`)
- Fix: `0710e4cae` (W4).
- Evidence: a model-aware pricing source now exists: `modules/policy/pricing.py:25-63` (`price_per_1k`/`estimate_cost_usd` off the DB `llm_models` registry, never guesses). But the July-cited Mission dollar gate is byte-identical in behavior: `coordinator_service.py:2424-2428` `_estimate_cost_usd` still returns `tokens/1000 * Config.COORDINATOR_COST_PER_1K_TOKENS` (flat 0.003 default, `config.py:721`), feeding `evaluate_approval` at `coordinator_service.py:2354-2364`. The same flat rate also prices the NEW W11 governance: `dispatcher.py:452,457` (Mission budget bands) and `recipe_executor.py:1002` `_tokens_to_usd` (Playbook ceiling). The other hardcoded tables survive: `manager._MODEL_COST_MAP` (`core/llm/manager.py:713`, deliberately kept per the in-code F059 note at `:702-711`) and `logging_utils.TOKEN_COSTS` (`core/utils/logging_utils.py:469-484`, stale 2024 prices defaulting unknown models to gpt-4). The registry-backed source is consumed only by the flag-gated `PolicyGate` budget check, and no caller passes token estimates to it (see F086) — in practice it prices nothing on main paths.
- Notes: "single model-aware pricing source" was not achieved — the count of pricing paths effectively went up. The approval/budget decisions that matter — Mission auto-approve ceilings and Playbook ceilings — are still denominated in the model-blind flat dollars the finding described.

#### F040 — PARTIAL
**Main-plane rate limiter inert: Limiter constructed but `SlowAPIMiddleware` never added to the app.** (July: `main.py:762-776`)
- Fix: `0710e4cae` (W4).
- Evidence: `SlowAPIMiddleware` is now registered — but only when the flag is ON: `main.py:837-838` `if _policy_plane_on: app.add_middleware(SlowAPIMiddleware)`, with `swallow_errors=not _policy_plane_on` (`main.py:827-831`) so fail-closed evaluation also only applies under the flag. With the default OFF, deployments keep exactly the July placebo: Limiter constructed (`main.py:827-832`), exception handler registered, `default_limits=['60/minute']` never enforced. No per-route `@limiter.limit` decorators exist anywhere, so there is no partial enforcement path either. The in-code comment admits it: "Registered ONLY under the policy-plane flag; OFF keeps today's (placebo) behaviour" (`main.py:834-836`). Widget-plane rate limiting (`WidgetRateLimitMiddleware`, `main.py:801`) was already separate and unconditional.
- Notes: the regression test is weak — `test_prd174_flag_gating.py:107-113` asserts F040 by string-grepping `main.py` source for "SlowAPIMiddleware", not by exercising a rate limit. Until the flag flips, the platform's HTTP edge remains fail-open.

#### F014 — PARTIAL
**`admin_only` tool gate is a no-op for agent calls: `is_admin` auto-flips true whenever the workspace has any owner/admin member, not the calling principal.** (July: `platform_executor.py:641-645`)
- Fix: `0710e4cae` (W4).
- Evidence: the execution-side fix exists but is flag-gated. The admin gate (`modules/tools/discovery/platform_executor.py:675-709`): with explicit `caller_context` it checks the caller's own roles (`:683-686`); with no caller identity it calls `_agent_inherits_admin` (`:694`). `_agent_inherits_admin` (`:563-593`): plane ON → requires the explicit, default-OFF `agents_inherit_admin` workspace policy (`policy_document.py:82`) AND a real admin owner; plane OFF (default) → falls through at `:592-593` to `_workspace_has_admin_owner` (`:529-561`) — the exact July auto-flip. `PolicyGate` mirrors the same logic under the flag (`gate.py:122-160`). By default, `admin_only` is still a no-op for identity-less agent calls (heartbeat/agent-factory paths), verbatim.
- Notes: a cousin survives unconditionally in the tool-VISIBILITY layer: `modules/tools/tool_router.py:369-388` auto-flips `is_admin` from workspace membership to decide which admin tool schemas are exposed to the model (`exclude_admin=not is_admin` at `:420`). Exposure not authorization, but with plane OFF both layers resolve admin from workspace membership.

#### F042 — PARTIAL
**Empty SDK-key permissions fork: allow-all on the widget plane, deny-all on the board plane — one minted no-permission key is god-key on one, null-key on the other.** (July: `widgets/auth.py:239`; `api_key_service.py:237-241`; `hybrid.py:537-539`)
- Fix: `0710e4cae` (W4).
- Evidence: one empty-permission semantic exists only under the flag. Widget plane: `api/widgets/auth.py:236-264` `require_permission` — plane ON uses `modules/policy/roles.has_permission` (empty = deny, `:254`); plane OFF keeps the historical "empty list = unrestricted" god-key (`:257`), explicitly to avoid breaking already-issued keys (comment `:239-244`). Flag defaults OFF, so the July fork is the live behavior: widget empty = allow-all vs board `_sdk_key_has_scope` empty = deny-all (`core/auth/hybrid.py:527-539`, unchanged, with its SECURITY comment still describing the divergence).
- Notes: `ApiKeyService.check_permissions` (`core/services/api_key_service.py:233-243`) still hardcodes "empty or None permissions = ALL permissions granted" with no flag gate. It currently has zero callers — dormant — but it is a loaded allow-all helper left in source; anyone wiring it re-opens the god-key on a third surface. See §4.C.19.

#### F043 — PARTIAL
**Seven routers gate admin functions with strict `system_role=='admin'`, 403ing super_admin out entirely (role hierarchy forked).**
- Fix: `0710e4cae` (W4).
- Evidence: a single shared helper now exists and is adopted: `core/auth/roles.py:28-45` `caller_is_admin`, called by all 8 formerly-forked routers (`workspace_plugins.py:34`, `marketplace_plugins.py:33`, `workspace_skills.py:44`, `widget_marketplace.py:58`, `admin_workspaces.py:49`, `admin_plugins.py:55`, `marketplace.py:41`, `skills.py:526-527`). A straggler sweep of `orchestrator/api/` finds no remaining strict equality checks (only `clerk.py:205`, a different platform-staff-sync concern). BUT the behavioral fix is flag-gated: `roles.py:37-45` — plane ON uses the super_admin ⊇ admin hierarchy; plane OFF (default) returns `role == 'admin'`, i.e. super_admin is STILL 403'd out of all 8 routers exactly as in July, by documented design (`roles.py:8-10`).
- Notes: the consolidation (one choke point instead of 8 copies) is real and unconditional — future fixes are one line. The user-visible defect persists in every default deployment until the flag flips.

#### F060 — PARTIAL
**Governance covers Missions only — Auto-created board tasks and Playbook runs have no dollar ceiling or approval gate; five execution engines coexist and two (legacy workflow runner, `/api/tasks` direct-step) sit outside all governance yet are mounted.** (July: `tasks.py:33`; `main.py:885`)
- Fix: `5378424d3` (W11 S2) + `0710e4cae` (W4) + `19ea48825` (W2).
- Evidence — landed: board tasks are genuinely gated, unconditionally (NOT flag-gated). `_launch_task_execution` calls `_board_task_blocked_pending_approval` before executing (`api/board_tasks.py:1019`, gate at `:943-994`); it runs the SAME `evaluate_approval` Missions use via `services/board_approval.py:63-147`, creates a durable/expiring/revocable ApprovalGrant (`core/services/approval_grants.py`, model `core/models/approval_grants.py`, migration `prd181_s2_approval_grants`), blocks the task, and `api/approval_grants.py` (mounted `router_manifest.py:78`) re-queues on grant / fails on deny; the dispatcher routes through the same launcher (`board_dispatcher.py:472-474`). Playbooks: a per-step dollar ceiling exists (`api/recipe_executor.py:1184-1208` via `services/budget_ceiling.playbook_can_afford`). Engines: W2 `19ea48825` deleted the three legacy execute endpoints and `execute_workflow_with_progress` is a no-op stub (`workflows.py:1354-1363`).
- Evidence — gaps: (1) board dollar ceiling is **vacuous** — `board_tasks.py:968` never passes `estimated_cost_usd`, so it defaults 0.0 (`board_approval.py:68`) and only an `always_ask` policy ever creates a grant, never the ceiling; (2) the board gate **fails OPEN** on any error (`board_tasks.py:989-994`); (3) Playbooks get NO approval gate — `SUBJECT_PLAYBOOK_RUN` (`approval_grants.py:54`) has zero non-model references — and the ceiling is opt-in (`cost_ceiling` absent = unlimited, `recipe_executor.py:1005-1010`) and flat-rate priced (`:1002`); Playbook Composio steps also execute outside the PolicyGate (`recipe_executor.py:655`); (4) scheduled/webhook agents are explicitly future work (`board_approval.py:12`; no approval/grant/budget references in `scheduled_task_service.py` or `playbook_scheduler.py`); (5) `/api/tasks` direct-step is still mounted (`main.py:1000`) with zero governance — auth + backend check only, raw shell/git steps straight to the worker (`api/tasks.py:62-139`); (6) the legacy workflow engine is still mounted (`main.py:947`) and execute-advanced (`workflows.py:955-1032`) enqueues agent tasks with no admission gate. Engine-count reduction: not done (and was unclaimed).
- Notes: the board half is the most complete fix in this cluster — unconditional, durable, audited, idempotent. But "dollar ceiling for board" in the commit message overstates: with the estimate hardwired to 0.0 the ceiling branch is unreachable for board tasks.

#### F071 — NOT DONE
**Three hardcoded verb→capability/app maps bake tool-routing policy into source.** (July: report §12.3)
- Evidence: all three maps are still hardcoded and load-bearing: `INTENT_TO_CAPABILITIES` (`modules/tools/capabilities/taxonomy.py:288`, consumed at `taxonomy.py:413` and `action_capability_filter.py:165-171`), `_HINT_TO_APPS` (`modules/tools/services/composio_tool_service.py:80`, applied at `:174-183`), and the hint-service keyword constants (`composio_hint_service.py:44-56`). Neither file touched since pre-review PR #303 (last change `b2e69a1f3`). No wave PRD or commit cites F071. W7 added learned edges/chain hints ALONGSIDE these maps; the adjacent alphabetical enrichment (F074 site) is also still live — the maps were not replaced by ranking.
- Notes: honest negative — nothing claimed this and nothing fixed it. The W4 plane is orthogonal (governs whether a routed action may run, not how verbs map to capabilities/apps).

#### F072 — PARTIAL
**`REQUIRES_CONFIRMATION`/`DESTRUCTIVE_CAPABILITIES` hardcode Composio approval policy in source — "guardrails as policy" untunable without a deploy.** (July: `taxonomy.py:259,230`)
- Adjacent fixes: `52cb4e114` (W7 S3) + `0710e4cae` (W4) — neither cites F072.
- Evidence: the cited sets are still hardcoded: `DESTRUCTIVE_CAPABILITIES` at `taxonomy.py:230-257` and `REQUIRES_CONFIRMATION` at `taxonomy.py:268-280` (line numbers shifted by W7's insertion of `DESTRUCTIVE_INTENT_KEYWORDS` at `:263`) — changing which Composio capabilities require confirmation still requires a deploy. Two adjacent real improvements: (1) W7 made the destructive gate **fail-CLOSED** on an empty/unavailable metadata table (`COMPOSIO_DESTRUCTIVE_FAIL_CLOSED` default True, unconditional; enforcement at `action_capability_filter.py:286,312` via `taxonomy.intent_is_destructive`), so the taxonomy is no longer inert-by-default; (2) W4 built a genuinely data-driven per-workspace policy: `workspace.settings.policy_plane` posture + `route_overrides` (`policy_document.py:77-160`, setter at `:163-201`), under which ALL Composio calls classify as `external_side_effect` and route to ask under Balanced (`classify_action` at `:215-249`, `gate.py:192-225`) — tunable per workspace without a deploy — but only when the flag is ON, and it layers on top of rather than replacing the taxonomy sets.
- Notes: the taxonomy did NOT become data-driven. What changed is posture (fail-closed) and a parallel flag-gated tunable layer at coarser granularity (risk-class routing, not per-capability confirmation lists). Under default flag-OFF, the finding's substance is fully intact.

#### F077 — NOT DONE
**Hardcoded policy/SaaS-topology cluster member (F068, F071–F077 range) — never individually described in the report.** (July: report line 153)
- Evidence: F077 exists only as a member of the merged hardcoded-policy cluster (report line 153; dedup note line 775). No wave PRD or commit cites it; rg across `docs/` and `orchestrator/` finds nothing outside the report. The cluster's verifiable exemplars remain hardcoded on main: taxonomy confirmation sets (`taxonomy.py:230,268`), verb/app maps (`taxonomy.py:288`, `composio_tool_service.py:80`), flat token-price rate (`config.py:721` driving `coordinator_service.py:2428`, `dispatcher.py:452,457`, `recipe_executor.py:1002`).
- Notes: with no per-finding definition, F077's exact site cannot be pinpointed — this is a cluster-level judgment (graded NOT DONE rather than UNVERIFIABLE because the cluster's remaining exemplars are demonstrably unchanged and nothing claimed it). One sibling in the same cluster (F076) WAS individually fixed when claimed — F077 was not claimed.

#### F041 — NOT DONE
**PRD-143 router-wide super_admin lock swallowed user-facing heartbeat schedule endpoints (over-gated user surface).**
- Evidence: the router-wide lock is untouched: `api/heartbeat.py:27-32` still declares the whole `/api/heartbeat` router with `dependencies=[Depends(require_super_admin)]`, and `require_super_admin` remains a strict `system_role=='super_admin'` check (`core/auth/super_admin.py:17-29`). git log on `heartbeat.py` shows the last change is the PRD-143 S6 lock itself (`e57a94989`); no commit cites F041. The over-gated USER surface is still actively called by the product UI: the agent configuration modal reads/writes `/api/heartbeat/agents/{id}/config` (`frontend/components/agents/agent-configuration-modal.tsx:353,604`) and runs `/api/heartbeat/agents/{id}/run` (`:620`); the heartbeats hook lists `/api/heartbeat/workspace` and toggles `/api/heartbeat/{id}/toggle` (`frontend/hooks/use-heartbeats-api.ts:84,121`). A non-super-admin workspace owner configuring their own agent's heartbeat schedule gets 403 on every one of these.
- Notes: W4's `caller_is_admin` work is unrelated (this is a super_admin router lock, not an admin equality check). The fix remains what July implied: move the user-facing agent-heartbeat CRUD off the observability-tier router or give it its own workspace-scoped dependency.

### 2.3 Cluster: tenancy-auth-secrets

#### F002 — FIXED
**CRITICAL — skills router takes ctx and never uses it — cross-workspace SKILL.md attach/read (prompt-injection/exfiltration); one DELETE can deactivate a global builtin skill for every workspace.** (July: `skills.py:838,731,604`)
- Fix: `19ea48825` — W2 PRD-172, PR #464.
- Evidence: `skills.py` now uses ctx via helpers `_skill_visible_to` (`orchestrator/api/skills.py:160-170`) and `_assert_agent_in_workspace` (`:173-182`). All three July-cited endpoints closed: `get_skill_content:604→674` and `get_skill_details:611-616` gate on `_skill_visible_to` (404, existence-hiding); `deactivate_skill` (July `:731`) at `:795-812` requires super-admin to delete a global (`workspace_id IS NULL`) skill and 404s another workspace's private skill; `assign_skills_to_agent` (July `:838`) at `:932+942-948` asserts agent ownership and filters attachable skills to global-or-own-workspace; `remove_skills_from_agent:1014` asserts ownership. July baseline (`37fdecc4e`) had none of these.
- Notes — residual outside F002's cited scope: `cleanup_old_skill_mappings` (`skills.py:730`) is "/admin/"-named but has NO admin gate — takes ctx and never checks it, deletes `agent_skills` rows for all legacy `source='Unknown'` skills platform-wide. See §4.C.3.

#### F003 — FIXED
**CRITICAL — four Shopify sync routes declare only `db=Depends(get_db)` — guessed workspace UUID triggers costly bulk-ops and overwrites that workspace's Knowledge Graph via `import_graph(merge=False)`.** (July: `shopify.py:584-991`)
- Fix: `19ea48825` (W2).
- Evidence: all four sync routes now carry `get_request_context_hybrid` and derive the workspace from ctx via `_resolve_sync_workspace` (`orchestrator/api/shopify.py:61-82`), called at `start_product_sync:466`, `get_product_sync_status:636`, `start_orders_sync:708`, `get_orders_sync_status:909`. A non-admin caller supplying a mismatched `workspace_id` param is rejected 403 (`:77-81`); admins may target explicit workspaces. Heavy bodies extracted to `_product_sync_impl`/`_orders_sync_impl` for the trusted in-process auto-trigger.
- Notes: guessed-UUID cross-tenant sync + `import_graph(merge=False)` overwrite is closed.

#### F004 — FIXED
**CRITICAL(adj) — `SHOPIFY_INTERNAL_API_KEY` fail-open when unset ('' default; `_verify_internal_key` returns early on falsy) — any Authorization value reaches /provision, /connect, /deactivate.** (July: `config.py:441`; `shopify.py:43-45`)
- Fix: `19ea48825` (W2).
- Evidence: `_verify_internal_key` (`orchestrator/api/shopify.py:85-101`) dropped the fail-open branch: if the configured key is empty it raises 503, and it compares with `hmac.compare_digest` — an arbitrary `Authorization: Bearer x` can no longer reach the internal routes. This runtime guard is the real fail-closed guarantee. Spot-checked in the pinned tree during assembly: confirmed.
- Notes — the advertised boot-abort is overstated: `config.validate_security()` (`config.py:1029`) does raise RuntimeError, and `main.py:178` calls it inside `_boot_phase_1_core`, but `run_stage` (`core/models/bootstrap.py:115-137`) CATCHES the exception and only records the stage "failed" without re-raising; lifespan (`main.py:507`) never checks the DATABASE_INIT result, and the trust gate (`main.py:301-330`) checks only DB reachability. So an unset key does NOT abort boot — only the runtime 503 protects the endpoints. Same swallow affects the F005/F008 boot guards. See §4.C.5.

#### F005 — FIXED
**CRITICAL(adj) — `S3VectorsBackend.search` accepts filters and never applies them; isolation rests on unvalidated bucket-name template — shared bucket leaks cross-workspace chunk text (gated by `S3_VECTORS_ENABLED`).** (July: `s3_vectors_backend.py:123-146`; `rag/service.py:316-317`)
- Fix: `19ea48825` (W2).
- Evidence: two real runtime guards, independent of the (swallowed) boot check. (1) `S3VectorsBackend.__init__` refuses a bucket template with no `{workspace_id}` placeholder (`orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py:51-55`). (2) `search()` now enforces the filter it previously ignored: rejects a `filters['workspace_id']` that disagrees with the backend's bound workspace (`:158-169`) and drops any hit whose metadata `workspace_id` differs (`:190-196`). The RAG choke point passes the filter (`modules/rag/service.py:934-941`) and `_expand_to_parent_context` joins documents and pins `d.workspace_id` (`service.py:1058-1064`) — the parent-context hydration was previously an unscoped join. `add_documents` stamps `workspace_id` into metadata (`:251`).
- Notes: genuine at both the vector-search and parent-expansion layers. Still gated by `S3_VECTORS_ENABLED` (default false) as before — the fix is in the enabled path.

#### F006 — FIXED
**CRITICAL(adj) — legacy workflow execute endpoints fetch workflow with no workspace filter and select first active agent from ANY workspace — cross-tenant oracle/enqueue.** (July: `workflows.py:1059-1072`)
- Fix: `19ea48825` (W2).
- Evidence: the legacy PRD-125 execute surface at the cited lines is DELETED: `execute_workflow` / `execute_workflow_general` / `create_execution` removed (deletion notes at `orchestrator/api/workflows.py:1050-1058` and `:1103-1105`). `github_webhooks` no longer imports the symbol and returns an honest `workflow_execution_disabled` (`api/github_webhooks.py:129-148`). The surviving `execute_workflow_advanced` scopes the workflow to `ctx.workspace_id` (`workflows.py:966`).
- Notes — residual, same class but not the cited lines: `execute_workflow_advanced` still selects the fallback agent with `db.query(Agent).filter(Agent.status=='active').first()` (`workflows.py:973`) and looks up an explicit `agent_id` with no workspace filter (`:978`) — a cross-workspace agent can be bound to a (correctly workspace-scoped) execution when `agent_id` is omitted. See §4.C.4.

#### F007 — FIXED
**CRITICAL(adj) — GET /api/alerts unconditionally unauthenticated — `ALERT_INGEST_TOKEN` guards only the ingest POST.**
- Fix: `19ea48825` (W2).
- Evidence: GET `/api/alerts` is now super-admin only: `@router.get('/alerts', dependencies=[Depends(require_super_admin)])` (`orchestrator/core/monitoring/automatos_alerts.py:212`). The Loki log routes are gated at router level: `APIRouter(..., dependencies=[Depends(require_super_admin)])` (`core/monitoring/automatos_logs_api.py:95`, covering `/logs/query`, `/logs/labels`, `/logs/label/{name}/values`). `require_super_admin` (`core/auth/super_admin.py:17-29`) is fail-closed (API-key "admin" and "service" principals refuse). `ALERT_INGEST_TOKEN` stays ONLY on the ingest POST (`automatos_alerts.py:159-162`).
- Notes: read surface is now obs-tier super_admin per PRD-143; ingest keeps its bearer token.

#### F019 — PARTIAL
**NL2SQL SELECT-only validator passes side-effecting functions (`query_to_xml('UPDATE ...')`); no read-only DB role enforcement.** (July: `validator.py:30-36,212-214`)
- Fix: `19ea48825` (W2) — validator half only.
- Evidence — done: `SQLValidator.validate_and_rewrite` walks the whole sqlglot AST and rejects a denylist of side-effecting functions (`_FORBIDDEN_FUNCTIONS` at `orchestrator/modules/nl2sql/query/validator.py:48-66`; `_forbidden_function_in:98-105`; enforced `:286-290`). The named exploit plus `dblink_exec`, `lo_export`, `pg_read_file`, `pg_sleep`, `copy`, `xp_cmdshell` are blocked; covered by `tests/security/test_nl2sql_validator.py`.
- Evidence — not done: `_run_sql_with_guards` (`modules/nl2sql/service.py:271-303`) still executes on a `create_engine()` built from the source's own credentials with only `statement_timeout` + an EXPLAIN dry-run — no `SET TRANSACTION READ ONLY` and no separate read-only role. A denylist is enumerated, not exhaustive; a side-effecting function not on the list would still pass the SELECT gate and run with write-capable credentials.
- Notes: the concrete July payload is blocked, but the review explicitly asked for the denylist "behind a read-only DB role" as defense-in-depth — that layer is absent, so an incomplete denylist has no backstop.

#### F039 — PARTIAL
**`/api/v1/memory` authenticates but never uses `ctx.workspace_id` — process-global store keyed on caller-supplied `session_id`.**
- Fix: `19ea48825` (W2) — write path only.
- Evidence: `_scoped_session` prefixes `{workspace_id}::` (`orchestrator/api/memory.py:68-81`) but is applied ONLY on the write path `store_memory` (`memory.py:101`). The read path `retrieve_memories` (`/retrieve/{session_id}`) passes the RAW path-param `session_id` straight to `manager.retrieve_memory` (`memory.py:143`; manager keys on it via access_optimizer at `modules/memory/storage/manager.py:189-194`), and `consolidate_memories` passes raw `session_id` (`memory.py:283/298`). Net effect: (a) **functional break** — a workspace that stored "abc" (now keyed `{ws}::abc`) then retrieves "abc" finds nothing; (b) **residual cross-tenant read** — workspace B can call GET `/retrieve/{ws_A}::abc` and the process-global store returns A's memories, since retrieve never scopes and workspace ids are not secrets. The W2 test (`tests/security/test_prd172_tenant_isolation.py:378-396`) only unit-tests `_scoped_session`, not the retrieve/consolidate paths.
- Notes: write scoped, read/consolidate not — the store↔retrieve asymmetry both breaks intra-tenant recall and leaves a cross-tenant read oracle. Report §14 flagged this endpoint as a delete-candidate; it was neither fully scoped nor deleted. See §4.C.14.

#### F045 — FIXED
**`api/context.py` mounts seven endpoints with no auth dependency and an unscoped SELECT COUNT(*).**
- Fix: `19ea48825` (W2).
- Evidence: all previously-unauthenticated endpoints now carry `get_request_context_hybrid`: `add_to_context:58`, `get_context_stats:89`, `get_rag_performance:116`, `get_context_sources:131`, `get_recent_queries:146`, `get_context_patterns:160`, `get_context_system_health:197`. The unscoped COUNT is closed: `get_retrieval_stats(db, workspace_id)` scopes the documents count to the caller's workspace (`modules/rag/service.py:1171-1192`), passed from `/stats:97-98` and `/system/health:203-204` (`admin_all_workspaces` → unfiltered aggregate).

#### F011 — UNVERIFIABLE
**CRITICAL(adj) — mem0 fork PRD-156 token-auth + PRD-159 metadata patches live only on branch `fix/pool-exhaustion@16b27eb2`, not fork main — deployed OpenMemory may be unauthenticated and metadata-dropping.** (sibling repo `automatos-mem0`)
- Claimed fix: `35a0d707e` (W3 PRD-173, PR #462) — delivered as a human runbook.
- Evidence: `docs/runbooks/W3-HUMAN-STEPS.md` §3 (lines 129-177) documents merging `automatos-mem0 fix/pool-exhaustion@16b27eb2` → fork main, pinning Railway to the merged SHA, and a curl 401-without-token boot probe. The actual fix lives in the separate fork and a live Railway deploy — NOT in this pinned tree. No security boot probe landed in-repo — the only in-repo mem0 probe is a reachability/circuit-breaker check (`services/heartbeat_service.py:186-201` → `client.run_health_probe()`; `config.py:850-854`), which pings for liveness and does NOT assert auth-required.
- Notes — what would verify it: (1) fork main at the pinned SHA contains the token-auth + metadata patches; (2) curl to the deployed OpenMemory base returns 401 for an unauthenticated router call. Both are outside this read-only tree. Flag for the memory dossier team.

#### F012 — PARTIAL
**CRITICAL(adj) — committed Clerk auth artifact `tests/e2e/.auth/user.json` (expired 60s JWTs + `__clerk_db_jwt`) — committed-artifact and re-commit hazard, not live credential.**
- Fix: `35a0d707e` (W3) — mitigations only.
- Evidence: mitigations landed: `.gitignore` now lists `tests/e2e/.auth/` (`.gitignore:133-134`); full-history secret-scanning lane added (`.github/workflows/gitleaks.yml` with fetch-depth:0) plus CodeQL. BUT the artifact is STILL committed: `git cat-file -t HEAD:tests/e2e/.auth/user.json` returns blob and `git ls-files` lists it; it still contains `__clerk_db_jwt` and clerk tokens. A .gitignore is a no-op for an already-tracked file — no `git rm --cached` was done — so the re-commit hazard is NOT closed. `gitleaks.yml`'s own comment concedes it: "once tests/e2e/.auth/user.json is purged, this job goes and stays green" — the lane is currently red pending a human history purge.
- Notes: file removal (named in the expected fix) did not happen; history purge/revocation is human-deferred (§4.D.3). JWTs were short-lived (60s) so live-credential risk is low.

#### F058 — FIXED
**Merchant Shopify Admin token (147 write-scopes) stored plaintext in `workspace.settings` while the docstring claims encryption.** (July: `shopify.py:357-369`)
- Fix: `35a0d707e` (W3).
- Evidence: `/connect` now encrypts the Admin token before persisting: `settings['shopify_access_token'] = _encrypt_secret(request.access_token)` (`orchestrator/api/shopify.py:208`), where `_encrypt_secret` delegates to the canonical Fernet EncryptionService (`shopify.py:45-49` → `core/credentials/encryption.py`). Round-trip covered by `tests/test_prd173_shopify_token_encryption.py`.
- Notes: no production `_decrypt_secret` caller exists (the settings-stored copy looks vestigial; the Composio path uses a separate encrypted credential store); encryption depends on `CREDENTIAL_ENCRYPTION_KEY` (else a local `.credential_key` file, else auto-generated) — same posture as all platform secrets.

#### F008 — FIXED
**CRITICAL — ClerkProvider mounted unconditionally with `auth.protect()` on all non-public routes, zero edition flag — UI serves nothing without a Clerk tenant (open-core blocker).** (July: `providers.tsx:33`)
- Fix: `4ba8cd320` — W5 PRD-175, PR #482.
- Evidence: frontend: ClerkProvider is now inside an edition-gated AuthBoundary — `isLocal` renders LocalAuthProvider with no Clerk symbol, `isSaaS` renders ClerkProvider (`frontend/components/providers.tsx:28-75`; spot-checked: AuthBoundary at `:29`, LocalAuthProvider at `:31`, boundary mounted at `:88`); middleware is edition-conditional (`frontend/middleware.ts`: `localMiddleware = NextResponse.next()`, `auth.protect()` only in saasMiddleware); edition read from `NEXT_PUBLIC_AUTH_EDITION` with unknown→saas fail-safe (`frontend/lib/auth-edition.ts`). Backend: `AUTH_EDITION` flag (`config.py:157-166`) forces `REQUIRE_AUTH=false` in local; `validate_auth_edition` boot guard (`config.py:1064-1104`); `hybrid.py` resolves the loginless request to `DEFAULT_WORKSPACE_ID` (`core/auth/hybrid.py:787,821-824`). Local edition boots with no Clerk tenant.
- Notes: default stays saas (correct posture for the running product); local is opt-in. The `validate_auth_edition` boot guard shares the run_stage swallow weakness (see F004) but the frontend gating + `REQUIRE_AUTH=false` make local functional regardless.

#### F075 — FIXED
**Literal 'admin' role domain-locked to @automatos.app in code.** (July: `clerk.py:201`)
- Fix: `4ba8cd320` (W5).
- Evidence: the hardcoded domain lock is now configuration: `staff_domain = config.PLATFORM_STAFF_EMAIL_DOMAIN` and `is_platform_staff = email.endswith('@'+staff_domain)` (`orchestrator/core/auth/clerk.py:203-204`); config default `automatos.app`, env-overridable (`config.py:172-173`). A self-hosted operator sets their own staff domain; the defence-in-depth admin-promotion gate is preserved.

#### F053 — NOT DONE
**Public widget keys skip the domain check when the Origin header is absent — a leaked `ak_pub_` key works from curl.** (July: `widgets/auth.py:183`)
- Evidence: unchanged from baseline and confirmed present: `orchestrator/api/widgets/auth.py:182-183` reads `origin = _extract_origin(request)` then `if origin and not ApiKeyService.check_domain(...)`. `_extract_origin` (`core/auth/hybrid.py`) returns None when neither Origin nor Referer is present, so a curl request with no headers short-circuits the guard and a leaked `ak_pub_` key authenticates from anywhere. The only commit to touch this file since baseline (`0710e4cae`, W4) addressed other findings; no wave PRD cites F053.
- Notes: fix shape — for a public key with non-empty `allowed_domains`, treat an absent Origin/Referer as deny rather than pass.

### 2.4 Cluster: operating-graph-tools-learning

#### F015 — PARTIAL
**Learned edges reach only the FILTERED strategy and offline eval; prompt-catalog chain hints gated behind `TOOL_ROUTING_GRAPH` default-false; edge reads carry no workspace filter.** (July: `modes.py:40-46`; `config.py:680,693`; `graph_router.py:150-171`)
- Fix: `998f79a68` (S5 workspace filter) + `8d12c4a5f` (S4, test-only) + `8db62cc95` (S6 eval) — W7 PRD-177, PR #494.
- Evidence — workspace-filter half genuinely fixed: `GraphRouter.rank_chains` now takes a REQUIRED `workspace_id` kwarg (`orchestrator/modules/tools/discovery/graph_router.py:129-142`); edge reads scoped `ToolRoutingEdge.workspace_id == workspace_id` else IS-NULL-only (`:315-320`), affinity reads likewise (`:359-384`), cache key includes ws (`:72-80`). Live call sites thread real ids: `platform_actions.py:142-147`, `smart_tool_router.py:228`, and the FILTERED-strategy loader `modules/context/sections/tools.py:200-208`. Per-tenant reads are coherent with per-workspace writes (`edge_builder.py:135-149`).
- Evidence — chain-hints half unchanged in posture: `TOOL_ROUTING_GRAPH` still default false (`config.py:757`); commit `8d12c4a5f` is test-only and its message says the flag flip is gated on the S6 uplift eval. S6 exists: `orchestrator/evals/operating_graph_uplift.py` (17KB), non-required CI job (`test.yml:218-246`). Learned edges still reach the primary chat path (FILTERED) under `SEMANTIC_TOOL_ROUTING` default true (`config.py:751`; `smart_tool_router.py:225-231`) — unchanged from July's adjusted note.
- Notes: the flag stays OFF **on evidence, not neglect** — the S6 commit honestly reports mean uplift −32.9 points on the bundled 47-case offline fixture, below the +5 gate, so chain hints remain dark. Right call per the eval; the tenancy defect is the part that is fully closed.

#### F016 — FIXED
**Composio telemetry logs the meta-tool name — the 856-app surface collapses to a single `composio_execute` graph node; cross-app used_after/affinity learning impossible.** (July: `telemetry.py:56-64`)
- Fix: `fcfccc6d3` (W7 S1, PR #494).
- Evidence: `resolve_action_name(tool_name, parameters)` at `orchestrator/modules/tools/execution/telemetry.py:20-44` pulls the real action from `parameters['action'|'action_name']` for composio_execute, uppercased; `write_telemetry` logs it as `action_name` (`telemetry.py:70,89`) and derives `app_name` from the RESOLVED prefix (SLACK_SEND_MESSAGE → SLACK, `:72-79`). Fired for every execution via the finally-block in `unified_executor.py:595-603`. Key shape matches the executor (`exec_composio.py:158`). The second learning path also resolves: `ToolRouter._record_tool_signal` at `tool_router.py:812-826`. Per-action composio tools dispatch with `{'action': tool_name,...}` (`unified_executor.py:515-521`). The edge builder consumes `ToolExecutionLog.action_name` unchanged, so used_after/affinity edges now form on per-action nodes. Test: `orchestrator/tests/test_prd177_composio_telemetry.py`.
- Notes: keys-only privacy posture preserved (`input_parameters` still logs only parameter keys, `telemetry.py:92`); malformed calls without an action key fall back to the meta-tool name rather than crashing.

#### F017 — FIXED
**Chat threads only `user_id` into caller_context — no user_query/turn_id, so intent-conditioned edges never materialize.** (July: `chatbot/service.py:1317`)
- Fix: `ff7730417` (W7 S2, PR #494).
- Evidence: `build_tool_caller_context` at `orchestrator/consumers/chatbot/service.py:97-135` constructs `{user_query, conversation_id, turn_id, user_id, prior_action}`. The live chat loop mints one `_turn_id` per user turn (`service.py:1277-1281`) and the tool callback threads the full context into every `execute_and_format` call (`service.py:1340-1352`), with `conversation_id=chat_id` at the `_stream_tool_loop` call site (`:2111-2116`) and `prior_action` tracked per resolved action (`:1353-1355`). Telemetry persists conversation/turn ids into `router_decision` (`telemetry.py:133-136`); the edge builder consumes them: `user_query` for intent clustering (`edge_builder.py:153,403-409`), `turn_id` preferred for used_after pairing (`:164-167`). The two remaining un-threaded `execute_and_format` sites (`service.py:2235` plain `stream_response` fallback; `:2311` `_execute_pretriggered_tools`) have zero non-test callers.
- Notes: `succeeds_for_intent` affinities can now materialize from real chat traffic; with F016, cross-app used_after pairs form in exact turn order. The dead legacy paths should eventually be deleted rather than left un-threaded.

#### F018 — FIXED
**`ComposioActionMetadata` writer has no scheduler and its claimed sync endpoint does not exist; destructive-action confirm gate fails open on the empty table.** (July: `jobs/sync_composio_actions.py`)
- Fix: `52cb4e114` (W7 S3, PR #494).
- Evidence: scheduler: new `orchestrator/services/composio_sync_scheduler.py` registers a daily cron (default 04:00 UTC, `COMPOSIO_SYNC_HOUR_UTC` `config.py:770`) running `sync_all_composio_actions` on the shared UnifiedScheduler; wired in `main.py:414-423` behind `COMPOSIO_SYNC_ENABLED` default TRUE (`config.py:769`). Fail-closed gate, default ON via `COMPOSIO_DESTRUCTIVE_FAIL_CLOSED=true` (`config.py:768`): empty-metadata-table branch denies destructive-reading intents (`action_capability_filter.py:272-295`); filter-unavailable and exception branches likewise via `_destructive_fail_closed` (`tool_router.py:953-977`, applied `:1000-1014`, `:1036-1053`). The gate sits on the live chat composio path: `execute_and_format` routes composio tools through `execute_tool_with_validation` with `allow_destructive=False` when `original_intent` is present (`tool_router.py:664-673`), and chat always threads `original_intent=user_text` (`consumers/chatbot/service.py:1344`). Destructive-keyword heuristic consolidated into `taxonomy.intent_is_destructive` / `DESTRUCTIVE_INTENT_KEYWORDS` (`taxonomy.py:263-267,429-439`). Tests: `orchestrator/tests/test_prd177_metadata_gate.py`.
- Notes — two honest caveats: (1) the fail-closed decision on an unclassified action keys off the INTENT TEXT via an 8-keyword heuristic — a destructive ACTION under neutral wording ("issue a refund", "create a 10% discount") still passes on an empty/unsynced table; the July §10 refund/discount exemplar is fully closed only once the daily sync populates rows and `metadata.destructive` applies (`action_capability_filter.py:310-313`). Deliberate cold-start trade-off; the residual window is real (§4.C.13). (2) The docstring-claimed "POST /api/admin/sync-composio-actions" (`jobs/sync_composio_actions.py:11`) still does not exist — stale documentation, no longer a functional hole.

#### F047 — NOT DONE
**NL2SQL admin-instructions semantic layer (BINDING Q59 headline) is read-wired but has no product write path.** (July: `nl2sql/service.py:670-674`)
- Evidence: no commit or wave PRD cites it. The read side is wired: the generation prompt consumes `semantic_layer['instructions']` or `['business_context']` (`orchestrator/modules/nl2sql/query/nl2sql_service.py:199-207`); the per-connection layer is injected at generation (`modules/nl2sql/service.py:496-500`). But the only product write path — POST `/{source_id}/semantic` (`api/database_knowledge.py:294-333`) → `update_semantic_layer` (`modules/nl2sql/service.py:654-678`) — accepts only metrics + dimensions, and line 670 REPLACES the whole `semantic_layer` dict with `{'metrics','dimensions','updated_at'}`. The frontend editor (`frontend/components/knowledge/SemanticLayerBuilder.tsx`) likewise has metrics/dimensions state only, no instructions field.
- Notes: slightly worse than write-path-absent — because the updater overwrites the whole JSONB dict, any `instructions` value set out-of-band (SQL console) is silently clobbered the next time an admin saves metrics/dimensions.

#### F048 — PARTIAL
**HARNESS self-management dark-launched off-by-default; approved prescriptions escalate admins to a surface that always returns HTTP 409 — never actuate.** (July: `config.py:583`; `harness_service.py:1679`)
- Fix: `0b07357bc` (W9 PRD-179, PR #496).
- Evidence: the actuation path is real: `/approve` now routes through the W4 policy plane — `orchestrator/api/harness_commands.py:188-290` (`_approve`): idempotency via the applied-tasks ledger (`:215-222`), unresolved-target refusal (`:226-231`), `evaluate_approval` call (`:238-240`), actuation via `_auto_apply_prescription` (`:252`), board task marked done WITH the actuation result (`:263-267`), ledger entry with policy_verdict (`:268-284`). Test: `orchestrator/tests/test_prd179_harness_actuation.py`. BUT the whole surface remains dark by default: `HARNESS_SELF_MANAGEMENT_ENABLED` defaults false (`config.py:627`); with defaults, `handle_harness_command` returns disabled (`harness_commands.py:133-134`), mapped to HTTP 409 (`api/harness.py:72-75`) — the July symptom still reproduces on out-of-the-box config. PRD-179 S3 explicitly specified "enable behind the existing config flag" — the flag decision was made: stay dark.
- Notes — quality flag: the commit and docstring claim "a policy decline still blocks actuation (fail-safe)", but `evaluate_approval` is invoked with `override_auto_approve=True` (`harness_commands.py:238-240`) and `approval_policy.py:163-164` short-circuits that to unconditional approve — the decline branch at `:241-250` is unreachable dead code. The policy-plane routing is ceremonial; effective gates are admin authz (`:136-148`), the idempotency ledger, and unresolved-target refusal. Semantically defensible (a human admin's explicit approve IS the approval), but the fail-safe claim as written is false. See §4.C.9.

#### F049 — FIXED
**Mission-synthesis flywheel ingests three arbitrary COMPLETED runs with no ORDER BY and no already-ingested exclusion — starves once more than three accumulate.** (July: `coordinator_service.py:1136-1152`)
- Fix: `61843897e` (W9, PR #496).
- Evidence: `_save_pending_output_documents` (`orchestrator/services/coordinator_service.py:1150-1181`): candidates filtered SQL-side to COMPLETED runs carrying NONE of three terminal markers — `config['output_document_id']` / `['output_ingest']` / `['output_ingest_failed']` all IS NULL (`:1166-1171`) — with ORDER BY created_at DESC (`:1172`) and batch via `Config.FLYWHEEL_INGEST_BATCH` (`:1173`; default 3, `config.py:884`). Markers: success stamps `output_document_id` (`:847`), workspace opt-out stamps `output_ingest` (`:802`), exceptions stamp `output_ingest_failed` so a poison run drops out next tick (`:866-879`). Test: `orchestrator/tests/test_prd179_flywheel_order.py`.
- Notes — residual markerless edge, narrower but same failure mode: a COMPLETED run with zero VERIFIED tasks returns None with NO marker (`coordinator_service.py:820-821`), and `ingest_agent_output` has fail-soft return-None paths that also leave no marker (`services/knowledge_flywheel.py:159,166,223`). Batch-size such runs at the newest created_at positions would permanently occupy the DESC-ordered batch and re-starve older backlog. Worth a marker on the not-ingestable shapes. See §4.C.22.

#### F070 — FIXED
**`rag_feedback` stores signals nothing feeds back into ranking or any eval (write-only endpoint).** (July: `rag_feedback.py:50-70`)
- Fix: `28fb99e26` (W9, PR #496).
- Evidence: rag_feedback now shapes the live retrieval hot path: `RAGService.retrieve` applies `_apply_feedback_penalty` after rerank (`orchestrator/modules/rag/service.py:331-334`; spot-checked `:334,:415,:459,:478`). `_negative_feedback_doc_ids` (`:415-450`) does a workspace-scoped read of docs with thumbs_down or rating ≤ `RAG_FEEDBACK_NEGATIVE_RATING_MAX` within `RAG_FEEDBACK_LOOKBACK_DAYS` (UNNEST over document_ids, CAST-style binds, fail-soft empty set). `_apply_feedback_penalty` (`:459-509`) multiplies score/similarity/rerank_score/rrf_score by `RAG_FEEDBACK_PENALTY_FACTOR`, immutably, then re-sorts (`:500-504`). Defaults ON: factor 0.5, rating_max 2, lookback 90d (`config.py:974-978`); disabled only if factor ≥ 1.0. Write-read contract matches the endpoint exactly (`api/rag_feedback.py:60-80`). Test: `orchestrator/tests/test_prd179_rag_feedback_rank.py`.
- Notes — scope, not defects: the loop is negative-only (thumbs_up/corrections still influence nothing — no boost), and the finding's "or any eval" half is untouched. The named write-only defect is genuinely closed on the live path.

#### F074 — NOT DONE
**Dispatcher enum enrichment is alphabetical top-50 rather than learned-edge ranked.** (July: `tool_router.py:322-358`)
- Evidence: still alphabetical: `orchestrator/modules/tools/tool_router.py:323-331` — `.order_by(ComposioActionCache.app_name, ComposioActionCache.display_name).limit(50)` with max 10 per app (`:349`), unchanged since May (`b2e69a1f3`). The platform dispatcher's enum narrowing remains pure-semantic via ActionSemanticIndex (`_rank_actions_for_dispatcher`, `tool_router.py:125-156`, applied `:404-412`) — W7 S4/S5 added no learned-edge re-rank to either enrichment; learned edges reach tool SELECTION only through `GraphRouter.rank_chains` in the FILTERED strategy and the flag-gated chain hints, never the dispatcher enum ordering. No commit or PRD cites F074.
- Notes: with F016/F017 now feeding per-action, intent-conditioned edges, this is the next cheapest place learned data could reach the live surface — the data-side prerequisites are in place, the read-side re-rank is not.

#### F054 — NOT DONE
**Phantom `Skill.priority` ordering — advertised but never effective.**
- Evidence: the phantom stands exactly as described. `modules/context/sections/skills.py` sorts active skills by `getattr(s, "priority", 0) or 0` (`:63-67`) and its docstring advertises "the primary skill (highest-priority active skill) is rendered uncapped" (`:24-25`) — but the Skill ORM model has NO priority column (`orchestrator/core/models/core.py:340-385`) and the agent_skills association table is bare (`core.py:29-32`). getattr returns 0 for every skill, the sort is a constant-key no-op, and "primary" is whatever unordered relationship-load order yields. No migration adds the column; no commit or PRD cites F054. (The priority that IS effective — `AgentAssignedPlugin.priority`, `plugin_context_service.py:53` — is a different mechanism, for plugins.)
- Notes: two honest fixes exist: delete the phantom sort + docstring claim (truthful, zero risk), or add the column and an ordering UI (real feature). As-is, which skill gets the uncapped "primary" token budget is nondeterministic — a silent quality lever on multi-skill agents.

#### F055 — NOT DONE
**Triple-format `tools_schema` incoherence — the same column holds three incompatible formats across the tool surface.**
- Evidence: intact, with mutually-blind consumers observable right now: (1) the writer `skill_materializer.py:124-127` normalizes YAML into DICT form `{"tools": [...]}`; (2) consumer `modules/context/sections/skills.py:103-114` accepts ONLY dict form and reads top-level `tool['name']`; (3) consumer `modules/agents/registry/agent_registry.py:103-118` accepts ONLY LIST form of OpenAI-style `{'function': {'name': ...}}` specs — plus a json.loads branch for STRING-serialized rows (`:107-108`), the third format. A materializer-written skill's tools are invisible to the registry extraction and a legacy list-form skill is invisible to the prompt section. No normalizer added; no commit since the waves touches `tools_schema`; no commit or PRD cites F055.
- Notes: practical effect — which tools an agent is told it has (prompt section) vs which the registry reports can disagree for the same skill row, depending on which era wrote it. A single normalize-on-read helper (or a one-shot migration to the materializer's dict form) would close it.

#### F082 — NOT DONE
**Learning decoys: `modules/learning` patterns/ and feedback/ are 0-byte packages, PlaybookMiner mines hardcoded demo sequences, `modules/evaluation` is a TODO scaffold with zero importers; real loops live elsewhere.**
- Evidence: all three decoys still present: `orchestrator/modules/learning/patterns/__init__.py` and `feedback/__init__.py` are 0-byte files; `PlaybookMiner._fetch_sequences` still returns hardcoded demo sequences with "TODO: Implement real sequence fetching" (`modules/learning/playbooks/miner.py:24-34`); `modules/evaluation/__init__.py` is still the TODO scaffold (`:24`, `__all__ = []` `:29`). The sole non-test importer of `modules.learning` remains the dead-on-arrival api_playbooks surface (`api/api_playbooks.py:12`, mounted via `main.py:55`) plus the `modules/__init__.py` re-export. PRD-184 never authored; git log --grep 'PRD-184|F082' returns nothing.
- Notes: the July calibration holds on the adjacent adjusted point — `fails_for_intent` negative affinities ARE consumed in ranking (`graph_router.py:413`) while `failed_after` edges remain write-only bookkeeping, by design. The directory literally named "learning" still contains none of the platform's real learning loops (which live in `harness_service.py`, `core/services/edge_builder.py`, `modules/tools/discovery` — all actively hardened by W7/W9), so the misleading-signpost argument for deletion has only strengthened.

### 2.5 Cluster: memory-field-codegraph

#### F020 — FIXED
**Field tools bind field_id to an arbitrary running Mission — `.first()` on state=='running' with no ordering and no link to the calling task (cross-Mission bleed).** (July: `platform_executor.py:842-856`)
- Fix: `e180a613a` (W8 PRD-178, PR #495).
- Evidence: the `.first()`-on-any-running-Mission block is deleted, not shimmed (git show confirms removal). Current injection at `orchestrator/modules/tools/discovery/platform_executor.py:897-904` reads field_id ONLY from `caller_context['field_context']`, built on the serial DB path from the calling task's OWN run: `orchestrator/services/coordinator_service.py:1875` (field_id from `run.config`) and `:2005-2012` (`field_context={field_id, mission_id=run.id}`), threaded via `:1458` and `:2046` → `agent_factory.py:1041-1043,1061` → `unified_executor.py:358,484-489` → platform_executor. With no threaded context there is no injection; `platform_field_query` then falls back to workspace-persistent recall (`handlers_field.py:70-84`), which also removes the running-Mission-shadows-workspace-recall symptom. field_id is not schema-required (`actions_field.py:33,71`), so no LLM dead-end. Test: `orchestrator/tests/test_prd178_field_binding.py`.
- Notes — residual: the IDENTICAL `.first()`-on-state=='running' arbitrary-Mission lookup survives one block up for `_agent_id` on graph/document tools (`platform_executor.py:864-888`, PRD-124) — cross-Mission agent-identity bleed persists for graph-tool team scoping. Same defect family, outside F020's cited scope. See §4.C.2.

#### F021 — FIXED
**HEARTBEAT mode deliberately excludes memory — recurring agents are amnesiac by config; PRD-164's Q60 read-half never merged.** (July: `modes.py:76-88`)
- Fix: `a23934b47` (W9, PR #496).
- Evidence: HEARTBEAT_AGENT mode now includes a `field_memory` section (`orchestrator/modules/context/modes.py:80-88`; PLANNING too at `:144-152`). The section is real code: `modules/context/sections/field_memory.py:32-126` reads the workspace-persistent field via `query_workspace` and renders through the shared `field_scoring.budget_results/format_digest` pipeline; registered in `sections/__init__.py:48`; consumed by the one assembler (`modules/context/service.py:95,376`). `query_workspace`'s agent_id defaults safely (`adapters/vector_field.py:388-395`). Heartbeat agents actually hit this path: `services/heartbeat_service.py:928` passes `ContextMode.HEARTBEAT_AGENT` into `execute_with_prompt` with no system_prompt, so ContextService builds the prompt (`agent_factory.py:910-930`). No feature flag gates it. Test: `orchestrator/tests/test_prd179_field_read.py`.
- Notes: fixed as scoped (the read-half) — recurring agents now inherit workspace-field digests. The chat/user "memory" section is still deliberately excluded from heartbeat mode to keep the tick lean (`modes.py:76-79`) — heartbeat agents get accumulated Mission patterns, not conversational memory. Section render swallows all exceptions to '' (`field_memory.py:56-62`) — correct for a prompt build, but a backend outage degrades silently to no digest.

#### F022 — FIXED
**Codegraph webhook auto-reindex can never fire — `auto_reindex` has no setter and is only ever read; agents reason over stale graphs.** (July: `codegraph.py:683`)
- Fix: `47587e453` (W13 PRD-183 S4, PR #500).
- Evidence: `auto_reindex` now has three setter paths: `CodeGraphService.set_auto_reindex` — workspace-guarded UPDATE (`orchestrator/modules/codegraph/codegraph_service.py:1477-1502`, WHERE id AND workspace_id, rowcount-checked; spot-checked `:1477`); PATCH `/projects/{project_id}/auto-reindex` (`orchestrator/api/codegraph.py:510-533`, hybrid-auth, ctx.workspace_id); and agent tool `platform_codegraph_set_auto_reindex` (`actions_codegraph.py:215-240`, handler `handlers_codegraph.py:321-344` resolving project by name within the caller's workspace). The GitHub push webhook (`api/codegraph.py:667-741`) filters rows on `auto_reindex` and fires background `index_github_project` per match — the flag it reads is now settable, so the path can fire. Test: `orchestrator/tests/test_prd183_s4_codegraph_reindex.py`.
- Notes — two caveats: (1) the webhook's signature check is skip-if-unset — with `GITHUB_WEBHOOK_SECRET` unconfigured, unsigned payloads are accepted (`api/codegraph.py:679-684`); unauthenticated reindex-triggering is a cost/DoS surface (§4.C.15). (2) End-to-end firing (GitHub-side webhook registration + a real push) needs runtime — the code path is complete but live firing was not observable here.

#### F062 — FIXED
**Retrieval-trace inspector mutates the field it observes via the writing field.query path.**
- Fix: `5d6e74b22` (W8, PR #495).
- Evidence: the trace inspector (POST `/missions/{mission_id}/field/query`, `orchestrator/api/missions.py:1069`) now queries with `record_access=False` (`missions.py:1094`). `record_access` is part of the SharedContextPort contract (`core/ports/context.py:59-68`), threaded through the instrumentation wrapper (`modules/context/instrumentation.py:171-190`, which also tags the op 'trace' so traces don't skew query KPIs) and gates Hebbian reinforcement in the vector adapter: `_reinforce_batch` only runs when record_access is true (`adapters/vector_field.py:279-283,460-466`). Default stays True, so live agent queries still reinforce. Test: `orchestrator/tests/test_prd178_field_trace_readonly.py`.
- Notes: clean fix — a real read-only path, not a snapshot shim; observing the field no longer mutates the access_count/strength it reports.

#### F063 — FIXED
**Field compaction lacks a workspace-scoped resume cursor.** (July: `adapters/vector_field.py`, report §12.5)
- Fix: `76ec9fb65` (W8, PR #495); sibling `89ce5dee8` (promotion seam).
- Evidence: `compact()` now takes workspace_id + resume_offset + max_scan and does a bounded, workspace-filtered Qdrant scroll returning `CompactionResult.next_offset` (`orchestrator/modules/context/adapters/vector_field.py:496-568`; scan budget `FIELD_COMPACTION_MAX_SCAN`). The cursor is persisted per workspace as a system_settings row (`modules/context/compaction_cursor.py:31-85`, category 'field_compaction'). The hourly coordinator tick enumerates workspaces with field data, loads the cursor, compacts, saves next_offset, commits per workspace with rollback isolation (`services/coordinator_service.py:3530-3590`); the sole non-test `compact()` caller uses the new signature (`:3579`). Test: `orchestrator/tests/test_prd178_field_compaction.py`.
- Notes: the sibling field-to-durable promotion (the arm that stops compaction hard-deleting before promotion) is real and default-ON: `orchestrator/jobs/promote_field_memory.py` (345 lines) scheduled in `main.py:400`, `FIELD_PROMOTION_ENABLED` default true with a taint gate on untrusted provenance (`config.py:856-874`). Minor: the hourly throttle is in-process per worker, so multi-worker deployments can run overlapping sweeps — deletes are idempotent and the cursor is last-writer-wins, so this is waste not corruption, but the cursor has no lock.

#### F064 — NOT DONE
**Codegraph fallback INSERT omits NOT NULL workspace_id.** (July: report line 155)
- Evidence: unchanged: when batch embedding fails, `orchestrator/modules/codegraph/codegraph_service.py:1016-1031` inserts into `codegraph_symbols` with columns (project_id, symbol_type, name, qualified_name, file_path, line_number, signature, docstring, code_snippet, metadata) — no workspace_id — while `codegraph_symbols.workspace_id` is `nullable=False` (`orchestrator/alembic/versions/20260218_fix_codegraph_schema_v2.py:85`). Every fallback row violates NOT NULL, so the "store without embeddings" fallback can never succeed: an embedding outage silently loses the whole batch (error logged at `:1015`, then the fallback itself raises). Untouched since PR #303 (pre-waves); no commit or wave PRD cites F064.
- Notes: doubly broken — besides the missing column, `metadata` is bound as Python `str(dict)` (single-quoted repr, not JSON) to a JSONB column (`:1029`), a second guaranteed failure. The primary path (`:986-1002`) binds both correctly — the fix is copying two lines. Real autonomy cost: exactly when embeddings degrade, indexing loses symbols entirely instead of degrading to non-semantic search.

#### F065 — UNVERIFIABLE
**Field/codegraph correctness cluster member (F062–F065 range) — never individually described in the report body.** (July: report line 155)
- Evidence: F065 is never individually defined anywhere in the pinned tree: the report names it only inside the theme range at `reports/PLATFORM_OS_REVIEW_2026-07-01.md:155`, with exemplars only for F062 and F064; Appendix B (unverified leads) and Appendix C (statistics) carry no F065 entry; no PRD cites it; git log --grep F065 returns zero commits. There is no defect description to verify a fix against.
- Notes: what would verify it: the review run's underlying findings register. Of the named cluster, F062 and F063 are fixed and F064 is not-done; the adjacent field-correctness gap visible in July code (compaction hard-deleting before promotion) was closed by `89ce5dee8` — but matching F065 to any of these would be conjecture.

#### F087 — FIXED
**No agent-side codegraph write/index/reindex tool — onboarding a repo is POST /index/github only; ask-Auto vs manual-UI parity broken.** (July: `codegraph.py:115`)
- Fix: `47587e453` (W13 S4, PR #500).
- Evidence: three agent-side write tools exist and are wired end-to-end: `platform_codegraph_index`, `platform_codegraph_reindex`, `platform_codegraph_set_auto_reindex` registered with promoted=True / permission_level='write' and sane schemas (`orchestrator/modules/tools/discovery/actions_codegraph.py:165-240`; spot-checked `:165,:192`). Handlers do real work, workspace-scoped: index runs the clone→parse→embed pipeline for the executor's workspace (`handlers_codegraph.py:247-277`), reindex resolves by name within the workspace (`:285-313`), set_auto_reindex flips the webhook flag (`:321-344`). Dispatch map wired at `platform_executor.py:524-526`; delivery rides the same promoted first-class-schema path as the read tools (`tool_router.py:457-470` via `action_registry.to_first_class_schemas:138-158`; not admin-filtered). "Index this repo and tell me what calls X" is now possible through Auto's own surface.
- Notes — two residuals: (1) parity is not total — the UI can delete a project but there is no agent-side delete tool (destructive op, plausibly deliberate; unstated); (2) `codegraph_index` awaits the full indexing pipeline inline in the tool call — same synchronous model as the REST route (`api/codegraph.py:115-120`), so no path asymmetry, but a large repo can outlive the tool-loop timeout in both.

### 2.6 Cluster: channels-widget-shopify

#### F013 — FIXED
**CRITICAL(adj) — standalone Shopify Remix app does not build (no app/routes.ts); subscribed webhook URIs would 404; GDPR handlers no-op; OAuth redirect unwired; install-time provisioning dead.** (sibling repo `automatos-shopify`)
- Fix: platform: `2c7028b00` + `31c48fe47` (W11 PRD-181 S3+S4, PR #499), `02dd93bec` (W13, PR #500); sibling repo: `3f90b9a` + `8733604` + `67abb40` (PR #10, merge `8dd304c`).
- Evidence — platform seams (pinned tree): fail-closed internal-key auth `orchestrator/api/shopify.py:85-101`; machine-to-machine GDPR surface POST `/api/verticals/{v}/gdpr/erase-subject|erase` + GET export at `orchestrator/api/verticals.py:158-248` (spot-checked: `:158` erase-subject route), resolving workspace by shop domain with 404-never-fallback (`:34-77`), delegating to the W11 cascade `services/gdpr_service.py:198,219,252` (field-memory + mem0 + SQL legs, gaps reported, audited); generic provision route `api/verticals.py:95` mounted via `router_manifest.py:53`; `/events` contract + catalog re-sync `api/shopify.py:355-396`.
- Evidence — sibling repo (`automatos-shopify` origin/main @ `8dd304c`, PR #10 merged 2026-07-03, CI job "routes + build + typecheck + test" SUCCESS, run 28685161381): `app/routes.ts` with flatRoutes() (the exact build fix, `3f90b9a`); `shopify.app.toml` webhook subscriptions map to real route modules (`webhooks.compliance.tsx`, `webhooks.catalog.tsx`, `webhooks.orders.create.tsx`, `webhooks.app.uninstalled.tsx`); the compliance handler HMAC-verifies then calls the platform GDPR endpoints (`webhooks.compliance.tsx:35-49` → `/api/verticals/shopify/gdpr/*`, repointed in `67abb40`); OAuth wired — `auth.callback.tsx:21` calls provisionAndStore → `/api/verticals/shopify/provision`; CI build gate runs `npx react-router routes` + build + tests (`8733604`).
- Notes: sibling evidence is outside the pinned tree (read-only; build proven by the sibling's green CI, not a local run). Quality caveats: customers/data_request export is workspace-scoped, not per-customer (documented at `api/verticals.py:229-233`); the compliance handler materializes+audits the export bundle but delivers it nowhere; actual webhook registration with Shopify (app-config deploy) is runtime state this review cannot verify.

#### F026 — NOT DONE
**Channel replies gated on the legacy `workspace.settings.integrations` bag being non-empty — new-style `channel_connections` process inbound but drop replies.** (July: `webhooks.py:417,447,478`)
- Evidence: `orchestrator/api/webhooks.py:371` still reads integrations from the legacy settings bag, and all reply sites still gate on `if platform and integrations:` — `:417`, `:447`, `:478` (plus the error path `:499`) — so a workspace whose creds live only in `channel_connections` processes inbound but never fires a reply. Ironically the gate is now pure residue: `_deliver_reply` itself routes through `channels.sender`/`channel_connections` and explicitly ignores the integrations param ("accepted for backward compat but unused", `webhooks.py:165-181,198-208`). Nothing writes `settings.integrations` on new-style connect (`api/channels.py` has no such write), so the gate stays False for new-style workspaces. Zero commits to `webhooks.py` since 2026-06-25; no commit mentions F026.
- Notes: the fix is now trivial — the gate no longer protects anything since delivery reads `channel_connections`; keying it off platform-only (or a `channel_connections` row) would restore replies.

#### F027 — NOT DONE
**`ChannelManager.start_all` runs in all four uvicorn workers (outside the leader lock) — Telegram 409 polling loops.** (July: `main.py:440-448`)
- Evidence: `orchestrator/main.py:467-476` still calls `get_channel_manager().start_all()` when CHANNELS_ENABLED, positioned after the boot leader lock closes (`~main.py:203-276`) and outside the unified-scheduler flock (`main.py:458-461`) — every uvicorn worker runs it. `ChannelManager.start_all` (`orchestrator/channels/manager.py:32-72`) has no leader/worker gating: it loads all active polling rows and starts adapters. Four workers × Telegram getUpdates polling = the 409 loop. The docstring at `manager.py:36-40` only addresses skipping webhook-mode rows, not multi-worker duplication. No commits since the review; no commit mentions F027.

#### F028 — NOT DONE
**Composio V3 webhook signature failures are logged then allowed through "for debugging" on an unauthenticated endpoint — forged agentic events accepted.** (July: `composio.py:630-632`)
- Evidence: `orchestrator/api/composio.py:629-632` is byte-identical to the July cite: V3 signature mismatch logs "V3 webhook signature mismatch — allowing through for debugging" and falls through; verification exceptions likewise (`:632`). Worse, the check only runs at all when a webhook-signature or x-composio-signature header is present (`composio.py:618,633`) — omitting both headers skips verification entirely on the unauthenticated POST /webhook. Forged agentic events are still accepted. No commits to `composio.py` since 2026-06-28; no commit mentions F028.
- Notes: this remains the sharpest edge in the cluster — an unauthenticated endpoint that documents its own bypass, feeding the trigger→ingestor→agent dispatch path. See §4.C.6 (top of the channels list).

#### F029 — NOT DONE
**`channel_connections.default_agent_id` is written and advertised but no routing path ever reads it — per-channel agent pinning severed.** (July: `channels.py:38`)
- Evidence: column still written and advertised, never read for routing: model `orchestrator/core/models/channels.py:38`; write paths `api/channels.py:319,355,501-503` and tool handlers `handlers_channels.py:85,117-119`; advertised in tool schemas `actions_channels.py:64,105`. Zero reads: grep across `orchestrator/channels/` and `orchestrator/core/routing/` returns nothing; adapters never touch agent ids; inbound webhook routing uses UniversalRouter or `get_default_agent_id(db, workspace_id)` — the workspace Auto agent by slug (`api/chat.py:91-103`, used at `webhooks.py:552-561`) — and widget pinning reads `sdk_api_keys.default_agent_id` (a different table).
- Notes: the similarly-named workspace/API-key `default_agent_id` reads are easy to mistake for a fix; they are different mechanisms.

#### F032 — FIXED
**Shopify catalog webhooks never update the commerce graph — /events schedules typeless pending dicts the incremental builder ignores; the vertical's graph is manual-refresh-only.** (July: `shopify.py:491-518`)
- Fix: `503a231cb` (W13 PRD-183 S1, PR #500, merge `79124b9d3`).
- Evidence: `orchestrator/api/shopify.py:315-319` defines `CATALOG_EVENTS` (products/collections create|update|delete + inventory_levels/update; spot-checked `:315,:374`); `/events` (`:355-396`, internal-key authed) resolves the workspace by shop domain and fires `_sync_catalog_for_workspace` (`:322-352`), which runs the real graph pipeline `_product_sync_impl` (Bulk Op → `map_shopify_catalog` → `import_graph`, `:470-568`) on its own SessionLocal — replacing the old typeless pending-dict that `partition_pending_sources` dropped. Seam verified end-to-end: the sibling Remix app forwards event names in the exact matching format (`automatos-shopify` `app/automatos.server.ts` onCatalogEvent builds `products/update` etc.; `webhooks.catalog.tsx` maps PRODUCTS_UPDATE→products/update). Tests: `orchestrator/tests/test_prd183_s1_catalog_webhook.py`.
- Notes — quality: each webhook triggers a FULL catalog re-sync with no debounce/coalescing and no already-running guard (`_product_sync_impl` sets status=running at `shopify.py:517` but never checks it) — a bulk edit emitting N webhooks fires N concurrent full Bulk-Op syncs, relying on Shopify's one-bulk-op-per-shop limit with the losers swallowed as warnings (`:346-350`). Also `asyncio.create_task` at `:388` keeps no task reference (GC-collectable mid-flight). Freshness is achieved; efficiency under webhook storms is not. See §4.C.12.

#### F033 — FIXED
**First-connect auto-sync passes a request-scoped DB session into a detached background task that is torn down mid-flight.** (July: `tools.py:270-288`)
- Fix: `dbd41f5d3` (W13 S2, PR #500).
- Evidence: `orchestrator/api/tools.py:39-67` — new `_fire_shopify_autosync` opens its OWN SessionLocal inside the detached task and closes it in finally; the call site (`:306-317`, SHOPIFY pending→active) no longer passes the request-scoped `Depends(get_db)` session. Behavioral tests assert the task uses the SessionLocal-minted session, closes it, and swallows errors: `orchestrator/tests/test_prd183_s2_autosync_session.py:70-129`.
- Notes: same fire-and-forget nit as F032 (`create_task` at `tools.py:67` without a held reference); a failed autosync only logs a warning — acceptable for a best-effort first-connect kick since manual /sync and the F088 tool exist as recovery paths.

#### F052 — NOT DONE
**npm/React widget distribution unpublished; React wrapper bypasses the loader and the whole proactive engine — only the CDN script path carries the feature set.** (sibling repo `automatos-widget-sdk`)
- Evidence: no wave PRD or commit addresses it. npm registry confirms unpublished: `https://registry.npmjs.org/@automatos%2Fwidget-sdk` returns `{"error":"Not found"}` (checked 2026-07-04). Sibling repo (read-only): zero commits since the review (HEAD `bb641d6`, a pre-review loader 0.4.1 bump); `packages/react/src/react.tsx` still directly instantiates `new ChatWidget(config)` in both the component and the hook — bypassing the CDN loader and carrying no proactive/page-context/plugin-dispatch config; package.json still `@automatos/widget-sdk 0.1.0`.
- Notes: a React-embedding merchant still gets the degraded widget; only the CDN script path carries the proactive feature set.

#### F066 — NOT DONE
**No inbound email channel exists — zero code; refund-email journey has no autonomous path.** (July: `channels.py:49-53,374-386`)
- Evidence: `orchestrator/api/channels.py:48-53` — `_SUPPORTED_PLATFORMS` still has no email entry (telegram/slack/discord/teams/google_chat/signal/imessage/irc/matrix/line/whatsapp/webhook); `channels/` contains no email adapter; the Composio trigger webhook still routes ONLY `JIRA_`-prefixed triggers to an ingestor — everything else (including Gmail) dead-letters as UnroutedEvent (`orchestrator/api/composio.py:724-746`). No commit mentions F066. The refund-email journey still has no autonomous inbound path — email works only pasted into chat.

#### F076 — PARTIAL
**Vertical #2 must fork `api/shopify.py` wholesale — agent roster, widget defaults, provision/connect/sync routes, key permissions all Shopify-shaped; mappers live in generic `graph_extraction.py`.** (July: `shopify.py:115,131-147,275`; `graph_extraction.py:503,693`)
- Fix: `1642728f8` (W13 S5, PR #500).
- Evidence — landed: generic VerticalProvisioner protocol + PROVISIONER_REGISTRY + one provision flow (`orchestrator/integrations/provisioning.py:42,74,215-310`); ShopifyProvisioner declaring roster/widget-defaults/key-permissions/ops-manager/site-type/allowed-origins (`orchestrator/integrations/shopify/provision.py:30-107`); generic POST `/api/verticals/{v}/provision` (`api/verticals.py:95-126`, mounted `router_manifest.py:53`); `api/shopify.py` /provision reduced to a thin delegate (`:148-180`) with the agent-slug constants / `_seed_shopify_agents` / the `shopify-ops` special-case all gone from it (grep 0 hits); catalog+orders mappers resolved via GRAPH_SOURCE_MAPPERS registry, not hardcoded imports (`api/shopify.py:488-496,731-736`; registered in `shopify/provision.py:114-129`); acceptance test provisions a mock vertical through the generic path (`orchestrator/tests/test_prd183_s5_vertical_provision.py:125`).
- Evidence — not landed: the mapper CODE still physically lives in generic `modules/knowledge/graph_extraction.py:503,693` — only resolution moved, and the CI vertical-isolation gate explicitly excludes that file ("map_shopify_catalog, outside scope", `scripts/ci/check-no-shopify-in-generic.sh` header); the proactive trigger vocabulary remains hardcoded commerce-shaped in generic code (`PROACTIVE_TRIGGER_REASONS` with 'cart_idle' at `orchestrator/api/widgets/chat.py:72-75`) despite PRD-183 S5 naming "make proactive triggers plugin-declared"; sync stays a Shopify-specific executor (hardcoded bulk-op query constants, `api/shopify.py:409-435,527-530`) with shopify-named rather than generic "graph source" tools; connect/deactivate/sync/events remain per-vertical routes.
- Notes: the substantive fork cost — the provision lifecycle — is genuinely extracted and proven by a mock-vertical test. But two sub-items the finding and the PRD itself name are unfinished (mapper physical location, plugin-declared triggers), so vertical #2 still touches generic files for its trigger vocabulary and still writes its own data-plane sync.

#### F081 — NOT DONE
**Channels legacy scaffolding: seven driverless adapters (1,589 lines: teams/google_chat/signal/imessage/irc/matrix/line) reachable only via manual start; `_ping_platform_legacy` fully dead.** (July: `channels.py:143`; `manager.py:139-151`)
- Evidence: all seven driverless adapters intact at exactly the July line count: teams(202)+google_chat(224)+signal(226)+imessage(227)+irc(261)+matrix(215)+line(234) = 1,589 lines under `orchestrator/channels/`, still instantiable via `_ADAPTER_MAP` (`channels/manager.py:143-149`). `_ping_platform_legacy` still defined at `orchestrator/api/channels.py:143` with zero callers. No PRD-184 exists; no dead-code-kill-list commit exists.
- Notes: the roadmap parked this in the never-authored W14/PRD-184 kill list; W14 was spent on PRD-170 Code Canvas instead. Even the zero-risk slice (deleting the caller-less `_ping_platform_legacy`) wasn't taken.

#### F088 — FIXED
**Shopify sync and codegraph indexing are bare HTTP routes, not platform tools — Auto cannot trigger a sync or check graph freshness through its own surface.**
- Fix: `dc0e3946f` (W13 S3) + `47587e453` (S4), PR #500.
- Evidence: Shopify leg: two promoted ActionDefinitions `platform_shopify_sync_catalog` (write) + `platform_shopify_sync_status` (read) at `orchestrator/modules/tools/discovery/actions_shopify.py:15-54`; handlers run `_product_sync_impl` / read `workspace.settings.product_sync` with workspace_id threaded from the authenticated executor context, never params (`handlers_shopify.py:34-80`); registered into the platform registry (`platform_actions.py:36,73`) and the executor handler map (`platform_executor.py:514-515`). Codegraph leg (built as S4 for F087/F022): index/reindex tools imported at `platform_executor.py:229-230` and mapped alongside the read tools (`:517-522`). Tests: `test_prd183_s3_shopify_tools.py` (registration, promotion/scoping, executor wiring, sync-runs-and-reports-deltas, status incl. never_synced) and `test_prd183_s4_codegraph_reindex.py`.
- Notes: Auto can now refresh the catalog graph, report deltas, and check freshness through its own tool surface — genuine parity, and the tool lane is authenticated (unlike the July-era bare route). Sync executes synchronously inside the tool call (minutes-long Bulk Op) — by design so the agent can report what changed, but a slow store makes it a long-blocking tool.

#### F091 — NOT DONE
**Commerce-KG ingestion boundary unverified: `map_shopify_catalog`/`map_shopify_orders` have zero direct tests and both headline commerce journeys are skipped — a mapper regression silently corrupts every opener/cross-sell.** (July: `graph_extraction.py:503,693`; `test_golden_journeys.py:101,221`)
- Evidence: both mappers still sit at `modules/knowledge/graph_extraction.py:503,693` with zero behavioral tests: the only new "coverage" is a registry identity check asserting callable + `__name__` (`test_prd183_s5_vertical_provision.py:79-88`); `test_prd183_s1_catalog_webhook.py` monkeypatches the sync away and never feeds the mapper fixture data; repo-wide grep finds no other mapper test. Both headline journeys remain skipped: J3 widget-plugin at `orchestrator/tests/test_golden_journeys.py:101-102` and J9 Shopify-sync→FBT-opener at `:221-228`, whose own skip message still reads "the highest-value gap to close next". No PRD or commit cites F091; PRD-182 doesn't touch it.
- Notes: **the exposure has WIDENED, not shrunk** — F032's fix now runs the untested `map_shopify_catalog` automatically on every catalog webhook, so a silent mapper regression corrupts every opener/cross-sell faster and without a human in the loop. See §4.C.16.

### 2.7 Cluster: deploy-ci-observability

#### F009 — PARTIAL
**CRITICAL — fresh clone cannot boot — docker-compose mounts the initdb schema from a nonexistent path.** (July: `docker-compose.yml:35`)
- Fix: `f27cb308b` (W6 PRD-176, PR #483) — cited defect only.
- Evidence — fixed: `docker-compose.yml:35` now mounts `./orchestrator/core/database/init_complete_schema.sql` (file exists, 1,964 lines).
- Evidence — headline behavior STILL TRUE: the wave's own smoke lane (`smoke-fresh-clone.yml`, added `1cddbec0d`) failed on its latest run (2026-07-03, job 85061322908, on `ac6a2b906` whose compose/entrypoint blobs are identical to pinned main): the backend container dies at start with `exec: "docker-entrypoint.sh": executable file not found in $PATH` then "FAIL: docker compose up failed", masked by continue-on-error (job conclusion "success"). Root cause: `docker-compose.yml:188` bind-mounts the repo's `docker-entrypoint.sh` (git mode 100644 — non-executable, per `git ls-files -s`) over the image's chmod+x stub (`orchestrator/Dockerfile:103-104`, ENTRYPOINT at `:130`). That mount predates the wave (`b2e69a1f3`, 2026-05-09) — a second boot blocker the July review didn't cite and the wave didn't catch.
- Notes: one-line class of fix outstanding (make the entrypoint executable in git, or exec via bash). Until then W6's acceptance ("fresh-clone docker compose up returns 200 on /health") is unmet and the smoke lane reports green while failing inside. See §4.B.

#### F010 — PARTIAL
**CRITICAL(adj) — Alembic cannot replay from zero — 132 revisions form a four-headed forest, core tables ALTERed but CREATEd by none, no from-zero CI.**
- Fix: `fd4428644` + `a050bb647` (+ CI lane `1cddbec0d`) — single-head half only.
- Evidence — done: `prd176_merge_heads.py:30-37` (commit `fd4428644`) merges the four July heads; `e773c09189a9_merge_prd176_prd181_heads.py:15` (`a050bb647`) merges the post-wave pair. AST parse of all 137 revision files in `orchestrator/alembic/versions` confirms exactly ONE head (`e773c09189a9`); CI hard-asserts it (`test.yml:335-343`; pinned-main run log "alembic head count: 1").
- Evidence — still broken: the from-zero half — the ADJUSTED core of the finding — fails: on the pinned-main run (28684977751, job 85075865428) the "Replay all migrations from an empty database" step exited 1 with psycopg2 `relation "marketplace_installs" does not exist`, masked by continue-on-error (`test.yml:355`). The Step-2 single-baseline squash reconciling ALTERed-but-never-CREATEd tables has not landed; the lane is also not a required check (live branch protection contexts: orchestrator-tests, ioc-scan only).
- Notes: the 1-head invariant now has a real hard regression guard. The from-zero lane is honest instrumentation of a still-failing replay — flag: the failing step's job-level API conclusion reads "success", so only the log shows the truth. See §4.B.

#### F030 — NOT DONE
**Presigned deliverable URLs 404 after an hour — generated Deliverables become unreachable.**
- Evidence: unchanged since July. `modules/documents/generation_service.py:643-645` still overwrites the stable app path with the raw presigned S3 URL as the Deliverable `download_url` when upload succeeds; `:697-705` generates it with `ExpiresIn=3600` — the persisted Deliverable link dies after an hour. git log since 2026-07-01 on `generation_service.py` and `api/document_generation.py` returns zero commits. A per-request re-minting endpoint exists (`api/document_generation.py:559-596` redirects to a fresh presign) but Deliverable records don't reference it — the durable artifact is still the expiring URL, flowing into tool results via `agent_platform_tools.py` and `result_formatter.py`.
- Notes: cheapest real fix already half-exists — persist the `/api/documents`-relative path and let the re-minting endpoint own presigning.

#### F034 — PARTIAL
**Zero frontend CI plus `ignoreBuildErrors:true` — ~400 TS errors on 148k lines go unenforced.**
- Fix: `073370dc0` (W12 PRD-182 S1, PR #498).
- Evidence: "zero frontend CI" is no longer true: frontend-ci job (`test.yml:383-421`) runs on every push/PR — vitest as a hard gate (`package.json:9`), `tsc --noEmit` baselined via `frontend/scripts/tsc-baseline-check.js` against measured floor 554 (`frontend/.tsc-baseline.json:4`; exits 1 when count exceeds maxErrors), eslint report-only (`test.yml:416`). Job GREEN on pinned main (run 28684977751). Two enforcement gaps keep it partial: (1) frontend-ci is NOT a required check — live branch protection requires only orchestrator-tests + ioc-scan, so a red frontend-ci cannot block a merge; (2) `next.config.js:19` `ignoreBuildErrors:true` and `:27` `ignoreDuringBuilds:true` remain (documented posture — the deploy build still never type-checks), and the 554-error debt is frozen, not reduced.
- Notes: the baselined-not-aspirational design matches the review's own §7 prescription; the missing leg is the required-check flip, bundled into the F057 owner action.

#### F044 — FIXED
**Frontend→backend route-contract net never built — PRD-155 S2 landed zero frontend files; nothing consumes route-manifest.json.**
- Fix: `073370dc0` (W12 S3).
- Evidence: the net now exists and is consumed: `frontend/scripts/check-route-contract.js` reads `orchestrator/reports/route-manifest.json` (committed, actively refreshed — e.g. `20a204364` "refresh manifest post-main-merge"), normalises `${expr}` vs `{param}` segments, and fails on any NEW api-client call to a path absent from manifest+baseline; 35 pre-existing drifts recorded in `frontend/scripts/route-contract-baseline.json`. Wired as a HARD step in frontend-ci (`test.yml:420-421`) and green on pinned main. The single-file scope (`lib/api-client.ts`) is sound because `.eslintrc.json:4-12` bans raw `fetch('/api…')` via no-restricted-syntax, making api-client the transport choke point.
- Notes — two seams to watch: (1) the manifest is a committed artifact — `test_route_manifest.py` proves deterministic generation but no CI step diffs committed-vs-fresh, so it can drift stale; (2) the enclosing frontend-ci job is not a required check (see F057), so the gate warns rather than blocks until branch protection flips.

#### F050 — FIXED
**No backup or disaster recovery anywhere in the repo — no pg_dump tooling, no restore runbook, for main pgvector DB or mem0 instance.**
- Fix: `62d8671dc` (W6).
- Evidence: `scripts/dr/backup.sh` (pg_dump -Fc of DATABASE_URL + optional first-class MEM0_DATABASE_URL target) and `scripts/dr/restore.sh` (pg_restore into a fresh DB, CREATE EXTENSION vector first) — both executable (`-rwxr-xr-x`, spot-checked), env-driven, no hardcoded creds. `docs/runbooks/DR-postgres.md` states RPO ≤24h / RTO ≤30min for primary and mem0 in parallel (`:21-25`), restore procedure ending in the wait-migrate-seed entrypoint, pgvector/raw-DDL gotchas, and schedule guidance (§3.2). Restore is tested: `orchestrator/tests/test_dr_restore.py:138` does a full dump→restore→row-parity round trip (skips without reachable Postgres).
- Notes — honest scope: nothing in-repo actually schedules the nightly dump — runbook §3.2 assigns it to cron/Railway scheduled job (operator action, §4.D.5), and WAL/PITR is surfaced as an explicit open decision rather than silently deferred. The finding's literal asks are fully met.

#### F051 — PARTIAL
**Schema truth split across four mechanisms mutating overlapping tables — no single wait-migrate-seed lifecycle.**
- Fix: `f27cb308b` (W6).
- Evidence: the single lifecycle exists on the compose path: `docker-entrypoint.sh` is now wait_for_postgres → check_database → run_migrations (fail-closed, `alembic upgrade heads` aborts startup on failure, `:51-61`) → load_seed_data → exec app (`:125-147`), with structural tests in `orchestrator/tests/test_deployability_w6.py`. But schema truth is NOT consolidated to it: `main.py:181` still runs `create_tables()` (`Base.metadata.create_all`) on every boot plus inline `ALTER TABLE … ADD COLUMN IF NOT EXISTS` at `main.py:186-193` plus a seed gate — so initdb SQL, alembic, create_all-at-boot, and seeders all still mutate overlapping tables. And the lifecycle has never run green end-to-end: the smoke lane fails before the entrypoint executes (non-executable bind-mount, see F009), and `init_complete_schema.sql` contains no `alembic_version` stamp, so on a fresh volume the fail-closed "upgrade heads" must replay the full 137-revision forest — which the from-zero CI lane proves currently crashes (marketplace_installs, job 85075865428). The fail-closed gate would therefore likely abort a fresh-volume boot even after the exec bug is fixed.
- Notes — quality flag: fail-closed migration is the right posture, but bolted onto an unstamped initdb schema it converts the July silent-drift problem into a hard boot abort on the exact path (fresh clone) the wave was meant to open. Needs either an `alembic_version` stamp in the initdb SQL or the Step-2 baseline squash. See §4.C.7.

#### F056 — FIXED
**Two whole test trees never collected — `orchestrator/modules/*/tests` (83 fns) and `integrations/*/tests` (34 fns) outside pytest.ini testpaths.** (July: `pytest.ini:11`)
- Fix: `073370dc0` (W12 S2) + `51b2d5a80` (ci-health, PR #502).
- Evidence: `pytest.ini:22-25` testpaths widened to `tests, modules, integrations` so default collection sees both formerly-orphaned trees; dedicated orchestrator-module-tests CI job runs them against Postgres with a collect-proof step (`test.yml:431-506`). The job is GREEN on pinned main `77bc9c6d5` (run 28684977751), made green at root cause by `51b2d5a80`: `codegraph_*` tables (migration-only, no SQLAlchemy model) added to `scripts/init_test_db.py` as raw DDL mirroring the migration; pgvector/embedder/benchmark-dependent codegraph tests skipped with honest environment probes in `modules/codegraph/tests/conftest.py`; 17 `test_math_foundations` tests skipped as written against a `core.math` API that never shipped (rewrite flagged as a human authorship decision, not deleted).
- Notes — two residuals, both deliberate and documented: the job is non-required (required gate stays pinned to `pytest tests`, `test.yml:124`), and the skipped subset is collected but does not execute in this CI environment.

#### F057 — NOT DONE
**Required checks run strict=false — the stale-merge window behind the recent red-main incidents.** (live GitHub state)
- Evidence: live branch protection verified via `gh api repos/AutomatosAI/automatos-ai/branches/main/protection` on 2026-07-04: `"strict":false`, required contexts only `["orchestrator-tests","ioc-scan"]`, enforce_admins false, required_approving_review_count 0. The stale-merge window is unchanged. What the wave shipped is documentation: `docs/runbooks/W12-BRANCH-PROTECTION.md` (explicitly "surfaced here for the repo owner to run — deliberately not applied by the wave") and `PRD-182-WAVE-12-CI-TEST-BAR.md:63-66` with the exact ready-to-run command.
- Notes: graded on behavior, which the expected-fix field itself anticipated ("DOCUMENTED, NOT applied"). A 30-second repo-admin action; while it stays off, none of the new W12 lanes (frontend-ci, module-tests, alembic-from-zero) can block a merge either, and green-against-stale-base merges remain possible. §4.D.1.

#### F068 — FIXED
**Nine railway.internal defaults hardcoded in config — vendor topology baked into source, not local-safe.** (July: `config.py:407-884`)
- Fix: `68909e55d` (W6).
- Evidence: all nine railway.internal config defaults flipped to localhost and `LOG_RELAY_ENABLED` defaulted false: `config.py:442` INTERNAL_API_HOSTNAME, `:443` INTERNAL_FRONTEND_HOSTNAME, `:543` LOKI_URL, `:544` PROMETHEUS_URL, `:557-560` LOG_RELAY_URL, `:561` LOG_RELAY_ENABLED 'false', `:578` LOKI_QUERY_URL, `:809` AGENT_OPT_WORKER_URL, `:833` MEM0_API_URL, `:1000` VOICE_SERVICE_URL. Only remaining railway.internal strings in config.py are comments (`:440`, `:555`). Guarded by tests including a consolidated no-railway.internal-in-any-default assertion (`orchestrator/tests/test_config_env_centralization.py:52,194-212`). SaaS behavior preserved via env override.
- Notes — trivial residue, not behavior-changing: two dead fallbacks (`or http://loki|prometheus.railway.internal…`) in `modules/tools/discovery/handlers_monitoring.py:153,207` (unreachable while the config attrs exist and are non-empty) and an example string in `tool_registry.py:1102` docs.

#### F089 — FIXED
**No local object store (MinIO/S3_ENDPOINT_URL) — knowledge flywheel fail-softs to None on every output; generated documents live only on ephemeral disk.** (July: `knowledge_flywheel.py:216-223`)
- Fix: `f27cb308b` + `68909e55d` (W6).
- Evidence: the local object store and seam are complete and default-on for compose: `config.py:792` S3_ENDPOINT_URL; DocumentManager threads endpoint_url + path-style addressing + fast-fail timeouts into the boto client (`modules/rag/ingestion/manager.py:424-445`); docker-compose adds a minio service + healthcheck + one-shot bucket-create init + volume (`docker-compose.yml:83-124`) and wires the backend with `S3_ENDPOINT_URL` default `http://minio:9000` + creds (`:161-168`, backend depends_on minio `:140`); the knowledge flywheel persists through `DocumentManager.upload_document` (`services/knowledge_flywheel.py:187`). Prod unchanged (S3_ENDPOINT_URL unset ⇒ real AWS S3). Structural tests in `test_deployability_w6.py`.
- Notes — runtime caveat: end-to-end "flywheel persists locally" has never been demonstrated, because the compose backend currently cannot boot at all (F009 residual; MinIO itself reached Healthy in that run). Bare-metal local runs without compose still fail-soft to None (`knowledge_flywheel.py:223`) since S3_ENDPOINT_URL defaults empty — by design.

#### F090 — FIXED
**Board SSE endpoint has zero frontend subscribers; every ops surface polls on 8-60s; the SSE is a timed ping, not LISTEN/NOTIFY — "Streaming live" claim decorative.** (July: `command-center-shell.tsx:110`; `board_tasks.py:417`)
- Fix: `88628913f` + `91b506504` + `61f3a01ab` (W10 PRD-180 S1, PR #497).
- Evidence: backend is real LISTEN/NOTIFY, not a timed ping: every board-task mutation fires pg_notify on the board_events channel (`services/board_events.py:62`; spot-checked; call sites `api/board_tasks.py:355,576,878`), a dedicated worker thread LISTENs (`board_events.py:96-98`) and bridges into the async SSE generator; per-workspace isolation enforced in pure `frame_for_payload` (`board_events.py:183-195` — other tenants' events dropped); heartbeat comments are liveness-only. SSE endpoint at `board_tasks.py:432-451`. The LISTEN/NOTIFY pool-autocommit poison is fixed by resetting autocommit before return-to-pool (`board_events.py:115-123`, commit `61f3a01ab`). Frontend now actually subscribes: `hooks/use-board-event-stream.ts` (fetch + ReadableStream because EventSource can't carry the auth header; reconnect with capped backoff; react-query invalidation on board_changed), mounted at `components/command-center/command-center-shell.tsx:74-75`, and the old 60s poll is removed with a regression test asserting `use-board-tasks.ts` contains no refetchInterval (`components/command-center/__tests__/board-event-stream.test.tsx:153-162`).
- Notes: "Streaming live" is now honest for the Command Center board — the finding's cited surface. Platform-wide push is not finished: the Activity board-task-viewer still polls at 5s (`components/activity/board/board-task-viewer.tsx:33`) and heartbeats at 30/60s (`hooks/use-heartbeats-api.ts:86,166`).

#### F092 — PARTIAL
**CRITICAL(adj) — no coverage tooling against the 80% doctrine, no SAST/dependabot/gitleaks, no migration-replay lane (CI initialises schema via create_all, never Alembic) — why four heads shipped undetected.** (July: `test.yml:99-100`; `requirements.txt:54-56`)
- Fix: `35a0d707e` (W3) + `073370dc0` (W12 S4) + `1cddbec0d` (CI lane); note `084dabb7c` later trimmed Dependabot to security-only.
- Evidence — three legs, mixed. (a) Supply-chain: FIXED — CodeQL python+js on PR/push/weekly cron (`codeql.yml:24-26,47`) and gitleaks full-history (`gitleaks.yml`, fetch-depth:0) landed in W3 and are green on pinned main; malware-scan's ioc-scan is even a required branch-protection context; the Dependabot version-update lane was deliberately removed (`084dabb7c`) but the security posture is live-verified: `gh api` shows vulnerability-alerts enabled and automated-security-fixes `{"enabled":true,"paused":false}` (2026-07-04). (b) Coverage: MEASURED, NOT ENFORCED — `pytest-cov==4.1.0` (`orchestrator/requirements.txt:63`), --cov + .coveragerc wired into the REQUIRED orchestrator-tests job (`test.yml:129-133,142-143`), but `orchestrator/.coverage-baseline` still holds the SEED token, so `scripts/check_coverage_baseline.py` prints the number and exits 0 every run — the ratchet is unarmed until a human commits the measured floor. (c) Migration replay: LANE EXISTS, GATE DOESN'T BITE — alembic-from-zero job (`test.yml:276-359`) hard-asserts the single head, but the actual from-zero replay runs continue-on-error and is currently FAILING (marketplace_installs, job 85075865428), and the required job still initializes schema via create_all in `scripts/init_test_db.py` (`test.yml:109-110`) — Alembic still never gates a merge.
- Notes: the four-heads-shipped-undetected failure mode is now guarded (hard single-head assert). The two soft legs share one arming step each: commit the measured coverage number over SEED, and drop continue-on-error once the Step-2 baseline squash makes from-zero replay green — both explicitly documented in-tree as follow-ups. §4.D.2/§4.D.7.

### 2.8 Cluster: frontend-truth-api-honesty

#### F031 — NOT DONE
**`generate_document` ToolSpec omits `template_id` though the handler parses it and the schema-discovery tool hands the agent an id — id-driven flow relies on an undeclared parameter.** (July: `tool_registry.py:1188-1219`)
- Evidence: `orchestrator/modules/tools/registry/tool_registry.py:1174-1218` — the generate_document ToolSpec parameters block (`:1185-1218`) still declares only title/format/data/template_name; no template_id. The handler still parses it: `modules/agents/services/agent_platform_tools.py:726` (`parameters.get("template_id")`) and resolves it to a UUID at `:767-787`. The schema-discovery tool still hands the agent an id: `actions_documents.py:246-263` (`platform_get_template_schema`, required ["template_id"], description: "Use this after platform_list_templates and before generate_document"). `git log -S "template_id"` on the registry file returns nothing — the ToolSpec was never touched. This lane is LLM-facing: `tool_router.py:282` converts registry ToolSpecs via `tool.to_openai_format()`.
- Notes — nuance verified honestly: the chatbot lane's separate inline schema at `agent_platform_tools.py:209-238` DOES declare template_id (`:227`, added by PRD-167 pre-review), so id-driven generation works on that lane. The gap is specifically the registry/tool_router lane, exactly as filed.

#### F035 — FIXED
**Chat hardcodes `initialChatModel='gpt-4'` and silently sends a placebo model selector overriding the real default.**
- Fix: `39d42eb99` (W10 PRD-180 S3, PR #497).
- Evidence: zero live-code hits for initialChatModel or selectedChatModel in frontend/orchestrator (spot-checked: the only file is the guard test `frontend/components/chatbot/__tests__/model-selector-removed.test.tsx:38-59`, which asserts no chat surface hardcodes initialChatModel and hooks.ts sends no override). `frontend/components/chatbot/model-selector.tsx` is deleted. `frontend/lib/chat/hooks.ts:115-116` confirms no client model override is sent — model resolves server-side via the Auto tier. Backend: zero hits for selectedChatModel or `_parse_model_selection`. The commit touched all 3 chat callers including `frontend/app/chat/[id]/page.tsx`.
- Notes: behavior genuinely corrected — the placebo was deleted, not hidden. Cosmetic residue only: `frontend/types/chat.ts:251` still declares a dead optional `selectedChatModel?: string` nothing writes, and `frontend/lib/ai/models.ts` still ships a gpt-4 model catalog with zero importers (orphaned dead code, no behavior).

#### F036 — NOT DONE
**`/chat/[id]` SSR route survived its intended deletion and 401s every request (token getter is client-only) — dead-but-routed zombie.**
- Evidence: `frontend/app/chat/[id]/page.tsx` still exists as an async server component (line 7, no 'use client') fetching via getChat/getChatMessages → `apiClient.request` (`frontend/lib/chat/api.ts:28-29,56-57`). apiClient auth is client-only: `setClerkTokenGetter` is invoked only from client providers (`clerk-api-client-provider.tsx:23`, `local-auth-provider.tsx:24`); server-side the getter stays null (`frontend/lib/api-client.ts:92`), so the SSR fetch carries no Authorization → backend rejects → catch → `notFound()` (`page.tsx:34`). Every request renders 404. It is not merely routed but **actively linked**: `components/activity/activity-feed.tsx:236`, `activity/execution-detail.tsx:333`, and `activity/widgets/recent-activity-widget.tsx:71` all `router.push(\`/chat/${item.source_id}\`)`. PRD-184 never authored; PRD-180's finding table covers only F035/F037/F038.
- Notes — sharper than "still open": W10's commit `39d42eb99` edited this exact file (removed the initialChatModel prop) and left the zombie routed. Runtime caveat: under AUTH_EDITION=local the backend might accept the unauthenticated SSR fetch; on the SaaS default the 401→404 mechanism stands from static evidence. See §4.C.18.

#### F037 — FIXED
**`composio_execute` calls are name-filtered out of the running/error indicators — every external-app action invisible while it runs, no error chip on failure.** (July: `message.tsx:278-280`)
- Fix: `61d87e0fe` (W10 S4, PR #497).
- Evidence: `frontend/components/chatbot/message.tsx:289-293` — renderToolCalls now filters by state only (`tc.state === 'running'` / `'error'`); the composio_execute name-filter is gone. composio_execute renders a running chip and per-tool error chips (`:297-319`), and formatToolLabel (`:264-282`) enriches the label with the resolved external action from input.action/tool_slug/app/action_name — e.g. "Composio · GMAIL_SEND_EMAIL", falling back to plain "Composio". Guard test: `frontend/components/chatbot/__tests__/composio-indicator.test.tsx` (real render of running chip, error chip, fallback).
- Notes: goes beyond un-filtering — external-app actions are now identified by their real action name while running and on failure. No shim, no swallowed error.

#### F038 — FIXED
**Studio sidebar renders fabricated metrics by default — tick 5s, $/dec $0.0027, cache 68%, v0.11.** (July: `studio-sidebar.tsx:45`)
- Fix: `bfec060b6` (W10 S2, PR #497).
- Evidence: `frontend/components/layout/studio-sidebar.tsx` — grep for mini-stats/0.0027/68%/v0.11/showStats finds no live code, only removal markers: `:70-72` (fabricated v0.11 literal removed, "no truthful version source exists") and `:180-182` (fabricated mini-stats block removed). Orphaned CSS removed with marker at `frontend/app/globals.css:1134`. The dead showStats prop is gone. Guard test `frontend/components/layout/__tests__/studio-sidebar.test.tsx` asserts none of the fabricated literals render.
- Notes — same-class residue not covered by the July finding: the default `workspaceMeta = 'pilot · 11 op'` (`studio-sidebar.tsx:44`) is itself a fabricated descriptor and still renders in the workspace pill (`:96`) because `main-layout.tsx:161` renders `<StudioSidebar />` with no props. The four named F038 literals are genuinely gone; this leftover is a smaller lie of the same species. See §4.C.17.

#### F046 — NOT DONE
**Two endpoints call a nonexistent `RAGService.test_rag_config` and 500 on every hit, atop a placebo RAG-config CRUD.**
- Evidence: both endpoints still call the nonexistent method: `orchestrator/api/context.py:182` and `orchestrator/api/system.py:330` call `rag_service.test_rag_config(config_id, query, db)` (spot-checked both sites; `rg 'def test_rag_config'` across the orchestrator matches only the two endpoint functions themselves — `RAGService` at `modules/rag/service.py:152` has no such method). The AttributeError falls into the generic `except Exception` → HTTP 500 on every hit. Both routers are mounted: `main.py:958` (system) and `:970` (context).
- Notes — aggravating detail: W2 commit `19ea48825` edited the very signature of `test_rag_configuration` in `api/context.py` (added the hybrid-context dependency) without repairing the broken call one line below — the endpoint was touched during the waves and left 500ing.

#### F079 — NOT DONE
**Rival context stacks: ContextAssembler char-budget assembly, SearchService/ContextRetrievalEngine (673 lines, zero external callers), ContextOptimizer (950 lines, constructed but zero method calls) — three answers to "context budget".**
- Evidence: all three still present. (1) ContextAssembler: `modules/search/services/context_assembler.py`, exposed via `api/context_policy.py`, mounted at `main.py:52/965`. (2) ContextRetrievalEngine: `modules/search/retrieval/context_retrieval_engine.py` still exactly 673 lines; zero references outside modules/search. (3) ContextOptimizer: `modules/search/optimization/context_optimizer.py` still exactly 950 lines; constructed at `modules/rag/service.py:185` and used only as a truthiness gate at `:342` — `rg 'self\._context_optimizer\.'` finds ZERO method calls; the misleadingly named `_optimize_with_context_optimizer` (`service.py:699`) is inline PRD-157 budgeter code that never touches the optimizer instance. PRD-184 never authored; no consolidation commit.
- Notes: the 950-line ContextOptimizer now functions purely as a boolean flag deciding which of two OTHER code paths runs — arguably worse than dead, since it looks load-bearing.

#### F080 — NOT DONE
**Dead NL2SQL intelligence/ package — 1,641 LOC decoy (SmartNL2SQLAgent/QueryClarifier/ResultExplainer) plus legacy PRD-21 service methods calling undefined `_build_connection_string`.** (July: `nl2sql/__init__.py:61-68`; `service.py:680-883`)
- Evidence: `orchestrator/modules/nl2sql/intelligence/` alive: agent.py/clarifier.py/explainer.py/rephraser.py/visualizer.py = 1,687 LOC total (slightly LARGER than July's 1,641). Still exported from `modules/nl2sql/__init__.py:59-68` — zero callers outside modules/nl2sql, so it remains a decoy surface. Legacy leg unchanged: `modules/nl2sql/service.py:762` still calls `self._build_connection_string(credentials, dialect)` and `def _build_connection_string` does not exist anywhere in the orchestrator — the legacy PRD-21 path crashes with AttributeError if ever reached. No kill commit.
- Notes: the decoy grew by ~46 lines rather than shrinking.

#### F083 — PARTIAL
**API mount-honesty: `api/workspace_exec.py` real router never imported (live duplicate at `workspace_files.py:160`); `api/anthropic_client.py` misfiled provider class importing nonexistent `api/base`; `analytics.py` mounted at bare /analytics (sites.py leg REFUTED — leave alone).**
- Fix: `ad2ac1100` (W14 P170-S7, PR #501) — workspace_exec leg only.
- Evidence: Leg 1 FIXED: `orchestrator/api/workspace_exec.py` is deleted (file absent); commit `ad2ac1100` "one exec surface; delete the dead duplicate router (Q85)" removed it with a reachability guard test (`tests/test_prd170_exec_surface.py`) proving POST `/api/workspaces/{id}/exec` is served exactly once by `workspace_files.py` (live route at `:175`). Clean delete, not a shim. Leg 2 NOT fixed: `orchestrator/api/anthropic_client.py` still exists with `from .base import BaseLLMProvider, LLMConfig, LLMResponse` at line 12 while `orchestrator/api/base.py` does not exist — a misfiled dead provider (the real client is `core/llm/clients/anthropic_client.py`); zero importers, so the broken import is latent rather than crashing. Leg 3 NOT fixed: `orchestrator/api/analytics.py:27-31` still `APIRouter(prefix="/analytics")` — bare mount without /api — included at `main.py:962`.
- Notes: the analytics router is now fail-closed super-admin-only via the PRD-143 S6 router dependency (`analytics.py:26-31`), which limits exposure but leaves the anomalous bare mount exactly as filed. Per the July finding, the sites.py leg was refuted and correctly left alone.

#### F084 — NOT DONE
**Dead-but-routed frontend relics: /api-control page, empty Prisma vertical (0 models, 0 importers), usePageAPI no-op across 17 pages, 442-line /styleguide in prod, three coexisting lockfiles (nondeterministic builds).**
- Evidence: every leg reproduces. (1) /api-control: `frontend/app/api-control/page.jsx` alive and routed; it drives useAllApiToggles from `frontend/hooks/use-api-toggle.js` — toggle controls for the mock system PRD-168 deleted, i.e. a placebo control panel. (2) Prisma: `frontend/prisma/schema.prisma` is 10 lines with zero model blocks, yet prisma 5.3.1 and @prisma/client 5.3.1 remain in `frontend/package.json` (lines 38, 56) plus a prisma config block (line 15). (3) usePageAPI: still imported by 17 app pages + chat-page-content (19 files total); the hook forwards to `apiClient.setCurrentPage` which is an explicit no-op — `frontend/lib/api-client.ts:131-133` ("PRD-168 S3: mock control removed. setCurrentPage kept as a no-op"). (4) `frontend/app/styleguide/page.tsx`: still 442 lines, still routed in prod. (5) Lockfiles: package-lock.json, yarn.lock, and pnpm-lock.yaml all coexist — nondeterministic-build risk unchanged. PRD-184 never authored; no commit cites F084.
- Notes: the usePageAPI no-op was a deliberate PRD-168 keep, but 17 pages calling a function that does nothing — and an /api-control page toggling a deleted mock system — are exactly the frontend-truth debt the finding named. Nothing moved.

---

## 3. Per-capability rollup

Findings may appear under more than one capability; each rollup answers: what the waves changed, what remains open, and what is flag-gated off.

### 3.1 Memory (field + durable)

**Changed:** W8/W9 did real, unconditional work. Field tools now bind to the calling task's own run instead of an arbitrary running Mission (F020, `.first()` block deleted); heartbeat and planning modes get a workspace-field digest so recurring agents are no longer amnesiac (F021); the trace inspector is genuinely read-only (`record_access=False` threaded through the port, F062); compaction is workspace-scoped with a persisted resume cursor (F063); and a default-ON field-to-durable promotion job with a taint guard on untrusted provenance landed as a sibling (`jobs/promote_field_memory.py`, `config.py:856-874`). This is a direct autonomy win: Missions stop cross-contaminating each other's field context, and heartbeat agents accumulate workspace patterns between ticks.
**Open:** `/api/v1/memory` is half-scoped — writes are workspace-prefixed, reads/consolidate are not (F039) — which simultaneously breaks intra-tenant recall through that endpoint and leaves a cross-tenant read oracle; it should be finished or deleted. The identical arbitrary-running-Mission lookup survives for `_agent_id` on graph/document tools (`platform_executor.py:864-888`). The mem0 fork's auth/metadata state is unverifiable from this tree (F011 — human runbook; the in-repo probe checks liveness, not auth).
**Flags:** `FIELD_PROMOTION_ENABLED` default true (ON). Nothing memory-side is dark.

### 3.2 RAG / retrieval

**Changed:** the feedback loop is closed on the live path: negative rag_feedback (thumbs-down / low ratings) now penalizes ranking in `RAGService.retrieve`, workspace-scoped, defaults ON (F070). S3-vectors isolation was fixed at both the search and parent-expansion layers (F005). `api/context.py` observability endpoints are authenticated and workspace-scoped (F045).
**Open:** the RAG-config CRUD remains a placebo whose test endpoints 500 on every hit — touched by W2 and left broken (F046). Three rival context stacks still coexist (F079); the 950-line ContextOptimizer is constructed and used only as a truthiness gate. Positive feedback and corrections still influence nothing, and no eval consumes feedback (F070 scope note). For output quality, the negative-feedback penalty is the first real usage→retrieval loop; the decoy stacks mainly cost maintainer comprehension.
**Flags:** `S3_VECTORS_ENABLED` default false (the F005 fix lives in the gated path). `RAG_FEEDBACK_PENALTY_FACTOR` 0.5 default ON.

### 3.3 Graphs (Knowledge Graph, commerce graph, codegraph)

**Changed:** codegraph gained agent-side write parity — index/reindex/set-auto-reindex platform tools, a PATCH route, and a workspace-guarded setter, making the GitHub-push auto-reindex path actually fireable (F087, F022). The commerce graph now re-syncs on catalog webhooks end-to-end from the sibling Shopify app (F032), and sync/freshness are platform tools (F088) — Auto can keep its own graph fresh and report deltas, a straight autonomy win.
**Open:** the codegraph fallback INSERT still violates NOT NULL workspace_id and binds `str(dict)` into JSONB — an embedding outage loses whole symbol batches instead of degrading (F064, two-line fix). The commerce mappers have zero behavioral tests while F032 now runs them automatically on every webhook — the F091 exposure widened. The codegraph webhook accepts unsigned payloads when `GITHUB_WEBHOOK_SECRET` is unset. Catalog re-sync has no debounce/coalescing under webhook storms.
**Flags:** none dark.

### 3.4 Tool selection / operating graph

**Changed:** W7 fixed the data side of the learning loop: Composio telemetry resolves per-action names so the 856-app surface no longer collapses to one node (F016); chat threads user_query/conversation/turn ids so intent-conditioned and used_after edges materialize in exact turn order (F017); the metadata sync scheduler exists (daily cron, default ON) and the destructive-action gate fails CLOSED on an empty table (F018); edge reads are workspace-scoped (F015 S5).
**Open:** the read side mostly doesn't consume the new data yet: chain hints stay dark behind `TOOL_ROUTING_GRAPH` (held honestly by the S6 eval's −32.9 uplift), and the dispatcher enum enrichment is still alphabetical top-50 (F074) — the cheapest next place learned edges could reach the live surface. Verb→capability/app maps (F071) and the confirmation taxonomy (F072) remain source-baked policy. `Skill.priority` is a phantom (F054) and `tools_schema` still holds three incompatible formats (F055) — both silent quality levers on multi-skill agents. `modules/learning`/`modules/evaluation` remain decoy packages (F082).
**Flags:** `TOOL_ROUTING_GRAPH` default false (evidence-held); `COMPOSIO_DESTRUCTIVE_FAIL_CLOSED` and `COMPOSIO_SYNC_ENABLED` default true.

### 3.5 Execution spine

**Changed:** W1 closed all four defects genuinely: the ctor-kwarg crash that killed every non-chat execution is fixed with a through-the-factory test plus a seam guard (F001); failed executions now mark failed with the error surfaced through the same failure-dispatch path as crashes (F023); a lease heartbeat renews for the life of every run so long runs are no longer swept and double-executed (F024); and Mission-mirror rows are excluded from both PATCH launch paths (F025). Headless autonomy — board, Missions, scheduled, webhooks, inter-agent — actually executes again; this was the single highest-leverage fix of the program.
**Open:** a narrower double-execution path survives via the dispatcher claim loop and Run-Now on mirror rows (recipe-only exclusions). The dead-on-arrival `api_playbooks` router is still mounted (F069), and the legacy workflow engine remains a mounted fifth engine with one live execute endpoint and a Composio-webhook fallback leg (F078 — its cross-tenant leg was closed by W2's endpoint deletions).

### 3.6 Missions / board

**Changed:** board tasks got a real, unconditional approval gate running the same `evaluate_approval` Missions use, with durable/expiring/revocable ApprovalGrants and a grant-resolution surface (F060 board half, W11). The Mission-synthesis flywheel no longer starves — SQL-side terminal markers, DESC ordering, failure markers (F049). HARNESS prescriptions have a real actuation path through approve (F048) — but see flags. Board "Streaming live" is now honest LISTEN/NOTIFY with the Command Center subscribed and the poll deleted (F090).
**Open:** the board dollar ceiling is vacuous (estimated cost hardwired 0.0) and the gate fails open on error; Playbook runs have an opt-in flat-rate ceiling but no ask-gate; scheduled/webhook agents are explicitly ungoverned future work; `/api/tasks` direct-step remains mounted with zero governance. Mission auto-approve ceilings still price with the model-blind flat rate (F059).
**Flags:** `HARNESS_SELF_MANAGEMENT_ENABLED` default false — the approve surface still 409s out of the box; and the actuation path's policy check is ceremonial (`override_auto_approve=True` makes the decline branch unreachable).

### 3.7 Channels

**Changed:** nothing. Zero commits to the channels surface since the review.
**Open:** the whole July list reproduces verbatim: replies drop for new-style `channel_connections` workspaces because of a now-pure-residue legacy gate (F026 — trivial fix since delivery already reads channel_connections); `start_all` runs in all four workers so Telegram polling 409-loops (F027); the Composio webhook documents its own signature bypass and skips verification entirely when headers are absent — forged agentic events accepted on an unauthenticated endpoint (F028, the sharpest edge here); per-channel agent pinning is severed (`default_agent_id` written, never read — F029); there is no inbound email channel at all, so the refund-email journey has no autonomous path (F066); and 1,589 lines of driverless adapters plus a dead ping function await the never-authored kill list (F081). For a platform whose North Star is autonomous client-facing operation, channels is the largest capability gap left standing.

### 3.8 Widget / Shopify

**Changed:** the biggest cross-repo win of the program: the sibling Remix app now builds with a CI gate, real webhook routes, HMAC-verified GDPR handlers calling a new platform machine-to-machine GDPR surface, and wired OAuth provisioning (F013). Catalog webhooks re-sync the commerce graph (F032); first-connect auto-sync owns its DB session (F033); sync/status are platform tools (F088); the Admin token is encrypted at rest (F058); sync routes are workspace-authenticated (F003); the internal key is fail-closed (F004). Provisioning is behind a generic VerticalProvisioner registry with a mock-vertical acceptance test (F076 landed half).
**Open:** mappers physically remain in generic `graph_extraction.py` and proactive trigger vocabulary is still commerce-shaped in generic code (F076 residue); mapper behavioral tests still zero while auto-sync now exercises them (F091 — widened); the npm/React widget lane is still unpublished and bypasses the proactive engine (F052); a leaked `ak_pub_` key still works from curl when Origin is absent (F053); GDPR data_request export is workspace-scoped rather than per-customer and is delivered nowhere.

### 3.9 Governance / policy

**Changed:** W4 built a real policy plane — PolicyGate chokepoint in the unified executor, budget admission with a real ceiling model, a registry-backed model-aware pricing source, one `caller_is_admin` helper adopted by all 8 forked routers, one empty-permission semantic, SlowAPI middleware registration — and W11 added the unconditional board approval gate and durable grants.
**Open — the honest headline:** every behavioral change in W4 is behind `AUTOMATOS_POLICY_PLANE`, default OFF, and nothing in the deploy surfaces sets it — so default deployments keep July behavior byte-for-byte on the inert rate limiter (F040), the workspace-membership admin auto-flip (F014), the widget god-key semantic (F042), and the super_admin 403 fork (F043). Even ON, the gate fails open on internal error, no caller threads token estimates (projected cost always 0), and the chat per-action Composio shortcut, Playbook steps, widget email, and `/api/tasks` bypass the chokepoint. Pricing paths went up, not down (F059). Verb maps and the confirmation taxonomy remain deploy-coupled (F071/F072). Heartbeat user endpoints are still super_admin-locked (F041). Judged by the North Star, the plane is scaffolding for autonomy-with-guardrails that is not yet load-bearing: the one gate agents actually hit today is the board approval gate.

### 3.10 Auth / tenancy

**Changed:** the strongest cluster — 10 of 15 fixed. W2 closed every cited cross-tenant surface: skills attach/read/delete (F002), Shopify sync (F003), internal-key fail-open (F004), S3-vectors filters (F005), the legacy execute oracle (deleted, F006), the unauthenticated alerts read plane (F007), and `api/context.py` (F045); the NL2SQL function denylist landed (F019 half). W3 encrypted the Shopify Admin token (F058) and added gitleaks/CodeQL. W5 delivered the `AUTH_EDITION local|saas` seam — the open-core blocker is gone: local boots loginless with no Clerk tenant (F008), and the staff domain moved to config (F075).
**Open:** `/api/v1/memory` read-path scoping (F039); the widget Origin-absent bypass (F053); NL2SQL still runs on write-capable credentials with no read-only role backstop (F019 half); the Clerk artifact is still tracked in git pending a human history purge (F012). Cross-cutting hazard: security boot guards are swallowed by `run_stage` — `validate_security()` raises but boot proceeds — so all "aborts at boot" claims are actually runtime-guard-only.

### 3.11 Observability

**Changed:** the alerts/logs read plane is super_admin-gated fail-closed (F007); board push is real LISTEN/NOTIFY with per-workspace frame isolation and the Command Center subscribed (F090); tool telemetry is per-action (F016); the fabricated Studio sidebar stats are deleted with a guard test (F038); CodeQL/gitleaks lanes run on every push (F092 leg a).
**Open:** the Activity board-task-viewer still polls at 5s and heartbeat surfaces at 30/60s (platform-wide push unfinished); the fabricated `pilot · 11 op` workspace pill still renders; presigned Deliverable URLs still die after an hour, which quietly breaks the client-facing artifact trail (F030); HARNESS's own health surface is dark by default (F048). Two CI lanes report green while failing inside (§4.B) — an observability problem about the platform's own delivery pipeline.

### 3.12 Deployability

**Changed:** W6 moved real ground: correct initdb mount (F009 half), a fail-closed wait-migrate-seed entrypoint (F051 half), pg_dump/restore tooling + tested DR runbook covering both DBs (F050), all nine railway.internal defaults flipped to localhost (F068), MinIO + `S3_ENDPOINT_URL` seam so the knowledge flywheel can persist locally (F089), and a single alembic head with a hard CI assert (F010 half). W5's local edition removes the Clerk dependency for OSS boots (F008).
**Open — the honest headline:** a fresh clone still cannot boot. The non-executable `docker-entrypoint.sh` bind-mount kills the backend container before the new entrypoint ever runs (second blocker, pre-wave, uncited in July), the from-zero migration replay still crashes (`marketplace_installs`), and both failures are masked by continue-on-error lanes that report green. Worse, the fail-closed migration gate on an unstamped initdb schema converts July's silent drift into a hard boot abort on exactly the fresh path once the exec bug is fixed — it needs the alembic stamp or the Step-2 squash. Schema truth still has four writers (initdb SQL, alembic, create_all-at-boot + inline ALTERs, seeders). Branch protection remains strict=false with only two required contexts, so none of the new lanes can block a merge (F057).

### 3.13 Frontend truth

**Changed:** W10 deleted the placebo model selector (server-side model resolution is now what the UI implies, F035), surfaced composio_execute running/error chips with real action names (F037), and deleted the fabricated sidebar stats (F038) — all with guard tests. W12 gave the frontend its first CI (vitest hard gate, tsc baselined at 554, F034) and the route-contract net catching new frontend→backend drift (F044). W14's Code Canvas work deleted the duplicate exec router (F083 leg 1).
**Open:** the `/chat/[id]` SSR zombie is still routed and actively linked from three activity surfaces — W10 edited that exact file and left it (F036); the RAG-config placebo still 500s (F046); the registry-lane `generate_document` schema still omits `template_id` (F031); the NL2SQL decoy grew (F080); and the whole F084 relic set reproduces (api-control placebo panel, empty Prisma, usePageAPI no-op on 17 pages, /styleguide in prod, three lockfiles). `ignoreBuildErrors:true` remains the deploy posture. For client-facing quality, F036 and F030 are the two that actively mislead users rather than just costing hygiene.

---

## 4. Watch list

### 4.A Fixes that exist but are OFF by default

Dossier teams: when you read "fixed in W4/W7/W9", check this table first. Grading in §2 already accounts for it.

| Flag | Default | What stays dark while OFF | Findings |
|---|---|---|---|
| `AUTOMATOS_POLICY_PLANE` (`config.py:645`; fail-safe-to-OFF `modules/policy/flag.py:20-32`) | **false** — nothing in envs/, docker-compose.yml, railway.json, Dockerfile sets it | The entire W4 plane: SlowAPI rate limiter registration (`main.py:837-838`), caller-identity admin gate (`platform_executor.py:675-709`), widget empty-permissions=deny (`widgets/auth.py:236-264`), super_admin ⊇ admin hierarchy (`core/auth/roles.py:37-45`), pre-call budget admission (`gate.py:162-190`), per-workspace risk-class routing (`policy_document.py:77-160`), Art.12 audit handler (`main.py:519`) | F040, F014, F042, F043, F085, F086, F072 (tunable layer) |
| `TOOL_ROUTING_GRAPH` (`config.py:757`) | **false** | Learned-edge chain hints in the prompt catalog. Held deliberately: S6 offline eval measured −32.9 mean uplift vs a +5 gate — dark on evidence, not neglect | F015 |
| `HARNESS_SELF_MANAGEMENT_ENABLED` (`config.py:627`) | **false** | The whole HARNESS actuation surface; `/approve` returns 409 out of the box, reproducing the July symptom | F048 |
| `S3_VECTORS_ENABLED` | **false** | The S3-vectors backend whose isolation W2 fixed (fix lives in the enabled path) | F005 |
| `NEXT_PUBLIC_AUTH_EDITION` / `AUTH_EDITION` (`config.py:157-166`) | **saas** | The loginless local edition (opt-in by design — correct posture, listed for completeness) | F008 |

Default-ON safety flags worth knowing (new since July): `COMPOSIO_DESTRUCTIVE_FAIL_CLOSED=true` (`config.py:768`), `COMPOSIO_SYNC_ENABLED=true` (`:769`), `FIELD_PROMOTION_ENABLED=true` (`config.py:856-874`), RAG negative-feedback penalty factor 0.5 (`config.py:974-978`), and the board approval gate + lease heartbeat + board SSE, which are unconditional.

### 4.B CI lanes that report green while failing inside

1. **smoke-fresh-clone** (`smoke-fresh-clone.yml`): backend container dies at start (`exec: "docker-entrypoint.sh": executable file not found`), then "FAIL: docker compose up failed" — masked by continue-on-error; latest run 2026-07-03 job 85061322908 concluded "success". Root cause: `docker-compose.yml:188` bind-mounts a git-mode-100644 entrypoint over the image's chmod+x copy (`orchestrator/Dockerfile:103-104`).
2. **alembic-from-zero replay step** (`test.yml:276-359`): from-zero replay exits 1 (`relation "marketplace_installs" does not exist`, pinned-main job 85075865428) under continue-on-error (`test.yml:355`). The single-head assert in the same job IS hard and green.
3. **Coverage ratchet**: `orchestrator/.coverage-baseline` still holds the SEED token — `scripts/check_coverage_baseline.py` prints and exits 0 every run; measured, never enforced.
4. **gitleaks**: currently red by design pending the human purge of `tests/e2e/.auth/user.json` (the workflow's own comment says so).
5. **Required checks**: live branch protection (read 2026-07-04) requires only `orchestrator-tests` + `ioc-scan`, `strict=false` — so frontend-ci, route-contract, module-tests, and alembic-from-zero cannot block any merge (F057).

### 4.C New or adjacent defects surfaced during verification (not in the July 94)

Candidate leads for Phase-2 dossiers; each verified with file:line in the pinned tree.

1. **Board mirror double-execution, residual paths** — claim loop filters only `source_type <> 'recipe'` (`board_dispatcher.py:92,117,146`); `run_task_now` (`board_tasks.py:828`) and the inbox→assigned auto-transition (`:543-544,:592-599`) can put a Mission-mirror row where the loop claims it. (from F025)
2. **Cross-Mission agent-identity bleed on graph/document tools** — the same `.first()`-on-state=='running' lookup F020 killed survives for `_agent_id` (`platform_executor.py:864-888`). (from F020)
3. **Ungated platform-wide delete** — `cleanup_old_skill_mappings` (`api/skills.py:730`) is "/admin/"-named, takes ctx, never checks it; deletes `agent_skills` rows across all workspaces. (from F002)
4. **Unscoped agent fallback in execute-advanced** — `workflows.py:973` picks any active agent from any workspace; `:978` fetches explicit agent_id with no workspace filter. (from F006)
5. **Boot guards swallowed** — `run_stage` (`core/models/bootstrap.py:115-137`) catches `validate_security()`'s RuntimeError; lifespan never checks the stage result (`main.py:178,507,301-330`). Every "aborts at boot" security claim is actually runtime-guard-only. (from F004; affects F005/F008 claims)
6. **Composio webhook skips verification when headers absent** — `composio.py:618,633`: no webhook-signature/x-composio-signature header ⇒ no check at all, on top of the documented mismatch bypass at `:629-632`. (from F028)
7. **Fail-closed migrations + unstamped initdb = fresh-volume boot abort** — `init_complete_schema.sql` carries no `alembic_version` stamp; the entrypoint's fail-closed `alembic upgrade heads` must replay the whole forest, which from-zero CI proves crashes. (from F051)
8. **Board dollar ceiling vacuous + gate fails open** — `board_tasks.py:968` never passes `estimated_cost_usd` (defaults 0.0 at `board_approval.py:68`); gate error path proceeds (`board_tasks.py:989-994`). (from F060)
9. **HARNESS "fail-safe" claim false** — `override_auto_approve=True` (`harness_commands.py:238-240`) short-circuits `approval_policy.py:163-164` to unconditional approve; the decline branch `:241-250` is unreachable. (from F048)
10. **PolicyGate fails open on internal error** — `unified_executor.py:275-280`: any exception in the plane is logged and treated as proceed, even when the flag is ON. (from F085)
11. **Composio lanes outside the policy chokepoint** — chat per-action shortcut (`consumers/chatbot/service.py:1321-1334,1550-1565`), Playbook steps (`recipe_executor.py:655`), widget email (`widget_email.py:286,340,388,437`), `/api/tasks` direct-step (`api/tasks.py:62-124`). (from F085/F060)
12. **Catalog webhook storm amplification** — full Bulk-Op re-sync per event, no debounce or already-running check (`shopify.py:517,:346-350`); unreferenced `asyncio.create_task` (`:388`; also `tools.py:67`). (from F032/F033)
13. **Destructive-gate cold-start window** — unclassified actions are allowed unless the INTENT TEXT matches an 8-keyword list; neutral-worded destructive actions ("issue a refund") pass until the daily sync populates metadata (`action_capability_filter.py:310-313`). (from F018)
14. **Memory store↔retrieve asymmetry** — writes scoped (`api/memory.py:101`), reads/consolidate raw (`:143,:283/298`): breaks intra-tenant recall AND leaves a cross-tenant read oracle via `GET /retrieve/{ws_A}::abc`. (from F039)
15. **Codegraph webhook signature skip-if-unset** — unsigned payloads accepted when `GITHUB_WEBHOOK_SECRET` unset (`api/codegraph.py:679-684`); now consequential because auto_reindex is settable. (from F022)
16. **F091 exposure widened** — the untested `map_shopify_catalog` now runs automatically on every catalog webhook; a mapper regression corrupts openers/cross-sells with no human in the loop. (from F032+F091)
17. **Fabricated workspace pill** — default `workspaceMeta = 'pilot · 11 op'` still renders (`studio-sidebar.tsx:44,96`; `main-layout.tsx:161` passes no props). (from F038)
18. **Zombie route actively linked** — `/chat/[id]` 404s every SaaS request yet three activity surfaces `router.push` into it (`activity-feed.tsx:236`, `execution-detail.tsx:333`, `recent-activity-widget.tsx:71`). (from F036)
19. **Dormant allow-all helper** — `ApiKeyService.check_permissions` (`api_key_service.py:233-243`) still hardcodes empty=ALL with zero callers; wiring it anywhere re-opens the god-key on a third surface. (from F042)
20. **Admin tool-exposure auto-flip** — `tool_router.py:369-388` still derives is_admin from workspace membership for schema exposure, unconditionally. (from F014)
21. **Flywheel markerless re-starve edge** — COMPLETED runs with zero VERIFIED tasks (or fail-soft ingest Nones) leave no marker and can permanently occupy the DESC batch (`coordinator_service.py:820-821`; `knowledge_flywheel.py:159,166,223`). (from F049)
22. **Route-manifest staleness seam** — the committed `route-manifest.json` is never diffed against a fresh generation in CI. (from F044)

### 4.D Deliberately human-deferred owner actions (documented in-tree, not applied)

1. Flip branch protection `strict=true` and add the new required contexts — ready-to-run command in `docs/runbooks/W12-BRANCH-PROTECTION.md` and `PRD-182:63-66`. Arms F057, and gives F034/F044/F056/F010 lanes teeth.
2. Commit the measured coverage floor over the SEED token in `orchestrator/.coverage-baseline`. Arms the F092 ratchet.
3. Purge `tests/e2e/.auth/user.json` from git history and `git rm --cached` it. Greens the gitleaks lane; closes F012.
4. Merge/pin the mem0 fork per `docs/runbooks/W3-HUMAN-STEPS.md` §3 and run the 401 boot probe. Resolves F011.
5. Schedule the nightly DR dump (runbook §3.2) and decide WAL/PITR. Arms F050 operationally.
6. Make `docker-entrypoint.sh` executable in git (or exec via bash) — the one-line class fix for the F009 residual; then de-mask the smoke lane.
7. Author the Step-2 alembic baseline squash (or stamp the initdb schema), then drop continue-on-error on the from-zero replay. Closes F010/F051 residuals.
8. Decide the `AUTOMATOS_POLICY_PLANE` rollout (staging soak → default ON), and separately whether `HARNESS_SELF_MANAGEMENT_ENABLED` ever flips.
9. Author PRD-184 or equivalent: the dead-code kill list is the single owner of F069, F078 (retirement leg), F081, F082, F036, F079, F080, F084 and the F083 remainder — none of which any wave claimed.

---

## 5. Honest caveats

### 5.1 The register covers 90 of a claimed 94

The July report's own statistics (Appendix C.1) state `findings_total: 94`, but only **90 distinct finding ids** appear anywhere in the published report — F061, F067 and F073 were never assigned in the text, and there is no F094. The authoritative 94-entry `findings[]` register the report references (Appendix C.4, "as reported") was never committed to the repo. This map therefore verifies **all 90 enumerable findings**; up to 4 register entries cannot be enumerated, let alone verified. Recovering the verification-workflow register or the 2026-07-01/02 workflow transcripts would close the gap.

### 5.2 The three UNVERIFIABLE gradings, and why

- **F093** and **F065** exist only as members of merged theme lines (report lines 154 and 155 respectively) with no individual description, file:line, or symptom anywhere in the tree; no PRD or commit cites them and nothing claims to fix them. Their named siblings split both ways (F069 not-done vs F070 fixed; F062/F063 fixed vs F064 not-done), so cluster membership predicts nothing. Any status other than unverifiable would be conjecture.
- **F011** (mem0 fork auth/metadata) was deliberately delivered as a human runbook (`docs/runbooks/W3-HUMAN-STEPS.md` §3): the fix lives in the separate `automatos-mem0` fork and a Railway deploy, both outside this pinned read-only tree. The only in-repo probe is liveness, not auth-asserting. The memory dossier team should verify fork main at the pinned SHA and curl the deployed base for a 401.

### 5.3 Evidence boundaries

- **F013** is graded FIXED partly on sibling-repo evidence (`automatos-shopify` origin/main @ `8dd304c`, PR #10, CI run 28685161381 green) — outside the pinned tree, read-only, and flagged as such in the entry. Actual webhook registration with Shopify is runtime state no static review can see.
- **F057** and the F092 Dependabot-posture check are graded on **live GitHub state** (`gh api`, 2026-07-04) — mutable after this snapshot.
- **F052**'s unpublished-npm claim cites the public registry response (2026-07-04).
- All other claims are file:line into the pinned tree at `77bc9c6d5`.

### 5.4 Method limits

This was static analysis only — no servers, builds, or test suites were run (CI is the platform's only execution gate, by project policy). Claims about runtime behavior (fresh-clone boot failure, from-zero replay failure, lane conclusions) rest on the waves' own CI run logs read via `gh`, plus code reading. Where a fix's end-to-end behavior needs runtime (F022 webhook firing, F089 flywheel-persists-locally, F013 webhook delivery), the entry says so explicitly. The 2026-06-09 structure graph (`graphify-out/graph.json`) predates all waves and was not used for any claim in this document.

### 5.5 Grading conventions that shape the numbers

Statuses are graded on **default-config behavior**, so the large PARTIAL block in policy-governance reflects the `AUTOMATOS_POLICY_PLANE` flag posture, not absent code — the code is real and some of it is good; it just is not what a default deployment runs. Conversely, several FIXED gradings carry residual notes (F002, F006, F020, F025, F032, F038, F049) — those residuals are adjacent defects outside the finding's cited scope, catalogued in §4.C rather than diluting the grade. Where the worklist's expected-fix mapping and the verifier disagreed, the verifier's code-level evidence won (e.g., F078 graded PARTIAL because W2's F006 fix deleted the July-cited endpoint; F013 graded FIXED on sibling CI evidence the worklist did not anticipate).
