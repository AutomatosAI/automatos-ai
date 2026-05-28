# PRD-141: Platform Reliability, Single-Path Consolidation & Auto Self-Management

**Status:** Draft (phased rewrite)
**Original diagnosis:** Devin (review session 2026-05-28) — verified accurate against source
**Phased rewrite + risk review:** Claude (2026-05-28)
**Priority:** P0 — Stability & consistency before new features
**Scope:** Backend orchestrator, memory subsystem, tool routing, HARNESS, context management
**Depends on:** PRD-121 (HARNESS), PRD-138/139 (Semantic Tool Routing), PRD-128 (Notifications)
**Ralph config:** `scripts/ralph/prd-141-reliability.json`
**Branch:** `ralph/prd-141-platform-reliability`

---

## 1. Problem Statement

The platform has grown to **919 Python files**, 105 models, 103 API routers, and 116 platform
actions — largely built by one developer. Verified problems (all confirmed against source on
2026-05-28):

1. **Mem0 instability** — `Mem0Client` uses the synchronous `requests` library (line 16) with
   `time.sleep()` in retry backoff (lines 156, 173), wrapped in `loop.run_in_executor(...)`
   (35 call sites across the memory module). Under load this starves the thread pool and stalls
   *all* async work, not just memory.
2. **Duplicate tool-ranking paths** — `SmartToolRouter._rank_tools_by_similarity()` and
   `ActionSemanticIndex.rank_actions()` both embed and cosine-rank tools independently.
3. **Hardcoded resource limits** — fixed token budgets, iteration caps, and context thresholds
   cause agents to silently bail instead of adapting or reporting.
4. **No negative feedback in tool routing** — `fails_for_intent` affinity exists in the model but
   is never queried.
5. **24 bare `except:` blocks** (verified exactly) and ~1,941 `except Exception` blocks — errors
   are swallowed, debugging is hard.
6. **HARNESS raises tickets it can't act on** — `_apply_approved_board_tasks()` (line 1179) logs
   `"Approved but not auto-applied — manual action required (v1)"` (line 1201) and does nothing.

**Goal:** Make every subsystem use a single, shared, testable pipeline. Stop the crashes first.
Give Auto self-management *last*, behind a flag, only once the base is stable.

---

## 2. What changed in this rewrite (and why)

The original draft was technically accurate but bundled three different goals under one
"reliability" banner and ordered them by *effort*, not by *stability payoff*. This rewrite
re-sequences by impact on "running smoother" and fixes spec bugs found during verification.

### 2.1 Re-prioritisation

| Workstream | Original wave | New phase | Reason |
|---|---|---|---|
| WS-1 Mem0 Async | Wave 2 | **Phase 1** | This is *the* crash source. It ships first. |
| WS-6 Exception Hygiene | Wave 1 | **Phase 0** (scoped) | Telemetry + 24 bare excepts only — gives us the error metrics to *measure* the WS-1 fix. The 1,941-handler rewrite is **out of scope** (regression risk > value). |
| WS-5 Dynamic Limits | Wave 4 | **Phase 2** | "Silent bail" is user-visible. Report-to-user lands before risky budget maths. |
| WS-2 Unified Tool Pipeline | Wave 3 | **Phase 3** | Code health, not stability. After the base is solid. |
| WS-4 Negative Signals | Wave 5 | **Phase 4** | Accuracy, additive. Depends on WS-2. |
| WS-3 HARNESS Self-Mgmt | Wave 6 | **Phase 5** (flag-gated) | Highest risk. Adds autonomy = adds risk surface. Opt-in, last, after soak. |

### 2.2 Spec bugs fixed (verified against source)

- **WS-3 used wrong platform-action names.** The draft called `platform_assign_tool` /
  `platform_unassign_tool`. The real actions are **`platform_assign_tool_to_agent`** /
  **`platform_unassign_tool_from_agent`** (`modules/tools/discovery/actions_assignments.py`).
  Stories use the correct names.
- **WS-3 referenced `harness._apply_single_approved_task()` — that method does not exist.** The
  approve/reject path reuses the existing `_auto_apply_prescription()` flow instead.
- **WS-3 `platform_create_routing_rule` does not exist.** `routing_rule_add` is **dropped from the
  v1 prescription vocabulary** — adding it would be net-new platform-action work outside this PRD.
- **WS-3 `/approve` `/reject` handler had no authorization.** Any inbound webhook payload could
  approve a privilege change. Stories add an authz check (verified channel identity → workspace
  admin) before any mutation.
- **WS-4 used `asyncio.ensure_future(...)` to open a DB session per tool call.** Under load this
  exhausts the connection pool — the exact failure WS-1 fixes. Replaced with a **batched recorder**
  (single background drain, one session per flush). `wilson_lower_bound` confirmed already present
  in `edge_builder.py`, so confidence maths is reused.

### 2.3 Out of scope (moved from "do everything")

- Mass rewrite of ~1,941 `except Exception` handlers. Phase 0 builds the telemetry util and fixes
  the 24 *bare* `except:`; hot-path handlers adopt `record_error()` opportunistically, not in a
  big-bang sweep.
- `routing_rule_add` prescription type (no backing platform action).
- Any UI work (PRD is backend-only).

---

## 3. Phase Plan Overview

Each phase is independently shippable and testable. Phases land in order; later phases depend on
earlier ones only where noted.

| Phase | Theme | Stories | Risk | Ships value |
|---|---|---|---|---|
| **0** | Error Telemetry Foundation (WS-6 core) | US-001–002 | Low | Observability to measure everything else |
| **1** | Mem0 Async Stability (WS-1) | US-003–008 | Low | **Stops the production crashes** |
| **2** | Dynamic Resource Limits (WS-5) | US-009–012 | Medium | Kills silent agent bails |
| **3** | Unified Tool Selection Pipeline (WS-2) | US-013–016 | Medium | Removes #1 code duplication |
| **4** | Tool Routing Negative Signals (WS-4) | US-017–019 | Low | Better tool-selection accuracy |
| **5** | HARNESS Self-Management (WS-3) | US-020–026 | **High** | Auto self-manages (flag-gated) |

**Each story carries:** verifiable acceptance criteria, unit tests, "type checks pass", "pytest
green". **Each phase ends with a review/gate** (code-reviewer agent; canary soak for Phases 1 & 5).

---

## 4. Phase 0 — Error Telemetry Foundation

**Goal:** Build structured error telemetry and kill the 24 bare `except:` blocks. This is the
*safe* half of WS-6. It ships first because it gives us the error-rate metric that proves Phase 1.

**Explicitly NOT in this phase:** rewriting the ~1,941 `except Exception` handlers. That is a
regression risk (every previously-swallowed error becomes a potential new crash) and is deferred to
opportunistic hot-path adoption.

### US-001 — Create `record_error()` structured telemetry util
- **File:** `orchestrator/core/utils/exception_telemetry.py` (new)
- `record_error(*, subsystem, operation, error, workspace_id=None, agent_id=None, action_name=None, extra=None)`
  logs to the `automatos.errors` logger with `exc_info=True` and a `structured_error` extra dict.
- Truncates `error_message` to 500 chars; tolerates `None` workspace_id.
- **Tests:** `tests/test_exception_telemetry.py` — structured log emitted; message truncated; None
  workspace doesn't crash.
- **AC:** unit tests green; type checks pass.

### US-002 — Replace all 24 bare `except:` and add CI gate
- Replace every bare `except:` with `except Exception:` across `orchestrator/`.
- **Add CI gate:** `scripts/ci/check-no-bare-except.sh` — fails if
  `grep -rn "except:" orchestrator/ --include="*.py" | grep -v "except " | grep -v __pycache__`
  returns any hits.
- **AC:** grep gate returns zero; existing tests still green; type checks pass.

**Phase 0 review gate:** code-reviewer agent on the telemetry util; confirm no behaviour change
from the bare-except replacements (each must be a pure widening, not a logic change).

---

## 5. Phase 1 — Mem0 Async Stability (THE crash fix)

**Goal:** Make Mem0 access natively async, per-workspace isolated, and self-healing. Self-contained.
This is the highest-leverage stability work in the PRD.

### US-003 — Convert `Mem0Client` to `httpx.AsyncClient`
- **File:** `orchestrator/modules/memory/integrations/mem0_client.py`
- Replace `import requests` with `import httpx`; `_request()` → `async def`; use a pooled
  `httpx.AsyncClient`; replace `time.sleep(wait)` (lines 156, 173) with `await asyncio.sleep(wait)`.
- Convert `add()`, `search()`, `get_all()`, `delete()` to `async def`.
- **AC:** `grep -rn "import requests" orchestrator/modules/memory/` → zero;
  `grep -rn "time\.sleep" orchestrator/modules/memory/` → zero; tests green; type checks pass.

### US-004 — Per-workspace circuit breaker
- Remove module-level `_breaker` singleton. Add `_breakers: Dict[str, _CircuitBreaker]` keyed by
  `workspace_id`; `_request()` accepts `workspace_id`.
- **Tests:** workspace A open breaker doesn't affect workspace B; opens after threshold; half-open
  probe after cooldown.
- **AC:** isolation test green; type checks pass.

### US-005 — Drop `run_in_executor` wrappers in `UnifiedMemoryService`
- **File:** `orchestrator/modules/memory/unified_memory_service.py`
- Replace every `loop.run_in_executor(None, lambda: self._mem0.xxx(...))` with `await self._mem0.xxx(...)`.
  Pass `workspace_id` on each call.
- **AC:** `grep -rn "run_in_executor.*_mem0\|run_in_executor.*mem0" orchestrator/` → zero; tests green.

### US-006 — Proactive Mem0 health probe
- **File:** `orchestrator/services/heartbeat_service.py` (extend existing tick)
- Every `MEM0_HEALTH_PROBE_INTERVAL_SECONDS` (default 30), ping Mem0 health; on failure trip all
  breakers; on recovery reset. Config flags `MEM0_HEALTH_PROBE_ENABLED` (default true).
- **AC:** probe-trips-all and probe-resets tests green; config read via `config.py` only.

### US-007 — Tighten Mem0 timeouts and cooldown
- **File:** `orchestrator/config.py` — `MEM0_WRITE_TIMEOUT_SECONDS` 15.0 → **5.0**;
  `MEM0_CIRCUIT_COOLDOWN_SECONDS` 300 → **60**.
- **AC:** write-timeout-respected test green; no `os.getenv()` outside `config.py`.

### US-008 — [GATE] Load test + canary soak
- Manual: 50 concurrent Mem0 searches complete with no thread-pool starvation.
- code-reviewer agent on the full Phase 1 diff.
- INBUILD canary: deploy, 24h soak, watch widget/chat error rate via `record_error(subsystem="memory")`.
- **AC:** load test passes; soak shows no error-rate regression; rollback path documented.

**Deletions:** `import requests`, module-level `_breaker`, all Mem0 `run_in_executor` wrappers,
`time.sleep` in memory retry.

---

## 6. Phase 2 — Dynamic Resource Limits (kills silent bails)

**Goal:** Limits become model-aware and configurable; agents *report* when they hit a wall instead
of silently stopping. Sequence the safe user-facing message before the riskier budget maths.

### US-009 — Report to user when hitting limits
- **File:** `orchestrator/consumers/chatbot/service.py` — on `iteration >= max_iterations`, emit a
  `{"type": "limit_reached", ...}` user-visible event.
- **File:** `orchestrator/services/coordinator_service.py` — on budget exceeded, emit a
  `BUDGET_WARNING` event before pausing.
- **AC:** iteration-limit-reports + budget-event tests green.

### US-010 — Configurable power-mode caps via `system_settings`
- **File:** `orchestrator/services/coordinator_service.py` — replace hardcoded `_POWER_MODE_CAPS`
  dict with `_get_power_mode_caps(power_mode)` reading `system_settings` with hardcoded defaults.
- **AC:** system_settings override respected; falls back to defaults; tests green.

### US-011 — Model-proportional context budgets
- **File:** `orchestrator/modules/memory/context_router.py` — `_compute_budgets(context_window)`
  allocates section budgets as proportions of the usable window. Static `CONTEXT_BUDGET_*` become
  fallbacks only.
- **AC:** 128K model gets larger budgets than 8K; unknown model uses defaults; tests green.

### US-012 — Adaptive context-guard thresholds
- **File:** `orchestrator/core/context_guard.py` — `_thresholds_for_model(context_window)` returns
  `(compact_threshold, keep_recent_turns)` scaled by window (0.90/12 at 200K → 0.70/3 at <8K).
- **AC:** threshold/turns adapt per window; existing context-guard tests updated and green.
- **Note:** highest-risk story in this phase — wrong thresholds cause provider 400s or OOM context.
  Verify against an 8K and a 128K model before merge.

**Phase 2 review gate:** code-reviewer agent; manual 8K-vs-128K verification.

**Deletions:** hardcoded `_POWER_MODE_CAPS`, static `COMPACT_THRESHOLD`/`KEEP_RECENT_TURNS` as sole source.

---

## 7. Phase 3 — Unified Tool Selection Pipeline (dedup)

**Goal:** One ranking pipeline: `ActionSemanticIndex` → `GraphRouter`. `SmartToolRouter` keeps only
intent classification, tool_choice, and ALWAYS_INCLUDE enforcement.

### US-013 — Add `get_by_category()` / `get_by_tags()` to `ActionRegistry`
- **File:** `orchestrator/modules/tools/discovery/action_registry.py`
- **AC:** `get_by_category("harness")` and `get_by_tags(["monitoring"])` return expected actions; tests green.

### US-014 — `SmartToolRouter` delegates ranking to `GraphRouter`; delete embedding path
- **File:** `orchestrator/consumers/chatbot/smart_tool_router.py`
- Delete `_embedding_manager`, `_tool_embeddings`, `_embeddings_initialized`, `_embeddings_init_lock`,
  `_ensure_embeddings()` (lines 139–182), `_rank_tools_by_similarity()` (lines 184–229). Route via
  `GraphRouter.rank_chains()` with a category-filter fallback.
- **AC:** `grep` for those symbols → zero; no import from `core.math.vector_operations` or
  `core.llm.embedding_manager`; tests green.

### US-015 — Delete `TOOL_CATEGORIES`/`INTENT_TO_TOOLS`; filter via registry
- Replace `_filter_tools_by_categories()` with `_filter_tools_by_intent()` reading `ActionRegistry`
  categories via `_INTENT_TO_REGISTRY_CATEGORIES`.
- **AC:** `grep TOOL_CATEGORIES smart_tool_router.py` → zero; a newly-registered action in a category
  is auto-discoverable without code change; tests green.

### US-016 — [GATE] Deletion + review gate
- **AC:** all Phase 3 acceptance greps return zero; code-reviewer agent confirms no new duplicate
  ranking path introduced; existing chatbot routing tests green.

---

## 8. Phase 4 — Tool Routing Negative Signals (accuracy)

**Goal:** The graph learns from failure, not just success. Additive. Depends on Phase 3.

### US-017 — Query and penalize negative affinities in `GraphRouter`
- **File:** `orchestrator/modules/tools/discovery/graph_router.py` — `_query_affinities()` returns
  `(positive, negative)`; `_expand_with_graph()` subtracts `fails_for_intent` penalties.
- **AC:** tool with `fails_for_intent` scores lower than without; tests green.

### US-018 — Add `failed_after` edge type to batch builder
- **File:** `orchestrator/core/services/edge_builder.py` — compute `failed_after` pairs in the
  nightly batch; reuse existing `wilson_lower_bound`.
- **AC:** `failed_after` edges produced; `failed_after` not followed during expansion; tests green.

### US-019 — Batched incremental signal recorder (NOT fire-and-forget)
- **File:** `orchestrator/modules/tools/tool_router.py`
- After tool execution, enqueue a lightweight signal on an in-process `asyncio.Queue`. A **single
  background drain task** batches upserts (every N signals or T seconds) using **one DB session per
  flush**. Do **not** open a DB session per tool call; do **not** use bare `asyncio.ensure_future`.
- **AC:** success → `used_after`; failure → `failed_after` + `fails_for_intent`; repeated signals
  increment `sample_count` (no duplicate edges); a flush uses exactly one session; tests green.
- **Note:** this story explicitly fixes the connection-pool-exhaustion risk in the original draft.

**Phase 4 review gate:** code-reviewer agent; confirm no per-call DB session.

---

## 9. Phase 5 — HARNESS Self-Management (HIGH RISK — flag-gated, last)

**Goal:** Let Auto execute approved board tasks, escalate high-risk changes, roll back regressions,
and prescribe a wider vocabulary — **all behind `HARNESS_SELF_MANAGEMENT_ENABLED` (default false)**,
enabled per-workspace only after Phases 1–2 have soaked.

**Why last and gated:** self-management *adds* risk surface. An agent that reassigns tools, changes
power modes, and auto-rolls-back is a reliability liability on an unstable base. It ships only once
the base is proven, and only opt-in.

### US-020 — Feature flag + `_parse_harness_task()`
- Add `HARNESS_SELF_MANAGEMENT_ENABLED` to `config.py` (default false).
- Add `_parse_harness_task(task)` → prescription dict (parses `[HARNESS] {change_type} for {target}`
  title + Current/Proposed from description; resolves `target_id`).
- **AC:** valid task parses; non-HARNESS task → None; flag defaults false; tests green.

### US-021 — Execute approved board tasks (flag-gated) + snapshot current value
- **File:** `orchestrator/services/harness_service.py`
- Replace the warning-only loop (lines 1193–1202) with execution via the **existing**
  `_auto_apply_prescription()`. Mark task `done` + `harness-applied` tag on success. Before applying,
  snapshot current state into the changelog (`current_value_before`) for rollback.
- **AC:** approved task is applied (not just logged) when flag on; nothing happens when flag off;
  snapshot recorded; tests green.

### US-022 — Auto-rollback on regression
- In `_phase_diagnose()`, flag agents that regressed after a last-run auto-apply; in
  `_phase_prescribe()`, emit a `rollback` prescription (risk=1) reverting to `current_value_before`.
- **AC:** regression-after-apply → rollback prescription with risk=1 reverting to snapshot; tests green.

### US-023 — Expanded prescription vocabulary (correct action names)
- Add handlers in `_auto_apply_prescription()` for: `tool_assignment_add` →
  **`platform_assign_tool_to_agent`**; `tool_assignment_remove` →
  **`platform_unassign_tool_from_agent`**; `power_mode_upgrade`/`power_mode_downgrade` →
  `platform_update_agent`; `rollback`.
- **`routing_rule_add` is dropped for v1** (no `platform_create_routing_rule` action exists).
- **AC:** each change type maps to a real, existing platform action; routing_rule_add absent; tests green.

### US-024 — Telegram/Slack escalation for high-risk changes
- In `_phase_apply()`, when `risk_score >= 4` and the workspace has a channel, call the **existing**
  `core.services.notification_service.send_workspace_notification(...)` with approve/reject instructions.
- **AC:** high-risk → notification sent; no channel → skipped; tests green.

### US-025 — `/approve` `/reject` command handler WITH authz
- **File:** `orchestrator/api/harness_commands.py` (new)
- **Authorization (mandatory):** verify the inbound command's channel identity maps to a workspace
  **admin** before any mutation. Reject unauthenticated/unauthorized commands.
- `/approve` finds the board task by `rx:{rx_id}` tag, marks done, applies via the **existing**
  `_auto_apply_prescription()` flow (NOT an invented `_apply_single_approved_task`). `/reject` marks
  rejected and blocks re-proposal. Idempotent (duplicate approve = no double-apply).
- **AC:** unauthorized command rejected; approve applies once (idempotent); reject blocks; unknown
  rx_id → "not found"; tests green.

### US-026 — [GATE] HARNESS self-management canary
- Enable `HARNESS_SELF_MANAGEMENT_ENABLED` on **one** non-critical workspace. Create an approved
  board task, verify it executes; verify a forced regression triggers rollback; verify escalation +
  authz on approve. 48h soak. code-reviewer + security-auditor agents on the full Phase 5 diff
  (focus: authz on the command path).
- **AC:** canary executes approved task; rollback fires; unauthorized approve blocked; soak clean.

**Deletions:** the `"not yet auto-applied (v1 limitation)"` warning and `approved_pending` changelog
entry (replaced by real execution).

---

## 10. Cross-Cutting Review & Test Gates

Every story:
- **Unit tests** for the specified scenarios; `pytest <path>` green.
- **Type checks pass** (mypy/pyright).
- No `os.getenv()` outside `config.py`; no hardcoded values; no backward-compat shims (delete what
  you replace in the same PR).

Every phase:
- **code-reviewer agent** on the phase diff; CRITICAL/HIGH addressed before merge.
- **Deletion verification:** run the acceptance greps — zero hits = pass.

Risky phases (1, 5):
- **Canary soak** on a single workspace before broad rollout. Rollback path documented and tested.

Phase 5 additionally:
- **security-auditor agent** on `api/harness_commands.py` (the authz surface).

---

## 11. Out of Scope

- Mass rewrite of ~1,941 `except Exception` handlers (telemetry adoption is opportunistic).
- `routing_rule_add` prescription / `platform_create_routing_rule` action.
- New UI components (backend-only PRD).
- New LLM providers or model integrations.
- Cross-repo work (automatos-shopify, automatos-widget-sdk, automatos-skills).

---

## 12. Success Metrics

| Metric | Current | Target | How measured |
|---|---|---|---|
| Mem0-related stalls under load | crashes | 0 thread-starvation events | 50-concurrent load test (US-008) |
| Mem0 error rate | unknown | <0.1% requests | `record_error(subsystem="memory")` count |
| Tool-ranking duplicate paths | 2 | 1 | grep for embedding gen in chatbot/ (US-016) |
| Bare `except:` blocks | 24 | 0 | CI gate (US-002) |
| Agent "silent bail" rate | unknown | measurable then reducible | `limit_reached` event count (US-009) |
| HARNESS approved tasks executed | 0% | 100% (flag-on workspaces) | changelog `applied_from_approved` (US-021) |
| HARNESS unauthorized mutations | possible | 0 | authz tests + canary (US-025/026) |

---

**End of PRD-141 (phased rewrite).**
