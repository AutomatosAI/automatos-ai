# PRD-177: Wave 7 — Operating-Graph Learning Loop Closure

**Phase:** C — Moat Compounding (weeks 16–24)
**Branch:** `feat/w7-operating-graph-loop` · **Worktree:** `automatos-ai-prd177`
**Dependencies:** Waves 1 + 4 (F001 spine fix + unified policy plane) — **both merged to main (`5768c2d5b`)**
**Build size:** M · **Risk:** Medium
**OS Review refs:** §8 (graph-native moat), §12.3, roadmap Phase C

---

## Overview

The learning loop is **write-mostly**: telemetry logs every execution, edges recompute nightly, `GraphRouter` reads them — but the loop is starved at four inputs so the operating graph never compounds from live traffic. This wave closes those inputs so the per-tenant, outcome-labeled edge dataset — the platform's **only defensible moat** (§8) — actually accumulates from production.

**Owner decision (locked 2026-07-03): learned edges are PER-TENANT.** GraphRouter reads must filter by `workspace_id`. A global graph is not a moat a competitor can't cold-start; a per-tenant one is.

**⚠️ Framing correction vs the source draft:** Do **NOT** "move `GraphRouter.rank_chains` from offline eval to the live hot path." The report (§8, and `graph_router.py` itself) confirms learned `used_after`/affinity edges **already** reach default live routing via `rank_chains` under the default-true `SEMANTIC_TOOL_ROUTING` flag. That code works — leave it. The only F015 remainder is the narrow prompt-catalog **chain-hints** path gated behind `TOOL_ROUTING_GRAPH` (default false, `config.py:741`). Verify current behavior before editing.

---

## Ownership boundary (parallel-safe)

This wave runs concurrently with W8 (PRD-178) and W9 (PRD-179). To avoid collisions:

- **W7 OWNS:** `modules/tools/execution/telemetry.py`, `modules/tools/discovery/graph_router.py`, `core/services/edge_builder.py`, `modules/tools/sync/composio_action_sync.py`, `jobs/sync_composio_actions.py`, `modules/tools/tool_router.py`, `consumers/chatbot/service.py` (caller_context write site), `config.py` (routing flags), **and `modules/context/modes.py` lines ~40–46 ONLY** (the chain-hints gate).
- **W7 MUST NOT TOUCH:** `modes.py` heartbeat/planning regions (~76–88, ~141–149 — W9 owns), `platform_executor.py` (W8 owns), `modules/context/service.py` (W9 owns), field/vector_field (W8 owns).

Confirm line numbers by grep before editing — the report's numbers have drifted.

---

## Findings & Scope

| Finding | Issue (verified) | Fix |
|---|---|---|
| **F016** | `telemetry.py:~59` logs `action_name=tool_name[:255]` → every Composio action collapses to one `composio_execute` node; the 856-app surface never learns | Resolve and log the real action name (e.g. `SLACK_SEND_MESSAGE`) |
| **F017** | Chat call site threads only `{user_id}` into `caller_context`; `user_query`/`turn_id` never reach the edge builder, so `succeeds_for_intent` affinities never materialize | Thread `user_query` + `conversation_id`/`turn_id` into `caller_context` at the chat call site (`service.py:~1317`) |
| **F018** | `ComposioActionMetadata` sync has no scheduler; the destructive-action gate in `tool_router.py` returns **fail-OPEN** (`return True, "... (fail open)"`, ~lines 960–996) on an empty/missing metadata table | Register the sync on the scheduler; flip the gate to **fail-CLOSED** for destructive actions when metadata is absent |
| **F015** | Prompt-catalog chain hints gated behind `TOOL_ROUTING_GRAPH` (default false, `config.py:741`) | Enable/verify the gated chain-hints path folds learned edges. **Do not touch the already-live `rank_chains` default path.** |
| **Per-tenant** | `graph_router.py` has zero `workspace_id` references; edges are written per-workspace (`edge_builder.py:~207`) but read globally → cross-tenant leak + moat-defeater | Add `workspace_id` filter to all GraphRouter edge reads |

---

## Stories (test-first — write the failing test, make it green, refactor)

### S1 · Composio per-action telemetry (F016) — S
**Files:** `modules/tools/execution/telemetry.py`, plus the composio execution call path that knows the resolved action.
**Test:** `test_composio_action_telemetry` drives a `composio_execute` for `SLACK_SEND_MESSAGE`, asserts the `ToolExecutionLog.action_name` is the resolved action (not `composio_execute`), and the recomputed edge carries that action.
**Notes:** The resolved action is available in the execution parameters (only param *keys* are logged today). Thread the resolved action name to `log_tool_execution`. Do not log secret param *values* — keep the keys-only privacy posture.

### S2 · Intent threading into caller_context (F017) — S
**Files:** `consumers/chatbot/service.py:~1317` (write site); `core/services/edge_builder.py:161-171` (already consumes intent — do not rewrite, just feed it).
**Test:** `test_intent_affinity_capture` runs a turn with a known `user_query`, asserts `caller_context` carries `user_query` + `conversation_id`, and after nightly recompute a `succeeds_for_intent` affinity edge exists for the tool that answered.
**Notes:** `telemetry.py` already reads `ctx.get("user_query")` — the gap is purely the chat write site not populating it.

### S3 · Metadata sync scheduler + fail-closed gate (F018) — S
**Files:** `modules/tools/sync/composio_action_sync.py` and/or `jobs/sync_composio_actions.py` (scheduler entry), `modules/tools/tool_router.py:~960-996` (gate).
**Test:** `test_metadata_sync_scheduled` asserts a sync job is registered on the scheduler (alongside the nightly edge recompute). `test_destructive_gate_fail_closed` asserts that with an empty `ComposioActionMetadata` table a destructive action is **denied / requires confirmation**, not allowed.
**Notes:** Register via the same scheduler that runs `nightly_edge_recompute`. Add a config flag for the fail-closed default (via `config.py`, never `os.getenv` inline). Preserve fail-open only for **non-destructive** actions if needed; destructive = fail-closed.

### S4 · Chain-hints gate verification (F015) — S
**Files:** `config.py:741` (`TOOL_ROUTING_GRAPH`), `modules/context/modes.py:~40-46` (chain-hints), `graph_router.py` (chain expansion).
**Test:** `test_chain_hints_use_learned_edges` sets `TOOL_ROUTING_GRAPH=true`, asserts prompt-catalog chain hints reflect a learned `used_after` edge; with the flag false, current behavior is unchanged.
**Notes:** This is a wiring + verification story, not a rewrite. If the gated path already works, the deliverable is the test proving it + flipping the default only if the eval (S6) supports it.

### S5 · Per-tenant workspace filter on GraphRouter reads — S
**Files:** `modules/tools/discovery/graph_router.py` (all edge-read methods, incl. `rank_chains` and chain expansion at `~280`).
**Test:** `test_graph_router_tenant_isolation` seeds edges in workspaces A and B, asserts `rank_chains(..., workspace_id=A)` never returns B's edges and vice versa.
**Notes:** Add `workspace_id` as a required parameter to edge reads; filter `.where(...workspace_id == workspace_id)`. Thread `workspace_id` from the call sites (context/service already carries it). Remove any unfiltered global-read fallback.

### S6 · Uplift eval — the business gate — M
**Files:** new `orchestrator/evals/operating_graph_uplift.py` (or extend the existing `experiment.py` harness if it fits), + a non-required CI job.
**Test / deliverable:** an offline eval measuring learned-edge selection accuracy vs **BM25** and **embedding** baselines, **per tenant**, emitting an uplift number.
**Gate:** If uplift is **< 5–10 points**, the moat claim fails honest review — record the number, do **not** flip `TOOL_ROUTING_GRAPH` on, and flag it. Do not fabricate a passing number. A published sub-threshold number is a valid, honest outcome.

---

## Verification (no servers, no Docker, no live browser)

Per project convention: verify with `python -m py_compile` on every changed file + **pure pytest** (no DB/network/Qdrant/Composio calls — mock at the boundary). CI is the integration gate. Do **not** run `next dev`, headless Chromium, or spin services.

```
python -m py_compile <changed files>
python -m pytest orchestrator/modules/tools -k "telemetry or graph_router or metadata" -q
python -m pytest orchestrator/... <new tests> -q
```

---

## Conventions (non-negotiable — see automatos-ai/CLAUDE.md)

- No `os.getenv()` outside `config.py`. New flags go through the canonical config module.
- No backward-compat shims; delete what you replace in the same commit.
- Immutable patterns; small focused functions; comprehensive error handling.
- No new tables if an existing one fits; no new tools where an existing one extends.
- Commit to `feat/w7-operating-graph-loop`. **Do not push or open a PR** — stop after local verify and report.

## Success metrics
- Composio actions carry per-action names in the graph (not collapsed).
- Intent affinity edges materialize from real chat traffic.
- Destructive-action gate denies on absent metadata (fail-closed).
- Zero cross-tenant edge leakage.
- A published per-tenant uplift number vs BM25/embedding baselines.
