# PRD-142 Wave 0 — Measurement First ("Is It Working?" Dashboard)

> **Parent:** `PRD-142-CORE-DESIGN-REVIEW.md` §12, Wave 0.
> **Status:** Build-PRD — drafted 2026-05-29. Gerard signed off the design review; the build
> gate (both PRD-141s merged to `main`: #394 platform-reliability, #395 widget-vertical, #396
> widget fix) is **CLEARED**. This is the first wave's build spec.
> **Type:** Extension + instrumentation. Backend + a single frontend dashboard section. **No new
> features, no new frontend page.** Reuse-first per CLAUDE.md §2.
> **Verified against:** branch `docs/prd-142-design-and-wave0`, code reads 2026-05-29.
> **Ralph config:** `scripts/ralph/prd-142-wave0.json`.

---

## 1. The founding question

The design review's whole premise is that we cannot today answer **"is the vision working?"** with a
number. Wave 0 fixes exactly that — and nothing else. It extends the PRD-141 Phase-0 telemetry
foundation (`record_error`, `widget_event_log`, `heartbeat_results`) into **one dashboard section
that answers the founding question** and gives every later wave a measurable before/after.

Five metrics, in priority order:

| # | Metric | The question it answers | Status today | Wave 0 action |
|---|---|---|---|---|
| 1 | **Error rate by subsystem** | "What is actually failing, and where?" | `record_error` logs to the `automatos.errors` logger only — **no queryable sink** (`core/utils/exception_telemetry.py:22,68`) | Add a queryable sink + aggregation endpoint |
| 2 | **Mission success rate** | "Do the things users start actually finish?" | **Real** endpoint exists, workspace-scoped, union of missions+workflows w/ 7-day trend (`api/analytics_real.py:53`) — but the dashboard card shows a **fake `85.0`** (`core/services/analytics_engine.py:118`) | Surface the real one; **delete the placeholder** |
| 3 | **Widget engagement** | "Are the widgets we ship being used?" | `widget_event_log` table + writer exist (9 event types, indexed) — **no aggregation endpoint** (`core/models/widget_event_log.py`, `modules/widgets/telemetry.py`) | Add an aggregation endpoint over the existing index |
| 4 | **Activation** | "Do new workspaces reach first value?" | **GAP** — no definition, no query | Define + compute first-successful-mission rate |
| 5 | **Per-primitive health** | "Which of the 8 primitives is green right now?" | **Partial** — `heartbeat_results` + `GET /api/heartbeat/analytics` exist (`api/heartbeat.py`) | Roll heartbeat findings up per primitive |

Then one capstone: **assemble the five into a dashboard section** on the **existing** dashboard page.

---

## 2. Reuse map (read before writing a line of code)

Everything below already exists. Wave 0 **extends** it; it does not rebuild it.

| Concern | Reuse this | Verdict |
|---|---|---|
| Structured error emit | `record_error(*, subsystem, operation, error, workspace_id, agent_id, action_name, extra)` → `automatos.errors` logger, `structured_error` extra dict (`core/utils/exception_telemetry.py:29`) | **Extend** — give it a queryable sink; do **not** change its signature or its never-raises contract |
| Widget events | `WidgetEventLog` + `WIDGET_EVENT_TYPES` (9 types) + `idx_widget_event_log_type_created` (`core/models/widget_event_log.py`); writer `log_widget_event` (`modules/widgets/telemetry.py:35`) | **Reuse** — read-only aggregation; no schema change |
| Mission success rate | `GET /api/analytics/dashboard/success-rate` (`api/analytics_real.py:53`) — already unions `OrchestrationRun` + `WorkflowExecution`, 7-day trend, workspace-scoped | **Reuse as-is** |
| Real workflow/mission metrics | `AnalyticsEngine._get_workflow_metrics` (`core/services/analytics_engine.py:144-174`) — real `OrchestrationRun.state=="completed"` / `OrchestrationTask.state=="verified"` | **Reuse** the real half; **delete** the fake `_get_agent_metrics` placeholders (`:118-121`) |
| Per-primitive health | `heartbeat_results` table + `GET /api/heartbeat/analytics` (`api/heartbeat.py:284`) | **Reuse** — roll up, don't rebuild |
| Dashboard page | `frontend/app/dashboard/page.tsx` → `frontend/components/dashboard/dashboard.tsx` (`Dashboard`, `MetricCards`, recharts, `useSystemHealth`/`useAllMetrics`) | **Extend** — add ONE section to this page; **no new route** |
| Analytics hooks | `frontend/hooks/use-analytics-api.ts`, `use-unified-analytics.ts` | **Extend** — add hooks for the new endpoints |

**Canonical-term note:** the live tables are still named `Workflow`/`WorkflowExecution`. Wave 0 reads
them where the existing union already does, but every **user-facing tile label** uses the canonical
noun **Mission**. The legacy tables are migrated-and-dropped in **Wave 3** (per
`PLAYBOOK-ENGINE-DESIGN.md` §4.2) — **out of scope here**. Do not add new `WorkflowExecution` reads.

---

## 3. Definition of Done (the whole wave)

- [ ] The dashboard shows **five real numbers** for the founding question — zero placeholders.
- [ ] The fake `successRate: 85.0` (and the sibling hardcoded `avgExecutionTime`/`totalTokensUsed`/
      `recentExecutions` placeholders) are **deleted** from `analytics_engine.py`. A grep gate proves it.
- [ ] Error-rate-by-subsystem is queryable from a persisted sink, not just log scraping.
- [ ] Widget engagement and activation each have one endpoint backed by a real query.
- [ ] Per-primitive health rolls up from `heartbeat_results` with no rebuild of the heartbeat system.
- [ ] The dashboard section lives on the **existing** dashboard page — **no new route, no new page**.
- [ ] Every new endpoint is workspace-scoped and has a pytest; tenant-isolation is asserted.
- [ ] Type checks pass; `pytest` green; existing suites stay green.

---

## 4. User stories

### Phase A — Error rate by subsystem (the one true gap in telemetry)

**US-001 — Persist `record_error` to a queryable sink**
- *As the platform, I need structured errors stored in a queryable table so the dashboard can show
  error rate by subsystem instead of scraping logs.*
- **Reuse-checked:** no existing error-event table exists (verified 2026-05-29); the closest is the
  `automatos.errors` logger. Follow the `widget_event_log` / `ToolExecutionLog` single-table-JSONB
  append-only pattern for consistency.
- AC:
  - Create `error_events` table (Alembic migration): `id BIGSERIAL PK`, `subsystem VARCHAR(64) NOT NULL`,
    `operation VARCHAR(128) NOT NULL`, `error_type VARCHAR(128)`, `error_message VARCHAR(500)`,
    `workspace_id UUID NULL`, `agent_id INT NULL`, `action_name VARCHAR(128) NULL`,
    `event_data JSONB NOT NULL DEFAULT '{}'`, `created_at TIMESTAMP NOT NULL DEFAULT now()`.
  - Index `idx_error_events_subsystem_created (subsystem, created_at)` and
    `idx_error_events_workspace_created (workspace_id, created_at)`.
  - Add an opt-in persistence path so `record_error` rows also land in `error_events` **without**
    changing `record_error`'s signature or its **never-raises** contract — a buffered/best-effort
    writer (mirroring `log_widget_event`: catch-all, rollback, swallow). The logger emit stays.
  - `record_error` is migration-safe: if the table/sink is unavailable, it logs and returns (no crash).
  - Test `orchestrator/tests/test_error_events_sink.py`: `test_record_error_persists_row`,
    `test_record_error_sink_failure_is_swallowed`, `test_record_error_truncates_message_in_db`,
    `test_none_workspace_persists`.
  - Type checks pass; `pytest orchestrator/tests/test_error_events_sink.py` green.

**US-002 — Error-rate-by-subsystem aggregation endpoint**
- *As an operator, I need one endpoint that returns error counts grouped by subsystem over a window.*
- AC:
  - `GET /api/analytics/errors/by-subsystem?window=24h` (workspace-scoped via
    `get_request_context_hybrid`) returns `{ window, total, by_subsystem: [{subsystem, count, rate}], generated_at }`.
  - `rate` = subsystem count ÷ total over the window (0 when total is 0; no divide-by-zero).
  - Query uses `idx_error_events_subsystem_created`; no full-table scan.
  - Test `orchestrator/tests/test_errors_by_subsystem_endpoint.py`: `test_groups_by_subsystem`,
    `test_window_filtering`, `test_empty_window_returns_zero`, `test_workspace_isolation`.
  - Type checks pass; `pytest` green.

### Phase B — Reuse + de-fake the metrics that already exist

**US-003 — Delete the fake agent-metrics placeholders**
- *As a user, I need the dashboard to show real numbers, not a hardcoded 85%.*
- AC:
  - In `core/services/analytics_engine.py` `_get_agent_metrics` (`:107`): remove the hardcoded
    `successRate: 85.0`, `avgExecutionTime: 2.5`, `totalTokensUsed: 0`, `recentExecutions: 0`.
  - Replace with real values where a real source exists (mission success via the same union as
    `/api/analytics/dashboard/success-rate`); **omit** any field with no real source rather than fake it.
  - `grep -n "85.0\|2.5.*Placeholder\|# Placeholder" orchestrator/core/services/analytics_engine.py`
    returns **zero** matches.
  - Test `orchestrator/tests/test_analytics_engine_real_metrics.py`: `test_agent_metrics_no_placeholders`,
    `test_success_rate_matches_orchestration_runs`.
  - Type checks pass; `pytest` green; existing analytics tests green.

**US-004 — Widget-engagement aggregation endpoint**
- *As an operator, I need widget engagement counts so I can see if shipped widgets are used.*
- **Reuse-checked:** `widget_event_log` + `idx_widget_event_log_type_created` exist; only the read side is missing.
- AC:
  - `GET /api/analytics/widget-engagement?window=7d` (workspace-scoped; resolve sites for the
    workspace) returns `{ window, by_event_type: [{event_type, count}], sessions, generated_at }`
    grouped over `WIDGET_EVENT_TYPES`.
  - Query uses `idx_widget_event_log_type_created`; read-only (no writes to `widget_event_log`).
  - Test `orchestrator/tests/test_widget_engagement_endpoint.py`: `test_groups_by_event_type`,
    `test_window_filtering`, `test_workspace_site_scoping`, `test_empty_returns_zero`.
  - Type checks pass; `pytest` green.

### Phase C — Fill the two real gaps

**US-005 — Activation metric (define + compute)**
- *As a founder, I need an activation number so I know whether new workspaces reach first value.*
- AC:
  - **Definition (documented in the endpoint docstring):** a workspace is *activated* when it has
    ≥1 `OrchestrationRun` with `state == "completed"`. Activation rate = activated workspaces ÷
    provisioned workspaces.
  - `GET /api/analytics/activation` returns `{ activated, total_workspaces, rate, generated_at }`.
  - Computed from `OrchestrationRun` (no new table; no fake fallback — 0 when no data).
  - Test `orchestrator/tests/test_activation_endpoint.py`: `test_activation_counts_completed_missions`,
    `test_rate_zero_when_no_workspaces`, `test_workspace_with_no_completed_run_not_activated`.
  - Type checks pass; `pytest` green.

**US-006 — Per-primitive health rollup**
- *As an operator, I need each of the 8 primitives shown green/degraded/down from existing heartbeats.*
- **Reuse-checked:** `heartbeat_results` + `GET /api/heartbeat/analytics` (`api/heartbeat.py:284`) exist.
- AC:
  - `GET /api/analytics/primitive-health` rolls the latest `heartbeat_results` findings into a
    per-primitive status (chat, memory, RAG, NL2SQL, graph, missions, playbooks, channels):
    `{ primitives: [{name, status, last_checked}], generated_at }`, `status ∈ {green, degraded, down, unknown}`.
  - Primitives with no heartbeat coverage report `unknown` (honest — not faked green).
  - No change to the heartbeat writer or schedule; read-only rollup.
  - Test `orchestrator/tests/test_primitive_health_endpoint.py`: `test_maps_findings_to_primitives`,
    `test_missing_coverage_is_unknown`, `test_latest_result_wins`.
  - Type checks pass; `pytest` green.

### Phase D — Assemble the dashboard section (capstone)

**US-007 — "Is it working?" section on the existing dashboard page**
- *As a user, I need the five metrics on one screen, on the dashboard I already use.*
- AC:
  - Add **one** section to `frontend/components/dashboard/dashboard.tsx` (or a new child component
    rendered by it) titled "Is it working?" with five tiles: Activation, Mission success rate,
    Per-primitive health, Error rate by subsystem, Widget engagement.
  - **No new route, no new page.** `grep -rn "app/.*health.*page\|new dashboard route"` style check: no
    new `page.tsx` is added under `frontend/app/`.
  - Add hooks to `frontend/hooks/use-analytics-api.ts` (or `use-unified-analytics.ts`) calling the
    four new endpoints + the existing `/dashboard/success-rate`. Reuse the existing `Card` /
    `MetricCards` / recharts patterns and the `ApiResponse<T>` envelope.
  - Tile labels use canonical nouns (**Mission**, not Workflow). Loading + empty states handled
    (no crash on zero data); no `console.log`; no hardcoded metric values in the component.
  - Type checks pass (`tsc`). Full Playwright E2E for this section is **Wave 2** (TEST-PLAN J-series),
    not here — Wave 0 verifies live in US-008.

### Gate

**US-008 — Live verification gate (operational, not code)**
- *As Gerard, I need to see five real numbers on a live workspace before Wave 0 is done.*
- AC:
  - On the deployed Railway frontend, the dashboard "Is it working?" section renders five real values
    for a workspace with real data (push → Railway → verify live URL; no localhost).
  - `grep -rn "85.0\|# Placeholder" orchestrator/core/services/analytics_engine.py` returns zero.
  - All four new endpoints return 200 with real shapes for that workspace; tenant isolation spot-checked
    (a second workspace sees its own numbers).
  - `passes` flips only after the live screen is confirmed.

---

## 5. Sequencing & dependencies

- **US-001 → US-002** (sink before aggregation).
- **US-003** independent (de-fake) — can land first; it is the cheapest honesty win.
- **US-004, US-005, US-006** independent of each other; all backend.
- **US-007** depends on US-002/003/004/005/006 (renders all five).
- **US-008** is the gate; depends on US-007 deployed.

Suggested order: US-003 (de-fake) → US-001/002 (error sink) → US-004/005/006 (the other tiles) →
US-007 (assemble) → US-008 (verify live).

---

## 6. Out of scope (explicit — do not let Ralph wander)

- **No new frontend page or route.** Extend the existing dashboard only.
- **No migrate-and-drop of `WorkflowExecution`/`Workflow`.** That is Wave 3 (`PLAYBOOK-ENGINE-DESIGN.md`
  §4.2). Wave 0 only *reads* via the existing union; it adds **no new** `WorkflowExecution` reads.
- **No Grafana board.** PRD-142 §12 mentions a Grafana tile per primitive as the *eventual* shape;
  Wave 0 ships the in-app dashboard section. Grafana wiring is later.
- **No HARNESS / learning-loop work.** `knowledge_nodes/edges` is Wave 4 (fix, not cut). The 3
  learning `COUNT(*)` tiles stay hidden/labelled until the loop is live (`KNOWLEDGE-GRAPH-CANONICAL.md` §8).
- **No new exception-handler adoption sweep.** `record_error` already exists; Wave 0 only gives it a
  sink. Mass-adopting it across handlers is out of scope (as it was in PRD-141 US-001).
- **No full Playwright net.** That is Wave 2.

---

## 7. Risk notes

- **`record_error` must never start raising.** US-001's sink is best-effort and swallows everything,
  exactly like `log_widget_event`. A telemetry write that crashes a business path is a regression.
- **`error_events` growth.** Append-only; size is bounded by error volume. A retention/rollup job is a
  *follow-up* (note it; do not build it in Wave 0). Indexes keep the window queries cheap meanwhile.
- **Touching `analytics_engine.py`.** It is read by the live dashboard. US-003 deletes placeholders but
  must keep the return shape stable for fields the frontend already consumes — verify against
  `MetricCards` before removing a key the UI reads.
