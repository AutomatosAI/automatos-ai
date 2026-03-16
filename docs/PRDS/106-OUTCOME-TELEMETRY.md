# PRD-106 — Outcome Telemetry & Learning Foundation

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P1
**Dependencies:** PRD-101 (Mission Schema), PRD-103 (Verification & Quality), PRD-105 (Budget & Governance)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-15

---

## 1. Problem Statement

### 1.1 The Gap

Automatos captures per-call LLM usage (`llm_usage` table — `core/models/core.py:138`) and stores JSONB findings in heartbeat results, but nothing correlates a multi-step mission to its aggregate cost, duration, quality, or human acceptance. The platform records individual API calls — it never answers "which model performs best for research tasks?" or "what's the average cost of a compliance mission?"

| Gap | Current State | Impact |
|-----|---------------|--------|
| No mission-level outcome record | `llm_usage` has `agent_id` but no `mission_task_id` | Cannot compute total cost/tokens/duration for a mission |
| No per-task structured outcome | `board_tasks.result` is free text | No machine-readable quality score, token spend, or retry count |
| No agent performance attribution | `llm_usage.agent_id` exists but unlinked to task outcomes | Cannot rank agents by task-type performance |
| No model comparison data | `llm_usage.model_id` exists but unlinked to quality scores | Cannot answer "Claude Opus or GPT-4 for research?" |
| No human feedback loop closure | `votes.is_upvoted` (`core/models/core.py:1157`) captures chat-level thumbs up/down | No mission/task-level acceptance signal |
| Prometheus metrics defined but unwired | 6 counters/histograms in `automatos_metrics.py:63-132` | Zero application metrics in Grafana |
| `heartbeat_results.cost` always 0.0 | Column exists but `_store_heartbeat_result()` never populates it | Heartbeat costs invisible |
| Context window telemetry ephemeral | `ContextResult.token_estimate` computed per-request | Never persisted — context optimization is blind |

### 1.2 Why This Matters Now

PRD-100 Section 3: *"No fancy learning engine — just data. Query it for patterns later."*

This PRD defines what "just data" means concretely — the schema, capture points, and query patterns that make future optimization possible without building the optimization engine now. Without structured telemetry:
- Mission Mode ships blind — no way to measure if it works
- Model routing remains manual forever — no data to automate it
- Cost optimization is guesswork — no per-task-type cost benchmarks
- Phase 3 learning foundation (bandit-style selection) has no training signal

---

## 2. Prior Art Analysis

### 2.1 Systems Studied

| System | Key Insight | What We Adopt | What We Reject |
|--------|-------------|---------------|----------------|
| **MLflow** | Three-tier storage: metrics (append-only time-series), params (immutable config), tags (mutable state) | Immutable config vs. mutable outcome separation on mission_tasks | Metric history as separate entity — too much schema for v1; use event log instead |
| **Weights & Biases** | Every run has `summary` (final/aggregate for cross-run comparison) and `history` (per-step time-series). Summary uses configurable aggregation (`min`, `max`, `mean`, `last`) | Summary columns on `mission_tasks` + event log for deep analysis = hybrid storage | W&B's custom aggregation DSL — our summary columns are explicit, not computed |
| **OpenTelemetry** | Trace/span model with `gen_ai.*` semantic conventions: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` | Attribute naming conventions for event metadata; mission=trace / task=span mental model | OTel transport layer — Postgres event log is simpler and matches our infra |
| **Honeycomb** | Schema-free columnar storage enables high-cardinality GROUP BY without pre-aggregation. No cardinality explosion because there's no pre-defined rollup | JSONB `metadata` escape hatch on mission_events for flexible querying via `jsonb_path_query` | Full columnar engine — Postgres JSONB with GIN indexes is sufficient at our scale |
| **Eppo** | Outcomes joined to treatments at analysis time via SQL, not stored as FK relations. Attribution window on metric definition, not on event | Store raw events with timestamps; join mission→outcome at query time for flexibility | Complex attribution window DSL — our missions have natural start/end boundaries |
| **Deng (Microsoft ExP)** | Minimal sufficient statistics: store `metric_sum` + `metric_sum_squares` alongside `mean` for variance computation without re-reading raw data | Aggregation views include sum and sum-of-squares for numeric metrics | Online variance algorithms — batch recomputation from `llm_usage` is sufficient |

### 2.2 Key Design Decision

**Hybrid storage (Option C from outline):** Summary columns on `mission_tasks` for dashboard queries + separate `mission_events` append-only table for detailed per-step telemetry.

Rationale:
- Summary columns enable fast `GROUP BY model_id, task_type` without JSONB path extraction
- Event log enables deep debugging, retry analysis, and future ML training
- Matches the W&B summary/history split — the most proven pattern in ML experiment tracking
- `llm_usage` remains the source of truth for per-call data — mission telemetry aggregates it, never duplicates it

---

## 3. Telemetry Schema

### 3.1 Summary Columns on `mission_tasks` (PRD-101 Extension)

These columns live on the `mission_tasks` table defined in PRD-101. They are the "W&B summary" — final/aggregate values for fast dashboard queries.

```sql
-- Migration: add telemetry summary columns to mission_tasks
ALTER TABLE mission_tasks
  ADD COLUMN model_id        VARCHAR(255),     -- e.g. 'anthropic/claude-sonnet-4-20250514'
  ADD COLUMN task_type        VARCHAR(50),      -- e.g. 'research', 'writing', 'coding', 'review'
  ADD COLUMN tokens_in        INTEGER DEFAULT 0,
  ADD COLUMN tokens_out       INTEGER DEFAULT 0,
  ADD COLUMN cost_usd         NUMERIC(12,8) DEFAULT 0,
  ADD COLUMN duration_ms      INTEGER,          -- completed_at - started_at
  ADD COLUMN verifier_score   REAL,             -- 0.0-1.0 from PRD-103, NULL until verified
  ADD COLUMN human_accepted   BOOLEAN,          -- NULL until reviewed
  ADD COLUMN error_type       VARCHAR(100),     -- NULL on success; structured enum
  ADD COLUMN retry_count      SMALLINT DEFAULT 0,
  ADD COLUMN context_tokens   INTEGER,          -- context window consumed
  ADD COLUMN tools_used       TEXT[];            -- accumulated tool names

-- Indexes for analytical queries (Section 5 query patterns)
CREATE INDEX idx_mt_model_type ON mission_tasks(model_id, task_type);
CREATE INDEX idx_mt_agent ON mission_tasks(agent_id);
CREATE INDEX idx_mt_verifier ON mission_tasks(verifier_score) WHERE verifier_score IS NOT NULL;
```

Field sourcing:

| Field | Source | When Set |
|-------|--------|----------|
| `model_id` | `LLMManager.generate_response()` → first call determines primary model | Task assignment |
| `task_type` | Coordinator decomposition (PRD-102) | Task creation |
| `tokens_in/out` | Running total from `UsageTracker.track()` | Incremented per LLM call |
| `cost_usd` | Running total from `UsageTracker.track()` | Incremented per LLM call |
| `duration_ms` | `completed_at - started_at` | Task completion |
| `verifier_score` | `VerificationService.verify()` (PRD-103) | Post-execution verification |
| `human_accepted` | Human review API endpoint | Post-review |
| `error_type` | Exception handler in `AgentFactory.execute_with_prompt()` | On failure |
| `retry_count` | `AgentFactory` tool loop (`core/agents/factory.py`) | Incremented per retry |
| `context_tokens` | `ContextService.get_context()` return value | Task execution start |
| `tools_used` | `_execute_tool_calls()` in `AgentFactory` | Appended per tool call |

### 3.2 Mission Run Summary Columns (PRD-101 Extension)

These columns live on the `mission_runs` table. They are aggregate-of-aggregates — computed from `mission_tasks` rows.

```sql
ALTER TABLE mission_runs
  ADD COLUMN total_tasks         INTEGER DEFAULT 0,
  ADD COLUMN tasks_passed        INTEGER DEFAULT 0,
  ADD COLUMN tasks_failed        INTEGER DEFAULT 0,
  ADD COLUMN total_tokens        INTEGER DEFAULT 0,
  ADD COLUMN total_cost_usd      NUMERIC(12,8) DEFAULT 0,
  ADD COLUMN verification_cost   NUMERIC(12,8) DEFAULT 0,
  ADD COLUMN coordination_cost   NUMERIC(12,8) DEFAULT 0,
  ADD COLUMN human_verdict       VARCHAR(20),  -- accepted, rejected, partial, pending
  ADD COLUMN converted_to_routine BOOLEAN DEFAULT FALSE;
```

These are maintained via two mechanisms:
1. **Real-time increment:** When `UsageTracker.track()` fires with a `mission_task_id`, it also increments the parent `mission_runs` totals
2. **Batch reconciliation:** A daily job recomputes all mission_runs summaries from `llm_usage WHERE mission_task_id IN (SELECT id FROM mission_tasks WHERE mission_run_id = ?)` — catches any drift from failed increments

### 3.3 Event Log Table: `mission_events`

The "W&B history" — append-only, per-step telemetry for debugging and future ML training.

```sql
CREATE TABLE mission_events (
  id              BIGSERIAL PRIMARY KEY,
  mission_run_id  BIGINT NOT NULL REFERENCES mission_runs(id) ON DELETE CASCADE,
  mission_task_id BIGINT REFERENCES mission_tasks(id) ON DELETE CASCADE,  -- NULL for run-level events
  event_type      VARCHAR(50) NOT NULL,

  -- OTel-inspired attributes (gen_ai.* naming convention)
  attributes      JSONB NOT NULL DEFAULT '{}',

  -- Common fields extracted for fast filtering (avoid JSONB path queries)
  agent_id        INTEGER,
  model_id        VARCHAR(255),

  -- Numeric payload (for aggregation queries)
  tokens_in       INTEGER,
  tokens_out      INTEGER,
  cost_usd        NUMERIC(12,8),
  duration_ms     INTEGER,

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partitioned by month for retention management
-- (Postgres 12+ declarative partitioning)
-- CREATE TABLE mission_events (...) PARTITION BY RANGE (created_at);
-- CREATE TABLE mission_events_2026_03 PARTITION OF mission_events
--   FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

-- Indexes
CREATE INDEX idx_me_run       ON mission_events(mission_run_id);
CREATE INDEX idx_me_task      ON mission_events(mission_task_id) WHERE mission_task_id IS NOT NULL;
CREATE INDEX idx_me_type      ON mission_events(event_type);
CREATE INDEX idx_me_created   ON mission_events USING BRIN(created_at);
CREATE INDEX idx_me_attrs     ON mission_events USING GIN(attributes jsonb_path_ops);
```

### 3.4 Event Types

| event_type | Level | Emitted By | Key Attributes |
|------------|-------|------------|----------------|
| `mission.created` | Run | Coordinator | `{goal, plan_hash, config}` |
| `mission.plan_generated` | Run | Coordinator | `{task_count, estimated_cost, plan_version}` |
| `mission.replanned` | Run | Coordinator | `{reason, tasks_added, tasks_removed, plan_version}` |
| `mission.completed` | Run | Coordinator | `{status, total_cost, total_duration_ms}` |
| `task.assigned` | Task | Coordinator | `{agent_id, model_id, task_type, priority}` |
| `task.started` | Task | AgentFactory | `{context_tokens, tools_available}` |
| `task.llm_call` | Task | UsageTracker | `{model_id, tokens_in, tokens_out, cost_usd, latency_ms}` |
| `task.tool_called` | Task | UnifiedExecutor | `{tool_name, duration_ms, status, cache_hit}` |
| `task.retry` | Task | AgentFactory | `{attempt, reason, error_type}` |
| `task.verified` | Task | VerificationService | `{score, verdict, verifier_model, verifier_cost}` |
| `task.completed` | Task | AgentFactory | `{status, tokens_total, cost_total, duration_ms}` |
| `task.failed` | Task | AgentFactory | `{error_type, error_message_truncated, attempt}` |
| `budget.warning` | Run | BudgetManager | `{threshold_pct, spent, limit}` |
| `budget.throttled` | Run | BudgetManager | `{spent, limit, action}` |
| `human.reviewed` | Task | API endpoint | `{accepted, feedback_text_length}` |
| `human.verdict` | Run | API endpoint | `{verdict, tasks_accepted, tasks_rejected}` |

### 3.5 `llm_usage` Extension

Add a nullable FK to link per-call data to mission tasks:

```sql
-- Migration: link llm_usage to mission tasks
ALTER TABLE llm_usage
  ADD COLUMN mission_task_id BIGINT REFERENCES mission_tasks(id) ON DELETE SET NULL;

CREATE INDEX idx_lu_mission_task ON llm_usage(mission_task_id)
  WHERE mission_task_id IS NOT NULL;
```

**Critical constraints:**
- **Nullable:** Thousands of existing rows (chatbot, heartbeat, routing, embedding calls) have no mission context. Non-nullable would require impossible backfill.
- **No backfill:** Historical `llm_usage` rows predate missions. Backfilling would create false attributions. Only new LLM calls made within mission execution get the FK set.
- **`ON DELETE SET NULL`:** If a mission_task is deleted (cascaded from mission_run deletion), the `llm_usage` rows survive for historical billing queries.

SQLAlchemy model change (`core/models/core.py:138`):

```python
class LLMUsage(Base):
    __tablename__ = 'llm_usage'
    # ... existing columns unchanged ...

    # PRD-106: Link to mission task for outcome telemetry
    mission_task_id = Column(BigInteger, ForeignKey('mission_tasks.id', ondelete='SET NULL'), nullable=True)

    # PRD-106: Extend request_type enum
    # request_type values: chat, agent, recipe, routing, embedding, coordinator, verifier
```

### 3.6 `request_type` Extension

The existing `request_type` column on `llm_usage` (`core/models/core.py:151`) is a `VARCHAR(50)` — not a database enum, so no migration needed to add values. The application code must use two new values:

| New Value | Used By | Purpose |
|-----------|---------|---------|
| `coordinator` | `CoordinatorService` (PRD-102) planning and monitoring calls | Separate coordination overhead from task execution cost |
| `verifier` | `VerificationService` (PRD-103) scoring calls | Separate verification cost from task execution cost |

This enables computing `coordination_cost_usd` and `verification_cost_usd` as separate line items on `mission_runs` via:

```sql
SELECT
  SUM(CASE WHEN request_type = 'coordinator' THEN total_cost ELSE 0 END) AS coordination_cost,
  SUM(CASE WHEN request_type = 'verifier' THEN total_cost ELSE 0 END) AS verification_cost,
  SUM(CASE WHEN request_type NOT IN ('coordinator', 'verifier') THEN total_cost ELSE 0 END) AS execution_cost
FROM llm_usage
WHERE mission_task_id IN (SELECT id FROM mission_tasks WHERE mission_run_id = :run_id);
```

---

## 4. Data Privacy Policy

**Rule: Never store prompt content or model outputs in telemetry.**

Telemetry captures metadata only. Content belongs in agent reports (PRD-76) and workspace files.

| Stored | NOT Stored |
|--------|-----------|
| Model ID, provider | Prompt text |
| Token counts (in/out) | Model output text |
| Cost in USD | User messages |
| Duration in ms | File contents |
| Tool names | Tool parameter values |
| Error type enum | Full error stack traces |
| Error message (truncated to 500 chars) | PII or IP |
| Verifier score (0.0-1.0) | Verifier reasoning text |
| Success criteria text (from mission definition) | Task output content |

**JSONB `attributes` on `mission_events`:** Must not contain prompt content, model output, or PII. Allowed keys are defined per `event_type` in Section 3.4. The `TelemetryService` validates keys against an allowlist before insertion.

---

## 5. Query Patterns

The schema is designed to answer specific questions. Every query below has been validated against the DDL in Section 3.

### 5.1 Operational (Day 1)

```sql
-- Q1: What did mission X cost?
SELECT total_cost_usd, total_tokens,
       coordination_cost, verification_cost
FROM mission_runs WHERE id = :mission_id;

-- Q2: Which tasks failed in mission X?
SELECT id, title, error_type, retry_count, cost_usd
FROM mission_tasks
WHERE mission_run_id = :mission_id AND status = 'failed';

-- Q3: Cost breakdown by task
SELECT id, title, task_type, model_id,
       tokens_in, tokens_out, cost_usd, duration_ms
FROM mission_tasks
WHERE mission_run_id = :mission_id
ORDER BY cost_usd DESC;

-- Q4: How long did mission X take?
SELECT EXTRACT(EPOCH FROM (completed_at - started_at)) * 1000 AS duration_ms
FROM mission_runs WHERE id = :mission_id;
```

### 5.2 Analytical (Week 1+)

```sql
-- Q5: Best model for research tasks? (cost-quality Pareto)
SELECT model_id,
       COUNT(*) AS task_count,
       AVG(verifier_score) AS avg_quality,
       AVG(cost_usd) AS avg_cost,
       AVG(duration_ms) AS avg_duration
FROM mission_tasks
WHERE task_type = 'research'
  AND verifier_score IS NOT NULL
GROUP BY model_id
ORDER BY avg_quality DESC;

-- Q6: Average cost per mission type?
SELECT config->>'mission_type' AS mission_type,
       COUNT(*) AS mission_count,
       AVG(total_cost_usd) AS avg_cost,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_cost_usd) AS median_cost
FROM mission_runs
WHERE status = 'completed'
GROUP BY config->>'mission_type';

-- Q7: Agent acceptance rate
SELECT mt.agent_id, a.name,
       COUNT(*) AS tasks_total,
       AVG(mt.human_accepted::int) AS acceptance_rate,
       AVG(mt.verifier_score) AS avg_quality
FROM mission_tasks mt
JOIN agents a ON a.id = mt.agent_id
WHERE mt.human_accepted IS NOT NULL
GROUP BY mt.agent_id, a.name
ORDER BY acceptance_rate DESC;

-- Q8: Failure patterns by tool
SELECT unnest(tools_used) AS tool_name,
       error_type,
       COUNT(*) AS failure_count
FROM mission_tasks
WHERE error_type IS NOT NULL
GROUP BY tool_name, error_type
ORDER BY failure_count DESC;

-- Q9: Verification cost as percentage of task cost
SELECT task_type,
       AVG(cost_usd) AS avg_task_cost,
       AVG(verifier_cost_from_events) AS avg_verify_cost,
       AVG(verifier_cost_from_events / NULLIF(cost_usd, 0)) * 100 AS verify_pct
FROM (
  SELECT mt.task_type, mt.cost_usd,
    (SELECT SUM(me.cost_usd) FROM mission_events me
     WHERE me.mission_task_id = mt.id AND me.event_type = 'task.verified') AS verifier_cost_from_events
  FROM mission_tasks mt
  WHERE mt.status = 'completed'
) sub
GROUP BY task_type;
```

### 5.3 Strategic (Month 1+)

```sql
-- Q10: Cost trend over time
SELECT DATE_TRUNC('week', created_at) AS week,
       COUNT(*) AS missions,
       SUM(total_cost_usd) AS total_cost,
       AVG(total_cost_usd) AS avg_cost
FROM mission_runs
WHERE status = 'completed'
GROUP BY week ORDER BY week;

-- Q11: Model cost-quality Pareto frontier
SELECT model_id,
       COUNT(*) AS n,
       AVG(verifier_score) AS avg_quality,
       AVG(cost_usd) AS avg_cost,
       STDDEV(verifier_score) AS quality_stddev,
       SUM(cost_usd) AS total_sum,
       SUM(cost_usd * cost_usd) AS total_sum_sq  -- sufficient statistics
FROM mission_tasks
WHERE verifier_score IS NOT NULL
GROUP BY model_id
HAVING COUNT(*) >= 10;  -- minimum sample size

-- Q12: Retry rate by model (identifies unreliable models)
SELECT model_id,
       COUNT(*) AS tasks,
       AVG(retry_count) AS avg_retries,
       SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS retry_rate
FROM mission_tasks
GROUP BY model_id
HAVING COUNT(*) >= 10
ORDER BY retry_rate DESC;

-- Q13: Which task types should become routines?
SELECT config->>'mission_type' AS mission_type,
       COUNT(*) AS total_runs,
       SUM(converted_to_routine::int) AS converted,
       AVG(total_cost_usd) AS avg_cost
FROM mission_runs
WHERE status = 'completed'
GROUP BY config->>'mission_type'
ORDER BY converted DESC;
```

### 5.4 Materialized Views for Dashboards

For queries that scan large tables (Q10-Q13), create materialized views refreshed by a daily batch job:

```sql
-- Refreshed daily by task_reconciler cron
CREATE MATERIALIZED VIEW mv_model_performance AS
SELECT model_id,
       task_type,
       COUNT(*) AS n,
       AVG(verifier_score) AS avg_quality,
       STDDEV(verifier_score) AS stddev_quality,
       AVG(cost_usd) AS avg_cost,
       AVG(duration_ms) AS avg_duration,
       AVG(retry_count) AS avg_retries,
       SUM(cost_usd) AS sum_cost,
       SUM(cost_usd * cost_usd) AS sum_sq_cost  -- sufficient statistics for variance
FROM mission_tasks
WHERE verifier_score IS NOT NULL
  AND status IN ('completed', 'failed')
GROUP BY model_id, task_type;

CREATE UNIQUE INDEX idx_mv_mp ON mv_model_performance(model_id, task_type);

-- Refresh command (run in daily reconciliation job)
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_model_performance;
```

---

## 6. Capture Points & Data Flow

### 6.1 Sequence Diagram: Telemetry Capture During Task Execution

```
Coordinator          AgentFactory          UsageTracker         mission_events
    │                     │                     │                     │
    │ assign_task(task)    │                     │                     │
    │────────────────────>│                     │                     │
    │                     │                     │              emit(task.assigned)
    │                     │                     │<────────────────────│
    │                     │                     │                     │
    │                     │ execute_with_prompt()│                     │
    │                     │─────────────────────>│                     │
    │                     │                     │              emit(task.started)
    │                     │                     │                     │
    │                     │   [tool loop iter]   │                     │
    │                     │   LLM call ──────────>  track(             │
    │                     │                     │    mission_task_id,  │
    │                     │                     │    tokens, cost)     │
    │                     │                     │   ┌─────────────┐    │
    │                     │                     │   │ INSERT       │    │
    │                     │                     │   │ llm_usage    │    │
    │                     │                     │   │ INCREMENT    │    │
    │                     │                     │   │ mission_tasks│    │
    │                     │                     │   │ summary cols │    │
    │                     │                     │   └─────────────┘    │
    │                     │                     │              emit(task.llm_call)
    │                     │                     │                     │
    │                     │   tool_call ─────────────────────> emit(task.tool_called)
    │                     │                     │                     │
    │                     │   [completion]       │                     │
    │                     │────────────────────────────────── emit(task.completed)
    │                     │                     │                     │
    │ verify(task) ───────────────────────────────────────── emit(task.verified)
    │                     │                     │                     │
```

### 6.2 Code Changes Required

| File | Current | Change | Lines |
|------|---------|--------|-------|
| `core/llm/usage_tracker.py:21` | `track()` has no mission context | Add `mission_task_id: Optional[int] = None` parameter. When set: (1) pass to `LLMUsage` row, (2) increment `mission_tasks.tokens_in/out/cost_usd` via `UPDATE ... SET tokens_in = tokens_in + :delta` | 21-80 |
| `core/llm/manager.py` | Calls `_track_usage()` | Pass `mission_task_id` through `generate_response()` → `_track_usage()` call chain. Source: `execution_context` dict already threaded through | 643-671 |
| `core/agents/factory.py` | Tool loop tracks retries | (1) Set `mission_task_id` on execution context, (2) append to `tools_used` array, (3) increment `retry_count`, (4) emit `task.started/completed/failed` events | Tool loop |
| `core/monitoring/automatos_metrics.py:75-132` | Prometheus counters defined, never incremented | Wire into `UsageTracker.track()`: `AGENT_TOKEN_USAGE.labels(agent_id, model, 'input').inc(tokens_in)`, `LLM_REQUEST_DURATION.labels(model, provider).observe(latency_ms/1000)` | 75-132 |
| `services/heartbeat_service.py:913-942` | `_store_heartbeat_result()` inserts with `cost=0.0` | After insert, query `SUM(total_cost) FROM llm_usage WHERE execution_id = :eid` and update `heartbeat_results.cost` | 913-942 |
| `api/llm_analytics.py` | Usage/cost endpoints only | Add `/api/missions/{id}/telemetry` and `/api/analytics/mission-outcomes` endpoints | New |

### 6.3 UsageTracker Extension

```python
# core/llm/usage_tracker.py — extended track() signature

class UsageTracker:
    @staticmethod
    def track(
        workspace_id: UUID,
        model_id: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: Optional[int] = None,
        execution_id: Optional[str] = None,
        request_type: str = "chat",
        latency_ms: Optional[int] = None,
        status: str = "success",
        is_byok: bool = False,
        error_message: Optional[str] = None,
        tier: str = "direct",
        mission_task_id: Optional[int] = None,  # PRD-106
    ) -> None:
        try:
            from core.database.database import SessionLocal
            from core.models.core import LLMUsage, LLMModel

            db = SessionLocal()
            try:
                # ... existing cost calculation unchanged ...

                row = LLMUsage(
                    # ... existing fields unchanged ...
                    mission_task_id=mission_task_id,  # PRD-106
                )
                db.add(row)

                # PRD-106: Increment mission_task summary columns
                if mission_task_id is not None:
                    db.execute(
                        text("""
                            UPDATE mission_tasks
                            SET tokens_in = tokens_in + :tin,
                                tokens_out = tokens_out + :tout,
                                cost_usd = cost_usd + :cost
                            WHERE id = :task_id
                        """),
                        {"tin": input_tokens, "tout": output_tokens,
                         "cost": total_cost, "task_id": mission_task_id}
                    )

                # PRD-106: Wire Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    from core.monitoring.automatos_metrics import (
                        AGENT_TOKEN_USAGE, LLM_REQUEST_DURATION, LLM_TOKEN_USAGE
                    )
                    LLM_TOKEN_USAGE.labels(
                        model=model_id, provider=provider, direction="input"
                    ).inc(input_tokens)
                    LLM_TOKEN_USAGE.labels(
                        model=model_id, provider=provider, direction="output"
                    ).inc(output_tokens)
                    if latency_ms:
                        LLM_REQUEST_DURATION.labels(
                            model=model_id, provider=provider
                        ).observe(latency_ms / 1000)
                    if agent_id:
                        AGENT_TOKEN_USAGE.labels(
                            agent_id=str(agent_id), model=model_id, direction="input"
                        ).inc(input_tokens)
                        AGENT_TOKEN_USAGE.labels(
                            agent_id=str(agent_id), model=model_id, direction="output"
                        ).inc(output_tokens)

                # ... existing agent stats update + commit unchanged ...
```

### 6.4 TelemetryService

New service — single write path for all `mission_events` rows:

```python
# services/telemetry_service.py

from typing import Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class TelemetryEvent:
    """Immutable telemetry event — validated before insertion."""
    mission_run_id: int
    event_type: str
    mission_task_id: Optional[int] = None
    agent_id: Optional[int] = None
    model_id: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None
    duration_ms: Optional[int] = None
    attributes: dict = None

    def __post_init__(self):
        if self.attributes is None:
            object.__setattr__(self, 'attributes', {})

# Allowlist per event_type — keys not in allowlist are stripped
EVENT_ATTRIBUTE_ALLOWLIST = {
    "mission.created": {"goal", "plan_hash", "config"},
    "mission.plan_generated": {"task_count", "estimated_cost", "plan_version"},
    "mission.replanned": {"reason", "tasks_added", "tasks_removed", "plan_version"},
    "mission.completed": {"status", "total_cost", "total_duration_ms"},
    "task.assigned": {"agent_id", "model_id", "task_type", "priority"},
    "task.started": {"context_tokens", "tools_available"},
    "task.llm_call": {"model_id", "tokens_in", "tokens_out", "cost_usd", "latency_ms"},
    "task.tool_called": {"tool_name", "duration_ms", "status", "cache_hit"},
    "task.retry": {"attempt", "reason", "error_type"},
    "task.verified": {"score", "verdict", "verifier_model", "verifier_cost"},
    "task.completed": {"status", "tokens_total", "cost_total", "duration_ms"},
    "task.failed": {"error_type", "error_message_truncated", "attempt"},
    "budget.warning": {"threshold_pct", "spent", "limit"},
    "budget.throttled": {"spent", "limit", "action"},
    "human.reviewed": {"accepted", "feedback_text_length"},
    "human.verdict": {"verdict", "tasks_accepted", "tasks_rejected"},
}


class TelemetryService:
    """Append-only mission event log.

    Single write path for all mission_events.
    Validates attributes against allowlist to prevent PII leakage.
    """

    @staticmethod
    def emit(event: TelemetryEvent) -> None:
        """Insert a validated event into mission_events."""
        from core.database.database import SessionLocal

        # Validate and filter attributes
        allowed = EVENT_ATTRIBUTE_ALLOWLIST.get(event.event_type, set())
        clean_attrs = {k: v for k, v in event.attributes.items() if k in allowed}

        db = SessionLocal()
        try:
            db.execute(
                text("""
                    INSERT INTO mission_events
                        (mission_run_id, mission_task_id, event_type,
                         agent_id, model_id, tokens_in, tokens_out,
                         cost_usd, duration_ms, attributes)
                    VALUES (:run_id, :task_id, :etype,
                            :agent, :model, :tin, :tout,
                            :cost, :dur, :attrs::jsonb)
                """),
                {
                    "run_id": event.mission_run_id,
                    "task_id": event.mission_task_id,
                    "etype": event.event_type,
                    "agent": event.agent_id,
                    "model": event.model_id,
                    "tin": event.tokens_in,
                    "tout": event.tokens_out,
                    "cost": event.cost_usd,
                    "dur": event.duration_ms,
                    "attrs": json.dumps(clean_attrs),
                }
            )
            db.commit()
        except Exception as e:
            logger.warning(f"Telemetry emit failed (non-fatal): {e}")
            db.rollback()
        finally:
            db.close()
```

---

## 7. Retention Policy

### 7.1 Three-Tier Retention

| Tier | Data | Retention | Rationale |
|------|------|-----------|-----------|
| **Permanent** | `mission_runs` summary columns, `mission_tasks` summary columns | Forever | Tiny rows (~500 bytes each). Essential for long-term analytics. Storage cost negligible |
| **Hot** | `mission_events` rows | 90 days in Postgres | Event log rows are larger (~200 bytes each). 1000 events/day × 90 days = ~18MB — manageable but grows linearly with mission volume |
| **Cold archive** | `mission_events` rows older than 90 days | S3 archive (`s3://automatos-ai/telemetry/archive/`) | JSONL export before partition drop. Available for future ML training or forensic analysis |

### 7.2 Retention Automation

```python
# Run as part of daily reconciliation in task_reconciler.py

async def archive_old_events(db: Session, s3_client, days: int = 90):
    """Archive mission_events older than retention window to S3, then delete."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    # 1. Export to JSONL
    rows = db.execute(
        text("SELECT * FROM mission_events WHERE created_at < :cutoff ORDER BY id"),
        {"cutoff": cutoff}
    ).fetchall()

    if not rows:
        return 0

    # 2. Write JSONL to S3
    archive_key = f"telemetry/archive/{cutoff.strftime('%Y/%m')}/events.jsonl"
    # ... S3 upload logic ...

    # 3. Delete archived rows
    db.execute(
        text("DELETE FROM mission_events WHERE created_at < :cutoff"),
        {"cutoff": cutoff}
    )
    db.commit()

    return len(rows)
```

---

## 8. Prometheus Metrics Wiring

### 8.1 Current State

Six counters/histograms are defined in `core/monitoring/automatos_metrics.py:63-132` but **never incremented anywhere in the codebase**:

| Metric | Defined At | Status |
|--------|-----------|--------|
| `automatos_agent_heartbeat_total` | Line 63 | Unwired |
| `automatos_agent_heartbeat_duration_seconds` | Line 69 | Unwired |
| `automatos_agent_token_usage_total` | Line 75 | Unwired |
| `automatos_active_agents` | Line 81 | Unwired |
| `automatos_llm_request_duration_seconds` | Line 121 | Unwired |
| `automatos_llm_tokens_total` | Line 128 | Unwired |

### 8.2 Wiring Plan

| Metric | Wire Into | How |
|--------|-----------|-----|
| `agent_heartbeat_total` | `heartbeat_service.py` after each tick | `.labels(agent_id=str(id), status=status).inc()` |
| `agent_heartbeat_duration_seconds` | `heartbeat_service.py` around tick execution | `.labels(agent_id=str(id)).observe(duration)` |
| `agent_token_usage_total` | `usage_tracker.py:track()` (see Section 6.3) | `.labels(agent_id, model, direction).inc(tokens)` |
| `active_agents` | `heartbeat_service.py` at service start | `.set(count_of_active_agents)` |
| `llm_request_duration_seconds` | `usage_tracker.py:track()` (see Section 6.3) | `.labels(model, provider).observe(latency/1000)` |
| `llm_tokens_total` | `usage_tracker.py:track()` (see Section 6.3) | `.labels(model, provider, direction).inc(tokens)` |

### 8.3 New Mission-Specific Metrics

```python
# Added to automatos_metrics.py
if PROMETHEUS_AVAILABLE:
    MISSION_TASKS_TOTAL = Counter(
        "automatos_mission_tasks_total",
        "Total mission tasks by status",
        ["task_type", "status"],
    )

    MISSION_COST_USD = Counter(
        "automatos_mission_cost_usd_total",
        "Cumulative mission cost in USD",
        ["model", "request_type"],
    )

    MISSION_DURATION = Histogram(
        "automatos_mission_duration_seconds",
        "Mission total duration",
        buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
    )
```

---

## 9. Heartbeat Cost Fix

### 9.1 Current Bug

`heartbeat_service.py:913-942` inserts `heartbeat_results` rows but never populates the `cost` column — it's always `0.0`.

The `cost` column exists in the raw SQL INSERT:
```sql
INSERT INTO heartbeat_results
    (source_type, source_id, workspace_id, status,
     findings, actions_taken, tokens_used, created_at)
```

Note: `cost` is not even in the INSERT column list — it defaults to whatever the column default is (likely 0.0 or NULL).

### 9.2 Fix

After the heartbeat LLM call completes, query the `llm_usage` table for the cost of the call just made (using `execution_id` correlation), then include `cost` in the INSERT:

```python
# In heartbeat_service.py, after LLM call:
heartbeat_cost = db.execute(
    text("SELECT SUM(total_cost) FROM llm_usage WHERE execution_id = :eid"),
    {"eid": execution_id}
).scalar() or 0.0

# Include in INSERT:
# ... cost = :hb_cost ...
```

---

## 10. API Endpoints

### 10.1 Mission Telemetry

```
GET /api/missions/{mission_id}/telemetry
```

Response:

```json
{
  "mission_id": 42,
  "status": "completed",
  "summary": {
    "total_tasks": 5,
    "tasks_passed": 4,
    "tasks_failed": 1,
    "total_tokens": 48230,
    "total_cost_usd": 0.1847,
    "coordination_cost_usd": 0.0123,
    "verification_cost_usd": 0.0089,
    "execution_cost_usd": 0.1635,
    "duration_ms": 127400,
    "human_verdict": "accepted"
  },
  "tasks": [
    {
      "task_id": 101,
      "title": "Research EU AI Act",
      "task_type": "research",
      "agent_id": 12,
      "model_id": "anthropic/claude-sonnet-4-20250514",
      "tokens_in": 12400,
      "tokens_out": 3200,
      "cost_usd": 0.0456,
      "duration_ms": 34200,
      "verifier_score": 0.87,
      "human_accepted": true,
      "retry_count": 0,
      "tools_used": ["workspace_read_file", "platform_search_memory"]
    }
  ]
}
```

### 10.2 Mission Outcome Analytics

```
GET /api/analytics/mission-outcomes?period=30d&group_by=model
```

Response:

```json
{
  "period": "30d",
  "group_by": "model",
  "groups": [
    {
      "key": "anthropic/claude-sonnet-4-20250514",
      "task_count": 142,
      "avg_quality": 0.82,
      "avg_cost": 0.034,
      "avg_duration_ms": 28400,
      "retry_rate": 0.07,
      "acceptance_rate": 0.91
    }
  ]
}
```

### 10.3 Human Review Endpoints

```
POST /api/missions/{mission_id}/tasks/{task_id}/review
{
  "accepted": true,
  "feedback": "Good research but missed section 3.2 of the regulation"
}
```

```
POST /api/missions/{mission_id}/verdict
{
  "verdict": "accepted",  // accepted | rejected | partial
  "save_as_routine": true
}
```

Both endpoints emit `human.reviewed` / `human.verdict` telemetry events.

---

## 11. Cross-PRD Integration

### 11.1 Dependencies

| PRD | Integration Point | Direction |
|-----|-------------------|-----------|
| **PRD-101** | `mission_runs` and `mission_tasks` tables receive summary columns | PRD-106 extends PRD-101 schema |
| **PRD-102** | Coordinator emits `mission.created/plan_generated/replanned` events, uses `request_type='coordinator'` | PRD-102 writes → PRD-106 captures |
| **PRD-103** | VerificationService sets `verifier_score`, emits `task.verified`, uses `request_type='verifier'` | PRD-103 writes → PRD-106 captures |
| **PRD-104** | Ephemeral contractors must log `agent_id` before cleanup. Contractor LLM calls must include `mission_task_id` | PRD-104 must preserve telemetry before destroying contractor |
| **PRD-105** | Budget enforcement uses real-time `cost_usd` from same increment path. `budget.warning/throttled` events | Shared write path — PRD-105 reads what PRD-106 writes |
| **PRD-107** | Context interface exposes `context_tokens` and `sections_trimmed` for telemetry capture | PRD-107 provides → PRD-106 persists |

### 11.2 Telemetry for Ephemeral Contractors (PRD-104 Constraint)

Ephemeral contractors are destroyed after task completion. The telemetry must be captured before cleanup:

1. All LLM calls during contractor execution include `mission_task_id` → `llm_usage` rows survive contractor deletion
2. `mission_tasks.agent_id` is set at assignment time → survives contractor deletion (FK is nullable, not cascading)
3. `mission_events` reference `agent_id` as an integer field, not a FK → no referential integrity issue when contractor is cleaned up

### 11.3 Propensity Logging (Future-Proofing)

The schema supports future `action_probability` without migration:

```sql
-- When coordinator selects a model, log the probability in mission_events attributes:
-- event_type = 'task.assigned'
-- attributes = {"agent_id": 12, "model_id": "claude-sonnet", "task_type": "research",
--               "action_probability": 0.6, "candidates_considered": 3}
```

The `attributes` JSONB field accepts any key in the `task.assigned` allowlist. Adding `action_probability` to the allowlist is a one-line code change, not a migration.

---

## 12. Batch Reconciliation

### 12.1 Purpose

Real-time increments in `UsageTracker.track()` can drift from truth if:
- A DB session fails after INSERT but before COMMIT
- The `UPDATE mission_tasks SET tokens_in = tokens_in + :delta` races with a concurrent update
- A retry loop double-counts

### 12.2 Reconciliation Job

Run daily as part of `task_reconciler.py`:

```python
async def reconcile_mission_telemetry(db: Session):
    """Recompute mission_tasks summary from llm_usage source of truth."""

    # Recompute per-task summaries
    db.execute(text("""
        UPDATE mission_tasks mt SET
            tokens_in = COALESCE(agg.tin, 0),
            tokens_out = COALESCE(agg.tout, 0),
            cost_usd = COALESCE(agg.cost, 0)
        FROM (
            SELECT mission_task_id,
                   SUM(input_tokens) AS tin,
                   SUM(output_tokens) AS tout,
                   SUM(total_cost) AS cost
            FROM llm_usage
            WHERE mission_task_id IS NOT NULL
            GROUP BY mission_task_id
        ) agg
        WHERE mt.id = agg.mission_task_id
          AND (mt.tokens_in != COALESCE(agg.tin, 0)
               OR mt.tokens_out != COALESCE(agg.tout, 0)
               OR mt.cost_usd != COALESCE(agg.cost, 0))
    """))

    # Recompute per-run summaries
    db.execute(text("""
        UPDATE mission_runs mr SET
            total_tasks = agg.total,
            tasks_passed = agg.passed,
            tasks_failed = agg.failed,
            total_tokens = agg.tokens,
            total_cost_usd = agg.cost
        FROM (
            SELECT mission_run_id,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE verifier_score >= 0.7) AS passed,
                   COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                   SUM(tokens_in + tokens_out) AS tokens,
                   SUM(cost_usd) AS cost
            FROM mission_tasks
            GROUP BY mission_run_id
        ) agg
        WHERE mr.id = agg.mission_run_id
    """))

    # Refresh materialized views
    db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_model_performance"))

    db.commit()
```

---

## 13. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Event log volume overwhelms Postgres | High | Medium | Monthly partitioning + 90-day retention + S3 archive. At 1000 events/day, 90 days = 90K rows (~18MB) — well within Postgres comfort zone |
| 2 | Telemetry write overhead slows LLM call path | High | Low | `UsageTracker` already writes in a separate DB session. `mission_task_id` is one extra field on existing INSERT + one UPDATE. No new network call |
| 3 | Schema stores too little — can't answer questions not yet thought of | Medium | Medium | Hybrid approach (summary + events + JSONB attributes) hedges this. `attributes` is the escape hatch for ad-hoc data |
| 4 | GDPR/privacy concerns with outcome data | Medium | Low | Strict "no content" policy (Section 4). Telemetry is metadata only. No prompt text, no model output, no PII |
| 5 | Premature optimization — building learning engine too early | Medium | High | PRD-100 explicitly forbids this. PRD-106 captures data and defines queries. It does NOT build recommendation or bandit algorithms |
| 6 | `llm_usage` table is write-hot — adding FK + index may slow writes | Medium | Medium | Nullable FK, index only on non-null values (`WHERE mission_task_id IS NOT NULL`). Existing queries are unaffected — they never filter by `mission_task_id` |
| 7 | Real-time increment drifts from raw data | Medium | Medium | Daily batch reconciliation (Section 12) recomputes from `llm_usage` source of truth |
| 8 | Prometheus cardinality explosion from `agent_id` × `model` labels | Medium | Low | Agent label uses string ID (bounded by roster size ~20). Model label uses `model_id` (bounded by installed models ~10). Max cardinality: 200 series |

---

## 14. Acceptance Criteria

### Must Have

- [x] `mission_events` table DDL with all fields, types, constraints, and indices (Section 3.3)
- [x] `mission_task_id` nullable FK added to `llm_usage` with migration DDL (Section 3.5)
- [x] Per-task summary fields on `mission_tasks` with migration DDL (Section 3.1)
- [x] Per-run summary fields on `mission_runs` with migration DDL (Section 3.2)
- [x] 16 event types defined with attribute schemas (Section 3.4)
- [x] `request_type` extended with `coordinator`, `verifier` values (Section 3.6)
- [x] All 13 query patterns shown as working SQL against the schema (Section 5)
- [x] Capture points documented with sequence diagram (Section 6)
- [x] Retention policy: 90-day events, permanent summaries, S3 archive (Section 7)
- [x] Privacy policy: explicit no-content rule with allowlist (Section 4)

### Should Have

- [x] `TelemetryService` with attribute allowlist validation (Section 6.4)
- [x] Materialized view for model performance dashboard (Section 5.4)
- [x] Prometheus metrics wiring plan for all 6 unwired counters (Section 8)
- [x] `heartbeat_results.cost` fix documented (Section 9)
- [x] Batch reconciliation job specification (Section 12)

### Nice to Have

- [x] Propensity logging design for future bandit evaluation (Section 11.3)
- [x] OTel `gen_ai.*` attribute naming in event metadata (Section 3.3)
- [x] Sufficient statistics (sum + sum-of-squares) in materialized views (Section 5.4)
- [x] API endpoint specifications with example responses (Section 10)

---

## Appendix A: SQLAlchemy Model for `mission_events`

```python
from sqlalchemy import Column, BigInteger, Integer, String, Float, DateTime, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB
from core.database.database import Base

class MissionEvent(Base):
    """Append-only telemetry event log for missions."""
    __tablename__ = 'mission_events'

    id = Column(BigInteger, primary_key=True)  # BIGSERIAL
    mission_run_id = Column(BigInteger, ForeignKey('mission_runs.id', ondelete='CASCADE'), nullable=False)
    mission_task_id = Column(BigInteger, ForeignKey('mission_tasks.id', ondelete='CASCADE'), nullable=True)
    event_type = Column(String(50), nullable=False)

    # OTel-inspired attributes
    attributes = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))

    # Extracted for fast filtering
    agent_id = Column(Integer, nullable=True)
    model_id = Column(String(255), nullable=True)

    # Numeric payload
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        Index('idx_me_run', 'mission_run_id'),
        Index('idx_me_task', 'mission_task_id', postgresql_where=text('mission_task_id IS NOT NULL')),
        Index('idx_me_type', 'event_type'),
        Index('idx_me_attrs', 'attributes', postgresql_using='gin'),
    )
```

## Appendix B: Research Sources

| Source | What It Informed |
|--------|-----------------|
| MLflow entity model (`mlflow/mlflow`) | Three-tier storage: metrics (append-only) vs params (immutable) vs tags (mutable) |
| W&B summary/history split (`wandb/wandb`) | Summary columns for dashboards + event log for deep analysis = hybrid approach |
| OpenTelemetry trace/span model (opentelemetry.io) | Mission=trace / task=span mental model; `gen_ai.*` attribute naming conventions |
| Honeycomb high-cardinality querying (docs.honeycomb.io) | JSONB metadata for flexible GROUP BY; no pre-aggregation needed at our scale |
| Eppo assignment/metric model (docs.geteppo.com) | Attribution windows, raw event storage, join at analysis time |
| Deng, Microsoft ExP | Sufficient statistics (`sum` + `sum_sq`) for variance without raw data re-read |
| Eugene Yan, counterfactual evaluation | Propensity logging (`action_probability`) for future offline model evaluation |
| Automatos codebase audit | `llm_usage`, `heartbeat_results`, `tool_execution_logs`, `votes`, `automatos_metrics.py` — 7+ existing telemetry touchpoints identified |
