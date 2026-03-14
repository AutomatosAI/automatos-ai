# PRD-106 Outline: Outcome Telemetry & Learning Foundation

**Type:** Research + Design Outline
**Status:** Outline
**Depends On:** PRD-101 (Mission Schema), PRD-103 (Verification & Quality), PRD-100 (Master Research)
**Feeds Into:** PRD-82D (Complexity Detection + Outcome Telemetry), future model routing & agent selection systems

---

## 1. Problem Statement

Automatos has **no structured outcome telemetry for missions**. The `llm_usage` table captures per-call token/cost data, and `heartbeat_results` stores JSONB findings, but nothing correlates a multi-step mission to its aggregate cost, duration, quality, or human acceptance. The platform records individual API calls — it never answers "which model performs best for research tasks" or "what's the average cost of a compliance mission."

### What's Missing

| Gap | Impact |
|-----|--------|
| No mission-level outcome record | Cannot correlate total cost/tokens/duration to a mission's success or failure |
| No per-task structured outcome | `board_tasks.result` is free text — no machine-readable quality score, token spend, or retry count |
| No agent performance attribution | `llm_usage.agent_id` exists but no query path links agent → task type → outcome quality |
| No model comparison data | Cannot answer "did Claude Opus or GPT-4 produce better research outputs?" — no structured outcome-to-model linkage |
| No human feedback loop closure | `votes.is_upvoted` captures chat-level feedback; no mission/task-level acceptance signal feeds back to inform future assignments |
| Prometheus agent metrics defined but unwired | `automatos_agent_heartbeat_total`, `automatos_agent_token_usage_total`, `automatos_llm_request_duration_seconds` exist in `automatos_metrics.py` but are never incremented |
| `heartbeat_results.cost` column always 0.0 | Schema includes cost field but `_store_heartbeat_result()` never populates it |
| Context window telemetry ephemeral | `ContextResult.token_estimate` and `sections_trimmed` computed per-request but never persisted |

### Why This Matters Now

PRD-100 Section 3 explicitly states: *"No fancy learning engine — just data. Query it for patterns later."* This PRD defines what "just data" means concretely — the schema, capture points, and query patterns that make future optimization possible without building the optimization engine now.

Without structured telemetry:
- Mission Mode ships blind — no way to measure if it's working
- Model routing remains manual forever — no data to automate it
- Cost optimization is guesswork — no per-task-type cost benchmarks
- The Phase 3 learning foundation (recommendation, bandit-style selection) has no training signal

---

## 2. Prior Art Research Targets

### Systems to Study

| System | Source | Focus Areas | Key Question |
|--------|--------|-------------|--------------|
| **MLflow** | `mlflow/mlflow` GitHub | Experiment/run/metric entity model, append-only metrics, params (immutable) vs tags (mutable), nested runs for parent-child | How do they separate config (inputs) from outcomes (outputs) in a queryable way? |
| **Weights & Biases** | `wandb/wandb` GitHub | Run config/summary/history tiers, auto-captured system metrics, `define_metric` aggregation control, MongoDB-style query API | How does W&B enable cross-run comparison with dynamic grouping? |
| **OpenTelemetry** | opentelemetry.io spec | Trace/span model, attribute semantic conventions, `gen_ai.*` namespace, span events and links | Can mission=trace / task=span give us distributed tracing for free? |
| **Honeycomb** | docs.honeycomb.io | High-cardinality querying without pre-aggregation, columnar span storage, tail-based sampling | How do they enable "group by any attribute" at scale? |
| **Eppo / Statsig** | docs.geteppo.com, docs.statsig.com | Assignment logs, metric source contracts, attribution windows, SRM checks, sufficient statistics | What data model supports offline counterfactual evaluation of model choices? |
| **Existing Automatos** | Codebase audit | `llm_usage`, `heartbeat_results`, `agent_reports`, `votes`, Prometheus metrics, `RecipeQualityService` | What's already captured that we can extend vs. what needs new infrastructure? |

### Key Patterns Discovered in Research

**MLflow's Three-Tier Storage (MLflow docs):** Metrics are append-only time-series (token spend per retry step), params are immutable config (agent_id, model, task_type), tags are mutable state (human_accepted, review_status). This separation enables clean querying — filter by config, aggregate by outcome, update status post-hoc. **Adopt:** immutable config vs. mutable outcome separation.

**W&B Summary vs. History (W&B docs):** Every run has a `summary` dict (final/aggregate values for cross-run comparison) and a `history` log (per-step time-series). Summary uses configurable aggregation (`min`, `max`, `mean`, `last`). **Adopt:** mission_tasks should have both a summary row (final cost, score, status) and detailed event history (per-tool-call, per-retry).

**OpenTelemetry gen_ai Semantic Conventions (OTel spec, emerging):** Standard attribute names for AI workloads: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`. The trace=mission / span=task mapping is natural. **Adopt:** attribute naming conventions even if we don't use OTel transport initially.

**Honeycomb's Schema-Free Columnar Model (Honeycomb docs):** No pre-aggregation — raw events stored columnar, GROUP BY computed at query time. Enables high-cardinality queries (group by model × task_type × agent) without cardinality explosion. **Adopt:** JSONB attributes on telemetry rows (queryable via Postgres jsonb_path operators) rather than fixed columns for every metric.

**Eppo's Attribution Window Pattern (Eppo docs):** Outcomes are joined to treatments at analysis time via SQL, not stored as FK relations. Attribution window is on the metric definition, not the event. **Adopt:** store raw events with timestamps; join mission→outcome at query time rather than denormalizing everything.

**A/B Testing Sufficient Statistics (Deng, Microsoft ExP):** Store `metric_sum` and `metric_sum_squares` alongside `mean` — these are the minimal sufficient statistics for computing variance without re-reading raw data. **Adopt:** aggregated telemetry views should include sum and sum-of-squares for numeric metrics.

**Propensity Logging (Open Bandit Dataset, Eugene Yan):** If we want future offline evaluation of model choices ("what if we'd used Sonnet instead of Opus?"), we need `action_probability` logged at serving time — the probability that the coordinator chose this specific model/agent. **Note for future:** not needed in v1, but the schema should not preclude adding it.

---

## 3. Telemetry Schema

### 3.1 What to Capture Per Task

Based on PRD-100 Section 3 requirements and research findings:

| Field | Type | Source | Immutable? | Notes |
|-------|------|--------|------------|-------|
| `mission_run_id` | FK → mission_runs | Coordinator | Yes | Links to PRD-101 schema |
| `mission_task_id` | FK → mission_tasks | Coordinator | Yes | Links to PRD-101 schema |
| `agent_id` | FK → agents | Assignment | Yes | Roster or contractor |
| `model_id` | string | LLMManager | Yes | e.g., `anthropic/claude-sonnet-4-20250514` |
| `task_type` | string | Coordinator | Yes | e.g., `research`, `writing`, `coding`, `review` |
| `tools_used` | string[] | AgentFactory tool loop | No (append) | Accumulated during execution |
| `tokens_in` | int | UsageTracker aggregation | No (increment) | Sum of input tokens across all LLM calls |
| `tokens_out` | int | UsageTracker aggregation | No (increment) | Sum of output tokens |
| `cost_usd` | decimal | UsageTracker aggregation | No (increment) | Sum of per-call costs |
| `duration_ms` | int | Wall clock | Yes (at completion) | `completed_at - started_at` |
| `verifier_score` | float 0.0-1.0 | PRD-103 verifier | No (set post-execution) | Null until verified |
| `human_accepted` | bool | Human review | No (set post-review) | Null until reviewed |
| `error_type` | string | Exception handler | Yes (at failure) | Null on success; structured enum |
| `retry_count` | int | AgentFactory tool loop | No (increment) | Number of retry attempts |
| `context_tokens_used` | int | ContextService | Yes | How much context window was consumed |
| `context_sections_trimmed` | int | TokenBudgetManager | Yes | Sections dropped due to budget |

### 3.2 What to Capture Per Mission (Aggregated)

| Field | Type | Derivation | Notes |
|-------|------|------------|-------|
| `total_tasks` | int | COUNT(mission_tasks) | Including retried tasks |
| `tasks_passed` | int | COUNT WHERE verifier_score >= threshold | PRD-103 pass threshold |
| `tasks_failed` | int | COUNT WHERE status = 'failed' | |
| `total_tokens` | int | SUM(tokens_in + tokens_out) | Across all tasks |
| `total_cost_usd` | decimal | SUM(cost_usd) | Across all tasks |
| `total_duration_ms` | int | Wall clock for entire mission | Not sum of tasks (parallel) |
| `verification_cost_usd` | decimal | SUM of verifier LLM calls | Separated from task cost |
| `coordination_cost_usd` | decimal | SUM of coordinator LLM calls | Planning + monitoring overhead |
| `human_verdict` | enum | Human review | `accepted`, `rejected`, `partial`, `pending` |
| `converted_to_routine` | bool | User action | PRD-100: "Save as routine?" |

### 3.3 Storage Strategy

```
Option A: Separate telemetry table (mission_telemetry_events)
  + Clean separation of concerns
  + Append-only, immutable events
  + Easy retention policy (truncate old events without touching core tables)
  - Extra JOINs for dashboard queries
  - Duplication risk with llm_usage

Option B: JSONB fields on mission_tasks (task_telemetry JSONB)
  + Zero extra tables
  + Single query for task + telemetry
  + Matches existing patterns (heartbeat_results.findings, recipe_executions.step_results)
  - JSONB indexing limitations for aggregate queries
  - Harder to enforce schema consistency

Option C: Hybrid — summary columns on mission_tasks + event log table
  + Best of both: fast dashboard queries via columns, deep analysis via event log
  + Matches W&B summary/history pattern
  - Two write paths to maintain
```

**Recommendation: Option C (Hybrid).** Summary columns on `mission_tasks` for dashboard queries (cost, tokens, score, status). Separate `mission_events` append-only table for detailed per-step telemetry (each tool call, each retry, each verifier invocation). This matches the W&B summary/history split and aligns with the MLflow append-only metric pattern.

---

## 4. Query Patterns

### Questions the Data Must Answer

The telemetry schema is only useful if it answers real questions. These are the queries PRD-100 envisions:

#### Operational (Day 1)
| Question | Query Shape | Tables |
|----------|-------------|--------|
| What did mission X cost? | `SUM(cost_usd) WHERE mission_run_id = X` | mission_tasks |
| Which tasks failed in mission X? | `WHERE mission_run_id = X AND status = 'failed'` | mission_tasks |
| How long did mission X take? | `completed_at - started_at` | mission_runs |
| What's the cost breakdown by task? | `GROUP BY mission_task_id` | mission_tasks |

#### Analytical (Week 1+)
| Question | Query Shape | Tables |
|----------|-------------|--------|
| Best model for research tasks? | `AVG(verifier_score) GROUP BY model_id WHERE task_type = 'research'` | mission_tasks |
| Average cost per mission type? | `AVG(total_cost_usd) GROUP BY mission_type` | mission_runs |
| Which agents have highest acceptance rate? | `AVG(human_accepted::int) GROUP BY agent_id` | mission_tasks |
| Failure patterns by tool? | `COUNT(*) WHERE error_type IS NOT NULL GROUP BY tools_used, error_type` | mission_tasks + mission_events |
| Verification cost as % of task cost? | `SUM(verification_cost) / SUM(task_cost) GROUP BY task_type` | mission_tasks |

#### Strategic (Month 1+)
| Question | Query Shape | Tables |
|----------|-------------|--------|
| Cost trend over time? | `SUM(cost_usd) GROUP BY DATE_TRUNC('week', created_at)` | mission_tasks |
| Model cost-quality Pareto frontier? | `AVG(verifier_score), AVG(cost_usd) GROUP BY model_id` | mission_tasks |
| Which task types should be automated as routines? | `WHERE converted_to_routine = true GROUP BY mission_type` | mission_runs |
| Retry rate by model? | `AVG(retry_count) GROUP BY model_id` | mission_tasks |

### Indexing Requirements

Based on the query patterns above:
```sql
-- Operational queries (high frequency)
CREATE INDEX idx_mt_mission_run ON mission_tasks(mission_run_id);
CREATE INDEX idx_mt_status ON mission_tasks(status);

-- Analytical queries (medium frequency)
CREATE INDEX idx_mt_model_type ON mission_tasks(model_id, task_type);
CREATE INDEX idx_mt_agent ON mission_tasks(agent_id);

-- Time-range queries
CREATE INDEX idx_mt_created ON mission_tasks(created_at);
CREATE INDEX idx_me_created ON mission_events(created_at);

-- Event log queries
CREATE INDEX idx_me_task ON mission_events(mission_task_id);
CREATE INDEX idx_me_event_type ON mission_events(event_type);
```

---

## 5. Key Design Questions

### Q1: Separate telemetry table or JSONB on existing tables?

**Options:**
- **Separate table** — `mission_telemetry_events` with typed columns
- **JSONB on mission_tasks** — `telemetry JSONB` field (matches `heartbeat_results` pattern)
- **Hybrid (recommended)** — summary columns on `mission_tasks` + `mission_events` append-only log

**Recommendation: Hybrid.** Summary columns enable fast `GROUP BY` queries without JSONB path extraction. Event log enables deep debugging and future ML training. The `llm_usage` table already exists for per-call data — mission telemetry aggregates it, not duplicates it.

### Q2: Retention policy — how long to keep detailed events?

**Options:**
- **Forever** — storage is cheap, disk grows linearly
- **Rolling window** — keep 90 days of events, archive summaries permanently
- **Tiered (recommended)** — full events for 90 days, aggregated summaries forever, raw events archived to cold storage (S3) for ML training

**Recommendation: Tiered.** Mission summary rows are tiny and permanent. Event log rows are larger and time-bounded. Archive to S3 (already configured) before deletion. This matches Honeycomb's sampling philosophy — keep all interesting data, sample routine data.

### Q3: Real-time aggregation or batch?

**Options:**
- **Real-time** — `UsageTracker` increments mission_task summary columns on every LLM call
- **Batch** — periodic job aggregates `llm_usage` rows into mission summaries
- **Hybrid (recommended)** — real-time increment of running totals on mission_tasks, batch job for cross-mission analytics materialized views

**Recommendation: Hybrid.** The `UsageTracker` already fires on every LLM call. Adding a `mission_task_id` column to `llm_usage` and incrementing summary columns is cheap. Batch materialized views for analytics dashboards avoid slow GROUP BY on hot tables.

### Q4: What NOT to store (privacy and size)?

**Answer: Never store prompt content or model outputs in telemetry.** These belong in the agent reports system (PRD-76) and workspace files. Telemetry captures metadata only: model, tokens, cost, duration, score. This keeps the telemetry tables small and avoids PII/IP concerns in analytics queries.

Exceptions:
- Error messages (truncated to 500 chars)
- Tool names and parameters (no parameter values)
- Success criteria text (from mission definition, not from outputs)

### Q5: How to link telemetry to existing `llm_usage` without duplication?

**Options:**
- **Add `mission_task_id` FK to `llm_usage`** — direct linkage, zero duplication
- **Copy fields from `llm_usage` into telemetry** — full denormalization, fast queries
- **Reference via `execution_id` (recommended)** — `llm_usage` already has `execution_id`; mission_events reference the same ID

**Recommendation: Add `mission_task_id` FK to `llm_usage`.** This is the simplest path — one new nullable column on an existing table. Mission-level aggregates can be computed via `SUM(cost) WHERE mission_task_id = X`. No data duplication.

### Q6: How to handle telemetry for coordinator and verifier LLM calls?

**Answer: Tag them separately.** The coordinator's planning calls and the verifier's scoring calls are NOT task execution — they're mission overhead. Use `llm_usage.request_type` (already supports `chat`, `agent`, `recipe`, `routing`, `embedding`) extended with `coordinator` and `verifier` types. This enables computing `coordination_cost_usd` and `verification_cost_usd` as separate line items.

---

## 6. Existing Codebase Touchpoints

### Tables That Already Capture Telemetry

| Table | Key Fields | Relevance to PRD-106 |
|-------|-----------|---------------------|
| `llm_usage` | `input_tokens`, `output_tokens`, `total_cost`, `latency_ms`, `status`, `model_id`, `agent_id`, `request_type` | **Primary data source.** Add `mission_task_id` FK. Extend `request_type` enum with `coordinator`, `verifier` |
| `heartbeat_results` | `findings` JSONB, `actions_taken` JSONB, `tokens_used`, `cost` (always 0.0), `status` | **Fix:** populate `cost` field. **Bridge:** heartbeat-driven tasks → mission telemetry |
| `agent_reports` | `metrics` JSONB, `grade`, `grade_notes`, `report_type` | **Existing outcome signal.** Human grading of agent output — feed into telemetry |
| `votes` | `is_upvoted`, `message_id` | **Existing feedback.** Chat-level only — mission telemetry supersedes for mission tasks |
| `agents.model_usage_stats` | JSONB: `total_tokens`, `total_cost`, `total_requests` | **Existing aggregation.** Per-agent lifetime stats — mission telemetry adds per-task granularity |
| `workflow_recipes.quality_score` | 5-dimension rolling average | **Existing quality metric.** PRD-103 unifies this with mission verification scoring |
| `tool_execution_logs` | `execution_time_ms`, `token_usage` JSONB, `status`, `cache_hit` | **Tool-level telemetry.** Can be linked to mission_events via `execution_id` |

### Code That Writes Telemetry

| File | What It Does | Change Needed |
|------|-------------|---------------|
| `orchestrator/core/llm/usage_tracker.py` | Inserts `llm_usage` row after every LLM call | Add `mission_task_id` parameter; propagate from `LLMManager.generate_response()` |
| `orchestrator/core/llm/manager.py:643-671` | Calls `_track_usage()` with token/cost data | Pass `mission_task_id` through call chain |
| `orchestrator/services/heartbeat_service.py:904-942` | Stores `heartbeat_results` | Fix: populate `cost` field from `llm_usage` aggregation |
| `orchestrator/core/monitoring/automatos_metrics.py` | Defines Prometheus counters (unwired) | Wire `AGENT_TOKEN_USAGE`, `LLM_REQUEST_DURATION` into `UsageTracker` |
| `orchestrator/api/llm_analytics.py` | Analytics query endpoints | Extend with mission-level analytics views |

### Existing Query Infrastructure

| File | What It Does | Extension Point |
|------|-------------|-----------------|
| `orchestrator/api/llm_analytics.py` | REST endpoints for usage analytics | Add `/api/missions/{id}/telemetry`, `/api/analytics/mission-outcomes` |
| `orchestrator/modules/memory/nl2sql/` | Natural language to SQL queries | Train on telemetry tables for "what's my most expensive mission type?" queries |

---

## 7. Acceptance Criteria for Full PRD-106

### Must Have
- [ ] **Telemetry schema defined** — mission_events table DDL with all fields, types, constraints, and indices
- [ ] **`mission_task_id` added to `llm_usage`** — nullable FK, migration script, backwards-compatible
- [ ] **Per-task summary fields** — `tokens_in`, `tokens_out`, `cost_usd`, `duration_ms`, `verifier_score`, `human_accepted` on mission_tasks (PRD-101 schema extension)
- [ ] **Capture points documented** — exactly which code paths write telemetry, with sequence diagram
- [ ] **`request_type` enum extended** — add `coordinator`, `verifier` to `llm_usage.request_type`
- [ ] **Query patterns validated** — all 12+ queries from Section 4 shown as working SQL against the schema
- [ ] **Retention policy defined** — event log TTL, summary permanence, archive strategy

### Should Have
- [ ] **Aggregation strategy** — materialized views or denormalized summary tables for dashboard queries
- [ ] **Prometheus metrics wired** — existing unwired counters connected to `UsageTracker`
- [ ] **`heartbeat_results.cost` fix** — populate the existing but unused cost field
- [ ] **Privacy policy** — explicit list of what is and is not stored in telemetry (no prompt content, no model output)

### Nice to Have
- [ ] **Propensity logging design** — schema supports future `action_probability` field for bandit-style evaluation
- [ ] **OTel attribute naming** — field names follow `gen_ai.*` semantic conventions where applicable
- [ ] **NL2SQL training data** — example natural language → SQL pairs for telemetry queries
- [ ] **Export format** — telemetry exportable as JSONL for external analysis (MLflow, W&B, Jupyter)

---

## 8. Risks & Dependencies

### Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Data volume from event log overwhelms Postgres | High | Medium | Tiered retention (90-day events, permanent summaries), partitioning by `created_at` month |
| 2 | Telemetry write overhead slows LLM call path | High | Low | `UsageTracker` already writes async (separate DB session). Mission_task_id is one extra field, not a new write |
| 3 | Storing too little — can't answer questions we haven't thought of yet | Medium | Medium | Hybrid approach (summary + events) hedges this. JSONB `metadata` escape hatch on events |
| 4 | Storing too much — GDPR/privacy concerns with outcome data | Medium | Low | Strict "no content" policy. Telemetry is metadata only |
| 5 | Premature optimization of learning algorithms | Medium | High | PRD-100 explicitly forbids this. Schema enables learning; PRD-106 does NOT build learning |
| 6 | `llm_usage` table already large — adding FK and index may slow existing queries | Medium | Medium | Nullable FK, added index is on new column only, existing queries unaffected |
| 7 | Aggregated metrics drift from raw data if real-time increment has bugs | Medium | Medium | Batch reconciliation job (daily) re-computes summaries from `llm_usage` source of truth |

### Dependencies

| Dependency | PRD | Why |
|------------|-----|-----|
| `mission_runs` / `mission_tasks` tables | PRD-101 | Telemetry attaches to these entities — they must exist first |
| Verification scoring | PRD-103 | `verifier_score` field populated by the verifier system |
| Budget tracking | PRD-105 | Budget consumed vs. allocated is a telemetry dimension |
| Existing `llm_usage` table | Built | Primary data source — add `mission_task_id` column |
| Existing `UsageTracker` | Built | Primary write path — extend with mission context |

### Cross-PRD Notes

- PRD-101 must include `mission_task_id` as a field that `llm_usage` can FK to
- PRD-103's verifier score is a first-class telemetry field — schema alignment needed
- PRD-105's budget enforcement needs real-time cost aggregation — same data path as telemetry
- PRD-104's contractor agents need telemetry capture despite being ephemeral — `agent_id` must be logged before contractor is destroyed
- PRD-107's context interface should expose `context_tokens_used` and `sections_trimmed` for telemetry capture (currently ephemeral)

---

## Appendix: Research Sources

| Source | What It Informed |
|--------|-----------------|
| MLflow entity model (mlflow/mlflow) | Three-tier storage: metrics (append-only) vs params (immutable) vs tags (mutable) |
| W&B summary/history split (wandb/wandb) | Summary columns for dashboards + event log for deep analysis = hybrid approach |
| OpenTelemetry trace/span model (opentelemetry.io) | Mission=trace / task=span mapping; `gen_ai.*` attribute naming conventions |
| Honeycomb high-cardinality querying (docs.honeycomb.io) | JSONB metadata for flexible GROUP BY without pre-aggregation; tail-based sampling for retention |
| Eppo assignment/metric model (docs.geteppo.com) | Attribution windows, sufficient statistics (sum + sum-of-squares), SRM checks |
| Statsig pipeline (docs.statsig.com) | Metric source annotation pattern — outcomes joined to treatments at analysis time |
| Deng, Microsoft ExP | Minimal sufficient statistics for variance computation without raw data re-read |
| Eugene Yan, counterfactual evaluation | Propensity logging (`action_probability`) for future offline model evaluation |
| Automatos codebase audit | 10+ existing telemetry touchpoints identified; `llm_usage` is the foundation to extend |
