# PRD-105 Outline: Budget & Governance

**Type:** Research + Design Outline
**Status:** Outline
**Depends On:** PRD-101 (Mission Schema), PRD-100 (Master Research)
**Feeds Into:** PRD-82C (Parallel Execution + Budget + Contractors)

---

## 1. Problem Statement

Automatos has **no per-mission budget enforcement**. Cost data flows from LLM responses into the `llm_usage` table (via `UsageTracker`), and analytics endpoints surface spending trends, but nothing blocks a mission from spending beyond any limit. The platform records what was spent — it never prevents overspending.

### What's Missing

| Gap | Impact |
|-----|--------|
| No pre-call budget check | A runaway mission can exhaust an entire workspace's LLM credits in minutes |
| No per-mission cost cap | Coordinator-spawned tasks have no aggregate spending boundary |
| No tool policy layering | Every agent gets every tool assigned in DB — no mission-scoped restrictions |
| No approval gates beyond chat | Complex missions auto-execute with no human checkpoint before expensive operations |
| `Workspace.plan_limits` JSONB exists but is never read | The schema hook for enforcement is present but unwired |
| Two `TokenBudgetManager` classes serve different purposes | `modules/context/budget.py` (context-window packing) vs `modules/orchestrator/stages/token_budget_manager.py` (workflow tokens, in-memory only, has latent `AttributeError` bugs) — confusion risk |

### Why This Matters Now

Mission Mode (PRD-102) introduces a coordinator that decomposes goals into multiple tasks, each consuming LLM calls. Without budget enforcement:
- A 10-task mission using GPT-4-class models could cost $5-50 depending on complexity
- Users have no visibility into projected cost before execution
- There's no mechanism to halt a mission that's burning faster than expected
- Multi-tenant workspaces cannot isolate cost between users/missions

---

## 2. Prior Art Research Targets

### 2.1 OpenClaw 8-Stage Tool Policy Chain

**Source:** [OpenClaw docs](https://docs.openclaw.ai/tools/multi-agent-sandbox-tools), [GitHub](https://github.com/openclaw/openclaw)

OpenClaw implements an **8-stage monotonically narrowing** tool policy chain (originally documented as "6 tiers" in PRD-100 — actually 8):

| Stage | Name | Controls |
|-------|------|----------|
| 1 | Tool Profile (global) | Base allowlist template (`minimal`, `coding`, `messaging`, `full`) |
| 2 | Provider Tool Profile | Narrows tools per LLM provider/model |
| 3 | Global Tool Policy | Explicit `allow`/`deny` rules across all agents |
| 4 | Provider Tool Policy | Per-provider `allow`/`deny` beyond profile |
| 5 | Agent-Specific Policy | Per-agent `allow`/`deny` and profile override |
| 6 | Agent Provider Policy | Per-agent per-provider restriction |
| 7 | Sandbox Policy | Tools allowed inside Docker-sandboxed execution |
| 8 | Subagent Policy | Tools passed to spawned child agents (cannot exceed parent's set) |

**Key design principle:** Each stage can only **narrow** the tool set — never expand. Deny always wins over allow. Enforcement happens at **tool-set construction** (tools passed to LLM `tools=` param), not post-hoc interception. A denied tool never appears in the model's function schema.

**What to adopt for Automatos:**
- Monotonic narrowing invariant (workspace → mission → task → agent)
- Tool group shorthand (`group:fs`, `group:web`, etc.) for policy configuration
- Enforcement at tool-set construction (already how `get_tools_for_agent()` works in `tool_router.py`)

**What doesn't apply:**
- No temporal/budget dimension — OpenClaw controls which tools, not how often or at what cost
- No per-mission scoping — policies are static config, not runtime-dynamic
- Single-user gateway model — no multi-tenancy

### 2.2 AWS Budgets & Cost Management

**Source:** [AWS Budgets API docs](https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/)

AWS implements budget enforcement through **soft caps with automated actions**:

- **Budget types:** COST (dollars) and USAGE (quantity) — both relevant to mission budgeting
- **`CUSTOM` time period:** Fixed start/end, no auto-renew — maps to mission lifecycle
- **Graduated thresholds:** Up to 5 per budget (e.g., warn at 50%, alert at 80%, act at 100%)
- **Budget Actions:** `APPLY_IAM_POLICY` (deny access), `RUN_SSM_DOCUMENTS` (stop instances), `APPLY_SCP_POLICY` (org-level block)
- **Approval models:** `AUTOMATIC` (fire immediately) or `MANUAL` (queue for human)
- **Cost allocation tags:** Per-resource tagging for attribution (e.g., `MissionId`, `AgentId`)

**Critical lesson:** AWS has **no true hard cap** — billing data updates every 8-12 hours. For LLM missions that can exhaust budgets in seconds, this lag is fatal. **We need synchronous pre-call checks**, not post-hoc billing scrapes.

**Adoptable patterns:**
- Graduated soft/hard cap design (warn → throttle → stop)
- Separate action thresholds from notification thresholds
- `AUTOMATIC` vs `MANUAL` approval model per budget tier
- Dual COST + USAGE budget types (track dollars AND tokens independently)
- Tag-based attribution for post-hoc analysis

### 2.3 Kubernetes Resource Quotas & Admission Control

**Source:** [K8s docs](https://kubernetes.io/docs/concepts/policy/resource-quotas/)

K8s enforces resource limits through **synchronous admission control** — the single most applicable pattern:

```
Request → AuthN → AuthZ → Mutating Admission → Validating Admission → etcd write
                              ↑ LimitRanger              ↑ ResourceQuota
                           (inject defaults)         (check aggregate)
```

**Key properties:**
- **Hard rejection, not queuing:** API server returns HTTP 403 synchronously. The resource is never created.
- **Two-layer limits:** `ResourceQuota` (namespace aggregate) + `LimitRange` (per-pod/container defaults and maximums)
- **Quota scopes:** `PriorityClass`-based quotas let you reserve budget for high-priority operations
- **Quota does not retroactively evict:** Lowering quota doesn't kill running workloads — enforcement fires on the *next* admission

**Direct translation to mission budgeting:**

| K8s Concept | Mission Equivalent |
|-------------|-------------------|
| Namespace | Mission (isolated budget boundary) |
| `ResourceQuota spec.hard` | Mission budget: `max_tokens`, `max_cost_usd`, `max_wall_time_s` |
| `LimitRange default + max` | Per-agent defaults and ceilings within a mission |
| Mutating admission | Budget middleware: inject default allocation to unspecified agents |
| Validating admission | Pre-call check: `current_spend + estimated_cost ≤ ceiling`, else reject |
| HTTP 403 | Raise `BudgetExceededError` before LLM call |
| Quota scopes | Priority sub-budgets: coordinator/verifier vs worker agents |

### 2.4 Rate Limiting Algorithms

**Sources:** Cloudflare engineering blog, Stripe rate limiting docs, Anthropic API docs

| Algorithm | Best For | Limitation |
|-----------|----------|------------|
| Fixed Window Counter | Simple per-minute/hour caps | "Boundary burst" — 2x at window edges |
| Sliding Window Counter | Production rate limiting at scale (Cloudflare: 0.003% error on 400M reqs) | Approximation, not exact |
| Token Bucket | Bursty-but-bounded traffic; **Anthropic and Stripe use this** | Two params to tune |
| Leaky Bucket | Constant throughput enforcement | No burst tolerance |
| Adaptive | Backends with their own limits (e.g., OpenRouter) | Complex, oscillation risk |

**For mission budgeting, use a cost-denominated token bucket:**
- Bucket capacity = mission budget in dollars
- Refill disabled (missions have fixed, non-replenishing budgets)
- Each LLM call consumes `estimated_cost` tokens from the bucket
- After call, reconcile with actual cost from response

**Pre-estimation formula:**
```python
estimated_cost = (input_tokens × model_input_rate) + (max_output_tokens × model_output_rate)
```
- Input tokens: countable exactly pre-call via tokenizer
- Output tokens: use `max_tokens` as worst case, reconcile after

**Anthropic tier structure (for reference):**
- Tier 1: 50 RPM, 30K ITPM, $100/mo cap
- Tier 4: 4,000 RPM, 2M ITPM, $200K/mo cap
- Cached input tokens do NOT count toward ITPM

### 2.5 LiteLLM BudgetManager

**Source:** [LiteLLM docs](https://docs.litellm.ai/docs/budget_manager)

LiteLLM implements a two-phase budget pattern:
1. `projected_cost(model, messages, user)` — pre-call estimate
2. `update_cost(completion_response, user)` — post-call reconciliation

This is the closest existing implementation to what Automatos needs for per-mission budget enforcement.

---

## 3. Budget Model

### 3.1 What to Track

| Dimension | Granularity | Source |
|-----------|-------------|--------|
| **Token consumption** (input + output) | Per-call, per-task, per-mission | LLM response `usage` field |
| **Cost (USD)** | Per-call, per-task, per-mission | `llm_models.input_cost_per_1k_tokens` × tokens |
| **API calls** | Per-task, per-mission | Counter increment per LLM invocation |
| **Tool invocations** | Per-task, per-mission | Counter per `execute_tool()` call |
| **Wall time** | Per-task, per-mission | `started_at` → `completed_at` delta |
| **Verification cost** | Per-task (separate from generation) | Track verifier LLM calls separately |

### 3.2 Budget Hierarchy

```
Workspace Budget (plan_limits JSONB — monthly cap)
  └─ Mission Budget (budget_config on orchestration_runs — per-mission cap)
       └─ Task Budget (allocation from mission budget — per-task sub-cap)
            └─ Agent Budget (per-agent-per-task limit — prevents single agent runaway)
```

### 3.3 Budget Lifecycle

```
CREATE MISSION
  → Coordinator estimates cost ("~15 LLM calls, $2-4 estimated")
  → User sets or accepts budget cap
  → Budget record created with cap + $0.00 spent

BEFORE EACH LLM CALL (admission gate)
  → Pre-estimate cost from model pricing + input token count + max_output_tokens
  → Check: current_spent + estimated_cost ≤ mission_cap
  → If yes: proceed
  → If no: raise BudgetExceededError → coordinator decides (abort, downgrade model, skip optional task)

AFTER EACH LLM CALL (reconciliation)
  → Record actual cost from response usage
  → Update running total
  → Check status thresholds (HEALTHY → WARNING → CRITICAL → EXCEEDED)

MISSION COMPLETE
  → Final cost reconciliation
  → Budget summary in mission report
```

### 3.4 Data Model Requirements (feeds PRD-101 schema)

```python
# On orchestration_runs (PRD-101 table)
budget_config: JSONB = {
    "max_cost_usd": 5.00,
    "max_tokens": 500_000,
    "max_wall_time_s": 3600,
    "warn_threshold_pct": 50,
    "throttle_threshold_pct": 80,
    "model_downgrade_enabled": True,
    "approval_model": "automatic"  # or "manual"
}
budget_spent: JSONB = {
    "cost_usd": 2.34,
    "input_tokens": 150_000,
    "output_tokens": 45_000,
    "api_calls": 12,
    "tool_invocations": 8,
    "verification_cost_usd": 0.45,
    "wall_time_s": 342
}
budget_status: String = "warning"  # healthy/warning/critical/exceeded
```

---

## 4. Governance Layers

### 4.1 Tool Policy Layering (inspired by OpenClaw, adapted for multi-tenant)

Automatos needs a **4-tier monotonically narrowing** tool policy:

| Tier | Scope | Who Sets It | Example |
|------|-------|------------|---------|
| 1 | **Workspace** | Workspace admin | "No browser tools in this workspace" |
| 2 | **Mission** | Mission creator / coordinator | "This research mission only needs web_search and document tools" |
| 3 | **Task** | Coordinator (per-task assignment) | "This writing task doesn't need code execution" |
| 4 | **Agent** | Existing DB agent config | Current `get_tools_for_agent()` behavior — intersection with above |

**Enforcement point:** `tool_router.py:get_tools_for_agent()` — already the single source of truth. Add policy intersection before returning tools.

### 4.2 Model Access Policies

| Policy | Purpose |
|--------|---------|
| Workspace model allowlist | Which models this workspace can use (already: `LLMModelInstall`) |
| Mission model preferences | Per-role model selection (planner=cheap, coder=mid, reviewer=different-family) |
| Budget-triggered downgrade | Auto-switch to `BUDGET_MODELS` when spend exceeds threshold |

### 4.3 Human Approval Gates

| Gate | When | Default |
|------|------|---------|
| Mission plan approval | After coordinator generates task decomposition | ON (show plan, wait for approval) |
| Budget exceeded | When spend hits 100% of cap | ON (halt + notify) |
| High-cost tool use | Tool invocation estimated > $X threshold | OFF (opt-in) |
| Cross-agent data sharing | Agent A reads Agent B's reports | OFF (always allowed within mission) |

### 4.4 Governance Config Storage

**Recommendation:** DB (JSONB on workspace/mission), not YAML files.
- Workspaces already have `plan_limits` JSONB (unwired)
- Missions will have `budget_config` JSONB (PRD-101)
- Tool policies as JSONB arrays on workspace + mission tables
- Human-readable, queryable, API-manageable

---

## 5. Key Design Questions

### Q1: Hard cap vs soft cap?
- **Hard cap** on cost: Mission cannot exceed `max_cost_usd`. Pre-call admission gate rejects.
- **Soft cap** on tokens: Warning when approaching, but don't reject (token counts are less directly meaningful to users than dollars).
- **Hybrid:** Hard on dollars, soft on everything else.

### Q2: Pre-estimation accuracy — how good can it be?
- Input tokens: exact (tokenizer count)
- Output tokens: worst case = `max_tokens`, typical = 30-50% of max
- OpenRouter returns pricing per model — `llm_models` table has `input_cost_per_1k_tokens` / `output_cost_per_1k_tokens`
- **Risk:** Model pricing changes without DB update → stale cost estimates
- **Mitigation:** Sync pricing from OpenRouter periodically; use worst-case estimates

### Q3: What happens when budget exceeded mid-task?
Options (coordinator decides based on `approval_model`):
1. **Abort mission** — mark as `budget_exceeded`, save partial results
2. **Downgrade model** — switch remaining tasks to `BUDGET_MODELS`
3. **Pause for human** — halt execution, notify user, wait for budget increase
4. **Complete current task, stop** — finish in-flight work, don't start new tasks
- K8s pattern: in-flight work completes; next admission is rejected. **Adopt this.**

### Q4: Per-model cost tracking with OpenRouter pricing?
- `UsageTracker` already reads `LLMModel.input_cost_per_1k_tokens` — this is the cost source
- OpenRouter returns `usage.total_tokens` in responses — already parsed by `LLMManager`
- **Gap:** `UsageTracker` doesn't tag calls with `mission_id` — needs a new column or tag field
- **Gap:** No pre-call cost estimation path exists — must build the admission gate

### Q5: Governance config — DB vs YAML?
- **DB wins** for multi-tenant SaaS (per-workspace, per-mission configs)
- YAML is for self-hosted/single-tenant (OpenClaw pattern)
- Use `Workspace.plan_limits` JSONB (already exists, unwired) for workspace-level
- Use `budget_config` JSONB on `orchestration_runs` (PRD-101) for mission-level

### Q6: How does budget interact with verification costs?
- PRD-103 defines verification as 10-30% of task generation cost
- Budget must account for verification: `task_cost = generation + verification`
- **Option A:** Include verification in the same budget pool
- **Option B:** Reserve a separate verification sub-budget (like K8s `PriorityClass` quotas)
- **Recommendation:** Option A (simpler), but track `verification_cost_usd` separately in `budget_spent`

### Q7: BudgetMLAgent cascade pattern — adopt?
- Pattern: free model → cheap model → expensive model, escalating only when quality is insufficient
- RouteLLM (ICLR 2025): 75% cost reduction at 95% quality with static role→model mapping
- BudgetMLAgent: 96% cost reduction with cascade
- **Recommendation:** Static role→model mapping for v1 (PRD-104 scope), cascade for v2

---

## 6. Existing Codebase Touchpoints

### Budget & Cost Infrastructure

| File | What It Does | Relevance to PRD-105 |
|------|-------------|---------------------|
| `orchestrator/modules/context/budget.py` | Context-window packing budget (TokenBudgetManager) | **Name collision risk** — mission budget is a different concept. Consider renaming or namespacing. |
| `orchestrator/modules/orchestrator/stages/token_budget_manager.py` | Workflow-scoped token allocation (in-memory) | **Structural template** for mission budget manager. Has latent bugs (`config.TOKEN_BUDGET_DEFAULT` doesn't exist in `config.py`). |
| `orchestrator/core/llm/usage_tracker.py` | Per-call cost recording to `llm_usage` table | **Post-call recording path** — extend with `mission_id` tag. Wire pre-call check here. |
| `orchestrator/core/llm/manager.py:643-671` | Calls `UsageTracker.track()` after LLM response | **Integration point** for pre-call admission gate (add check before `_call_provider`). |
| `orchestrator/core/models/core.py:138-170` | `llm_usage` table schema | Needs `mission_id` / `mission_task_id` foreign keys for attribution |
| `orchestrator/core/models/core.py:43-90` | `llm_models` table with pricing per 1K tokens | **Cost source** for pre-estimation |
| `orchestrator/api/llm_analytics.py` | Cost analytics endpoints | Extend with per-mission cost breakdown |
| `orchestrator/core/llm/openrouter_analytics.py` | OpenRouter credit/activity sync | Source for model pricing updates |

### Rate Limiting & Security

| File | What It Does | Relevance to PRD-105 |
|------|-------------|---------------------|
| `orchestrator/core/security/rate_limiter.py` | Redis sliding-window rate limiter | **Pattern to extend** for mission cost rate limiting |
| `orchestrator/api/widgets/rate_limit.py` | Widget-specific rate limiting middleware | Not directly relevant (operational, not cost-based) |

### Governance & Access Control

| File | What It Does | Relevance to PRD-105 |
|------|-------------|---------------------|
| `orchestrator/core/workspaces/permissions.py` | RBAC: OWNER/ADMIN/EDITOR/VIEWER | Add `budget:set`, `budget:override` permissions |
| `orchestrator/core/models/workspaces.py:32-33` | `Workspace.plan`/`plan_limits` JSONB | **Unwired hook** — wire for workspace-level budget enforcement |
| `orchestrator/modules/tools/tool_router.py:140` | `get_tools_for_agent()` — single source of truth for agent tools | **Policy enforcement point** — add tool policy intersection |
| `orchestrator/config.py:435-445` | `PREMIUM_MODELS`, `BUDGET_MODELS`, savings ratio | **Model tier data** for downgrade-on-budget-pressure pattern |

### Agent & Execution

| File | What It Does | Relevance to PRD-105 |
|------|-------------|---------------------|
| `orchestrator/modules/agents/factory/agent_factory.py` | Agent execution with tool loop (max 10 iterations) | Each iteration = potential LLM call = budget check needed |
| `orchestrator/modules/tools/execution/unified_executor.py` | Tool dispatch by prefix | Tool invocation counting for governance |
| `orchestrator/services/heartbeat_service.py` | Orchestrator + agent ticks with rate limiting | Existing rate limiting pattern to reference |

---

## 7. Acceptance Criteria for Full PRD-105

### Must Have

- [ ] **Budget admission gate design:** Synchronous pre-call check with estimated cost, integrated into `LLMManager` call path
- [ ] **Budget data model:** `budget_config` and `budget_spent` JSONB schemas (feeds PRD-101), `budget_status` enum
- [ ] **Budget lifecycle:** Create → estimate → enforce → reconcile → report, with state machine
- [ ] **Graduated thresholds:** Configurable warn/throttle/stop percentages with distinct actions per threshold
- [ ] **Pre-estimation algorithm:** Input token counting + output worst-case + model pricing lookup
- [ ] **Post-call reconciliation:** Actual cost recording with mission/task attribution
- [ ] **Budget exceeded handling:** At least 3 strategies (abort, downgrade, pause) with coordinator decision logic
- [ ] **Tool policy layering design:** 4-tier narrowing model (workspace → mission → task → agent) with enforcement at `get_tools_for_agent()`
- [ ] **`Workspace.plan_limits` activation:** Design for wiring the existing JSONB field to enforcement
- [ ] **RBAC extensions:** `budget:set`, `budget:override`, `budget:view` permissions
- [ ] **Cost attribution:** How `llm_usage` rows link to mission runs and tasks

### Should Have

- [ ] **Human approval gates design:** Mission plan approval and budget-exceeded notification flows
- [ ] **Model downgrade strategy:** Static role→model mapping with budget-triggered fallback to `BUDGET_MODELS`
- [ ] **Mission cost estimation endpoint:** Pre-execution cost projection for user review
- [ ] **Budget dashboard design:** Per-mission spend tracking, remaining budget, burn rate

### Nice to Have

- [ ] **Adaptive rate limiting design:** Adjust call frequency based on budget burn rate
- [ ] **Cross-mission budget aggregation:** Monthly workspace-level spending reports
- [ ] **Budget templates:** Reusable budget configs for common mission types

---

## 8. Risks & Dependencies

### Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Pre-estimation inaccuracy | Medium — over-estimate blocks legitimate work, under-estimate allows overspend | High | Use worst-case (max_tokens) for output estimate; reconcile after each call; allow 10% overage buffer |
| 2 | Stale model pricing in DB | Medium — cost calculations wrong if prices change | Medium | Periodic sync from OpenRouter API; timestamp pricing data; alert on age > 7 days |
| 3 | Budget check latency | Low — adds round-trip per LLM call | Medium | Redis-based running total (sub-ms read); avoid DB query per call |
| 4 | Governance overhead / user friction | High — too many approval gates → users disable everything | Medium | Defaults should be minimal (plan approval ON, everything else OFF); progressive disclosure |
| 5 | Context-budget vs cost-budget confusion | Low — two `TokenBudgetManager` classes | High | Clear naming: `ContextBudgetManager` vs `MissionBudgetManager`; document distinction |
| 6 | Verification cost unpredictable | Low — verifier can use more tokens than expected | Medium | Cap verification at 30% of task generation cost; separate tracking |
| 7 | In-flight work when budget exceeded | Medium — can't interrupt an LLM call mid-stream | Low | K8s pattern: in-flight completes, next admission rejected; track overage |

### Dependencies

| Dependency | PRD | Why |
|------------|-----|-----|
| `orchestration_runs` table with `budget_config` JSONB | PRD-101 | Budget needs a home in the data model |
| `orchestration_tasks` with `cost_spent` tracking | PRD-101 | Per-task cost attribution |
| Coordinator service that creates/manages missions | PRD-102 | Coordinator is the budget consumer — it checks budget before spawning tasks |
| Verification cost as a budget dimension | PRD-103 | Verification adds to mission cost; must be budgeted |
| Contractor agent lifecycle | PRD-104 | Contractors inherit mission budget constraints |
| `mission_events` for budget audit trail | PRD-101 | Every budget check/alert/exceed should be an event |

### Cross-PRD Notes

- PRD-101 must include `budget_config`, `budget_spent`, `budget_status` fields on `orchestration_runs`
- PRD-102 coordinator must call budget admission gate before each agent execution
- PRD-103 verification cost should be tracked separately within the budget (`verification_cost_usd`)
- PRD-104 contractor agents must inherit the mission's remaining budget as their ceiling
- PRD-106 telemetry must capture budget utilization metrics for pattern analysis
- The stages `TokenBudgetManager` (`stages/token_budget_manager.py`) has latent `AttributeError` bugs — `config.TOKEN_BUDGET_DEFAULT` etc. don't exist in `config.py`. PRD-105 implementation should either fix or replace this class.
- `Workspace.plan_limits` JSONB is the existing hook for workspace-level budget — wire it, don't create a new field.

---

## Appendix: Research Sources

| Source | What It Informed |
|--------|-----------------|
| OpenClaw docs (docs.openclaw.ai) | 8-stage tool policy chain, monotonic narrowing, enforcement at tool-set construction |
| AWS Budgets API | Graduated thresholds, AUTOMATIC vs MANUAL actions, cost allocation tags, CUSTOM budget periods |
| K8s ResourceQuota + LimitRange | Synchronous admission control, hard rejection, two-layer limits, scope-based quotas |
| Cloudflare rate limiting blog | Sliding window counter (0.003% error at scale), algorithm comparison |
| Anthropic API docs | Token bucket rate limiting, tier structure, cached tokens excluded from ITPM |
| LiteLLM BudgetManager | `projected_cost()` + `update_cost()` two-phase pattern |
| RouteLLM (ICLR 2025) | 75% cost reduction at 95% quality with static model routing |
| BudgetMLAgent | Cascade pattern: free → cheap → expensive, 96% cost reduction |
| Automatos codebase | UsageTracker, LLMManager, TokenBudgetManager(s), rate_limiter, plan_limits, PREMIUM/BUDGET_MODELS |
