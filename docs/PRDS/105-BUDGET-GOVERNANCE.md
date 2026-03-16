# PRD-105 — Budget & Governance

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P0
**Dependencies:** PRD-101 (Mission Schema), PRD-100 (Research Master)
**Feeds Into:** PRD-82C (Parallel Execution + Budget + Contractors)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-15

---

## 1. Problem Statement

### 1.1 The Gap

Automatos has **no per-mission budget enforcement**. Cost data flows from LLM responses into the `llm_usage` table (via `UsageTracker`), and analytics endpoints surface spending trends, but nothing blocks a mission from spending beyond any limit. The platform records what was spent — it never prevents overspending.

| Gap | Impact |
|---|---|
| No pre-call budget check | A runaway mission can exhaust workspace LLM credits in minutes |
| No per-mission cost cap | Coordinator-spawned tasks have no aggregate spending boundary |
| No tool policy layering | Every agent gets every assigned tool — no mission-scoped restrictions |
| No approval gates beyond chat | Complex missions auto-execute with no human checkpoint |
| `Workspace.plan_limits` JSONB exists but is never read | The schema hook for enforcement is present but unwired |
| Two `TokenBudgetManager` classes serve different purposes | `modules/context/budget.py` (context-window packing) vs `stages/token_budget_manager.py` (workflow tokens, latent bugs) |

### 1.2 What This PRD Delivers

1. **Budget admission gate** — synchronous pre-call check that blocks LLM calls when budget is exhausted
2. **Budget data model** — `budget_config` and `budget_spent` JSONB schemas on `orchestration_runs`
3. **Graduated thresholds** — warn at 50%, throttle at 80%, stop at 100%
4. **Tool policy layering** — 4-tier monotonically narrowing model (workspace → mission → task → agent)
5. **Pre-estimation algorithm** — predict mission cost before execution for human approval
6. **Post-call reconciliation** — actual cost recorded with mission/task attribution
7. **`Workspace.plan_limits` activation** — wire the existing but unused JSONB field

---

## 2. Prior Art: Budget & Governance Patterns

### 2.1 System-by-System Analysis

#### OpenClaw 8-Stage Tool Policy Chain

OpenClaw implements an **8-stage monotonically narrowing** tool policy chain. Each stage can only narrow the tool set — never expand. Deny always wins over allow. Enforcement happens at tool-set construction (tools passed to LLM `tools=` param), not post-hoc interception.

**What we adopt:** Monotonic narrowing invariant (workspace → mission → task → agent). Tool group shorthand for policy configuration. Enforcement at tool-set construction (already how `get_tools_for_agent()` works).

**What we reject:** 8 stages (overkill). No temporal/budget dimension. No per-mission scoping.

#### K8s ResourceQuota & Admission Control

K8s enforces resource limits through **synchronous admission control** — the API server returns HTTP 403 before the resource is created. Two layers: `ResourceQuota` (namespace aggregate) + `LimitRange` (per-pod defaults/maximums). Quota does not retroactively evict running workloads.

**Direct translation:**

| K8s Concept | Mission Equivalent |
|---|---|
| Namespace | Mission (isolated budget boundary) |
| `ResourceQuota spec.hard` | `max_cost_usd`, `max_tokens`, `max_wall_time_s` |
| `LimitRange default + max` | Per-agent defaults and ceilings within a mission |
| Validating admission | Pre-call check: `current_spend + estimated_cost ≤ ceiling` |
| HTTP 403 | Raise `BudgetExceededError` before LLM call |
| Quota scopes | Priority sub-budgets (coordinator/verifier vs worker agents) |

**What we adopt:** Synchronous admission gate. Hard rejection (not queuing). Two-layer limits (mission aggregate + per-task). In-flight work completes; next admission rejected.

#### AWS Budgets

AWS implements budget enforcement through **soft caps with graduated automated actions**. Up to 5 thresholds per budget. `AUTOMATIC` or `MANUAL` approval model per action. **Critical lesson:** AWS has no true hard cap — billing data updates every 8-12 hours. For LLM missions that exhaust budgets in seconds, this lag is fatal.

**What we adopt:** Graduated thresholds (warn → throttle → stop). Separate action thresholds from notification thresholds. Dual COST + USAGE tracking.

**What we reject:** Post-hoc billing approach. We need synchronous pre-call checks.

#### Token Bucket (Anthropic/Stripe Pattern)

For mission budgeting, use a **cost-denominated token bucket** with no refill:
- Bucket capacity = mission budget in dollars
- Each LLM call consumes `estimated_cost` from the bucket
- After call, reconcile estimated vs actual cost
- Refill disabled (missions have fixed, non-replenishing budgets)

#### LiteLLM BudgetManager

Two-phase pattern: `projected_cost(model, messages, user)` pre-call, `update_cost(completion_response, user)` post-call. This is the closest existing implementation to what we need.

**What we adopt:** The two-phase pattern (estimate → check → execute → reconcile).

### 2.2 Architectural Decisions Summary

| Decision | Choice | Source | Rationale |
|---|---|---|---|
| **Enforcement model** | Synchronous pre-call admission gate | K8s admission control | Prevents overspend before it happens |
| **Budget type** | Hard cap on USD, soft cap on tokens | AWS Budgets + K8s | Dollars matter to users; tokens are internal |
| **Threshold model** | Graduated: warn 50%, throttle 80%, stop 100% | AWS Budgets | Progressive response prevents surprise stops |
| **In-flight handling** | Complete current call, reject next | K8s quota | Interrupting LLM calls wastes tokens already consumed |
| **Running total** | Redis-based for sub-ms reads | Rate limiter pattern | Can't add DB round-trip to every LLM call |
| **Tool policy** | 4-tier monotonic narrowing | OpenClaw | Each scope can only restrict, never expand |
| **Pre-estimation** | Input tokens (exact) + output estimate (max × 0.7) | LiteLLM | Conservative but not worst-case |

---

## 3. Budget Admission Gate

### 3.1 Architecture

```
Agent needs LLM call
    │
    ▼
┌─────────────────────────────────┐
│ MissionBudgetManager.check()    │
│                                 │
│ 1. Get running total from Redis │
│    Key: budget:{mission_id}     │
│    Value: {cost_usd, tokens_in, │
│            tokens_out, calls}   │
│                                 │
│ 2. Estimate this call's cost:   │
│    input_tokens × model_input_rate │
│    + (max_tokens × 0.7) × model_output_rate │
│                                 │
│ 3. Check: spent + estimated ≤ cap │
│    ├─ ≤ 50%: HEALTHY → proceed  │
│    ├─ ≤ 80%: WARNING → proceed + emit event │
│    ├─ ≤ 100%: CRITICAL → proceed + emit event │
│    └─ > 100%: EXCEEDED → raise BudgetExceededError │
└──────────────┬──────────────────┘
               │ proceed
               ▼
         LLM call executes
               │
               ▼
┌─────────────────────────────────┐
│ MissionBudgetManager.reconcile()│
│                                 │
│ 1. Read actual cost from response │
│ 2. Update Redis running total   │
│ 3. Update orchestration_tasks   │
│    cost fields (async)          │
│ 4. Re-check threshold status    │
└─────────────────────────────────┘
```

### 3.2 MissionBudgetManager Interface

```python
class BudgetStatus(StrEnum):
    HEALTHY = "healthy"       # < 50% spent
    WARNING = "warning"       # 50-80% spent
    CRITICAL = "critical"     # 80-100% spent
    EXCEEDED = "exceeded"     # > 100% spent


class BudgetExceededError(Exception):
    """Raised when a mission's budget is exhausted."""
    def __init__(self, mission_id: int, spent: float, cap: float):
        self.mission_id = mission_id
        self.spent = spent
        self.cap = cap
        super().__init__(
            f"Mission {mission_id} budget exceeded: ${spent:.2f} / ${cap:.2f}"
        )


class MissionBudgetManager:
    """
    Per-mission budget enforcement.

    Uses Redis for sub-millisecond running total reads.
    Reconciles with DB asynchronously after each call.

    NOT to be confused with:
    - ContextBudgetManager (modules/context/budget.py) — context window packing
    - TokenBudgetManager (stages/) — workflow token allocation (legacy, has bugs)
    """

    def __init__(self, redis_client, db_session_factory):
        self._redis = redis_client
        self._db_factory = db_session_factory

    async def initialize_budget(
        self, mission_id: int, config: BudgetConfig
    ) -> None:
        """
        Create budget tracking for a new mission.
        Called once when mission transitions to 'running'.
        """
        key = f"budget:{mission_id}"
        await self._redis.hset(key, mapping={
            "cost_usd": "0.0",
            "tokens_in": "0",
            "tokens_out": "0",
            "api_calls": "0",
            "tool_invocations": "0",
            "verification_cost_usd": "0.0",
            "cap_cost_usd": str(config.max_cost_usd),
            "cap_tokens": str(config.max_tokens),
            "warn_pct": str(config.warn_threshold_pct),
            "throttle_pct": str(config.throttle_threshold_pct),
        })
        # TTL = mission max wall time + 1 hour buffer
        await self._redis.expire(key, config.max_wall_time_s + 3600)

    async def check(
        self,
        mission_id: int,
        model: str,
        input_tokens: int,
        max_output_tokens: int,
        is_verification: bool = False,
    ) -> BudgetStatus:
        """
        Pre-call admission check.

        Returns BudgetStatus if call is allowed.
        Raises BudgetExceededError if budget would be exceeded.

        Cost estimation:
        - Input tokens: exact (counted by tokenizer)
        - Output tokens: max_tokens × 0.7 (empirical median)
        """
        budget = await self._get_budget(mission_id)
        if budget is None:
            return BudgetStatus.HEALTHY  # No budget configured

        # Estimate cost
        pricing = await self._get_model_pricing(model)
        estimated_output = int(max_output_tokens * 0.7)
        estimated_cost = (
            (input_tokens * pricing.input_per_1k / 1000)
            + (estimated_output * pricing.output_per_1k / 1000)
        )

        # Check against cap
        projected_total = budget.cost_usd + estimated_cost
        pct = (projected_total / budget.cap_cost_usd) * 100

        if pct > 100:
            raise BudgetExceededError(mission_id, budget.cost_usd, budget.cap_cost_usd)

        status = BudgetStatus.HEALTHY
        if pct > budget.throttle_pct:
            status = BudgetStatus.CRITICAL
        elif pct > budget.warn_pct:
            status = BudgetStatus.WARNING

        return status

    async def reconcile(
        self,
        mission_id: int,
        task_id: int,
        actual_input_tokens: int,
        actual_output_tokens: int,
        actual_cost_usd: float,
        is_verification: bool = False,
    ) -> BudgetStatus:
        """
        Post-call reconciliation.

        Updates running total with actual cost.
        Returns current budget status after update.
        """
        key = f"budget:{mission_id}"
        pipe = self._redis.pipeline()
        pipe.hincrbyfloat(key, "cost_usd", actual_cost_usd)
        pipe.hincrby(key, "tokens_in", actual_input_tokens)
        pipe.hincrby(key, "tokens_out", actual_output_tokens)
        pipe.hincrby(key, "api_calls", 1)
        if is_verification:
            pipe.hincrbyfloat(key, "verification_cost_usd", actual_cost_usd)
        await pipe.execute()

        # Async DB update (non-blocking)
        asyncio.create_task(
            self._update_db_budget(mission_id, task_id, actual_cost_usd)
        )

        # Re-check status
        return await self._compute_status(mission_id)
```

### 3.3 Integration with LLMManager

The budget check inserts into the existing LLM call path:

```python
# In orchestrator/core/llm/manager.py

async def generate_response(self, messages, model, max_tokens, **kwargs):
    mission_context = kwargs.get("mission_context")

    # PRD-105: Budget admission gate
    if mission_context:
        input_tokens = self._count_tokens(messages, model)
        status = await self._budget_manager.check(
            mission_id=mission_context["mission_id"],
            model=model,
            input_tokens=input_tokens,
            max_output_tokens=max_tokens,
            is_verification=mission_context.get("is_verification", False),
        )
        if status == BudgetStatus.WARNING:
            # Emit warning event (non-blocking)
            asyncio.create_task(self._emit_budget_warning(mission_context))

    # Existing LLM call
    response = await self._call_provider(messages, model, max_tokens, **kwargs)

    # PRD-105: Budget reconciliation
    if mission_context:
        await self._budget_manager.reconcile(
            mission_id=mission_context["mission_id"],
            task_id=mission_context.get("task_id"),
            actual_input_tokens=response.usage.input_tokens,
            actual_output_tokens=response.usage.output_tokens,
            actual_cost_usd=self._compute_cost(response, model),
            is_verification=mission_context.get("is_verification", False),
        )

    return response
```

---

## 4. Budget Data Model

### 4.1 Budget Config (on orchestration_runs)

```python
@dataclass(frozen=True)
class BudgetConfig:
    """Mission budget configuration. Stored as JSONB on orchestration_runs."""
    max_cost_usd: float = 5.00          # Hard cap in dollars
    max_tokens: int = 500_000            # Soft cap in total tokens
    max_wall_time_s: int = 3600          # Hard cap in seconds (1 hour default)
    warn_threshold_pct: int = 50         # Emit warning event
    throttle_threshold_pct: int = 80     # Emit critical event + coordinator may downgrade models
    model_downgrade_enabled: bool = True # Allow auto-downgrade to BUDGET_MODELS
    approval_model: str = "automatic"    # "automatic" (fire immediately) or "manual" (queue for human)


BUDGET_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "max_cost_usd": {"type": "number", "minimum": 0.01, "maximum": 1000.00},
        "max_tokens": {"type": "integer", "minimum": 1000},
        "max_wall_time_s": {"type": "integer", "minimum": 60, "maximum": 86400},
        "warn_threshold_pct": {"type": "integer", "minimum": 10, "maximum": 90},
        "throttle_threshold_pct": {"type": "integer", "minimum": 50, "maximum": 99},
        "model_downgrade_enabled": {"type": "boolean"},
        "approval_model": {"type": "string", "enum": ["automatic", "manual"]},
    },
}
```

### 4.2 Budget Spent (on orchestration_runs)

```python
@dataclass
class BudgetSpent:
    """Running budget consumption. Stored as JSONB, updated after each LLM call."""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    tool_invocations: int = 0
    verification_cost_usd: float = 0.0
    coordination_cost_usd: float = 0.0
    wall_time_s: int = 0
```

### 4.3 Budget Status (on orchestration_runs)

```python
class BudgetStatusEnum(StrEnum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"
    NOT_CONFIGURED = "not_configured"
```

---

## 5. Pre-Estimation Algorithm

### 5.1 Per-Call Estimate

```python
def estimate_call_cost(
    model: str,
    input_tokens: int,
    max_output_tokens: int,
) -> float:
    """
    Estimate cost of a single LLM call.

    Input tokens: exact (counted by tokenizer pre-call).
    Output tokens: max_tokens × 0.7 (empirical median for agent tasks).

    Why 0.7 and not 1.0 (worst case)?
    Worst-case over-reserves budget. Missions would stall at 70% actual
    spend because the gate thinks 100% is committed. Reconcile actual vs
    estimated after each call; the running total uses real numbers.
    """
    pricing = get_model_pricing(model)
    estimated_output = int(max_output_tokens * 0.7)

    return (
        (input_tokens * pricing.input_per_1k / 1000)
        + (estimated_output * pricing.output_per_1k / 1000)
    )
```

### 5.2 Per-Mission Estimate

Computed at plan approval time for the user:

```
Mission Cost Estimate
─────────────────────────────────────────────
Task 1: Research EU AI Act       $0.50 (sonnet)
Task 2: Analyze requirements     $0.30 (sonnet)
Task 3: Write report             $0.40 (sonnet)
Task 4: Review report            $0.15 (gpt-4o)
─────────────────────────────────────────────
Subtotal (tasks)                 $1.35
Verification (~3%)               $0.04
Coordination (~2 calls)          $0.10
─────────────────────────────────────────────
Estimated total                  $1.49
Recommended budget               $3.00 (2x buffer)
```

The 2x buffer accounts for retries, replanning, and estimation error.

---

## 6. Budget Exceeded Handling

### 6.1 Coordinator Response to BudgetExceededError

```python
async def _handle_budget_exceeded(
    self, run: OrchestrationRun, task: OrchestrationTask
) -> None:
    """
    Called when budget gate rejects an LLM call.

    K8s pattern: in-flight work completes; next admission rejected.
    The current task's in-progress LLM call (if any) finishes.
    No new LLM calls are allowed until budget is resolved.
    """
    config = BudgetConfig(**run.budget_config)

    if config.model_downgrade_enabled:
        # Strategy 1: Downgrade remaining tasks to BUDGET_MODELS
        remaining_tasks = await self._get_pending_tasks(run.id)
        for t in remaining_tasks:
            t.contractor_config["model"] = self._get_budget_model(t.task_type)
        await self._emit_event(run.id, "budget_model_downgrade", {
            "remaining_tasks": len(remaining_tasks),
        })
        # Re-estimate with cheaper models — may fit within budget
        new_estimate = await self._re_estimate_remaining(run, remaining_tasks)
        if new_estimate <= run.budget_remaining:
            return  # Downgrade resolved it

    if config.approval_model == "manual":
        # Strategy 2: Pause for human decision
        await self._transition_run(run, RunState.BUDGET_EXCEEDED)
        await self._notify_user(run, "budget_exceeded", {
            "spent": run.budget_spent["cost_usd"],
            "cap": config.max_cost_usd,
            "options": ["increase_budget", "cancel_mission", "downgrade_models"],
        })
    else:
        # Strategy 3: Auto-abort
        await self._transition_run(run, RunState.FAILED,
            reason="Budget exceeded and no resolution available")
```

### 6.2 Human Resolution Options

| Action | API Endpoint | Effect |
|---|---|---|
| Increase budget | `POST /missions/{id}/budget` | Updates `budget_config.max_cost_usd`, resumes mission |
| Cancel mission | `DELETE /missions/{id}` | Cancels all pending tasks, mission → cancelled |
| Downgrade models | `POST /missions/{id}/budget` with `downgrade=true` | Switch remaining tasks to BUDGET_MODELS, resume |
| Complete current + stop | `POST /missions/{id}/budget` with `complete_current=true` | Finish in-flight tasks, cancel remaining |

---

## 7. Tool Policy Layering

### 7.1 Four-Tier Narrowing Model

```
Tier 1: Workspace Policy
  Tools: [all platform_*, workspace_*, composio_*]
  Deny: [workspace_exec]  ← admin removed shell access

  └─ Tier 2: Mission Policy
       Tools: [platform_search_web, workspace_read_file, workspace_write_file]
       ← coordinator scoped to research/writing tools

       └─ Tier 3: Task Policy
            Tools: [platform_search_web, workspace_read_file]
            ← this specific task only needs search + read

            └─ Tier 4: Agent Policy
                 Tools: [platform_search_web, workspace_read_file]
                 ← agent's DB tool assignments intersected with above
                 ← for contractors: explicit tools from coordinator
```

**Invariant: each tier can only narrow, never expand.** The effective tool set is the intersection of all tiers.

### 7.2 Enforcement Point

```python
# In orchestrator/modules/tools/tool_router.py

async def get_tools_for_agent(
    self,
    agent_id: int,
    workspace_id: int,
    mission_policy: Optional[ToolPolicy] = None,  # NEW
    task_policy: Optional[ToolPolicy] = None,      # NEW
    explicit_tools: Optional[list[str]] = None,    # NEW (PRD-104 contractors)
) -> list[dict]:
    """
    Resolve tools for an agent with policy layering.

    For roster agents: DB tools → intersect with policies
    For contractors: explicit_tools → intersect with policies
    """
    # Start with full available tools
    if explicit_tools is not None:
        # Contractor: coordinator-specified tools
        base_tools = await self._resolve_tool_names(explicit_tools, workspace_id)
    else:
        # Roster agent: DB-backed tool assignments
        base_tools = await self._get_agent_db_tools(agent_id, workspace_id)

    # Apply policy layers (narrowing only)
    workspace_policy = await self._get_workspace_tool_policy(workspace_id)
    effective_tools = self._intersect_policies(
        base_tools,
        workspace_policy,
        mission_policy,
        task_policy,
    )

    return effective_tools

def _intersect_policies(
    self,
    tools: list[dict],
    *policies: Optional[ToolPolicy],
) -> list[dict]:
    """Intersect tool set with each policy layer. None = no restriction."""
    result = tools
    for policy in policies:
        if policy is None:
            continue
        if policy.allowed:
            # Allowlist: keep only tools in the allowed set
            allowed_names = set(policy.allowed)
            result = [t for t in result if t["name"] in allowed_names]
        if policy.denied:
            # Denylist: remove denied tools (deny wins over allow)
            denied_names = set(policy.denied)
            result = [t for t in result if t["name"] not in denied_names]
    return result
```

### 7.3 ToolPolicy Schema

```python
@dataclass(frozen=True)
class ToolPolicy:
    """Tool access policy for a specific scope."""
    allowed: Optional[list[str]] = None  # None = no allowlist restriction
    denied: Optional[list[str]] = None   # None = no denylist
    groups: Optional[list[str]] = None   # Tool group shorthand: "web", "filesystem", "code"

# Tool groups (shorthand for common sets)
TOOL_GROUPS: dict[str, list[str]] = {
    "web": ["platform_search_web", "platform_browse_url"],
    "filesystem": ["workspace_read_file", "workspace_write_file", "workspace_list_dir"],
    "code": ["workspace_exec", "workspace_git"],
    "search": ["platform_search_documents", "platform_search_web"],
    "communication": ["platform_send_message", "platform_create_report"],
}
```

---

## 8. Workspace Plan Limits Activation

### 8.1 Current State

`Workspace.plan_limits` JSONB exists on the `workspaces` table but is never read by any code path.

### 8.2 Wiring Design

```python
# In orchestrator/core/models/workspaces.py
# plan_limits JSONB already exists at line ~32-33

# Activation: workspace-level budget enforcement
PLAN_LIMITS_SCHEMA = {
    "type": "object",
    "properties": {
        "max_monthly_cost_usd": {"type": "number"},          # Monthly spending cap
        "max_concurrent_missions": {"type": "integer"},       # Max active missions
        "max_mission_cost_usd": {"type": "number"},           # Per-mission default cap
        "default_tool_policy": {"type": "object"},            # Tier 1 tool policy
        "allowed_models": {"type": "array", "items": {"type": "string"}},
    },
}

# Enforcement: checked at mission creation
async def check_workspace_limits(
    workspace_id: int, proposed_budget: float
) -> None:
    """
    Check workspace-level limits before creating a mission.
    Raises WorkspaceLimitError if any limit would be exceeded.
    """
    workspace = await get_workspace(workspace_id)
    limits = workspace.plan_limits or {}

    # Check monthly spending
    if "max_monthly_cost_usd" in limits:
        month_spend = await get_workspace_monthly_spend(workspace_id)
        if month_spend + proposed_budget > limits["max_monthly_cost_usd"]:
            raise WorkspaceLimitError(
                f"Monthly budget would be exceeded: "
                f"${month_spend:.2f} + ${proposed_budget:.2f} > "
                f"${limits['max_monthly_cost_usd']:.2f}"
            )

    # Check concurrent missions
    if "max_concurrent_missions" in limits:
        active = await count_active_missions(workspace_id)
        if active >= limits["max_concurrent_missions"]:
            raise WorkspaceLimitError(
                f"Max concurrent missions reached: {active}/{limits['max_concurrent_missions']}"
            )
```

---

## 9. RBAC Extensions

### 9.1 New Permissions

| Permission | Who Can | Description |
|---|---|---|
| `mission:create` | EDITOR, ADMIN, OWNER | Create a new mission |
| `mission:approve` | ADMIN, OWNER | Approve/reject mission plans |
| `mission:review` | EDITOR, ADMIN, OWNER | Review mission results |
| `budget:view` | VIEWER, EDITOR, ADMIN, OWNER | See budget status |
| `budget:set` | ADMIN, OWNER | Set mission budget |
| `budget:override` | OWNER | Increase budget beyond workspace limits |
| `tool_policy:set` | ADMIN, OWNER | Configure workspace tool policy |

### 9.2 Integration with Existing RBAC

The existing `permissions.py` uses role-based access (OWNER/ADMIN/EDITOR/VIEWER). New permissions map to existing roles — no new permission system needed, just new permission checks in mission API endpoints.

---

## 10. Cost Attribution

### 10.1 Linking llm_usage to Missions

**Decision: Add nullable `mission_task_id` FK to `llm_usage`.**

```sql
ALTER TABLE llm_usage ADD COLUMN mission_task_id INTEGER
    REFERENCES orchestration_tasks(id) ON DELETE SET NULL;

CREATE INDEX CONCURRENTLY idx_llm_usage_mission_task
    ON llm_usage(mission_task_id) WHERE mission_task_id IS NOT NULL;
```

**Why nullable:** Thousands of existing `llm_usage` rows from non-mission calls (chatbot, heartbeat, routing). These have no mission context and never will. Non-nullable would require backfill.

### 10.2 Request Type Extension

```sql
-- Extend existing request_type enum (or VARCHAR)
-- Existing: 'chat', 'agent', 'recipe', 'routing', 'embedding'
-- New: 'coordinator', 'verifier', 'mission_task'
```

This enables computing:
- `coordination_cost_usd` = SUM(cost WHERE request_type = 'coordinator')
- `verification_cost_usd` = SUM(cost WHERE request_type = 'verifier')
- `task_cost_usd` = SUM(cost WHERE request_type = 'mission_task')

---

## 11. Acceptance Criteria

### Must Have

- [x] **Budget admission gate** — synchronous pre-call check with estimated cost, raises BudgetExceededError
- [x] **Budget data model** — `BudgetConfig` and `BudgetSpent` schemas with JSONB validation
- [x] **Budget lifecycle** — initialize → estimate → enforce → reconcile → report
- [x] **Graduated thresholds** — warn/throttle/stop with configurable percentages and distinct actions
- [x] **Pre-estimation algorithm** — input tokens (exact) + output (max × 0.7) + model pricing
- [x] **Post-call reconciliation** — actual cost recording with Redis running total + async DB update
- [x] **Budget exceeded handling** — 3 strategies (downgrade, pause, abort) with coordinator decision logic
- [x] **Tool policy layering** — 4-tier narrowing model with enforcement at `get_tools_for_agent()`
- [x] **`Workspace.plan_limits` activation** — wiring design for existing JSONB field
- [x] **RBAC extensions** — 7 new permissions mapped to existing roles
- [x] **Cost attribution** — `mission_task_id` FK on `llm_usage`, request_type extension

### Should Have

- [x] **Human resolution options** — increase budget, cancel, downgrade, complete-and-stop
- [x] **Mission cost estimation** — pre-execution projection with 2x buffer recommendation
- [ ] Budget dashboard design — per-mission spend tracking, burn rate

### Nice to Have

- [ ] Adaptive rate limiting — adjust call frequency based on budget burn rate
- [ ] Budget templates — reusable configs for common mission types
- [ ] Cross-mission workspace spending aggregation

---

## 12. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **Pre-estimation inaccuracy** — over-estimate blocks work, under-estimate allows overspend | Medium | High | 0.7 median estimate (not worst case). Reconcile actual after each call. 10% overage buffer. |
| 2 | **Stale model pricing** | Medium | Medium | Periodic sync from OpenRouter API. Timestamp pricing. Alert on age > 7 days. |
| 3 | **Budget check latency** | Low | Low | Redis-based running total (sub-ms). No DB query per call. |
| 4 | **Governance overhead / user friction** | High | Medium | Minimal defaults: plan approval ON, everything else OFF. Progressive disclosure. |
| 5 | **Context-budget vs cost-budget confusion** | Low | High | Clear naming: `ContextBudgetManager` vs `MissionBudgetManager`. Document distinction. |
| 6 | **In-flight work when budget exceeded** | Medium | Low | K8s pattern: in-flight completes, next rejected. Track overage. |

---

## 13. Dependencies

| Dependency | Direction | Notes |
|---|---|---|
| PRD-101 (Mission Schema) | Uses | `budget_config`, `budget_spent` JSONB on orchestration_runs |
| PRD-102 (Coordinator) | Uses | Coordinator calls budget gate before spawning tasks |
| PRD-103 (Verification) | Uses | Verification cost tracked separately within budget |
| PRD-104 (Contractors) | Uses | Contractors inherit mission budget constraints |
| PRD-106 (Telemetry) | Feeds | Budget utilization is a telemetry dimension |
| `UsageTracker` | Extension | Add `mission_task_id` to tracking path |
| `LLMManager` | Extension | Insert budget check before `_call_provider` |
| `tool_router.py` | Extension | Add policy intersection to `get_tools_for_agent()` |

---

## Appendix: Research Sources

| Source | What It Informed |
|--------|-----------------|
| OpenClaw 8-stage tool policy (docs.openclaw.ai) | Monotonic narrowing, enforcement at tool-set construction |
| K8s ResourceQuota + admission control (kubernetes.io) | Synchronous hard rejection, two-layer limits, quota scopes |
| AWS Budgets API | Graduated thresholds, AUTOMATIC/MANUAL actions, cost allocation |
| Token bucket (Anthropic/Stripe) | Cost-denominated bucket with no refill for fixed budgets |
| LiteLLM BudgetManager | `projected_cost()` + `update_cost()` two-phase pattern |
| RouteLLM (ICLR 2025) | 75% cost reduction validates model downgrade strategy |
| Automatos UsageTracker | Existing cost recording path to extend |
| Automatos config.py | PREMIUM_MODELS, BUDGET_MODELS for downgrade targets |
| Automatos Workspace.plan_limits | Existing unwired JSONB field to activate |
