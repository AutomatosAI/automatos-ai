# Loop 0: Design Outlines for PRDs 101-108

## Progress

- [x] US-001: Design PRD-101 outline (Mission Schema & Data Model)
- [x] US-002: Design PRD-102 outline (Coordinator Architecture)
- [x] US-003: Design PRD-103 outline (Verification & Quality)
- [x] US-004: Design PRD-104 outline (Ephemeral Agents & Model Selection)
- [x] US-005: Design PRD-105 outline (Budget & Governance)
- [x] US-006: Design PRD-106 outline (Outcome Telemetry & Learning)
- [x] US-007: Design PRD-107 outline (Context Interface Abstraction)
- [ ] US-008: Design PRD-108 outline (Memory Field Prototype)

## Discoveries

- Temporal and Dagster both use full event sourcing; we adopt the hybrid approach (CRUD + audit events) for simplicity
- Prefect's `task_inputs` JSONB pattern is ideal for our scale (3-20 tasks per mission) — avoids separate edges table
- Symphony's continuation vs retry distinction is critical for AI agents and should be adopted
- Dagster's `run_tags` table pattern keeps schema stable while enabling rich metadata
- Existing `board_tasks.planning_data` JSONB and `source_type` column were designed for this extension
- `board_tasks.parent_task_id` already supports subtask trees — missions can use this
- `task_reconciler.py` currently only handles `recipe_executions` — must extend for missions

- LbMAS (2025) validates blackboard pattern for LLM multi-agent: 5% improvement over static systems, fewer tokens than autonomous workflow search
- ChatHTN (2025) proves LLM+symbolic hybrid is sound: LLM fills gaps in method library, symbolic engine validates
- Hsiao et al. (2025): hand-coded HTNs enable 20-70B models to outperform 120B baselines — structure dramatically improves LLM planning
- ChatBDI (AAMAS 2025): BDI provides interpretability/safety that raw LLM agents lack — "Think BDI, Talk LLM"
- Symphony's two-phase tick (dispatch + reconcile) is the cleanest coordination loop pattern found
- CrewAI's hierarchical mode uses LLM-as-manager which is non-deterministic and untestable — reject this, use deterministic agent matching
- LangGraph's typed state + checkpoint model is the right durability pattern — our DB schema (PRD-101) already implements this
- AutoGen's Swarm handoff pattern (agents explicitly name next agent) is interesting but wrong for our case — coordinator should always decide
- `_orchestrator_tick_llm` (heartbeat_service.py:382) is the structural template for coordinator tick — same skeleton, different prompt/tools/budget
- `AgentCommunicationProtocol` and `CollaborativeReasoner` exist but are not wired into live code paths — coordinator could activate these
- `SharedContextManager` has 2h Redis TTL — insufficient for multi-day missions, needs DB persistence

- Rubric-based absolute scoring is more stable than pairwise (9% vs 35% flip rate) — Zheng et al. 2023
- Self-preference bias is empirically proven — MUST use different model for verification than execution
- Deterministic-first pattern (OpenAI Evals): check format/length/schema before expensive LLM judge calls
- DeepEval's DAG (decision-tree evaluation) maps naturally to success criteria checking pipelines
- FutureAGI live traffic scoring (`POST /score`) is the existing pattern to extend — add `POST /verify-task`
- Three existing quality scoring systems (recipe_quality, quality_assessor, report grading) overlap — PRD-103 must clarify boundaries
- Verification cost budget: 10-30% of task generation cost (industry benchmark)
- RAGAS achieves 95% human agreement on faithfulness — useful for RAG-heavy mission tasks
- DeepEval supports ~14 pre-built metrics + custom DAG metrics; RAGAS specializes in RAG with ~8 metrics

- ContextService has 8 modes, 12 sections, all rendered in parallel via asyncio.gather — adapter wraps this without changing internals
- `ContextResult` (frozen dataclass) is the current public contract — `AgentContext` in ports module becomes the new one
- 4 callers directly import ContextService: smart_orchestrator, heartbeat_service, recipe_executor, routing/engine — all must migrate to ContextProvider port
- SharedContextManager has 3 merge strategies (override, append, consensus) + 3 consensus methods (majority, average, union) — all preserved behind SharedContextPort
- AgentCommunicationProtocol and CollaborativeReasoner exist in inter_agent.py but are NOT wired into live code paths — coordinator (PRD-102) could activate these
- LangGraph's two-tier model (checkpointer + store) validates our two-port design: ContextProvider (per-agent) + SharedContextPort (cross-agent)
- AutoGen's `update_context()` preprocessor pattern confirms the approach: memory/shared context injects as system message
- Context Engineering theory requires resonance-based query primitive ("what resonates with X?") — different from traditional retrieval, but same interface shape
- Non-commutativity of field operations (inject order matters) — interface must preserve operation ordering
- BaseSection ABC already exists with `async def render(self, ctx: SectionContext) -> str` — good precedent for port pattern

## Cross-PRD Dependencies Found

- PRD-102 (Coordinator) will need `COORDINATOR` context mode in ContextService — note for PRD-107
- PRD-103 (Verification) needs `success_criteria` JSONB schema defined in PRD-101's mission_tasks
- PRD-104 (Ephemeral Agents) needs `contractor_config` JSONB schema defined in PRD-101's mission_tasks
- PRD-105 (Budget) needs `budget_config`/`budget_spent` JSONB schemas from PRD-101's mission_runs
- PRD-106 (Telemetry) will query mission_events + cost/token fields — schema must support efficient aggregation
- Existing `llm_usage` table tracks per-call costs — mission tasks should reference or aggregate these

- `AgentFactory.execute_with_prompt()` already accepts `AgentRuntime` directly (line 711) — zero changes needed for contractor execution path
- Agent Zero conversation sealing (progressive compression: 50% current / 30% topics / 20% bulks) and utility model pattern should be adopted
- AutoGen's agent-as-config-dict pattern `(name, system_message, llm_config, tools, description)` is the contractor config model
- K8s Job lifecycle patterns (TTL cleanup, activeDeadlineSeconds, podFailurePolicy) map directly to contractor lifecycle
- RouteLLM (ICLR 2025): 75% cost reduction at 95% quality with static role→model mapping — validates our approach
- BudgetMLAgent: cascade pattern (free → cheap → expensive) achieves 96% cost reduction — relevant to PRD-105
- OpenRouter's `sort`/`max_price`/`preferred_min_throughput` provider params are the v1 model selection interface
- Cognitive diversity is a hard constraint: reviewer model family MUST differ from coder model family
- Hybrid contractor creation recommended: in-memory AgentRuntime for speed + async DB audit row for observability
- `agents` table has 45+ columns — contractors only need ~8 fields. Consider minimal DB row or in-memory-only mode
- Existing `inter_agent.py` AgentCommunicationProtocol and CollaborativeReasoner are unwired — future option for contractor coordination

- OpenClaw actually has 8 policy stages (not 6 as stated in PRD-100) — all monotonically narrowing, deny always wins
- OpenClaw enforces at tool-set construction (tools passed to LLM) — same as our `get_tools_for_agent()` pattern
- AWS Budgets has NO true hard cap — 8-12 hour billing lag makes it useless for real-time LLM enforcement
- K8s admission control pattern is the right model: synchronous pre-call check, hard rejection, in-flight work completes
- Two `TokenBudgetManager` classes exist — `modules/context/budget.py` (context-window) and `stages/token_budget_manager.py` (workflow, broken)
- `stages/token_budget_manager.py` has latent `AttributeError` — references `config.TOKEN_BUDGET_DEFAULT` etc. that don't exist in `config.py`
- `Workspace.plan_limits` JSONB exists but is completely unwired — the enforcement hook is already in the schema
- `llm_usage` table has no `mission_id` column — must add for per-mission cost attribution
- `UsageTracker` is fire-and-forget (post-call only) — budget enforcement needs a pre-call admission gate in `LLMManager`
- LiteLLM BudgetManager's `projected_cost()` + `update_cost()` two-phase pattern is the reference implementation
- Anthropic uses token bucket natively; cached input tokens excluded from ITPM — relevant for mission cost optimization
- Cost-denominated token bucket with disabled refill is the right algorithm for fixed-budget missions

- MLflow's three-tier storage (metrics=append-only, params=immutable, tags=mutable) maps cleanly to mission telemetry: config fields (agent, model, task_type) are immutable, outcome fields (score, acceptance) are mutable post-execution
- W&B summary/history split is the right pattern: summary columns on mission_tasks for dashboards, event log table for deep analysis
- `llm_usage` table is the foundation — adding nullable `mission_task_id` FK is the lowest-friction integration path
- `heartbeat_results.cost` column exists but is never populated — quick fix opportunity
- Prometheus agent metrics (`automatos_metrics.py`) are defined but never `.inc()`'d — wire them into UsageTracker
- Eppo/Statsig pattern: store `metric_sum` + `metric_sum_squares` for variance computation without raw data re-read
- Propensity logging (`action_probability`) not needed in v1 but schema should not preclude it for future bandit evaluation
- OpenTelemetry `gen_ai.*` semantic conventions are emerging — worth adopting attribute naming even without OTel transport
- 10+ existing telemetry touchpoints already in codebase — extend, don't duplicate
