# PRD-123 — Harness Pattern Adoption

**Version:** 1.2
**Type:** Implementation
**Status:** Draft
**Priority:** P1
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-31
**Extends:** PRD-82A (Coordinator), PRD-79 (Memory), PRD-35 (Tools), PRD-55 (Channels), PRD-58 (Prompt Management)
**Touches:** PRD-77 (Scheduled Tasks), PRD-06 (Dashboard), PRD-37 (API Keys), PRD-29 (FutureAGI)
**Research Base:** Claude Code harness pattern analysis (instructkr/claude-code Python port)

---

## 1. Goal

Adopt 12 architectural patterns, 3 prompt engineering patterns, and 2 system-awareness features identified from studying the Claude Code agent harness and Automatos onboarding gaps. The architectural patterns address runtime gaps (silent permission failures, ambiguous stop reasons, monolithic startup). The prompt patterns address intelligence gaps — the LLM makes poor memory decisions, gets minimal tool-use guidance, and has no anti-pattern awareness for business operations. The system-awareness features address a critical UX gap — Auto doesn't know what users can accomplish with the platform and has no guided onboarding for new workspaces.

This PRD hardens platform internals, improves agent intelligence via prompt content, **and** introduces a goal-oriented platform awareness prompt plus a dynamic Mission Zero onboarding flow — all delivered through the existing ContextService section architecture and PromptRegistry (PRD-58).

## 2. Research Context

The patterns were extracted from `instructkr/claude-code`, a Python clean-room port of the Claude Code TypeScript harness. The analysis compared each pattern against the current Automatos codebase to identify gaps, adaptation strategies, and expected impact.

Full research saved at: `claude-code/` repo memory (`research_claude_code_patterns.md`).

**Key insight:** Claude Code is a single-user CLI tool. Automatos is a multi-tenant platform with orchestration. The patterns that matter most are the ones that scale — structured denials, named stop reasons, trust gates, and typed events all become *more* valuable in multi-agent, multi-tenant contexts.

## 3. What Ships

| # | Pattern | Phase | Files Touched | Depends On |
|---|---------|-------|---------------|------------|
| 5 | Permission Denial as First-Class Data | Quick Win | 4-6 | None |
| 6 | Named Stop Reasons | Quick Win | 3-5 | None |
| 2 | Trust-Gated Initialization | Quick Win | 1-2 | None |
| 10 | Bootstrap Named Stages | Quick Win | 2-3 | None |
| 4 | Tool Tier Stratification | Next Sprint | 5-8 | #5 |
| 11 | Streaming Typed Events | Next Sprint | 8-12 | None |
| 1 | Frozen State Transition Models | Next Sprint | 10-15 | #6 |
| 7 | Proactive Transcript Compaction | Next Sprint | 2-4 | None |
| 8 | Session Checkpointing | Backlog | 4-6 | #1 |
| 3 | Tool Manifest Snapshots | Backlog | 2-3 | #4 |
| 9 | PRD Parity Audit | Backlog | 3-5 | None |
| 12 | Tool Execution Cost Tracking | Backlog | 2-3 | None |
| **E** | **Memory Decision Framework** | **Quick Win** | **2-3** | **None** |
| **B** | **Business Tool Behavioral Contracts** | **Quick Win** | **3-5** | **None** |
| **F** | **Section-Level Anti-Patterns** | **Quick Win** | **1-2** | **E, B** |
| **H** | **Platform Awareness Prompt** | **Quick Win** | **2-3** | **None** |
| **I** | **Mission Zero Onboarding** | **Next Sprint** | **5-8** | **H** |

> [!IMPORTANT]
> Patterns A (Sectioned Prompts) and D (Skills as Prompt Expansion) were identified in research but **already exist** in Automatos. `ContextService` + `SECTION_REGISTRY` implements sectioned assembly with 14 sections, 9 modes, parallel rendering, and priority-based token trimming. `SkillsSection` (Priority 4) already injects SKILL.md content into the system prompt. Pattern G (Prompt-Level Cost Awareness) is deferred — users configure agent LLMs at design time; on-the-fly model switching is a separate feature. Pattern C (Conditional Context Injection) is deferred — the section system handles this adequately for now.

## 4. What Does NOT Ship (Deferred)

| Deferred | Why |
|----------|-----|
| Full event-sourcing migration | Too large; frozen models + events are sufficient for now |
| CQRS read/write separation | Premature; current DB load doesn't warrant it |
| Tool sandboxing / runtime isolation | Separate security PRD needed |
| Frontend UI changes for new events | Frontend PRD follows after backend ships |
| Pattern A: Sectioned Prompt Assembly | **Already built.** ContextService + SECTION_REGISTRY with 14 sections, 9 modes |
| Pattern D: Skills as Prompt Expansion | **Already built.** SkillsSection (P4) injects SKILL.md into system prompt |
| Pattern G: On-the-fly LLM switching | Users configure agent LLMs at design time; different feature scope |
| Pattern C: Mid-conversation context injection | Current section system handles this; revisit when needed |

---

## 5. Phase 1 — Quick Wins

### 5.1 Pattern #5: Permission Denial as First-Class Data

#### Problem

When the coordinator assigns a task to an agent and that agent lacks access to a required tool, the failure is silent. The tool simply doesn't appear in the agent's available tools. The coordinator sees "task failed" with no explanation of *why*. Debugging requires manually cross-referencing `agent_tool_assignments` against the task requirements.

#### What Claude Code Does

Every permission check produces a `PermissionDenial(tool_name, reason)` frozen dataclass. These denials flow through `TurnResult` and are surfaced in the query engine output. The system always knows what was blocked and why.

#### What Automatos Does Today

- `agent_tool_assignments` controls tool access, but denial is silent (tool excluded from list)
- Workspace role checks return HTTP 403 with generic message
- `OrchestrationEvent` tracks task lifecycle but not permission failures
- No structured denial data in the orchestration flow

#### Design

**New dataclass:**

```python
@dataclass(frozen=True)
class PermissionDenial:
    """Immutable record of a denied tool/action access."""
    agent_id: int
    tool_name: str
    reason: str
    workspace_id: UUID
    denied_at: datetime
    context: str  # 'orchestration', 'chat', 'workflow'
```

**Emission points:**

1. **Tool resolution** (`function_registry.py`) — when agent requests a tool not in their assignments
2. **Coordinator dispatch** (`coordinator_service.py`) — when task requires a tool the matched agent can't access
3. **Chat execution** (`chat_service.py`) — when user-facing agent hits a tool boundary

**Storage:**

Emit as `OrchestrationEvent` with `event_type='permission_denied'`:

```python
await emit_event(
    run_id=run.id,
    task_id=task.id,
    event_type='permission_denied',
    actor_type='system',
    actor_id='tool_resolver',
    details={
        'agent_id': agent.id,
        'agent_name': agent.name,
        'tool_name': required_tool,
        'reason': 'Tool not assigned to agent',
        'resolution_hint': 'Assign tool via workspace settings or reassign task to agent with access',
    }
)
```

**Self-healing in coordinator:**

When a denial is recorded during dispatch, the coordinator should attempt reassignment:

```python
async def dispatch_task(task: OrchestrationTask, run: OrchestrationRun):
    agent = await match_agent(task)
    required_tools = extract_tool_requirements(task)

    denials = check_tool_access(agent, required_tools)
    if denials:
        for denial in denials:
            await emit_permission_denial_event(denial)

        # Try next-best agent
        agent = await match_agent(task, exclude=[agent.id])
        if not agent:
            await transition_task(task, 'assignment_failed',
                reason=f"No agent has access to: {[d.tool_name for d in denials]}")
            return
```

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/models/orchestration.py` | Add `PermissionDenial` dataclass |
| `orchestrator/core/llm/function_registry.py` | Emit denial when tool not found in assignments |
| `orchestrator/core/services/coordinator_service.py` | Check tool access before dispatch, attempt reassignment |
| `orchestrator/core/services/state_machine_service.py` | Add `permission_denied` event type |
| `orchestrator/api/missions.py` | Include denials in mission status response |

#### Acceptance Criteria

1. When an agent is denied a tool, a `PermissionDenial` record is created (not just logged)
2. `OrchestrationEvent` with `event_type='permission_denied'` appears in mission event trail
3. Coordinator attempts reassignment before failing the task
4. `GET /api/missions/{id}` response includes `permission_denials` array
5. No existing tests break

---

### 5.2 Pattern #6: Named Stop Reasons

#### Problem

When an `OrchestrationRun` ends, the state is either `completed` or `failed`. There's no distinction between "ran out of budget," "hit max retries," "human cancelled," or "all tasks succeeded." Users see "failed" and have to dig through events to understand why. The `token_budget_estimate` field exists but isn't enforced as a stop condition.

#### What Claude Code Does

`QueryEngineConfig` defines `max_turns` and `max_budget_tokens`. When the engine stops, it returns a named `stop_reason`: `completed`, `max_turns_reached`, or `max_budget_reached`. Every exit is explicit.

#### What Automatos Does Today

- `RunState` has `completed`, `failed`, `cancelled` as terminal states
- `token_budget_estimate` and `tokens_used` exist on `OrchestrationRun` but budget is not enforced
- `max_retries` exists but exhaustion just sets state to `failed`
- No `stop_reason` field — you have to infer from events

#### Design

**New enum:**

```python
class StopReason(str, Enum):
    """Why a mission stopped. Set on terminal transition."""
    COMPLETED = 'completed'                    # All tasks verified successfully
    BUDGET_EXHAUSTED = 'budget_exhausted'      # tokens_used >= token_budget_estimate
    MAX_RETRIES_EXCEEDED = 'max_retries'       # Task hit retry limit, no recovery
    TIMEOUT = 'timeout'                        # Wall-clock deadline exceeded
    HUMAN_CANCELLED = 'human_cancelled'        # User cancelled via API/UI
    COORDINATOR_ERROR = 'coordinator_error'    # Internal coordinator failure
    NO_CAPABLE_AGENT = 'no_capable_agent'      # No agent matched task requirements
    DEPENDENCY_FAILED = 'dependency_failed'    # Upstream task failed, downstream can't run
```

**Schema change:**

```sql
ALTER TABLE orchestration_runs ADD COLUMN IF NOT EXISTS stop_reason VARCHAR(50);
ALTER TABLE orchestration_runs ADD COLUMN IF NOT EXISTS stop_detail TEXT;
```

**Enforcement points:**

```python
# In coordinator tick loop:
async def coordinator_tick(run: OrchestrationRun):
    # Budget check BEFORE dispatching next task
    if run.tokens_used >= run.token_budget_estimate:
        await transition_run(run, RunState.FAILED,
            stop_reason=StopReason.BUDGET_EXHAUSTED,
            stop_detail=f"Used {run.tokens_used} of {run.token_budget_estimate} budget tokens")
        return

    # Retry exhaustion
    if task.retry_count >= run.max_retries:
        await transition_run(run, RunState.FAILED,
            stop_reason=StopReason.MAX_RETRIES_EXCEEDED,
            stop_detail=f"Task '{task.title}' failed after {run.max_retries} retries")
        return

    # All tasks verified
    if all_tasks_terminal(run):
        if all_tasks_verified(run):
            await transition_run(run, RunState.COMPLETED,
                stop_reason=StopReason.COMPLETED)
        else:
            await transition_run(run, RunState.FAILED,
                stop_reason=StopReason.DEPENDENCY_FAILED,
                stop_detail=f"Tasks failed: {failed_task_names}")
        return
```

**API response enrichment:**

```json
{
  "id": "run-uuid",
  "state": "failed",
  "stop_reason": "budget_exhausted",
  "stop_detail": "Used 15,230 of 10,000 budget tokens",
  "tokens_used": 15230,
  "token_budget_estimate": 10000
}
```

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/models/orchestration.py` | Add `StopReason` enum, `stop_reason` + `stop_detail` columns |
| `orchestrator/core/services/coordinator_service.py` | Enforce budget, set stop_reason at every exit point |
| `orchestrator/core/services/state_machine_service.py` | Accept `stop_reason` in `transition_run()` |
| `orchestrator/api/missions.py` | Include `stop_reason` and `stop_detail` in responses |
| `orchestrator/core/database/init_database.py` | Migration: add columns |

#### Acceptance Criteria

1. Every terminal `OrchestrationRun` has a non-null `stop_reason`
2. Budget enforcement: run stops with `BUDGET_EXHAUSTED` when `tokens_used >= token_budget_estimate`
3. `GET /api/missions/{id}` includes `stop_reason` and `stop_detail`
4. Retry exhaustion produces `MAX_RETRIES_EXCEEDED` (not generic `failed`)
5. Human cancel via `POST /api/missions/{id}/cancel` sets `HUMAN_CANCELLED`

---

### 5.3 Pattern #2: Trust-Gated Initialization

#### Problem

Automatos boots in a single phase. DB schema, seeding, scheduler, channels, Composio sync, Git-backed skill loading — all run in one `lifespan` block. If a third-party Git skill repo is compromised, its code runs at startup alongside core database initialization. If a channel OAuth token is expired, the error can delay or break the entire startup sequence.

#### What Claude Code Does

Boot is split into phases with an explicit trust gate between core initialization and extension loading. Plugins, skills, MCP servers, and hooks only load *after* the trust gate passes. This is tracked via `DeferredInitResult(plugins_loaded, skills_loaded, mcp_connected, hooks_registered)`.

#### What Automatos Does Today

- `main.py` lifespan runs everything linearly
- No separation between core (DB, config) and extensions (skills, channels, Composio)
- If `start_all_channels()` throws, it can crash the startup
- Skill source seeding (`seed_skill_sources`) runs before health checks

#### Design

**Two-phase boot with trust gate:**

```python
# Phase 1: TRUSTED CORE (must succeed)
async def boot_phase_1_core(app):
    """Core platform — no third-party code, no network calls to external services."""
    await init_database()           # Schema + migrations
    await seed_system_prompts()     # PRD-58
    await seed_system_agents()      # PRD-67
    await verify_credentials()     # Check DB has required provider keys
    app.state.core_ready = True
    logger.info("Phase 1 complete: core ready")

# TRUST GATE
async def trust_gate(app):
    """Verify preconditions before loading extensions."""
    checks = {
        'database': app.state.core_ready,
        'credentials': await check_provider_credentials(),
        'config': validate_feature_flags(),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        logger.error(f"Trust gate failed: {failed}")
        app.state.trust_passed = False
        return  # Platform runs in degraded mode (core only)
    app.state.trust_passed = True

# Phase 2: EXTENSIONS (can fail gracefully)
async def boot_phase_2_extensions(app):
    """Third-party integrations — isolated failures don't crash core."""
    if not app.state.trust_passed:
        logger.warning("Skipping extensions: trust gate failed")
        return

    results = DeferredInitResult()

    # Each extension wrapped in try/except — failure is logged, not fatal
    try:
        await seed_skill_sources()
        results.skills_loaded = True
    except Exception as e:
        logger.error(f"Skill loading failed: {e}")
        results.skills_loaded = False

    try:
        await start_composio_sync()
        results.tools_synced = True
    except Exception as e:
        logger.error(f"Composio sync failed: {e}")
        results.tools_synced = False

    try:
        await start_all_channels()
        results.channels_connected = True
    except Exception as e:
        logger.error(f"Channel startup failed: {e}")
        results.channels_connected = False

    try:
        await start_unified_scheduler()
        results.scheduler_started = True
    except Exception as e:
        logger.error(f"Scheduler failed: {e}")
        results.scheduler_started = False

    app.state.deferred_init = results
```

**New dataclass:**

```python
@dataclass
class DeferredInitResult:
    """Tracks which extensions loaded successfully."""
    skills_loaded: bool = False
    tools_synced: bool = False
    channels_connected: bool = False
    scheduler_started: bool = False
    embeddings_computed: bool = False

    @property
    def all_healthy(self) -> bool:
        return all([
            self.skills_loaded,
            self.tools_synced,
            self.channels_connected,
            self.scheduler_started,
        ])

    def as_dict(self) -> dict:
        return {
            'skills_loaded': self.skills_loaded,
            'tools_synced': self.tools_synced,
            'channels_connected': self.channels_connected,
            'scheduler_started': self.scheduler_started,
            'embeddings_computed': self.embeddings_computed,
            'all_healthy': self.all_healthy,
        }
```

**Health endpoint update:**

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if app.state.trust_passed else "degraded",
        "core_ready": app.state.core_ready,
        "trust_passed": app.state.trust_passed,
        "extensions": app.state.deferred_init.as_dict()
            if hasattr(app.state, 'deferred_init') else None,
    }
```

#### Files to Change

| File | Change |
|------|--------|
| `main.py` | Split lifespan into `boot_phase_1_core`, `trust_gate`, `boot_phase_2_extensions` |
| `orchestrator/core/models/` (new file) | Add `DeferredInitResult` dataclass |
| `orchestrator/api/health.py` or `main.py` | Update `/health` to include extension status |

#### Acceptance Criteria

1. If Composio sync fails, the platform still starts (degraded mode)
2. If channel OAuth is expired, core API endpoints still work
3. `/health` reports which extensions loaded and which failed
4. No third-party code executes before `trust_gate` passes
5. Startup logs clearly show Phase 1 / Trust Gate / Phase 2 boundaries

---

### 5.4 Pattern #10: Bootstrap Named Stages

#### Problem

Startup is a linear block in `main.py`. If seeding fails at line 47 of a 120-line function, the error says "startup failed" with a traceback. In production (Railway), diagnosing which *phase* failed requires scrolling logs. There's no way to ask "did the scheduler start?" without checking log text.

#### What Claude Code Does

`BootstrapGraph` defines ordered named stages with descriptions. Each stage is individually reportable. The system can render a full bootstrap report showing what completed and what didn't.

#### What Automatos Does Today

- Monolithic `lifespan` function in `main.py`
- Stages exist implicitly but aren't named, timed, or reportable
- No `/health/bootstrap` endpoint

#### Design

**Named stages enum + timing:**

```python
class BootstrapStage(str, Enum):
    CONFIG_LOAD = 'config_load'
    DATABASE_INIT = 'database_init'
    SCHEMA_MIGRATION = 'schema_migration'
    SEED_SYSTEM_DATA = 'seed_system_data'
    TRUST_GATE = 'trust_gate'
    SKILL_SOURCES = 'skill_sources'
    TOOL_SYNC = 'tool_sync'
    SEMANTIC_EMBEDDINGS = 'semantic_embeddings'
    SCHEDULER_INIT = 'scheduler_init'
    CHANNEL_CONNECT = 'channel_connect'
    READY = 'ready'

@dataclass(frozen=True)
class StageResult:
    stage: BootstrapStage
    status: str          # 'success', 'failed', 'skipped'
    duration_ms: int
    error: str | None = None

@dataclass
class BootstrapReport:
    stages: list[StageResult]
    started_at: datetime
    ready_at: datetime | None = None

    @property
    def total_duration_ms(self) -> int:
        return sum(s.duration_ms for s in self.stages)

    @property
    def failed_stages(self) -> list[StageResult]:
        return [s for s in self.stages if s.status == 'failed']

    def as_dict(self) -> dict:
        return {
            'started_at': self.started_at.isoformat(),
            'ready_at': self.ready_at.isoformat() if self.ready_at else None,
            'total_duration_ms': self.total_duration_ms,
            'stages': [
                {
                    'stage': s.stage.value,
                    'status': s.status,
                    'duration_ms': s.duration_ms,
                    'error': s.error,
                }
                for s in self.stages
            ],
            'failed': [s.stage.value for s in self.failed_stages],
        }
```

**Execution wrapper:**

```python
async def run_stage(report: BootstrapReport, stage: BootstrapStage, func):
    """Execute a bootstrap stage with timing and error capture."""
    start = time.monotonic()
    try:
        await func()
        elapsed = int((time.monotonic() - start) * 1000)
        report.stages.append(StageResult(stage, 'success', elapsed))
        logger.info(f"Bootstrap [{stage.value}] completed in {elapsed}ms")
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        report.stages.append(StageResult(stage, 'failed', elapsed, str(e)))
        logger.error(f"Bootstrap [{stage.value}] failed after {elapsed}ms: {e}")
```

**Endpoint:**

```python
@app.get("/health/bootstrap")
async def bootstrap_status():
    """Returns detailed bootstrap stage report."""
    return app.state.bootstrap_report.as_dict()
```

#### Files to Change

| File | Change |
|------|--------|
| `main.py` | Wrap each startup task in `run_stage()`, store `BootstrapReport` on `app.state` |
| `orchestrator/core/models/` (new file or extend) | Add `BootstrapStage`, `StageResult`, `BootstrapReport` |
| `main.py` or `orchestrator/api/health.py` | Add `/health/bootstrap` endpoint |

#### Acceptance Criteria

1. Every startup phase is named and timed in logs: `Bootstrap [database_init] completed in 340ms`
2. `GET /health/bootstrap` returns full stage report with durations
3. Failed stages show error message in the report
4. Total startup time visible in one API call

---

### 5.5 Pattern #E: Memory Decision Framework (Prompt Content)

#### Problem

The LLM gets 3 lines of memory guidance in `get_self_learning_instruction()`:

```
- Workspace discoveries (what agents exist, what's configured, user preferences)
- Task outcomes (what worked, what failed, what the user liked)
- Platform patterns (common requests, effective tool combinations)
```

The result: the model stores garbage. Real examples from production Mem0:

- `"Appreciates clean, modern, minimal aesthetic in design"` — vague, no context about when/where this applies
- `"Need a blog image for a post titled 'The Top 5 AI Agent Frameworks in 2025'"` — ephemeral task, not a memory
- `"On the left: 3 abstract agent nodes in sequence (A → B → C) passing fragmented..."` — raw artifact content, not a fact

The backend auto-saves ~95% of memories via `SmartMemoryManager.store_conversation()`. But `platform_store_memory` is the LLM's only way to intentionally store high-quality, curated facts. It needs a real decision framework, not a 3-bullet nudge.

#### What Claude Code Does

The memory system prompt is ~2000 words. It defines 4 named memory types (user, feedback, project, reference), each with:
- Description of what belongs in this type
- **When to save** — specific trigger conditions
- **When to use** — retrieval conditions
- **Examples** — input → memory action pairs
- **Anti-patterns** — what NOT to save
- **Body structure** — how to format the memory content

#### What Automatos Does Today

- `get_self_learning_instruction()` in `personality.py` (lines 300-313): 3 bullets, ~60 words
- `platform_store_memory` tool description: `"Store a piece of information in the workspace memory system."` — 10 words
- No memory type taxonomy, no examples, no anti-patterns, no format guidance
- Backend `SmartMemoryManager` classifies as "global" or "agent-specific" — but the LLM doesn't know these categories exist

#### Design

Replace `get_self_learning_instruction()` with a full memory decision framework. Delivered as a new PromptRegistry version for the relevant chatbot personality slugs.

**New prompt content (~800 tokens, replaces ~60):**

```markdown
## Memory — What to Remember

I have a memory system. When I learn something worth keeping, I store it using
`platform_store_memory`. Not everything is worth storing — I'm selective.

### Memory Types

**User Facts** — Who this person is, what they care about, how they work.
- When to save: User shares their role, team, goals, preferences, or constraints.
- Format: Start with the fact, then context. "CFO of a 12-person SaaS startup. Cares about burn rate and runway."
- NOT: Greetings, single-turn task requests, or things I can infer from the workspace.

**Decisions & Outcomes** — What was decided, what worked, what didn't.
- When to save: A task completes and the user confirms the result was good (or bad). A strategy is chosen. A workflow is approved.
- Format: Lead with the decision, then the outcome. "Chose weekly email digest over daily — user said daily was too noisy."
- NOT: Intermediate steps, failed tool calls, or routine completions.

**Workspace Knowledge** — How this workspace is set up, what the patterns are.
- When to save: I discover something about the workspace that took effort to find — which agents handle what, naming conventions, recurring workflows, integration details.
- Format: Lead with the fact, then where it applies. "Marketing reports go to the #growth Slack channel every Monday via the Reports Agent."
- NOT: Things already in the agent config, skill descriptions, or tool schemas.

**Preferences & Corrections** — How the user wants things done.
- When to save: User corrects my approach, confirms a non-obvious choice, or states a preference for future interactions.
- Format: Lead with the rule, then why. "Always include cost estimates in proposals — user's CEO requires them."
- NOT: One-time formatting requests or trivial style preferences.

### What I Never Store

- Raw tool outputs, JSON, or API responses
- The content of documents, emails, or messages (store the *takeaway*, not the text)
- Anything I'd need to update every day (volatile metrics, counts, statuses)
- Speculative context ("user might want...") — only store confirmed facts
- Task artifacts (generated images, drafted emails) — these live in the conversation, not memory

### How I Decide

Before calling `platform_store_memory`, I ask myself:
1. Will this help me in a **future conversation**, not just this one?
2. Is this a **fact** or just a **task detail**?
3. Could I find this by checking the workspace config? If yes, don't store it.
4. Would storing this make my next interaction **noticeably better**?

If all four answers aren't yes, I skip the store.
```

**Tool description upgrade for `platform_store_memory`:**

Current (in `actions_workspace.py`):
```python
description="Store a piece of information in the workspace memory system."
```

New:
```python
description=(
    "Store a curated fact in workspace memory for future conversations. "
    "Use for: user facts, confirmed decisions, workspace patterns, and user corrections. "
    "Do NOT use for: task artifacts, raw tool outputs, volatile data, or speculative context. "
    "Format: lead with the fact, then context. Keep under 200 characters."
)
```

#### Delivery Mechanism

1. Replace `get_self_learning_instruction()` in `personality.py` with the new framework text
2. Create a new PromptRegistry version for `chatbot-friendly`, `chatbot-professional`, `chatbot-technical` that includes the memory framework
3. Update `platform_store_memory` tool description in `actions_workspace.py`
4. Use FutureAGI (PRD-29) to A/B test: run `assess` on conversations before/after the change to measure memory quality improvement

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/consumers/chatbot/personality.py` | Replace `get_self_learning_instruction()` (~60 words → ~800 words) |
| `orchestrator/modules/tools/discovery/actions_workspace.py` | Upgrade `platform_store_memory` description (10 words → 50 words) |
| `orchestrator/core/seeds/seed_system_prompts.py` | Update PROMPT_MANIFEST entries for chatbot-* slugs to include memory framework |

> [!NOTE]
> The backend `SmartMemoryManager` auto-store logic is unchanged. This pattern only improves what the LLM **intentionally** stores via `platform_store_memory`. Over time, as the LLM stores better memories, the auto-stored conversation data becomes less important relative to curated facts.

#### Acceptance Criteria

1. `get_self_learning_instruction()` returns the full memory decision framework (~800 tokens)
2. `platform_store_memory` tool description includes type guidance and anti-patterns
3. New PromptRegistry versions created for all three chatbot personality slugs
4. FutureAGI `assess` run on 10 sample conversations shows improvement in `is_helpful` score
5. After 1 week of production use, sample 50 new Mem0 entries — fewer than 20% should be ephemeral task artifacts (baseline: ~60%)

---

### 5.6 Pattern #B: Business Tool Behavioral Contracts (Prompt Content)

#### Problem

Tool descriptions are minimal. `platform_execute` has a 1-line description. `composio_execute` lists available actions but gives no behavioral guidance. The LLM doesn't know:

- When to use a tool vs. just answering from knowledge
- How to handle tool failures gracefully
- What information to include in tool calls (e.g., always include workspace context for Slack)
- What NOT to do with tools (e.g., don't search the knowledge base for "good morning")

Claude Code's Bash tool description is ~1500 words of behavioral rules. Automatos is not a coding tool, but the same principle applies — **rich tool descriptions produce expert tool usage**.

#### What Automatos Does Today

- `get_tool_guidance_prompt()` in `personality.py` (lines 245-269): 3 bullets, ~40 words
- `PlatformActionsSection` (Priority 5): renders `ActionRegistry.build_prompt_summary()` — a markdown catalog of action names and 1-line descriptions
- `ComposioSection`: lists available Composio actions by name
- No behavioral rules, no anti-patterns, no workflow guidance for any tool

#### Design

Upgrade `get_tool_guidance_prompt()` with business-focused behavioral contracts. This is not coding-specific — it's about how an AI assistant should use tools when running a business.

**New prompt content (~600 tokens, replaces ~40):**

```markdown
## Tools — How I Use Them

### When I Reach for Tools

- **Action requests** ("send an email", "create an agent", "check Slack") → Use the tool immediately, then confirm what I did.
- **Information requests** ("what agents do we have?", "show me costs") → Use platform_execute or knowledge search, then summarize in plain language.
- **Conversations** ("good morning", "what do you think about X?") → Just talk. No tool calls for greetings, opinions, or brainstorming.
- **Ambiguous requests** ("help with marketing") → Clarify the goal first, then pick tools. Don't spray tool calls hoping something sticks.

### How I Use Tools Well

- **One tool at a time** unless the task clearly requires multiple. Chain results, don't blast parallel calls.
- **Include context** in every tool call — workspace ID, agent name, date range. Vague tool calls produce vague results.
- **Read results before responding** — if a tool returns unexpected data, investigate before presenting it as fact.
- **Fail gracefully** — if a tool errors, explain what happened in plain language and suggest an alternative. Never show raw error payloads to the user.

### What I Never Do with Tools

- Search the knowledge base to answer "how are you?" or other conversational messages
- Call `platform_store_memory` for every interaction — only for facts worth keeping (see Memory section)
- Make multiple identical tool calls hoping for a different result
- Show raw JSON, function names, or API details to the user — always translate to plain language
- Use tools to verify things I already know from memory or context
```

**Upgrade `PlatformActionsSection` rendering:**

Currently `ActionRegistry.build_prompt_summary()` returns a flat list. Add a preamble:

```markdown
## Platform Actions

You can execute these actions via `platform_execute`. Always specify the action name
and include all required parameters. If an action fails, check the error and retry
with corrected parameters — do not guess or fabricate results.

[existing action catalog follows]
```

#### Delivery Mechanism

1. Replace `get_tool_guidance_prompt()` in `personality.py` with the behavioral contract
2. Add preamble to `PlatformActionsSection._build()` before the action catalog
3. Create new PromptRegistry versions via seed update
4. FutureAGI assessment before/after

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/consumers/chatbot/personality.py` | Replace `get_tool_guidance_prompt()` (~40 words → ~600 words) |
| `orchestrator/modules/context/sections/platform_actions.py` | Add behavioral preamble before action catalog |
| `orchestrator/core/seeds/seed_system_prompts.py` | Update PROMPT_MANIFEST for chatbot-* slugs |

#### Acceptance Criteria

1. `get_tool_guidance_prompt()` includes when-to-use, how-to-use, and never-do sections
2. `PlatformActionsSection` includes a behavioral preamble
3. Chatbot stops calling knowledge search for greetings (testable: send "good morning", verify no tool call)
4. Tool failures produce plain-language explanations (not raw error JSON)
5. FutureAGI `assess` shows improvement in `is_helpful` and `is_concise` scores

---

### 5.7 Pattern #F: Section-Level Anti-Patterns (Prompt Content)

#### Problem

System prompts tell agents what TO do. They rarely say what NOT to do. The `Response Rules` section in `get_base_system_prompt()` has one anti-pattern ("NEVER show code"), but there's no systematic anti-pattern documentation for business operations.

Without negative examples, the LLM defaults to training biases: over-explaining, unsolicited suggestions, redundant tool calls, and verbose responses.

#### What Claude Code Does

Explicit anti-patterns throughout the system prompt:
- "Do NOT use Bash to run commands when a dedicated tool is provided"
- "NEVER create documentation files unless explicitly requested"
- "Do NOT propose changes to code you haven't read"
- "Avoid over-engineering"

#### Design

Add anti-pattern blocks to the Identity section and Memory section. These are business-focused, not coding-focused.

**New anti-pattern block for Identity section (~200 tokens):**

```markdown
## What I Avoid

- **Over-researching simple requests** — "What time is it?" doesn't need a knowledge search
- **Unsolicited suggestions** — If asked to send an email, send the email. Don't also suggest a Slack message, a calendar invite, and a follow-up task unless asked
- **Repeating what the user said** — "You asked me to create an agent. I'll create an agent." → Just create the agent
- **Explaining how tools work** — "I'm going to use the platform_execute action to..." → Just do it and share the result
- **Being overly cautious** — "Are you sure you want me to...?" for routine operations. Confirm only for destructive or irreversible actions (deleting agents, removing integrations)
- **Long responses when short ones work** — If the answer is "Done. Agent created." then say that, not a 3-paragraph confirmation
```

**Integration into `personality.py`:**

Add as a new static method `get_anti_patterns()` called from `IdentitySection._build_chatbot_identity()`, appended after `get_action_response_style()`.

#### Delivery Mechanism

1. Add `get_anti_patterns()` to `AutomatosPersonality` class
2. Call from `IdentitySection._build_chatbot_identity()` in the parts list
3. Seed new PromptRegistry versions
4. FutureAGI assessment on `is_concise` metric

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/consumers/chatbot/personality.py` | Add `get_anti_patterns()` static method (~200 tokens) |
| `orchestrator/modules/context/sections/identity.py` | Add `get_anti_patterns()` call in `_build_chatbot_identity()` parts list |
| `orchestrator/core/seeds/seed_system_prompts.py` | Update PROMPT_MANIFEST |

#### Acceptance Criteria

1. Anti-pattern block is injected in chatbot identity section
2. Chatbot responds to "create an agent named X" with action + confirmation, not a preamble about what it's about to do
3. Chatbot doesn't call tools for simple greetings (testable)
4. FutureAGI `is_concise` score improves by >0.1 on sample conversations
5. No regression in `is_helpful` score

---

### 5.8 Pattern #H: Platform Awareness Prompt (Prompt Content)

#### Problem

`get_platform_skill()` in `personality.py` tells Auto what tools exist — "Agent management", "Skills & plugins", "Knowledge base" — but not what users can **accomplish**. A new user who says "help me run my business" gets a generic response because Auto's self-knowledge is tool-centric, not goal-centric.

Current prompt (~400 tokens):
```
**My capabilities:**
- Agent management: Create, configure, update, and delete AI agents
- Skills & plugins: Browse the marketplace, install to workspace, assign to agents
- Knowledge base: Search documents, codebase, and semantic indexes
- Recipes/workflows: Create and execute multi-step automation pipelines
- Memory: I remember users, preferences, and workspace context
- External integrations: Email, Slack, GitHub, Calendar via Composio
- Observability: Usage stats, costs, health checks, activity feeds
```

This reads like a feature list, not an assistant offering help. The user doesn't care about "agent management" — they care about "set up my team to handle customer emails automatically."

#### What Should Change

Rewrite `get_platform_skill()` as a **goal-oriented** capability map. Organized by what users want to do, not what API endpoints exist. Include the full breadth of 100+ platform actions grouped into achievable outcomes.

#### Design

**New `get_platform_skill()` content (~600 tokens, replaces ~400):**

```markdown
## Platform Skill — What I Am

I am **Auto**, the orchestrator brain of the **Automatos AI Platform**. I'm not a generic chatbot — I'm the platform itself.

### What I Can Do For You

**Set up your business operations:**
- Create AI agents for different roles — sales, support, marketing, ops, engineering, research
- Assign skills, plugins, and integrations to each agent
- Configure agent heartbeats for autonomous monitoring and reporting
- Apply governance blueprints to enforce quality and budget rules

**Automate your workflows:**
- Build playbooks — multi-step automation pipelines with triggers and schedules
- Schedule recurring tasks (cron or one-shot)
- Launch missions — complex multi-agent projects where I decompose the goal, assign agents, and deliver results

**Connect your tools:**
- 100+ integrations via Composio: Gmail, Slack, GitHub, Jira, Linear, Salesforce, HubSpot, Google Drive, Notion, Stripe, and more
- Browse and install from the marketplace — agents, skills, and plugins ready to use
- Upload documents to the knowledge base for semantic search

**Track everything:**
- Real-time analytics: costs, token usage, agent performance, success rates, efficiency scores
- System health monitoring with predictive alerts and bottleneck detection
- Task boards with priority management and SLA compliance tracking
- Reports from agents: standups, research, incidents, audits

**Manage content:**
- Publish blog posts, write long-form content
- Generate documents and reports
- Search conversation history and stored memories

**My tools are real.** I have platform_* tools for reading AND writing. When asked to create an agent, install a skill, or check workspace data — I call the tool and do it.

**For new workspaces**, I can run Mission Zero — a guided setup where I learn about your business, research the marketplace for the right agents and integrations, and build your operating environment from scratch. Just say "set up my workspace" or "help me get started."
```

**Key differences from current:**

| Aspect | Current | New |
|--------|---------|-----|
| Organization | By tool category | By user intent |
| Language | Technical ("agent management") | Goal-oriented ("set up your business") |
| Scope | 7 bullet points | 5 goal sections with specifics |
| Integrations | "Email, Slack, GitHub, Calendar" | "100+ integrations" with named examples |
| Missions | Not mentioned | Prominently featured |
| Onboarding | Not mentioned | Mission Zero teased for new users |
| Analytics | "Usage stats, costs" | "Real-time analytics: costs, token usage, success rates, efficiency scores" |
| Marketplace | "Browse the marketplace" | "Browse and install — agents, skills, plugins ready to use" |

#### Delivery Mechanism

1. Replace `get_platform_skill()` in `personality.py` with the goal-oriented version
2. Create new PromptRegistry version for `chatbot-friendly`, `chatbot-professional`, `chatbot-technical`
3. FutureAGI (PRD-29) A/B test: measure if new users engage more features in first 5 conversations

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/consumers/chatbot/personality.py` | Rewrite `get_platform_skill()` (~400 tokens → ~600 tokens, restructured) |
| `orchestrator/core/seeds/seed_system_prompts.py` | Update PROMPT_MANIFEST entries for chatbot-* slugs |

#### Acceptance Criteria

1. `get_platform_skill()` is organized by user intent, not tool category
2. Missions are explicitly mentioned as a capability
3. Mission Zero is teased for new workspaces ("Just say 'set up my workspace'")
4. Composio integrations mention 100+ apps with specific named examples
5. Analytics capabilities include success rates, efficiency scores, predictive alerts (not just "usage stats")
6. New PromptRegistry versions created for all chatbot personality modes
7. FutureAGI assessment: users in test group use 2+ more platform features in first 5 conversations vs control

---

### 5.9 Pattern #I: Mission Zero Onboarding (Next Sprint — documented here for context)

> [!NOTE]
> Mission Zero is listed here for completeness but is a **Next Sprint** item (Section 6.5). It depends on Pattern H (Platform Awareness Prompt) being live so Auto knows to offer it.

See Section 6.5 for the full design.

---

## 6. Phase 2 — Next Sprint

### 6.1 Pattern #4: Tool Tier Stratification

#### Problem

All tools are flat in `agent_tool_assignments`. A system health-check tool sits alongside a third-party Salesforce connector with no trust distinction. There's no way to enforce "system tools always available" or "marketplace tools require explicit approval." The `owner_type` field on agents distinguishes workspace vs marketplace, but tools have no equivalent.

#### What Claude Code Does

`CommandGraph` stratifies into builtins / plugin-like / skill-like. Each tier is independently togglable via flags (`include_plugin_commands`, `include_skill_commands`).

#### Design

**New enum on tools:**

```python
class ToolTier(str, Enum):
    SYSTEM = 'system'           # Platform internals (health, memory, context). Always available. Cannot be disabled.
    PLATFORM = 'platform'       # Automatos-managed tools (document gen, NL2SQL). Available by default. Can be disabled per-workspace.
    MARKETPLACE = 'marketplace' # Third-party via Composio/Adapter. Requires explicit enablement. Credential required.
    CUSTOM = 'custom'           # Workspace-created tools. Requires workspace-owner approval.
```

**Schema change:**

```sql
ALTER TABLE tools ADD COLUMN IF NOT EXISTS tier VARCHAR(20) DEFAULT 'marketplace';
-- Backfill system tools
UPDATE tools SET tier = 'system' WHERE name IN ('health_check', 'memory_store', 'memory_retrieve', 'context_lookup');
-- Backfill platform tools
UPDATE tools SET tier = 'platform' WHERE source = 'automatos' AND tier = 'marketplace';
```

**Enforcement rules:**

| Tier | Assignment Required? | Credential Required? | Can Disable? | Rate Limit |
|------|---------------------|---------------------|--------------|------------|
| `system` | No (always available) | No | No | None |
| `platform` | No (default on) | Per-tool | Yes (workspace setting) | Standard |
| `marketplace` | Yes (explicit) | Yes | Yes | Lower (2x less) |
| `custom` | Yes (owner approval) | Per-tool | Yes | Standard |

**Integration with Pattern #5:**

When a tool is blocked due to tier policy, emit a `PermissionDenial` with reason `tier_policy`:

```python
PermissionDenial(
    agent_id=agent.id,
    tool_name='salesforce_query',
    reason='Marketplace tool not enabled for this agent. Enable in workspace settings.',
    workspace_id=workspace.id,
    denied_at=now,
    context='orchestration'
)
```

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/models/core.py` | Add `ToolTier` enum, `tier` column to tools table |
| `orchestrator/core/llm/function_registry.py` | Filter by tier during tool resolution |
| `orchestrator/core/services/tool_service.py` | Tier-based policy enforcement |
| `orchestrator/core/database/init_database.py` | Migration + backfill |
| `orchestrator/api/tools.py` | Include tier in API responses |
| `orchestrator/core/services/coordinator_service.py` | Tier-aware agent matching |

#### Acceptance Criteria

1. System tools always appear in agent tool lists regardless of assignments
2. Marketplace tools require explicit `agent_tool_assignments` entry
3. Tier is visible in `GET /api/tools` response
4. Tier enforcement produces `PermissionDenial` events (Pattern #5)
5. Backfill migration correctly classifies existing tools

---

### 6.2 Pattern #11: Streaming Typed Events

#### Problem

SSE streaming currently emits 4 event types: `token`, `thinking`, `tool_call`, `done`. The frontend activity board (PRD-06) polls for updates. Users can't see memory operations, permission denials, context compaction, or budget warnings in real-time.

#### What Claude Code Does

`stream_submit_message()` yields fine-grained typed events: `message_start`, `command_match`, `tool_match`, `permission_denial`, `message_delta`, `message_stop`.

#### Design

**Expanded event vocabulary:**

```python
class StreamEventType(str, Enum):
    # Existing
    TOKEN = 'token'                            # LLM output token
    THINKING = 'thinking'                      # Extended thinking content
    TOOL_CALL = 'tool_call'                    # Tool invocation
    DONE = 'done'                              # Stream complete

    # New: Agent lifecycle
    AGENT_ASSIGNED = 'agent_assigned'          # Router picked an agent
    AGENT_THINKING = 'agent_thinking'          # Agent reasoning phase started

    # New: Tool lifecycle
    TOOL_RESOLVED = 'tool_resolved'            # Tool found in registry
    TOOL_PERMISSION_DENIED = 'tool_permission_denied'  # Pattern #5 integration
    TOOL_EXECUTING = 'tool_executing'          # Tool call in progress
    TOOL_RESULT = 'tool_result'               # Tool returned result

    # New: Memory operations
    MEMORY_INJECTED = 'memory_injected'        # Context from L1-L4 added to prompt
    MEMORY_STORED = 'memory_stored'            # New fact stored in memory

    # New: Context management
    CONTEXT_COMPACTED = 'context_compacted'    # Compaction triggered (Pattern #7)
    BUDGET_WARNING = 'budget_warning'          # Approaching token/cost limit

    # New: Orchestration (mission context)
    TASK_STATE_CHANGE = 'task_state_change'    # Task transitioned state
    MISSION_STOP = 'mission_stop'             # Mission ended (with stop_reason from Pattern #6)
```

**Event schema:**

```python
@dataclass(frozen=True)
class StreamEvent:
    type: StreamEventType
    content: str | None = None
    metadata: dict | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_sse(self) -> str:
        payload = {
            'type': self.type.value,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
        }
        return f"data: {json.dumps(payload)}\n\n"
```

**Emission example:**

```python
# In chat execution pipeline
yield StreamEvent(
    type=StreamEventType.AGENT_ASSIGNED,
    metadata={'agent_id': agent.id, 'agent_name': agent.name, 'route_method': 'semantic'}
)

yield StreamEvent(
    type=StreamEventType.MEMORY_INJECTED,
    metadata={'layers': ['L1', 'L3'], 'token_count': 340, 'facts_count': 5}
)

# ... LLM streaming tokens ...

yield StreamEvent(
    type=StreamEventType.TOOL_PERMISSION_DENIED,
    metadata={'tool_name': 'salesforce', 'reason': 'Not assigned', 'agent_id': agent.id}
)

yield StreamEvent(
    type=StreamEventType.DONE,
    metadata={'stop_reason': 'completed', 'tokens_used': 1240, 'cost': 0.0037}
)
```

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/models/` (new) | `StreamEventType` enum, `StreamEvent` dataclass |
| `orchestrator/core/services/chat_service.py` | Emit new events during execution |
| `orchestrator/core/llm/function_registry.py` | Emit `tool_resolved`, `tool_permission_denied` |
| `orchestrator/core/services/memory_service.py` | Emit `memory_injected`, `memory_stored` |
| `orchestrator/core/services/coordinator_service.py` | Emit `task_state_change`, `mission_stop` |
| `orchestrator/core/context/context_guard.py` | Emit `context_compacted`, `budget_warning` |
| `orchestrator/api/chat.py` | Update SSE generator to use `StreamEvent.to_sse()` |

> [!NOTE]
> Frontend changes to consume new events are deferred to a separate frontend PRD. The backend ships the events; frontend can adopt incrementally.

#### Acceptance Criteria

1. SSE stream includes `agent_assigned` event before first token
2. Memory injection visible as `memory_injected` event with layer info
3. Tool permission denials appear as `tool_permission_denied` event in stream
4. All events include `timestamp` for frontend ordering
5. Existing `token`, `thinking`, `tool_call`, `done` events unchanged (backward compatible)

---

### 6.3 Pattern #1: Frozen State Transition Models

#### Problem

`OrchestrationTask` and `OrchestrationRun` state fields are mutated in-place. In the coordinator tick loop, if two workers (unlikely with fcntl lock, but possible in future scaling) process the same run, they can race on state transitions. Even without races, in-place mutation makes it impossible to reconstruct "what state was the task in when the coordinator made this decision?"

#### What Claude Code Does

Every dataclass is `@dataclass(frozen=True)`. State changes produce new objects. Combined with event-sourcing, every state is a snapshot.

#### Design

**Introduce frozen transition records alongside mutable ORM models:**

```python
@dataclass(frozen=True)
class TaskTransition:
    """Immutable record of a state change. Never modify — always create new."""
    task_id: UUID
    from_state: str
    to_state: str
    triggered_by: str       # 'coordinator', 'agent', 'user', 'system'
    reason: str | None
    stop_reason: str | None  # Pattern #6 integration
    timestamp: datetime
    metadata: dict | None = None

@dataclass(frozen=True)
class RunTransition:
    """Immutable record of a run state change."""
    run_id: UUID
    from_state: str
    to_state: str
    triggered_by: str
    stop_reason: str | None  # Pattern #6
    timestamp: datetime
    metadata: dict | None = None
```

**Usage in state machine service:**

```python
async def transition_task(task, new_state, **kwargs) -> TaskTransition:
    transition = TaskTransition(
        task_id=task.id,
        from_state=task.state,
        to_state=new_state,
        triggered_by=kwargs.get('triggered_by', 'system'),
        reason=kwargs.get('reason'),
        stop_reason=kwargs.get('stop_reason'),
        timestamp=datetime.utcnow(),
        metadata=kwargs.get('metadata'),
    )

    # Validate transition is legal
    if not is_valid_transition(transition.from_state, transition.to_state):
        raise InvalidTransitionError(transition)

    # Apply to ORM (single mutation point)
    task.state = new_state
    task.updated_at = transition.timestamp
    db.commit()

    # Emit as event (immutable audit trail)
    await emit_event_from_transition(transition)

    return transition  # Caller gets frozen record
```

**Key principle:** The ORM model is still mutable (SQLAlchemy requires it), but every mutation goes through `transition_*()` which produces a frozen `Transition` record. The coordinator loop works with frozen records, not mutable ORM objects.

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/models/orchestration.py` | Add `TaskTransition`, `RunTransition` frozen dataclasses |
| `orchestrator/core/services/state_machine_service.py` | Return `Transition` from all state changes |
| `orchestrator/core/services/coordinator_service.py` | Work with `Transition` records in tick loop |
| `orchestrator/core/services/mission_dispatcher.py` | Accept frozen transitions |

#### Acceptance Criteria

1. Every `transition_task()` and `transition_run()` returns a frozen `Transition` dataclass
2. No direct `task.state = X` mutations outside the state machine service
3. All transitions are recorded as `OrchestrationEvent` entries
4. Coordinator tick loop receives and logs `Transition` records

---

### 6.4 Pattern #7: Proactive Transcript Compaction

#### Problem

`ContextGuard` compacts at 80% context window usage — reactive, not proactive. For a long session (30+ turns), all turns stay in context until the panic threshold fires. By then, context quality has already degraded (LLMs perform worse at high context utilization). The L1→L2 consolidation runs hourly — too slow for active sessions.

#### What Claude Code Does

`TranscriptStore.compact()` proactively keeps only last N entries. The query engine auto-compacts after a configurable turn count, before context limits are reached.

#### Design

**Proactive compaction after N turns:**

```python
# Configuration (add to config.py or session settings)
PROACTIVE_COMPACT_AFTER_TURNS: int = 8      # Compact every 8 turns
PROACTIVE_COMPACT_KEEP_RECENT: int = 4      # Keep last 4 turns verbatim
PROACTIVE_COMPACT_SUMMARY_TOKENS: int = 300 # Budget for summary of older turns

async def maybe_compact_session(
    messages: list[dict],
    turn_count: int,
    llm_manager: LLMManager,
) -> tuple[list[dict], bool]:
    """
    Proactive compaction: after every N turns, summarize oldest turns.
    Returns (compacted_messages, was_compacted).
    """
    if turn_count < PROACTIVE_COMPACT_AFTER_TURNS:
        return messages, False

    if turn_count % PROACTIVE_COMPACT_AFTER_TURNS != 0:
        return messages, False

    # Split: system prompt + old turns + recent turns
    system_msg = messages[0]  # system prompt
    recent = messages[-PROACTIVE_COMPACT_KEEP_RECENT * 2:]  # last N user+assistant pairs
    old = messages[1:-PROACTIVE_COMPACT_KEEP_RECENT * 2]

    if len(old) < 4:  # not enough to compact
        return messages, False

    # Summarize old turns
    summary = await llm_manager.quick_complete(
        prompt=f"Summarize this conversation concisely, preserving key decisions and facts:\n\n"
               + format_messages(old),
        max_tokens=PROACTIVE_COMPACT_SUMMARY_TOKENS,
    )

    # Rebuild: system + summary + recent
    compacted = [
        system_msg,
        {"role": "system", "content": f"[Conversation summary: {summary}]"},
        *recent,
    ]

    return compacted, True
```

**Integration point — in chat handler, before LLM call:**

```python
async def handle_chat_message(session, message, llm_manager):
    session.turn_count += 1
    session.messages.append({"role": "user", "content": message})

    # Proactive compaction (Pattern #7)
    session.messages, was_compacted = await maybe_compact_session(
        session.messages, session.turn_count, llm_manager
    )
    if was_compacted:
        yield StreamEvent(type=StreamEventType.CONTEXT_COMPACTED,  # Pattern #11
            metadata={'turn_count': session.turn_count, 'messages_before': len(old), 'messages_after': len(session.messages)})

    # Existing ContextGuard check (kept as safety net at 80%)
    session.messages, emergency_compact = await context_guard.check_and_compact(...)
```

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/context/` (new or extend) | `maybe_compact_session()` function |
| `orchestrator/core/services/chat_service.py` | Call compaction before LLM, emit event |
| `orchestrator/core/session_queue.py` | Add `turn_count` to session state |
| `config.py` | Add compaction configuration constants |

#### Acceptance Criteria

1. After 8 turns, oldest turns are summarized and replaced
2. Last 4 turns always kept verbatim (no loss of recent context)
3. `ContextGuard` 80% check still exists as safety net
4. `context_compacted` SSE event emitted when compaction occurs
5. Token usage per-session decreases for long conversations (measurable)

---

### 6.5 Pattern #I: Mission Zero Onboarding

#### Problem

When a new user signs up, they land in an empty workspace. Auto greets them but has no structured way to:
1. Learn about their business and goals
2. Research the marketplace for matching tools
3. Propose a workspace setup (agents, integrations, skills, playbooks)
4. Let the user iterate on the proposal ("I use Google Drive, not Dropbox")
5. Execute the approved plan as a mission

The `is_new_workspace` signal already exists (`GET /api/workspaces` returns `is_new_workspace: true` when `agent_count == 0`), but nothing acts on it. The frontend triggers a basic onboarding UI, but Auto itself has no onboarding intelligence.

The Mission Zero research (`docs/PRDS/Research/MISSION-ZERO/`) demonstrated this concept with a 14-agent roster for Automatos' own workspace — proving the pattern works. This pattern generalizes it for any user.

#### What Mission Zero Does

Mission Zero is **Auto as both planner and executor** for initial workspace setup. It's a coordinator-mode mission where Auto:

1. **Detects** a new or unconfigured workspace
2. **Discovers** the user's business through structured questions
3. **Researches** the marketplace dynamically to match needs
4. **Proposes** a complete setup in plan mode for user review
5. **Iterates** based on user feedback ("swap Jira for Linear", "skip the blog agent")
6. **Executes** the approved plan as a standard mission via existing orchestration infrastructure

#### Design

**Phase 1: Detection & Trigger**

Mission Zero activates when:
- `is_new_workspace == true` (no agents created yet), OR
- User explicitly says "set up my workspace", "help me get started", "mission zero"

Add a new prompt block to `get_platform_skill()` (Pattern H) that tells Auto about Mission Zero. Add detection logic in the chatbot consumer to inject the Mission Zero prompt when the workspace is empty.

```python
# In identity.py or a new onboarding section
class OnboardingSection(BaseSection):
    """Injected when workspace has no agents — triggers Mission Zero flow."""
    priority = 2  # High priority, after identity
    max_tokens = 800

    async def _build(self) -> str:
        # Check if workspace is empty
        agent_count = await get_agent_count(self.context.workspace_id)
        if agent_count > 0:
            return ""  # Not a new workspace, skip

        return MISSION_ZERO_PROMPT

MISSION_ZERO_PROMPT = """
## Mission Zero — Let's Set Up Your Workspace

This is a new workspace. My job is to help you set it up so it works for your business.

### What I'll Do

1. **Ask you about your business** — What do you do? What tools do you use? What do you want automated?
2. **Research the marketplace** — I'll search for agents, skills, plugins, and integrations that match your needs
3. **Propose a setup** — I'll present a plan with specific agents, roles, integrations, and workflows
4. **You review and adjust** — Swap tools, remove agents, change priorities — it's your workspace
5. **I build it** — Once you approve, I execute the plan as a mission and set everything up

### Discovery Questions

Start by asking these questions naturally (not as a numbered list — weave them into conversation):

- **Business type**: What kind of business or project is this for? (agency, SaaS, ecommerce, consulting, personal productivity, etc.)
- **Team size**: How many people will use Automatos? Just you, or a team?
- **Current tools**: What tools do you already use? (email provider, project management, messaging, CRM, file storage, etc.)
- **Priority automation**: What's the first thing you'd love to automate? (email triage, content creation, customer support, reporting, etc.)
- **Budget sensitivity**: Are you cost-conscious or willing to use premium models for better quality?
- **Existing content**: Do you have documents, knowledge bases, or codebases you want agents to access?

### How to Research & Propose

After gathering answers, research the marketplace:

1. Use `platform_browse_marketplace_agents` to find agent templates matching their business type
2. Use `platform_browse_marketplace_skills` to find relevant skills for their priority automations
3. Use `platform_browse_marketplace_plugins` to find plugins matching their workflow
4. Use `platform_list_connected_apps` to check which Composio integrations match their stated tools
5. Use `platform_list_llms` to recommend models based on budget sensitivity

Then present a structured proposal:

```
Here's what I'd set up for your [business type]:

**Agents:**
- [Agent Name] ([Model]) — [What it does for them]
- [Agent Name] ([Model]) — [What it does for them]
- ...

**Integrations:**
- [App] ✓ (you mentioned this)
- [App] ✓ (matched to your [need])
- ...

**Skills & Plugins:**
- [Skill] → assigned to [Agent]
- [Plugin] → workspace-wide
- ...

**Playbooks:**
- [Workflow name] — [trigger] → [steps]
- ...

**Estimated monthly cost:** ~$XX at typical usage

Would you like to adjust anything, or shall I set this up?
```

### Executing the Plan

When the user approves, create a mission:
- Goal: "Set up workspace for [business type]: create [N] agents, connect [N] integrations, install [N] skills, configure [N] playbooks"
- The coordinator decomposes this into tasks: one per agent creation, one per integration, one per skill install, etc.
- Each task uses existing platform tools (`platform_create_agent`, `platform_install_skill`, `platform_assign_skill_to_agent`, etc.)
- Mission completes with a summary of everything created

### User Iteration

If the user modifies the plan before approval:
- "I don't use Dropbox, I use Google Drive" → Swap the integration, re-present
- "Skip the blog agent" → Remove it, adjust cost estimate
- "Add a customer support chatbot" → Search marketplace for support agent templates, add to plan
- "What else can you do?" → Reference the Platform Awareness prompt (Pattern H) for full capabilities

Don't re-ask discovery questions for minor adjustments. Only re-research the marketplace if the user changes their business type or fundamentally shifts priorities.
"""
```

**Phase 2: Plan Mode Integration**

The Mission Zero proposal is presented in a structured format that maps directly to executable actions. When the user says "approve" or "set this up", Auto calls `platform_create_mission` with the full plan as the goal.

The mission coordinator already handles decomposition — it will break "create 5 agents, connect 3 integrations, install 4 skills" into individual tasks assigned to Auto itself (as the coordinator agent).

**Phase 3: Post-Setup**

After Mission Zero completes:
- Auto stores key facts about the business via `platform_store_memory` (Pattern E)
- The `OnboardingSection` stops injecting (agent_count > 0)
- Auto's regular Platform Awareness prompt (Pattern H) takes over for ongoing assistance
- Auto offers: "Your workspace is ready. Want me to run a quick tour of what each agent does?"

#### Marketplace Research Tools (Already Exist)

| Tool | Purpose in Mission Zero |
|------|------------------------|
| `platform_browse_marketplace_agents` | Find agent templates by business type (e.g., "marketing", "support") |
| `platform_browse_marketplace_skills` | Find skills for priority automations (e.g., "SEO", "email triage") |
| `platform_browse_marketplace_plugins` | Find plugins matching workflow needs |
| `platform_list_connected_apps` | Check which Composio integrations match user's stated tools |
| `platform_list_llms` | Recommend models based on budget sensitivity |
| `platform_create_agent` | Create agents from approved plan |
| `platform_install_skill` | Install marketplace skills |
| `platform_assign_skill_to_agent` | Wire skills to agents |
| `platform_create_playbook` | Set up automation workflows |
| `platform_configure_agent_heartbeat` | Set up autonomous agent monitoring |
| `platform_create_mission` | Execute the full setup as a coordinated mission |

All 11 tools already exist. **No new infrastructure needed.**

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/modules/context/sections/onboarding.py` (new) | `OnboardingSection` — injects Mission Zero prompt when `agent_count == 0` |
| `orchestrator/modules/context/sections/__init__.py` | Register `OnboardingSection` in `SECTION_REGISTRY` |
| `orchestrator/modules/context/modes.py` | Add `OnboardingSection` to `CHATBOT` mode's section list |
| `orchestrator/consumers/chatbot/personality.py` | Add Mission Zero reference in `get_platform_skill()` (done in Pattern H) |
| `orchestrator/core/seeds/seed_system_prompts.py` | Update PROMPT_MANIFEST with onboarding prompt content |

#### Acceptance Criteria

1. When a new user sends their first message, Auto proactively offers to set up the workspace (doesn't wait to be asked)
2. Auto asks 4-6 discovery questions conversationally (not a numbered form)
3. Auto uses `platform_browse_marketplace_*` tools to research matching agents, skills, and plugins
4. Auto presents a structured proposal with agents, integrations, skills, playbooks, and cost estimate
5. User can modify the proposal ("swap X for Y", "remove Z") and Auto re-presents without re-asking all questions
6. On user approval, Auto creates a mission that executes the setup
7. After mission completes, Auto stores business context in memory (Pattern E) and stops showing the onboarding prompt
8. `OnboardingSection` returns empty string for workspaces with agents (no prompt bloat for existing users)
9. User can trigger Mission Zero manually by saying "set up my workspace" even in a non-empty workspace (for re-configuration)

#### Research Context

The Mission Zero concept was validated in `docs/PRDS/Research/MISSION-ZERO/`:
- `MISSION-0.1-PROMPT.md` — 14-agent roster tested for Automatos' own workspace
- `MISSION-ZERO-RESULTS.md` — Full operating model with KPIs, review cadence, channel matrix
- `MISSION-ZERO-REVIEW.md` — Governance assessment with 48 acceptance criteria
- `PLATFORM-CAPABILITIES-DEFINITIVE.md` — 98 platform tools across 18 domains confirmed operational
- `PLATFORM-READINESS-REPORT.md` — 3-phase platform build confirmed complete

The key difference: the research was a hardcoded plan for one specific workspace. Pattern I generalizes this into a **dynamic, marketplace-driven flow** that works for any business type.

---

## 7. Phase 3 — Backlog

### 7.1 Pattern #8: Session Checkpointing

#### Problem

If the coordinator crashes mid-mission, task state is preserved in PostgreSQL but the conversation context for each agent is lost (L1 Redis may have expired). Long-running missions (10+ tasks, 30+ minutes) are vulnerable to context loss on restart.

#### Design

At key milestones (task completion, plan change, tool result), write a checkpoint to S3:

```python
@dataclass(frozen=True)
class SessionCheckpoint:
    run_id: UUID
    task_id: UUID | None
    messages: tuple[dict, ...]
    memory_snapshot: dict        # L1 state at checkpoint time
    tokens_used: int
    checkpoint_number: int
    created_at: datetime

async def write_checkpoint(run: OrchestrationRun, task: OrchestrationTask | None):
    checkpoint = SessionCheckpoint(
        run_id=run.id,
        task_id=task.id if task else None,
        messages=tuple(session.messages),
        memory_snapshot=await redis.get_session_state(run.session_id),
        tokens_used=run.tokens_used,
        checkpoint_number=run.checkpoint_count + 1,
        created_at=datetime.utcnow(),
    )
    key = f"checkpoints/{run.id}/{checkpoint.checkpoint_number}.json"
    await s3.put_object(key, json.dumps(asdict(checkpoint)))
    run.checkpoint_count += 1
```

**Resume from checkpoint:** `POST /api/missions/{id}/resume?from_checkpoint=latest`

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/models/orchestration.py` | `SessionCheckpoint` dataclass, `checkpoint_count` on run |
| `orchestrator/core/services/coordinator_service.py` | Write checkpoint after each task completion |
| `orchestrator/core/services/checkpoint_service.py` (new) | S3 read/write, resume logic |
| `orchestrator/api/missions.py` | Add resume endpoint |

#### Acceptance Criteria

1. Checkpoint written to S3 after each verified task
2. Resume from checkpoint restores conversation context
3. Checkpoint includes L1 memory snapshot
4. `GET /api/missions/{id}/checkpoints` lists available checkpoints

---

### 7.2 Pattern #3: Tool Manifest Snapshots

#### Problem

Tool definitions come from Composio sync and the Adapter catalog. These change over time (Composio updates tool schemas, Adapter adds new tools). If an agent behaved differently yesterday, there's no way to know if the tool definitions changed.

#### Design

After each Composio sync or Adapter catalog refresh, write a versioned manifest:

```python
async def snapshot_tool_manifest(workspace_id: UUID):
    tools = await get_all_tools(workspace_id)
    manifest = {
        'version': datetime.utcnow().isoformat(),
        'workspace_id': str(workspace_id),
        'tool_count': len(tools),
        'tools': [
            {
                'id': t.id,
                'name': t.name,
                'tier': t.tier,              # Pattern #4
                'schema_hash': hashlib.sha256(json.dumps(t.schema).encode()).hexdigest(),
                'provider': t.provider,
            }
            for t in tools
        ],
    }
    key = f"manifests/{workspace_id}/{manifest['version']}.json"
    await s3.put_object(key, json.dumps(manifest))
    return manifest
```

**Diff endpoint:** `GET /api/tools/manifest/diff?from=2026-03-30&to=2026-03-31`

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/services/tool_service.py` | `snapshot_tool_manifest()` after sync |
| `orchestrator/api/tools.py` | Manifest list + diff endpoints |

---

### 7.3 Pattern #9: PRD Parity Audit

#### Problem

PRDs define what should exist. There's no automated check of feature completeness. Tracking is manual.

#### Design

A CLI script that parses PRD files and checks for implementation markers:

```python
# scripts/prd_parity.py
async def audit_prd(prd_path: str) -> ParityResult:
    """
    Parse PRD, extract 'What Ships' table entries,
    check if corresponding files/endpoints/models exist.
    """
    prd = parse_prd_markdown(prd_path)
    entries = extract_ships_table(prd)

    results = []
    for entry in entries:
        found = await check_implementation(entry)
        results.append(ParityEntry(
            component=entry.component,
            expected=entry.description,
            found=found,
            coverage=compute_coverage(found),
        ))

    return ParityResult(prd=prd.id, entries=results)
```

Run as: `python scripts/prd_parity.py docs/PRDS/ --output parity_report.json`

#### Files to Change

| File | Change |
|------|--------|
| `scripts/prd_parity.py` (new) | Parser + checker |
| `CI pipeline` | Add parity check step (optional) |

---

### 7.4 Pattern #12: Tool Execution Cost Tracking

#### Problem

`LLMUsage` tracks LLM call costs perfectly. But tool executions (Composio API calls, Adapter REST calls) have no cost attribution. A mission might cost $2.40 in LLM but trigger 50 Salesforce API calls with their own rate-limit implications.

#### Design

Extend `tool_usage_logs` with cost data:

```sql
ALTER TABLE tool_usage_logs ADD COLUMN IF NOT EXISTS estimated_cost FLOAT DEFAULT 0.0;
ALTER TABLE tool_usage_logs ADD COLUMN IF NOT EXISTS rate_limit_remaining INT;
ALTER TABLE tool_usage_logs ADD COLUMN IF NOT EXISTS execution_ms INT;
```

Map costs from provider pricing (Composio reports usage per tool).

#### Files to Change

| File | Change |
|------|--------|
| `orchestrator/core/models/core.py` | Add columns to `tool_usage_logs` |
| `orchestrator/core/services/tool_service.py` | Record cost + latency after execution |
| `orchestrator/api/analytics.py` | Include tool costs in cost aggregation |

---

## 8. Implementation Phases Summary

```
PHASE 1 — QUICK WINS (1 week)
├── #H Platform Awareness Prompt           (2-3 files, 0.5 day)      ← PROMPT
├── #E Memory Decision Framework           (2-3 files, 1 day)        ← PROMPT
├── #B Business Tool Behavioral Contracts  (3-5 files, 1 day)        ← PROMPT
├── #F Section-Level Anti-Patterns         (1-2 files, 0.5 day)      ← PROMPT
├── #5 Permission Denial as Data           (4-6 files, 1-2 days)
├── #6 Named Stop Reasons                  (3-5 files, 1 day)
├── #2 Trust-Gated Init                    (1-2 files, 0.5 day)
└── #10 Bootstrap Named Stages             (2-3 files, 0.5 day)

PHASE 2 — NEXT SPRINT (2 weeks)
├── #I Mission Zero Onboarding             (5-8 files, 2-3 days)     ← ONBOARDING
├── #4 Tool Tier Stratification            (5-8 files, 2-3 days)
├── #11 Streaming Typed Events             (8-12 files, 3-4 days)
├── #1 Frozen State Transitions            (10-15 files, 2-3 days)
└── #7 Proactive Compaction                (2-4 files, 1-2 days)

PHASE 3 — BACKLOG (as needed)
├── #8 Session Checkpointing               (4-6 files, 2-3 days)
├── #3 Tool Manifest Snapshots             (2-3 files, 1 day)
├── #9 PRD Parity Audit                    (3-5 files, 2-3 days)
└── #12 Tool Cost Tracking                 (2-3 files, 1 day)
```

> [!NOTE]
> Prompt patterns (E, B, F) are the fastest wins — they're content changes to existing files, not architectural changes. They ship as new PromptRegistry versions (PRD-58) and can be A/B tested via FutureAGI (PRD-29) before activation. Roll back with one click if metrics regress.

## 9. Testing Strategy

### Unit Tests

| Pattern | Test |
|---------|------|
| #H | `test_platform_skill_organized_by_user_intent()`, `test_platform_skill_mentions_missions()`, `test_platform_skill_mentions_mission_zero()` |
| #E | `test_memory_framework_in_system_prompt()`, `test_platform_store_memory_description_includes_anti_patterns()` |
| #B | `test_tool_guidance_includes_behavioral_contracts()`, `test_platform_actions_has_preamble()` |
| #F | `test_anti_patterns_in_chatbot_identity()`, `test_anti_patterns_not_in_task_execution()` |
| #I | `test_onboarding_section_injected_for_new_workspace()`, `test_onboarding_section_empty_for_existing_workspace()`, `test_mission_zero_prompt_includes_discovery_questions()` |
| #5 | `test_permission_denial_created_on_missing_tool()`, `test_coordinator_reassigns_on_denial()` |
| #6 | `test_budget_exhausted_stop_reason()`, `test_max_retries_stop_reason()`, `test_completed_stop_reason()` |
| #2 | `test_core_boots_without_extensions()`, `test_failed_extension_doesnt_crash_startup()` |
| #10 | `test_bootstrap_report_captures_all_stages()`, `test_failed_stage_recorded()` |
| #4 | `test_system_tools_always_available()`, `test_marketplace_requires_assignment()` |
| #11 | `test_agent_assigned_event_emitted()`, `test_memory_injected_event_emitted()` |
| #1 | `test_transition_returns_frozen_record()`, `test_invalid_transition_raises()` |
| #7 | `test_compaction_after_n_turns()`, `test_recent_turns_preserved()` |

### Integration Tests

| Test | Patterns Covered |
|------|-----------------|
| Full mission lifecycle with budget limit | #5, #6, #1 |
| Startup with failed Composio sync | #2, #10 |
| 20-turn conversation with compaction | #7, #11 |
| Agent tool resolution with tier enforcement | #4, #5 |
| New workspace first message triggers Mission Zero flow | #H, #I |
| Mission Zero marketplace research + plan generation | #I |
| Mission Zero plan approval creates executable mission | #I |

### Prompt Quality Tests (via FutureAGI)

| Test | Patterns Covered | Method |
|------|-----------------|--------|
| Memory storage quality: sample 50 new Mem0 entries, <20% ephemeral artifacts | #E | FutureAGI `assess` on live traffic after 1 week |
| No tool calls for greetings: send 10 greetings, verify 0 tool calls | #B, #F | Manual test + FutureAGI `safety` check |
| Response conciseness: compare avg response length before/after | #F | FutureAGI `is_concise` metric |
| Tool failure handling: trigger 5 known tool errors, verify plain-language response | #B | Manual test |
| New user feature engagement: test group uses 2+ more features in first 5 convos | #H | FutureAGI `assess` on new user cohort |
| Mission Zero completion: new workspace → approved plan → executed mission | #I | Manual E2E test with fresh workspace |
| Mission Zero marketplace matching: user says "I use Slack" → Slack integration proposed | #I | Manual test + assertion on marketplace tool calls |

## 10. Success Criteria

1. **Smarter memory** — Mem0 entries shift from ephemeral artifacts to curated facts; <20% garbage after 1 week (baseline: ~60%)
2. **Better tool usage** — no tool calls for greetings; tool failures produce plain-language responses; fewer redundant tool calls
3. **Concise responses** — measurable improvement in FutureAGI `is_concise` metric across chatbot conversations
4. **Platform-aware Auto** — Auto describes capabilities in terms of user goals, not API endpoints; new users understand what they can do in the first conversation
5. **Zero-to-operational onboarding** — a new user can go from empty workspace to fully configured (agents, integrations, skills, playbooks) in one Mission Zero conversation
6. **Dynamic marketplace matching** — Mission Zero proposals are built from live marketplace data, not hardcoded templates; user modifications ("swap X for Y") are handled without re-asking discovery questions
7. **Zero silent failures** — every permission denial, budget hit, and retry exhaustion is recorded as structured data
8. **Startup resilience** — platform runs in degraded mode if any extension fails; `/health/bootstrap` shows exactly what's broken
9. **Debuggable orchestration** — any mission failure can be explained from `stop_reason` + `permission_denials` without log diving
10. **Real-time transparency** — SSE stream includes 10+ event types covering the full agent execution lifecycle
11. **Context quality** — long conversations (20+ turns) maintain response quality via proactive compaction
12. **No regressions** — all existing tests pass, all existing SSE consumers backward compatible, FutureAGI `is_helpful` score does not decrease

## 11. Risks

| Risk | Mitigation |
|------|-----------|
| Memory framework prompt is too long (800 tokens) | Token budget for identity section is 600 + no limit for chatbot mode. Memory framework goes in self_learning which is outside the max_tokens cap. Monitor total prompt size. |
| Anti-patterns make the agent too passive | Keep the list tight (6 items). Focus on "don't over-do" not "don't do." FutureAGI `is_helpful` metric catches regression. |
| Tool behavioral contracts conflict with specific skill instructions | Skills (Priority 4) override general tool guidance. Behavioral contracts are defaults; skill-specific instructions take precedence. |
| Prompt changes break existing personality modes | Deliver as new PromptRegistry versions. Old versions remain archived. One-click rollback via admin UI. |
| Frozen dataclasses add verbosity | Only freeze state transitions, not ORM models. Minimal overhead. |
| Proactive compaction loses important context | Keep last 4 turns verbatim. Summary preserves key facts. 80% guard remains as safety net. |
| Too many SSE events overwhelm frontend | Frontend consumes incrementally. New events are additive. Existing consumers unchanged. |
| Trust gate causes "degraded mode" confusion | Clear `/health` reporting. Log warnings. Dashboard shows extension status. |
| Backfill of tool tiers misclassifies tools | Manual review of backfill SQL. Default to `marketplace` (most restrictive). |
| Mission Zero prompt too large for context | OnboardingSection has max_tokens=800. Prompt is ~700 tokens. Only injected for empty workspaces — zero cost for existing users. |
| Mission Zero marketplace results change between proposal and execution | Cache marketplace results for the session. If items are removed between proposal and execution, Auto reports what couldn't be installed and suggests alternatives. |
| User abandons Mission Zero mid-flow | No harm — nothing is created until explicit approval. Auto stores partial business context in memory for future attempts. |
| Mission Zero over-provisions (too many agents for a solo user) | Discovery question about team size calibrates the proposal. Solo users get 3-5 agents; teams get more. Budget estimate is shown upfront. |

## 12. Open Questions

1. Should `PermissionDenial` records be persisted in their own table or only as `OrchestrationEvent` entries?
2. What's the right `PROACTIVE_COMPACT_AFTER_TURNS` value? 8 is a starting point — needs tuning per-model.
3. Should tool manifest snapshots be per-workspace or global?
4. Do we need a "resume from degraded mode" mechanism when a failed extension recovers?
5. Should the memory decision framework differ per personality mode? (e.g., `professional` mode might store more formal decisions, `friendly` might store more personal context)
6. Should anti-patterns also apply to `TASK_EXECUTION` mode agents, or only `CHATBOT`? Task agents might need different anti-patterns (e.g., "don't skip verification steps").
7. How long before we measure memory quality improvement? 1 week proposed, but may need 2 weeks for statistical significance.
8. Should Mission Zero be re-runnable? ("I want to add a support department" on an existing workspace) — currently designed to also trigger on explicit "set up my workspace" command.
9. Should Mission Zero proposals be saved as blueprints so users can share their setup templates with others?
10. Should the OnboardingSection inject for workspaces with agents but no integrations (partially configured)? Could use a readiness score instead of a simple agent_count check.
11. Should Mission Zero have a "quick start" mode (skip questions, use sensible defaults for common business types) vs the full discovery flow?

## 13. References

- **Research:** Claude Code harness pattern analysis (`research_claude_code_patterns.md` in project memory)
- **Source repo:** `instructkr/claude-code` (Python port of Claude Code harness)
- **Mission Zero Research:** `docs/PRDS/Research/MISSION-ZERO/` — 8 files documenting the Mission Zero concept, 14-agent roster, governance assessment, platform capabilities inventory, and readiness verdicts
- **Related PRDs:** 82A (Coordinator), 79 (Memory), 35 (Tools), 55 (Channels), 06 (Dashboard), 77 (Scheduled Tasks)
- **Prompt Management:** PRD-58 (PromptRegistry, versioning, seeding), PRD-29 (FutureAGI observability)
- **Existing Architecture:** `ContextService` (`modules/context/service.py`), `SECTION_REGISTRY` (`modules/context/sections/__init__.py`), `AutomatosPersonality` (`consumers/chatbot/personality.py`)
- **Workspace Onboarding Signal:** `api/workspaces.py` — `is_new_workspace: true` when `agent_count == 0`
- **Marketplace Tools:** `actions_marketplace.py` — `platform_browse_marketplace_agents`, `platform_browse_marketplace_skills`, `platform_browse_marketplace_plugins`
