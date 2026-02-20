# PRD-58: System Prompt Management & FutureAGI Evaluation Integration

**Version:** 3.0
**Status:** In Progress
**Date:** February 19, 2026 (updated from v2.0 Feb 18)
**Author:** Claude Code + Gerard
**Prerequisites:** PRD-29 (FutureAGI Observability), PRD-55 (Autonomous Assistant / Soul Designer)
**Branch:** `futureAGI` (worktree: `automatos-ai-futureAGI`)
**FutureAGI Access:** 6-month free trial (expires ~August 2026)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 17 | Initial draft — SDK-based approach |
| 2.0 | Feb 18 | Complete rewrite of Phase 1B. Dropped SDK (crashes Docker). Direct HTTP API. Live traffic eval with toggle. Intelligent self-healing vision. |
| 3.0 | Feb 19 | Phase 1B marked complete. Full detailed specs for Phase 1C (Self-Healing) and Phase 2 (User-Facing Features). Architecture evolved to worker service pattern. |

---

## Executive Summary

Automatos has 17 system prompts hardcoded across the orchestrator that control every decision the platform makes — routing, task decomposition, agent selection, quality assessment, SQL generation, memory injection, personality, and more. Today these prompts:

- Live in Python source code, scattered across 17 files
- Can only be changed by a developer deploying code
- Have zero quality data — they were tuned by feel, never evaluated
- Cannot be A/B tested, versioned, or rolled back without git

This PRD introduces three capabilities:

1. **Admin Prompt Management System** (Phase 1A) — Move all 17 system prompts into the database with a full management UI, version history, and instant rollback. No code deploys to change how Automatos thinks. **STATUS: SHIPPED**

2. **FutureAGI Live Traffic Evaluation** (Phase 1B) — Per-prompt toggle that sends real chat input/output pairs to FutureAGI for quality scoring in the background. Scores accumulate over time as a live quality dashboard. Safety scanning on prompt text. **STATUS: COMPLETE**

3. **Intelligent Self-Healing Prompt System** (Phase 1C) — The orchestrator monitors rolling quality scores per prompt. When it detects degradation (score drops, error rate spikes), it automatically enables eval, runs optimization, and can swap in improved prompt versions. A self-healing AI. **STATUS: PLANNED**

### Architecture Decision: Worker Service Pattern

The FutureAGI Python SDK (`futureagi==0.6.0`) was evaluated and rejected (v2.0). The architecture then evolved further:

1. **v1.0**: SDK in orchestrator → crashed Docker, broken APIs
2. **v2.0**: Direct HTTP in orchestrator → worked but polluted orchestrator deps
3. **v3.0 (current)**: Isolated `agent-opt-worker` service handles all FutureAGI calls. Orchestrator calls worker via internal HTTP. Zero FutureAGI deps in orchestrator.

**Decision:** Orchestrator → Worker service (`AGENT_OPT_WORKER_URL`) via `httpx`. Worker owns FutureAGI API keys and SDK concerns. Orchestrator owns DB operations and dispatch logic.

---

## Current Implementation Status

### Phase 1A: Admin Prompt Management — SHIPPED ✅

Everything in the original Phase 1A is deployed and working:

| Component | Status | Commits |
|-----------|--------|---------|
| DB tables (system_prompts, versions, eval_runs) | ✅ Deployed | `a02a002` |
| Seed script (15 prompts across 4 categories) | ✅ Deployed | `a02a002` |
| PromptRegistry service with caching + fallback | ✅ Deployed | Pre-existing |
| Admin API endpoints (CRUD + versions + rollback) | ✅ Deployed | `5157daa` |
| Admin UI: SystemPromptsTab in Settings | ✅ Deployed | `b80b1d7`, `d4eac9a` |
| Auth: Clerk/API-key users can manage prompts | ✅ Deployed | `5157daa` |
| Startup: create_tables + seed on Railway boot | ✅ Deployed | `a02a002` |

### Phase 1B: FutureAGI Integration — COMPLETE ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Direct HTTP service (no SDK) | ✅ Deployed | Routes through `agent-opt-worker` service |
| Per-template config (keys + models) | ✅ Deployed | `6b8936d` |
| Concurrent execution (asyncio.gather) | ✅ Deployed | `6b8936d` |
| Safety scanning (4 checks) | ✅ Working | toxicity, bias, injection, moderation |
| Assessment (3 metrics) | ✅ Working | completeness, is_helpful, is_concise via worker |
| Optimize | ✅ Working | Uses worker `/optimize` with dataset from live traffic |
| Frontend: per-run-type rendering | ✅ Deployed | `d4eac9a` |
| Frontend: polling + loading states | ✅ Deployed | `60a46e2` |
| Live traffic eval toggle | ✅ Deployed | Per-prompt toggle in assessments tab |
| Hook into chat pipeline | ✅ Deployed | Fire-and-forget at 3 hook points in service.py |
| Idempotent column migration | ✅ Deployed | `ALTER TABLE ADD COLUMN IF NOT EXISTS` in main.py |

---

## The 17 System Prompts

(Unchanged from v1.0 — see original section)

Every prompt that drives Automatos decision-making:

### Core Orchestration (the "brain")

| # | Prompt | File | What It Decides | Impact |
|---|--------|------|----------------|--------|
| 1 | **Personality/Soul** | `consumers/chatbot/personality.py` | Tone, style, warmth of every response | Every message |
| 2 | **Default System Prompt** | `consumers/chatbot/prompt_analyzer.py` | Fallback identity when no personality set | Every message |
| 3 | **Routing Classifier** | `core/routing/engine.py` | Which agent handles each user message | Every message |
| 4 | **Intent Classifier Patterns** | `consumers/chatbot/intent_classifier.py` | Simple vs complex, tool needs, memory needs | Every message |
| 5 | **Tool Ranking Logic** | `consumers/chatbot/prompt_analyzer.py` | Which tools get suggested to the LLM | Every tool call |

### Multi-Agent Orchestration (complex tasks)

| # | Prompt | File | What It Decides | Impact |
|---|--------|------|----------------|--------|
| 6 | **Task Decomposer** | `modules/orchestrator/stages/task_decomposer.py` | How complex tasks break into subtasks | Every multi-step task |
| 7 | **Complexity Analyzer** | `modules/orchestrator/stages/complexity_analyzer.py` | Simple vs moderate vs complex classification | Every task |
| 8 | **Agent Selector** | `modules/orchestrator/llm/llm_agent_selector.py` | Which agent runs each subtask | Every subtask |
| 9 | **Strategy Planner** | `modules/orchestrator/llm/master_orchestrator.py` | Speed/quality/cost tradeoff strategy | Every orchestrated workflow |
| 10 | **Quality Assessor** | `modules/orchestrator/stages/quality_assessor.py` | Whether outputs meet quality threshold | Every output |
| 11 | **Context Optimizer** | `modules/search/optimization/context_optimizer.py` | What context gets injected, how much, what format | Every context-enriched request |

### Domain-Specific

| # | Prompt | File | What It Decides | Impact |
|---|--------|------|----------------|--------|
| 12 | **NL2SQL Generator** | `modules/nl2sql/query/nl2sql_service.py` | Natural language → SQL translation | Every data query |
| 13 | **Memory Injection Template** | `modules/memory/operations/prompt_injection.py` | How memories get formatted into context | Every memory-aware request |
| 14 | **Agent Factory Builder** | `modules/agents/factory/agent_factory.py` | Dynamic agent system prompt assembly | Every agent execution |
| 15 | **Execution Manager** | `modules/agents/execution/execution_manager.py` | Default professional execution prompt | Every agent run |

### Personas & Templates

| # | Prompt | File | What It Decides | Impact |
|---|--------|------|----------------|--------|
| 16 | **Persona Presets** (x4) | `core/seeds/seed_personas.py` | Engineer, Sales, Marketing, Support personas | Agent creation |
| 17 | **Recipe Learning** | `core/services/recipe_learning_service.py` | How improvement suggestions are generated | Recipe optimization |

---

## Phase 1A: Admin Prompt Management System — SHIPPED ✅

(Unchanged from v1.0 — fully deployed. See original PRD for data model, API endpoints, and UI specs.)

Key endpoints deployed:
```
GET    /api/admin/prompts                              — List all system prompts
GET    /api/admin/prompts/:id                          — Get prompt with active version
GET    /api/admin/prompts/:id/versions                 — List all versions
POST   /api/admin/prompts/:id/versions                 — Create new version (draft)
POST   /api/admin/prompts/:id/versions/:vid/activate   — Activate a version
POST   /api/admin/prompts/:id/rollback                 — Rollback to previous version
DELETE /api/admin/prompts/:id/versions/:vid             — Delete a draft version
GET    /api/admin/prompts/:id/assessment-runs           — List eval runs
POST   /api/admin/prompts/:id/assess                    — Trigger assessment run
```

---

## Phase 1B: FutureAGI Live Traffic Evaluation — COMPLETE ✅

### 1B.1 The Problem with Synthetic Testing

The original PRD assumed we'd create test datasets per prompt and evaluate against those. In practice:
- FutureAGI evaluates **input/output pairs**, not prompts in isolation
- Sending fake output (`"System prompt assessed successfully."`) produces garbage scores
- Creating realistic test datasets for 17 prompts is weeks of manual work
- Synthetic tests don't reflect real-world usage patterns

### 1B.2 The Solution: Live Traffic Evaluation

**Instead of synthetic tests, evaluate real conversations.**

When a system prompt has FutureAGI evaluation enabled:
1. A user sends a chat message
2. The orchestrator selects the system prompt, sends it + user message to LLM
3. LLM responds
4. **Fire-and-forget**: send the real (user input, LLM output) to FutureAGI for scoring
5. Score is stored in `system_prompt_eval_runs` linked to the prompt version
6. Scores accumulate over time → live quality dashboard

Zero impact on chat latency — the eval call is fully async in the background.

### 1B.3 Per-Prompt Eval Toggle

New field on `SystemPrompt` model:

```python
# Add to SystemPrompt model
futureagi_eval_enabled = Column(Boolean, default=False)
```

**Admin UI**: Toggle switch per prompt in the detail view. When ON, every real chat interaction using that prompt gets scored.

**API**:
```
PATCH  /api/admin/prompts/:id/eval-toggle   — Toggle FutureAGI eval on/off
```

### 1B.4 Hook Point: Chat Pipeline

The hook goes into `consumers/chatbot/service.py` where `SmartChatIntegration` prepares the orchestrated request and the LLM response comes back.

```python
# In service.py, after LLM response is complete:

# Fire-and-forget FutureAGI eval if the prompt has eval enabled
if orchestrated.system_prompt_slug:  # set by SmartChatIntegration
    try:
        from core.services.futureagi_service import futureagi_service
        if futureagi_service.is_available:
            asyncio.create_task(
                futureagi_service.eval_live_traffic(
                    prompt_slug=orchestrated.system_prompt_slug,
                    input_text=user_message,
                    output_text=assistant_response,
                    context_text=orchestrated.system_prompt,
                )
            )
    except Exception:
        pass  # Never block chat for eval
```

### 1B.5 FutureAGI Service: Direct HTTP (No SDK)

```python
# orchestrator/core/services/futureagi_service.py

FUTUREAGI_BASE_URL = "https://api.futureagi.com"
ASSESSMENT_ENDPOINT = f"{FUTUREAGI_BASE_URL}/sdk/api/v1/new-eval/"

# Per-template configuration (sourced from GET /sdk/api/v1/get-evals/)
TEMPLATE_CONFIG = {
    # Quality assessment
    "completeness":      {"keys": ["input", "output"], "model": "turing_large"},
    "is_helpful":        {"keys": ["input", "output"], "model": "turing_large"},
    "is_concise":        {"keys": ["output"], "model": "turing_large"},
    "factual_accuracy":  {"keys": ["input", "output"], "model": "turing_large"},
    "groundedness":      {"keys": ["input", "output", "context"], "model": "turing_large"},
    "prompt_adherence":  {"keys": ["input", "output"], "model": "turing_large"},  # NOTE: has server-side bug
    "summary_quality":   {"keys": ["input", "output"], "model": "turing_large"},
    # Safety
    "toxicity":          {"keys": ["output"], "model": "protect"},
    "bias_detection":    {"keys": ["output"], "model": "protect_flash"},
    "prompt_injection":  {"keys": ["input"], "model": "protect"},
    "content_moderation": {"keys": ["output"], "model": "protect"},
}

# Auth: X-Api-Key + X-Secret-Key headers
# Each template requires specific input keys and model — see TEMPLATE_CONFIG
# All eval calls use asyncio.gather() for concurrent execution
```

### 1B.6 Live Traffic Eval Method

New method on `FutureAGIService`:

```python
async def eval_live_traffic(
    self,
    prompt_slug: str,
    input_text: str,
    output_text: str,
    context_text: str | None = None,
) -> None:
    """
    Called fire-and-forget after each chat response.
    Checks if eval is enabled for this prompt, runs metrics, stores results.
    """
    # 1. Check if eval is enabled for this prompt slug
    # 2. Run quality metrics: completeness, is_helpful, is_concise
    # 3. Store results in system_prompt_eval_runs
    # 4. Update rolling average on prompt version
```

### 1B.7 Safety Scanning — Working ✅

Safety scanning runs on prompt text directly (no real I/O needed):
- **toxicity** — `protect` model, checks output text
- **bias_detection** — `protect_flash` model, checks output text
- **prompt_injection** — `protect` model, checks input text
- **content_moderation** — `protect` model, checks output text

All 4 checks run concurrently via `asyncio.gather()`.

### 1B.8 Optimize — Deferred

FutureAGI's improve-prompt API has gone async (returns job IDs, no inline results). Two options for later:
1. Poll FutureAGI for results (no polling endpoint found yet)
2. Use own LLMs to optimize based on accumulated assessment feedback

**Decision**: Park optimize for now. Focus on getting live traffic eval working — that's the foundation everything else builds on.

### 1B.9 Frontend: Assessment Dashboard

The Assessments tab per prompt shows:
- **Toggle**: FutureAGI Eval ON/OFF switch
- **Live scores**: Rolling quality metrics from real traffic
- **Assessment runs**: Individual eval results with reasons
- **Safety results**: Per-check pass/fail with detailed reasons
- Auto-polling every 3s while runs are pending/running

### 1B.10 Implementation Steps (Phase 1B — what we're building now)

| Step | What | Status |
|------|------|--------|
| 1 | Add `futureagi_eval_enabled` column to SystemPrompt | 🔨 Building |
| 2 | Add PATCH `/eval-toggle` API endpoint | 🔨 Building |
| 3 | Add toggle switch to frontend prompt detail view | 🔨 Building |
| 4 | Add `eval_live_traffic()` method to FutureAGIService | 🔨 Building |
| 5 | Hook into chat pipeline (service.py after LLM response) | 🔨 Building |
| 6 | Store per-message eval results in eval_runs table | 🔨 Building |
| 7 | Display accumulated scores in assessments dashboard | 🔨 Building |
| 8 | Test: send chat messages with eval ON, verify scores appear | 🔨 Building |

### 1B.11 Known Issues

| Issue | Impact | Workaround |
|-------|--------|-----------|
| `prompt_adherence` template returns server-side error | Can't use this metric | Replaced with `is_helpful` in defaults |
| `bias_detection` can timeout (>90s) | Occasional missing safety check | 90s timeout, graceful degradation |
| FutureAGI optimize API returns async job IDs | Can't get optimized prompts inline | Deferred to Phase 1C |
| FutureAGI API has no rate limiting docs | Unknown throughput limits | Start with eval on 1-2 prompts, monitor |

---

## Phase 1C: Intelligent Self-Healing Prompt System — PLANNED

### Vision

The orchestrator becomes self-aware about prompt quality. Instead of an admin manually checking scores, the system:

1. **Monitors** rolling average quality scores per prompt version
2. **Detects** degradation — score drops below threshold, error rate spikes
3. **Responds** automatically:
   - Enables FutureAGI eval if not already on
   - Triggers optimization (using own LLMs to rewrite based on failure patterns)
   - Creates a new prompt version candidate
   - A/B tests the candidate against the current version
   - If candidate scores better → activates it
   - If not → discards and alerts admin
4. **Reports** what it did and why via audit trail

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Chat Pipeline                              │
│  User → System Prompt → LLM → Response                           │
│                    │                │                              │
│                    │    ┌───────────┘                              │
│                    │    │ (fire-and-forget)                        │
│                    ▼    ▼                                          │
│              ┌─────────────────┐                                  │
│              │  FutureAGI Eval │                                  │
│              │  (per-message)  │                                  │
│              └────────┬────────┘                                  │
│                       │                                           │
│                       ▼                                           │
│              ┌─────────────────┐                                  │
│              │  Score Store    │                                  │
│              │  (eval_runs)    │                                  │
│              └────────┬────────┘                                  │
│                       │                                           │
│              ┌────────┴─────────────────┐                         │
│              │  Prompt Health Monitor   │ ← APScheduler (5 min)   │
│              │                          │                         │
│              │  1. Compute rolling avg  │                         │
│              │  2. Compare to baseline  │                         │
│              │  3. Detect degradation   │                         │
│              └──────────┬───────────────┘                         │
│                         │                                         │
│                  ┌──────┴───────┐                                  │
│                  │              │                                  │
│              Healthy       Degraded                                │
│           (log + skip)        │                                    │
│                               ▼                                   │
│              ┌──────────────────────────┐                         │
│              │  Auto-Optimizer          │                         │
│              │  1. Collect failure runs │                         │
│              │  2. Analyze patterns     │ ← LLM-as-judge (Claude) │
│              │  3. Generate new prompt  │                         │
│              │  4. Create draft version │                         │
│              └──────────┬───────────────┘                         │
│                         │                                         │
│                         ▼                                         │
│              ┌──────────────────────────┐                         │
│              │  A/B Test Runner         │                         │
│              │  • 50% traffic → current │                         │
│              │  • 50% traffic → candidate│                        │
│              │  • Over N messages       │                         │
│              │  • Auto-promote or       │                         │
│              │    discard + alert       │                         │
│              └──────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
```

### 1C.1 Data Model Changes

New table for health monitoring + audit trail:

```python
# Add to core/models/system_prompts.py

class PromptHealthSnapshot(Base):
    """
    Periodic health snapshot per prompt version.
    Created by the PromptHealthMonitor every 5 minutes.
    """
    __tablename__ = "prompt_health_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompts.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("system_prompt_versions.id", ondelete="SET NULL"), nullable=True)

    # Rolling averages (computed from eval_runs)
    window_size = Column(Integer, nullable=False)           # number of runs in window
    avg_completeness = Column(Float, nullable=True)
    avg_helpfulness = Column(Float, nullable=True)
    avg_conciseness = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)          # weighted average of above
    failure_rate = Column(Float, nullable=True)             # % of runs where failure=true
    eval_count = Column(Integer, default=0)                 # total evals in window

    # Health status
    health_status = Column(String(20), nullable=False, default="healthy")  # healthy | degraded | critical | recovering
    degradation_delta = Column(Float, nullable=True)        # % drop from baseline

    # Actions taken
    action_taken = Column(String(50), nullable=True)        # none | eval_enabled | optimization_triggered | ab_test_started | version_promoted | version_discarded
    action_details = Column(JSONB, nullable=True)           # audit trail details

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PromptABTest(Base):
    """
    Tracks an A/B test between two prompt versions.
    """
    __tablename__ = "prompt_ab_tests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompts.id", ondelete="CASCADE"), nullable=False, index=True)

    control_version_id = Column(UUID(as_uuid=True), ForeignKey("system_prompt_versions.id"), nullable=False)
    candidate_version_id = Column(UUID(as_uuid=True), ForeignKey("system_prompt_versions.id"), nullable=False)

    status = Column(String(20), nullable=False, default="running")  # running | promoting | discarded | completed
    traffic_split = Column(Float, default=0.5)              # % of traffic to candidate

    # Results
    control_eval_count = Column(Integer, default=0)
    control_composite_score = Column(Float, nullable=True)
    candidate_eval_count = Column(Integer, default=0)
    candidate_composite_score = Column(Float, nullable=True)

    # Config
    min_evals = Column(Integer, default=50)                 # minimum evals before decision
    promotion_threshold = Column(Float, default=0.10)       # candidate must score 10% better

    # Outcome
    outcome = Column(String(20), nullable=True)             # promoted | discarded | timeout
    outcome_reason = Column(Text, nullable=True)

    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
```

New columns on existing models:

```python
# Add to SystemPrompt model
health_status = Column(String(20), default="unknown")       # unknown | healthy | degraded | critical
baseline_score = Column(Float, nullable=True)               # established baseline composite score
ab_test_id = Column(UUID(as_uuid=True), nullable=True)      # active A/B test if any

# Add to SystemPromptVersion model
is_candidate = Column(Boolean, default=False)               # true if auto-generated candidate
generated_by = Column(String(50), nullable=True)            # "auto_optimizer" | "admin" | "system"
generation_context = Column(JSONB, nullable=True)           # failure patterns that triggered generation
```

### 1C.2 Prompt Health Monitor Service

Uses APScheduler (already in codebase via `HeartbeatService`) to run every 5 minutes.

```python
# New file: orchestrator/core/services/prompt_health_monitor.py

class PromptHealthMonitor:
    """
    Scheduled service that monitors prompt quality and triggers self-healing.
    Runs every 5 minutes via APScheduler.
    """

    # Configurable thresholds
    WINDOW_SIZE = 20                    # last N eval runs to average
    DEGRADATION_THRESHOLD = 0.15       # 15% drop from baseline triggers action
    CRITICAL_THRESHOLD = 0.30          # 30% drop = critical
    FAILURE_RATE_THRESHOLD = 0.10      # 10% failure rate triggers action
    MIN_EVALS_FOR_BASELINE = 10        # need this many evals before establishing baseline
    STALE_HOURS = 24                   # no evals in 24h = stale

    async def check_all_prompts(self) -> None:
        """Main entry point — called by APScheduler every 5 min."""
        # 1. Query all prompts with eval enabled
        # 2. For each, compute rolling averages from eval_runs
        # 3. Compare to baseline
        # 4. Create PromptHealthSnapshot
        # 5. If degraded → trigger_healing()

    async def compute_rolling_scores(self, prompt_id, version_id) -> dict:
        """
        Query last WINDOW_SIZE eval_runs for this version.
        Returns: {
            "avg_completeness": float,
            "avg_helpfulness": float,
            "avg_conciseness": float,
            "composite_score": float,  # weighted: helpfulness*0.4 + completeness*0.35 + conciseness*0.25
            "failure_rate": float,
            "eval_count": int,
        }
        """

    async def establish_baseline(self, prompt_id, composite_score) -> None:
        """Set baseline_score on SystemPrompt once we have MIN_EVALS_FOR_BASELINE."""

    async def trigger_healing(self, prompt, snapshot) -> None:
        """
        Respond to degradation:
        1. If eval not enabled → enable it
        2. If degraded (>15% drop) → trigger optimization
        3. If critical (>30% drop) → alert admin + fast-track optimization
        """

    async def trigger_optimization(self, prompt, failure_runs) -> None:
        """
        Use LLM-as-judge to analyze failure patterns and generate improved prompt.
        1. Collect last N failed/low-score eval runs with reasons
        2. Send to Claude/GPT: "Here's the prompt, here are the failures, rewrite it"
        3. Create new SystemPromptVersion (status="draft", is_candidate=True)
        4. Create PromptABTest
        5. Log action in PromptHealthSnapshot
        """
```

### 1C.3 LLM-as-Judge Optimization

When FutureAGI optimize is unavailable or as a primary strategy, use own LLMs:

```python
# Add to prompt_health_monitor.py

OPTIMIZATION_PROMPT = """You are a prompt engineering expert. Analyze these evaluation results
for a system prompt and generate an improved version.

## Current System Prompt
{current_prompt}

## Purpose
{prompt_description}

## Recent Evaluation Failures (last {window_size} messages)
{failure_analysis}

## Score Trends
- Composite score: {composite_score} (baseline: {baseline_score}, delta: {degradation_delta}%)
- Helpfulness: {avg_helpfulness}
- Completeness: {avg_completeness}
- Conciseness: {avg_conciseness}
- Failure rate: {failure_rate}%

## Common Failure Reasons
{failure_reasons}

## Instructions
1. Identify the root causes of quality degradation
2. Rewrite the system prompt to address these issues
3. Preserve the core intent and personality
4. Do NOT change any variable placeholders like {{agent_name}} or {{tools_list}}

Return ONLY the improved prompt text, no explanation."""

async def generate_improved_prompt(self, prompt, failure_runs) -> str:
    """
    Call Claude/GPT to generate improved prompt based on failure analysis.
    Uses the orchestrator's existing LLM integration.
    """
    # Aggregate failure reasons from eval runs
    failure_reasons = []
    for run in failure_runs:
        if run.scores and "scores" in run.scores:
            for metric, data in run.scores["scores"].items():
                if data.get("failure"):
                    failure_reasons.append(f"- {metric}: {data.get('reason', 'no reason')}")

    # Call LLM (via existing orchestrator LLM client)
    from modules.orchestrator.llm.master_orchestrator import _call_llm  # or equivalent
    improved = await _call_llm(
        system_prompt="You are a prompt engineering expert.",
        user_message=OPTIMIZATION_PROMPT.format(...),
        model="claude-sonnet-4-20250514",  # fast + capable enough
    )
    return improved
```

### 1C.4 A/B Testing Infrastructure

The chat pipeline already selects the active prompt version. For A/B testing, we intercept that selection:

```python
# Modify PromptRegistry.get_prompt() or smart_orchestrator.py

async def get_prompt_version_for_request(self, prompt_slug: str) -> SystemPromptVersion:
    """
    Returns the prompt version to use.
    If an A/B test is running, randomly assigns traffic based on split ratio.
    """
    prompt = self.get_by_slug(prompt_slug)
    if not prompt or not prompt.ab_test_id:
        # Normal path — return active version
        return self.get_active_version(prompt_slug)

    ab_test = db.query(PromptABTest).filter(PromptABTest.id == prompt.ab_test_id).first()
    if not ab_test or ab_test.status != "running":
        return self.get_active_version(prompt_slug)

    # Random traffic split
    import random
    if random.random() < ab_test.traffic_split:
        return db.query(SystemPromptVersion).filter(
            SystemPromptVersion.id == ab_test.candidate_version_id
        ).first()
    else:
        return db.query(SystemPromptVersion).filter(
            SystemPromptVersion.id == ab_test.control_version_id
        ).first()
```

The eval_live_traffic hook already stores `version_id` on each `SystemPromptEvalRun`, so scores naturally accumulate per version. The Health Monitor compares scores between control and candidate versions to decide promotion.

### 1C.5 A/B Test Resolution

```python
# Add to prompt_health_monitor.py

async def check_ab_tests(self) -> None:
    """Check all running A/B tests for resolution."""
    # Query all running PromptABTests
    # For each:
    #   1. Count eval runs per version since test started
    #   2. If both versions have >= min_evals:
    #      a. Compare composite scores
    #      b. If candidate is >promotion_threshold better → promote
    #      c. If candidate is same or worse → discard
    #      d. If after 7 days neither → timeout + discard
    #   3. Log outcome

async def promote_candidate(self, ab_test) -> None:
    """Candidate won. Activate it, archive the old version."""
    # 1. Set candidate version status = "active"
    # 2. Set control version status = "archived"
    # 3. Update ab_test status = "completed", outcome = "promoted"
    # 4. Clear prompt.ab_test_id
    # 5. Reset prompt.baseline_score to candidate's composite score
    # 6. Log audit event

async def discard_candidate(self, ab_test, reason: str) -> None:
    """Candidate lost or timed out. Delete it."""
    # 1. Delete candidate version
    # 2. Update ab_test status = "completed", outcome = "discarded"
    # 3. Clear prompt.ab_test_id
    # 4. Log audit event
```

### 1C.6 Health Monitor Triggers

| Signal | Threshold | Action |
|--------|-----------|--------|
| Rolling quality score drops >15% from baseline | Over 20-eval window | Enable eval if off, trigger optimization |
| Rolling quality score drops >30% from baseline | Over 20-eval window | Critical alert to admin + fast-track optimization |
| Failure rate >10% | Over last 50 evals | Enable eval, trigger optimization |
| No eval data for >24h (stale) | Time-based | Auto-enable eval for heartbeat data |
| A/B candidate scores >10% better | After min 50 evals per version | Auto-promote candidate version |
| A/B test running >7 days with no winner | Time-based | Discard candidate, alert admin |

### 1C.7 Admin API Endpoints

```
GET    /api/admin/prompts/:id/health               — Health status + rolling scores + trend
GET    /api/admin/prompts/:id/health/history        — Historical health snapshots (for trend chart)
GET    /api/admin/prompts/health/overview            — All prompts health summary (dashboard)
GET    /api/admin/prompts/:id/ab-test               — Current A/B test status + scores
POST   /api/admin/prompts/:id/ab-test/resolve        — Manually resolve A/B test (promote/discard)
POST   /api/admin/prompts/:id/optimize               — Manually trigger optimization cycle
PATCH  /api/admin/prompts/:id/health/thresholds      — Adjust thresholds per prompt
GET    /api/admin/prompts/health/audit-log            — Global audit trail of auto-actions
```

### 1C.8 Frontend: Prompt Health Dashboard

New "Health" sub-tab in the prompt detail view (alongside existing Editor, Versions, Assessments):

**Health Overview Card**:
- Health status badge (healthy / degraded / critical / recovering)
- Composite score gauge (0-100) with baseline indicator
- Trend sparkline (last 24h of health snapshots)

**Rolling Metrics Chart** (Recharts — already in frontend deps):
- Line chart showing completeness, helpfulness, conciseness over time
- Baseline reference line
- Degradation threshold markers

**A/B Test Panel** (when active):
- Side-by-side score comparison: Control vs Candidate
- Eval count per version
- Projected winner based on current trend
- Manual override buttons: "Promote Now" / "Discard"
- Diff view of control vs candidate prompt text

**Audit Trail**:
- Chronological list of auto-healing actions
- Each entry: timestamp, action taken, reason, outcome
- Link to relevant prompt version

**Global Health Dashboard** (new section in Settings → System Prompts):
- Table of all prompts with health status, composite score, trend arrow
- Filter by status (healthy/degraded/critical)
- Sort by composite score or degradation delta

### 1C.9 Notifications

When the system takes auto-healing actions, notify admins:

| Event | Channel | Content |
|-------|---------|---------|
| Prompt degraded | In-app toast + audit log | "Personality prompt quality dropped 18%. Optimization triggered." |
| Optimization complete | In-app toast + audit log | "New candidate version v4 generated for Routing Classifier." |
| A/B test started | Audit log | "A/B test started: v3 (control) vs v4 (candidate), 50/50 split." |
| A/B test resolved | In-app toast + audit log | "Candidate v4 promoted — scored 14% better than v3." |
| Critical degradation | In-app toast + audit log | "CRITICAL: Task Decomposer quality dropped 35%. Manual review recommended." |

### 1C.10 Startup Integration

```python
# Add to main.py startup

# PRD-58 Phase 1C: Start Prompt Health Monitor
try:
    from core.services.prompt_health_monitor import prompt_health_monitor
    await prompt_health_monitor.start()  # registers APScheduler job
    logger.info("Prompt Health Monitor started (5-min interval)")
except Exception as e:
    logger.warning(f"Prompt Health Monitor failed to start: {e}")
```

### 1C.11 Implementation Steps

| # | Step | Files | Notes |
|---|------|-------|-------|
| 1 | Add `PromptHealthSnapshot` and `PromptABTest` models | `core/models/system_prompts.py` | New tables + Pydantic schemas |
| 2 | Add `health_status`, `baseline_score`, `ab_test_id` to `SystemPrompt` | `core/models/system_prompts.py` | Idempotent migration in main.py |
| 3 | Add `is_candidate`, `generated_by`, `generation_context` to `SystemPromptVersion` | `core/models/system_prompts.py` | Idempotent migration in main.py |
| 4 | Create `PromptHealthMonitor` service | `core/services/prompt_health_monitor.py` | APScheduler, rolling averages, degradation detection |
| 5 | Add LLM-as-judge optimization logic | `core/services/prompt_health_monitor.py` | Uses existing LLM client + Claude Sonnet |
| 6 | Add A/B test traffic splitting to PromptRegistry | `core/services/prompt_registry.py` or `smart_orchestrator.py` | Random split based on `traffic_split` |
| 7 | Add A/B test resolution logic | `core/services/prompt_health_monitor.py` | Check running tests on each monitor cycle |
| 8 | Add health API endpoints | `api/admin_prompts.py` | Health overview, history, A/B test, audit |
| 9 | Update `eval_live_traffic()` to tag version_id from A/B test | `core/services/futureagi_service.py` | Version already tracked, just verify A/B path |
| 10 | Frontend: Health sub-tab with metrics chart | `SystemPromptsTab.tsx` | Recharts line chart + health status badge |
| 11 | Frontend: A/B test panel | `SystemPromptsTab.tsx` | Side-by-side scores, promote/discard buttons |
| 12 | Frontend: Global health dashboard | `SystemPromptsTab.tsx` | Summary table of all prompt health statuses |
| 13 | Frontend: Audit trail list | `SystemPromptsTab.tsx` | Chronological list of auto-actions |
| 14 | Register health monitor on app startup | `main.py` | APScheduler job, 5-min interval |
| 15 | Test: degrade a prompt, verify auto-optimization fires | Manual | Temporarily worsen a prompt, observe healing |
| 16 | Test: A/B test lifecycle end-to-end | Manual | Candidate generated → traffic split → promoted/discarded |

### 1C.12 LLM-as-Judge Fallback Strategy

FutureAGI trial expires ~August 2026. The self-healing system must survive without it:

| Capability | FutureAGI Path | LLM-as-Judge Fallback |
|------------|---------------|----------------------|
| Live traffic scoring | Worker `/score` endpoint | Claude Haiku judges (input, output) against rubric |
| Safety scanning | Worker `/safety` endpoint | Claude scans for toxicity/injection using system prompt |
| Optimization | Worker `/optimize` endpoint | Claude Sonnet rewrites based on failure patterns (already built in 1C) |
| Quality metrics | FutureAGI templates (completeness, etc.) | Custom rubrics scored 0-1 by Claude |

**Fallback activation**: When FutureAGI worker returns errors for >1 hour, auto-switch to LLM-as-judge mode. Log the switch. Admin can manually toggle in settings.

```python
# Rubric for LLM-as-judge fallback
LLM_JUDGE_RUBRIC = """Score this response on a scale of 0.0 to 1.0 for each metric.

Input: {input}
Output: {output}

Metrics:
1. completeness (0-1): Does the response fully address the input?
2. helpfulness (0-1): Is the response practically useful?
3. conciseness (0-1): Is it appropriately concise without losing value?

Return JSON: {"completeness": 0.X, "helpfulness": 0.X, "conciseness": 0.X, "failure": bool, "reason": "..."}"""
```

---

## Phase 2: User-Facing Features — PLANNED

**Prerequisite**: Phase 1B complete (eval data flowing), Phase 1C desirable but not blocking.

Phase 2 brings prompt quality data out of the admin settings and into user-facing surfaces where it drives real product value.

### 2A. Model Comparison Modal

**Goal**: Wire the existing marketplace Compare button to a real comparison view backed by FutureAGI quality scores.

**Current State**: `frontend/components/marketplace/marketplace-llms-tab.tsx` and `llm-model-detail-modal.tsx` exist. Compare button present but wired to static/heuristic data.

#### 2A.1 How It Works

1. Admin selects 2-3 models to compare in the LLM Marketplace
2. System runs the same set of test prompts through each model
3. FutureAGI scores each model's output on completeness, helpfulness, conciseness
4. Results displayed side-by-side in a comparison modal

#### 2A.2 Backend

```python
# New file: orchestrator/api/model_comparison.py

@router.post("/api/marketplace/models/compare")
async def compare_models(
    request: ModelComparisonRequest,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """
    Compare 2-3 models by running the same prompts through each
    and scoring with FutureAGI.
    """

class ModelComparisonRequest(BaseModel):
    model_ids: List[str]                    # e.g. ["gpt-4o", "claude-sonnet", "gemini-pro"]
    test_prompts: Optional[List[str]]       # custom prompts, or use defaults
    metrics: List[str] = ["completeness", "is_helpful", "is_concise"]

class ModelComparisonResult(BaseModel):
    comparison_id: str
    status: str                             # pending | running | completed
    models: List[ModelScorecard]

class ModelScorecard(BaseModel):
    model_id: str
    model_name: str
    scores: Dict[str, float]               # metric → average score
    composite_score: float
    per_prompt_results: List[PromptResult]  # individual prompt scores
    avg_latency_ms: float
    estimated_cost_per_1k: float
```

**Test Prompt Defaults** (used when admin doesn't provide custom):
```python
DEFAULT_COMPARISON_PROMPTS = [
    "Explain quantum computing to a 10-year-old in 3 sentences.",
    "Write a Python function that finds the longest palindromic substring.",
    "Summarize the key differences between REST and GraphQL APIs.",
    "Draft a professional email declining a meeting invitation politely.",
    "Analyze this dataset trend: [Q1: 100, Q2: 95, Q3: 110, Q4: 130]",
]
```

#### 2A.3 Frontend

**Comparison Modal** (extends `llm-model-detail-modal.tsx`):
- Side-by-side columns, one per model
- Radar chart (Recharts) showing metric scores overlaid
- Bar chart comparing composite scores
- Expandable rows showing individual prompt results
- Latency and cost comparison row
- "Select Winner" button that highlights recommended model

#### 2A.4 Implementation Steps

| # | Step | Files |
|---|------|-------|
| 1 | Create `ModelComparisonRequest`/`Result` schemas | `core/models/marketplace.py` (or new file) |
| 2 | Create `/api/marketplace/models/compare` endpoint | `api/model_comparison.py` |
| 3 | Background worker: run prompts through each model, score with FutureAGI | `core/services/model_comparison_service.py` |
| 4 | Store comparison results (new table or use eval_runs with run_type="comparison") | `core/models/` |
| 5 | Frontend: Comparison modal with radar + bar charts | `marketplace/model-comparison-modal.tsx` |
| 6 | Wire Compare button in `marketplace-llms-tab.tsx` | `marketplace/marketplace-llms-tab.tsx` |

---

### 2B. Enhanced Recipe Suggestions

**Goal**: Replace heuristic `quality_score` in recipe learning with real FutureAGI eval scores after recipe execution.

**Current State**: `recipe_learning_service.py`, `recipe_quality_service.py`, and `recipe_memory_service.py` exist. Quality scores are computed via heuristics in `orchestrator/modules/orchestrator/tracker.py` (~line 238: `quality_score: float`). Recipes tab in frontend at `frontend/components/workflows/recipes-tab.tsx`.

#### 2B.1 How It Works

1. User executes a recipe (workflow)
2. Orchestrator runs the recipe, produces output
3. **New**: Fire-and-forget FutureAGI eval on recipe input/output (same as live chat eval)
4. Score stored alongside recipe execution result
5. Recipe suggestions panel shows actual quality scores instead of heuristic
6. Recipe learning uses real scores to rank improvement suggestions

#### 2B.2 Backend Changes

```python
# Modify: orchestrator/core/services/recipe_quality_service.py

async def score_recipe_execution(
    self,
    recipe_id: str,
    input_text: str,
    output_text: str,
    execution_id: str,
) -> Dict[str, float]:
    """
    Score a recipe execution using FutureAGI.
    Called fire-and-forget after recipe completes.
    Returns: {"completeness": 0.X, "helpfulness": 0.X, "conciseness": 0.X, "composite": 0.X}
    """
    from core.services.futureagi_service import futureagi_service

    if not futureagi_service.is_available:
        return self._heuristic_score(output_text)  # fallback

    result = await futureagi_service._call_worker("/score", {
        "input_text": input_text,
        "output_text": output_text,
        "metrics": ["completeness", "is_helpful", "is_concise"],
    })

    scores = result.get("scores", {})
    # Store in recipe_executions table
    # Update rolling recipe quality average
    return scores
```

```python
# Modify: orchestrator/core/services/recipe_learning_service.py

# Replace heuristic quality_score references with FutureAGI scores
# When generating improvement suggestions, include real eval data:
# "This recipe scored 0.65 on helpfulness (below 0.8 threshold). Suggest improvements."
```

#### 2B.3 Frontend Changes

- `recipes-tab.tsx`: Show actual quality scores with color coding (green >0.8, yellow 0.6-0.8, red <0.6)
- `recipe-suggestions-panel.tsx`: Rank suggestions by real quality delta, not heuristic
- `recipe-preview-panel.tsx`: Show per-execution quality trend chart
- `view-recipe-modal.tsx`: Quality score breakdown in execution history

#### 2B.4 Implementation Steps

| # | Step | Files |
|---|------|-------|
| 1 | Add FutureAGI scoring hook after recipe execution | `core/services/recipe_quality_service.py` |
| 2 | Store real scores in recipe_executions table | `core/models/` (add eval_scores JSONB column) |
| 3 | Update recipe_learning_service to use real scores | `core/services/recipe_learning_service.py` |
| 4 | Frontend: Quality score badges on recipe cards | `workflows/recipes-tab.tsx` |
| 5 | Frontend: Quality trend chart in recipe detail | `workflows/recipe-preview-panel.tsx` |
| 6 | Frontend: Score-ranked suggestion panel | `workflows/recipe-suggestions-panel.tsx` |

---

### 2C. Agent "Test My Prompt" Button

**Goal**: In the agent creation wizard, let users test their system prompt with generated scenarios and get quality scores before deploying.

**Current State**: `frontend/components/agents/create-agent-modal.tsx` has the agent creation flow. Backend agent factory at `modules/agents/factory/agent_factory.py`.

#### 2C.1 How It Works

1. User writes a system prompt in agent creation modal
2. Clicks "Test My Prompt" button
3. System generates 3-5 test scenarios relevant to the prompt's purpose
4. Runs each scenario through the prompt + selected LLM
5. Scores each response with FutureAGI
6. Displays results inline: pass/fail per scenario, overall quality score
7. User can iterate on prompt, re-test, then save

#### 2C.2 Backend

```python
# New endpoint in api/agents.py or new file api/prompt_testing.py

@router.post("/api/agents/test-prompt")
async def test_agent_prompt(
    request: TestPromptRequest,
    background_tasks: BackgroundTasks,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Generate test scenarios and evaluate a system prompt."""

class TestPromptRequest(BaseModel):
    system_prompt: str
    model_id: str = "gpt-4o-mini"           # model to test with
    purpose: Optional[str] = None            # "customer support agent", etc.
    num_scenarios: int = 5

class TestPromptResponse(BaseModel):
    test_id: str
    status: str
    scenarios: List[ScenarioResult]
    overall_score: float
    recommendations: List[str]

class ScenarioResult(BaseModel):
    scenario: str                            # the test input
    response: str                            # LLM's response
    scores: Dict[str, float]                 # completeness, helpfulness, conciseness
    passed: bool                             # composite > 0.7
    feedback: str                            # specific improvement suggestion
```

**Scenario Generation** (uses LLM):
```python
SCENARIO_GEN_PROMPT = """Generate {num} realistic test scenarios for this AI agent.

System Prompt: {system_prompt}
Purpose: {purpose}

For each scenario, create a user message that would test a different capability
of this agent. Include edge cases and challenging inputs.

Return JSON array: [{"scenario": "user message", "expected_behavior": "what good looks like"}]"""
```

#### 2C.3 Frontend Changes

Add to `create-agent-modal.tsx`:
- "Test My Prompt" button below the system prompt textarea
- Loading state with progress (testing scenario 2/5...)
- Results panel showing:
  - Per-scenario: input, response, score badges, pass/fail
  - Overall score with gauge visualization
  - Specific recommendations for improvement
  - "Re-test" button after edits
- Quality gate: warning if overall score < 0.7 when saving

#### 2C.4 Implementation Steps

| # | Step | Files |
|---|------|-------|
| 1 | Create test-prompt endpoint with scenario generation | `api/prompt_testing.py` |
| 2 | LLM scenario generator | `core/services/prompt_testing_service.py` |
| 3 | Run scenarios through model + FutureAGI scoring | `core/services/prompt_testing_service.py` |
| 4 | Frontend: Test button + results panel in create-agent-modal | `agents/create-agent-modal.tsx` |
| 5 | Frontend: Quality gate warning on save | `agents/create-agent-modal.tsx` |
| 6 | Cache test results so re-opening modal shows last test | `api/prompt_testing.py` |

---

### 2D. Quality-Aware Model Recommendations

**Goal**: Replace cost-only model recommendations in analytics with recommendations backed by real eval data.

**Current State**: `frontend/components/analytics/analytics-page.tsx` and related analytics components exist. `api/benchmarking.py` has quality_score fields (heuristic). The orchestrator tracker records quality metrics per execution.

#### 2D.1 How It Works

1. Live traffic eval (Phase 1B) accumulates quality scores per model used
2. Analytics dashboard aggregates: model × quality × cost × latency
3. Recommendations engine: "Switch Routing Classifier from GPT-4o to Claude Sonnet — 12% better quality at 40% lower cost"
4. Confidence intervals based on eval volume

#### 2D.2 Backend

```python
# New file or extend: orchestrator/api/model_recommendations.py

@router.get("/api/analytics/model-recommendations")
async def get_model_recommendations(
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """
    Analyze eval data across models and generate optimization recommendations.
    """

class ModelRecommendation(BaseModel):
    prompt_slug: str                         # which system prompt
    current_model: str
    recommended_model: str
    quality_delta: float                     # +12% = recommended is 12% better
    cost_delta: float                        # -40% = recommended is 40% cheaper
    latency_delta: float                     # -200ms = recommended is 200ms faster
    confidence: str                          # low | medium | high (based on eval count)
    eval_count: int                          # how many evals back this recommendation
    reasoning: str                           # human-readable explanation
```

**Recommendation Logic**:
```python
async def generate_recommendations(self) -> List[ModelRecommendation]:
    """
    For each system prompt with eval data:
    1. Group eval_runs by model used (tag model in eval metadata)
    2. Compare composite scores across models
    3. Factor in cost (from model pricing table) and latency
    4. Recommend if: better quality OR same quality + lower cost
    5. Require minimum 20 evals per model for medium confidence, 50 for high
    """
```

#### 2D.3 Frontend Changes

**New section in Analytics page** (`analytics-page.tsx`):
- "Model Optimization Recommendations" card
- Table: prompt name, current model, recommended model, quality delta, cost delta
- Confidence badge per recommendation
- "Apply" button that updates the prompt's model config
- Historical view: quality scores per model over time (line chart)

**Enhance existing analytics**:
- `analytics-overview.tsx`: Add "Prompt Quality" section with aggregate scores
- `analytics-costs.tsx`: Overlay quality scores on cost charts to show cost/quality tradeoff
- `analytics-llm-usage.tsx`: Quality column in model usage breakdown table

#### 2D.4 Prerequisites

This feature requires tracking which model produced each eval run. Needs a small change to `eval_live_traffic()`:

```python
# In futureagi_service.py, add model info to eval run metadata
run = SystemPromptEvalRun(
    ...
    metadata_={"source": "live_traffic", "model": model_id, "latency_ms": latency},
)
```

#### 2D.5 Implementation Steps

| # | Step | Files |
|---|------|-------|
| 1 | Tag model + latency in live eval run metadata | `core/services/futureagi_service.py` |
| 2 | Create model recommendations endpoint | `api/model_recommendations.py` |
| 3 | Recommendation engine: aggregate scores by model, compare | `core/services/model_recommendation_service.py` |
| 4 | Frontend: Recommendations card in analytics | `analytics/analytics-page.tsx` |
| 5 | Frontend: Quality column in LLM usage table | `analytics/analytics-llm-usage.tsx` |
| 6 | Frontend: Cost/quality tradeoff chart | `analytics/analytics-costs.tsx` |

---

### Phase 2 Priority Order

| Phase | Value | Effort | Depends On | Recommended Order |
|-------|-------|--------|------------|-------------------|
| 2C — Test My Prompt | High (user-facing quality) | Medium | 1B only | 1st — immediate user value |
| 2B — Recipe Quality | High (real scores > heuristics) | Medium | 1B only | 2nd — replaces hack with real data |
| 2D — Model Recommendations | High (cost savings) | Medium-High | 1B + model tagging | 3rd — needs eval data volume |
| 2A — Model Comparison | Medium (nice-to-have) | High | 1B + multi-model infra | 4th — most infrastructure work |

---

## Environment Variables

```bash
# Worker service (Railway internal network)
AGENT_OPT_WORKER_URL=http://agent-opt-worker.railway.internal:8080

# FutureAGI credentials (on worker service, not orchestrator)
FUTUREAGI_API_KEY=9f3b49d56df74dff9dd671dc631efb02
FUTUREAGI_SECRET_KEY=650ff31fcc554d42b8e71cb6b6c86c0f
```

---

## FutureAGI API Reference (Direct HTTP)

Reverse-engineered from SDK source + API testing on Feb 18, 2026.

### Evaluation Endpoint
```
POST https://api.futureagi.com/sdk/api/v1/new-eval/
Headers: X-Api-Key, X-Secret-Key, Content-Type: application/json
Body: {
  "eval_name": "<template_name>",
  "inputs": {"input": ["..."], "output": ["..."], "context": ["..."]},
  "model": "<model_name>"
}
Response: {
  "result": [{
    "evaluations": [{
      "failure": bool,
      "reason": "detailed explanation",
      "metrics": [{"id": "metric_name", "value": 0.0-1.0}],
      "metadata": {}
    }]
  }]
}
```

### Available Templates
```
GET https://api.futureagi.com/sdk/api/v1/get-evals/
```

### Improve Prompt (ASYNC — returns job ID)
```
POST https://api.futureagi.com/model-hub/prompt-templates/improve-prompt/
Body: {"existing_prompt": "...", "improvement_requirements": "..."}
Response: {"status": true, "result": {"improveId": "improve_xxx"}}
```

### Models
- Quality assessment: `turing_large`, `turing_small`, `turing_flash`
- Safety scanning: `protect`, `protect_flash`

---

## Security & Privacy

- All `/api/admin/prompts/*` endpoints require authenticated Clerk/API-key users
- FutureAGI API keys live on the worker service only, not in orchestrator
- Live traffic eval sends user messages to worker → FutureAGI API — confirm acceptable under data policy
- Eval toggle is per-prompt, admin-controlled — not automatically enabled
- Phase 1C auto-optimization creates full audit trail of every decision (PromptHealthSnapshot + PromptABTest)
- Phase 1C auto-generated prompts always start as "draft" candidates, never go live without A/B validation
- Phase 2C "Test My Prompt" generates synthetic scenarios, does NOT use real user data
- Safety scanning runs on prompt text only, not user content

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| FutureAGI trial expires (Aug 2026) | Lose eval capability | LLM-as-judge fallback designed in Phase 1C.12. Auto-switches on worker failure. |
| FutureAGI API rate limits unknown | Eval calls may be throttled | Start with 1-2 prompts enabled, monitor. Sampling strategy in Phase 1B. |
| Live eval sends user data to FutureAGI | Privacy concern | Admin-controlled toggle. Can be disabled entirely. Worker isolates data flow. |
| FutureAGI API instability | 400 errors, timeouts | Worker handles retries. Graceful degradation — never blocks chat. |
| Worker service unavailable | All eval stops | Orchestrator checks `is_available`. Chat unaffected. Health monitor logs gaps. |
| Auto-optimizer makes things worse (1C) | Quality degrades further | A/B testing with min 50 evals before promotion. Auto-rollback if candidate worse. Admin override available. |
| A/B test introduces inconsistency | Users get different quality | 50/50 split means short exposure. Max 7-day test window with auto-timeout. |
| Health monitor false positives | Unnecessary optimizations | Conservative thresholds (15% degradation). Min 10 evals for baseline. Audit trail for review. |
| Phase 2C token costs | User-facing cost for prompt testing | 5 scenarios × model cost per test. Consider warning or limiting daily tests. |

---

## Open Questions (Updated)

1. ~~Dataset creation~~ → **Resolved: Live traffic eval replaces synthetic datasets**
2. ~~Eval frequency~~ → **Resolved: Every message when toggle is ON**
3. **Rate limiting**: What's FutureAGI's API rate limit? Need to test before enabling on high-traffic prompts
4. **Sampling**: For high-traffic prompts, should we eval every message or sample (e.g., 1 in 10)?
5. **Privacy**: User messages sent to FutureAGI — need to confirm data handling policy
6. ~~Phase 1C triggers~~ → **Resolved: Detailed thresholds defined in 1C.6 — start conservative**
7. ~~FutureAGI fallback~~ → **Resolved: LLM-as-judge fallback designed in 1C.12**
8. **A/B test traffic split**: Should split be configurable per test or fixed at 50/50? (Spec says configurable via `traffic_split` field)
9. **Multi-model eval tagging**: Phase 2D needs model ID tagged on each eval run — requires chat pipeline to pass model info through to eval hook
10. **Phase 2C cost**: Test My Prompt runs real LLM calls (5 scenarios × model cost) — should we warn user about token usage?

---

## Fresh Context Quick Start

**If starting a new Claude Code session, read this section first.**

### Branch & Worktree

```bash
# ALL work happens here — NOT in automatos-ai
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai-futureAGI
git branch  # Must be: futureAGI
```

Railway deploys `automatos-ai-api` and `automotas-ai-frontend` from this branch.

### Key File Map

| File | What | Notes |
|------|------|-------|
| `orchestrator/core/models/system_prompts.py` | ORM models: `SystemPrompt`, `SystemPromptVersion`, `SystemPromptEvalRun` + Pydantic schemas | Phase 1C adds `PromptHealthSnapshot`, `PromptABTest` here |
| `orchestrator/api/admin_prompts.py` | FastAPI endpoints for prompt CRUD + assessment + eval-toggle | Phase 1C adds health/ab-test endpoints |
| `orchestrator/core/services/futureagi_service.py` | Routes all FutureAGI ops to worker service. Has `eval_live_traffic()`, `assess_prompt()`, `optimize_prompt()` | Singleton at `futureagi_service` |
| `orchestrator/services/heartbeat_service.py` | APScheduler-based periodic task runner | Pattern to follow for Phase 1C health monitor |
| `orchestrator/core/metadata_cache/scheduler.py` | `schedule` lib-based daily sync | Alternative scheduling pattern |
| `orchestrator/core/seeds/seed_system_prompts.py` | Seeds 15 prompts on startup | |
| `orchestrator/core/database/database.py` | `SessionLocal`, `get_db`, `create_tables` | Use `SessionLocal()` in background tasks |
| `orchestrator/consumers/chatbot/service.py` | Chat pipeline with 3 eval hook points (lines 746, 1226, 1499) | Fire-and-forget via `asyncio.create_task()` |
| `orchestrator/consumers/chatbot/integration.py` | `SmartChatIntegration` + `OrchestratedRequest` dataclass | Carries `system_prompt` field |
| `orchestrator/consumers/chatbot/smart_orchestrator.py` | Builds system prompt via `get_happy_system_prompt()` | Phase 1C A/B test version selection goes here |
| `orchestrator/main.py` | FastAPI app — imports routers, runs startup | Phase 1C adds health monitor startup |
| `frontend/components/settings/SystemPromptsTab.tsx` | Full prompt management UI — list, editor, versions, assessments, eval toggle | Phase 1C adds Health sub-tab |
| `frontend/lib/api-client.ts` | API client class | **GOTCHA**: Use `apiClient.request()` NOT `apiClient()` |
| `frontend/components/agents/create-agent-modal.tsx` | Agent creation wizard | Phase 2C adds "Test My Prompt" button |
| `frontend/components/marketplace/marketplace-llms-tab.tsx` | LLM model marketplace | Phase 2A adds comparison modal |
| `frontend/components/workflows/recipes-tab.tsx` | Recipe management | Phase 2B adds real quality scores |
| `frontend/components/analytics/analytics-page.tsx` | Analytics dashboard | Phase 2D adds model recommendations |

### Critical Gotchas (learned the hard way)

1. **Wrong directory**: Work in `automatos-ai-futureAGI`, NOT `automatos-ai`. Shell cwd resets between commands.
2. **apiClient pattern**: `apiClient` is `new ApiClient()` — call `.request(endpoint, options)`, not as a function.
3. **BackgroundTasks**: The admin_prompts.py `trigger_assessment` uses FastAPI `BackgroundTasks` with `asyncio.run()` inside a sync wrapper. This is the correct pattern for dispatching async work from sync endpoints.
4. **SessionLocal in background**: Background tasks can't use request-scoped `get_db`. Use `SessionLocal()` directly and close in `finally`.
5. **Worker service**: FutureAGI calls go to `agent-opt-worker` at `AGENT_OPT_WORKER_URL` (Railway internal). Worker owns API keys. Orchestrator just posts to `/assess`, `/safety`, `/optimize`, `/score`.
6. **APScheduler in codebase**: `HeartbeatService` uses `AsyncIOScheduler` from APScheduler. Follow this pattern for Phase 1C health monitor.
7. **Recharts in frontend**: Already a dependency — use for Phase 1C health charts and Phase 2 quality visualizations.
8. **Optimize works via worker**: Worker `/optimize` collects dataset from live traffic and runs FutureAGI optimize. No longer broken (was broken with direct SDK).

### Database Access (Railway)

```bash
PGPASSWORD=alrckxcy2fgvy7zxzhhv0wa37gtc690w psql -h shortline.proxy.rlwy.net -p 47906 -U postgres -d railway
```

### Railway CLI

```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai-futureAGI
railway link --project <project-id> --service automatos-ai-api  # if not linked
railway logs -d  # deploy logs
railway logs -d --filter "FutureAGI"  # filtered
```

### Worker Service

```bash
# FutureAGI API keys live on the worker service, not orchestrator
# Orchestrator env var:
AGENT_OPT_WORKER_URL=http://agent-opt-worker.railway.internal:8080

# Worker env vars:
FUTUREAGI_API_KEY=9f3b49d56df74dff9dd671dc631efb02
FUTUREAGI_SECRET_KEY=650ff31fcc554d42b8e71cb6b6c86c0f
```

### Git History (futureAGI branch, most recent first)

```
d4eac9a fix: Render assess/safety/optimize results correctly per run type
60a46e2 fix: Add assessment polling, loading states, and auto-tab switch
0667f74 fix: Replace prompt_adherence (broken API), fix optimize payload, add logging
6b8936d fix: Per-template input keys/models + concurrent execution for FutureAGI
1d80cb6 fix: Use FutureAGI model names (turing_large/protect) and fix optimize payload
f076e47 fix: Replace FutureAGI SDK with direct HTTP API calls
b80b1d7 fix: Use apiClient.request() instead of apiClient() in SystemPromptsTab
a02a002 fix: Run create_tables + seed_system_prompts on startup for Railway
5157daa fix: Allow authenticated Clerk/API-key users to access admin prompts
```

---

## Build Progress Tracker

### Phase 1A — COMPLETE ✅
All components shipped. See Phase 1A section above.

### Phase 1B — COMPLETE ✅

| # | Step | Status | Files | Notes |
|---|------|--------|-------|-------|
| 1 | Add `futureagi_eval_enabled` column to `SystemPrompt` model | ✅ DONE | `core/models/system_prompts.py` | Boolean, default False. Added to `PromptResponse` schema. |
| 2 | Add PATCH `/futureagi-toggle` API endpoint | ✅ DONE | `api/admin_prompts.py` | Toggle on/off, returns updated prompt |
| 3 | Add toggle switch to frontend prompt detail view | ✅ DONE | `SystemPromptsTab.tsx` | Toggle in assessments tab with green/grey styling |
| 4 | Add `eval_live_traffic()` method to FutureAGIService | ✅ DONE | `core/services/futureagi_service.py` | Routes to worker `/score`, stores as "live" run |
| 5 | Hook into chat pipeline after LLM response | ✅ DONE | `consumers/chatbot/service.py` | Fire-and-forget at 3 hook points (lines 746, 1226, 1499) |
| 6 | Store per-message eval results in eval_runs table | ✅ DONE | Covered by step 4 | Links to prompt_id + version_id, run_type="live" |
| 7 | Display accumulated scores in assessments dashboard | ✅ DONE | `SystemPromptsTab.tsx` | "live" runs render same as "assess" runs |
| 8 | Test end-to-end: toggle ON → send chat → verify scores | ✅ DONE | Manual | Deployed and working via worker service |
| 9 | Idempotent column migration on startup | ✅ DONE | `main.py` | ALTER TABLE ADD COLUMN IF NOT EXISTS |

### Phase 1C — NOT STARTED

See Phase 1C section above for 16 implementation steps.

**Next up:** Step 1 — Add `PromptHealthSnapshot` and `PromptABTest` models

### Phase 2 — NOT STARTED

See Phase 2 section above. Recommended order: 2C → 2B → 2D → 2A.

**Last updated:** 2026-02-19
**Architecture:** Orchestrator → `agent-opt-worker` service (Railway internal HTTP)
**Note:** Architecture evolved from direct FutureAGI HTTP to isolated worker service. All FutureAGI API concerns live in the worker.
