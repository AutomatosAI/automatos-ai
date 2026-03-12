# PRD-80: Unified Context Service

**Version:** 1.0
**Status:** Draft
**Priority:** P0
**Author:** Gar Kavanagh + Claude
**Created:** 2026-03-12
**Updated:** 2026-03-12
**Dependencies:** PRD-64 (Unified Action Discovery — COMPLETE), PRD-68 (Progressive Complexity Routing — COMPLETE), PRD-71 (Unified Skills — COMPLETE), PRD-76 (Agent Reporting — COMPLETE)

---

## Executive Summary

Every time we fix a prompt, tool loading, or memory injection bug, we have to patch it in 3–5 different places. The chatbot builds prompts one way (`smart_orchestrator.py` → `personality.py`). Agent task execution builds them another way (`agent_factory.py` → `_build_agent_system_prompt`). Heartbeats copy-paste from the factory. Recipes have their own path. The orchestrator stages have yet another. This fragmentation has caused:

- **Chatbot missing platform actions** — fixed in commit `4a8d7e3` but only because someone noticed Auto couldn't see `platform_execute`
- **Composio tools loaded differently** per code path — factory built typed schemas, chatbot used generic `composio_execute`
- **Memory injected in different formats** — chatbot via `get_happy_system_prompt(memories=...)`, factory via string concatenation, heartbeat skipped entirely
- **Tool count explosion** — factory sent 107 tools, chatbot sent 76, heartbeat sent all platform tools individually before the dispatcher fix
- **Daily logs, action summaries, personality** all wired independently into each path with different bugs each time

This PRD introduces a **single `ContextService`** that every LLM-calling code path uses. One place to build prompts, load tools, inject memory, manage token budgets. Fix it once, fixed everywhere.

### What We're Building

1. **`modules/context/` package** — new module containing the unified context service
2. **`ContextService` class** — single entry point: `build_context(agent, mode, messages) → ContextResult`
3. **Composable prompt sections** — identity, skills, platform actions, memory, tools, task context assembled declaratively
4. **Token budget manager** — sections have priority weights; low-priority content gets trimmed first when approaching limits
5. **`ContextResult` dataclass** — contains `system_prompt`, `tools`, `tool_choice`, `messages`, `metadata` ready for any LLM call
6. **Migration of all 9 code paths** to use `ContextService` instead of building prompts/tools themselves
7. **Dead code cleanup** — remove `_build_agent_system_prompt`, `get_happy_system_prompt` complexity, duplicated tool loading

### What We're NOT Building

- A prompt versioning UI (future PRD — admin prompt editing stays in `system_prompts` table)
- A/B testing framework for prompts (future)
- LLM-specific prompt formatting (we target OpenAI chat format; provider adapters stay in `llm_manager.py`)
- New memory system (Mem0 stays; we just standardize how memories are injected into context)

---

## 1. The Problem: 9 Fragmented Code Paths

Every path that calls an LLM independently builds its own prompt, loads its own tools, and injects its own context. Here's the current state:

| # | Code Path | File | Builds Prompt | Loads Tools | Injects Memory | Injects Platform Actions |
|---|-----------|------|---------------|-------------|----------------|--------------------------|
| 1 | **Chatbot (Auto)** | `consumers/chatbot/smart_orchestrator.py` | `get_happy_system_prompt()` | `smart_tool_router.route()` | `smart_memory.retrieve_memories()` | `build_prompt_summary()` (added 2026-03-12) |
| 2 | **Agent Task Execution** | `modules/agents/factory/agent_factory.py` | `_build_agent_system_prompt()` | `get_tools_for_agent()` | String concatenation in prompt | `build_prompt_summary()` |
| 3 | **Heartbeat Service** | `services/heartbeat_service.py` | Inline f-string | `to_dispatcher_schema()` only | None | Inline summary |
| 4 | **Recipe Executor** | `api/recipe_executor.py` | Recipe-specific prompt + agent prompt | Inherits from factory | None | None |
| 5 | **Execution Manager** | `modules/agents/execution/execution_manager.py` | Delegates to factory | Delegates to factory | None | Via factory |
| 6 | **Universal Router** | `core/routing/engine.py` | Per-tier prompts | Per-tier tool selection | None | None |
| 7 | **Orchestrator Stages** | `modules/orchestrator/stages/*.py` | Per-stage prompts | None (LLM-only) | None | None |
| 8 | **Board Task Chat** | Via chatbot path | Via chatbot | Via chatbot | Via chatbot | Via chatbot |
| 9 | **NL2SQL** | `modules/nl2sql/service.py` | Schema-specific prompt | None (text-only) | None | None |

### What Goes Wrong

1. **Feature added to one path, missing from others** — platform action summary was added to factory + chatbot but not heartbeat or recipes until manually patched
2. **Memory format divergence** — chatbot formats memories as bullet list via `personality.py`, factory dumps raw strings, heartbeat gets nothing
3. **Tool count inconsistency** — factory sends `get_tools_for_agent()` (core + platform dispatcher + composio), chatbot sends `smart_tool_router.route()` (filtered subset), heartbeat sends only platform dispatcher
4. **Personality applied inconsistently** — chatbot uses `AutomatosPersonality` with workspace settings, factory uses basic system prompt, heartbeat uses hardcoded text
5. **No token awareness** — prompts grow unbounded; adding platform action summary + daily logs + memory + skill instructions can exceed model context with zero warnings

---

## 2. Architecture: The Unified Context Service

### 2.1 Design Principles (from Context Engineering)

Inspired by David Kamm & IBM's [Context Engineering](https://github.com/dkamm/context-engineering) framework:

1. **Composable Sections** — Context is built from independent sections (atoms → molecules → cells), each responsible for one concern. Sections can be included, excluded, or reordered without touching other sections.
2. **Token Budgets as First-Class Constraints** — Every section declares its priority and max token allocation. The assembly pipeline respects a total budget and trims low-priority sections first.
3. **Declarative Mode-Based Assembly** — Each "mode" (chatbot, task_execution, heartbeat, etc.) declares which sections it needs and in what order. No imperative if/else chains.
4. **Single Source of Truth** — One module owns prompt construction. Callers provide context (agent, task, messages), service returns ready-to-send payload.

### 2.2 Module Structure

```
orchestrator/modules/context/
├── __init__.py                  # Exports: ContextService, ContextResult, ContextMode
├── service.py                   # ContextService — the main entry point
├── result.py                    # ContextResult dataclass
├── modes.py                     # ContextMode enum + mode configurations
├── budget.py                    # TokenBudgetManager — estimates + trims sections
├── sections/
│   ├── __init__.py
│   ├── base.py                  # BaseSection ABC — priority, max_tokens, render()
│   ├── identity.py              # Agent name, role, persona, personality
│   ├── skills.py                # SKILL.md content from agent's assigned skill
│   ├── platform_actions.py      # platform_execute action catalog (from ActionRegistry)
│   ├── memory.py                # Mem0 memories + daily logs
│   ├── tools.py                 # Tool schema assembly (core + dispatcher + composio)
│   ├── task_context.py          # Current task description, board context
│   ├── recipe_context.py        # Recipe step instructions
│   ├── datetime_context.py      # Current UTC datetime
│   ├── conversation.py          # Message history formatting + trimming
│   └── custom.py                # Workspace-level custom prompt sections (DB-driven)
└── estimator.py                 # Token count estimator (tiktoken or char-based)
```

### 2.3 Core Interfaces

```python
# --- result.py ---

@dataclass(frozen=True)
class ContextResult:
    """Immutable result from ContextService.build_context()."""

    # Ready for LLM call
    system_prompt: str
    messages: list[dict[str, str]]
    tools: list[dict[str, Any]]
    tool_choice: str                   # "auto", "required", "none"

    # Metadata for logging/debugging
    mode: str
    sections_included: list[str]       # Which sections made it into the prompt
    sections_trimmed: list[str]        # Which sections were cut for budget
    token_estimate: int                # Estimated total tokens
    token_budget: int                  # Max allowed
    memory_context: Optional[str]      # For SSE events (US-015)
    user_name: Optional[str]
    preparation_time_ms: float


# --- modes.py ---

class ContextMode(str, Enum):
    CHATBOT = "chatbot"
    TASK_EXECUTION = "task_execution"
    HEARTBEAT = "heartbeat"
    RECIPE = "recipe"
    ROUTER = "router"
    ORCHESTRATOR_STAGE = "orchestrator_stage"
    NL2SQL = "nl2sql"


# Mode → ordered list of section classes + config overrides
MODE_CONFIGS: dict[ContextMode, ModeConfig] = {
    ContextMode.CHATBOT: ModeConfig(
        sections=[
            "identity", "skills", "platform_actions", "memory",
            "datetime_context", "conversation"
        ],
        tool_loading="filtered",        # Uses intent-based filtering
        personality=True,
        max_tokens=None,                # Use model default
    ),
    ContextMode.TASK_EXECUTION: ModeConfig(
        sections=[
            "identity", "skills", "platform_actions", "memory",
            "task_context", "datetime_context", "conversation"
        ],
        tool_loading="full",            # All assigned tools
        personality=False,
        max_tokens=None,
    ),
    ContextMode.HEARTBEAT: ModeConfig(
        sections=[
            "identity", "skills", "platform_actions", "datetime_context"
        ],
        tool_loading="dispatcher_only", # Only platform_execute
        personality=False,
        max_tokens=8000,                # Heartbeats should be cheap
    ),
    ContextMode.RECIPE: ModeConfig(
        sections=[
            "identity", "skills", "platform_actions", "recipe_context",
            "datetime_context"
        ],
        tool_loading="full",
        personality=False,
        max_tokens=None,
    ),
    # ... etc
}


# --- sections/base.py ---

class BaseSection(ABC):
    """Base class for composable prompt sections."""

    name: str                       # Unique identifier
    priority: int                   # 1 (highest) to 10 (lowest) — trimmed low-to-high
    max_tokens: Optional[int]       # Hard cap for this section

    @abstractmethod
    async def render(self, ctx: SectionContext) -> str:
        """Render this section's content as a string."""
        ...

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for rendered content."""
        ...


# --- service.py ---

class ContextService:
    """
    Single entry point for building LLM context.

    Usage:
        service = ContextService(db_session)
        result = await service.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent_record,
            workspace_id="ws_123",
            messages=conversation_messages,
            task_description="Search the web and write a report",
        )
        # result.system_prompt, result.tools, result.messages → ready for LLM
    """

    async def build_context(
        self,
        mode: ContextMode,
        agent: Any,                         # Agent DB record
        workspace_id: str,
        messages: list[dict] = None,
        task_description: str = None,
        recipe_step: dict = None,
        complexity_assessment: Any = None,
        tool_hints: list[str] = None,
        widget_mode: bool = False,
        **kwargs,
    ) -> ContextResult:
        ...
```

### 2.4 Token Budget Manager

```python
# --- budget.py ---

@dataclass
class TokenBudget:
    total: int                          # e.g. 120000 for GPT-4
    reserved_for_response: int          # e.g. 4096
    reserved_for_messages: int          # Estimated from message history
    available_for_sections: int         # total - response - messages

class TokenBudgetManager:
    """
    Manages token allocation across sections.

    Algorithm:
    1. Estimate tokens for each rendered section
    2. If total fits within budget → include all
    3. If over budget → trim sections lowest-priority-first:
       a. Truncate section content to its max_tokens cap
       b. If still over → drop lowest-priority sections entirely
    4. Log which sections were trimmed/dropped
    """

    def allocate(
        self,
        sections: list[RenderedSection],
        budget: TokenBudget,
    ) -> tuple[list[RenderedSection], list[str]]:
        """Returns (included_sections, trimmed_section_names)."""
        ...
```

**Priority assignments:**

| Priority | Section | Rationale |
|----------|---------|-----------|
| 1 | Identity | Agent must know who it is |
| 2 | Task Context / Recipe Context | Must know what to do |
| 3 | Tools (schemas) | Must know what tools are available |
| 4 | Skills | SKILL.md instructions guide behaviour |
| 5 | Platform Actions | Action catalog for platform_execute |
| 6 | Memory | User context, preferences |
| 7 | Daily Logs | Recent activity for awareness |
| 8 | Datetime | Nice-to-have temporal context |
| 9 | Custom | Workspace-level custom prompts |

### 2.5 Tool Loading Strategy

Tool loading is unified in the `tools` section but varies by mode:

```python
class ToolLoadingStrategy(str, Enum):
    FULL = "full"                       # All assigned tools (core + dispatcher + composio)
    FILTERED = "filtered"               # Intent-based filtering (chatbot)
    DISPATCHER_ONLY = "dispatcher_only" # Only platform_execute (heartbeat)
    NONE = "none"                       # No tools (NL2SQL, orchestrator stages)
```

**Tool assembly (single implementation, used by all modes):**

```python
async def load_tools(
    self,
    agent_id: int,
    workspace_id: str,
    strategy: ToolLoadingStrategy,
    intent_result: Optional[IntentResult] = None,
    tool_hints: Optional[list[str]] = None,
) -> tuple[list[dict], str]:
    """
    Returns (tool_schemas, tool_choice).

    Steps:
    1. Load core tools from ToolRegistry (if strategy != NONE)
    2. Add platform_execute dispatcher schema (if strategy != NONE)
    3. Load Composio per-action schemas for assigned apps (if strategy != DISPATCHER_ONLY)
    4. Apply intent filtering (if strategy == FILTERED and intent_result provided)
    5. Apply tool_hints override (if complexity_assessment provided)
    """
    ...
```

This replaces:
- `get_tools_for_agent()` in `tool_router.py`
- `smart_tool_router.route()` in chatbot path
- Inline `to_dispatcher_schema()` calls in heartbeat
- Tool assembly in `agent_factory.py`

### 2.6 Memory Integration

Memory retrieval is unified in the `memory` section:

```python
class MemorySection(BaseSection):
    name = "memory"
    priority = 6
    max_tokens = 1500              # Cap memory injection at ~1500 tokens

    async def render(self, ctx: SectionContext) -> str:
        """
        Retrieves and formats memories.

        Steps:
        1. Call smart_memory.retrieve_memories() (existing service)
        2. Format as consistent markdown bullet list
        3. Append daily logs if config.INJECT_DAILY_LOGS is True
        4. Truncate to max_tokens
        """
        ...
```

This replaces:
- Memory retrieval + formatting in `smart_orchestrator.py:157-178`
- Memory string concatenation in `agent_factory._build_agent_system_prompt()`
- Missing memory in heartbeat/recipe paths

---

## 3. Section Details

### 3.1 Identity Section

```python
class IdentitySection(BaseSection):
    """Agent identity: name, role, persona, personality."""
    name = "identity"
    priority = 1
    max_tokens = 500
```

**Renders:**
```
You are {agent_name}, an AI agent on the Automatos platform.
Your role: {agent_role}
Workspace: {workspace_name}

{persona_text}                    # From personas table if assigned
{personality_adjustments}         # From workspace orchestrator_settings (tone, verbosity, etc.)
```

**Replaces:**
- `get_happy_system_prompt()` identity portion in `personality.py`
- `_build_agent_system_prompt()` opening in `agent_factory.py`
- Hardcoded "You are a helpful AI assistant" in heartbeat

### 3.2 Skills Section

```python
class SkillsSection(BaseSection):
    """SKILL.md content for the agent's assigned skill."""
    name = "skills"
    priority = 4
    max_tokens = 3000
```

**Renders:** The full SKILL.md text from the agent's assigned skill (loaded from `agent_skills` table → `skills` table → `content` field).

**Replaces:**
- Skill injection in `agent_factory._build_agent_system_prompt()`
- Missing skill injection in heartbeat (currently heartbeat gets skill content via its own path)

### 3.3 Platform Actions Section

```python
class PlatformActionsSection(BaseSection):
    """Markdown catalog of available platform_execute actions."""
    name = "platform_actions"
    priority = 5
    max_tokens = 2000
```

**Renders:** Output of `ActionRegistry.build_prompt_summary()` — grouped by category with parameter hints.

**Replaces:**
- `build_prompt_summary()` injection in `smart_orchestrator.py:262-268`
- `build_prompt_summary()` injection in `agent_factory.py`
- Inline summary in `heartbeat_service.py`

### 3.4 Memory Section

See §2.6 above. Renders:
```
## What You Know About This User

- {memory_1}
- {memory_2}
- ...

## Recent Activity

{daily_logs_trimmed}
```

### 3.5 Task Context Section

```python
class TaskContextSection(BaseSection):
    """Current task description and board context."""
    name = "task_context"
    priority = 2
    max_tokens = 1000
```

**Renders:**
```
## Current Task

{task_description}

Status: {task_status}
Priority: {task_priority}
Board: {board_name}
```

### 3.6 Recipe Context Section

```python
class RecipeContextSection(BaseSection):
    """Recipe step instructions for multi-step workflows."""
    name = "recipe_context"
    priority = 2
    max_tokens = 2000
```

**Renders:**
```
## Recipe: {recipe_name}

### Current Step: {step_number}/{total_steps} — {step_name}

{step_instructions}

### Previous Step Results:
{previous_step_output_summary}
```

### 3.7 Conversation Section

```python
class ConversationSection(BaseSection):
    """Message history formatting and trimming."""
    name = "conversation"
    priority = 3
    max_tokens = None              # Dynamic — uses remaining budget
```

**Renders:** Formatted message history, filtered and converted:
- Strips system messages (we build our own)
- Converts `parts` format to plain text
- Trims oldest messages if exceeding token budget

---

## 4. How Callers Change

### 4.1 Agent Factory (Task Execution)

**Before:**
```python
# agent_factory.py — 50+ lines of prompt building
system_prompt = self._build_agent_system_prompt(agent, skill_content, composio_hints)
tools = get_tools_for_agent(agent.id, workspace_id, db)
# ... manually inject platform summary, memory, etc.
```

**After:**
```python
# agent_factory.py — 3 lines
from modules.context import ContextService, ContextMode

context = await ContextService(db).build_context(
    mode=ContextMode.TASK_EXECUTION,
    agent=agent,
    workspace_id=workspace_id,
    messages=messages,
    task_description=task_description,
)
# context.system_prompt, context.tools, context.messages → pass to LLM
```

### 4.2 Smart Chat Orchestrator (Chatbot)

**Before:**
```python
# smart_orchestrator.py — 100+ lines in prepare_request()
intent_result = self.classifier.classify(...)
memory_result = await self.memory_manager.retrieve_memories(...)
tool_result = await self.tool_router.route(...)
system_prompt = get_happy_system_prompt(...)
# ... manually inject daily logs, platform summary, datetime
```

**After:**
```python
# smart_orchestrator.py — simplified prepare_request()
context = await ContextService(db).build_context(
    mode=ContextMode.CHATBOT,
    agent=chatbot_agent,
    workspace_id=self.workspace_id,
    messages=messages,
    complexity_assessment=complexity_assessment,
    widget_mode=self.widget_mode,
)
# Intent classification still runs (needed for tool_choice + response routing)
# but prompt building, tool loading, memory retrieval all delegated
```

### 4.3 Heartbeat Service

**Before:**
```python
# heartbeat_service.py — 40+ lines of inline prompt
system_prompt = f"You are {agent.name}... {skill_content}... Current time: {now}..."
tools = [registry.to_dispatcher_schema()]
```

**After:**
```python
context = await ContextService(db).build_context(
    mode=ContextMode.HEARTBEAT,
    agent=agent,
    workspace_id=workspace_id,
)
```

### 4.4 Recipe Executor

**Before:**
```python
# recipe_executor.py — builds prompt from recipe step + agent
prompt = f"{agent_system_prompt}\n\n{recipe_step_instructions}"
# Tools inherited from factory path
```

**After:**
```python
context = await ContextService(db).build_context(
    mode=ContextMode.RECIPE,
    agent=agent,
    workspace_id=workspace_id,
    recipe_step=current_step,
)
```

---

## 5. Migration Strategy

### Phase 1: Build the Module (No Breaking Changes)

**Goal:** Create `modules/context/` with full ContextService. All existing code paths continue working unchanged.

**Files created:**
- `modules/context/__init__.py`
- `modules/context/service.py`
- `modules/context/result.py`
- `modules/context/modes.py`
- `modules/context/budget.py`
- `modules/context/estimator.py`
- `modules/context/sections/base.py`
- `modules/context/sections/identity.py`
- `modules/context/sections/skills.py`
- `modules/context/sections/platform_actions.py`
- `modules/context/sections/memory.py`
- `modules/context/sections/tools.py`
- `modules/context/sections/task_context.py`
- `modules/context/sections/recipe_context.py`
- `modules/context/sections/datetime_context.py`
- `modules/context/sections/conversation.py`
- `modules/context/sections/custom.py`

**Verification:** Unit tests for each section + integration test that `build_context()` produces equivalent output to current paths.

### Phase 2: Migrate Callers (One at a Time)

Each migration follows the same pattern:
1. Add `ContextService` call alongside existing code
2. Log both outputs, verify equivalence
3. Switch to `ContextService` output
4. Remove old code

**Migration order (least risk → most risk):**

| Order | Caller | Risk | Rationale |
|-------|--------|------|-----------|
| 1 | Heartbeat Service | LOW | Runs on schedule, easy to test, simple prompt |
| 2 | Agent Factory | MEDIUM | Core execution path, well-tested |
| 3 | Recipe Executor | MEDIUM | Uses factory internally, limited usage |
| 4 | Execution Manager | LOW | Delegates to factory, thin wrapper |
| 5 | Smart Orchestrator (Chatbot) | HIGH | User-facing, intent classification interplay |
| 6 | Universal Router | LOW | Tier routing, independent prompts |
| 7 | Orchestrator Stages | LOW | Internal LLM calls, no tools |
| 8 | NL2SQL | LOW | Isolated, schema-specific |
| 9 | Channels (Telegram, etc.) | MEDIUM | Uses factory, needs testing |

### Phase 3: Cleanup

- Delete `_build_agent_system_prompt()` from `agent_factory.py`
- Delete `get_happy_system_prompt()` from `personality.py` (move personality logic to `IdentitySection`)
- Delete `smart_tool_router.py` (filtering moves to `ToolLoadingStrategy.FILTERED`)
- Delete tool loading from `tool_router.py:get_tools_for_agent()` (moves to `ToolsSection`)
- Consolidate `build_prompt_summary()` into `PlatformActionsSection`
- Remove memory injection from `smart_orchestrator.py` (moves to `MemorySection`)

### Phase 4: Advanced Features (Future)

- Prompt versioning via `system_prompts` table integration
- A/B testing section variants
- Per-workspace section overrides (admin can disable/reorder sections)
- Token usage analytics (which sections consume the most tokens per mode)

---

## 6. Token Budget Model

### 6.1 Default Budgets by Mode

| Mode | Model Context | Response Reserve | Message Reserve | Section Budget |
|------|--------------|------------------|-----------------|----------------|
| Chatbot | 128K | 4K | 60K | 64K |
| Task Execution | 128K | 4K | 20K | 104K |
| Heartbeat | 128K | 2K | 0 | 8K |
| Recipe | 128K | 4K | 10K | 40K |
| NL2SQL | 128K | 2K | 2K | 8K |

### 6.2 Token Estimation

We use a character-based estimator (4 chars ≈ 1 token) as the fast path, with optional `tiktoken` for precise estimation when the rough estimate is within 10% of the budget.

```python
class TokenEstimator:
    def estimate(self, text: str) -> int:
        """Fast estimate: len(text) / 4."""
        return len(text) // 4

    def precise(self, text: str, model: str = "gpt-4") -> int:
        """Precise estimate using tiktoken (if available)."""
        ...
```

### 6.3 Trimming Behaviour

When total section tokens exceed the budget:

1. **Soft trim** — Sections with `max_tokens` caps get truncated to their cap
2. **Hard trim** — If still over, drop sections from priority 10 → 1 until within budget
3. **Never drop** — Priority 1-2 sections (identity, task context) are never dropped
4. **Log warnings** — Every trim/drop is logged with section name and tokens saved

---

## 7. Observability

### 7.1 Logging

Every `build_context()` call logs:
```
[ContextService] mode=task_execution agent=141 sections=[identity,skills,platform_actions,memory,task_context,datetime_context]
  trimmed=[] token_estimate=4200/104000 tools=19 prep_time=45ms
```

### 7.2 SSE Events

The chatbot path currently emits `memory_retrieved` SSE events. `ContextResult` includes `memory_context` so the chatbot can continue emitting these events without reaching into internals.

### 7.3 Metrics (Future)

- `context_build_duration_ms` — histogram by mode
- `context_tokens_used` — gauge by mode + section
- `context_sections_trimmed` — counter by section name

---

## 8. Testing Strategy

### 8.1 Unit Tests

Each section gets its own test file:
```
tests/test_context/
├── test_identity_section.py
├── test_skills_section.py
├── test_platform_actions_section.py
├── test_memory_section.py
├── test_tools_section.py
├── test_budget_manager.py
├── test_estimator.py
└── test_service.py
```

**Key assertions:**
- Each section renders expected content given known inputs
- Budget manager trims lowest-priority sections first
- Budget manager never drops priority 1-2 sections
- Token estimator is within 20% of tiktoken for sample texts

### 8.2 Integration Tests

- `build_context(CHATBOT)` produces prompt containing identity, memory, platform actions
- `build_context(TASK_EXECUTION)` includes task description and full tool set
- `build_context(HEARTBEAT)` produces prompt under 8K tokens
- Tool schemas match expected structure (OpenAI function calling format)

### 8.3 Equivalence Tests (Migration Phase)

For each caller migration:
1. Capture current output (prompt + tools + messages) for 5 representative inputs
2. Run same inputs through `ContextService`
3. Assert semantic equivalence (exact match not required; key sections must be present)

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking existing prompts during migration | HIGH | Migrate one caller at a time, run equivalence tests, dual-write during transition |
| Token estimator inaccuracy | MEDIUM | Use conservative estimates (overcount by 10%), log actual vs estimated |
| Circular imports | MEDIUM | `modules/context/` depends on `modules/tools/`, `modules/memory/`, `core/models/` — keep dependency direction clear, no reverse imports |
| Performance regression (async section rendering) | LOW | Sections that need DB/API calls run in parallel via `asyncio.gather()` |
| Mode config drift (new features added to config but not to service) | MEDIUM | All prompt modifications must go through section classes — no direct string injection |

---

## 10. Success Criteria

| Metric | Target |
|--------|--------|
| Code paths using ContextService | 9/9 (100%) |
| Lines of prompt-building code deleted | > 500 |
| Time to add new prompt section to all agents | < 30 minutes (add 1 section class + register in modes) |
| Token budget violations (prompts exceeding model context) | 0 |
| Bug requiring multi-file prompt fix | 0 (fix in section class, affects all modes) |

---

## 11. File Impact Summary

### New Files
| File | Purpose |
|------|---------|
| `modules/context/__init__.py` | Package exports |
| `modules/context/service.py` | ContextService |
| `modules/context/result.py` | ContextResult dataclass |
| `modules/context/modes.py` | ContextMode enum + configs |
| `modules/context/budget.py` | TokenBudgetManager |
| `modules/context/estimator.py` | Token estimator |
| `modules/context/sections/*.py` | 11 section classes |
| `tests/test_context/*.py` | Unit + integration tests |

### Modified Files
| File | Change |
|------|--------|
| `modules/agents/factory/agent_factory.py` | Replace `_build_agent_system_prompt` + tool loading with `ContextService.build_context()` |
| `consumers/chatbot/smart_orchestrator.py` | Replace prompt building + memory + tool routing with `ContextService.build_context()` |
| `services/heartbeat_service.py` | Replace inline prompt + tool loading with `ContextService.build_context()` |
| `api/recipe_executor.py` | Replace prompt building with `ContextService.build_context()` |
| `modules/agents/execution/execution_manager.py` | Delegate to `ContextService` |
| `core/routing/engine.py` | Use `ContextService` for per-tier prompts |
| `modules/orchestrator/stages/*.py` | Use `ContextService` for stage prompts |
| `modules/nl2sql/service.py` | Use `ContextService` for schema prompt |

### Deleted Files (Phase 3)
| File | Reason |
|------|--------|
| `consumers/chatbot/smart_tool_router.py` | Filtering moves to `ToolsSection` |
| Parts of `consumers/chatbot/personality.py` | Personality moves to `IdentitySection` |

### Files NOT Touched
| File | Reason |
|------|--------|
| `modules/tools/execution/unified_executor.py` | Tool execution stays separate from context building |
| `modules/tools/discovery/action_registry.py` | Keeps `build_prompt_summary()` — consumed by `PlatformActionsSection` |
| `core/composio/client.py` | Composio SDK stays; tool schemas consumed by `ToolsSection` |
| `modules/memory/` | Memory services stay; consumed by `MemorySection` |

---

## 12. Relationship to Other PRDs

| PRD | Relationship |
|-----|-------------|
| **PRD-03** (Context Engineering Layer) | PRD-80 supersedes PRD-03's prompt management aspects. PRD-03 was theoretical; PRD-80 is the concrete implementation. |
| **PRD-51** (Orchestrator Unification) | PRD-80 is complementary — PRD-51 unified the routing/execution flow, PRD-80 unifies the context/prompt flow. |
| **PRD-58** (Prompt Management) | PRD-80 subsumes PRD-58. The FutureAGI integration and prompt versioning UI remain future work. |
| **PRD-64** (Unified Action Discovery) | PRD-80 consumes PRD-64's `ActionRegistry` via `PlatformActionsSection`. |
| **PRD-68** (Progressive Complexity) | PRD-80's modes support complexity-aware context (e.g., skip memory for simple queries via `complexity_assessment`). |
| **PRD-69** (Agent Intelligence Layer) | PRD-80 provides the context backbone that PRD-69's intelligence features would plug into. |

---

## Appendix A: Context Engineering Patterns Applied

From David Kamm & IBM's Context Engineering framework:

| Pattern | How We Apply It |
|---------|----------------|
| **Atoms → Molecules → Cells** | Sections are atoms; mode configs compose atoms into molecules; `build_context()` is the cell |
| **Token budgets as constraints** | `TokenBudgetManager` enforces hard limits with priority-based trimming |
| **Declarative assembly** | `MODE_CONFIGS` dict declares section composition per mode — no imperative if/else |
| **Schema-driven context** | `ContextResult` is a typed schema; sections implement `BaseSection` interface |
| **Separation of concerns** | Each section owns exactly one type of context; no section reaches into another |

## Appendix B: Current Prompt Sizes (Estimated)

Measured from production logs and code analysis:

| Component | Tokens (est.) | Notes |
|-----------|--------------|-------|
| Identity / personality | ~200 | `get_happy_system_prompt` base |
| Skill content (SENTINEL) | ~1,800 | Full SKILL.md |
| Platform action summary | ~1,200 | 58 actions grouped by category |
| Memory injection | ~400 | 5-10 memories as bullets |
| Daily logs | ~500 | Last 2000 chars |
| Datetime context | ~30 | Single line |
| Tool schemas (19 tools) | ~3,000 | OpenAI function format |
| **Total (task execution)** | **~7,130** | Well within budget |
| **Total (chatbot, 20 msgs)** | **~12,000** | Includes message history |
