# PRD-107 — Context Interface Abstraction

**Version:** 1.0
**Type:** Research + Design
**Status:** Complete — Ready for Peer Review
**Priority:** P1
**Dependencies:** PRD-102 (Coordinator Architecture), PRD-100 (Master Research)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-15

---

## 1. Problem Statement

### 1.1 The Gap

Phase 3 swaps message-passing for shared semantic fields, but the coordinator (PRD-102) shouldn't know or care which context implementation runs behind the interface. Today, **every consumer directly imports `ContextService`** and calls `build_context()` with a specific `ContextMode` enum value. There is no abstraction boundary — replacing the context layer means rewriting every caller.

### 1.2 Direct Coupling Points

| Coupling Point | File | Line | Problem |
|---------------|------|------|---------|
| Direct `ContextService(db)` instantiation | `smart_orchestrator.py` | 194 | Creates `ContextService(self._db_session)` directly — no injection, no interface |
| Direct `ContextService(db)` instantiation | `heartbeat_service.py` | 439 | Same pattern — `ContextService(db).build_context(...)` |
| Direct `ContextService(db)` instantiation | `recipe_executor.py` | 143 | Same pattern |
| Direct `ContextService(db)` instantiation | `routing/engine.py` | 525 | Same pattern — `ContextService(self._db).build_context(...)` |
| Direct `ContextService(db)` instantiation | `agent_factory.py` | 748 | Same pattern |
| Direct `ContextService()` instantiation | `task_decomposer.py` | 181 | Even uses `ContextService()` with no DB session |
| Direct `ContextService()` instantiation | `agent_negotiation.py` | 125 | Same pattern |
| Direct `ContextService()` instantiation | `quality_assessor.py` | 264 | Same pattern |
| Direct `ContextService()` instantiation | `complexity_analyzer.py` | 427 | Same pattern |
| Direct `ContextService()` instantiation | `nl2sql/service.py` | 294 | Same pattern |
| `ContextMode` enum dependency | `modules/context/modes.py:13` | 13 | Callers import and pass specific enum values — adding a mode requires touching callers |
| `ContextResult` structure assumption | `modules/context/result.py:10` | 10 | Callers destructure `.system_prompt`, `.tools`, `.messages` |
| `SharedContextManager` disconnected | `inter_agent.py:400` | 400 | Exists but is NOT integrated into `ContextService` pipeline |

**10 production callers** directly instantiate `ContextService`. Swapping the backend means changing all 10.

### 1.3 Why This Matters Now

Without this abstraction:
- Phase 3 requires rewriting all 10 callers of `ContextService`
- The coordinator (PRD-102) would be built against a concrete implementation, making Phase 3 a breaking change
- No way to A/B test message-passing vs. field-based context (PRD-108 experiment requires running both)
- No clean path to inject mission-level shared context into agent prompts

### 1.4 What This PRD Delivers

A **port/adapter boundary** between context consumers (coordinator, chatbot, heartbeat, recipe, router, agent factory, orchestrator stages) and context providers (current ContextService for Phase 2, neural field engine for Phase 3). Consumers call the port. The adapter maps to whichever backend is active. Zero consumer changes when Phase 3 ships.

---

## 2. Prior Art Analysis

### 2.1 Architecture Patterns

| Pattern | Source | What We Adopt | What We Reject |
|---------|--------|---------------|----------------|
| **Hexagonal Architecture** (Cockburn 2005) | Port ownership by domain layer, adapter as infrastructure, driving vs. driven ports | Port defined in `core/ports/` — coordinator owns the contract. Adapter lives in `modules/context/adapters/` — infrastructure layer | Symmetric port model — our ports are asymmetric: `ContextProvider` is a driven port (coordinator drives it), not a driving port |
| **Clean Architecture** (Martin 2017) | Dependency Rule: source code dependencies point inward only. The domain layer never imports from infrastructure | Coordinator imports `ContextProvider` from `core/ports/`. Never imports `ContextService` from `modules/context/` | Full 4-ring layering — our codebase doesn't have formal ring boundaries. We adopt the dependency direction rule, not the full structure |
| **Strategy Pattern** (GoF 1994) | Runtime backend swap via config-driven factory | Factory function `create_context_provider(config)` selects adapter at startup based on `CONTEXT_BACKEND` env var | Formal Strategy interface hierarchy — our ABC ports serve the same role without GoF boilerplate |
| **Repository Pattern** (Evans 2003) | Collection-like interface over diverse backends | `SharedContextPort.query()` hides whether backend is Redis dict lookup or Qdrant vector search | Repository-per-aggregate — context isn't a DDD aggregate; the pattern is borrowed, not applied literally |
| **Cosmic Python** (Percival & Gregory) | Python-specific ports/adapters with ABC, manual composition root, in-memory test fakes | ABC-based ports, `InMemoryContextProvider` for testing, factory function as composition root — no DI framework | `dependency-injector` library — manual factory is sufficient at our scale |

### 2.2 Multi-Agent Framework Context Abstractions

| Framework | Key Insight | What We Adopt | What We Reject |
|-----------|-------------|---------------|----------------|
| **LangGraph** | Two-tier separation: `BaseCheckpointSaver` (thread-scoped state) + `BaseStore` (cross-thread memory). Both injected at `compile()` time via constructor | Two ports: `ContextProvider` (per-agent, per-call) + `SharedContextPort` (mission-scoped, cross-agent). Separate concerns, separate lifecycles | Sync/async dual API — our codebase is async-first; sync wrapper is unnecessary complexity |
| **AutoGen** | `Memory` ABC with 5 methods: `add()`, `query()`, `update_context()`, `clear()`, `close()`. `update_context(model_context)` is a context preprocessor — it mutates the model's context by injecting memories | The preprocessor concept: both Phase 2 (inject retrieved messages) and Phase 3 (inject field query results) are context preprocessing. But we use immutable returns, not mutation | `MemoryContent` with MIME types — over-engineered for our use case. Our context is always text/JSON |
| **CrewAI** | Hierarchical scopes: `memory.scope("workspace/agent/task")` restricts operations to a subtree. `memory.slice()` creates read-only views across scopes | Scope concept for future: workspace → mission → agent context hierarchy maps cleanly to our data model | Composite scoring (semantic + recency + importance) in the port — that's adapter-internal, not port-level |

### 2.3 Context Engineering Theory (from repo chapters 08-14)

| Concept | Implication for Interface |
|---------|--------------------------|
| **Field operations** (inject, decay, resonate, attractor, boundary) | `SharedContextPort` must support `inject()` and `query()` — the minimal operations that both message-passing AND field-based backends can implement |
| **Resonance as query** | Phase 3 queries via resonance ("what resonates with X?"), not retrieval ("get context for agent X"). The `query()` method must accept a natural language query, not just a key lookup — so both backends can implement it |
| **Non-commutativity** | `inject(A); inject(B)` ≠ `inject(B); inject(A)` because attractors formed by A change how B resonates. The interface preserves operation ordering — no batched unordered writes |
| **Decay** | Phase 2 uses Redis TTL (2h). Phase 3 uses exponential decay (`S(t) = S₀ × e^(-λt)`). The interface doesn't expose decay — it's adapter-internal |
| **Boundary permeability** | Phase 3 can configure which agents see which field patterns. Phase 2 uses team membership lists. The `create_context()` method accepts `team_agent_ids` — both backends enforce access control their own way |

### 2.4 Key Design Decision

**Two ports, ABC-based, async-only, constructor-injected via manual factory.**

Rationale:
- Two ports (not one) because `ContextProvider` and `SharedContextPort` have different consumers, lifecycles, and Phase 3 implementations
- ABC (not Protocol) because port contracts are foundational — explicit inheritance catches missing methods at class definition time, not at runtime
- Async-only because all backends are I/O-bound and all callers already use `await`
- Manual factory (not DI framework) because Cosmic Python's pattern works and we have <15 callers to wire

---

## 3. Port Interfaces

### 3.1 Core Port: `ContextProvider`

Lives at `orchestrator/core/ports/context.py`. Owned by the domain layer.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional


class ContextModeType(StrEnum):
    """Port-level mode names. Decoupled from ContextMode enum in modes.py."""
    CHATBOT = "chatbot"
    TASK_EXECUTION = "task_execution"
    HEARTBEAT_ORCHESTRATOR = "heartbeat_orchestrator"
    HEARTBEAT_AGENT = "heartbeat_agent"
    RECIPE = "recipe"
    ROUTER = "router"
    ORCHESTRATOR_STAGE = "orchestrator_stage"
    NL2SQL = "nl2sql"
    COORDINATOR = "coordinator"        # PRD-102
    VERIFIER = "verifier"              # PRD-103


@dataclass(frozen=True)
class AgentContext:
    """Immutable domain object — the coordinator's view of context.

    Any backend must produce this shape. Replaces ContextResult as the
    public contract. ContextResult becomes adapter-internal.
    """
    system_prompt: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "auto"

    # Metadata for observability and telemetry (PRD-106)
    mode: str = ""
    sections_included: list[str] = field(default_factory=list)
    sections_trimmed: list[str] = field(default_factory=list)
    token_estimate: int = 0
    token_budget: int = 0
    preparation_time_ms: float = 0.0
    memory_context: Optional[str] = None
    user_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextProvider(ABC):
    """Driven port: consumers call this to build LLM-ready context.

    Phase 2: DefaultContextProvider wraps ContextService
    Phase 3: NeuralFieldContextProvider queries neural field
    """

    @abstractmethod
    async def build_context(
        self,
        mode: str,
        agent: Any,
        workspace_id: str,
        messages: Optional[list[dict]] = None,
        task_description: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentContext:
        """Build complete LLM-ready context for the given mode.

        Args:
            mode: Context mode name (ContextModeType value or string).
            agent: Agent record (SQLAlchemy model or dict).
            workspace_id: UUID string for workspace scoping.
            messages: Conversation history for chatbot modes.
            task_description: Task text for task execution modes.
            **kwargs: Backend-specific overrides (recipe_step, etc.).

        Returns:
            Immutable AgentContext ready for LLM call.
        """
        ...
```

### 3.2 Secondary Port: `SharedContextPort`

Mission-level shared context between agents. Separate from per-agent context.

```python
class SharedContextPort(ABC):
    """Cross-agent context sharing within a mission.

    Phase 2: RedisSharedContext wraps SharedContextManager
    Phase 3: NeuralFieldSharedContext injects/queries neural field
    """

    @abstractmethod
    async def inject(
        self,
        context_id: str,
        key: str,
        value: Any,
        agent_id: int,
        strength: float = 1.0,
    ) -> None:
        """Inject a finding/result into shared context.

        Phase 2: Redis HSET with 2h TTL.
        Phase 3: Neural field injection with strength parameter.
        """
        ...

    @abstractmethod
    async def query(
        self,
        context_id: str,
        query: str,
        agent_id: int,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query shared context relevant to the query string.

        Phase 2: Key-value lookup from Redis/in-memory dict.
        Phase 3: Resonance-based retrieval from neural field.

        Returns list of {key, value, score?} dicts, ranked by relevance.
        """
        ...

    @abstractmethod
    async def create_context(
        self,
        team_agent_ids: list[int],
        initial_data: Optional[dict] = None,
    ) -> str:
        """Create a scoped shared context for a team of agents.

        Returns: context_id (opaque string).
        """
        ...

    @abstractmethod
    async def destroy_context(self, context_id: str) -> None:
        """Destroy a shared context and release resources."""
        ...
```

### 3.3 What the Interface Does NOT Expose

| Hidden Detail | Why |
|---------------|-----|
| Section-level control | Callers don't pick sections. The adapter decides based on mode config |
| Token budget internals | Callers don't set per-section budgets. The adapter's `TokenBudgetManager` handles it |
| Resonance/decay parameters | Phase 3 internals — adapter tunes them via its own config |
| Storage backend (Redis, Postgres, Qdrant) | Invisible to callers — the whole point of ports/adapters |
| `ContextMode` enum values | Callers use `ContextModeType` strings; adapter maps internally |
| `ContextResult` dataclass | Adapter-internal — callers see `AgentContext` only |

### 3.4 Interface Segregation

| Consumer | Uses `ContextProvider` | Uses `SharedContextPort` |
|----------|----------------------|-------------------------|
| Coordinator (PRD-102) | Yes — builds context for task agents | Yes — manages mission-level shared context |
| Chatbot (`smart_orchestrator.py`) | Yes — chatbot context | No — no mission shared context |
| Heartbeat (`heartbeat_service.py`) | Yes — heartbeat context | No |
| Recipe (`recipe_executor.py`) | Yes — task execution context | No |
| Router (`routing/engine.py`) | Yes — router context | No |
| AgentFactory (`agent_factory.py`) | Yes — task execution context | No (receives mission context via kwargs) |
| Orchestrator stages | Yes — orchestrator_stage context | No |

Most consumers only need `ContextProvider`. Only the coordinator needs both. Interface Segregation Principle: don't force agents to depend on shared context methods they don't use.

---

## 4. Phase 2 Adapters

### 4.1 `DefaultContextProvider`

Wraps the existing `ContextService` (`modules/context/service.py:47`). Zero behavior change — every existing test continues to pass.

```python
# orchestrator/modules/context/adapters/default.py

from core.ports.context import ContextProvider, AgentContext
from modules.context.service import ContextService
from modules.context.modes import ContextMode


class DefaultContextProvider(ContextProvider):
    """Phase 2 adapter: delegates to existing ContextService.

    Maps AgentContext (domain contract) ← ContextResult (internal).
    All existing behavior, sections, budget management, tool loading
    is preserved. This is a thin translation layer.
    """

    def __init__(self, db_session=None):
        self._service = ContextService(db_session)

    async def build_context(
        self,
        mode: str,
        agent,
        workspace_id: str,
        messages=None,
        task_description=None,
        **kwargs,
    ) -> AgentContext:
        # Map string mode → ContextMode enum (backwards compat)
        try:
            context_mode = ContextMode(mode)
        except ValueError:
            # New modes (coordinator, verifier) not yet in enum
            # Fall back to TASK_EXECUTION as default
            context_mode = ContextMode.TASK_EXECUTION

        result = await self._service.build_context(
            mode=context_mode,
            agent=agent,
            workspace_id=workspace_id,
            messages=messages,
            task_description=task_description,
            **kwargs,
        )

        # Map ContextResult → AgentContext
        return AgentContext(
            system_prompt=result.system_prompt,
            messages=result.messages,
            tools=result.tools,
            tool_choice=result.tool_choice,
            mode=result.mode,
            sections_included=result.sections_included,
            sections_trimmed=result.sections_trimmed,
            token_estimate=result.token_estimate,
            token_budget=result.token_budget,
            preparation_time_ms=result.preparation_time_ms,
            memory_context=result.memory_context,
            user_name=result.user_name,
        )
```

### 4.2 `RedisSharedContext`

Wraps the existing `SharedContextManager` (`inter_agent.py:400`). Preserves 2h TTL behavior.

```python
# orchestrator/modules/context/adapters/shared_redis.py

from core.ports.context import SharedContextPort
from modules.agents.communication.inter_agent import SharedContextManager


class RedisSharedContext(SharedContextPort):
    """Phase 2 adapter: Redis + in-memory shared context.

    Wraps SharedContextManager. 2h TTL via Redis EXPIRE.
    query() returns all entries (no semantic ranking in Phase 2).
    """

    def __init__(self, db_session=None):
        self._manager = SharedContextManager(db_session)

    async def inject(self, context_id, key, value, agent_id, strength=1.0):
        # strength is ignored in Phase 2 — Redis doesn't support weighted entries
        await self._manager.update_shared_context(
            context_id=context_id,
            agent=agent_id,
            updates={key: value},
            merge_strategy="append",
        )

    async def query(self, context_id, query, agent_id, top_k=10):
        ctx = self._manager.contexts.get(context_id)
        if not ctx:
            # Try Redis fallback
            if self._manager._redis:
                import json as _json
                redis_key = f"swarm:{context_id}:shared_context"
                raw = self._manager._redis.hget(redis_key, "context_data")
                if raw:
                    data = _json.loads(raw)
                    return [{"key": k, "value": v, "score": 1.0} for k, v in data.items()][:top_k]
            return []

        # Phase 2: return all entries, no semantic ranking
        return [
            {"key": k, "value": v, "score": 1.0}
            for k, v in ctx.context_data.items()
        ][:top_k]

    async def create_context(self, team_agent_ids, initial_data=None):
        ctx = await self._manager.create_shared_context(
            team=team_agent_ids,
            initial_context=initial_data or {},
        )
        return ctx.id

    async def destroy_context(self, context_id):
        # Remove from in-memory dict
        self._manager.contexts.pop(context_id, None)
        # Remove from Redis
        if self._manager._redis:
            redis_key = f"swarm:{context_id}:shared_context"
            self._manager._redis.delete(redis_key)
```

### 4.3 `InMemoryContextProvider` — Test Fake

For unit testing coordinators without DB/Redis:

```python
# orchestrator/modules/context/adapters/memory.py

from core.ports.context import ContextProvider, SharedContextPort, AgentContext


class InMemoryContextProvider(ContextProvider):
    """Test fake: returns configurable context without DB or LLM calls."""

    def __init__(self, system_prompt: str = "You are a test agent.", tools: list = None):
        self._system_prompt = system_prompt
        self._tools = tools or []

    async def build_context(self, mode, agent, workspace_id,
                            messages=None, task_description=None, **kwargs):
        return AgentContext(
            system_prompt=self._system_prompt,
            tools=self._tools,
            messages=messages or [],
            mode=mode,
            token_estimate=100,
            token_budget=4096,
        )


class InMemorySharedContext(SharedContextPort):
    """Test fake: in-memory dict-based shared context."""

    def __init__(self):
        self._contexts: dict[str, dict] = {}

    async def inject(self, context_id, key, value, agent_id, strength=1.0):
        if context_id not in self._contexts:
            raise ValueError(f"Context {context_id} not found")
        self._contexts[context_id][key] = value

    async def query(self, context_id, query, agent_id, top_k=10):
        data = self._contexts.get(context_id, {})
        return [{"key": k, "value": v, "score": 1.0} for k, v in data.items()][:top_k]

    async def create_context(self, team_agent_ids, initial_data=None):
        import uuid
        ctx_id = str(uuid.uuid4())
        self._contexts[ctx_id] = initial_data or {}
        return ctx_id

    async def destroy_context(self, context_id):
        self._contexts.pop(context_id, None)
```

---

## 5. Factory Function & Composition Root

### 5.1 Factory

```python
# orchestrator/core/factories/context.py

from core.ports.context import ContextProvider, SharedContextPort
from config import config


def create_context_provider(
    db_session=None,
    backend: str = None,
) -> ContextProvider:
    """Create the context provider based on config.

    Args:
        db_session: SQLAlchemy session for DB-backed adapters.
        backend: Override CONTEXT_BACKEND config. One of:
            "default" — Phase 2 message-passing (ContextService)
            "neural_field" — Phase 3 neural field (PRD-108+)

    Returns:
        ContextProvider instance.
    """
    backend = backend or getattr(config, 'CONTEXT_BACKEND', 'default')

    if backend == "neural_field":
        from modules.context.adapters.neural_field import NeuralFieldContextProvider
        return NeuralFieldContextProvider()
    else:
        from modules.context.adapters.default import DefaultContextProvider
        return DefaultContextProvider(db_session)


def create_shared_context(
    db_session=None,
    backend: str = None,
) -> SharedContextPort:
    """Create the shared context port based on config."""
    backend = backend or getattr(config, 'CONTEXT_BACKEND', 'default')

    if backend == "neural_field":
        from modules.context.adapters.neural_field import NeuralFieldSharedContext
        return NeuralFieldSharedContext()
    else:
        from modules.context.adapters.shared_redis import RedisSharedContext
        return RedisSharedContext(db_session)
```

### 5.2 Config Extension

```python
# In config.py
CONTEXT_BACKEND = os.getenv("CONTEXT_BACKEND", "default")  # "default" | "neural_field"
```

### 5.3 Per-Mission Override (for PRD-108 A/B Testing)

The factory accepts an explicit `backend` parameter. The coordinator can pass a mission-level override:

```python
# In CoordinatorService
mission_backend = mission_config.get("context_backend", None)  # None = use default
context_provider = create_context_provider(db, backend=mission_backend)
shared_context = create_shared_context(db, backend=mission_backend)
```

This enables PRD-108's controlled experiment: specific missions run through the neural field adapter while everything else uses the default.

---

## 6. MissionContextSection

New section that injects shared mission context into an agent's system prompt via `SharedContextPort`.

```python
# orchestrator/modules/context/sections/mission_context.py

from modules.context.sections import BaseSection, SectionContext


class MissionContextSection(BaseSection):
    """Injects shared mission context into agent's system prompt.

    Activated when kwargs contains 'mission_context_id' and 'shared_context_port'.
    Priority 3: after identity/task, before skills/plugins.
    """
    name = "mission_context"
    priority = 3
    max_tokens = 2000

    async def render(self, ctx: SectionContext) -> str:
        mission_context_id = ctx.kwargs.get("mission_context_id")
        shared_port = ctx.kwargs.get("shared_context_port")

        if not mission_context_id or not shared_port:
            return ""

        entries = await shared_port.query(
            context_id=mission_context_id,
            query=ctx.task_description or "",
            agent_id=ctx.agent.id if hasattr(ctx.agent, 'id') else 0,
            top_k=5,
        )

        if not entries:
            return ""

        lines = ["## Mission Shared Context", ""]
        for entry in entries:
            lines.append(f"**{entry['key']}:** {entry['value']}")
            lines.append("")

        return "\n".join(lines)
```

Register in `SECTION_REGISTRY`:

```python
# In modules/context/sections/__init__.py
from .mission_context import MissionContextSection
SECTION_REGISTRY["mission_context"] = MissionContextSection
```

Add to mode configs that need it:

```python
# In modules/context/modes.py — MODE_CONFIGS
ContextMode.COORDINATOR: ModeConfig(
    sections=["identity", "task", "mission_context", "skills", "tools", "memory"],
    tool_loading="full",
),
```

---

## 7. Migration Path

### 7.1 Strategy: Incremental, Non-Breaking

The adapter wraps `ContextService` — `ContextService` itself is unchanged. Callers migrate one-by-one. Both old and new patterns work simultaneously.

### 7.2 Migration Steps Per Caller

Each caller follows the same 3-step pattern:

**Step 1:** Accept `ContextProvider` via constructor (with default fallback).

```python
# Before:
class SmartChatOrchestrator:
    def __init__(self, db_session, ...):
        self._db_session = db_session

    async def _build_context(self):
        context = await ContextService(self._db_session).build_context(
            mode=ContextMode.CHATBOT, ...
        )

# After:
class SmartChatOrchestrator:
    def __init__(self, db_session, context_provider: ContextProvider = None, ...):
        self._db_session = db_session
        self._context = context_provider or create_context_provider(db_session)

    async def _build_context(self):
        context = await self._context.build_context(
            mode="chatbot", ...
        )
```

**Step 2:** Replace `ContextMode.X` enum with string `"x"`.

**Step 3:** Replace `ContextResult` type hints with `AgentContext`.

### 7.3 Migration Order

Migration order follows dependency criticality — highest-impact callers first:

| Priority | File | Caller | Why This Order |
|----------|------|--------|----------------|
| 1 | `agent_factory.py:748` | `AgentFactory` | All agent execution flows through here — highest blast radius |
| 2 | `smart_orchestrator.py:194` | Chatbot | User-facing — validates the pattern works end-to-end |
| 3 | `heartbeat_service.py:439` | Heartbeat | High-frequency caller — validates performance |
| 4 | `recipe_executor.py:143` | Recipe | Task execution path |
| 5 | `routing/engine.py:525` | Router | Routing path |
| 6-10 | Orchestrator stages | Various | Less critical — only active during orchestrated execution |

### 7.4 Backward Compatibility

During migration, `ContextResult` is NOT removed — it becomes adapter-internal. Callers that haven't migrated continue to work. `AgentContext` has the same fields as `ContextResult` plus `metadata`:

| `ContextResult` field | `AgentContext` field | Change |
|----------------------|---------------------|--------|
| `system_prompt` | `system_prompt` | Same |
| `messages` | `messages` | Same |
| `tools` | `tools` | Same |
| `tool_choice` | `tool_choice` | Same |
| `mode` | `mode` | Same |
| `sections_included` | `sections_included` | Same |
| `sections_trimmed` | `sections_trimmed` | Same |
| `token_estimate` | `token_estimate` | Same |
| `token_budget` | `token_budget` | Same |
| `memory_context` | `memory_context` | Same |
| `user_name` | `user_name` | Same |
| `preparation_time_ms` | `preparation_time_ms` | Same |
| — | `metadata` | **New** — extensible dict for adapter-specific data |

---

## 8. Phase 3 Adapter Skeleton

Stub implementation for PRD-108+ to fill in:

```python
# orchestrator/modules/context/adapters/neural_field.py

from core.ports.context import ContextProvider, SharedContextPort, AgentContext


class NeuralFieldContextProvider(ContextProvider):
    """Phase 3 adapter: builds context from neural field resonance.

    PRD-108 implements the prototype. This stub documents the contract.
    """

    async def build_context(self, mode, agent, workspace_id,
                            messages=None, task_description=None, **kwargs):
        raise NotImplementedError(
            "NeuralFieldContextProvider requires PRD-108 implementation. "
            "Set CONTEXT_BACKEND=default to use Phase 2 message-passing."
        )


class NeuralFieldSharedContext(SharedContextPort):
    """Phase 3 adapter: shared context via neural field injection.

    PRD-108 implements the prototype. This stub documents the contract.
    """

    async def inject(self, context_id, key, value, agent_id, strength=1.0):
        raise NotImplementedError("Requires PRD-108 implementation.")

    async def query(self, context_id, query, agent_id, top_k=10):
        raise NotImplementedError("Requires PRD-108 implementation.")

    async def create_context(self, team_agent_ids, initial_data=None):
        raise NotImplementedError("Requires PRD-108 implementation.")

    async def destroy_context(self, context_id):
        raise NotImplementedError("Requires PRD-108 implementation.")
```

**Why the interface holds for Phase 3:**

| Operation | Phase 2 Implementation | Phase 3 Implementation |
|-----------|----------------------|----------------------|
| `build_context(mode, agent, ...)` | 12 sections + budget manager → system prompt | Field query by agent pattern → resonant context assembly |
| `inject(context_id, key, value, ...)` | Redis HSET with 2h TTL | Neural field injection with strength, triggers attractor formation |
| `query(context_id, query, ...)` | Dict key-value lookup, return all | Resonance measurement, return ranked results |
| `create_context(team_ids, ...)` | Allocate Redis key + in-memory dict | Create field with boundary config for team |
| `destroy_context(id)` | Delete Redis key + dict entry | Trigger field decay + explicit cleanup |

Same 5 methods. Different backends. Zero coordinator changes.

---

## 9. Module Hierarchy

### 9.1 New Files

```
orchestrator/
├── core/
│   ├── ports/
│   │   ├── __init__.py              # Public exports
│   │   └── context.py               # ContextProvider, SharedContextPort, AgentContext, ContextModeType
│   └── factories/
│       └── context.py               # create_context_provider(), create_shared_context()
└── modules/
    └── context/
        ├── adapters/
        │   ├── __init__.py           # Adapter exports
        │   ├── default.py            # DefaultContextProvider (wraps ContextService)
        │   ├── shared_redis.py       # RedisSharedContext (wraps SharedContextManager)
        │   ├── memory.py             # InMemoryContextProvider, InMemorySharedContext (test fakes)
        │   └── neural_field.py       # NeuralFieldContextProvider stub (PRD-108+)
        └── sections/
            └── mission_context.py    # MissionContextSection
```

### 9.2 Dependency Direction

```
Coordinator (PRD-102)
    │
    │ imports
    ▼
core/ports/context.py  ←── THE BOUNDARY ──→  modules/context/adapters/
    │                                              │
    │ defines:                                     │ implements:
    │  - ContextProvider ABC                       │  - DefaultContextProvider
    │  - SharedContextPort ABC                     │  - RedisSharedContext
    │  - AgentContext dataclass                    │  - NeuralFieldContextProvider
    │  - ContextModeType StrEnum                   │
    │                                              │ wraps:
    │                                              │  - ContextService (unchanged)
    │                                              │  - SharedContextManager (unchanged)
    ▼
core/factories/context.py
    │
    │ selects adapter at startup
    │ based on config.CONTEXT_BACKEND
    ▼
CoordinatorService(context=adapter)
```

Source code dependencies point inward only: `adapters/ → ports/` (never `ports/ → adapters/`). The coordinator imports from `core/ports/`, never from `modules/context/`.

---

## 10. Cross-PRD Integration

| PRD | Integration | Direction |
|-----|-------------|-----------|
| **PRD-102** (Coordinator) | Coordinator receives `ContextProvider` + `SharedContextPort` via constructor. Uses `mode="coordinator"`. Creates mission shared context via `create_context()` | PRD-102 consumes → PRD-107 provides |
| **PRD-103** (Verification) | Verifier uses `ContextProvider` with `mode="verifier"`. New mode added to `ContextModeType` and `MODE_CONFIGS` | PRD-103 consumes → PRD-107 provides |
| **PRD-104** (Ephemeral Agents) | Contractors receive context through same `ContextProvider` — no special path. Shared context entries survive contractor destruction | PRD-104 consumes normally |
| **PRD-105** (Budget) | Budget enforcement can wrap `ContextProvider` as a decorator — check budget before `build_context()`, not inside it | PRD-105 decorates → PRD-107 core |
| **PRD-106** (Telemetry) | `AgentContext.token_estimate` and `sections_trimmed` are first-class telemetry fields. `metadata` dict carries additional observability data | PRD-106 reads → PRD-107 produces |
| **PRD-108** (Memory Field) | Neural field adapter implements both ports. PRD-108 experiment tests whether field-based context outperforms message-passing — same interface, different backend | PRD-108 implements → PRD-107 defines |

---

## 11. Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Over-abstraction — interface too generic to be useful | High | Medium | Start with the narrowest interface that satisfies the coordinator. 5 methods total across both ports. Expand only when Phase 3 demands it |
| 2 | Leaky abstraction — backend-specific types leak through `metadata` | Medium | High | Strict rule: `metadata` values are string/int/float/bool only. No `ContextResult`, no `SharedContext`, no `VectorSearchResult` |
| 3 | Phase 3 requirements break the interface | High | Medium | PRD-108 prototype IS the validation gate. If it can't implement the ports, the interface is updated before Phase 3 PRDs |
| 4 | Performance regression from adapter indirection | Low | Low | Adapter is one function call + one dataclass construction. Sub-microsecond overhead. Context assembly (100ms+) dominates |
| 5 | Migration disruption — changing all 10 callers at once | Medium | Medium | Incremental migration: adapter wraps ContextService (unchanged), callers migrate one-by-one, old pattern continues to work |
| 6 | `MissionContextSection` token budget competes with critical sections | Medium | Medium | Priority 3 (after identity/task, before skills). Max 2000 tokens. Budget manager drops it before identity or conversation |
| 7 | Two ports increase wiring complexity | Medium | Low | Single factory function creates both. Coordinator receives both from same factory call. Test fakes implement both in one import |

---

## 12. Acceptance Criteria

### Must Have

- [x] **Port interfaces defined** — `ContextProvider` ABC and `SharedContextPort` ABC with full signatures (Section 3)
- [x] **`AgentContext` domain object** — Frozen dataclass replacing `ContextResult` as public contract (Section 3.1)
- [x] **`ContextModeType` StrEnum** — Port-level mode names including `COORDINATOR` and `VERIFIER` (Section 3.1)
- [x] **`DefaultContextProvider` adapter** — Wraps `ContextService`, zero behavior change (Section 4.1)
- [x] **`RedisSharedContext` adapter** — Wraps `SharedContextManager`, preserves 2h TTL (Section 4.2)
- [x] **`InMemoryContextProvider` test fake** — For unit testing coordinators without DB/Redis (Section 4.3)
- [x] **Factory function** — `create_context_provider(config)` selects adapter (Section 5.1)
- [x] **`MissionContextSection`** — Injects shared context into agent prompts (Section 6)
- [x] **Migration path** — Step-by-step for all 10 callers (Section 7)
- [x] **Module hierarchy** — Files, locations, dependency direction (Section 9)

### Should Have

- [x] **Phase 3 adapter skeleton** — `NeuralFieldContextProvider` stub with `NotImplementedError` (Section 8)
- [x] **Per-mission adapter override** — Factory supports backend override for A/B testing (Section 5.3)
- [x] **Backward compatibility table** — `ContextResult` ↔ `AgentContext` field mapping (Section 7.4)
- [x] **Interface segregation analysis** — Which consumers use which ports (Section 3.4)

### Nice to Have

- [x] **Architecture Decision Record** — Why ports/adapters, ABC over Protocol, two ports over one (Section 2.4)
- [x] **Dependency direction diagram** — Visual of the boundary (Section 9.2)
- [x] **Cross-PRD integration map** — How each PRD interacts with the ports (Section 10)

---

## Appendix A: Research Sources

| Source | What It Informed |
|--------|-----------------|
| Alistair Cockburn, "Hexagonal Architecture" (2005) | Port ownership by domain, adapter as infrastructure, driving vs. driven ports |
| Robert C. Martin, *Clean Architecture* (2017) | Dependency Rule (always inward), boundary anatomy |
| Gamma et al., *Design Patterns* (1994) | Strategy Pattern for runtime backend swap |
| Eric Evans, *Domain-Driven Design* (2003) | Repository Pattern adapted for context access |
| Harry Percival & Bob Gregory, *Cosmic Python* (cosmicpython.com) | Python ABC-based ports, in-memory test fakes, manual composition root |
| LangGraph (`langchain-ai/langgraph`) | Two-tier separation: `BaseCheckpointSaver` + `BaseStore`, constructor injection |
| CrewAI (`crewAIInc/crewAI`) | Hierarchical scopes, `StorageBackend` protocol |
| AutoGen (`microsoft/autogen`) | `Memory` ABC (5 methods), `update_context()` preprocessor pattern |
| Context Engineering repo, chapters 08-10, 14 | Field operations catalog, resonance as query primitive, non-commutativity constraint |
| Automatos `ContextService` (`modules/context/service.py:47`) | 8 modes, 12 sections, `ContextResult` dataclass, `TokenBudgetManager` |
| Automatos `SharedContextManager` (`inter_agent.py:400`) | In-memory + Redis, 2h TTL, merge strategies, team access control |
| PEP 544 — Python Protocols | Structural typing alternative; decided against for critical ports |

## Appendix B: ContextMode Enum Values (Current)

From `modules/context/modes.py:13`:

```
CHATBOT, TASK_EXECUTION, HEARTBEAT_ORCHESTRATOR, HEARTBEAT_AGENT,
RECIPE, ROUTER, ORCHESTRATOR_STAGE, NL2SQL
```

New modes added by PRD-107 `ContextModeType`: `COORDINATOR`, `VERIFIER`.
