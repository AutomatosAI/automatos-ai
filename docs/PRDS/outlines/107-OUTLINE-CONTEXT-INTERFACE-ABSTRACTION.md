# PRD-107 Outline: Context Interface Abstraction

**Type:** Research + Design Outline
**Status:** Outline
**Depends On:** PRD-102 (Coordinator Architecture), PRD-100 (Master Research), Context Engineering repo (chapters 08-10, 14)
**Feeds Into:** PRD-108 (Memory Field Prototype), PRD-82B (Sequential Mission Coordinator), Phase 3 PRDs (110-116)

---

## 1. Problem Statement

Phase 3 swaps message-passing for shared semantic fields, but the coordinator (PRD-102) shouldn't know or care which context implementation runs behind the interface. Today, **every consumer directly imports `ContextService`** and calls `build_context()` with a specific `ContextMode` enum. There is no abstraction boundary — replacing the context layer means rewriting every caller.

### What's Tightly Coupled Today

| Coupling Point | File | Problem |
|---------------|------|---------|
| Direct `ContextService` instantiation | `smart_orchestrator.py`, `heartbeat_service.py`, `recipe_executor.py`, `routing/engine.py` | Every caller creates `ContextService(db_session)` directly — no injection, no interface |
| `ContextMode` enum dependency | `orchestrator/modules/context/modes.py` | Callers import and pass specific enum values — adding a neural field mode requires changing callers |
| `ContextResult` structure assumption | All callers | Callers destructure `context.system_prompt`, `context.tools`, `context.messages` — a field-based backend may not produce context in this shape |
| `SectionContext` internal exposure | All 12 section implementations | Sections reach into DB directly (agent records, skills, plugins) — no data access abstraction |
| `SharedContextManager` disconnected | `inter_agent.py` | Exists but is NOT integrated into `ContextService` — agents can't see peer context through the normal rendering pipeline |
| Tool loading strategy baked in | `ToolLoadingStrategy` enum per mode | Phase 3 may need different tool discovery (field-resonant tools vs. assigned tools) |

### Why This Matters

Without this abstraction:
- Phase 3 requires rewriting all 4+ callers of `ContextService`
- The coordinator (PRD-102) would be built against a concrete implementation, making Phase 3 a breaking change
- No way to A/B test message-passing vs. field-based context (PRD-108 experiment requires running both)
- No clean path to inject mission-level shared context (from `SharedContextManager`) into agent prompts

### What This PRD Delivers

A **port/adapter boundary** between context consumers (coordinator, chatbot, heartbeat) and context providers (current ContextService for Phase 2, neural field engine for Phase 3). The coordinator calls the port. The adapter maps to whichever backend is active. Zero coordinator changes when Phase 3 ships.

---

## 2. Prior Art Research Targets

### Architecture Patterns to Study

| Pattern | Source | Focus Areas | Key Question |
|---------|--------|-------------|--------------|
| **Hexagonal Architecture (Ports & Adapters)** | Alistair Cockburn (2005), alistair.cockburn.us | Port ownership, adapter composition, driving vs driven ports, dependency direction | How does the coordinator define what it needs without knowing how it's fulfilled? |
| **Clean Architecture Boundaries** | Robert C. Martin, *Clean Architecture* (2017), Chapters 17-22 | Dependency Rule, boundary anatomy, interface segregation at layer boundaries | What belongs in the domain layer vs. infrastructure layer for context? |
| **Strategy Pattern for Backend Selection** | Gamma et al., *Design Patterns* (1994) | Runtime backend swap, factory functions, config-driven selection | How to select message-passing vs. field adapter at startup? |
| **Repository Pattern (adapted for context)** | Eric Evans, *Domain-Driven Design* (2003) | Collection-like interface over diverse backends, query abstraction | Can context retrieval look like "get context for agent X" regardless of backend? |
| **Cosmic Python (Percival & Gregory)** | cosmicpython.com, free online | Python-specific ports/adapters with SQLAlchemy, ABC-based ports, in-memory test fakes | What's the Pythonic way to implement this without a DI framework? |

### Multi-Agent Framework Context Abstractions to Study

| Framework | Source | Focus Areas | Key Question |
|-----------|--------|-------------|--------------|
| **LangGraph** | `langchain-ai/langgraph` | `BaseCheckpointSaver` + `BaseStore` two-tier model, constructor injection at `compile()`, sync/async dual API | How does their two-tier separation (thread-scoped state vs. cross-thread memory) map to our per-agent vs. mission-shared context? |
| **CrewAI** | `crewAIInc/crewAI` | Unified `Memory` class, `StorageBackend` protocol, hierarchical scopes, composite scoring (semantic + recency + importance) | How does scope-based access control (`memory.scope("project/backend")`, `memory.slice()`) map to workspace/mission/agent context? |
| **AutoGen** | `microsoft/autogen` | `Memory` ABC (5 methods: `add`, `query`, `update_context`, `clear`, `close`), `MemoryContent` with MIME types, composable memory list | How does `update_context(model_context)` as context preprocessor pattern work for both message-passing and vector retrieval? |

### Context Engineering Theory to Study

| Document | Path | Focus Areas | Key Question |
|----------|------|-------------|--------------|
| **Neural Fields Foundations** | `docs/context-engineering/00_foundations/08_neural_fields_foundations.md` | Field operations (inject, query, decay), boundary permeability, resonance retrieval | What operations must the interface support for Phase 3? |
| **Persistence & Resonance** | `docs/context-engineering/00_foundations/09_persistence_and_resonance.md` | Decay formula `S(t) = S₀ × e^(-λt)`, resonance formula `R(A,B) = cos(θ) × |A| × |B| × S(A,B)`, attractor formation | What query primitives does resonance-based retrieval require vs. traditional retrieval? |
| **Field Orchestration** | `docs/context-engineering/00_foundations/10_field_orchestration.md` | Multi-field coordination, boundary configuration, field-to-field coupling, orchestrator-field data flow | How does a coordinator interact with multiple fields simultaneously? |
| **Unified Field Theory** | `docs/context-engineering/00_foundations/14_unified_field_theory.md` | Field collapse (continuous → concrete), three-layer flow (quantum → symbolic → field), non-commutativity of operations | What constraints does the unified theory place on the interface? |

### Key Patterns Discovered in Research

**LangGraph's Two-Tier Separation:** `BaseCheckpointSaver` handles thread-scoped state (conversation continuity, time-travel, fault tolerance). `BaseStore` handles cross-thread shared memory (user preferences, cross-agent knowledge). Both injected at `compile()` time. **Adopt:** per-conversation context (existing ContextService) is tier 1; mission-shared context (SharedContextManager → neural field) is tier 2. Separate interfaces, both injected.

**AutoGen's 5-Method Simplicity:** `add()`, `query()`, `update_context()`, `clear()`, `close()`. The `update_context(model_context)` pattern treats memory as a context preprocessor — it mutates the model's context by injecting retrieved memories as system messages. **Adopt:** the preprocessor pattern maps cleanly to both Phase 2 (inject retrieved messages) and Phase 3 (inject vector search results). But use immutable returns, not mutation.

**CrewAI's Hierarchical Scopes:** `memory.scope("workspace/agent/task")` restricts operations to a subtree. `memory.slice(["workspace/agent", "workspace/shared"])` creates read-only views across scopes. **Adopt:** scope-based access maps to workspace → mission → agent context hierarchy.

**Hexagonal + Strategy Hybrid (Cockburn + GoF):** Define the port (interface) owned by the coordinator domain. Use Strategy to select the adapter at startup via config. This gives clean dependency direction (Hexagonal) with simple runtime selection (Strategy). **Adopt:** manual factory function in composition root, no DI framework needed for our scale.

**Context Engineering Field Operations:** The theory defines 6 operation categories: pattern management (inject, decay), resonance operations (measure, detect, enhance), attractor operations (form, identify, track), boundary operations (configure permeability), field state operations (collapse, measure stability), and field integration (update, integrate document). **Key insight:** Phase 3 queries via resonance ("what resonates with X?"), not retrieval ("give me context for X"). The interface must support both paradigms.

**Non-Commutativity Warning (Unified Field Theory):** `inject(A); inject(B)` ≠ `inject(B); inject(A)` because attractors formed by A change how B resonates. The interface must preserve operation ordering — no batched unordered writes.

---

## 3. Interface Definition

### 3.1 Core Port: `ContextProvider`

The coordinator's primary interface to the context layer. Defined in domain layer (coordinator owns this contract).

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(frozen=True)
class AgentContext:
    """Immutable domain object — the coordinator's view of context.
    Any backend must produce this shape."""
    system_prompt: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "auto"

    # Metadata (observability, telemetry)
    mode: str = ""
    sections_included: list[str] = field(default_factory=list)
    sections_trimmed: list[str] = field(default_factory=list)
    token_estimate: int = 0
    token_budget: int = 0
    preparation_time_ms: float = 0.0
    memory_context: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)  # Extensible

class ContextProvider(ABC):
    """Driven port: coordinator calls this to build LLM-ready context."""

    @abstractmethod
    async def build_context(
        self,
        mode: str,
        agent: Any,
        workspace_id: str,
        messages: Optional[list[dict]] = None,
        task_description: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentContext: ...
```

### 3.2 Secondary Port: `SharedContextPort`

Mission-level shared context between agents. Separate from per-agent context because consumers differ.

```python
class SharedContextPort(ABC):
    """Cross-agent context sharing within a mission."""

    @abstractmethod
    async def inject(
        self, context_id: str, key: str, value: Any,
        agent_id: int, strength: float = 1.0
    ) -> None: ...

    @abstractmethod
    async def query(
        self, context_id: str, query: str, agent_id: int,
        top_k: int = 10
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def create_context(
        self, team_agent_ids: list[int],
        initial_data: Optional[dict] = None
    ) -> str: ...  # Returns context_id

    @abstractmethod
    async def destroy_context(self, context_id: str) -> None: ...
```

### 3.3 Operations the Interface Must Support

| Operation | Phase 2 (Message-Passing) | Phase 3 (Neural Field) | Interface Method |
|-----------|--------------------------|----------------------|-----------------|
| Build agent context | ContextService.build_context() → sections + budget | Field.query(agent_pattern) → resonant context | `ContextProvider.build_context()` |
| Share findings across agents | SharedContextManager.update_shared_context() | Field.inject(pattern, strength) | `SharedContextPort.inject()` |
| Query shared knowledge | SharedContextManager.get_shared_context() → dict lookup | Field.measure_resonance(query) → ranked results | `SharedContextPort.query()` |
| Create team workspace | SharedContextManager.create_shared_context() | Field.create() with boundary config | `SharedContextPort.create_context()` |
| Cleanup after mission | In-memory dict + Redis TTL expiry | Field decay + explicit destroy | `SharedContextPort.destroy_context()` |

### 3.4 What the Interface Does NOT Expose

- **Section-level control** — Callers don't pick sections. The adapter decides.
- **Token budget internals** — Callers don't set per-section budgets. The adapter manages.
- **Resonance/decay parameters** — Phase 3 internals. The adapter tunes them.
- **Storage backend details** — Redis, Postgres, FAISS, Qdrant — invisible to callers.

---

## 4. Phase 2 Implementation: Message-Passing Adapter

### 4.1 `DefaultContextProvider` — Wraps Existing ContextService

```python
class DefaultContextProvider(ContextProvider):
    """Phase 2 adapter: delegates to existing ContextService."""

    def __init__(self, db_session):
        self._service = ContextService(db_session)

    async def build_context(self, mode, agent, workspace_id,
                            messages=None, task_description=None, **kwargs):
        # Map string mode to ContextMode enum (backwards compat)
        context_mode = ContextMode[mode] if isinstance(mode, str) else mode
        result = await self._service.build_context(
            mode=context_mode,
            agent=agent,
            workspace_id=workspace_id,
            messages=messages,
            task_description=task_description,
            **kwargs,
        )
        # Map ContextResult → AgentContext (domain object)
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
        )
```

### 4.2 `RedisSharedContext` — Wraps Existing SharedContextManager

```python
class RedisSharedContext(SharedContextPort):
    """Phase 2 adapter: Redis-backed shared context with 2h TTL."""

    def __init__(self, shared_ctx_manager: SharedContextManager):
        self._manager = shared_ctx_manager

    async def inject(self, context_id, key, value, agent_id, strength=1.0):
        await self._manager.update_shared_context(
            context_id=context_id,
            agent=agent_id,
            updates={key: value},
            merge_strategy="append",
        )

    async def query(self, context_id, query, agent_id, top_k=10):
        ctx = await self._manager.get_shared_context(context_id)
        if not ctx:
            return []
        # Phase 2: simple key-value lookup, no semantic search
        return [{"key": k, "value": v} for k, v in ctx.context_data.items()]

    async def create_context(self, team_agent_ids, initial_data=None):
        ctx = await self._manager.create_shared_context(
            team=team_agent_ids,
            initial_context=initial_data or {},
        )
        return ctx.id

    async def destroy_context(self, context_id):
        # Redis TTL handles cleanup; explicit delete for immediate
        pass
```

### 4.3 Key Design Decision: MissionContextSection

New section for ContextService that reads from `SharedContextPort`:

```python
class MissionContextSection(BaseSection):
    """Injects shared mission context into agent's system prompt."""
    name = "mission_context"
    priority = 3  # After identity/task, before skills
    max_tokens = 2000

    async def render(self, ctx: SectionContext) -> str:
        mission_context_id = ctx.kwargs.get("mission_context_id")
        if not mission_context_id:
            return ""
        # Read shared context through the port
        shared_port: SharedContextPort = ctx.kwargs["shared_context_port"]
        entries = await shared_port.query(
            context_id=mission_context_id,
            query=ctx.task_description or "",
            agent_id=ctx.agent.id,
        )
        if not entries:
            return ""
        return self._format_shared_context(entries)
```

---

## 5. Phase 3 Implementation Preview: Neural Field Adapter

### 5.1 `NeuralFieldContextProvider`

```python
class NeuralFieldContextProvider(ContextProvider):
    """Phase 3 adapter: builds context from neural field resonance."""

    async def build_context(self, mode, agent, workspace_id,
                            messages=None, task_description=None, **kwargs):
        # Query field with agent pattern — resonant context surfaces
        query_pattern = self._build_query_pattern(agent, task_description)
        resonant_context = await self._field.measure_resonance(query_pattern)

        # Assemble into AgentContext (same shape as Phase 2)
        return AgentContext(
            system_prompt=self._assemble_prompt(resonant_context),
            tools=self._extract_resonant_tools(resonant_context),
            # ... same interface, different internals
        )
```

### 5.2 `NeuralFieldSharedContext`

```python
class NeuralFieldSharedContext(SharedContextPort):
    """Phase 3 adapter: shared context via neural field injection."""

    async def inject(self, context_id, key, value, agent_id, strength=1.0):
        # Inject as pattern with strength — resonance amplifies relevant info
        pattern = await self._embedder.embed(f"{key}: {value}")
        await self._field.inject(pattern, strength=strength)

    async def query(self, context_id, query, agent_id, top_k=10):
        # Resonance-based retrieval — "what resonates with this query?"
        query_pattern = await self._embedder.embed(query)
        return await self._field.query_resonance(query_pattern, top_k=top_k)
```

### 5.3 Why the Interface Holds

The same `ContextProvider.build_context()` and `SharedContextPort.inject()/query()` calls work for both:
- Phase 2: Assembles context from 12 sections + Redis shared state
- Phase 3: Queries neural field for resonant patterns + injects findings

The coordinator code is identical in both phases. Only the adapter (selected at startup) differs.

---

## 6. Key Design Questions

### Q1: ABC or Protocol for the port interface?

**Options:**
- **ABC (Abstract Base Class)** — Explicit inheritance, catches missing methods at class definition time, stronger contract
- **Protocol (typing.Protocol)** — Structural typing, no inheritance required, lighter weight, duck-typing with mypy support

**Recommendation: ABC.** For critical infrastructure ports (the context layer is foundational), explicit inheritance is safer. Protocol is better for internal adapters iterated quickly. The port must never silently fail to implement a method.

### Q2: One port or two ports?

**Options:**
- **Single `ContextProvider`** with all operations — simpler, one thing to inject
- **Two ports: `ContextProvider` + `SharedContextPort`** — different consumers, different lifecycles

**Recommendation: Two ports.** The coordinator uses both; individual agents only use `ContextProvider` for their own context. Shared context has a different lifecycle (mission-scoped) than per-agent context (per-call). Interface Segregation Principle applies — don't force agents to depend on shared context methods they don't use.

### Q3: Where does the interface live in the module hierarchy?

**Options:**
- `orchestrator/modules/context/ports.py` — close to current implementation
- `orchestrator/core/ports/context.py` — in a dedicated ports directory
- `orchestrator/domain/context.py` — Clean Architecture domain layer

**Recommendation: `orchestrator/core/ports/context.py`.** The `core/` directory already exists for cross-cutting concerns. A `ports/` subdirectory signals intent: these are contracts, not implementations. The coordinator (wherever it lives) imports from `core/ports/`, never from `modules/context/`.

### Q4: How does the coordinator get its adapter?

**Options:**
- **Constructor injection** — `CoordinatorService(context=adapter)` — simplest, testable
- **Factory function** — `create_context_provider(config)` returns the right adapter based on config
- **DI framework** — `dependency-injector` or `punq` library

**Recommendation: Constructor injection + factory function.** The factory reads config (`CONTEXT_BACKEND=default|neural_field`), constructs the adapter, and passes it to the coordinator. No DI framework needed at our scale. Matches Cosmic Python's manual composition root pattern.

### Q5: How to handle the `ContextMode` enum transition?

**Options:**
- **String modes** — `build_context(mode="chatbot")` instead of `ContextMode.CHATBOT`
- **Keep enum internally** — Port accepts strings, adapter maps to enum
- **New enum in ports** — Define mode constants in the port module

**Recommendation: Port accepts strings, adapter maps internally.** This decouples callers from the enum. New modes (e.g., `COORDINATOR`, `VERIFIER` from PRD-102/103) are added to the adapter config, not to the port interface. Phase 3 may define entirely different modes.

**Type safety consideration:** Define a `ContextModeType` string literal union or `StrEnum` in the ports module itself (not in `modes.py`). This gives callers autocomplete and typo-catching without coupling them to the concrete implementation's enum:

```python
# In orchestrator/core/ports/context.py
from enum import StrEnum

class ContextModeType(StrEnum):
    CHATBOT = "chatbot"
    HEARTBEAT = "heartbeat"
    TASK_EXECUTION = "task_execution"
    COORDINATOR = "coordinator"
    VERIFIER = "verifier"
    ROUTER = "router"
    # Phase 3 can add: FIELD_QUERY = "field_query"
```

The adapter maps these strings to its internal representation. Callers get type safety; the port stays decoupled from the backend.

### Q6: Sync vs. async interface?

**Recommendation: Async only.** Both backends are I/O-bound (Phase 2: DB queries + Redis; Phase 3: vector search + embedding). LangGraph's dual sync/async API is admirable but unnecessary complexity for our Python async-first codebase. All callers already use `await`.

### Q7: How to run Phase 2 and Phase 3 side-by-side for PRD-108 experiment?

**Options:**
- **Config toggle** — `CONTEXT_BACKEND=neural_field` switches everything
- **Per-mission adapter** — Mission config specifies which backend
- **Dual-write + A/B** — Both adapters run; results compared but only one used

**Recommendation: Per-mission adapter.** The factory can accept a mission-level override: `create_context_provider(config, mission_config)`. PRD-108's experiment runs specific missions through the neural field adapter while everything else uses the default. This enables controlled comparison.

---

## 7. Existing Codebase Touchpoints

### Files That Must Change

| File | Current State | Change Needed |
|------|-------------|---------------|
| `orchestrator/modules/context/service.py` | Direct instantiation by callers | Wrap in `DefaultContextProvider` adapter — ContextService itself unchanged |
| `orchestrator/modules/context/modes.py` | `ContextMode` enum imported by callers | Callers migrate to string mode names; adapter maps internally |
| `orchestrator/modules/context/result.py` | `ContextResult` frozen dataclass | `AgentContext` in ports module becomes the public contract; `ContextResult` becomes internal |
| `orchestrator/consumers/chatbot/smart_orchestrator.py` | `ContextService(self._db_session).build_context(mode=ContextMode.CHATBOT, ...)` | Accept `ContextProvider` via constructor; call `provider.build_context(mode="chatbot", ...)` |
| `orchestrator/services/heartbeat_service.py` | `ContextService(db).build_context(mode=ContextMode.HEARTBEAT_ORCHESTRATOR, ...)` | Same — inject `ContextProvider` |
| `orchestrator/api/recipe_executor.py` | `ContextService(db).build_context(mode=ContextMode.TASK_EXECUTION, ...)` | Same — inject `ContextProvider` |
| `orchestrator/core/routing/engine.py` | `ContextService(self._db).build_context(mode=ContextMode.ROUTER, ...)` | Same — inject `ContextProvider` |
| `orchestrator/modules/agents/communication/inter_agent.py` | `SharedContextManager` with in-memory + Redis | Wrap in `RedisSharedContext` adapter — SharedContextManager itself unchanged |
| `orchestrator/core/context_guard.py` | `ContextGuard` for auto-compaction | Must work with `AgentContext` not just `ContextResult` |

### Files That Must Be Created

| File | Purpose |
|------|---------|
| `orchestrator/core/ports/context.py` | `ContextProvider` ABC, `SharedContextPort` ABC, `AgentContext` dataclass |
| `orchestrator/core/ports/__init__.py` | Public exports |
| `orchestrator/modules/context/adapters/default.py` | `DefaultContextProvider` wrapping ContextService |
| `orchestrator/modules/context/adapters/shared_redis.py` | `RedisSharedContext` wrapping SharedContextManager |
| `orchestrator/modules/context/adapters/__init__.py` | Adapter exports |
| `orchestrator/modules/context/sections/mission_context.py` | `MissionContextSection` — injects shared mission context |
| `orchestrator/core/factories/context.py` | `create_context_provider(config)` factory function |

### Tables / Schema (No new tables)

This PRD introduces no new database tables. It's a code-level abstraction. However:
- The `MissionContextSection` will need `mission_context_id` passed via `kwargs` — this comes from PRD-101's mission_runs table
- Future: PRD-108 may need a `neural_field_embeddings` table, but that's PRD-108's scope

---

## 8. Acceptance Criteria for Full PRD-107

### Must Have
- [ ] **Port interfaces defined** — `ContextProvider` ABC and `SharedContextPort` ABC with full method signatures, type hints, docstrings
- [ ] **`AgentContext` domain object** — Immutable dataclass that replaces `ContextResult` as the public contract
- [ ] **`DefaultContextProvider` adapter** — Wraps existing `ContextService`, zero behavior change, all existing tests pass
- [ ] **`RedisSharedContext` adapter** — Wraps existing `SharedContextManager`, preserves 2h TTL behavior
- [ ] **Migration path documented** — Step-by-step guide to migrate each caller (chatbot, heartbeat, recipe, router) from direct `ContextService` to `ContextProvider`
- [ ] **Factory function** — `create_context_provider(config)` that selects adapter based on `CONTEXT_BACKEND` config
- [ ] **`MissionContextSection`** — New section that injects shared context into agent prompts via `SharedContextPort`
- [ ] **In-memory test fake** — `InMemoryContextProvider` for unit testing coordinators without DB/Redis
- [ ] **Architecture Decision Record** — Why ports/adapters over alternatives, with explicit tradeoffs

### Should Have
- [ ] **Mode string migration** — Callers use string mode names instead of `ContextMode` enum
- [ ] **`ContextGuard` compatibility** — Works with `AgentContext` (not just `ContextResult`)
- [ ] **Observability hooks** — Pre/post `build_context` callbacks for metrics (preparation_time_ms, sections_trimmed)
- [ ] **Phase 3 adapter skeleton** — `NeuralFieldContextProvider` stub with `NotImplementedError` methods, documenting what PRD-108+ must implement

### Nice to Have
- [ ] **Per-mission adapter override** — Factory supports mission-level config for A/B testing (Phase 2 vs Phase 3)
- [ ] **Composable shared context** — Multiple `SharedContextPort` instances per agent (fast recent + slow deep), following AutoGen's `memory=[A, B]` pattern
- [ ] **Scope-based access control** — `SharedContextPort.query()` respects hierarchical scopes (workspace → mission → agent), inspired by CrewAI
- [ ] **Async context cleanup** — Explicit `close()` on adapters for graceful shutdown

---

## 9. Risks & Dependencies

### Risks

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | Over-abstraction — interface too generic to be useful | High | Medium | Start with the narrowest interface that satisfies the coordinator. Expand only when Phase 3 demands it. "You Aren't Gonna Need It" for operations the coordinator doesn't call. |
| 2 | Leaky abstraction — backend-specific types leak through `AgentContext.metadata` | Medium | High | Strict rule: `metadata` contains only string/int/float/bool values. No `ContextResult`, no `SharedContext`, no `VectorSearchResult`. |
| 3 | Phase 3 requirements not yet understood well enough to design stable interface | High | Medium | PRD-108 prototype tests the interface. If it breaks, the interface is updated before Phase 3 PRDs. The prototype IS the validation gate. |
| 4 | Performance regression from adapter indirection | Low | Low | Adapter is a thin wrapper — one function call + dataclass mapping. Sub-millisecond overhead. Profile if concerned. |
| 5 | Migration disruption — changing all callers at once | Medium | Medium | Incremental migration: adapter wraps ContextService, callers migrate one-by-one, `ContextResult` deprecated but not removed until all callers migrate. |
| 6 | Two ports (ContextProvider + SharedContextPort) create coordination complexity | Medium | Low | Factory function wires both together. Coordinator receives both from the same factory. Test fake implements both. |
| 7 | `MissionContextSection` token budget competes with existing sections | Medium | Medium | Priority 3 (after identity/task, before skills). Max 2000 tokens. Budget manager drops it before critical sections. |

### Dependencies

| Dependency | PRD | Why |
|------------|-----|-----|
| Coordinator service design | PRD-102 | The coordinator is the primary consumer — its needs define the port interface |
| Mission schema | PRD-101 | `mission_context_id` comes from the mission_runs table |
| Context Engineering theory | Repo chapters 08-14 | Phase 3 operations define the upper bound of what the interface must eventually support |
| Existing ContextService | Built | Phase 2 adapter wraps it — must not break it |
| Existing SharedContextManager | Built | Phase 2 shared adapter wraps it — must not break it |

### Cross-PRD Notes

- PRD-102 (Coordinator): Must use `ContextProvider` port, not `ContextService` directly. New `COORDINATOR` mode added as a string, not an enum value.
- PRD-103 (Verification): Verifier context mode (`VERIFIER`) follows the same pattern — string mode, adapter maps internally.
- PRD-104 (Ephemeral Agents): Contractor agents receive context through same `ContextProvider` — no special path needed.
- PRD-105 (Budget): Budget enforcement hooks can wrap the `ContextProvider` (decorator pattern) — check budget before `build_context()`, not inside it.
- PRD-106 (Telemetry): `AgentContext.metadata` should include `context_tokens_used` and `sections_trimmed` for telemetry capture.
- PRD-108 (Memory Field Prototype): The neural field adapter implements `ContextProvider` + `SharedContextPort`. The experiment tests whether field-based context outperforms message-passing — same interface, different backend.

---

## Appendix: Research Sources

| Source | What It Informed |
|--------|-----------------|
| Alistair Cockburn, "Hexagonal Architecture" (2005) | Port ownership by domain, adapter as infrastructure, symmetric port model |
| Robert C. Martin, *Clean Architecture* (2017) | Dependency Rule (always inward), boundary anatomy, "Screaming Architecture" |
| Gamma et al., *Design Patterns* (1994) | Strategy Pattern for runtime backend swap |
| Eric Evans, *Domain-Driven Design* (2003) | Repository Pattern adapted for context access |
| Harry Percival & Bob Gregory, *Cosmic Python* (cosmicpython.com) | Python-specific ports/adapters, ABC-based ports, in-memory test fakes, manual composition root |
| LangGraph (`langchain-ai/langgraph`) | Two-tier separation: `BaseCheckpointSaver` (thread-scoped) + `BaseStore` (cross-thread), constructor injection at `compile()`, sync/async dual API |
| CrewAI (`crewAIInc/crewAI`) | Unified `Memory` class, hierarchical scopes, composite scoring, `StorageBackend` protocol |
| AutoGen (`microsoft/autogen`) | `Memory` ABC (5 methods), `update_context()` preprocessor pattern, composable memory list, `ComponentBase` config serialization |
| Context Engineering repo, chapters 08-10, 14 | Neural field operations catalog (inject, decay, resonate, attractor, boundary), orchestrator-field data flow, non-commutativity constraint, resonance as query primitive |
| Automatos ContextService (codebase) | 8 modes, 12 sections, `BaseSection` ABC, `ContextResult` dataclass, `TokenBudgetManager`, `SectionContext`, `ToolLoadingStrategy` — the concrete implementation the interface must wrap |
| Automatos SharedContextManager (codebase) | In-memory + Redis shared context, 2h TTL, merge strategies (override/append/consensus), team agent access control, audit trail — the concrete shared context the interface must wrap |
| PEP 544 — Python Protocols | Structural typing alternative to ABC; decided against for critical port interfaces |
