# PRD-68: Progressive Complexity Routing — Atom → Organism

**Version:** 2.0
**Status:** Ready for Implementation
**Priority:** P0
**Author:** Gar Kavanagh + Auto CTO
**Created:** 2026-02-28
**Updated:** 2026-02-28
**Dependencies:** PRD-50 (Universal Router — COMPLETE), PRD-59 (Workflow Engine V2 — MERGED), PRD-67 (CTO Agent — COMPLETE)
**Branch:** `ralph/progressive-complexity-routing`

---

## Executive Summary

Automatos has 850+ tools, 350+ models, multi-agent workflows, and a Neural Swarm architecture. But every chat message — from "Hi" to "Refactor the auth system" — runs through the same monolithic pipeline: regex intent classification → hardcoded tool category filter → full memory fetch → LLM with 15 pre-filtered tools.

This PRD makes the Atom → Organism progressive complexity model the **platform's primary routing architecture**. AutoBrain — which already sits at the front door of every chat request — evolves from a binary gate (RESPOND/DELEGATE) into an LLM-driven complexity assessor. The assessment threads through existing components that are **already wired to receive it** but currently get no data.

This is not a new system. The plumbing exists. We finish it.

### What We're Building

An AI platform that feels like ChatGPT for "Hi" (0.5s, 100 tokens) and seamlessly transforms into a multi-agent swarm for "Refactor the auth system" (5 min, 12K tokens). Dynamic scaling of intelligence — the user never knows they're using a different pipeline.

### What We're Deleting

~1,200 lines of dead code: mock API endpoints, deprecated streaming methods, legacy service stubs. One streaming format. One flow.

---

## 1. Current State (What's Wired Today)

### The Flow

```
api/chat.py:306    → AutoBrain.assess() → ComplexityAssessment
api/chat.py:406    → passes complexity_assessment to service
service.py:1259    → accepts complexity_assessment parameter
service.py:1371    → passes to smart_chat.prepare(complexity_assessment=...)
integration.py:94  → passes to smart_orchestrator
smart_orchestrator.py:162 → checks complexity_assessment.needs_memory ← FIELD DOESN'T EXIST YET
smart_orchestrator.py:199 → checks complexity_assessment.tool_hints  ← FIELD DOESN'T EXIST YET
```

**The wiring from chat.py → service.py → integration.py → smart_orchestrator.py is COMPLETE.** The downstream branching code is partially written — it checks `needs_memory` and `tool_hints` fields that don't exist on the `ComplexityAssessment` dataclass yet.

### What AutoBrain Returns Today

```python
# auto.py:55 — Current dataclass
@dataclass
class ComplexityAssessment:
    complexity: Complexity        # ATOM/MOLECULE/CELL/ORGAN/ORGANISM
    action: Action                # RESPOND/DELEGATE/WORKFLOW
    reasoning: str
    target_agent_id: Optional[int] = None
    target_agent_name: Optional[str] = None
    matched_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0
```

**Missing fields:** `needs_memory`, `tool_hints`, `needs_multi_agent`

### What AutoBrain Does Today

Pure regex. Returns RESPOND for atoms/platform/memory, DELEGATE for everything else. The `complexity` field is set but never drives downstream behavior. Everything that hits DELEGATE gets `Complexity.MOLECULE` regardless of actual complexity.

---

## 2. Dead Code Deletion (Do First)

Delete before building. Clean house.

### DELETE: `api/multi_agent.py` (567 lines)

**Why:** All 6 endpoints call `EnhancedOrchestratorService` methods that return `{"status": "legacy_mock"}`. Mounted in `main.py:696` but completely non-functional.

**Steps:**
1. Delete `orchestrator/api/multi_agent.py`
2. Remove import from `orchestrator/main.py` (line ~53): `from api.multi_agent import router as multi_agent_router`
3. Remove registration from `orchestrator/main.py` (line ~696): `app.include_router(multi_agent_router)`

### DELETE: `api/field_theory.py` (552 lines)

**Why:** Same as above. All endpoints return legacy mocks. Mounted in `main.py:697` but non-functional.

**Steps:**
1. Delete `orchestrator/api/field_theory.py`
2. Remove import from `orchestrator/main.py` (line ~54): `from api.field_theory import router as field_theory_router`
3. Remove registration from `orchestrator/main.py` (line ~697): `app.include_router(field_theory_router)`

### DELETE: Legacy mock methods in `modules/orchestrator/service.py`

**Why:** `FieldManager`, `CoordinationManager`, and 7 mock methods (`update_field_context`, `propagate_field_influence`, etc.) all return `{"status": "legacy_mock"}`.

**Steps:**
1. Delete lines ~298-347 (legacy class stubs and mock methods)push
2. Keep `EnhancedOrchestratorService` class itself — it's still imported by `api/orchestrator.py` for the task decomposition endpoint

### DELETE: `stream_response_aisdk()` in `consumers/chatbot/service.py`

**Why:** Explicitly deprecated at line 502. Emits deprecation warning. Not called from any active path. `stream_response_with_agent()` is the only active method.

**Steps:**
1. Delete the method (lines ~493-530)

### DELETE: Legacy SSE format support in `consumers/workflows/streaming.py`

**Why:** We standardize on AI SDK Data Stream format (`0:`, `d:`, `e:` prefixes). The legacy `data: {json}\n\n` format is not used by the active frontend.

**Steps:**
1. Remove any legacy format functions/comments
2. Keep only AI SDK format methods (`format_aisdk_*`)

**Total deletion: ~1,200 lines.**

---

## 3. The Implementation (5 Changes to 5 Files)

### Change 1: Evolve `ComplexityAssessment` dataclass (`auto.py`)

**File:** `orchestrator/consumers/chatbot/auto.py`
**Line:** 55

**BEFORE:**
```python
@dataclass
class ComplexityAssessment:
    complexity: Complexity
    action: Action
    reasoning: str
    target_agent_id: Optional[int] = None
    target_agent_name: Optional[str] = None
    matched_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0
```

**AFTER:**
```python
@dataclass
class ComplexityAssessment:
    complexity: Complexity
    action: Action
    reasoning: str
    target_agent_id: Optional[int] = None
    target_agent_name: Optional[str] = None
    matched_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0
    # PRD-68: Fields consumed by smart_orchestrator.py (lines 162, 190-199)
    needs_memory: bool = False
    tool_hints: List[str] = field(default_factory=list)
    needs_multi_agent: bool = False
```

**Why:** `smart_orchestrator.py` already checks `complexity_assessment.needs_memory` (line 162) and `complexity_assessment.tool_hints` (line 199). These fields just need to exist and be populated.

### Change 2: Add LLM-driven assessment to `AutoBrain.assess()` (`auto.py`)

**File:** `orchestrator/consumers/chatbot/auto.py`
**Method:** `assess()` at line 145

Keep the existing regex fast-paths for ATOM (greetings) and platform queries. Add a Redis cache layer and an LLM classification call for everything else.

**New `assess()` logic:**

```python
async def assess(self, message: str, conversation_length: int = 0) -> ComplexityAssessment:
    if not message or not message.strip():
        return ComplexityAssessment(
            complexity=Complexity.ATOM, action=Action.RESPOND,
            reasoning="Empty message", confidence=1.0
        )

    msg_lower = message.lower().strip()

    # ── Tier 1: Regex fast-paths (FREE, <5ms) ──
    # Keep existing _is_atom(), _match_platform_query(), _is_memory_recall()
    if self._is_atom(msg_lower):
        return ComplexityAssessment(
            complexity=Complexity.ATOM, action=Action.RESPOND,
            reasoning="Greeting or chitchat", confidence=0.95,
            needs_memory=False, tool_hints=[], needs_multi_agent=False
        )

    platform_tool = self._match_platform_query(msg_lower)
    if platform_tool:
        return ComplexityAssessment(
            complexity=Complexity.MOLECULE, action=Action.RESPOND,
            reasoning=f"Platform query ({platform_tool})",
            matched_tools=[platform_tool], confidence=0.90,
            needs_memory=False, tool_hints=[], needs_multi_agent=False
        )

    if self._is_memory_recall(msg_lower):
        return ComplexityAssessment(
            complexity=Complexity.CELL, action=Action.RESPOND,
            reasoning="Memory recall", confidence=0.85,
            needs_memory=True, tool_hints=[], needs_multi_agent=False
        )

    # ── Tier 2: Cache lookup (FREE, <5ms) ──
    cached = await self._cache_lookup(msg_lower)
    if cached:
        return cached

    # ── Tier 3: LLM classification (~$0.001, ~200ms) ──
    assessment = await self._llm_classify(message, conversation_length)

    # Cache the result
    await self._cache_store(msg_lower, assessment)
    return assessment
```

**The `_llm_classify()` method:**

```python
async def _llm_classify(self, message: str, conversation_length: int) -> ComplexityAssessment:
    """Use a lightweight LLM to classify complexity. Any model, any provider."""
    from core.llm.manager import create_llm_manager

    llm = create_llm_manager(service_name="complexity_assessor")

    # Build agent context (lightweight — names + one-line descriptions only)
    agent_summaries = self._get_agent_summaries()  # DB query, cached

    prompt = f"""Classify this user message for an AI platform.

Available agents: {agent_summaries}
Conversation turn: {conversation_length}

Message: "{message}"

Return ONLY valid JSON:
{{
  "complexity": "atom|molecule|cell|organ|organism",
  "action": "respond|delegate|workflow",
  "tool_hints": ["domain1", "domain2"],
  "needs_memory": true/false,
  "needs_multi_agent": true/false,
  "reasoning": "one sentence"
}}

Rules:
- atom: Greetings, chitchat, simple factual. No tools.
- molecule: Needs ONE tool/agent. "Send email", "check Jira", "search docs".
- cell: Needs tools + memory/conversation context. "Reply to that email we discussed".
- organ: Needs multiple agents coordinating. "Research bug, plan fix, open PR".
- organism: Enterprise-scale multi-step. "Refactor auth across all services".
- tool_hints: short domain keywords like "email", "github", "jira", "code", "database". Empty for atom.
- needs_memory: true if the message references past conversations or user preferences.
- needs_multi_agent: true only for organ/organism level tasks.
- action: "respond" for atom, "delegate" for molecule/cell, "workflow" for organ/organism."""

    try:
        response = await llm.generate_response(
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content)  # Parse JSON from LLM

        complexity = Complexity(result.get("complexity", "molecule"))
        action_str = result.get("action", "delegate")
        action = Action(action_str)

        return ComplexityAssessment(
            complexity=complexity,
            action=action,
            reasoning=result.get("reasoning", "LLM classified"),
            confidence=0.85,
            needs_memory=result.get("needs_memory", False),
            tool_hints=result.get("tool_hints", []),
            needs_multi_agent=result.get("needs_multi_agent", False),
        )
    except Exception:
        # LLM failed — fall back to DELEGATE as MOLECULE (current behavior)
        logger.exception("[AutoBrain] LLM classification failed, falling back to DELEGATE")
        return ComplexityAssessment(
            complexity=Complexity.MOLECULE, action=Action.DELEGATE,
            reasoning="LLM classification failed — defaulting to delegate",
            confidence=0.50, needs_memory=False, tool_hints=[], needs_multi_agent=False,
        )
```

**The `_cache_lookup()` and `_cache_store()` methods:**

```python
async def _cache_lookup(self, msg_lower: str) -> Optional[ComplexityAssessment]:
    """Check Redis for a cached complexity assessment."""
    try:
        from core.redis.client import get_redis_client
        redis = get_redis_client()
        if not redis:
            return None
        cache_key = f"complexity:{self._workspace_id}:{hashlib.sha256(msg_lower.encode()).hexdigest()[:16]}"
        cached = redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return ComplexityAssessment(
                complexity=Complexity(data["complexity"]),
                action=Action(data["action"]),
                reasoning=data.get("reasoning", "cached"),
                confidence=data.get("confidence", 0.90),
                needs_memory=data.get("needs_memory", False),
                tool_hints=data.get("tool_hints", []),
                needs_multi_agent=data.get("needs_multi_agent", False),
            )
    except Exception:
        logger.debug("[AutoBrain] Cache lookup failed, continuing to LLM")
    return None

async def _cache_store(self, msg_lower: str, assessment: ComplexityAssessment) -> None:
    """Cache a complexity assessment in Redis. TTL: 24 hours."""
    try:
        from core.redis.client import get_redis_client
        redis = get_redis_client()
        if not redis:
            return
        cache_key = f"complexity:{self._workspace_id}:{hashlib.sha256(msg_lower.encode()).hexdigest()[:16]}"
        data = {
            "complexity": assessment.complexity.value,
            "action": assessment.action.value,
            "reasoning": assessment.reasoning,
            "confidence": assessment.confidence,
            "needs_memory": assessment.needs_memory,
            "tool_hints": assessment.tool_hints,
            "needs_multi_agent": assessment.needs_multi_agent,
        }
        ttl = int(os.getenv("COMPLEXITY_CACHE_TTL_HOURS", "24")) * 3600
        redis.setex(cache_key, ttl, json.dumps(data))
    except Exception:
        logger.debug("[AutoBrain] Cache store failed, non-critical")
```

**The `_get_agent_summaries()` method:**

```python
def _get_agent_summaries(self) -> str:
    """Get lightweight agent descriptions for LLM context. Cached per request."""
    from core.models.agents import Agent
    agents = self._db.query(Agent.name, Agent.description).filter(
        Agent.workspace_id == self._workspace_id,
        Agent.is_active == True
    ).all()
    if not agents:
        return "No custom agents configured."
    return ", ".join(f"{a.name}: {(a.description or '')[:60]}" for a in agents)
```

**Model Configuration:**

Add to `core/llm/manager.py` `SERVICE_CATEGORY_MAP`:
```python
"complexity_assessor": "complexity_assessor_llm"
```

Default in `.env.example`:
```bash
# Complexity Assessor — any lightweight model. Change in System Settings UI.
COMPLEXITY_ASSESSOR_LLM_PROVIDER=openrouter
COMPLEXITY_ASSESSOR_LLM_MODEL=meta-llama/llama-3.1-8b-instruct
```

This means: any model, any provider, swappable at runtime via system settings. Free Llama 8B by default, swap to Haiku/Flash/whatever performs best.

### Change 3: ATOM fast-path in `service.py`

**File:** `orchestrator/consumers/chatbot/service.py`
**Location:** Lines 1340-1374 (the SmartChatIntegration block)

**BEFORE (lines 1353-1374):**
```python
is_simple = self.prompt_analyzer.is_simple_message(latest_text)
all_tools = None if is_simple else get_chat_tools(agent_id=agent_id, workspace_id=self.workspace_id)
if not is_simple and skill_tools:
    all_tools = (all_tools or []) + skill_tools

llm_messages = self.prompt_analyzer.convert_to_llm_messages(messages, system_prompt="", available_tools=all_tools)

orchestrated = await smart_chat.prepare(
    messages=llm_messages,
    available_tools=all_tools or [],
    chat_id=chat_id,
    complexity_assessment=complexity_assessment
)
llm_messages = apply_orchestration_to_messages(orchestrated)
use_tools = orchestrated.tools if orchestrated.requires_tools else None
```

**AFTER:**
```python
# ── PRD-68: Branch on complexity level ──
_complexity = (
    complexity_assessment.complexity
    if complexity_assessment
    else Complexity.MOLECULE  # default if no assessment (e.g., direct agent selection)
)

if _complexity == Complexity.ATOM:
    # ATOM: No tools, no memory, no SmartChatIntegration.
    # Just system prompt + conversation → LLM. Fastest path.
    logger.info("[PRD-68] ATOM path — skipping tools, memory, orchestration")
    llm_messages = self.prompt_analyzer.convert_to_llm_messages(
        messages, system_prompt="", available_tools=None
    )
    # Minimal system prompt (personality only, no tool instructions)
    from consumers.chatbot.personality import get_happy_system_prompt
    _sys_prompt = get_happy_system_prompt(
        user_name=None, agent_name=agent_runtime.metadata.name,
        msg_count=len(messages), memories=[], tool_names=[],
        orchestrator_settings=None
    )
    llm_messages.insert(0, {"role": "system", "content": _sys_prompt})
    use_tools = None
    orchestrated = None
else:
    # MOLECULE / CELL / ORGAN / ORGANISM: Run SmartChatIntegration
    from consumers.chatbot.tool_router import get_chat_tools
    all_tools = get_chat_tools(agent_id=agent_id, workspace_id=self.workspace_id)
    if skill_tools:
        all_tools = (all_tools or []) + skill_tools

    llm_messages = self.prompt_analyzer.convert_to_llm_messages(
        messages, system_prompt="", available_tools=all_tools
    )

    orchestrated = await smart_chat.prepare(
        messages=llm_messages,
        available_tools=all_tools or [],
        chat_id=chat_id,
        complexity_assessment=complexity_assessment
    )
    llm_messages = apply_orchestration_to_messages(orchestrated)
    use_tools = orchestrated.tools if orchestrated.requires_tools else None
```

**What this does:**
- ATOM: Skips `get_chat_tools()` (saves ~50ms DB query), skips `SmartChatIntegration.prepare()` (saves intent classification, memory fetch, tool routing). Just builds a minimal system prompt and goes straight to LLM.
- Everything else: Runs the existing pipeline with the complexity assessment flowing through (memory skip / tool_hints already handled by `smart_orchestrator.py`).

**Delete:** Remove the `is_simple = self.prompt_analyzer.is_simple_message(latest_text)` check entirely. AutoBrain's ATOM detection replaces it. `prompt_analyzer.is_simple_message()` becomes dead code for this path.

### Change 4: `tool_hints` integration in `SmartToolRouter` (`smart_tool_router.py`)

**File:** `orchestrator/consumers/chatbot/smart_tool_router.py`
**Method:** `route()` at line ~185

`smart_orchestrator.py:199` already passes `tool_hints` to the router. The router needs to use them.

**Add to the `route()` method — BEFORE existing logic:**

```python
async def route(self, query, available_tools, conversation_context=None, tool_hints=None):
    # ── PRD-68: tool_hints from AutoBrain take priority over regex ──
    if tool_hints:
        hint_matched = []
        for tool in available_tools:
            tool_name = tool.get("function", {}).get("name", "").lower()
            tool_desc = tool.get("function", {}).get("description", "").lower()
            for hint in tool_hints:
                hint_lower = hint.lower()
                if hint_lower in tool_name or hint_lower in tool_desc:
                    hint_matched.append(tool)
                    break
        if hint_matched:
            # Always include core tools alongside hint-matched tools
            core = [t for t in available_tools if t.get("function", {}).get("name") in self.CORE_TOOLS]
            combined = hint_matched + [c for c in core if c not in hint_matched]
            logger.info(f"[ToolRouter] PRD-68 hint match: {len(hint_matched)} tools for hints={tool_hints}")
            return ToolRoutingResult(
                should_include_tools=True,
                filtered_tools=combined[:15],
                tool_choice="auto",
                reasoning=f"Tool hints: {tool_hints}"
            )
        # Hints didn't match anything — fall through to existing logic

    # ... existing intent-based routing below (unchanged) ...
```

**What this does:** When AutoBrain says `tool_hints: ["email"]`, the router searches ALL available tools for "email" in their name or description. No hardcoded category dict. "email" finds `GMAIL_SEND_EMAIL`, `OUTLOOK_SEND_MAIL`, `composio_email_*`, etc. Tools become discoverable instead of filterable.

### Change 5: `api/chat.py` — ORGAN/ORGANISM workflow bridge

**File:** `orchestrator/api/chat.py`
**Location:** After line 313 (the existing action branching)

**BEFORE (line 314):**
```python
if complexity_assessment.action == Action.RESPOND:
    effective_agent_id = get_default_agent_id(db, ctx.workspace_id)
    use_system_llm = True
    ...
else:
    # DELEGATE — Universal Router picks the right specialized agent.
    ...
```

**AFTER:**
```python
if complexity_assessment.action == Action.RESPOND:
    effective_agent_id = get_default_agent_id(db, ctx.workspace_id)
    use_system_llm = True
    logger.info(
        f"[Auto] Direct response (complexity={complexity_assessment.complexity.value}): "
        f"agent_id={effective_agent_id} with orchestrator LLM"
    )
elif complexity_assessment.action == Action.WORKFLOW:
    # ── PRD-68 Phase 2: ORGAN/ORGANISM → workflow execution via chat ──
    # Create transient workflow and execute through PRD-59 pipeline.
    # Stream workflow events as AI SDK format back to chat.
    effective_agent_id = get_default_agent_id(db, ctx.workspace_id)
    use_system_llm = True
    # TODO Phase 2: Bridge to execute_workflow_with_progress()
    # For now, WORKFLOW falls through to DELEGATE behavior.
    # The complexity_assessment with needs_multi_agent=True is passed through,
    # letting the orchestrator handle it as a rich Cell-level response.
    logger.info(
        f"[Auto] WORKFLOW detected (complexity={complexity_assessment.complexity.value}) "
        f"— Phase 2 bridge not yet active, using delegate path"
    )
    # Fall through to DELEGATE logic below

if complexity_assessment.action in (Action.DELEGATE, Action.WORKFLOW):
    # DELEGATE / WORKFLOW — Universal Router picks the right specialized agent.
    try:
        ingestor = ChatbotIngestor()
        envelope = ingestor.ingest(
            message=message_text, agent_id=None,
            session_id=chat_id, request_context=ctx,
        )
        routing_request_id = str(envelope.id)
        universal_router = UniversalRouter(db, cache=get_routing_cache())
        routing_decision = await universal_router.route(envelope)
    except Exception:
        logger.exception("[chat] Router failed — falling back to default agent")
        routing_decision = None

    if routing_decision is not None and routing_decision.route_type == "agent" and routing_decision.agent_id is not None:
        effective_agent_id = routing_decision.agent_id
        logger.info(f"[Auto] Router → agent_id={effective_agent_id}")
    elif routing_decision is not None and routing_decision.route_type == "orchestrate":
        effective_agent_id = get_default_agent_id(db, ctx.workspace_id)
        use_system_llm = True
    else:
        effective_agent_id = get_default_agent_id(db, ctx.workspace_id)
        use_system_llm = True
```

**What this does:** Adds the WORKFLOW action handling as a clean branch. Phase 1 treats WORKFLOW same as DELEGATE (the LLM still handles it, just with richer context from the assessment). Phase 2 will bridge to `execute_workflow_with_progress()`.

---

## 4. CTO Agent Compatibility (PRD-67)

The CTO detection in `api/chat.py:298` runs BEFORE AutoBrain:

```python
elif _is_admin and (cto_id := _get_cto_agent_id(db)):
    effective_agent_id = cto_id
    use_system_llm = True
```

**CTO bypasses AutoBrain entirely.** This is correct — CTO Auto always gets full Cell-level context (tools + memory + codebase access). No complexity assessment needed for the platform builder.

The CTO path remains unchanged. PRD-68 only affects the `else` branch (non-admin, non-explicit-agent).

---

## 5. Streaming Format — Standardize on AI SDK

**Decision:** AI SDK Data Stream format (`0:`, `d:`, `e:` prefixes) for everything.

The chat frontend already handles:
- `0:"text chunk"` — streaming text
- `d:{"type":"tool-start",...}` — tool lifecycle
- `d:{"type":"tool-end",...}` — tool completion
- `d:{"type":"workflow-update",...}` — workflow progress (US-015 widget)
- `d:{"type":"finish",...}` — response complete

When Phase 2 bridges chat → workflow, workflow stage events map to `d:` events:

```
Workflow stage_start → d:{"type":"workflow-update","stage":"Planning","status":"started"}
Workflow subtask_update → d:{"type":"workflow-update","step":"Research agent analyzing..."}
Workflow stage_complete → d:{"type":"workflow-update","stage":"Planning","status":"complete"}
Workflow final result → 0:"Here's what I found..." (streamed text)
Workflow complete → d:{"type":"finish","finishReason":"stop"}
```

**No format adapter needed.** Same protocol, same parser, same frontend.

---

## 6. File Impact Summary

| File | Change | Phase |
|------|--------|-------|
| `api/multi_agent.py` | **DELETE** (567 lines) | 1 |
| `api/field_theory.py` | **DELETE** (552 lines) | 1 |
| `main.py` | Remove imports + registration for above 2 files | 1 |
| `modules/orchestrator/service.py` | Delete legacy mock methods + classes (~40 lines) | 1 |
| `consumers/chatbot/service.py` | Delete `stream_response_aisdk()` (~50 lines). Add ATOM branch (~30 lines). Remove `is_simple_message()` usage. | 1 |
| `consumers/chatbot/auto.py` | Add 3 fields to dataclass. Add `_llm_classify()`, `_cache_lookup()`, `_cache_store()`, `_get_agent_summaries()` methods. (~120 lines new) | 1 |
| `consumers/chatbot/smart_tool_router.py` | Add `tool_hints` parameter + hint-matching block (~25 lines) | 1 |
| `api/chat.py` | Add WORKFLOW action branch (~15 lines) | 1 |
| `core/llm/manager.py` | Add `"complexity_assessor"` to SERVICE_CATEGORY_MAP (1 line) | 1 |
| `.env.example` | Add COMPLEXITY_ASSESSOR_LLM_PROVIDER/MODEL (2 lines) | 1 |
| `consumers/workflows/streaming.py` | Remove legacy format code (cleanup) | 1 |

**Net: ~1,200 lines deleted. ~200 lines added.**

---

## 7. Phasing

### Phase 1: Core Routing (This PR)

1. Delete dead code (multi_agent.py, field_theory.py, legacy mocks, deprecated method)
2. Add fields to `ComplexityAssessment` dataclass
3. Add LLM classification + Redis cache to `AutoBrain.assess()`
4. Add ATOM fast-path in `service.py` (skip tools/memory/orchestration)
5. Add `tool_hints` to `SmartToolRouter`
6. Add WORKFLOW branch placeholder in `chat.py`
7. Add `complexity_assessor` to LLM manager service map

**Deliverable:** "Hi" → 0.5s, 100 tokens. "Send email" → correctly classified as MOLECULE with `tool_hints: ["email"]`, tools discovered by hint match. "Reply to that thread" → CELL with `needs_memory: true`. "Refactor auth" → ORGAN with `needs_multi_agent: true` (falls through to delegate for now).

### Phase 2: Workflow Bridge (Follow-up PR)

1. When `action == WORKFLOW`, create transient workflow from chat message
2. Execute via `execute_workflow_with_progress()` with PhaseSelector
3. Stream stage events as AI SDK `d:` events back to chat
4. Display multi-agent progress inline in chat UI

**Deliverable:** "Research the bug, plan a fix, open a PR" → user sees planning, execution, and results streamed back in chat, powered by the full PRD-59 Neural Swarm pipeline.

---

## 8. Verification & Metrics

| Metric | Before | After (Phase 1) |
|--------|--------|------------------|
| "Hi" latency | ~2s (loads tools, runs intent classifier, fetches memory) | <1s (ATOM bypass) |
| "Hi" token cost | ~500 tokens (system prompt + tool descriptions) | ~100 tokens (minimal prompt) |
| "Send email" tool accuracy | Depends on regex matching "EXTERNAL_ACTION" | LLM classifies with `tool_hints: ["email"]`, router discovers email tools |
| Cache hit rate (repeat patterns) | 0% (no cache) | >70% after 1 week |
| Complexity assessment cost | $0 (regex) | ~$0.001 per uncached message (Haiku/Llama) |
| Dead code | ~1,200 lines of mocks/stubs | 0 |

---

## 9. Non-Goals

- **NOT replacing the UniversalRouter** — AutoBrain classifies complexity + action. The router (for DELEGATE) still picks which agent handles it. Different jobs.
- **NOT replacing SmartIntentClassifier** — Intent classification still runs for MOLECULE/CELL paths. It helps tool routing when tool_hints aren't enough. But it no longer gates the pipeline.
- **NOT rewriting the workflow engine** — PRD-59's PhaseSelector and Neural Swarm remain the ORGAN/ORGANISM execution vehicle. Phase 2 just connects chat to it.
- **NOT building new modules** — No new files except possibly a test file. This is rewiring, not building.

---

## 10. Open Questions

1. **Cache scope:** Should the complexity cache be per-workspace (current design) or global? Same message in different workspaces might have different complexities based on available agents.
   **Recommendation:** Per-workspace. Different agent configurations = different complexity assessments.

2. **LLM fallback behavior:** If the complexity assessor model is unavailable and Redis cache misses, should we fall back to regex (current behavior) or error?
   **Recommendation:** Fall back to DELEGATE as MOLECULE (current behavior). Never block the user.

3. **Phase 2 transient workflows:** Should ORGAN/ORGANISM workflows created from chat be visible in the workflows UI, or ephemeral?
   **Recommendation:** Visible, with a "created from chat" tag. Users should be able to re-run them.
