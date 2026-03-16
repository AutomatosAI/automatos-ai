# System Audit: Context & Prompt Fragmentation

**Date:** 2026-03-12
**Scope:** All LLM-calling code paths in `orchestrator/`
**Purpose:** Map every location that builds prompts, loads tools, injects memory, or constructs LLM context — to inform PRD-80 (Unified Context Service)

---

## 1. Codebase Overview

| Metric | Value |
|--------|-------|
| Total `.py` files in `orchestrator/` | 651 |
| Total lines of Python | ~202,831 |
| Largest file | `modules/tools/discovery/platform_executor.py` (3,806 lines) |
| Files containing class definitions | ~60 |
| Code paths that call LLMs | 9 (confirmed) |

---

## 2. Discovery: The 9 Code Paths

### Path 1 — Chatbot (Auto)

**File:** `consumers/chatbot/smart_orchestrator.py` (569 lines)
**Entry point:** `SmartChatOrchestrator.prepare_request()`

| Capability | How it works | Lines |
|-----------|--------------|-------|
| **A. Prompt Building** | Calls `get_happy_system_prompt()` from `personality.py` which assembles 6 sections: base identity, platform skill, memory context, tool guidance, action response style, self-learning instruction. Then appends session context, daily logs, platform action summary, datetime as separate string concatenations. | 263-378 |
| **B. Tool Loading** | Receives `available_tools` from caller (`service.py:598` calls `get_tools_for_agent()`). Then `SmartToolRouter.route()` filters by intent classification, with semantic ranking (PRD-64) or keyword fallback. Platform tools always included. | 223-261 |
| **C. Memory Injection** | Two paths: (1) Context Router (`unified_memory.retrieve_context()`) returning a `ContextBundle` with L1/L2/L3 data, or (2) fallback `SmartMemoryManager.retrieve_memories()` doing two-tier Mem0 search. Memory formatted as bullet list via `_format_memories_for_llm()`. | 159-221 |
| **D. Daily Logs** | `smart_memory.get_daily_logs()` fetches from Mem0, max 2000 chars. Appended to system prompt as `## Recent Activity`. Only in fallback path; Context Router path gets daily logs from bundle. | 316-358 |
| **E. Personality** | `AutomatosPersonality` class in `personality.py` (401 lines). Workspace-level settings (personality_mode, communication_style) loaded with 2-min TTL cache from DB. PromptRegistry integration (PRD-58). | personality.py |
| **F. Platform Actions** | `get_action_registry().build_prompt_summary()` injected into system prompt. Markdown catalog of all platform_execute actions grouped by category. | 361-367 |
| **G. Datetime** | Injected as a system message at position 1 in the message array (not in system prompt). Format: `Current date/time (UTC): YYYY-MM-DD HH:MM`. | 373-379 |
| **H. Message Conversion** | `_convert_messages()` strips system messages, converts `parts` format to plain text, handles file attachments. | 431-465 |
| **I. Token Budget** | **NONE.** No token awareness. Prompt grows unbounded. |  |
| **J. Composio Hints** | Not in chatbot — handled by tool_router's `get_tools_for_agent()` which enriches `composio_execute` description with available actions. |  |
| **K. Skill Content** | Not in chatbot path. Auto uses `personality.py` platform skill, not agent skills. |  |

**Dependencies:** `intent_classifier.py`, `smart_memory.py`, `smart_tool_router.py`, `personality.py`, `unified_memory_service.py`, `context_router.py`

---

### Path 2 — Agent Task Execution (AgentFactory)

**File:** `modules/agents/factory/agent_factory.py` (1,424 lines)
**Entry point:** `AgentFactory.execute_with_prompt()` (line 714)

| Capability | How it works | Lines |
|-----------|--------------|-------|
| **A. Prompt Building** | `_build_agent_system_prompt()` assembles: identity (name, ID, type, description), persona (custom or from persona table), task context, skills (SKILL.md via `skill_loader`), plugins (tier1+tier2 from `PluginContextService`), Composio apps section, dependency context instructions, skill tool usage instructions, response formatting rules. Returns `(prompt_text, skill_tool_schemas)`. | 1136-1308 |
| **B. Tool Loading** | `get_tools_for_agent()` from `tool_router.py` — loads from `ToolRegistry`, validates per-agent access, enriches `composio_execute` with app actions, adds `platform_execute` dispatcher. Skill tool schemas appended separately. | 811-824 |
| **C. Memory Injection** | Short-term memory from `agent_runtime.memory[-3:]` injected as user/assistant message pairs. No long-term Mem0 retrieval (controlled by `use_memory` param, defaults True but only uses runtime memory). | 780-785 |
| **D. Daily Logs** | **NONE.** |  |
| **E. Personality** | **NONE.** No personality/workspace settings applied. Raw agent identity only. |  |
| **F. Platform Actions** | **NOT in system prompt.** But `platform_execute` dispatcher schema IS in the tool list via `get_tools_for_agent()`. The LLM sees the tool but has no markdown catalog unless the agent's skill mentions it. |  |
| **G. Datetime** | **NONE.** No datetime injection. |  |
| **H. Message Conversion** | Direct message construction: system prompt, optional memory messages, optional recipe context, then user prompt. | 759-845 |
| **I. Token Budget** | **NONE.** |  |
| **J. Composio Hints** | `_inject_composio_hints()` and `_inject_composio_recipe_hints()` — adds action enum + hints as system messages. | 826-839 |
| **K. Skill Content** | Full SKILL.md loaded via `skill_loader.load_skill_core()` for each active skill. Plus tool schemas extracted from `skill.tools_schema`. | 1172-1219 |

**Key finding:** `build_prompt_summary()` for platform actions is NOT called in agent_factory. Agents executing tasks cannot see the platform action catalog in their prompt, only the raw `platform_execute` tool schema. This means task-executing agents don't know what actions are available without the skill mentioning them.

---

### Path 3 — Heartbeat Service (Orchestrator Tick)

**File:** `services/heartbeat_service.py` (1,446 lines)
**Entry point:** `_orchestrator_tick_llm()` (line 382)

| Capability | How it works | Lines |
|-----------|--------------|-------|
| **A. Prompt Building** | Inline f-string: personality mode, level instruction (silent/notify/act_notify/autonomous), communication style suffix, checklist block, platform action summary. Hardcoded "You are the Automatos orchestrator performing a scheduled health check." | 426-433 |
| **B. Tool Loading** | `registry.to_dispatcher_schema()` — single platform_execute dispatcher only. No core tools, no composio. | 399-400 |
| **C. Memory Injection** | **NONE.** |  |
| **D. Daily Logs** | **NONE.** |  |
| **E. Personality** | Partial: loads `personality_mode` and `communication_style` from workspace settings. No full `AutomatosPersonality` template. | 393-396 |
| **F. Platform Actions** | `registry.build_prompt_summary()` injected into system prompt. | 425 |
| **G. Datetime** | Injected in user message: `Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}` | 437 |
| **H. Message Conversion** | Direct construction: system + user messages. Context trimming at exchange boundaries after 10 messages. | 435-524 |
| **I. Token Budget** | **NONE.** Max 5 tool loop iterations as implicit limit. Context trimming keeps last 2 exchanges. | 447, 510-524 |
| **J. Composio Hints** | **NONE.** |  |
| **K. Skill Content** | **NONE.** |  |

---

### Path 4 — Heartbeat Service (Agent Tick)

**File:** `services/heartbeat_service.py` (1,446 lines)
**Entry point:** `_agent_tick()` (line 579)

| Capability | How it works | Lines |
|-----------|--------------|-------|
| **A. Prompt Building** | Delegates to `AgentFactory.execute_with_prompt()` — the prompt is just the heartbeat instruction + board task context. System prompt comes from agent_factory's `_build_agent_system_prompt()`. | 676-698 |
| **B. Tool Loading** | Via AgentFactory which calls `get_tools_for_agent()`. | 688-698 |
| **C. Memory Injection** | `use_memory=False` passed to factory — no memory. | 698 |
| **D. Daily Logs** | **NONE.** |  |
| **E. Personality** | **NONE.** (Via agent_factory which has no personality.) |  |
| **F. Platform Actions** | Via agent_factory tool list (dispatcher schema). No markdown catalog in prompt. |  |
| **G. Datetime** | Injected in user prompt text: `Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}` | 678 |
| **H. Message Conversion** | Via agent_factory. |  |
| **I. Token Budget** | **NONE.** |  |
| **J. Composio Hints** | Via agent_factory. |  |
| **K. Skill Content** | Via agent_factory's `_build_agent_system_prompt()`. |  |

---

### Path 5 — Recipe Executor

**File:** `api/recipe_executor.py` (1,453 lines)
**Entry point:** `_execute_step()` (line 66)

| Capability | How it works | Lines |
|-----------|--------------|-------|
| **A. Prompt Building** | `_build_system_prompt()` (line 443) — uses AgentRuntime's pre-built `system_prompt` from factory activation. Falls back to minimal identity-only prompt. Step-level context injected via Scratchpad and prior step outputs. | 125-126, 443-466 |
| **B. Tool Loading** | `get_chat_tools()` (alias for `get_agent_tools()`) which wraps `get_tools_for_agent()`. Scratchpad write tool injected separately. | 228-239 |
| **C. Memory Injection** | Recipe memory via `RecipeMemoryService` — retrieves recipe-scoped and agent-scoped memories from Mem0. Injected as system message. | Not in _execute_step directly |
| **D. Daily Logs** | **NONE.** |  |
| **E. Personality** | **NONE.** |  |
| **F. Platform Actions** | Via tool list (dispatcher schema). No markdown catalog. |  |
| **G. Datetime** | **NONE.** |  |
| **H. Message Conversion** | Direct message construction with scratchpad context, prior step outputs, and user instruction. |  |
| **I. Token Budget** | **NONE.** |  |
| **J. Composio Hints** | Pre-resolved action names passed to factory via `composio_action_names` param. |  |
| **K. Skill Content** | Via agent_factory's pre-built prompt (from activation). |  |

---

### Path 6 — Execution Manager

**File:** `modules/agents/execution/execution_manager.py` (1,344 lines)
**Entry point:** `_execute_subtask()` (around line 810)

| Capability | How it works | Lines |
|-----------|--------------|-------|
| **A. Prompt Building** | Hardcoded "professional_system_prompt" (40 lines of instructions), then prepends `_build_agent_system_prompt()` from factory if agent has skills. Appends workspace guidance and conversion tool instructions. | 821-904 |
| **B. Tool Loading** | Via `AgentFactory.execute_with_prompt()` which calls `get_tools_for_agent()`. | 906 |
| **C. Memory Injection** | `use_memory=True` passed to factory — uses runtime memory. | 910 |
| **D. Daily Logs** | **NONE.** |  |
| **E. Personality** | **NONE.** Custom "professional" prompt replaces personality. | 821-847 |
| **F. Platform Actions** | Via factory tool list. No markdown catalog. |  |
| **G. Datetime** | **NONE.** |  |
| **H. Message Conversion** | Via factory. |  |
| **I. Token Budget** | **NONE.** |  |
| **J. Composio Hints** | Via factory. |  |
| **K. Skill Content** | Via factory's `_build_agent_system_prompt()` — DUPLICATED. The execution manager calls `_build_agent_system_prompt()` directly AND then passes a `system_prompt` to `execute_with_prompt()`, which means the factory won't rebuild it. But the execution_manager concatenation with `professional_system_prompt` creates a different prompt than what the factory alone would produce. | 863-868 |

**Key finding:** The execution_manager and agent_factory produce DIFFERENT system prompts for the same agent. The execution_manager prepends skill content and appends a hardcoded professional prompt, workspace guidance, and conversion tool instructions.

---

### Path 7 — Universal Router (Tier 3 — LLM Classification)

**File:** `core/routing/engine.py` (887 lines)
**Entry point:** `_classify_with_llm()` (line 452)

| Capability | How it works | Lines |
|-----------|--------------|-------|
| **A. Prompt Building** | `_build_classification_prompt()` — hardcoded routing prompt with agent descriptions, semantic hints. PRD-58 PromptRegistry integration attempted. | 662-726 |
| **B. Tool Loading** | **NONE.** Text-only LLM call. |  |
| **C-K** | **NONE.** Router is special-purpose — no memory, no tools, no personality, no datetime. |  |

---

### Path 8 — Orchestrator Stages

**Files:** `modules/orchestrator/stages/*.py` (11 files)
**Key files:** `task_decomposer.py`, `complexity_analyzer.py`, `quality_assessor.py`, `agent_negotiation.py`, `context_engineering.py`

| Capability | How it works |
|-----------|--------------|
| **A. Prompt Building** | Each stage builds its own prompt. `task_decomposer.py` has a hardcoded system prompt ("You are an expert task decomposition system") with PromptRegistry fallback. `complexity_analyzer.py`, `quality_assessor.py`, `agent_negotiation.py` all have custom `_build_*_prompt()` methods. `context_engineering.py` has `_build_enhanced_prompt()` for RAG-augmented subtask prompts. |
| **B. Tool Loading** | **NONE.** All stages are text-only LLM calls. |
| **C-K** | **NONE.** Stages are internal orchestration — no memory, tools, or personality. |

---

### Path 9 — NL2SQL

**File:** `modules/nl2sql/query/nl2sql_service.py`
**Entry point:** `NaturalLanguageToSQLService.generate_sql()`

| Capability | How it works |
|-----------|--------------|
| **A. Prompt Building** | `_build_prompt()` — schema-specific prompt with SQL rules, table schema, semantic layer, few-shot examples, error context for self-correction. Fully self-contained. |
| **B. Tool Loading** | **NONE.** Text-only. |
| **C-K** | **NONE.** |

---

## 3. Capability Matrix

| Capability | Path 1 (Chatbot) | Path 2 (Factory) | Path 3 (HB Orch) | Path 4 (HB Agent) | Path 5 (Recipe) | Path 6 (ExecMgr) | Path 7 (Router) | Path 8 (Stages) | Path 9 (NL2SQL) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A. Prompt Building | personality.py | _build_agent_system_prompt | inline f-string | via factory | via factory | hardcoded + factory | _build_classification_prompt | per-stage | _build_prompt |
| B. Tool Loading | SmartToolRouter | get_tools_for_agent | dispatcher only | via factory | get_chat_tools | via factory | none | none | none |
| C. Memory | Context Router / SmartMemory | runtime memory only | none | none | recipe memory | runtime memory | none | none | none |
| D. Daily Logs | smart_memory.get_daily_logs | none | none | none | none | none | none | none | none |
| E. Personality | full AutomatosPersonality | none | partial (mode+style) | none | none | hardcoded professional | none | none | none |
| F. Platform Actions (prompt) | build_prompt_summary | **MISSING** | build_prompt_summary | **MISSING** | **MISSING** | **MISSING** | none | none | none |
| G. Datetime | system message | none | in user msg | in user msg | none | none | none | none | none |
| H. Message Convert | _convert_messages | direct | direct | via factory | direct | via factory | direct | direct | direct |
| I. Token Budget | **NONE** | **NONE** | implicit (5 iter) | **NONE** | **NONE** | **NONE** | **NONE** | **NONE** | **NONE** |
| J. Composio Hints | via tool_router | _inject_composio_hints | none | via factory | pre-resolved | via factory | none | none | none |
| K. Skill Content | platform skill only | full SKILL.md | none | via factory | via factory | via factory | none | none | none |

---

## 4. Classification of Findings

### 4.1 Critical Fragmentation Issues

**F1: Platform action catalog missing from 4 of 6 agent-facing paths.**
- Chatbot and heartbeat orchestrator inject `build_prompt_summary()` into the system prompt.
- Agent factory, agent heartbeat, recipe executor, and execution manager do NOT inject it. The LLM gets the `platform_execute` tool schema but has no markdown catalog explaining what actions exist or what parameters they take.
- **Impact:** Task-executing agents cannot effectively use platform actions unless their SKILL.md happens to list them.

**F2: Memory injection inconsistency across 4 paths.**
- Chatbot: full Context Router or SmartMemory with two-tier Mem0 search.
- Factory: only runtime memory (last 3 exchanges, no Mem0).
- Heartbeat (agent): explicitly `use_memory=False`.
- Recipe: recipe-scoped Mem0, separate path.
- **Impact:** Agents performing tasks have no long-term user context. Heartbeat agents have no memory at all.

**F3: Personality applied to only 1.5 of 6 agent-facing paths.**
- Chatbot: full AutomatosPersonality with workspace settings.
- Heartbeat orchestrator: personality_mode and communication_style only (no template).
- All other paths: no personality at all.
- **Impact:** Agent responses have inconsistent tone depending on how they're invoked.

**F4: No token budget management anywhere.**
- Zero paths implement token budget awareness.
- Prompt sections grow unbounded.
- Adding platform action summary (~1200 tokens) + skill content (~1800 tokens) + memory (~400 tokens) + daily logs (~500 tokens) is fine now (~7K total for task execution), but adding more sections will silently overflow model context limits.
- Only the heartbeat orchestrator has implicit limits (5 tool iterations, context trimming at 10 messages).

### 4.2 Duplication

**D1: `_build_agent_system_prompt()` called from 3 different places with different wrapping.**
1. `agent_factory.activate_agent()` — stored on AgentRuntime, used as-is.
2. `agent_factory.execute_with_prompt()` — called as fallback when no cached prompt exists.
3. `execution_manager._execute_subtask()` — called directly, then prepended to a separate hardcoded prompt.
- The execution_manager produces a fundamentally different prompt than the factory alone.

**D2: Tool loading has 3 entry points that all resolve to `get_tools_for_agent()`.**
- `get_tools_for_agent()` in `tool_router.py` (primary)
- `get_agent_tools()` in `tool_router.py` (convenience wrapper)
- `get_chat_tools()` re-exported via `consumers/chatbot/tool_router.py`
- All resolve to the same function. This is fine structurally but confusing.

**D3: Platform action summary injected via string concatenation in 2 places.**
- `smart_orchestrator.py:363` — chatbot path
- `heartbeat_service.py:425` — orchestrator heartbeat path
- Both call `get_action_registry().build_prompt_summary()` and append to system prompt. Any change to injection format requires updating both.

**D4: Datetime context injected 3 different ways.**
- Chatbot: system message at position 1 in message array.
- Heartbeat orchestrator: in the user message text.
- Heartbeat agent: in the user prompt text.
- Factory, recipe, execution_manager, router, stages, NL2SQL: not injected at all.

### 4.3 Dead / Redundant Code

**R1: `SmartToolRouter` in `consumers/chatbot/smart_tool_router.py` (442 lines).**
- Used ONLY by `smart_orchestrator.py` for intent-based tool filtering.
- Contains its own `IntentClassifier`, `TOOL_CATEGORIES`, `INTENT_TO_TOOLS` mappings, and semantic embedding ranking.
- Could be replaced by a tool loading strategy in ContextService.

**R2: `AutomatosPersonality.get_platform_skill()` in `personality.py`.**
- Returns a hardcoded platform skill block. Only used by the chatbot.
- Should be a section in ContextService.

**R3: `professional_system_prompt` in `execution_manager.py` (lines 821-847).**
- 40 lines of hardcoded instructions. Completely different from any other path.
- Duplicates some instructions that should be in the agent's skill or a shared section.

### 4.4 Missing Capabilities

| Missing | Affected Paths | Severity |
|---------|---------------|----------|
| Platform action catalog in prompt | Factory, Recipe, ExecMgr, HB Agent | HIGH |
| Long-term memory (Mem0) | Factory, ExecMgr, HB Agent | MEDIUM |
| Daily logs | All except chatbot | MEDIUM |
| Personality/tone | Factory, Recipe, ExecMgr, HB Agent | MEDIUM |
| Datetime context | Factory, Recipe, ExecMgr | LOW |
| Token budget | ALL paths | HIGH |
| Session context (L1) | All except chatbot | LOW |

---

## 5. Dependency Map

```
smart_orchestrator.py
  -> personality.py (get_happy_system_prompt, load_orchestrator_settings)
  -> smart_memory.py (retrieve_memories, get_daily_logs, store_daily_summary)
  -> smart_tool_router.py (route)
  -> intent_classifier.py (classify)
  -> unified_memory_service.py (retrieve_context, store_exchange, update_session)
  -> action_registry.py (build_prompt_summary)

agent_factory.py
  -> tool_router.py (get_tools_for_agent)
  -> skill_loader.py (load_skill_core)
  -> plugin_context_service.py (get_assigned_plugins, build_tier1_summary, build_tier2_content_sync)
  -> composio_cache models (AgentAppAssignment, ComposioAppCache, ComposioActionCache)

heartbeat_service.py
  -> personality.py (load_orchestrator_settings)
  -> action_registry.py (to_dispatcher_schema, build_prompt_summary)
  -> agent_factory.py (execute_with_prompt)
  -> platform_executor.py (PlatformActionExecutor)
  -> LLMManager (direct)

recipe_executor.py
  -> agent_factory.py (activate_agent)
  -> tool_router.py (get_agent_tools)
  -> LLMManager (direct)

execution_manager.py
  -> agent_factory.py (_build_agent_system_prompt, execute_with_prompt)
  -> MemoryPromptInjector (imported but usage unclear in current code)

engine.py (UniversalRouter)
  -> LLMManager (create_llm_manager)
  -> IntentClassifier
  -> PromptRegistry

stages/*.py
  -> LLMManager variants
  -> ContextOptimizer (from search module)
```

---

## 6. File Inventory (All Files Inspected)

| File | Lines | Role in Context Building |
|------|-------|-------------------------|
| `consumers/chatbot/smart_orchestrator.py` | 569 | Central chatbot coordinator — prompt + tools + memory |
| `consumers/chatbot/personality.py` | 401 | AutomatosPersonality prompt template |
| `consumers/chatbot/smart_tool_router.py` | 442 | Intent-based tool filtering |
| `consumers/chatbot/smart_memory.py` | 788 | Two-tier Mem0 retrieval + daily logs |
| `consumers/chatbot/intent_classifier.py` | ~400 | Intent classification (9 intents) |
| `consumers/chatbot/service.py` | 1,976 | StreamingChatService, tool loop, SSE |
| `modules/agents/factory/agent_factory.py` | 1,424 | Agent activation, prompt building, tool loop |
| `modules/agents/execution/execution_manager.py` | 1,344 | Subtask execution with custom prompts |
| `services/heartbeat_service.py` | 1,446 | Scheduled ticks, LLM tool loop |
| `api/recipe_executor.py` | 1,453 | Step-by-step recipe execution |
| `core/routing/engine.py` | 887 | Universal router with LLM tier |
| `modules/orchestrator/stages/*.py` | ~6,000 | 9-stage pipeline (11 files) |
| `modules/nl2sql/query/nl2sql_service.py` | ~400 | Text-to-SQL prompt builder |
| `modules/tools/tool_router.py` | 762 | Tool loading + execution |
| `modules/tools/discovery/action_registry.py` | 192 | Platform action registry |
| `modules/memory/unified_memory_service.py` | 2,068 | 5-layer memory stack |
| `modules/memory/context_router.py` | ~500 | Context assembly (PRD-79) |
| `channels/base.py` | ~150 | Channel adapter base (delegates to router + factory) |

---

## 7. Key Metrics for Migration Planning

| Metric | Count |
|--------|-------|
| Distinct prompt-building functions | 8 |
| Distinct tool-loading entry points | 3 (all resolve to 1) |
| Distinct memory retrieval paths | 4 |
| Places `build_prompt_summary()` is called | 2 |
| Places `to_dispatcher_schema()` is called | 2 |
| Places `get_happy_system_prompt()` is called | 2 |
| Places `_build_agent_system_prompt()` is called | 3 |
| Lines of prompt-building code (estimated) | ~600 |
| Lines deletable after ContextService migration | ~450 |
