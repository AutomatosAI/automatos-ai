# Unified Pipeline Architecture (PRD-55)

> Cold-start reference for agents. Every entry point flows through the same pattern:
> **Ingest -> Route -> Execute -> Respond**

---

## Quick Context

This doc covers the **runtime pipeline** - how messages get from users to agents and back.
For the static layer model (API / Core / Modules / Consumers), see `ARCHITECTURE_OVERVIEW.md`.
For flow diagrams of workflows and deployment, see `FLOW_DIAGRAMS.md`.

---

## Entry Points

There are exactly 3 classes of entry point. All converge on the same execution layer.

| Entry Point | File | Protocol | Route Method |
|---|---|---|---|
| **Chatbot (Web UI)** | `api/chat.py` | SSE stream | User selects agent in UI |
| **Channel adapters** | `channels/<platform>_adapter.py` | Platform SDK | `UniversalRouter.route()` |
| **Webhooks** | `api/webhooks.py` | HTTP POST | `UniversalRouter.route()` |

---

## Pipeline Stages

### Stage 1: Ingest

Transform raw input into a normalized format.

**Chatbot path** (`api/chat.py` -> `consumers/chatbot/service.py`):
- Frontend sends messages array + `agent_id` + `chat_id` via POST
- `StreamingChatService.stream_response_with_agent()` is the entry method
- Messages are already structured `[{role, parts}]`

**Channel path** (`channels/base.py`):
- Platform SDK delivers raw message dict
- Each adapter implements `_to_envelope(platform_message)` -> `RequestEnvelope`
- `RequestEnvelope` is defined in `core/models/routing.py`:
  ```
  RequestEnvelope:
    id: UUID
    source: ChannelSource (TELEGRAM, SLACK, DISCORD, etc.)
    content: str
    workspace_id: UUID
    user: RequestUser
    override_agent_id: Optional[int]
    conversation_id: Optional[UUID]
  ```

**Webhook path** (`api/webhooks.py`):
- HTTP payload is converted directly to `RequestEnvelope`

### Stage 2: Route

Determine which agent handles the request.

**Chatbot**: No routing needed - `agent_id` comes from the frontend.

**Channels + Webhooks**: `UniversalRouter.route(envelope)` -> `RoutingDecision`

The router uses a 4-tier cascade (`core/routing/engine.py`):

| Tier | Name | Logic |
|---|---|---|
| 0 | Override | `envelope.override_agent_id` set explicitly |
| 1 | Cache | Hash-based lookup of previously routed identical messages |
| 2a | Rules | Pattern match on `routing_rules` table (source, keywords) |
| 2b | Channel default | `channel_connections.default_agent_id` |
| 3 | LLM classify | Intent classification -> best-fit agent for workspace |
| 4 | Fallback | First active agent in workspace |

**Output**: `RoutingDecision(route_type, agent_id, confidence, reasoning)`

### Stage 3: Execute

Run the agent with the user's content.

#### Chatbot Execution (full orchestration)

File: `consumers/chatbot/service.py` -> `stream_response_with_agent()`

```
1. Activate agent        -> AgentFactory.activate_agent(id) -> AgentRuntime
2. Build tools           -> get_chat_tools() + skill_tools from _build_agent_system_prompt()
3. SmartChatIntegration  -> .prepare(messages, tools, chat_id)
   - Intent classification (consumers/chatbot/intent_classifier.py)
   - Memory retrieval via SmartMemoryManager (consumers/chatbot/smart_memory.py)
   - Tool filtering via SmartToolRouter (consumers/chatbot/smart_tool_router.py)
   - Personality injection via personality.py (workspace settings)
   Returns: OrchestratedRequest(system_prompt, messages, tools, tool_choice)
4. Composio hints        -> ComposioHintService.build_hints() injected as system message
5. LLM call              -> agent_runtime.llm_manager.generate_response(messages, tools)
6. Tool loop             -> _handle_tool_calls_aisdk() (up to 5 iterations)
   - ToolExecutionTracker prevents loops (exact + semantic dedup)
   - ToolRouter.execute_and_format() -> UnifiedToolExecutor
   - Composio safety guard blocks destructive actions unless user asked
7. Store memory          -> smart_chat.store(user_msg, response, chat_id)
```

#### Channel Execution (simplified)

File: `channels/base.py` -> `handle_message()`

```
1. Route                 -> UniversalRouter.route(envelope) -> RoutingDecision
2. Execute               -> AgentFactory.execute_with_prompt(agent_id, content)
   - Internally: activate agent, build prompt, LLM call, tool loop
3. Respond               -> adapter.send_message(channel_id, response)
4. Track stats           -> UPDATE channel_connections SET message_count += 1
```

### Stage 4: Respond

**Chatbot**: AI SDK SSE stream format
- `0:"text chunk"\n` for text
- `d:{json}\n` for data (tool results, agent info)
- `e:{json}\n` for errors
- Managed by `consumers/chatbot/streaming.py`

**Channels**: `adapter.send_message(channel_id, text)` - platform SDK call

**Webhooks**: JSON HTTP response

---

## Key Components (file map)

### SmartChatIntegration

**File**: `consumers/chatbot/integration.py`

Drop-in orchestrator for the chatbot path. Replaces scattered memory/tool/prompt logic.

```python
smart_chat = SmartChatIntegration(workspace_id, agent_id, agent_name)

# Before LLM call:
orchestrated = await smart_chat.prepare(messages, tools, chat_id)
# orchestrated.system_prompt  -> personality + memory baked in
# orchestrated.messages       -> conversation with memory context injected
# orchestrated.tools          -> filtered subset of available tools
# orchestrated.tool_choice    -> "auto" | "none"
# orchestrated.requires_tools -> bool
# orchestrated.memory_context -> str or None
# orchestrated.intent         -> Intent enum value

# After LLM response:
await smart_chat.store(user_msg, assistant_response, chat_id)
```

**Internally coordinates** (all in `consumers/chatbot/`):

| Sub-component | File | Purpose |
|---|---|---|
| `SmartChatOrchestrator` | `smart_orchestrator.py` | Central coordinator |
| `SmartIntentClassifier` | `intent_classifier.py` | Classifies into 9 intents |
| `SmartMemoryManager` | `smart_memory.py` | Two-tier Mem0 retrieval/storage |
| `SmartToolRouter` | `smart_tool_router.py` | Filters 600+ tools to relevant subset |
| `AutomatosPersonality` | `personality.py` | Workspace-configured personality |
| `load_orchestrator_settings()` | `personality.py` | Cached workspace settings loader |

### Intent Classification

**File**: `consumers/chatbot/intent_classifier.py`

9 intent categories (rule-based, no LLM call):

| Intent | requires_tools | requires_memory | Example |
|---|---|---|---|
| `GREETING` | no | yes (name) | "Hello" |
| `CHITCHAT` | no | no | "What do you think about AI?" |
| `MEMORY_RECALL` | no | yes | "What's my name?" |
| `FACTUAL` | no | no | "What is Python?" |
| `DATA_QUERY` | yes | no | "Show me sales metrics" |
| `SEARCH` | yes | no | "Find docs about auth" |
| `EXTERNAL_ACTION` | yes | no | "Send a Slack message" |
| `CREATION` | yes | no | "Write a report" |
| `MULTI_STEP` | yes | yes | "Summarize emails and post to Slack" |

### SmartMemoryManager

**File**: `consumers/chatbot/smart_memory.py`

Two-tier system backed by Mem0:
- **Short-term**: Per-session, 2-minute cache TTL
- **Long-term**: Persistent per workspace+agent, stored in Mem0
- Scoped by `user_id = f"{workspace_id}:{agent_id or 'shared'}"`
- Extracts user facts (name, preferences) from conversation
- `retrieve_memories()` -> `MemoryResult(memories, user_context, formatted_context)`
- `store_conversation()` -> background task, non-blocking

### Personality System

**File**: `consumers/chatbot/personality.py`

- `load_orchestrator_settings(workspace_id)` -> cached dict from `workspace.settings.orchestrator`
- `get_happy_system_prompt()` builds the full system prompt with:
  - Base personality (friendly/professional/technical/custom)
  - Agent name and user name
  - Memory context (injected as bullet points)
  - Available tool names
  - Custom prompt if workspace has one

### AgentFactory

**File**: `modules/agents/factory/agent_factory.py`

Two main methods:

| Method | Used by | Returns |
|---|---|---|
| `activate_agent(id)` | Chatbot path | `AgentRuntime` (LLM manager, metadata, tools) |
| `execute_with_prompt(agent, prompt)` | Channel/webhook path | `{"response": str, "tool_results": [...]}` |

**AgentRuntime** (dataclass):
```
agent_id: int
metadata: AgentMetadata (name, type, description, skills)
llm_manager: LLMManager (configured for agent's model)
tools: List[Dict]  (assigned external apps - Composio metadata)
is_byok: bool
resolved_provider: str
```

### UniversalRouter

**File**: `core/routing/engine.py`

- Input: `RequestEnvelope`
- Output: `RoutingDecision(route_type, agent_id, confidence, reasoning)`
- 4-tier cascade (see Stage 2 above)
- Logs all decisions to `routing_decisions` table for analytics

### HeartbeatService

**File**: `services/heartbeat_service.py`

- APScheduler-based, singleton via `get_heartbeat_service()`
- **Orchestrator ticks**: Check agent health, run checklists
- **Agent ticks**: LLM-powered domain checks with configurable prompt
- **Daily summary**: Cron at 01:00 UTC, aggregates last 24h findings, stores in SmartMemoryManager
- Active-hours guard: skips ticks outside configured time window
- Results stored in `heartbeat_results` table

### Tool Execution

**File**: `consumers/chatbot/tool_router.py` (schema provider)
**File**: `modules/tools/` (execution)

- `get_chat_tools(agent_id, workspace_id)` -> OpenAI function schemas
- `ToolRouter.execute_and_format(name, args)` -> `{success, llm_context, frontend_data}`
- `UnifiedToolExecutor` handles Composio, file ops, DB queries, search
- `ToolExecutionTracker` prevents loops (exact hash + semantic similarity dedup)

---

## Channel Adapters

**Directory**: `channels/`

All 11 platforms supported, each inherits from `BaseChannelAdapter`:

| Platform | File | Required Config |
|---|---|---|
| Telegram | `telegram_adapter.py` | `bot_token` |
| Slack | `slack_adapter.py` | `bot_token`, `signing_secret` |
| Discord | `discord_adapter.py` | `bot_token` |
| Teams | `teams_adapter.py` | `app_id`, `app_password` |
| Google Chat | `google_chat_adapter.py` | `service_account_key` |
| Signal | `signal_adapter.py` | `phone_number` |
| iMessage | `imessage_adapter.py` | `apple_id` |
| IRC | `irc_adapter.py` | `server`, `channel`, `nickname` |
| Matrix | `matrix_adapter.py` | `homeserver_url`, `access_token` |
| LINE | `line_adapter.py` | `channel_access_token`, `channel_secret` |
| WhatsApp | `whatsapp_adapter.py` | `phone_number_id`, `access_token` |

**BaseChannelAdapter** abstract methods:
- `start()` / `stop()` - lifecycle
- `send_message(channel_id, text)` - outbound
- `test_connection()` -> `{"ok": bool, "detail": str}`
- `_to_envelope(platform_message)` -> `RequestEnvelope`

**ChannelManager** (`channels/manager.py`):
- Singleton via `get_channel_manager()`
- `start_all()` loads active connections from DB, spins up adapters
- `_create_adapter()` uses data-driven map with lazy imports

**API** (`api/channels.py`):
- CRUD endpoints at `/api/channels`
- `POST /{id}/start` and `POST /{id}/stop` for runtime control
- `POST /{id}/test` for connection validation
- `GET /analytics` for message stats

---

## Database Models (pipeline-relevant)

### channel_connections
```sql
id              UUID PK
workspace_id    UUID (indexed)
platform        VARCHAR(50)
config          JSON         -- encrypted credentials
status          VARCHAR(20)  -- active | inactive | error
metadata        JSON
default_agent_id INTEGER     -- US-027: default routing
message_count   INTEGER      -- incremented on each message
last_activity_at TIMESTAMP   -- updated on each message
created_at      TIMESTAMP
updated_at      TIMESTAMP
```

### routing_decisions
```sql
id              SERIAL PK
request_id      UUID (indexed)
envelope_hash   VARCHAR(64)
workspace_id    UUID
source          VARCHAR(50)
content         TEXT
route_type      VARCHAR(50)  -- agent | workflow | orchestrate
agent_id        INTEGER
confidence      FLOAT
cached          BOOLEAN
was_corrected   BOOLEAN
corrected_agent_id INTEGER
created_at      TIMESTAMP
```

### routing_rules
```sql
id              SERIAL PK
workspace_id    UUID
source_pattern  VARCHAR(100)
source_channel  VARCHAR(50)  -- channel-based routing
intent_keywords JSONB
target_agent_id INTEGER
target_workflow_id INTEGER
priority        INTEGER
```

### heartbeat_results
```sql
source_type     VARCHAR      -- orchestrator | agent
source_id       VARCHAR
workspace_id    UUID
status          VARCHAR      -- success | error | skipped
findings        JSON
actions_taken   JSON
tokens_used     INTEGER
created_at      TIMESTAMP
```

---

## Configuration

### Workspace Settings (stored in `workspaces.settings` JSON column)

```json
{
  "orchestrator": {
    "personality": {
      "mode": "friendly",           // friendly | professional | technical | custom
      "custom_prompt": "...",       // only when mode=custom
      "agent_name": "Automatos"
    },
    "heartbeat": {
      "enabled": true,
      "interval_minutes": 30,
      "active_hours_start": "08:00",
      "active_hours_end": "20:00",
      "timezone": "America/New_York",
      "checklist": "- Check agent health\n- Review pending tasks"
    }
  }
}
```

### Agent Configuration (stored in `agents.configuration` JSON column)

```json
{
  "model_config": {
    "provider": "openai",
    "model_id": "gpt-4o"
  },
  "heartbeat": {
    "enabled": true,
    "interval_minutes": 60,
    "prompt": "Check your domain for updates.",
    "auto_act": false
  }
}
```

### Environment Variables (pipeline-relevant)

| Variable | Default | Purpose |
|---|---|---|
| `HEARTBEAT_ENABLED` | `false` | Enable HeartbeatService at startup |
| `CHANNELS_ENABLED` | `false` | Enable ChannelManager at startup |
| `REDIS_URL` | - | Redis for heartbeat job store (falls back to memory) |
| `MEM0_API_KEY` | - | Mem0 API key for SmartMemoryManager |

---

## Startup Sequence

`main.py` lifespan handler:

```
1. Database ready (tables exist from docker-compose)
2. Redis lazy-init
3. Dashboard services init
4. if HEARTBEAT_ENABLED:
     HeartbeatService.start()  -> loads configs, schedules jobs, daily summary cron
5. if CHANNELS_ENABLED:
     ChannelManager.start_all()  -> loads active connections, starts adapters
```

Shutdown reverses: stop heartbeat -> stop channels -> shutdown dashboard.

---

## Deprecation Notes

- `get_memory_injector()` (`modules/memory/operations/injection.py`) is deprecated.
  Use `SmartChatIntegration` from `consumers/chatbot/integration.py` instead.
  The old injector still works in `stream_response_aisdk()` (non-agent chatbot path)
  but will be removed in a future release.

- `stream_response_aisdk()` is the legacy chatbot path (no agent activation, uses
  `memory_injector` directly). `stream_response_with_agent()` is the current path.

---

## Call Graph (chatbot path, simplified)

```
api/chat.py
  POST /api/chat
    -> StreamingChatService.stream_response_with_agent(chat_id, messages, agent_id, user_id)
        -> AgentFactory.activate_agent(agent_id) -> AgentRuntime
        -> _build_agent_system_prompt(agent_runtime) -> (system_prompt, skill_tools)
        -> get_chat_tools(agent_id, workspace_id) -> [OpenAI function schemas]
        -> SmartChatIntegration(workspace_id, agent_id, agent_name)
            -> SmartChatOrchestrator.__init__()
                -> SmartIntentClassifier
                -> SmartMemoryManager
                -> SmartToolRouter
            -> .prepare(messages, tools, chat_id)
                -> classifier.classify(query, messages) -> IntentResult
                -> memory_manager.retrieve_memories() -> MemoryResult
                -> tool_router.route(query, tools) -> ToolRoutingResult
                -> get_happy_system_prompt() -> str (with personality)
                -> OrchestratedRequest
            -> apply_orchestration_to_messages(orchestrated) -> llm_messages
        -> ComposioHintService.build_hints() -> system message
        -> agent_runtime.llm_manager.generate_response(llm_messages, use_tools)
        -> if tool_calls:
            -> _handle_tool_calls_aisdk()
                -> ToolExecutionTracker.should_skip_execution()
                -> ToolRouter.execute_and_format() -> UnifiedToolExecutor
                -> loop up to 5 iterations
        -> streaming_handler.stream_text_aisdk(full_response)
        -> smart_chat.store(user_msg, full_response, chat_id)
            -> SmartMemoryManager.store_conversation()
        -> chat_service.save_message() -> DB
```

## Call Graph (channel path, simplified)

```
channels/<platform>_adapter.py
  on_message callback
    -> BaseChannelAdapter.handle_message(platform_message)
        -> self._to_envelope(platform_message) -> RequestEnvelope
        -> UniversalRouter(db).route(envelope)
            -> tier0: check override_agent_id
            -> tier1: cache lookup
            -> tier2a: routing_rules match
            -> tier2b: channel default_agent_id
            -> tier3: LLM intent classification
            -> tier4: first active agent fallback
            -> RoutingDecision(agent_id)
        -> AgentFactory(db).execute_with_prompt(agent_id, content)
            -> activate_agent() -> AgentRuntime
            -> build prompt with context
            -> llm_manager.generate_response()
            -> handle tool calls if any
            -> {"response": str}
        -> self.send_message(reply_channel, response_text)
        -> self._update_activity_stats(db)
            -> UPDATE channel_connections SET message_count += 1
```
