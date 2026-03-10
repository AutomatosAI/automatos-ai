# Agent Zero Research — 2026-03-10

**Repo:** https://github.com/agent0ai/agent-zero (6k+ stars)
**What it is:** Personal, local-first agentic framework. Single agent runs on your machine, uses OS/terminal as primary tool, spawns subordinate agents in a hierarchy.

## Architecture Summary

- `agent.py` — core Agent class with message loop (`monologue()`)
- `python/tools/` — 23 tools as individual Python files
- `prompts/default/agent.system.md` — system prompt in editable markdown
- Extension system with lifecycle hooks: `agent_init`, `before_main_llm_call`, `tool_execute_before`, `tool_execute_after`, etc.
- 4 model slots per agent: chat, utility, browser, embedding
- WebSocket streaming with real-time user intervention

## Key Features for Future Reference

### 1. Hierarchical Agent Delegation (`call_subordinate.py`)
- Parent spawns subordinate with `Agent(number=self.number+1, config, context)`
- Bidirectional linking: subordinate stores ref to superior, parent stores ref to subordinate
- `profile` kwarg overrides system prompt for specialised tasks
- After completion, `subordinate.history.new_topic()` seals/compresses conversation
- **No agent selection logic** — parent decides via its own reasoning, not a roster

**Relevance to Automatos:** Auto already selects from roster (better). But conversation sealing pattern is worth adopting — compress sub-task context after delegation completes. Also: sub-agents can't currently request help from *other* agents mid-task. A `platform_request_agent_help` tool could enable this.

### 2. Behaviour Adjustment Tool (`behaviour_adjustment.py`)
- Agent modifies its own operating parameters mid-conversation
- Use cases: switch to creative mode for copywriting, cautious mode for debugging, verbose mode for research
- **Risks:** prompt injection via malicious documents, parameter drift over long sessions, no audit trail, gaming metrics
- **If built for Automatos:** whitelist adjustable params, log changes to activity feed, reset at conversation start, admin lock

### 3. A2A Protocol (`a2a_chat.py`)
- HTTP-based inter-instance agent comms using FastA2A protocol
- `send_message()` → `wait_for_completion()` (polling)
- Session persistence via cached `context_id` per remote agent URL
- **No service discovery** — requires explicit URL
- **Automatos equivalent:** Redis pub/sub channels (already have Redis scratchpads), `platform_message_agent` tool, internal A2A without network overhead

### 4. Task Scheduler (`scheduler.py`)
- Three task types: ScheduledTask (cron), AdHocTask (one-shot), PlannedTask (multi-step)
- Cron validation with regex
- `TaskScheduler` singleton manages all tasks
- **Key difference from Automatos:** agents can self-schedule future work mid-conversation
- **Automatos has:** UnifiedScheduler (APScheduler), HeartbeatService, RecipeSchedulerService — but agent-initiated scheduling not yet exposed as a tool

### 5. SKILL.md Open Standard
- Portable skill definitions compatible with Claude Code, Cursor, Goose
- Skills are declarative docs, execution handled by `code_execution_tool`
- Max 5 skills loaded at a time (context management)
- **Automatos approach:** DB-stored skills assigned to specific agents (avoids context bloat, admin-controlled risk)

### 6. Memory System
- Tools: `memory_save`, `memory_load`, `memory_delete`, `memory_forget`
- Storage via `Memory.get(agent)` abstraction
- Memories tagged with `area` (namespace) + arbitrary metadata
- **Memory Dashboard** in UI: view/search/delete/consolidate memories
- **Consolidation** — merge related/duplicate memories into single entries

### 7. Secrets Management
- Agents use credentials without seeing them — injected at runtime
- Project-scoped credential isolation
- Automatos uses Composio for OAuth but lacks a generic secrets vault

## Where Automatos Already Wins
- Multi-tenancy & workspaces (Agent Zero is single-user)
- Persistent agent identity & roster (Agent Zero agents are ephemeral numbered instances)
- Marketplace (plugins, skills, models)
- Composio integration (200+ third-party actions)
- RAG pipeline (S3 Vectors + chunking)
- Cloud-native SaaS on Railway
- Analytics & monitoring (PRD-72, 73, 74)
- Structured tool_use vs Agent Zero's dirty JSON parsing

## Related Automatos PRDs
- PRD-68: Progressive Complexity Routing (delegation patterns)
- PRD-50: Universal Router (agent selection)
- PRD-10: Workflow Engine (recipe chaining)
- PRD-05: Memory & Knowledge
- PRD-69: Agent Intelligence Layer (self-improvement)
- PRD-77: Agent Self-Scheduling & Memory Dashboard (NEW — inspired by this research)
