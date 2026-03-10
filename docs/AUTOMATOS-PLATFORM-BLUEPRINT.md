# Automatos Platform Blueprint

Reference document for Auto's RAG knowledge base. Covers platform architecture, capabilities, data model, and available tools.

---

## 1. What Automatos Is

Automatos is an AI agent platform that lets users create, manage, and orchestrate AI agents within isolated workspaces. Users interact primarily through **Auto** — the platform's orchestrator brain — which routes requests to specialist agents, executes platform operations, manages memory, and coordinates multi-step workflows (recipes).

Each workspace is a tenant boundary: agents, documents, tools, skills, plugins, and models are scoped to the workspace. Users authenticate via Clerk, and workspace membership controls access.

---

## 2. Architecture

### 2.1 Auto — The Orchestrator Brain

Auto is the entry point for all user messages. It uses **Progressive Complexity Routing** (ATOM to ORGANISM scale) to determine how much infrastructure a request needs:

| Level | Description | Token Budget | Example |
|-------|-------------|-------------|---------|
| **ATOM** | Direct response — greetings, chitchat | <200 | "hello", "how are you?" |
| **MOLECULE** | Single tool call, no deep memory | ~1K | "list my agents" |
| **CELL** | Memory + tools + reasoning | ~3K | "what did we discuss about the API?" |
| **ORGAN** | Multi-agent coordination | ~6K | "have the researcher analyze this, then the coder implement it" |
| **ORGANISM** | Full pipeline with learning + feedback | ~12K | Enterprise workflow orchestration |

**3-Tier Assessment** determines complexity before execution:
- **Tier 1**: Redis cache lookup (<5ms, free) — repeat patterns
- **Tier 2**: Regex fast-paths (<5ms, free) — greeting/chitchat detection
- **Tier 3**: LLM classification (~200ms, ~$0.001) — ambiguous requests

Auto decides one of three **actions**: `RESPOND` (answer directly), `DELEGATE` (route to a sub-agent), or `WORKFLOW` (trigger multi-agent pipeline).

### 2.2 Universal Router

Routes requests to specialist agents using a tiered strategy:

- **Tier 0**: User override — explicit agent_id or workflow_id set
- **Tier 1**: Cache lookup — RoutingCache hit from previous decisions
- **Tier 2a**: Routing rules table — workspace-configured source pattern matches
- **Tier 2b**: Trigger subscriptions — event-driven routing (e.g., Jira triggers)
- **Tier 2.5**: Semantic similarity — cosine similarity on agent description embeddings (2048-dim vectors)
- **Tier 2c**: IntentClassifier — keyword matching against routing rules
- **Tier 3**: LLM classification — fallback when all heuristic tiers fail

Returns `None` only when all tiers (including LLM) fail to route.

### 2.3 Agent System

Agents are the execution units. Each agent has:
- **name**, **description**, **agent_type**, **status** (active/inactive)
- **model_config**: provider, model_id, temperature, max_tokens, top_p, penalties, fallback_model_id
- **tags**: lightweight capability labels
- **persona_id** or **custom_persona_prompt**: behavioral personality
- **semantic_embedding**: 2048-float vector for routing similarity
- **workspace_id**: tenant isolation

**Agent types** (used in creation):
- `chatbot` — conversational, user-facing
- `worker` — task execution, background processing
- `researcher` — analysis, data gathering, investigation
- `coder` — development, code generation, debugging

**Marketplace agents** use single-table architecture: `owner_type` distinguishes workspace vs. marketplace agents. Marketplace agents can be cloned into workspaces.

**Agent heartbeats**: Agents can be configured with periodic heartbeat schedules — interval, prompt, active hours, proactive level (silent/notify/act_notify/autonomous), and notification channel.

### 2.4 Memory System

Two-tier memory powered by Mem0:

- **Global memory** (workspace-scoped): Personal facts about the user — name, role, company, location, preferences. Available across all agents.
- **Agent memory** (workspace + agent-scoped): Tool-specific patterns, workflow preferences, domain context. Scoped to a single agent.

Memory classification rules:
- Personal facts (name, job, company) → global
- Tool/workflow patterns (Slack channels, email contacts, repo preferences) → agent
- Preferences → both tiers

User IDs for memory scoping follow the pattern `ws_{workspace_id}_agent_{agent_id}`.

Additional memory infrastructure:
- `memory_items` table for structured storage
- `RecipeMemoryService` for workflow execution memory
- `WorkflowMemoryIntegrator` for cross-agent memory in pipelines
- Redis caching with 2-minute TTL for frequent retrievals

### 2.5 Tool System

**Platform Actions** — built-in operations Auto can perform on the Automatos platform itself. Registered via `ActionRegistry` with permission levels (`read`, `write`, `destructive`). Destructive actions require confirmation.

**Composio Integrations** — external service connections (Gmail, Slack, GitHub, Jira, Google Calendar, etc.). Agents are assigned Composio apps via `AgentAppAssignment`. Connections are per-workspace via `ComposioEntity`.

**Workspace Tools** — file I/O, code search, shell execution, and git operations proxied through the workspace worker HTTP API.

**Unified Executor** (`platform_executor.py`) dispatches all tool calls through a single entry point.

### 2.6 Knowledge Base

- **Document upload**: Files stored in S3 (`s3://automatos-ai/workspaces/{workspace_id}/documents/`)
- **Chunking and embedding**: Documents split into chunks, embedded with `qwen/qwen3-embedding-8b` (2048 dimensions) via OpenRouter
- **Vector storage**: AWS S3 Vectors
- **RAG search**: Semantic similarity search over document chunks
- **Database knowledge**: NL2SQL for connected databases — schema introspection, query templates, semantic dimensions
- **Cloud sync**: Google Drive, Dropbox integration via `cloud_sync_config` / `cloud_documents` tables
- **Supported formats**: PDF, DOCX, TXT, MD, code files, spreadsheets

### 2.7 Skills and Plugins

**Skills**: Code execution abilities attached to agents. Types: `cognitive`, `technical`, `communication`. Have versions, source files, and audit logs.

**Plugins**: Marketplace bundles of skills. Flow: marketplace catalog → workspace installation → agent assignment.

Tables:
- `marketplace_plugins` — global catalog with security scans
- `workspace_enabled_plugins` / `workspace_enabled_skills` — workspace installations
- `agent_assigned_plugins` — per-agent assignments
- `plugin_categories` — taxonomy

### 2.8 Recipes (Workflows)

Multi-step automation pipelines. Each recipe has:
- **Steps**: Ordered prompts, each optionally assigned to a specific agent
- **Triggers**: Manual, cron schedule, or event-driven (Composio triggers)
- **Variable substitution**: `{input.*}` for recipe inputs, `{steps[N].*}` for previous step outputs
- **Error handling per step**: stop, skip, or retry
- **Execution tracking**: `recipe_executions` table with status, step results, timing
- **Quality and learning**: `RecipeLearningService`, `RecipeQualityService` for continuous improvement

---

## 3. Platform Tools (Available to Auto)

### Agents
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_list_agents` | read | List all agents (filterable by status) |
| `platform_get_agent` | read | Get agent details by name or ID |
| `platform_create_agent` | write | Create agent with name, type, description, model, system_prompt, temperature, tags |
| `platform_update_agent` | write | Update agent configuration |
| `platform_delete_agent` | destructive | Permanently delete an agent (requires confirmation) |
| `platform_assign_tool_to_agent` | write | Assign Composio app to agent |
| `platform_assign_skill_to_agent` | write | Assign skill to agent |
| `platform_assign_plugin_to_agent` | write | Assign plugin to agent |
| `platform_configure_agent_heartbeat` | write | Configure periodic heartbeat schedule |

### Recipes / Workflows
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_list_recipes` | read | List all recipes (filterable by status) |
| `platform_get_recipe` | read | Get recipe details, steps, trigger config |
| `platform_create_recipe` | write | Create recipe with steps, triggers, schedule |
| `platform_update_recipe` | write | Update recipe config, schedule, execution settings |
| `platform_add_recipe_step` | write | Add step to recipe (prompt, agent, order, error handling) |
| `platform_update_recipe_step` | write | Modify existing step |
| `platform_delete_recipe_step` | write | Remove step (auto-reorders remaining) |
| `platform_execute_recipe` | write | Trigger async recipe run, returns execution_id |
| `platform_get_recipe_execution` | read | Check execution status and results |
| `platform_delete_recipe` | destructive | Delete recipe with full cleanup |

### Knowledge / Documents
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_list_datasources` | read | List documents (RAG) and databases (NL2SQL) |
| `platform_delete_document` | destructive | Delete document + chunks + vectors |
| `platform_reprocess_document` | write | Re-chunk and re-embed a document |

### Discovery
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_list_tools` | read | List all tools — platform, Composio, internal (filterable, searchable) |
| `platform_list_llms` | read | Browse OpenRouter model catalog — costs, capabilities, context lengths |
| `platform_list_connected_apps` | read | List Composio integrations with connection status |

### Analytics / Observability
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_get_llm_usage` | read | Token usage stats over time period |
| `platform_get_cost_breakdown` | read | Cost by model and agent |
| `platform_workspace_stats` | read | Dashboard — usage, top models, top agents, routing distribution |
| `platform_get_activity_feed` | read | Recent chats, recipe runs, routines |
| `platform_get_system_health` | read | DB, Redis, API, RAG pipeline, CPU/memory/disk |

### Infrastructure
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_get_logs` | read | Fetch Railway deployment logs (filterable by keyword) |
| `platform_list_services` | read | List Railway services |

### Marketplace
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_browse_marketplace_plugins` | read | Search/browse plugin catalog |
| `platform_browse_marketplace_skills` | read | Search/browse skills catalog |
| `platform_list_workspace_plugins` | read | List installed plugins |
| `platform_list_workspace_skills` | read | List installed skills |
| `platform_list_workspace_models` | read | List installed LLM models |
| `platform_install_plugin` | write | Enable plugin for workspace |
| `platform_install_skill` | write | Enable skill for workspace |
| `platform_install_model` | write | Install LLM model from OpenRouter |

### Memory
| Tool | Permission | Description |
|------|-----------|-------------|
| `platform_store_memory` | write | Store information in workspace memory |

### Workspace Files (Agent Tools)
| Tool | Permission | Description |
|------|-----------|-------------|
| `workspace_read_file` | read | Read file from workspace repo |
| `workspace_write_file` | write | Write/create file in workspace |
| `workspace_list_dir` | read | List directory contents |
| `workspace_grep` | read | Search code with regex patterns |
| `workspace_exec` | write | Execute shell commands |
| `workspace_git` | write | Git operations on workspace repos |

---

## 4. Agent Design Principles

### Configuration
Each agent's `model_config` JSON contains:
- `provider`: LLM provider (openai, anthropic, google, etc.)
- `model_id`: Specific model identifier
- `temperature`: 0.0 (deterministic) to 1.0 (creative)
- `max_tokens`: Response length cap
- `top_p`, `frequency_penalty`, `presence_penalty`: Sampling controls
- `fallback_model_id`: Backup model if primary fails

### Model Selection Guidance
| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Complex reasoning, planning | GPT-5.4, Claude Opus | Deepest reasoning |
| Coding, technical tasks | Claude Sonnet 4 | Best code model |
| Lightweight agents, high-volume | Claude Haiku 4.5 | Fast, cost-effective |
| General purpose | GPT-4o | Good balance |
| Budget-conscious | GPT-4o-mini | Cheapest capable model |

Models are sourced from OpenRouter — full catalog browsable via `platform_list_llms`, filterable by capability (tools, vision, reasoning, json_mode) and tier (free, budget, mid, premium).

### Capability Layering
1. **Base model** — raw LLM capability via model_config
2. **System prompt** — persona, constraints, behavioral instructions
3. **Skills** — code execution abilities (cognitive, technical, communication)
4. **Plugins** — marketplace bundles of skills
5. **Composio tools** — external service integrations (Gmail, Slack, GitHub, etc.)
6. **Memory** — persistent context across conversations

---

## 5. Key Database Tables

### Core
| Table | Purpose |
|-------|---------|
| `agents` | Agent definitions, config, model settings, marketplace fields, semantic embeddings |
| `skills` | Skill definitions (type: cognitive/technical/communication, implementation, parameters) |
| `skill_files` | Source files for skills |
| `skill_versions` | Version history |
| `skill_audit_log` | Skill change audit trail |
| `workflows` | Workflow definitions |
| `workflow_executions` | Execution tracking |
| `workflow_recipes` | Recipe definitions with steps, triggers, schedules |
| `recipe_executions` | Recipe run tracking with step results |
| `documents` | Uploaded document metadata (title, status, chunk_count, file_path) |
| `chats` | Chat session records |
| `messages` | Individual messages within chats |
| `artifacts` | Generated artifacts (code, documents) |
| `votes` | Message feedback |
| `tasks` | Background task tracking |
| `users` | User records |
| `memory_items` | Structured memory storage |
| `patterns` | Learned behavioral patterns |

### Workspace / Multi-Tenancy
| Table | Purpose |
|-------|---------|
| `workspaces` | Workspace definitions |
| `workspace_shares` | Cross-workspace sharing |
| `workspace_tool_config` | Per-workspace tool configuration |

### Marketplace
| Table | Purpose |
|-------|---------|
| `marketplace_plugins` | Plugin catalog with security scans |
| `marketplace_widgets` | Widget catalog |
| `plugin_categories` | Plugin taxonomy |
| `plugin_security_scans` | Security scan results |
| `plugin_sync_history` | Sync audit trail |
| `workspace_enabled_plugins` | Workspace-level plugin installations |
| `workspace_enabled_skills` | Workspace-level skill installations |
| `agent_assigned_plugins` | Per-agent plugin assignments |
| `widget_installations` | Widget installations per workspace |
| `widget_reviews` | Widget user reviews |

### Integrations
| Table | Purpose |
|-------|---------|
| `composio_entities` | Per-workspace Composio entity mappings |
| `composio_connections` | Active integration connections |
| `composio_apps_cache` | Cached Composio app catalog |
| `composio_actions_cache` | Cached Composio action definitions |
| `agent_app_assignments` | Agent-to-Composio-app mappings |
| `agent_app_features` | Feature flags per agent-app pair |
| `trigger_subscriptions` | Event-driven trigger configs |
| `tool_execution_logs` | Execution audit trail |
| `tool_execution_cache` | Cached execution results |
| `intent_classification_cache` | Cached intent classifications |
| `channel_connections` | Channel integration connections |

### Routing
| Table | Purpose |
|-------|---------|
| `routing_decisions` | Historical routing decisions for learning |
| `routing_rules` | Workspace-configured routing patterns |
| `unrouted_events` | Requests that failed to route (for improvement) |

### Knowledge
| Table | Purpose |
|-------|---------|
| `cloud_sync_config` | Cloud provider sync settings |
| `cloud_documents` | Synced cloud document tracking |
| `cloud_sync_jobs` | Sync job history |
| `database_knowledge_sources` | Connected database schemas |
| `database_relationships` | Foreign key / relationship mappings |
| `database_query_audit` | Query execution audit |
| `database_query_templates` | Reusable query templates |
| `semantic_metrics` | Semantic analysis metrics |
| `semantic_dimensions` | Dimensional model for analytics |
| `nl2sql_training_examples` | NL2SQL training data |
| `nl2sql_benchmark_runs` / `nl2sql_benchmark_results` | Accuracy benchmarks |
| `external_knowledge` | External knowledge source references |

### System
| Table | Purpose |
|-------|---------|
| `system_settings` | Platform-wide configuration (category/key/value) |
| `system_prompts` | Managed system prompt templates |
| `system_prompt_versions` | Prompt version history |
| `system_prompt_eval_runs` | Prompt evaluation results |
| `llm_models` | LLM model registry |
| `llm_usage` | Token usage and cost tracking |
| `workspace_models` | Per-workspace model availability |
| `user_api_keys` | User API key management |
| `sdk_api_keys` | SDK API key management |
| `openrouter_models_cache` | Cached OpenRouter model catalog |
| `openrouter_sync_jobs` | OpenRouter sync history |
| `personas` | Reusable agent personas |
| `context_policies` | Context window management rules |
| `credential_types` / `credentials` / `credential_audit_logs` | Credential management |
| `tools` / `tool_categories` / `tool_credentials` / `tool_configurations` | Tool registry |
| `agent_tool_permissions` / `permission_audit_logs` | Tool permission management |
| `tool_reviews` / `tool_installation_requests` | Tool marketplace |
| `code_symbols` / `code_edges` / `playbooks` | Code graph analysis |
| `document_templates` | Template library |
| `evaluation_results` / `benchmark_assessments` / `component_metrics` / `integration_analyses` | Platform evaluation |

---

## 6. Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI (Python), async, SQLAlchemy ORM |
| **Frontend** | Next.js (React), TypeScript, Tailwind CSS, shadcn/ui |
| **Database** | PostgreSQL |
| **Object Storage** | AWS S3 (documents) |
| **Vector Storage** | AWS S3 Vectors (embeddings) |
| **Cache** | Redis (routing cache, memory cache, session state) |
| **Embeddings** | qwen/qwen3-embedding-8b via OpenRouter (2048 dimensions) |
| **LLM Providers** | OpenRouter (multi-provider gateway — OpenAI, Anthropic, Google, etc.) |
| **External Integrations** | Composio (50+ apps — Gmail, Slack, GitHub, Jira, Calendar, etc.) |
| **Memory** | Mem0 (two-tier: global + agent-scoped) |
| **Auth** | Clerk (authentication + user management) |
| **Deployment** | Railway (services, workers, cron jobs) |
| **Monitoring** | Railway logs, built-in analytics engine |

### Key Configuration
All configuration flows through `config.py` reading from `.env`. No `os.getenv()` outside of `config.py`. No hardcoded model names, bucket names, or URLs in application code.

### Request Flow
```
User message
  → api/chat.py (auth, workspace resolution)
    → Auto brain (complexity assessment: ATOM→ORGANISM)
      → ATOM: respond directly
      → MOLECULE+: tool discovery + execution via platform_executor
      → CELL+: memory retrieval + tool execution
      → ORGAN+: Universal Router → specialist agent delegation
      → ORGANISM: multi-agent pipeline orchestration
    → Response streamed back to frontend
```
