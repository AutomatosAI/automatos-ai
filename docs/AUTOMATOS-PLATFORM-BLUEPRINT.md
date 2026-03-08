# Automatos AI Platform — Blueprint

## What Is Automatos?

Automatos is an AI agent orchestration platform. Users create, configure, and deploy AI agents that can access tools, search knowledge bases, integrate with external apps, and execute multi-step workflows. The platform is multi-tenant — each workspace is isolated with its own agents, documents, integrations, and memory.

## Architecture Overview

### Auto (The Brain)
Auto is the orchestrator — the default entry point for all user messages. It:
- Assesses message complexity (ATOM → ORGANISM scale)
- Routes to specialist agents when appropriate
- Handles platform management directly (creating agents, installing skills, workspace queries)
- Uses the system LLM (configurable per workspace)

### Progressive Complexity (ATOM → ORGANISM)
- **ATOM**: Greetings, chitchat — no tools needed
- **MOLECULE**: Single tool call (search docs, list agents)
- **CELL**: Tools + memory + reasoning
- **ORGAN**: Multi-agent coordination
- **ORGANISM**: Enterprise pipelines

### Universal Router
When Auto delegates, the Universal Router picks the best specialist agent using:
1. Tier 1: Keyword matching (instant)
2. Tier 2: Intent classification (fast)
3. Tier 3: LLM-based routing with agent descriptions (accurate)

### Agent System
Each agent has:
- **name**: Display name
- **agent_type**: chatbot | worker | researcher | coder
- **description**: What the agent does (used by router)
- **model_id**: LLM model (e.g. gpt-5.4, claude-sonnet-4-20250514, claude-haiku-4-5-20251001)
- **system_prompt**: Custom persona and instructions
- **temperature**: 0.0-2.0 (creativity vs determinism)
- **tags**: Categorization labels
- **skills**: Assigned skill capabilities
- **plugins**: Assigned external integrations
- **status**: active | inactive | archived

**Agent types:**
- **Chatbot**: Conversational, user-facing interactions
- **Worker**: Background task execution
- **Researcher**: Analysis, data gathering, report generation
- **Coder**: Code review, development assistance

### Memory System
Hierarchical, multi-tier memory:
- **Conversation**: Short-term, per-chat context
- **Agent**: Agent-specific learned knowledge
- **Workspace**: Shared across all agents in a workspace
- **Global**: Platform-wide knowledge (brand, configuration)

Memory is automatically extracted from conversations and stored as facts. The system retrieves relevant memories based on query similarity.

### Tool System
Tools are provided to agents via the Unified Executor:
- **Platform tools** (platform_*): Direct database operations for workspace management
- **Knowledge tools**: search_knowledge, semantic_search, search_codebase
- **Database tools**: query_database, smart_query_database (NL2SQL)
- **File tools**: workspace_read_file, workspace_write_file
- **External tools**: Composio integrations (email, Slack, GitHub, Calendar)
- **Document tools**: generate_document (PDF, DOCX, XLSX)

### Knowledge Base
- Documents uploaded via UI or API
- Chunked and embedded (qwen3-embedding-8b, 2048 dimensions)
- Stored in S3 standard (files) + S3 Vectors (embeddings)
- Searchable via RAG (search_knowledge, semantic_search)
- Supports: PDF, DOCX, TXT, MD, code files, spreadsheets

### Skills & Plugins
- **Skills**: Code-execution capabilities (e.g. create_implementation_plan)
- **Plugins**: External app integrations via Composio
- **Marketplace**: Browse and install available skills/plugins/models
- **Assignment**: Skills and plugins can be assigned to specific agents

### Recipes / Workflows
Multi-step automation pipelines:
- Each recipe has ordered steps
- Steps can call tools, agents, or external actions
- Execute via platform_execute_recipe
- Track status via platform_get_recipe_execution

## Platform Tools Reference

### Agent Management
| Tool | Action |
|------|--------|
| platform_list_agents | List all agents in workspace |
| platform_get_agent | Get agent details by ID or name |
| platform_create_agent | Create new agent (name, type, model, prompt) |
| platform_update_agent | Update agent config (name, status, model, prompt) |
| platform_delete_agent | Delete an agent (requires confirmation) |

### Marketplace & Assignment
| Tool | Action |
|------|--------|
| platform_browse_marketplace_plugins | Search available plugins |
| platform_browse_marketplace_skills | Search available skills |
| platform_list_workspace_plugins | List installed plugins |
| platform_list_workspace_skills | List installed skills |
| platform_list_workspace_models | List installed models |
| platform_install_plugin | Install plugin to workspace |
| platform_install_skill | Install skill to workspace |
| platform_install_model | Install model to workspace |
| platform_assign_tool_to_agent | Assign a tool to an agent |
| platform_assign_skill_to_agent | Assign a skill to an agent |
| platform_assign_plugin_to_agent | Assign a plugin to an agent |

### Recipes & Workflows
| Tool | Action |
|------|--------|
| platform_list_recipes | List all recipes |
| platform_get_recipe | Get recipe details |
| platform_create_recipe | Create new recipe |
| platform_update_recipe | Update recipe config |
| platform_add_recipe_step | Add step to recipe |
| platform_execute_recipe | Run a recipe |
| platform_get_recipe_execution | Check execution status |
| platform_delete_recipe | Delete recipe (requires confirmation) |

### Knowledge & Documents
| Tool | Action |
|------|--------|
| platform_list_documents | List uploaded documents |
| platform_delete_document | Remove a document |
| platform_reprocess_document | Re-embed a document |
| search_knowledge | RAG search across knowledge base |
| semantic_search | Vector similarity search |
| search_codebase | Search indexed code repositories |

### Workspace & Observability
| Tool | Action |
|------|--------|
| platform_get_workspace_info | Workspace details and config |
| platform_workspace_stats | Usage statistics |
| platform_get_system_health | Platform health check |
| platform_get_activity_feed | Recent activity log |
| platform_get_llm_usage | Token usage and costs |
| platform_get_cost_breakdown | Cost analysis by model/agent |
| platform_get_memory_stats | Memory system statistics |
| platform_list_connected_apps | Active Composio integrations |
| platform_list_tools | Available tools |
| platform_list_llms | Available LLM models |
| platform_list_datasources | Indexed data sources |

### Memory
| Tool | Action |
|------|--------|
| platform_store_memory | Save a fact to memory |
| platform_get_memory_stats | Memory system statistics |

## Tech Stack
- **Backend**: Python, FastAPI, SQLAlchemy
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui
- **Database**: PostgreSQL (Supabase)
- **Storage**: AWS S3 (documents), S3 Vectors (embeddings)
- **Cache**: Redis (routing cache, assessment cache)
- **Deployment**: Railway (auto-deploy from GitHub)
- **LLM Providers**: OpenRouter (multi-model), OpenAI, Anthropic
- **Integrations**: Composio (50+ external apps)
- **Embeddings**: qwen/qwen3-embedding-8b via OpenRouter (2048 dimensions)

## Model Selection Guide
| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Complex reasoning, planning | GPT-5.4, Claude Opus | Deepest reasoning |
| Coding, technical tasks | Claude Sonnet 4.6 | Best code model |
| Lightweight agents, routing | Claude Haiku 4.5 | Fast, cost-effective |
| General purpose | GPT-4o | Good balance |
| Budget-conscious | GPT-4o-mini | Cheapest capable model |

## Database Schema (Key Tables)
- **agents**: id, workspace_id, name, agent_type, description, status, model_config, custom_persona_prompt, configuration, tags
- **recipes**: id, workspace_id, name, description, status, trigger_type, steps (via recipe_steps)
- **recipe_steps**: id, recipe_id, step_order, step_type, configuration
- **documents**: id, workspace_id, title, status, chunk_count, file_path
- **document_chunks**: id, document_id, content, chunk_index, embedding
- **memories**: id, workspace_id, agent_id, memory (text), metadata
- **workspace_members**: workspace_id, user_id, role
- **system_settings**: category, key, value (LLM config, feature flags)
