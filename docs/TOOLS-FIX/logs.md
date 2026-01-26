nowledge' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:19:34,054 - modules.tools.execution.unified_executor - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - [tool-trace c5b6a52ac3e6] Parameters keys=['query', 'limit']
2026-01-26 11:19:34,055 - modules.tools.execution.unified_executor - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:19:34,055 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:19:34,056 - modules.tools.registry.tool_registry - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:19:34,056 - modules.tools.execution.unified_executor - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   🔧 Initializing research tools (RAG, CodeGraph)...
2026-01-26 11:19:35,162 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - RAGConfig loaded: max_tokens=4500, diversity=0.3, min_similarity=0.5
2026-01-26 11:19:36,682 - modules.codegraph.codegraph_service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - CodeGraphService using huggingface_local embeddings
2026-01-26 11:19:36,860 - modules.codegraph.codegraph_service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ CodeGraphService using EnhancedVectorStore for semantic search
2026-01-26 11:19:36,860 - modules.agents.services.agent_platform_tools - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 🔧 Agent 19 calling tool: search_knowledge
2026-01-26 11:19:36,860 - modules.agents.services.agent_platform_tools - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Parameters: {'query': 'workflows and how they work in Automatos', 'limit': 5}
2026-01-26 11:19:36,860 - modules.agents.services.agent_platform_tools - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   🔍 Searching knowledge base: 'workflows and how they work in Automatos' (limit: 5)
2026-01-26 11:19:36,861 - modules.search.optimization.context_optimizer - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Context Optimizer using huggingface_local embeddings (model: BAAI/bge-large-en-v1.5, dimension: 1024)
2026-01-26 11:19:36,861 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Using modules.search.ContextOptimizer (Knapsack, MMR, Entropy)
2026-01-26 11:19:36,861 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Using huggingface_local embeddings
2026-01-26 11:19:37,038 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Using modules.search.EnhancedVectorStore for vector search
2026-01-26 11:19:37,038 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Using modules.rag.SemanticChunker (5 strategies)
2026-01-26 11:19:37,038 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Using QueryEnhancer (HyDE, decomposition, concept extraction)
2026-01-26 11:19:37,754 - modules.rag.query_enhancer - WARNING - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - LLM not available for query enhancement: LLM provider not configured for service 'rag'. Please set rag.provider in system settings.
2026-01-26 11:19:37,754 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Enhanced query into 1 variations
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.35it/s]
2026-01-26 11:19:38,413 - modules.rag.service - WARNING - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - EnhancedVectorStore search failed, falling back to SQL: column "timestamp" does not exist
2026-01-26 11:19:38,413 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Using fallback SQL-based vector search
2026-01-26 11:19:38,685 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 🔎 Executing SQL vector similarity search: min_similarity=0.5, limit=15
2026-01-26 11:19:39,115 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 📊 Database returned 15 results
2026-01-26 11:19:39,116 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 📁 Candidate sources: {'17-DYNAMIC-TOOL-ASSIGNMENT.md': 1, '10-WORKFLOW-ORCHESTRATION-ENGINE.md': 3, 'DEVELOPER_GUIDE.md': 1, '12-PLAYBOOKS-PATTERN.md': 1, '22-Anthropic-Skills-Integration.md': 1, 'WEAVIATE_COMPARISON.md': 1, '30-MODULAR-ARCHITECTURE-REFACTOR.md': 3, 'EBOOK_SPECIFICATIONS.md': 1, '16-LLM-DRIVEN-ORCHESTRATOR.md': 1, 'EBOOK_CONTEXT_ENGINEERING.md': 1, 'DOCUMENTATION_RESTRUCTURE_COMPLETE.md': 1}
2026-01-26 11:19:39,116 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 📈 Similarity range: 0.722 - 0.786
2026-01-26 11:19:39,116 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Retrieved 15 candidates with SQL fallback
2026-01-26 11:19:39,163 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 🔍 Starting optimization: 15 candidates, max_chunks=5, max_tokens=2000
2026-01-26 11:19:39,163 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Candidate 1: base=0.786, quality=0.80, source_penalty=1.00, final=0.629, tokens=55, source=17-DYNAMIC-TOOL-ASSIGNMENT.md, preview='--- ## Executive Summary Transform Automatos AI from a research-focused platform to a **truly task-a...'
2026-01-26 11:19:39,163 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Candidate 2: base=0.785, quality=1.00, source_penalty=1.00, final=0.785, tokens=147, source=10-WORKFLOW-ORCHESTRATION-ENGINE.md, preview='- Automated workflow generation - Predictive optimization PRD-10 completes the Automatos AI platform...'
2026-01-26 11:19:39,164 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Candidate 3: base=0.760, quality=1.00, source_penalty=0.70, final=0.532, tokens=116, source=10-WORKFLOW-ORCHESTRATION-ENGINE.md, preview='failure recovery - Cost optimization AI - A/B testing for strategies - Workflow versioning ### Phase...'
2026-01-26 11:19:39,165 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 📊 Value range: 0.308 - 0.785, Weight range: 22 - 221 tokens
2026-01-26 11:19:39,165 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 🎯 Running 0/1 Knapsack DP algorithm with quality-adjusted scores...
2026-01-26 11:19:39,165 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 🎒 Knapsack DP: n=15 items, capacity=2000 tokens, max_items=5
2026-01-26 11:19:39,179 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Knapsack result: 5 items, total_value=3.143, total_weight=495 tokens
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - ✅ Knapsack selected 5 items: [0, 1, 2, 3, 4]
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Selected 1: idx=0, score=0.629, tokens=55, source=17-DYNAMIC-TOOL-ASSIGNMENT.md, preview='--- ## Executive Summary Transform Automatos AI from a research-focused platform...'
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Selected 2: idx=1, score=0.785, tokens=147, source=10-WORKFLOW-ORCHESTRATION-ENGINE.md, preview='- Automated workflow generation - Predictive optimization PRD-10 completes the A...'
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Selected 3: idx=2, score=0.532, tokens=116, source=10-WORKFLOW-ORCHESTRATION-ENGINE.md, preview='failure recovery - Cost optimization AI - A/B testing for strategies - Workflow ...'
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Selected 4: idx=3, score=0.449, tokens=29, source=DEVELOPER_GUIDE.md, preview='Automatos AI is built on **Modular Domain-Driven Design**. - **Everything is a M...'
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   Selected 5: idx=4, score=0.748, tokens=148, source=12-PLAYBOOKS-PATTERN.md, preview='based on proven patterns. This is how Automatos learns from your team's success....'
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 📈 Results: 5 chunks, 495 tokens, source_diversity=0.80
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 📁 Source distribution: {'17-DYNAMIC-TOOL-ASSIGNMENT.md': 1, '10-WORKFLOW-ORCHESTRATION-ENGINE.md': 2, 'DEVELOPER_GUIDE.md': 1, '12-PLAYBOOKS-PATTERN.md': 1}
2026-01-26 11:19:39,180 - modules.rag.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - 💡 Information gain: 0.210
2026-01-26 11:19:39,180 - modules.agents.services.agent_platform_tools - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   📊 RAG returned 5 results
2026-01-26 11:19:39,180 - modules.agents.services.agent_platform_tools - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   ✅ Returning 5 formatted results
2026-01-26 11:19:39,180 - modules.agents.services.agent_platform_tools - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   📄 Sample: --- ## Executive Summary Transform Automatos AI from a research-focused platform to a **truly task-a...
2026-01-26 11:19:39,180 - modules.tools.execution.unified_executor - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] -   ✅ Tool 'search_knowledge' executed successfully
2026-01-26 11:19:39,180 - consumers.chatbot.tool_router - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - [tool-trace c5b6a52ac3e6] execute_tool done tool=search_knowledge success=True
2026-01-26 11:19:39,186 - consumers.chatbot.tool_router - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - [tool-trace c5b6a52ac3e6] search_knowledge succeeded
2026-01-26 11:19:39,187 - consumers.chatbot.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Received final response from tool loop: True
2026-01-26 11:19:39,187 - consumers.chatbot.service - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - Using final response from tool loop (162 chars)
2026-01-26 11:19:39,948 - modules.memory.operations.injection - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - [Memory] Storing: show me documents for workflows and how they work ...
2026-01-26 11:19:39,948 - modules.memory.storage.knowledge_system - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - [Memory] store_experience called: type=conversation, importance=1.00
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.18it/s]
2026-01-26 11:19:40,422 - modules.memory.storage.knowledge_system - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored 3c08587b-277d-485b-9560-2c1337e26c28 (type=conversation, level=long_term, importance=1.00)
2026-01-26 11:19:40,488 - modules.memory.operations.injection - INFO - [req=1ed53c059e82 run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored (id=3c08587b-277d-485b-9560-2c1337e26c28): show me documents for workflows and how they work ...
2026-01-26 11:20:05,745 - core.auth.clerk - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Auto-assigned admin role to gerard@automatos.app based on domain
2026-01-26 11:20:05,946 - api.chat - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [chat] RequestContext workspace_id=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:20:05,947 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - StreamingChatService initialized with AgentFactory integration
2026-01-26 11:20:06,740 - api.chat - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Chat request - agentId: 19, model: gpt-4
2026-01-26 11:20:06,740 - api.chat - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Using agent-based streaming with agent_id=19
2026-01-26 11:20:07,200 - modules.agents.factory.agent_factory - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 📋 Using model from settings: gpt-4-turbo-preview (context: 128000)
2026-01-26 11:20:07,201 - modules.agents.factory.agent_factory - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Agent 19 using LLM: openai/gpt-4-turbo-preview (from system settings)
2026-01-26 11:20:07,306 - modules.agents.factory.agent_factory - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - ✅ Loaded 3 Composio app assignment(s) for agent 19
2026-01-26 11:20:07,307 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:20:07,502 - modules.agents.factory.agent_factory - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - ✅ Activated agent 19 (Context Engineer API) with gpt-4-turbo-preview
2026-01-26 11:20:07,503 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Trying ContextRetrievalEngine...
2026-01-26 11:20:07,503 - modules.search.retrieval.context_retrieval_engine - WARNING - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - No embedding provided for similarity-based retrieval
2026-01-26 11:20:07,504 - modules.search.retrieval.context_retrieval_engine - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Retrieved 0 contexts in 0.2ms
2026-01-26 11:20:07,504 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] ContextRetrievalEngine returned empty, trying basic memory...
2026-01-26 11:20:07,504 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Using HierarchicalMemorySystem...
2026-01-26 11:20:07,504 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Starting parallel retrieval: Semantic + Recent
Batches:   0%|                                                                                                                                           | 0/1 [00:00<?, ?it/s]INFO:     127.0.0.1:61594 - "POST /api/chat HTTP/1.1" 200 OK
2026-01-26 11:20:07,710 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Activating agent 19 for chat f8271d3e-40fe-4c37-a3eb-02e3cfcca90b
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.85it/s]
2026-01-26 11:20:08,841 - consumers.chatbot.tool_router - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [tool-trace 161415224a90] Loaded 16 tools (agent_id=19, denied=0, candidates=16, 1129ms)
2026-01-26 11:20:09,105 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Semantic search found 0 memories
2026-01-26 11:20:09,106 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Recent memories: 10
2026-01-26 11:20:09,106 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Total unique memories: 10
2026-01-26 11:20:09,106 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Injecting 2303 chars of shared memory
2026-01-26 11:20:09,107 - consumers.chatbot.prompt_analyzer - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🔍 Query: 'How may apps and tools does automatos have registered in the database.' | Top ranked tools: ['composio_execute (2)', 'query_database (2)', 'search_knowledge (1)', 'smart_query_database (1)']
2026-01-26 11:20:09,107 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🔍 Narrowing tools from 16 to 4 based on ranked candidates
2026-01-26 11:20:09,107 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Generating response with agent Context Engineer API
2026-01-26 11:20:09,107 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🔍 Agent tools - count: 4, is_simple: False
2026-01-26 11:20:09,107 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🔍 Available tools: ['search_knowledge', 'query_database', 'smart_query_database', 'composio_execute']
2026-01-26 11:20:09,122 - core.llm.clients.openai_client - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Initialized OpenAI client with model: gpt-4-turbo-preview
2026-01-26 11:20:09,123 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 4 tools. Sample: [{"type": "function", "function": {"name": "search_knowledge", "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform", "parameters": {"type"...
2026-01-26 11:20:10,856 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:20:10,860 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🔍 Agent LLM Response - has_tool_calls: True, content_length: 0, finish_reason: tool_calls
2026-01-26 11:20:10,860 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - ✅ Agent LLM requested 1 tool calls
2026-01-26 11:20:10,860 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Agent requested 1 tool calls
2026-01-26 11:20:10,860 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Tool iteration 1: 1 tool calls
2026-01-26 11:20:10,861 - consumers.chatbot.tool_router - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [tool-trace b3d13b17b32e] ToolRouter execute_and_format tool=smart_query_database agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['query']
2026-01-26 11:20:10,861 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:20:10,861 - consumers.chatbot.tool_router - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [tool-trace b3d13b17b32e] execute_tool start tool=smart_query_database agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['query']
2026-01-26 11:20:10,861 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [tool-trace b3d13b17b32e] Executing tool 'smart_query_database' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:20:10,861 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [tool-trace b3d13b17b32e] Parameters keys=['query']
2026-01-26 11:20:10,861 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:20:10,861 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:20:10,861 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:20:10,861 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:20:10,861 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:20:10,861 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:20:10,861 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:20:10,861 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:20:10,862 - modules.tools.registry.tool_registry - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:20:10,862 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 🧠 Smart DB Query: How many apps and tools does Automatos have regist...
2026-01-26 11:20:10,862 - modules.nl2sql.schema.provider - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - ✅ Schema Provider initialized
2026-01-26 11:21:38,480 - core.llm.manager - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Found credential 'development_openai' (type: openai_api) for provider 'openai'
2026-01-26 11:21:38,646 - core.llm.manager - WARNING - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Could not find credential for provider 'openai' using any variation. Tried: ['development_openai_api', 'development_openai', 'openai_api', 'openai', 'openai', 'Openai', 'Openai']
2026-01-26 11:21:38,655 - core.llm.clients.openai_client - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Initialized OpenAI client with model: gpt-4
2026-01-26 11:21:45,281 - httpx - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:21:52,967 - httpx - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:21:52,970 - modules.nl2sql.query.nl2sql_service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Generated SQL for question: What is the total count of unique apps and tools associated with Automatos that are registered in th...
2026-01-26 11:21:53,179 - modules.nl2sql.intelligence.agent - ERROR - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - SQL execution error: (psycopg2.errors.UndefinedColumn) column "provider" does not exist
LINE 1: ...DISTINCT app_name) FROM composio_apps_cache WHERE provider =...
                                                             ^

[SQL: SELECT (SELECT COUNT(DISTINCT app_name) FROM composio_apps_cache WHERE provider = 'Automatos' AND created_at >= (CURRENT_DATE - INTERVAL '30 days')) AS unique_apps_count, (SELECT COUNT(DISTINCT name) FROM mcp_tools_backup WHERE provider = 'Automatos' AND created_at >= (CURRENT_DATE - INTERVAL '30 days')) AS unique_tools_count LIMIT 1000]
(Background on this error at: https://sqlalche.me/e/20/f405)
2026-01-26 11:21:53,180 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - 📊 Smart query returning 0 rows
2026-01-26 11:21:53,180 - modules.tools.execution.unified_executor - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] -   ✅ Tool 'smart_query_database' executed successfully
2026-01-26 11:21:53,181 - consumers.chatbot.tool_router - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [tool-trace b3d13b17b32e] execute_tool done tool=smart_query_database success=True
2026-01-26 11:21:53,181 - consumers.chatbot.tool_router - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [tool-trace b3d13b17b32e] smart_query_database succeeded
2026-01-26 11:21:55,928 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:21:55,932 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Received final response from tool loop: True
2026-01-26 11:21:55,932 - consumers.chatbot.service - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - Using final response from tool loop (160 chars)
2026-01-26 11:21:56,683 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] Storing: How may apps and tools does automatos have registe...
2026-01-26 11:21:56,684 - modules.memory.storage.knowledge_system - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] store_experience called: type=conversation, importance=1.00
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.41it/s]
2026-01-26 11:21:57,707 - modules.memory.storage.knowledge_system - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored 245f6758-9410-4285-90d3-6eff9e2da78c (type=conversation, level=long_term, importance=1.00)
2026-01-26 11:21:57,758 - modules.memory.operations.injection - INFO - [req=fa7cd07b32c5 run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored (id=245f6758-9410-4285-90d3-6eff9e2da78c): How may apps and tools does automatos have registe...
2026-01-26 11:22:19,609 - core.auth.clerk - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Auto-assigned admin role to gerard@automatos.app based on domain
2026-01-26 11:22:19,813 - api.chat - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [chat] RequestContext workspace_id=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:22:19,814 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - StreamingChatService initialized with AgentFactory integration
2026-01-26 11:22:20,637 - api.chat - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Chat request - agentId: 19, model: gpt-4
2026-01-26 11:22:20,637 - api.chat - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Using agent-based streaming with agent_id=19
2026-01-26 11:22:21,108 - modules.agents.factory.agent_factory - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 📋 Using model from settings: gpt-4-turbo-preview (context: 128000)
2026-01-26 11:22:21,109 - modules.agents.factory.agent_factory - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Agent 19 using LLM: openai/gpt-4-turbo-preview (from system settings)
2026-01-26 11:22:21,202 - modules.agents.factory.agent_factory - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - ✅ Loaded 3 Composio app assignment(s) for agent 19
2026-01-26 11:22:21,202 - modules.tools.execution.unified_executor - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:22:21,412 - modules.agents.factory.agent_factory - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - ✅ Activated agent 19 (Context Engineer API) with gpt-4-turbo-preview
2026-01-26 11:22:21,412 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Trying ContextRetrievalEngine...
2026-01-26 11:22:21,413 - modules.search.retrieval.context_retrieval_engine - WARNING - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - No embedding provided for similarity-based retrieval
2026-01-26 11:22:21,413 - modules.search.retrieval.context_retrieval_engine - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Retrieved 0 contexts in 0.3ms
2026-01-26 11:22:21,413 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] ContextRetrievalEngine returned empty, trying basic memory...
2026-01-26 11:22:21,413 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Using HierarchicalMemorySystem...
2026-01-26 11:22:21,413 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Starting parallel retrieval: Semantic + Recent
Batches:   0%|                                                                                                                                           | 0/1 [00:00<?, ?it/s]INFO:     127.0.0.1:61646 - "POST /api/chat HTTP/1.1" 200 OK
2026-01-26 11:22:21,595 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Activating agent 19 for chat f8271d3e-40fe-4c37-a3eb-02e3cfcca90b
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.44it/s]
2026-01-26 11:22:22,519 - consumers.chatbot.tool_router - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [tool-trace 9fb02d85b121] Loaded 16 tools (agent_id=19, denied=0, candidates=16, 923ms)
2026-01-26 11:22:22,780 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Semantic search found 0 memories
2026-01-26 11:22:22,780 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Recent memories: 10
2026-01-26 11:22:22,780 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Total unique memories: 10
2026-01-26 11:22:22,780 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Injecting 2330 chars of shared memory
2026-01-26 11:22:22,780 - consumers.chatbot.prompt_analyzer - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔍 Query: 'Show me code for the AgentFactory and how it works' | Top ranked tools: ['search_codebase (1)', 'search_multimodal (1)']
2026-01-26 11:22:22,781 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔍 Narrowing tools from 16 to 2 based on ranked candidates
2026-01-26 11:22:22,781 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Generating response with agent Context Engineer API
2026-01-26 11:22:22,781 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔍 Agent tools - count: 2, is_simple: False
2026-01-26 11:22:22,781 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔍 Available tools: ['search_codebase', 'search_multimodal']
2026-01-26 11:22:22,794 - core.llm.clients.openai_client - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Initialized OpenAI client with model: gpt-4-turbo-preview
2026-01-26 11:22:22,794 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 2 tools. Sample: [{"type": "function", "function": {"name": "search_codebase", "description": "Search indexed codebases for functions, classes, and implementations", "parameters": {"type": "object", "properties": {"qu...
2026-01-26 11:22:24,342 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:22:24,346 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔍 Agent LLM Response - has_tool_calls: True, content_length: 0, finish_reason: tool_calls
2026-01-26 11:22:24,346 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - ✅ Agent LLM requested 1 tool calls
2026-01-26 11:22:24,346 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Agent requested 1 tool calls
2026-01-26 11:22:24,346 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Tool iteration 1: 1 tool calls
2026-01-26 11:22:24,346 - consumers.chatbot.tool_router - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [tool-trace 5d44a77491f3] ToolRouter execute_and_format tool=search_codebase agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['query', 'file_type']
2026-01-26 11:22:24,346 - modules.tools.execution.unified_executor - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:22:24,346 - consumers.chatbot.tool_router - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [tool-trace 5d44a77491f3] execute_tool start tool=search_codebase agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['query', 'file_type']
2026-01-26 11:22:24,346 - modules.tools.execution.unified_executor - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [tool-trace 5d44a77491f3] Executing tool 'search_codebase' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:22:24,347 - modules.tools.execution.unified_executor - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [tool-trace 5d44a77491f3] Parameters keys=['query', 'file_type']
2026-01-26 11:22:24,347 - modules.tools.execution.unified_executor - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:22:24,355 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:22:24,355 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:22:24,355 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:22:24,355 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:22:24,355 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:22:24,355 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:22:24,356 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:22:24,357 - modules.tools.registry.tool_registry - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:22:24,357 - modules.tools.execution.unified_executor - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] -   🔧 Initializing research tools (RAG, CodeGraph)...
2026-01-26 11:22:25,429 - modules.rag.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - RAGConfig loaded: max_tokens=4500, diversity=0.3, min_similarity=0.5
2026-01-26 11:22:26,840 - modules.codegraph.codegraph_service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - CodeGraphService using huggingface_local embeddings
2026-01-26 11:22:27,010 - modules.codegraph.codegraph_service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - ✅ CodeGraphService using EnhancedVectorStore for semantic search
2026-01-26 11:22:27,011 - modules.agents.services.agent_platform_tools - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - 🔧 Agent 19 calling tool: search_codebase
2026-01-26 11:22:27,011 - modules.agents.services.agent_platform_tools - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] -   Parameters: {'query': 'AgentFactory', 'file_type': 'py'}
2026-01-26 11:22:27,150 - modules.agents.services.agent_platform_tools - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] -   🔍 Searching codebase: 'AgentFactory' in project 'Automatos-ai'
2026-01-26 11:22:27,399 - modules.agents.services.agent_platform_tools - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] -   ✅ Found 4 code results
2026-01-26 11:22:27,399 - modules.tools.execution.unified_executor - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] -   ✅ Tool 'search_codebase' executed successfully
2026-01-26 11:22:27,399 - consumers.chatbot.tool_router - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [tool-trace 5d44a77491f3] execute_tool done tool=search_codebase success=True
2026-01-26 11:22:27,400 - consumers.chatbot.tool_router - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [tool-trace 5d44a77491f3] search_codebase succeeded
2026-01-26 11:22:27,401 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 2 tools. Sample: [{"type": "function", "function": {"name": "search_codebase", "description": "Search indexed codebases for functions, classes, and implementations", "parameters": {"type": "object", "properties": {"qu...
2026-01-26 11:22:29,072 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:22:29,074 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Iteration 1 complete. More tool calls: True, Has content: False
2026-01-26 11:22:29,074 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Tool iteration 2: 1 tool calls
2026-01-26 11:22:38,017 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:22:38,024 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Received final response from tool loop: True
2026-01-26 11:22:38,024 - consumers.chatbot.service - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - Using final response from tool loop (1616 chars)
2026-01-26 11:22:40,393 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] Storing: Show me code for the AgentFactory and how it works...
2026-01-26 11:22:40,393 - modules.memory.storage.knowledge_system - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] store_experience called: type=conversation, importance=1.00
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.00it/s]
2026-01-26 11:22:41,177 - modules.memory.storage.knowledge_system - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored 64afd47e-0dac-47fc-af73-becaa3ba96a9 (type=conversation, level=long_term, importance=1.00)
2026-01-26 11:22:41,227 - modules.memory.operations.injection - INFO - [req=7040fedfe06a run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored (id=64afd47e-0dac-47fc-af73-becaa3ba96a9): Show me code for the AgentFactory and how it works...
2026-01-26 11:23:45,488 - core.auth.clerk - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Auto-assigned admin role to gerard@automatos.app based on domain
2026-01-26 11:23:45,691 - api.chat - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [chat] RequestContext workspace_id=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:23:45,691 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - StreamingChatService initialized with AgentFactory integration
2026-01-26 11:23:46,494 - api.chat - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Chat request - agentId: 19, model: gpt-4
2026-01-26 11:23:46,494 - api.chat - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Using agent-based streaming with agent_id=19
2026-01-26 11:23:46,965 - modules.agents.factory.agent_factory - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 📋 Using model from settings: gpt-4-turbo-preview (context: 128000)
2026-01-26 11:23:46,970 - modules.agents.factory.agent_factory - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Agent 19 using LLM: openai/gpt-4-turbo-preview (from system settings)
2026-01-26 11:23:47,077 - modules.agents.factory.agent_factory - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - ✅ Loaded 3 Composio app assignment(s) for agent 19
2026-01-26 11:23:47,077 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:23:47,268 - modules.agents.factory.agent_factory - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - ✅ Activated agent 19 (Context Engineer API) with gpt-4-turbo-preview
2026-01-26 11:23:47,268 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Trying ContextRetrievalEngine...
2026-01-26 11:23:47,270 - modules.search.retrieval.context_retrieval_engine - WARNING - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - No embedding provided for similarity-based retrieval
2026-01-26 11:23:47,270 - modules.search.retrieval.context_retrieval_engine - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Retrieved 0 contexts in 0.4ms
2026-01-26 11:23:47,271 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] ContextRetrievalEngine returned empty, trying basic memory...
2026-01-26 11:23:47,272 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Using HierarchicalMemorySystem...
2026-01-26 11:23:47,272 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Starting parallel retrieval: Semantic + Recent
Batches:   0%|                                                                                                                                           | 0/1 [00:00<?, ?it/s]INFO:     127.0.0.1:61676 - "POST /api/chat HTTP/1.1" 200 OK
2026-01-26 11:23:47,452 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Activating agent 19 for chat f8271d3e-40fe-4c37-a3eb-02e3cfcca90b
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.15s/it]
2026-01-26 11:23:48,552 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 264a4d774ce6] Loaded 16 tools (agent_id=19, denied=0, candidates=16, 1100ms)
2026-01-26 11:23:48,825 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Semantic search found 0 memories
2026-01-26 11:23:48,825 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Recent memories: 10
2026-01-26 11:23:48,825 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Total unique memories: 10
2026-01-26 11:23:48,825 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Injecting 2346 chars of shared memory
2026-01-26 11:23:48,826 - consumers.chatbot.prompt_analyzer - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔍 Query: 'Can you create a file in /Users/gkavanagh/Development/Automatos-AI-Platform called TESTING-TOOLS.md and add text "We are testing tools"' | Top ranked tools: ['composio_execute (2)', 'search_knowledge (2)', 'create_directory (1)', 'query_database (1)', 'read_file (1)', 'search_images (1)', 'search_multimodal (1)', 'semantic_search (1)', 'smart_query_database (1)', 'write_file (1)']
2026-01-26 11:23:48,826 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔍 Narrowing tools from 16 to 10 based on ranked candidates
2026-01-26 11:23:48,826 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Generating response with agent Context Engineer API
2026-01-26 11:23:48,826 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔍 Agent tools - count: 10, is_simple: False
2026-01-26 11:23:48,826 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔍 Available tools: ['search_knowledge', 'semantic_search', 'search_images', 'search_multimodal', 'query_database', 'smart_query_database', 'read_file', 'write_file', 'create_directory', 'composio_execute']
2026-01-26 11:23:48,844 - core.llm.clients.openai_client - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Initialized OpenAI client with model: gpt-4-turbo-preview
2026-01-26 11:23:48,844 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 10 tools. Sample: [{"type": "function", "function": {"name": "search_knowledge", "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform", "parameters": {"type"...
2026-01-26 11:23:51,023 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:23:51,026 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔍 Agent LLM Response - has_tool_calls: True, content_length: 0, finish_reason: tool_calls
2026-01-26 11:23:51,027 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - ✅ Agent LLM requested 1 tool calls
2026-01-26 11:23:51,027 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Agent requested 1 tool calls
2026-01-26 11:23:51,030 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Tool iteration 1: 1 tool calls
2026-01-26 11:23:51,031 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 633af3f88995] ToolRouter execute_and_format tool=write_file agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['content', 'file_path']
2026-01-26 11:23:51,032 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:23:51,032 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 633af3f88995] execute_tool start tool=write_file agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['content', 'file_path']
2026-01-26 11:23:51,032 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 633af3f88995] Executing tool 'write_file' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:23:51,032 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 633af3f88995] Parameters keys=['content', 'file_path']
2026-01-26 11:23:51,032 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:23:51,032 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:23:51,034 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:23:51,034 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:23:51,034 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:23:51,034 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:23:51,034 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:23:51,035 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:23:51,035 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:23:51,035 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:23:51,035 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:23:51,036 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] -   🔧 Initializing file/shell executor...
2026-01-26 11:23:51,037 - modules.agents.services.agent_action_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Action executor initialized with workspace: /private/tmp/automatos_workspace
2026-01-26 11:23:51,037 - modules.agents.services.agent_action_executor - WARNING - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Action failed: write_file - {'path': '', 'size': 20} - [Errno 21] Is a directory: '/private/tmp/automatos_workspace'
2026-01-26 11:23:51,037 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] -   ✅ Tool 'write_file' executed successfully
2026-01-26 11:23:51,038 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 633af3f88995] execute_tool done tool=write_file success=False
2026-01-26 11:23:51,038 - consumers.chatbot.tool_router - WARNING - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 633af3f88995] write_file failed: Unknown error
2026-01-26 11:23:51,039 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 10 tools. Sample: [{"type": "function", "function": {"name": "search_knowledge", "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform", "parameters": {"type"...
2026-01-26 11:23:55,091 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:23:55,097 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Iteration 1 complete. More tool calls: True, Has content: False
2026-01-26 11:23:55,097 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Tool iteration 2: 1 tool calls
2026-01-26 11:23:55,098 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 0686889f56d6] ToolRouter execute_and_format tool=create_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:23:55,098 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:23:55,098 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 0686889f56d6] execute_tool start tool=create_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:23:55,098 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 0686889f56d6] Executing tool 'create_directory' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:23:55,098 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 0686889f56d6] Parameters keys=['dir_path']
2026-01-26 11:23:55,098 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:23:55,098 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:23:55,099 - modules.tools.registry.tool_registry - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:23:55,099 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] -   🔧 Initializing file/shell executor...
2026-01-26 11:23:55,099 - modules.agents.services.agent_action_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Action executor initialized with workspace: /private/tmp/automatos_workspace
2026-01-26 11:23:55,099 - modules.agents.services.agent_action_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Action executed: create_directory - {'path': '/private/tmp/automatos_workspace'}
2026-01-26 11:23:55,099 - modules.tools.execution.unified_executor - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] -   ✅ Tool 'create_directory' executed successfully
2026-01-26 11:23:55,099 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 0686889f56d6] execute_tool done tool=create_directory success=True
2026-01-26 11:23:55,099 - consumers.chatbot.tool_router - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [tool-trace 0686889f56d6] create_directory succeeded
2026-01-26 11:23:55,100 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 10 tools. Sample: [{"type": "function", "function": {"name": "search_knowledge", "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform", "parameters": {"type"...
2026-01-26 11:23:57,223 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:23:57,230 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Iteration 2 complete. More tool calls: True, Has content: False
2026-01-26 11:23:57,230 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Tool iteration 3: 1 tool calls
2026-01-26 11:23:59,284 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:23:59,286 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Received final response from tool loop: True
2026-01-26 11:23:59,286 - consumers.chatbot.service - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - Using final response from tool loop (159 chars)
2026-01-26 11:24:00,031 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] Storing: Can you create a file in /Users/gkavanagh/Developm...
2026-01-26 11:24:00,032 - modules.memory.storage.knowledge_system - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] store_experience called: type=conversation, importance=1.00
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.59it/s]
2026-01-26 11:24:00,934 - modules.memory.storage.knowledge_system - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored 57653eab-837b-485a-8bdd-1ae408083306 (type=conversation, level=long_term, importance=1.00)
2026-01-26 11:24:00,982 - modules.memory.operations.injection - INFO - [req=71cad2fd69fa run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored (id=57653eab-837b-485a-8bdd-1ae408083306): Can you create a file in /Users/gkavanagh/Developm...
2026-01-26 11:24:51,662 - core.auth.clerk - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Auto-assigned admin role to gerard@automatos.app based on domain
2026-01-26 11:24:51,952 - api.chat - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [chat] RequestContext workspace_id=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:24:51,953 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - StreamingChatService initialized with AgentFactory integration
2026-01-26 11:24:52,805 - api.chat - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Chat request - agentId: 19, model: gpt-4
2026-01-26 11:24:52,805 - api.chat - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Using agent-based streaming with agent_id=19
2026-01-26 11:24:53,477 - modules.agents.factory.agent_factory - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 📋 Using model from settings: gpt-4-turbo-preview (context: 128000)
2026-01-26 11:24:53,477 - modules.agents.factory.agent_factory - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Agent 19 using LLM: openai/gpt-4-turbo-preview (from system settings)
2026-01-26 11:24:53,575 - modules.agents.factory.agent_factory - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - ✅ Loaded 3 Composio app assignment(s) for agent 19
2026-01-26 11:24:53,575 - modules.tools.execution.unified_executor - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:24:54,084 - modules.agents.factory.agent_factory - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - ✅ Activated agent 19 (Context Engineer API) with gpt-4-turbo-preview
2026-01-26 11:24:54,089 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Trying ContextRetrievalEngine...
2026-01-26 11:24:54,090 - modules.search.retrieval.context_retrieval_engine - WARNING - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - No embedding provided for similarity-based retrieval
2026-01-26 11:24:54,090 - modules.search.retrieval.context_retrieval_engine - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Retrieved 0 contexts in 0.3ms
2026-01-26 11:24:54,091 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] ContextRetrievalEngine returned empty, trying basic memory...
2026-01-26 11:24:54,091 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Using HierarchicalMemorySystem...
2026-01-26 11:24:54,091 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Starting parallel retrieval: Semantic + Recent
Batches:   0%|                                                                                                                                           | 0/1 [00:00<?, ?it/s]INFO:     127.0.0.1:61703 - "POST /api/chat HTTP/1.1" 200 OK
2026-01-26 11:24:54,308 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Activating agent 19 for chat f8271d3e-40fe-4c37-a3eb-02e3cfcca90b
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.48it/s]
2026-01-26 11:24:55,325 - consumers.chatbot.tool_router - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [tool-trace 72e7acd3fa5c] Loaded 16 tools (agent_id=19, denied=0, candidates=16, 1016ms)
2026-01-26 11:24:55,610 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Semantic search found 0 memories
2026-01-26 11:24:55,611 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Recent memories: 10
2026-01-26 11:24:55,611 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Total unique memories: 10
2026-01-26 11:24:55,611 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Injecting 2442 chars of shared memory
2026-01-26 11:24:55,611 - consumers.chatbot.prompt_analyzer - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔍 Query: 'Can you test ssh commads' | Top ranked tools: ['search_multimodal (1)', 'smart_query_database (1)']
2026-01-26 11:24:55,611 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔍 Narrowing tools from 16 to 2 based on ranked candidates
2026-01-26 11:24:55,611 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Generating response with agent Context Engineer API
2026-01-26 11:24:55,611 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔍 Agent tools - count: 2, is_simple: False
2026-01-26 11:24:55,611 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔍 Available tools: ['search_multimodal', 'smart_query_database']
2026-01-26 11:24:55,622 - core.llm.clients.openai_client - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Initialized OpenAI client with model: gpt-4-turbo-preview
2026-01-26 11:24:55,622 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 2 tools. Sample: [{"type": "function", "function": {"name": "search_multimodal", "description": "Unified search across ALL knowledge types: documents, code, tables, images, formulas. Use this for comprehensive researc...
2026-01-26 11:24:57,204 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:24:57,214 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔍 Agent LLM Response - has_tool_calls: True, content_length: 0, finish_reason: tool_calls
2026-01-26 11:24:57,214 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - ✅ Agent LLM requested 1 tool calls
2026-01-26 11:24:57,214 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Agent requested 1 tool calls
2026-01-26 11:24:57,214 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Tool iteration 1: 1 tool calls
2026-01-26 11:24:57,215 - consumers.chatbot.tool_router - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [tool-trace 459146e6044f] ToolRouter execute_and_format tool=search_multimodal agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['query', 'kb_types']
2026-01-26 11:24:57,215 - modules.tools.execution.unified_executor - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:24:57,215 - consumers.chatbot.tool_router - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [tool-trace 459146e6044f] execute_tool start tool=search_multimodal agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['query', 'kb_types']
2026-01-26 11:24:57,215 - modules.tools.execution.unified_executor - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [tool-trace 459146e6044f] Executing tool 'search_multimodal' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:24:57,215 - modules.tools.execution.unified_executor - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [tool-trace 459146e6044f] Parameters keys=['query', 'kb_types']
2026-01-26 11:24:57,215 - modules.tools.execution.unified_executor - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:24:57,215 - modules.tools.registry.tool_registry - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:24:57,234 - modules.tools.execution.unified_executor - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - 🔧 Agent 19 executing multimodal tool: search_multimodal
2026-01-26 11:24:57,234 - modules.rag.services.multimodal_knowledge_tools.MultimodalKnowledgeTools - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Multimodal search: query='ssh commands testing', types=['code'], limit=10
2026-01-26 11:24:57,410 - modules.rag.services.multimodal_knowledge_tools.MultimodalKnowledgeTools - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Found 0 items across 0 types
2026-01-26 11:24:57,410 - modules.tools.execution.unified_executor - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] -   ✅ Tool 'search_multimodal' executed successfully
2026-01-26 11:24:57,410 - consumers.chatbot.tool_router - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [tool-trace 459146e6044f] execute_tool done tool=search_multimodal success=True
2026-01-26 11:24:57,410 - consumers.chatbot.tool_router - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [tool-trace 459146e6044f] search_multimodal succeeded
2026-01-26 11:24:57,411 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Received final response from tool loop: True
2026-01-26 11:24:57,411 - consumers.chatbot.service - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - Using final response from tool loop (217 chars)
2026-01-26 11:24:58,251 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] Storing: Can you test ssh commads...
2026-01-26 11:24:58,251 - modules.memory.storage.knowledge_system - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] store_experience called: type=conversation, importance=1.00
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.26it/s]
2026-01-26 11:24:58,762 - modules.memory.storage.knowledge_system - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored 332fc52e-1e5a-4022-a361-959581086e0b (type=conversation, level=long_term, importance=1.00)
2026-01-26 11:24:58,812 - modules.memory.operations.injection - INFO - [req=6a7448e1d5ee run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored (id=332fc52e-1e5a-4022-a361-959581086e0b): Can you test ssh commads...
2026-01-26 11:25:24,945 - core.auth.clerk - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Auto-assigned admin role to gerard@automatos.app based on domain
2026-01-26 11:25:25,126 - api.chat - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [chat] RequestContext workspace_id=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:25:25,127 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - StreamingChatService initialized with AgentFactory integration
2026-01-26 11:25:25,973 - api.chat - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Chat request - agentId: 19, model: gpt-4
2026-01-26 11:25:25,973 - api.chat - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Using agent-based streaming with agent_id=19
2026-01-26 11:25:26,473 - modules.agents.factory.agent_factory - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 📋 Using model from settings: gpt-4-turbo-preview (context: 128000)
2026-01-26 11:25:26,474 - modules.agents.factory.agent_factory - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Agent 19 using LLM: openai/gpt-4-turbo-preview (from system settings)
2026-01-26 11:25:26,569 - modules.agents.factory.agent_factory - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - ✅ Loaded 3 Composio app assignment(s) for agent 19
2026-01-26 11:25:26,569 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:25:26,745 - modules.agents.factory.agent_factory - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - ✅ Activated agent 19 (Context Engineer API) with gpt-4-turbo-preview
2026-01-26 11:25:26,745 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Trying ContextRetrievalEngine...
2026-01-26 11:25:26,746 - modules.search.retrieval.context_retrieval_engine - WARNING - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - No embedding provided for similarity-based retrieval
2026-01-26 11:25:26,746 - modules.search.retrieval.context_retrieval_engine - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Retrieved 0 contexts in 0.3ms
2026-01-26 11:25:26,746 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] ContextRetrievalEngine returned empty, trying basic memory...
2026-01-26 11:25:26,746 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Using HierarchicalMemorySystem...
2026-01-26 11:25:26,746 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Starting parallel retrieval: Semantic + Recent
2026-01-26 11:25:26,747 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Using cached recent memories
INFO:     127.0.0.1:61716 - "POST /api/chat HTTP/1.1" 200 OK
2026-01-26 11:25:26,752 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Activating agent 19 for chat f8271d3e-40fe-4c37-a3eb-02e3cfcca90b
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.38it/s]
2026-01-26 11:25:27,886 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 95dec190076b] Loaded 16 tools (agent_id=19, denied=0, candidates=16, 1128ms)
2026-01-26 11:25:28,154 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Semantic search found 0 memories
2026-01-26 11:25:28,155 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Recent memories: 10
2026-01-26 11:25:28,155 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Total unique memories: 10
2026-01-26 11:25:28,155 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Injecting 2442 chars of shared memory
2026-01-26 11:25:28,156 - consumers.chatbot.prompt_analyzer - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔍 Query: 'Can you list the available system tools ?' | Top ranked tools: ['composio_execute (1)', 'list_directory (1)', 'search_multimodal (1)', 'smart_query_database (1)']
2026-01-26 11:25:28,156 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔍 Narrowing tools from 16 to 4 based on ranked candidates
2026-01-26 11:25:28,156 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Generating response with agent Context Engineer API
2026-01-26 11:25:28,156 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔍 Agent tools - count: 4, is_simple: False
2026-01-26 11:25:28,156 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔍 Available tools: ['search_multimodal', 'smart_query_database', 'list_directory', 'composio_execute']
2026-01-26 11:25:28,172 - core.llm.clients.openai_client - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Initialized OpenAI client with model: gpt-4-turbo-preview
2026-01-26 11:25:28,173 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 4 tools. Sample: [{"type": "function", "function": {"name": "search_multimodal", "description": "Unified search across ALL knowledge types: documents, code, tables, images, formulas. Use this for comprehensive researc...
2026-01-26 11:25:29,407 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:25:29,417 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔍 Agent LLM Response - has_tool_calls: True, content_length: 0, finish_reason: tool_calls
2026-01-26 11:25:29,418 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - ✅ Agent LLM requested 1 tool calls
2026-01-26 11:25:29,418 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Agent requested 1 tool calls
2026-01-26 11:25:29,418 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Tool iteration 1: 1 tool calls
2026-01-26 11:25:29,420 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 171715ae9166] ToolRouter execute_and_format tool=list_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:25:29,420 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:25:29,420 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 171715ae9166] execute_tool start tool=list_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:25:29,420 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 171715ae9166] Executing tool 'list_directory' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:25:29,420 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 171715ae9166] Parameters keys=['dir_path']
2026-01-26 11:25:29,420 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:25:29,421 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:25:29,421 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:25:29,421 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:25:29,421 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:25:29,421 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:25:29,422 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:25:29,423 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:25:29,423 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:25:29,423 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] -   🔧 Initializing file/shell executor...
2026-01-26 11:25:29,423 - modules.agents.services.agent_action_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Action executor initialized with workspace: /private/tmp/automatos_workspace
2026-01-26 11:25:29,425 - modules.agents.services.agent_action_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Action executed: list_directory - {'path': '/private/tmp/automatos_workspace'}
2026-01-26 11:25:29,425 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] -   ✅ Tool 'list_directory' executed successfully
2026-01-26 11:25:29,425 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 171715ae9166] execute_tool done tool=list_directory success=True
2026-01-26 11:25:29,425 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 171715ae9166] list_directory succeeded
2026-01-26 11:25:29,426 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 4 tools. Sample: [{"type": "function", "function": {"name": "search_multimodal", "description": "Unified search across ALL knowledge types: documents, code, tables, images, formulas. Use this for comprehensive researc...
2026-01-26 11:25:31,110 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:25:31,117 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Iteration 1 complete. More tool calls: True, Has content: False
2026-01-26 11:25:31,117 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Tool iteration 2: 1 tool calls
2026-01-26 11:25:31,118 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] ToolRouter execute_and_format tool=composio_execute agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['action']
2026-01-26 11:25:31,118 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:25:31,118 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] execute_tool start tool=composio_execute agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['action']
2026-01-26 11:25:31,118 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] Executing tool 'composio_execute' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:25:31,118 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] Parameters keys=['action']
2026-01-26 11:25:31,118 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:25:31,118 - modules.tools.registry.tool_registry - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:25:31,118 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] Routing to Composio generic executor: composio_execute
2026-01-26 11:25:31,118 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] -   🔧 Initializing Composio executor (PRD-36)...
2026-01-26 11:25:31,119 - modules.tools.execution.unified_executor - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] Composio execute app=None action=list_system_tools agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 params_keys=[]
2026-01-26 11:25:31,299 - consumers.chatbot.tool_router - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] execute_tool done tool=composio_execute success=False
2026-01-26 11:25:31,299 - consumers.chatbot.tool_router - WARNING - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [tool-trace 13d8b7b8dd70] composio_execute failed: 'LIST' is not assigned to agent 19. Assign it to this agent before using it.
2026-01-26 11:25:31,299 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Received final response from tool loop: True
2026-01-26 11:25:31,300 - consumers.chatbot.service - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - Using final response from tool loop (76 chars)
2026-01-26 11:25:31,960 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] Storing: Can you list the available system tools ?...
2026-01-26 11:25:31,960 - modules.memory.storage.knowledge_system - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] store_experience called: type=conversation, importance=1.00
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.18it/s]
2026-01-26 11:25:32,477 - modules.memory.storage.knowledge_system - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored afc01a8a-6351-4150-9e6f-698d4325a713 (type=conversation, level=long_term, importance=1.00)
2026-01-26 11:25:32,527 - modules.memory.operations.injection - INFO - [req=c5473d980b4f run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored (id=afc01a8a-6351-4150-9e6f-698d4325a713): Can you list the available system tools ?...
2026-01-26 11:25:49,384 - core.auth.clerk - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Auto-assigned admin role to gerard@automatos.app based on domain
2026-01-26 11:25:49,558 - api.chat - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [chat] RequestContext workspace_id=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:25:49,558 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - StreamingChatService initialized with AgentFactory integration
2026-01-26 11:25:50,406 - api.chat - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Chat request - agentId: 19, model: gpt-4
2026-01-26 11:25:50,406 - api.chat - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Using agent-based streaming with agent_id=19
2026-01-26 11:25:50,873 - modules.agents.factory.agent_factory - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 📋 Using model from settings: gpt-4-turbo-preview (context: 128000)
2026-01-26 11:25:50,874 - modules.agents.factory.agent_factory - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Agent 19 using LLM: openai/gpt-4-turbo-preview (from system settings)
2026-01-26 11:25:50,962 - modules.agents.factory.agent_factory - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - ✅ Loaded 3 Composio app assignment(s) for agent 19
2026-01-26 11:25:50,963 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:25:51,158 - modules.agents.factory.agent_factory - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - ✅ Activated agent 19 (Context Engineer API) with gpt-4-turbo-preview
2026-01-26 11:25:51,158 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Trying ContextRetrievalEngine...
2026-01-26 11:25:51,158 - modules.search.retrieval.context_retrieval_engine - WARNING - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - No embedding provided for similarity-based retrieval
2026-01-26 11:25:51,158 - modules.search.retrieval.context_retrieval_engine - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Retrieved 0 contexts in 0.3ms
2026-01-26 11:25:51,158 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] ContextRetrievalEngine returned empty, trying basic memory...
2026-01-26 11:25:51,158 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Using HierarchicalMemorySystem...
2026-01-26 11:25:51,158 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Starting parallel retrieval: Semantic + Recent
2026-01-26 11:25:51,159 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Using cached recent memories
INFO:     127.0.0.1:61729 - "POST /api/chat HTTP/1.1" 200 OK
2026-01-26 11:25:51,173 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Activating agent 19 for chat f8271d3e-40fe-4c37-a3eb-02e3cfcca90b
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.57it/s]
2026-01-26 11:25:52,176 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace ec049fea6cf5] Loaded 16 tools (agent_id=19, denied=0, candidates=16, 999ms)
2026-01-26 11:25:52,463 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Semantic search found 0 memories
2026-01-26 11:25:52,463 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Recent memories: 10
2026-01-26 11:25:52,463 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Total unique memories: 10
2026-01-26 11:25:52,463 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Injecting 2442 chars of shared memory
2026-01-26 11:25:52,464 - consumers.chatbot.prompt_analyzer - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔍 Query: 'Can you list all file in - /Users/gkavanagh/Development/Automatos-AI-Platform' | Top ranked tools: ['search_knowledge (2)', 'search_multimodal (2)', 'semantic_search (2)', 'list_directory (1)', 'query_database (1)', 'read_file (1)', 'smart_query_database (1)', 'write_file (1)']
2026-01-26 11:25:52,464 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔍 Narrowing tools from 16 to 8 based on ranked candidates
2026-01-26 11:25:52,464 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Generating response with agent Context Engineer API
2026-01-26 11:25:52,465 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔍 Agent tools - count: 8, is_simple: False
2026-01-26 11:25:52,465 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔍 Available tools: ['search_knowledge', 'semantic_search', 'search_multimodal', 'query_database', 'smart_query_database', 'read_file', 'write_file', 'list_directory']
2026-01-26 11:25:52,483 - core.llm.clients.openai_client - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Initialized OpenAI client with model: gpt-4-turbo-preview
2026-01-26 11:25:52,483 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 8 tools. Sample: [{"type": "function", "function": {"name": "search_knowledge", "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform", "parameters": {"type"...
2026-01-26 11:25:54,190 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:25:54,202 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔍 Agent LLM Response - has_tool_calls: True, content_length: 0, finish_reason: tool_calls
2026-01-26 11:25:54,202 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - ✅ Agent LLM requested 1 tool calls
2026-01-26 11:25:54,202 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Agent requested 1 tool calls
2026-01-26 11:25:54,202 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Tool iteration 1: 1 tool calls
2026-01-26 11:25:54,203 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 8d898ab45b24] ToolRouter execute_and_format tool=list_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:25:54,203 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:25:54,203 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 8d898ab45b24] execute_tool start tool=list_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:25:54,203 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 8d898ab45b24] Executing tool 'list_directory' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:25:54,203 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 8d898ab45b24] Parameters keys=['dir_path']
2026-01-26 11:25:54,203 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:25:54,203 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:25:54,203 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:25:54,204 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:25:54,205 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:25:54,205 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] -   🔧 Initializing file/shell executor...
2026-01-26 11:25:54,213 - modules.agents.services.agent_action_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Action executor initialized with workspace: /private/tmp/automatos_workspace
2026-01-26 11:25:54,215 - modules.agents.services.agent_action_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Action executed: list_directory - {'path': '/private/tmp/automatos_workspace'}
2026-01-26 11:25:54,215 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] -   ✅ Tool 'list_directory' executed successfully
2026-01-26 11:25:54,215 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 8d898ab45b24] execute_tool done tool=list_directory success=True
2026-01-26 11:25:54,216 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 8d898ab45b24] list_directory succeeded
2026-01-26 11:25:54,217 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 8 tools. Sample: [{"type": "function", "function": {"name": "search_knowledge", "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform", "parameters": {"type"...
2026-01-26 11:25:55,489 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:25:55,494 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Iteration 1 complete. More tool calls: True, Has content: False
2026-01-26 11:25:55,494 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Tool iteration 2: 1 tool calls
2026-01-26 11:25:55,494 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 041854d79fb8] ToolRouter execute_and_format tool=list_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:25:55,494 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - 🔧 UnifiedToolExecutor initialized (lazy-loading enabled)
2026-01-26 11:25:55,494 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 041854d79fb8] execute_tool start tool=list_directory agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8 args=dict keys=['dir_path']
2026-01-26 11:25:55,494 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 041854d79fb8] Executing tool 'list_directory' for agent=19 workspace=ae8320bc-95e1-4de1-bbe9-396bef19cbf8
2026-01-26 11:25:55,494 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 041854d79fb8] Parameters keys=['dir_path']
2026-01-26 11:25:55,494 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] -   🔧 Initializing tool registry...
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_knowledge (category: research, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: semantic_search (category: research, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_codebase (category: research, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_tables (category: research, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_images (category: research, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_formulas (category: research, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: search_multimodal (category: research, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: query_database (category: database, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: smart_query_database (category: database, security: safe)
2026-01-26 11:25:55,494 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: read_file (category: file_ops, security: safe)
2026-01-26 11:25:55,495 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: write_file (category: file_ops, security: cautious)
2026-01-26 11:25:55,495 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: delete_file (category: file_ops, security: dangerous)
2026-01-26 11:25:55,495 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: list_directory (category: file_ops, security: safe)
2026-01-26 11:25:55,495 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: create_directory (category: file_ops, security: cautious)
2026-01-26 11:25:55,495 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: execute_command (category: shell, security: dangerous)
2026-01-26 11:25:55,495 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Registered tool: composio_execute (category: api, security: cautious)
2026-01-26 11:25:55,495 - modules.tools.registry.tool_registry - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - ToolRegistry initialized with 16 tools
2026-01-26 11:25:55,495 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] -   🔧 Initializing file/shell executor...
2026-01-26 11:25:55,495 - modules.agents.services.agent_action_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Action executor initialized with workspace: /private/tmp/automatos_workspace
2026-01-26 11:25:55,495 - modules.agents.services.agent_action_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Action executed: list_directory - {'path': '/private/tmp/automatos_workspace'}
2026-01-26 11:25:55,495 - modules.tools.execution.unified_executor - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] -   ✅ Tool 'list_directory' executed successfully
2026-01-26 11:25:55,495 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 041854d79fb8] execute_tool done tool=list_directory success=True
2026-01-26 11:25:55,495 - consumers.chatbot.tool_router - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [tool-trace 041854d79fb8] list_directory succeeded
2026-01-26 11:25:55,496 - core.llm.clients.openai_client - INFO - [req=- run=- agent=- wf=- tenant=-] - Sending tools to OpenAI (tool_choice=required): 8 tools. Sample: [{"type": "function", "function": {"name": "search_knowledge", "description": "Search the Automatos knowledge base for documentation, guides, and information about the platform", "parameters": {"type"...
2026-01-26 11:25:57,215 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:25:57,224 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Iteration 2 complete. More tool calls: True, Has content: False
2026-01-26 11:25:57,224 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Tool iteration 3: 1 tool calls
2026-01-26 11:25:59,198 - httpx - INFO - [req=- run=- agent=- wf=- tenant=-] - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-01-26 11:25:59,200 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Received final response from tool loop: True
2026-01-26 11:25:59,200 - consumers.chatbot.service - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - Using final response from tool loop (249 chars)
2026-01-26 11:26:00,028 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] Storing: Can you list all file in - /Users/gkavanagh/Develo...
2026-01-26 11:26:00,028 - modules.memory.storage.knowledge_system - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] store_experience called: type=conversation, importance=1.00
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.45it/s]
2026-01-26 11:26:00,708 - modules.memory.storage.knowledge_system - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored a0918387-8ad4-48cb-a363-337073940cad (type=conversation, level=long_term, importance=1.00)
2026-01-26 11:26:00,756 - modules.memory.operations.injection - INFO - [req=1e6dd7ddaf92 run=- agent=- wf=- tenant=-] - [Memory] ✅ Stored (id=a0918387-8ad4-48cb-a363-337073940cad): Can you list all file in - /Users/gkavanagh/Develo...
