# 🏗️ Automatos AI Architecture Overview

**Version:** 2.0.0 | **Last Updated:** December 2024 | **Status:** Production

---

## 📋 Table of Contents

1. [Architecture Principles](#architecture-principles)
2. [System Layers](#system-layers)
3. [Directory Structure](#directory-structure)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Scaling Patterns](#scaling-patterns)

---

## 🎯 Architecture Principles

Automatos AI follows a **modular, layered architecture** designed for:

- **Separation of Concerns**: Clear boundaries between API, business logic, and infrastructure
- **Scalability**: Horizontal scaling through stateless services and Redis pub/sub
- **Maintainability**: Domain-driven module organization
- **Extensibility**: Plugin-based tools and MCP integration
- **Real-time**: Server-Sent Events (SSE) for streaming updates

---

## 🏛️ System Layers

### Layer 1: API Layer (`orchestrator/api/`)

**Purpose**: HTTP REST endpoints and SSE streaming interfaces

**Components** (52 endpoint files):
- **Core APIs**: `agents.py`, `workflows.py`, `documents.py`, `system.py`
- **Streaming APIs**: `chat.py` (SSE), `chatbot_llm.py`
- **Feature APIs**: `tools.py`, `mcp_tools.py`, `credentials.py`, `skills.py`
- **Analytics APIs**: `analytics.py`, `benchmarking.py`, `statistics.py`
- **Advanced APIs**: `multi_agent.py`, `field_theory.py`, `codegraph.py`

**Responsibilities**:
- Request validation and routing
- Authentication/authorization
- Response formatting
- SSE stream management
- API documentation (OpenAPI/Swagger)

**Example**: Creating an agent
```python
# orchestrator/api/agents.py
@router.post("/", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate, db: Session = Depends(get_db)):
    # Validate, create, return
```

---

### Layer 2: Core Layer (`orchestrator/core/`)

**Purpose**: Foundational services and shared infrastructure

**Modules** (10 subdirectories):

#### `core/database/`
- SQLAlchemy ORM configuration
- Database migrations (Alembic)
- Connection pooling
- Session management

#### `core/models/`
- ORM models (Agent, Workflow, Document, etc.)
- Pydantic schemas for validation
- Enum types and constants

#### `core/llm/`
- LLM provider integrations (OpenAI, Anthropic, etc.)
- Model management and routing
- Token counting and cost tracking
- Streaming response handling

#### `core/credentials/`
- Encrypted credential storage
- Provider-specific credential resolution
- Credential validation and testing

#### `core/redis/`
- Redis client factory
- Pub/sub message handling
- Caching utilities
- Stream management

#### `core/services/`
- Analytics engine
- Workspace manager
- Dashboard integration
- Shared business services

#### `core/utils/`
- Logging adapters
- Request ID tracking
- Helper functions

#### `core/math/`
- Mathematical utilities
- Field theory calculations
- Optimization algorithms

#### `core/seeds/`
- Database seed data
- Initial configuration

**Example**: Getting LLM manager
```python
# Any module can use core services
from core.llm.manager import get_llm_manager

llm_manager = get_llm_manager()
response = await llm_manager.generate_completion(...)
```

---

### Layer 3: Modules Layer (`orchestrator/modules/`)

**Purpose**: Domain-specific business logic and features

**Domains** (12 subdirectories):

#### `modules/agents/`
- Agent factory and lifecycle
- Execution managers
- Skill assignment
- Agent coordination

#### `modules/tools/`
- **Tool registry** (single source of truth)
- Unified tool executor
- Tool categories (RESEARCH, DATABASE, FILE_OPS, etc.)
- Result formatting

#### `modules/rag/`
- Document processing pipeline
- Embedding generation
- Vector similarity search
- RAG service and configuration

#### `modules/memory/`
- Hierarchical memory systems
- Context storage
- Memory optimization
- Retrieval strategies

#### `modules/codegraph/`
- Code indexing (Python, TypeScript)
- Symbol extraction
- Semantic code search
- Call graph analysis

#### `modules/orchestrator/`
- Workflow execution engine
- Stage tracking
- Subtask coordination
- Progress monitoring

#### `modules/search/`
- Multi-source search integration
- Web search
- Knowledge base search
- Hybrid search strategies

#### `modules/nl2sql/`
- Natural language to SQL translation
- Query validation
- Schema introspection
- Result formatting

#### `modules/learning/`
- Continuous learning patterns
- Performance tracking
- Adaptive optimization

#### `modules/reasoning/`
- Multi-agent reasoning
- Consensus mechanisms
- Decision frameworks

#### `modules/evaluation/`
- Quality assessment
- Performance metrics
- Benchmarking

**Example**: Using tool registry
```python
# In any API endpoint
from modules.tools import ToolRegistry, UnifiedToolExecutor

registry = ToolRegistry()
executor = UnifiedToolExecutor(db_session)
result = await executor.execute_tool("search_web", {"query": "AI news"}, agent_id)
```

---

### Layer 4: Consumers Layer (`orchestrator/consumers/`)

**Purpose**: Background processing and event-driven workflows

**Domains**:

#### `consumers/chatbot/`
- Streaming chat service
- Tool execution routing
- Chat history management
- Artifact generation

#### `consumers/document_processor/`
- Async document processing
- Chunking and embedding
- Queue management

#### `consumers/workflows/`
- Workflow execution background tasks
- Stage coordination
- Progress streaming

**Example**: Background workflow execution
```python
# orchestrator/consumers/workflows/streaming.py
async def run_workflow_execution(...):
    # Execute workflow in background
    # Stream progress via SSE
    # Update database state
```

---

## 📁 Directory Structure

```
automatos-ai/
├── orchestrator/              # Backend Python application
│   ├── api/                   # 🔵 Layer 1: REST/SSE endpoints (52 files)
│   │   ├── agents.py          # Agent CRUD operations
│   │   ├── workflows.py       # Workflow management (122KB!)
│   │   ├── chat.py            # SSE streaming chat
│   │   ├── tools.py           # Tool management
│   │   ├── mcp_tools.py       # MCP protocol integration
│   │   └── ... (47 more)
│   │
│   ├── core/                  # 🟢 Layer 2: Foundational services
│   │   ├── database/          # ORM, migrations
│   │   ├── models/            # Data models
│   │   ├── llm/               # LLM integrations
│   │   ├── credentials/       # Credential management
│   │   ├── redis/             # Redis client
│   │   ├── services/          # Shared services
│   │   └── utils/             # Utilities
│   │
│   ├── modules/               # 🟡 Layer 3: Business logic
│   │   ├── agents/            # Agent orchestration
│   │   ├── tools/             # Tool registry & execution
│   │   ├── rag/               # RAG pipeline
│   │   ├── memory/            # Memory systems
│   │   ├── codegraph/         # Code intelligence
│   │   ├── orchestrator/      # Workflow engine
│   │   └── ... (6 more)
│   │
│   ├── consumers/             # 🔴 Layer 4: Background processors
│   │   ├── chatbot/           # Streaming chat
│   │   ├── document_processor/# Doc processing
│   │   └── workflows/         # Workflow execution
│   │
│   ├── main.py                # 🚀 FastAPI application
│   ├── config.py              # Configuration
│   ├── alembic/               # Database migrations
│   └── requirements.txt       # Python dependencies
│
├── frontend/                  # React/Next.js frontend
│   ├── app/                   # Next.js app router
│   ├── components/            # React components
│   ├── lib/                   # Client libraries
│   └── public/                # Static assets
│
└── docs/                      # 📚 Documentation
    ├── PRDS/                  # Product requirement docs (31 files)
    ├── README.md              # Documentation hub
    ├── architecture.md        # This file
    └── ... (guides)
```

---

## 🔄 Data Flow

### Request Flow (API → Response)

```
1. HTTP Request
   ↓
2. FastAPI Router (api/*.py)
   ↓
3. Authentication/Validation
   ↓
4. Business Logic (modules/*)
   ↓
5. Database (core/database)
   ↓
6. Response Formatting
   ↓
7. HTTP Response
```

### Streaming Flow (SSE)

```
1. Client connects to /api/chat/stream
   ↓
2. API creates SSE stream
   ↓
3. Consumer processes request
   ↓
4. Events published to Redis
   ↓
5. SSE manager broadcasts
   ↓
6. Client receives real-time updates
```

### Tool Execution Flow

```
1. LLM requests tool use
   ↓
2. API receives tool call (api/chat.py)
   ↓
3. Tool router validates (consumers/chatbot/tool_router.py)
   ↓
4. Tool registry resolves (modules/tools/)
   ↓
5. Unified executor runs tool
   ↓
6. Result formatted
   ↓
7. Injected into LLM context
```

### Workflow Execution Flow

```
1. User creates workflow
   ↓
2. API persists to DB (api/workflows.py)
   ↓
3. Execute endpoint triggered
   ↓
4. Background task spawned (consumers/workflows/)
   ↓
5. Orchestrator engine runs stages (modules/orchestrator/)
   ↓
6. Progress streamed via SSE
   ↓
7. Results stored in DB
```

---

## 🛠️ Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI + Uvicorn | Async REST API + SSE |
| **Database** | PostgreSQL 15+ | Relational data storage |
| **ORM** | SQLAlchemy 2.0 | Database abstraction |
| **Migrations** | Alembic | Schema versioning |
| **Caching** | Redis 7+ | Pub/sub + caching |
| **Streaming** | Server-Sent Events (SSE) | Real-time updates |
| **Vector DB** | pgvector | Semantic search | 
| **LLM** | OpenAI, Anthropic, etc | AI capabilities |
| **Auth** | API Key | Simple authentication |

### Frontend

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | Next.js 14+ | React framework |
| **Language** | TypeScript | Type safety |
| **State** | Zustand | State management |
| **UI** | Tailwind CSS + Radix UI | Styling + components |
| **SSE Client** | EventSource API | Real-time updates |

### Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Containerization** | Docker + Docker Compose | Local development |
| **Orchestration** | Kubernetes (optional) | Production scaling |
| **Monitoring** | Built-in analytics | System metrics |
| **Logging** | Python logging | Structured logs |

---

## 📈 Scaling Patterns

### Horizontal Scaling

**Stateless API Servers**:
```
Load Balancer
├── API Instance 1
├── API Instance 2
└── API Instance 3
     ↓
Shared PostgreSQL + Redis
```

**Worker Pool** (for background tasks):
```
Task Queue (Redis)
├── Worker 1 (document processing)
├── Worker 2 (workflow execution)
└── Worker 3 (agent coordination)
```

### Caching Strategy

1. **Redis Cache** - Frequently accessed data
2. **Database Connection Pool** - Reuse DB connections
3. **LLM Response Cache** - Cache identical queries

### Real-time Distribution

```
Multiple API Instances
     ↓
Redis Pub/Sub
     ↓
All SSE Connections
```

Each API instance:
- Subscribes to Redis channels
- Maintains SSE connections
- Broadcasts relevant events

---

## 🔒 Security Architecture

### Authentication
- API key-based (header: `X-API-Key`)
- Environment-based configuration
- Per-endpoint authorization

### Credential Management
- Encrypted at rest (core/credentials)
- Provider-specific resolution
- Secure credential injection

### Data Protection
- PostgreSQL encryption
- HTTPS in production
- Input validation (Pydantic)

---

## 🚀 Key Design Decisions

### ✅ SSE over WebSocket
- **Why**: Simpler, auto-reconnect, HTTP/2 multiplexing
- **Where**: `api/chat.py`, `consumers/chatbot/streaming.py`
- **Migration**: All WebSocket code removed

### ✅ Tool Registry Pattern
- **Why**: Single source of truth, DRY principle
- **Where**: `modules/tools/`
- **Benefit**: One place to add/modify tools

### ✅ Modular Architecture
- **Why**: Separation of concerns, testability
- **Layers**: API | Core | Modules | Consumers
- **Benefit**: Clear boundaries, easy to navigate

### ✅ Domain-Driven Modules
- **Why**: Business logic organization
- **Examples**: agents, tools, rag, memory
- **Benefit**: Features are self-contained

---

## 📚 Related Documentation

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Development workflow
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment
- **[Workflow System](WORKFLOW_SYSTEM_GUIDE.md)** - Workflow orchestration
- **[Tools Integration](TOOLS_INTEGRATION_GUIDE.md)** - Tool development

---

*Built with ❤️ by the Automatos AI community*
