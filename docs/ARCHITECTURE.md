# Automotas AI - System Architecture & Flow Documentation

## 🏗️ System Overview

Automotas AI is a comprehensive AI agent orchestration platform built with FastAPI, featuring advanced context engineering, real-time WebSocket communication, and sophisticated agent management capabilities.

## 📁 Directory Structure

```
automatos-ai/orchestrator/
├── main.py                    # Application entry point
├── docker-compose.yml         # Container orchestration  
├── requirements.txt           # Python dependencies
├── alembic/                   # Database migrations
├── src/                       # Organized source code
│   ├── api/                   # REST API endpoints
│   ├── database/              # Data models & database logic
│   ├── services/              # Core business services
│   ├── agents/                # Agent management & orchestration
│   ├── context/               # Context engineering & RAG
│   ├── utils/                 # Utility functions & tools
│   ├── monitoring/            # System monitoring & metrics
│   └── mcp/                   # Model Context Protocol server
├── logs/                      # Application logs
├── vector_stores/             # Vector database storage
├── projects/                  # Project workspaces
└── keys/                      # API keys & credentials
```

## 🔧 Core Components

### 1. Application Layer (main.py)
- **FastAPI Application**: Main ASGI application with CORS middleware
- **Router Registration**: Centralized API route management
- **WebSocket Support**: Real-time bidirectional communication
- **Lifecycle Management**: Database initialization and shutdown hooks
- **Logging Configuration**: Structured logging setup

### 2. API Layer (src/api/)
All REST endpoints organized by domain:

#### Agent Management (agents.py)
- GET /api/agents/ - List agents with filtering & pagination
- POST /api/agents/ - Create new agents
- GET /api/agents/types - Available agent types
- GET /api/agents/stats - Agent statistics
- GET /api/agents/{id}/status - Agent status monitoring
- POST /api/agents/{id}/execute - Execute agent tasks
- POST /api/agents/bulk - Bulk agent operations
- PUT /api/agents/{id} - Update agent configuration
- DELETE /api/agents/{id} - Remove agents

#### Workflow Orchestration (workflows.py)
- GET /api/workflows/ - List workflows
- POST /api/workflows/ - Create workflow definitions
- POST /api/workflows/{id}/execute - Execute workflows
- GET /api/workflows/{id}/executions - Execution history

#### Document Management (documents.py)
- POST /api/documents/upload - Upload documents
- GET /api/documents/ - List documents
- GET /api/documents/{id} - Retrieve document content
- DELETE /api/documents/{id} - Remove documents

#### Skills Management (skills.py)
- GET /api/skills/ - List available skills
- POST /api/skills/ - Create new skills
- GET /api/skills/categories - Skill categories
- POST /api/skills/bulk - Bulk skill creation

#### System Operations (system.py)
- GET /api/system/health - System health check
- GET /api/system/metrics - Performance metrics
- GET /api/system/config - System configuration

#### Context Engineering (context.py)
- POST /api/context/query - Context-aware queries
- GET /api/context/stats - Context statistics
- POST /api/context/index - Index new content

### 3. Database Layer (src/database/)

#### Models (models.py)
Core data entities with relationships:
- Agent: AI agents with capabilities
- Skill: Reusable agent skills
- Pattern: Coordination patterns
- Workflow: Multi-agent workflows
- WorkflowExecution: Execution tracking
- Document: Knowledge base documents
- SystemConfiguration: System settings
- RAGConfiguration: RAG system config

#### Database Management (database.py)
- Connection Management: SQLAlchemy engine & session management
- Schema Initialization: Table creation and migration support
- Seed Data: Default agents, skills, and configurations
- Health Checks: Database connectivity monitoring

### 4. Services Layer (src/services/)

#### LLM Provider (llm_provider.py)
- Multi-Provider Support: OpenAI, Anthropic, local models
- Request Routing: Intelligent model selection
- Response Processing: Standardized output formatting
- Error Handling: Graceful fallback mechanisms

#### WebSocket Manager (websocket_manager.py)
- Connection Management: Client lifecycle handling
- Event Broadcasting: Real-time updates to connected clients
- Room Management: Grouped communication channels
- Message Routing: Targeted message delivery

#### RAG Service (rag_service.py)
- Document Ingestion: Text processing and chunking
- Vector Storage: Embedding generation and indexing
- Similarity Search: Context retrieval for queries
- Integration: Seamless LLM context enhancement

#### System Health (system_health_service.py)
- Resource Monitoring: CPU, memory, disk usage
- Service Status: Component health checks
- Performance Metrics: Response times and throughput
- Alerting: Threshold-based notifications

#### MCP Bridge (mcp_bridge.py)
- Protocol Implementation: Model Context Protocol support
- External Integration: Third-party service connectivity
- Data Translation: Format conversion and mapping

### 5. Agent Management (src/agents/)

#### Orchestrator (orchestrator.py)
- Agent Lifecycle: Creation, configuration, destruction
- Task Distribution: Work allocation and load balancing
- Communication: Inter-agent message passing
- State Management: Agent status and context tracking

#### Agent Communication (agent_comm.py)
- Message Protocol: Structured agent-to-agent communication
- Event System: Publish-subscribe messaging
- Synchronization: Coordination mechanisms
- Conflict Resolution: Concurrent access handling

### 6. Context Engineering (src/context/)

#### Vector Store (vector_store.py)
- Embedding Storage: High-dimensional vector indexing
- Similarity Search: Efficient nearest neighbor retrieval
- Metadata Management: Document and chunk annotations
- Performance Optimization: Caching and indexing strategies

#### Embeddings (embeddings.py)
- Model Integration: Multiple embedding providers
- Batch Processing: Efficient bulk embedding generation
- Caching: Embedding result storage and reuse
- Quality Assurance: Embedding validation and metrics

#### Context Retriever (context_retriever.py)
- Query Processing: User intent understanding
- Context Assembly: Relevant information aggregation
- Ranking: Relevance scoring and prioritization
- Filtering: Content quality and safety checks

## 🔄 Data Flow & Request Lifecycle

### 1. Agent Creation Flow
```
User Request → API Router → Validation → Database → Agent Creation → Skills Assignment → Response
     ↓
WebSocket Notification → Connected Clients
```

### 2. Context-Aware Query Flow
```
Query → Context Engineering → Document Retrieval → Embedding Search → 
LLM Processing → Response Generation → Result Formatting → User Response
```

### 3. Workflow Execution Flow
```
Workflow Definition → Agent Selection → Task Distribution → 
Parallel Execution → Result Aggregation → Status Updates → Completion Notification
```

## 🌐 Integration Points

### External Services
- OpenAI API: GPT models for language processing
- Anthropic API: Claude models for advanced reasoning
- PostgreSQL: Primary data storage with pgvector
- Redis: Caching and session management
- Docker: Containerized deployment

### Internal Communication
- REST APIs: Synchronous request-response
- WebSockets: Real-time bidirectional communication
- Database Events: Trigger-based notifications
- Message Queues: Asynchronous task processing

## 🚀 Deployment Architecture

### Container Structure
```
├── backend_api          # Main FastAPI application
├── postgres            # PostgreSQL with pgvector
├── redis              # Caching and sessions
└── monitoring         # Prometheus & Grafana (optional)
```

## 🔗 Quick Start for Developers

1. Setup Environment: Configure database and API keys
2. Run Migrations: Initialize database schema
3. Start Services: Launch containers with docker-compose
4. Test APIs: Verify endpoint functionality
5. Monitor Health: Check system status and metrics

This architecture provides a robust, scalable foundation for AI agent orchestration with comprehensive monitoring, real-time communication, and advanced context engineering capabilities.
