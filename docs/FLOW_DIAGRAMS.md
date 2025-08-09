# Automatos AI - System Flow Diagrams

## 🔄 Request Flow Diagrams

### 1. Agent Creation & Management Flow

```mermaid
graph TD
    A[User Request] --> B[FastAPI Router]
    B --> C[Agent Controller]
    C --> D[Input Validation]
    D --> E{Valid Input?}
    E -->|No| F[Return 400 Error]
    E -->|Yes| G[Agent Model Creation]
    G --> H[Database Transaction]
    H --> I[Skills Assignment]
    I --> J[Agent Response Build]
    J --> K[WebSocket Notification]
    K --> L[Return Success Response]
    
    style A fill:#e1f5fe
    style L fill:#e8f5e8
    style F fill:#ffebee
```

### 2. Context-Aware Query Processing

```mermaid
graph TD
    A[User Query] --> B[Context Router]
    B --> C[Query Preprocessing]
    C --> D[Embedding Generation]
    D --> E[Vector Search]
    E --> F[Document Retrieval]
    F --> G[Context Assembly]
    G --> H[LLM Provider Selection]
    H --> I[LLM API Call]
    I --> J[Response Processing]
    J --> K[Result Formatting]
    K --> L[Cache Update]
    L --> M[Return Enhanced Response]
    
    style A fill:#e1f5fe
    style M fill:#e8f5e8
```

### 3. Multi-Agent Workflow Execution

```mermaid
graph TD
    A[Workflow Definition] --> B[Agent Selection]
    B --> C[Task Distribution]
    C --> D[Parallel Agent Execution]
    D --> E[Agent 1]
    D --> F[Agent 2]
    D --> G[Agent N]
    E --> H[Result Collection]
    F --> H
    G --> H
    H --> I[Result Aggregation]
    I --> J[Status Updates]
    J --> K[WebSocket Broadcast]
    K --> L[Workflow Completion]
    
    style A fill:#e1f5fe
    style L fill:#e8f5e8
```

### 4. Real-Time Communication Flow

```mermaid
graph TD
    A[Event Trigger] --> B[WebSocket Manager]
    B --> C[Client Filter]
    C --> D{Room/Channel?}
    D -->|Broadcast| E[All Clients]
    D -->|Targeted| F[Specific Clients]
    E --> G[Message Delivery]
    F --> G
    G --> H[Client Update]
    H --> I[Acknowledgment]
    
    style A fill:#e1f5fe
    style I fill:#e8f5e8
```

## 🏗️ System Architecture Diagram

```mermaid
graph TB
    subgraph Client Layer
        UI[Web UI]
        API_CLIENT[API Clients]
        WS_CLIENT[WebSocket Clients]
    end
    
    subgraph API Gateway
        NGINX[Nginx Reverse Proxy]
    end
    
    subgraph Application Layer
        MAIN[Main FastAPI App]
        subgraph API Routers
            AGENTS[Agents API]
            WORKFLOWS[Workflows API]
            DOCS[Documents API]
            MEMORY[Memory API]
            EVALUATION[Evaluation API]
            MULTIAGENT[Multi-Agent API]
            FIELD[Field Theory API]
            SYSTEM[System API]
            CONTEXT[Context Engineering API]
        end
    end
    
    subgraph Service Layer
        LLM[LLM Provider]
        WS_MGR[WebSocket Manager]
        RAG[RAG Service]
        HEALTH[Health Service]
        MCP[MCP Bridge]
    end
    
    subgraph Agent Layer
        ORCH[Orchestrator]
        COMM[Agent Communication]
        SEED[Professional Agents]
    end
    
    subgraph Context Layer
        VECTOR[Vector Store]
        EMBED[Embeddings]
        CHUNK[Chunking]
        RETRIEVER[Context Retriever]
        LEARNING[Learning Engine]
    end
    
    subgraph Data Layer
        POSTGRES[(PostgreSQL + pgvector)]
        REDIS[(Redis Cache)]
        FILES[File Storage]
    end
    
    subgraph External Services
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
    end
    
    UI --> NGINX
    API_CLIENT --> NGINX
    WS_CLIENT --> NGINX
    
    NGINX --> MAIN
    MAIN --> AGENTS
    MAIN --> WORKFLOWS
    MAIN --> DOCS
    MAIN --> SYSTEM
    MAIN --> CONTEXT
     MAIN --> MEMORY
     MAIN --> EVALUATION
     MAIN --> MULTIAGENT
     MAIN --> FIELD
    
    AGENTS --> LLM
    AGENTS --> WS_MGR
    CONTEXT --> RAG
    SYSTEM --> HEALTH
    
    RAG --> VECTOR
    RAG --> EMBED
    VECTOR --> POSTGRES
    EMBED --> POSTGRES
    
    LLM --> OPENAI
    LLM --> ANTHROPIC
    
    ORCH --> POSTGRES
    WS_MGR --> REDIS
    HEALTH --> POSTGRES
    HEALTH --> REDIS
    
    style UI fill:#e3f2fd
    style POSTGRES fill:#fff3e0
    style REDIS fill:#fff3e0
    style OPENAI fill:#f3e5f5
    style ANTHROPIC fill:#f3e5f5
```

## 📊 Data Flow Architecture

```mermaid
graph LR
    subgraph Input Sources
        USER[User Requests]
        DOCS[Documents]
        CONFIGS[Configuration]
    end
    
    subgraph Processing Pipeline
        VALIDATE[Validation]
        TRANSFORM[Transformation]
        ENRICH[Context Enrichment]
    end
    
    subgraph Storage Systems
        DB[(Primary Database)]
        CACHE[(Cache Layer)]
        VECTORS[(Vector Store)]
        FILES[(File Storage)]
    end
    
    subgraph Output Channels
        API_RESP[API Responses]
        WS_EVENTS[WebSocket Events]
        LOGS[System Logs]
    end
    
    USER --> VALIDATE
    DOCS --> VALIDATE
    CONFIGS --> VALIDATE
    
    VALIDATE --> TRANSFORM
    TRANSFORM --> ENRICH
    
    ENRICH --> DB
    ENRICH --> CACHE
    ENRICH --> VECTORS
    ENRICH --> FILES
    
    DB --> API_RESP
    CACHE --> API_RESP
    VECTORS --> API_RESP
    
    DB --> WS_EVENTS
    CACHE --> WS_EVENTS
    
    API_RESP --> LOGS
    WS_EVENTS --> LOGS
    
    style USER fill:#e8f5e8
    style API_RESP fill:#e8f5e8
    style WS_EVENTS fill:#e8f5e8
```

## 🔐 Security & Authentication Flow

```mermaid
graph TD
    A[Incoming Request] --> B[Rate Limiting]
    B --> C[CORS Validation]
    C --> D[API Key Check]
    D --> E{Valid Key?}
    E -->|No| F[401 Unauthorized]
    E -->|Yes| G[Input Sanitization]
    G --> H[Request Validation]
    H --> I{Valid Request?}
    I -->|No| J[400 Bad Request]
    I -->|Yes| K[Process Request]
    K --> L[Log Activity]
    L --> M[Return Response]
    
    style A fill:#e1f5fe
    style M fill:#e8f5e8
    style F fill:#ffebee
    style J fill:#ffebee
```

## 🚀 Deployment Flow

```mermaid
graph TD
    A[Code Repository] --> B[Docker Build]
    B --> C[Image Registry]
    C --> D[Container Orchestration]
    D --> E[Service Deployment]
    E --> F[Health Checks]
    F --> G{Healthy?}
    G -->|No| H[Rollback]
    G -->|Yes| I[Load Balancer Update]
    I --> J[Traffic Routing]
    J --> K[Monitor & Log]
    
    H --> E
    
    style A fill:#e1f5fe
    style K fill:#e8f5e8
    style H fill:#fff3e0
```

## 📈 Monitoring & Observability Flow

```mermaid
graph TD
    A[Application Events] --> B[Metrics Collection]
    A --> C[Log Aggregation]
    A --> D[Trace Collection]
    
    B --> E[Prometheus]
    C --> F[Log Storage]
    D --> G[Tracing System]
    
    E --> H[Grafana Dashboard]
    F --> I[Log Analysis]
    G --> J[Performance Analysis]
    
    H --> K[Alerting]
    I --> K
    J --> K
    
    K --> L[Incident Response]
    
    style A fill:#e1f5fe
    style L fill:#ffebee
```

## 🔄 Component Interaction Matrix

| Component | Depends On | Provides To | Communication |
|-----------|------------|-------------|---------------|
| **main.py** | All routers, database | Application entry | FastAPI |
| **agents.py** | Database, WebSocket | Agent CRUD | REST API |
| **workflows.py** | Database, Agents | Workflow management | REST API |
| **context.py** | RAG, Vector Store | Context queries | REST API |
| **database.py** | PostgreSQL | Data persistence | SQLAlchemy |
| **websocket_manager.py** | Redis | Real-time updates | WebSocket |
| **llm_provider.py** | OpenAI, Anthropic | LLM services | HTTP API |
| **rag_service.py** | Vector Store, Embeddings | Context retrieval | Function calls |
| **vector_store.py** | PostgreSQL pgvector | Vector operations | SQL/pgvector |

## 📝 Development Workflow

```mermaid
graph LR
    A[Local Development] --> B[Code Changes]
    B --> C[Unit Tests]
    C --> D[Integration Tests]
    D --> E[Docker Build]
    E --> F[Local Testing]
    F --> G[Git Commit]
    G --> H[CI/CD Pipeline]
    H --> I[Automated Testing]
    I --> J[Staging Deployment]
    J --> K[Production Deployment]
    
    style A fill:#e8f5e8
    style K fill:#e8f5e8
```

This comprehensive flow documentation provides developers with clear understanding of how data moves through the system, how components interact, and what the request lifecycle looks like for different operations.
