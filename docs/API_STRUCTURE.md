# 🔌 API Structure & Endpoints

**Version:** 2.0.0 | **Last Updated:** December 2024

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [API Organization](#api-organization)
3. [Endpoint Categories](#endpoint-categories)
4. [Authentication](#authentication)
5. [Streaming Architecture](#streaming-architecture)
6. [Error Handling](#error-handling)

---

## 🎯 Overview

Automatos AI exposes **52 REST API endpoints** organized into functional routers. All endpoints follow Open API 3.0 specification and are documented at `/docs` (Swagger UI).

**Base URL**: `http://localhost:8000` (development)  
**API Prefix**: Most endpoints use `/api/*`  
**Documentation**: `/docs` (Swagger), `/redoc` (ReDoc)

---

## 🗂️ API Organization

### Router Structure

All API routers are located in `orchestrator/api/` and registered in `main.py`:

```python
# orchestrator/main.py
app.include_router(agents_router)           # /api/agents
app.include_router(workflows_router)        # /api/workflows
app.include_router(chat_router)             # /api/chat
# ... 49 more routers
```

### Endpoint File Mapping

| File | Router Prefix | Description | Lines |
|------|---------------|-------------|-------|
| `workflows.py` | `/api/workflows` | Workflow CRUD + execution | 2632 |
| `agents.py` | `/api/agents` | Agent management | 800+ |
| `documents.py` | `/api/documents` | Document processing + RAG | 1500+ |
| `system.py` | `/api/system` | System config + health | 900+ |
| `chat.py` | `/api/chat` | SSE streaming chat | 600+ |
| `tools.py` | `/tools` | Tool management | 600+ |
| `mcp_tools.py` | `/api/mcp` | MCP protocol tools | 500+ |
| `credentials.py` | `/api/credentials` | Credential management | 700+ |
| `codegraph.py` | `/api/code-graph` | Code intelligence | 400+ |
| ... | ... | ... | ... |

---

## 📁 Endpoint Categories

### 1. Core Entities

#### 🤖 Agents (`/api/agents`)
```
GET    /api/agents              # List agents
POST   /api/agents              # Create agent
GET    /api/agents/{id}         # Get agent details
PUT    /api/agents/{id}         # Update agent
DELETE /api/agents/{id}         # Delete agent
POST   /api/agents/{id}/skills  # Assign skills
GET    /api/agents/{id}/stats   # Agent statistics
```

#### 🔄 Workflows (`/api/workflows`)
```
GET    /api/workflows                      # List workflows
POST   /api/workflows                      # Create workflow
GET    /api/workflows/{id}                 # Get workflow
PUT    /api/workflows/{id}                 # Update workflow
DELETE /api/workflows/{id}                 # Delete workflow
POST   /api/workflows/{id}/execute         # Execute workflow
GET    /api/workflows/{id}/executions      # List executions
GET    /api/workflows/executions/{exec_id} # Get execution details
POST   /api/workflows/executions/{exec_id}/cancel # Cancel execution
```

**SSE Streaming**:
```
GET /api/workflows/executions/{id}/stream        # JSON SSE stream
GET /api/workflows/executions/{id}/stream/aisdk  # AI SDK format
```

#### 📄 Documents (`/api/documents`)
```
GET    /api/documents              # List documents
POST   /api/documents              # Upload document
GET    /api/documents/{id}         # Get document
DELETE /api/documents/{id}         # Delete document
POST   /api/documents/{id}/reprocess # Re-process document
GET    /api/documents/{id}/content # Get content
POST   /api/documents/search       # Semantic search
```

---

### 2. AI Features

#### 💬 Chat (`/api/chat`)
```
POST /api/chat/stream              # SSE streaming chat
GET  /api/chat/sessions            # List chat sessions
POST /api/chat/sessions            # Create session
GET  /api/chat/sessions/{id}       # Get session + history
DELETE /api/chat/sessions/{id}     # Delete session
POST /api/chat/sessions/{id}/clear # Clear history
```

**Legacy** (backward compatibility):
```
POST /api/chatbot/stream  # Old endpoint, redirects to /api/chat/stream
```

#### 🛠️ Tools (`/tools`, `/api/mcp`)
```
GET  /tools                    # List all tools
GET  /tools/{id}               # Get tool details
POST /tools/{id}/install       # Install tool
POST /tools/{id}/configure     # Configure tool
DELETE /tools/{id}/uninstall   # Uninstall tool
GET  /tools/{id}/status        # Tool health status

GET  /api/mcp/servers          # List MCP servers
POST /api/mcp/servers          # Register MCP server
GET  /api/mcp/tools            # List MCP tools
POST /api/mcp/tools/execute    # Execute MCP tool
```

#### 💡 Skills (`/api/skills`)
```
GET    /api/skills           # List skills
POST   /api/skills           # Create skill
GET    /api/skills/{id}      # Get skill
PUT    /api/skills/{id}      # Update skill
DELETE /api/skills/{id}      # Delete skill
```

---

### 3. Knowledge & Intelligence

#### 🗄️ CodeGraph (`/api/code-graph`)
```
POST /api/code-graph/index/github          # Index GitHub repo
GET  /api/code-graph/search/symbols        # Symbol search
GET  /api/code-graph/search/semantic       # Semantic code search
GET  /api/code-graph/projects              # List indexed projects
DELETE /api/code-graph/projects/{id}       # Delete project
POST /api/code-graph/projects/{id}/reindex # Re-index project
GET  /api/code-graph/call-graph            # Get call graph
```

#### 🧠 Memory (`/api/memory`)
```
GET    /api/memory/items          # List memory items
POST   /api/memory/items          # Create memory
GET    /api/memory/items/{id}     # Get memory
DELETE /api/memory/items/{id}     # Delete memory
POST   /api/memory/search         # Search memories
GET    /api/memory/stats          # Memory statistics
```

#### 🗃️ Knowledge (`/api/knowledge`, `/api/knowledge-multimodal`)
```
GET  /api/knowledge/items              # List knowledge items
POST /api/knowledge/items              # Add knowledge
POST /api/knowledge/search             # Knowledge search
GET  /api/knowledge-multimodal/items   # Multimodal knowledge
POST /api/knowledge-multimodal/analyze # Analyze media
```

#### 🗺️ Knowledge Graph (`/api/knowledge-graph`)
```
GET  /api/knowledge-graph/nodes        # List nodes
POST /api/knowledge-graph/nodes        # Create node
POST /api/knowledge-graph/relationships # Create relationship
GET  /api/knowledge-graph/query        # Graph query
```

---

### 4. Advanced Features

#### 👥 Multi-Agent (`/api/multi-agent`)
```
POST /api/multi-agent/reasoning/collaborative # Collaborative reasoning
POST /api/multi-agent/coordination/execute    # Coordinate agents
GET  /api/multi-agent/behavior/monitor        # Monitor behavior
POST /api/multi-agent/optimize                # Optimize system
GET  /api/multi-agent/health                  # System health
```

#### 🌐 Field Theory (`/api/field-theory`)
```
POST /api/field-theory/fields/{id}        # Create field
GET  /api/field-theory/fields/{id}        # Get field
POST /api/field-theory/propagate          # Propagate field
POST /api/field-theory/interact           # Field interactions
GET  /api/field-theory/health             # Field system health
```

#### 🎯 Context Engineering (`/api/context-engineering`)
```
POST /api/context-engineering/assemble    # Assemble context
POST /api/context-engineering/optimize    # Optimize context
POST /api/context-engineering/vector-ops  # Vector operations
GET  /api/context-engineering/stats       # Context statistics
```

---

### 5. System & Configuration

#### ⚙️ System (`/api/system`)
```
GET  /api/system/config              # List system configs
POST /api/system/config              # Create config
GET  /api/system/config/{key}        # Get config
PUT  /api/system/config/{key}        # Update config

GET  /api/system/rag                 # List RAG configs
POST /api/system/rag                 # Create RAG config
GET  /api/system/rag/{id}            # Get RAG config

GET  /api/system/health              # System health
GET  /api/system/metrics             # System metrics (with time-series)
```

#### 🔐 Credentials (`/api/credentials`)
```
GET    /api/credentials              # List all credentials
POST   /api/credentials              # Create credential
GET    /api/credentials/{id}         # Get credential
PUT    /api/credentials/{id}         # Update credential
DELETE /api/credentials/{id}         # Delete credential
POST   /api/credentials/{id}/test    # Test credential
GET    /api/credentials/providers    # List providers
```

#### ⚙️ System Settings (`/api/settings`)
```
GET  /api/settings/llm               # Get LLM settings
PUT  /api/settings/llm               # Update LLM settings
GET  /api/settings/services          # List service configs
PUT  /api/settings/services/{name}   # Update service config
```

---

### 6. Analytics & Monitoring

#### 📊 Analytics (`/api/analytics`, `/api/analytics-real`)
```
GET /api/analytics/overview          # Dashboard overview
GET /api/analytics/context           # Context optimization metrics
GET /api/analytics/learning          # Learning system metrics
GET /api/analytics-real/realtime     # Real-time analytics stream
```

#### 📈 Statistics (`/api/statistics`)
```
GET /api/statistics/agents           # Agent statistics
GET /api/statistics/workflows        # Workflow statistics
GET /api/statistics/system           # System statistics
```

#### 🎯 Benchmarking (`/api/benchmarking`)
```
POST /api/benchmarking/run           # Run benchmark
GET  /api/benchmarking/results       # Get results
GET  /api/benchmarking/compare       # Compare benchmarks
```

---

## 🔐 Authentication

### API Key Auth

**Header**: `X-API-Key`  
**Configuration**: Set in environment variable `API_KEY`  
**Enforcement**: Controlled by `REQUIRE_API_KEY` env var

```bash
# Example request
curl -H "X-API-Key: your-api-key-here" \
     http://localhost:8000/api/agents
```

### Dependency

```python
# In API endpoints
from main import require_api_key

@router.get("/protected")
async def protected_endpoint(auth: bool = Depends(require_api_key)):
    # Endpoint logic
```

---

## 🌊 Streaming Architecture (SSE)

### Server-Sent Events (SSE)

Automatos AI uses **SSE (Server-Sent Events)** for real-time streaming, **not WebSocket**.

**Advantages**:
- Auto-reconnection
- HTTP/2 multiplexing
- Simpler protocol
- Built-in browser support

### Streaming Endpoints

#### Workflow Execution Streaming

```
GET /api/workflows/executions/{id}/stream
```

**Response Format** (JSON SSE):
```
data: {"type": "stage_start", "stage": 1, "message": "Starting..."}

data: {"type": "subtask_update", "progress": 50}

data: {"type": "execution_log", "message": "Processing..."}

data: {"type": "workflow_complete", "status": "completed"}
```

#### AI SDK Format Streaming

```
GET /api/workflows/executions/{id}/stream/aisdk
```

**Response Format** (AI SDK):
```
d:{"type":"execution_log","message":"..."}

e:{"error":"Something went wrong"}
```

#### Chat Streaming

```
POST /api/chat/stream
Content-Type: application/json

{
  "message": "Hello",
  "session_id": "uuid",
  "stream": true
}
```

**Response** (SSE):
```
data: {"type": "chunk", "content": "Hello"}

data: {"type": "tool_call", "tool": "search_web", "args": {...}}

data: {"type": "tool_result", "result": {...}}

data: {"type": "done", "message_id": "uuid"}
```

### Client Implementation (JavaScript)

```javascript
const eventSource = new EventSource('/api/workflows/executions/123/stream');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};

eventSource.onerror = (error) => {
  console.error('Stream error:', error);
  eventSource.close();
};
```

### Implementation (Backend)

```python
# orchestrator/consumers/chatbot/streaming.py
from core.redis.stream_manager import SSEStreamManager

stream_manager = SSEStreamManager()

async def stream_response(...):
    async for event in stream_manager.listen(execution_id):
        yield f"data: {json.dumps(event)}\n\n"
```

---

## ⚠️ Error Handling

### Standard Error Response

All endpoints return consistent error format:

```json
{
  "detail": "Error message",
  "status_code": 400,
  "error_type": "ValidationError"
}
```

### HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| `200` | OK | Successful GET/PUT |
| `201` | Created | Successful POST |
| `204` | No Content | Successful DELETE |
| `400` | Bad Request | Invalid input |
| `401` | Unauthorized | Missing/invalid API key |
| `404` | Not Found | Resource doesn't exist |
| `409` | Conflict | Resource already exists |
| `422` | Unprocessable Entity | Validation error |
| `500` | Internal Server Error | Server exception |

### Validation Errors (422)

Pydantic validation errors:

```json
{
  "detail": [
    {
      "loc": ["body", "name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## 🔍 API Discovery

### OpenAPI Specification

**JSON**: `GET /openapi.json`  
**Swagger UI**: `GET /docs`  
**ReDoc**: `GET /redoc`

### Health Endpoints

```
GET /health                      # Main health check
GET /api/system/health           # Detailed system health
GET /api/health/endpoints        # Per-endpoint health stats
```

### Root Endpoint

```
GET /                            # API overview + navigation
```

---

## 🚀 Quick Reference

### Most Common Endpoints

```bash
# Check system health
GET /health

# Create and execute workflow
POST /api/workflows
POST /api/workflows/{id}/execute
GET  /api/workflows/executions/{exec_id}/stream  # Watch progress

# Chat with streaming
POST /api/chat/stream

# Upload and search documents
POST /api/documents
POST /api/documents/search

# Manage agents
GET  /api/agents
POST /api/agents
POST /api/agents/{id}/skills
```

---

## 📚 Related Documentation

- **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** - System architecture
- **[Developer Guide](DEVELOPER_GUIDE.md)** - API development
- **[Streaming Guide](STREAMING_GUIDE.md)** - SSE implementation
- **[Authentication](SECURITY.md)** - Security configuration

---

*For complete API documentation, visit `/docs` when running the server*
