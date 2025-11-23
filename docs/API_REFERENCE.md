---
title: API Reference
description: Complete REST API documentation with 373 endpoints for agent management, workflows, memory, tools, and orchestration
---

# 📡 API Reference

*Complete API documentation for Automatos AI Platform - 373 endpoints and counting*

---

## 🎯 Overview

The Automatos AI API provides **373+ endpoints** across all platform features:

| Category | Endpoints | Base Path |
|----------|-----------|-----------|
| **Agents** | ~45 | `/api/v1/agents/*` |
| **Workflows** | ~38 | `/api/v1/workflows/*` |
| **Memory & Knowledge** | ~52 | `/api/v1/memory/*`, `/api/v1/knowledge/*` |
| **Tools & Credentials** | ~67 | `/api/v1/tools/*`, `/api/credentials/*` |
| **CodeGraph** | ~41 | `/api/codegraph/*` |
| **Documents & RAG** | ~35 | `/api/documents/*`, `/api/rag/*` |
| **Playbooks & Patterns** | ~24 | `/api/playbooks/*`, `/api/patterns/*` |
| **Analytics & Monitoring** | ~33 | `/api/analytics/*`, `/api/dashboard/*` |
| **System & Admin** | ~38 | `/api/system/*`, `/api/admin/*` |

---

## 📚 Auto-Generated API Documentation

**Automatos AI uses FastAPI**, which automatically generates comprehensive, interactive API documentation.

### Access Interactive API Docs

#### Option 1: Swagger UI (Recommended)

```
{API_URL}/docs
```

> **Note**: Replace `{API_URL}` with your API server URL (e.g., `https://api.automatos.app` for production, `http://localhost:8000` for local)

**Features**:
- ✅ Interactive "Try it out" functionality
- ✅ All 373 endpoints with full documentation
- ✅ Request/response examples
- ✅ Authentication testing
- ✅ Always up-to-date (auto-generated from code)

#### Option 2: ReDoc

```
{API_URL}/redoc
```

**Features**:
- ✅ Clean, readable documentation
- ✅ Better for reading/browsing
- ✅ Searchable
- ✅ Downloadable as PDF

#### Option 3: OpenAPI JSON

```
{API_URL}/openapi.json
```

**Use for**:
- Importing into Postman
- Generating client SDKs
- Integration with GitBook
- API testing tools

---

## 🔧 Generating OpenAPI Spec

### For GitBook Integration

Run the generation script:

```bash
cd docs
./generate_openapi.sh
```

This will:
1. Fetch latest OpenAPI spec from running server
2. Save to `docs/openapi.json`
3. Display summary (endpoints count, version, etc.)

### Manual Generation

```bash
# Option 1: From running server (replace $API_URL with your server URL)
curl -s ${API_URL:-https://your-api-url.com}/openapi.json -o docs/openapi.json

# Option 2: From local server
curl -s http://localhost:8000/openapi.json -o docs/openapi.json

# Option 3: From code (requires Python environment)
cd orchestrator
python3 -c "
from main import app
import json
with open('../docs/openapi.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
"
```

---

## 📖 API Quick Reference

### Base URLs

```
Production:  {API_URL} (e.g., https://api.automatos.app)
Local:       http://localhost:8000
```

> **Note**: Set `API_URL` environment variable or replace `{API_URL}` with your server URL in all examples

### Authentication

All endpoints require API key in header:

```bash
curl -H "X-API-Key: your_api_key_here" \
     ${API_URL:-https://your-api-url.com}/api/v1/agents
```

---

## 🎯 Core Endpoint Groups

### Agent Management

```http
GET    /api/v1/agents                    # List all agents
POST   /api/v1/agents/create-specialized # Create specialized agent
GET    /api/v1/agents/{id}               # Get agent details
PUT    /api/v1/agents/{id}               # Update agent
DELETE /api/v1/agents/{id}               # Delete agent
POST   /api/v1/agents/{id}/execute       # Execute agent task
GET    /api/v1/agents/{id}/performance   # Get performance metrics
PUT    /api/v1/agents/{id}/model-config  # Update model configuration
```

**See detailed guide**: [Agent System Guide](AGENT_SYSTEM_GUIDE.md)

### Workflow Orchestration

```http
GET    /api/v1/workflows                     # List workflows
POST   /api/v1/workflows                     # Create workflow
GET    /api/v1/workflows/{id}                # Get workflow details
POST   /api/v1/workflows/{id}/execute        # Execute workflow
GET    /api/v1/workflows/executions/{id}     # Get execution status
GET    /api/v1/workflows/executions/{id}/logs # Get execution logs
```

**See detailed guide**: [Workflow System Guide](WORKFLOW_SYSTEM_GUIDE.md)

### Memory & Knowledge

```http
POST   /api/v1/memory/store                  # Store memory
POST   /api/v1/memory/retrieve               # Retrieve memories
POST   /api/v1/memory/consolidate            # Consolidate memories
GET    /api/v1/memory/stats/{agent_id}       # Memory statistics
POST   /api/v1/knowledge/query               # Query knowledge graph
```

**See detailed guide**: [Memory & Knowledge Guide](MEMORY_KNOWLEDGE_GUIDE.md)

### Tools & Credentials

```http
GET    /api/v1/tools/registry                # List all tools
POST   /api/v1/tools/recommend               # Recommend tools for task
POST   /api/v1/agents/{id}/tools/execute     # Execute tool
GET    /api/credentials                      # List credentials
POST   /api/credentials                      # Create credential
POST   /api/credentials/{id}/test            # Test credential
```

**See detailed guide**: [Tools & Integration Guide](TOOLS_INTEGRATION_GUIDE.md)

### CodeGraph

```http
POST   /api/codegraph/index                  # Index codebase
GET    /api/codegraph/projects               # List projects
POST   /api/codegraph/search                 # Semantic code search
GET    /api/codegraph/symbols                # Get code symbols
```

**See detailed guide**: [CodeGraph Guide](CODEGRAPH_GUIDE.md)

### Documents & RAG

```http
POST   /api/documents/upload                 # Upload document
GET    /api/documents                        # List documents
POST   /api/documents/search                 # Semantic search
POST   /api/rag/query                        # RAG retrieval
GET    /api/documents/stats                  # Document statistics
```

**See detailed guide**: [Context Engineering Guide](CONTEXT_ENGINEERING_GUIDE.md)

### Playbooks & Patterns

```http
GET    /api/playbooks                        # List discovered playbooks
GET    /api/playbooks/{id}                   # Get playbook details
POST   /api/playbooks/mine                   # Mine new patterns
POST   /api/playbooks/{id}/create-workflow   # Create workflow from playbook
```

**See detailed guide**: [Playbooks Guide](PLAYBOOKS_GUIDE.md)

---

## 🔗 GitBook Integration

### Update book.json

Your `book.json` is configured for GitBook. To integrate the OpenAPI spec:

**Option 1**: Link to auto-generated docs

```json
{
  "variables": {
    "apiDocs": "{API_URL}/docs",
    "apiSpec": "{API_URL}/openapi.json"
  }
}
```

> **Note**: Replace `{API_URL}` with your actual API server URL

**Option 2**: GitBook OpenAPI Plugin

```bash
# Install GitBook OpenAPI plugin
npm install -g gitbook-plugin-openapi

# Add to book.json plugins
"plugins": [
  "openapi"
],
"pluginsConfig": {
  "openapi": {
    "spec": "./openapi.json"
  }
}
```

---

## 📊 API Statistics

**Total Endpoints**: 373+ (and growing)

**By HTTP Method**:
- GET: ~187 endpoints (read operations)
- POST: ~142 endpoints (create/execute)
- PUT: ~28 endpoints (update)
- DELETE: ~16 endpoints (delete)

**Response Time** (P95):
- Simple queries: <100ms
- Agent execution: 2-8s
- Workflow execution: 5-60s

**Success Rate**: 96.8% (over last 30 days)

---

## 🚀 Quick Start Examples

### Create and Execute Agent

```bash
# Set your API URL (or replace $API_URL in commands below)
export API_URL="https://your-api-url.com"  # or http://localhost:8000 for local

# 1. Create specialized agent
curl -X POST ${API_URL}/api/v1/agents/create-specialized \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{
    "name": "CodeReviewer",
    "type": "code_architect",
    "model_config": {
      "provider": "openai",
      "model_id": "gpt-4-turbo-preview"
    },
    "skills": ["code_analysis", "security_audit"]
  }'

# Response: {"id": 42, "status": "active", ...}

# 2. Execute task
curl -X POST ${API_URL}/api/v1/agents/42/execute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{
    "task": {
      "description": "Review this authentication code for security issues",
      "context": {"file_path": "auth.py"}
    }
  }'
```

### Execute Workflow

```bash
curl -X POST ${API_URL}/api/v1/workflows/15/execute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{
    "input_data": {},
    "execution_options": {
      "enable_communication": true,
      "use_memory": true
    }
  }'
```

---

## 📝 API Versioning

Current version: **v1**

All endpoints are prefixed with `/api/v1/` for future compatibility.

When breaking changes are introduced, a new version will be created (`/api/v2/`) while v1 remains available for 6 months.

---

## ⚡ Best Practices

### 1. Use Swagger UI for Exploration

Don't manually craft API requests - use the interactive Swagger UI at `/docs` to:
- Explore available endpoints
- Test requests interactively
- See real-time request/response examples
- Generate curl commands automatically

### 2. Generate Client SDKs

Use the OpenAPI spec to generate client libraries:

```bash
# Generate Python client
openapi-generator-cli generate \
  -i ${API_URL}/openapi.json \
  -g python \
  -o ./python-client

# Generate TypeScript client  
openapi-generator-cli generate \
  -i ${API_URL}/openapi.json \
  -g typescript-axios \
  -o ./ts-client
```

### 3. Keep OpenAPI Spec Updated

Re-generate the spec after deploying new features:

```bash
cd docs
./generate_openapi.sh
git add openapi.json
git commit -m "Update OpenAPI spec"
```

---

## 🔍 Finding Specific Endpoints

### Use Swagger UI Search

Navigate to `{API_URL}/docs` and use the search box to find endpoints by:
- Functionality (e.g., "create agent")
- Resource type (e.g., "workflow")
- HTTP method (e.g., "POST")

### Use OpenAPI JSON

```bash
# Download spec
curl -s ${API_URL}/openapi.json > openapi.json

# Search for specific endpoints
jq '.paths | keys[] | select(contains("agent"))' openapi.json

# Get endpoint details
jq '.paths["/api/v1/agents"]' openapi.json
```

---

## 📚 Detailed API Guides

For detailed usage of specific API categories, see our comprehensive guides:

- **[Agent System Guide](AGENT_SYSTEM_GUIDE.md)** - Agent management APIs
- **[Workflow System Guide](WORKFLOW_SYSTEM_GUIDE.md)** - Workflow execution APIs
- **[Tools & Integration Guide](TOOLS_INTEGRATION_GUIDE.md)** - Tool and credential APIs
- **[Memory & Knowledge Guide](MEMORY_KNOWLEDGE_GUIDE.md)** - Memory and knowledge APIs
- **[Context Engineering Guide](CONTEXT_ENGINEERING_GUIDE.md)** - RAG and document APIs
- **[Playbooks Guide](PLAYBOOKS_GUIDE.md)** - Pattern discovery APIs

---

## ❓ FAQ

### Q: Why isn't there a manual API reference like api.md?

**A**: Because FastAPI auto-generates perfect, always-up-to-date documentation at `/docs`. Maintaining a manual reference would be:
- ❌ Always outdated (you have 373 endpoints!)
- ❌ Duplicate effort
- ❌ Error-prone

The Swagger UI is superior because it's:
- ✅ Auto-generated from code
- ✅ Always current
- ✅ Interactive
- ✅ Complete with schemas

### Q: How do I integrate with GitBook?

**A**: Two approaches:

1. **Link to Swagger** (recommended):
   - Add link in SUMMARY.md pointing to `/docs`
   - Users access live, interactive docs

2. **Import OpenAPI spec**:
   - Run `./generate_openapi.sh` to create `openapi.json`
   - Use GitBook OpenAPI plugin to render spec

### Q: Can I export Postman collection?

**A**: Yes!

```bash
# Download OpenAPI spec
curl ${API_URL}/openapi.json > automatos-api.json

# Import in Postman:
# 1. Open Postman
# 2. File > Import
# 3. Select automatos-api.json
# 4. All 373 endpoints imported!
```

---

## 🚀 Next Steps

1. **📖 [Explore Swagger UI]({API_URL}/docs)** - Interactive API docs
2. **🤖 [Agent APIs](AGENT_SYSTEM_GUIDE.md#api-reference)** - Agent management
3. **🔄 [Workflow APIs](WORKFLOW_SYSTEM_GUIDE.md#api-reference)** - Workflow execution
4. **🔧 [Tool APIs](TOOLS_INTEGRATION_GUIDE.md#api-reference)** - Tool integration

---

**Built with ❤️ powered by FastAPI's auto-generated documentation**

*Always up-to-date with current codebase*

