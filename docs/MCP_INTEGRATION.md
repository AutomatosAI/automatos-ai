---
title: MCP Integration Guide
description: Complete guide to Model Context Protocol integration, IDE extensions, and headless automation for Automatos AI
---

# 🔌 MCP Integration Guide

*Seamless integration with IDEs and development tools via Model Context Protocol*

---

## 📖 Table of Contents

1. [What is MCP?](#what-is-mcp)
2. [MCP Server](#mcp-server)
3. [IDE Integrations](#ide-integrations)
4. [Available Tools](#available-tools)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)

---

## What is MCP?

**Model Context Protocol (MCP)** is a standard protocol for connecting AI models to external tools and data sources. Automatos AI implements MCP for:

- ✅ **IDE Integration**: Use Automatos from Cursor, VSCode, etc.
- ✅ **Headless Automation**: Programmatic access to platform
- ✅ **Tool Integration**: 400+ pre-integrated MCP servers
- ✅ **Standardization**: Industry-standard protocol

---

## MCP Server

### Starting the MCP Server

```bash
# Start MCP server (default port 8002)
cd automatos-ai/orchestrator
python -m mcp_server.main

# With custom configuration
MCP_SERVER_PORT=8003 \
MCP_API_TOKEN=your-secure-token \
python -m mcp_server.main
```

### Configuration

```python
# mcp_server/config.py
class MCPConfig:
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    
    # Authentication
    API_TOKEN: str = os.getenv("MCP_API_TOKEN")
    
    # Orchestrator connection
    ORCHESTRATOR_URL: str = "http://localhost:8000"
    ORCHESTRATOR_API_KEY: str = os.getenv("API_KEY")
    
    # Features
    ENABLE_WORKFLOWS: bool = True
    ENABLE_AGENTS: bool = True
    ENABLE_TOOLS: bool = True
```

---

## IDE Integrations

### Cursor IDE

**Setup** (1 minute):

1. Open Cursor Settings
2. Go to "Model Context Protocol"
3. Add server:
   ```json
   {
     "name": "Automatos AI",
     "type": "http",
     "url": "http://localhost:8002",
     "headers": {
       "Authorization": "Bearer your-mcp-token"
     }
   }
   ```
4. Save and restart Cursor

**Usage**:
```
# In Cursor chat:
@automatos create a workflow to deploy this repository

@automatos review this code for security issues

@automatos index this codebase for semantic search
```

### VSCode Integration

**Install Extension**:
```bash
# From VSCode Marketplace
code --install-extension automatos-ai.mcp-integration
```

**Configure**:
```json
// settings.json
{
  "automatos.mcp.serverUrl": "http://localhost:8002",
  "automatos.mcp.apiToken": "your-mcp-token",
  "automatos.mcp.autoStart": true
}
```

### JetBrains IDEs (IntelliJ, PyCharm, etc.)

**Install Plugin**:
1. File > Settings > Plugins
2. Search "Automatos AI MCP"
3. Install and restart

**Configure**:
- Tools > Automatos AI > Settings
- Enter MCP server URL and token

---

## Available Tools

### MCP Tool Categories

**400+ tools available via MCP**:

| Category | Count | Examples |
|----------|-------|----------|
| **Code & Version Control** | 52 | GitHub, GitLab, Bitbucket |
| **Cloud Infrastructure** | 68 | AWS, Azure, GCP |
| **Databases** | 42 | PostgreSQL, MySQL, MongoDB, Redis |
| **Communication** | 35 | Slack, Discord, Telegram |
| **CI/CD** | 28 | Jenkins, CircleCI, GitHub Actions |
| **Monitoring** | 24 | Datadog, Prometheus, Grafana |
| **Productivity** | 87 | Google Workspace, Notion, Airtable |
| **AI & ML** | 15 | OpenAI, Anthropic, HuggingFace |
| **Other** | 49 | Stripe, Twilio, SendGrid, etc. |

**See complete list**: [Tools & Integration Guide](TOOLS_INTEGRATION_GUIDE.md)

---

## Usage Examples

### Example 1: Deploy from Cursor

```typescript
// In Cursor chat:
User: @automatos deploy this Next.js app to production

// MCP server receives request
{
  "action": "create_workflow",
  "context": {
    "repository": "/current/working/directory",
    "task": "Deploy Next.js app to production"
  }
}

// Automatos responds:
Response: "Creating deployment workflow...
  ✓ Analyzed repository (Next.js detected)
  ✓ Created workflow: Production Deploy
  ✓ Assigned agents: InfrastructureManager, SecurityChecker
  ✓ Executing...
  
  Results available at: {APP_URL}/workflows/123  # Replace {APP_URL} with your frontend URL
"
```

### Example 2: Code Review via API

```python
from mcp_server.client import MCPClient

async with MCPClient("http://localhost:8002", token="your-token") as client:
    # Create code review workflow
    result = await client.execute_action(
        action="code_review",
        context={
            "repository": "https://github.com/acme/app",
            "pr_number": 456,
            "focus": ["security", "performance"]
        }
    )
    
    print(f"Review complete: {result['quality_score']}")
    print(f"Issues found: {len(result['issues'])}")
```

### Example 3: Index Codebase

```python
# Index current project
result = await client.execute_action(
    action="index_codebase",
    context={
        "path": "/path/to/project",
        "languages": ["python", "javascript"],
        "project_name": "my-app"
    }
)

# Response:
{
  "project_id": "proj_123",
  "symbols_found": 1247,
  "files_indexed": 89,
  "indexing_time": 12.3
}
```

---

## MCP Protocol Details

### Request Format

```json
{
  "protocol": "mcp",
  "version": "1.0",
  "action": "execute_workflow",
  "parameters": {
    "workflow_id": 42,
    "input_data": {}
  },
  "authentication": {
    "type": "bearer",
    "token": "your-mcp-token"
  }
}
```

### Response Format

```json
{
  "protocol": "mcp",
  "version": "1.0",
  "status": "success",
  "data": {
    "execution_id": 157,
    "workflow_id": 42,
    "status": "running"
  },
  "metadata": {
    "execution_time": 0.234,
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

---

## Advanced Features

### WebSocket Streaming

```python
# Stream workflow execution updates
async with client.stream_workflow(workflow_id=42) as stream:
    async for event in stream:
        if event.type == "subtask_completed":
            print(f"✓ {event.data.description}")
        elif event.type == "workflow_completed":
            print(f"✓ Workflow done: {event.data.overall_score}")
```

### Batch Operations

```python
# Execute multiple actions
results = await client.batch_execute([
    {"action": "create_agent", "params": {...}},
    {"action": "create_workflow", "params": {...}},
    {"action": "execute_workflow", "params": {...}}
])
```

---

## Configuration Reference

### MCP Server Environment Variables

```bash
# Server
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8002
MCP_SERVER_WORKERS=4

# Authentication
MCP_API_TOKEN=your-secure-mcp-token

# Backend connection
ORCHESTRATOR_URL=http://localhost:8000
ORCHESTRATOR_API_KEY=your-api-key

# Features
MCP_ENABLE_STREAMING=true
MCP_MAX_CONNECTIONS=100
MCP_REQUEST_TIMEOUT=300

# Logging
MCP_LOG_LEVEL=INFO
MCP_LOG_FORMAT=json
```

---

## Troubleshooting

### MCP Server Won't Start

```bash
# Check port not in use
lsof -i:8002

# Check logs
tail -f mcp_server.log

# Verify configuration
python -c "from mcp_server.config import MCPConfig; print(MCPConfig())"
```

### IDE Can't Connect

```bash
# Test MCP server health
curl http://localhost:8002/health

# Test with authentication
curl -H "Authorization: Bearer your-token" \
     http://localhost:8002/health
```

### Authentication Errors

```bash
# Verify token
echo $MCP_API_TOKEN

# Generate new token
openssl rand -hex 32
```

---

## Next Steps

1. **🔧 [Tools & Integration Guide](TOOLS_INTEGRATION_GUIDE.md)** - Complete MCP tool registry
2. **🔐 [Credential System](CREDENTIAL_SYSTEM_GUIDE.md)** - Credential management
3. **🤖 [Agent System](AGENT_SYSTEM_GUIDE.md)** - Using MCP tools with agents
4. **🔄 [Workflow System](WORKFLOW_SYSTEM_GUIDE.md)** - MCP in workflows

---

**Built with ❤️ for seamless IDE integration**

*Last updated: January 2025*

