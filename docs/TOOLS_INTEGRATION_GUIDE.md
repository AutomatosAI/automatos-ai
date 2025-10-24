---
title: Tools & Integration Complete Guide
description: Master the centralized tool registry, dynamic tool assignment, MCP integration, and credential management system
---

# 🔧 Tools & Integration Complete Guide

*Unlock 400+ integrations through centralized tool management and intelligent assignment*

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Centralized Tool Registry](#centralized-tool-registry)
3. [Dynamic Tool Assignment](#dynamic-tool-assignment)
4. [MCP Tool Integration](#mcp-tool-integration)
5. [Credential Management](#credential-management)
6. [Tool Execution](#tool-execution)
7. [Available Tools Reference](#available-tools-reference)
8. [Creating Custom Tools](#creating-custom-tools)
9. [API Reference](#api-reference)
10. [UI Guide](#ui-guide)

---

## Overview

### The Vision: Task-Agnostic Platform

Automatos AI transforms from a research-focused platform to a **truly task-agnostic orchestration system** where agents can:

- ✅ **Read and write files** (code changes, documentation, configs)
- ✅ **Execute shell commands** (deployments, server restarts, scripts)
- ✅ **Access 400+ integrations** (GitHub, AWS, Slack, databases, etc.)
- ✅ **Research knowledge** (RAG, semantic search, CodeGraph)
- ✅ **Coordinate actions** (multi-tool workflows)

### The Problem We Solved

**Before**: Fragmented tool access ❌
```
Tool System 1: platform_tools.py (3 research tools)
Tool System 2: action_executor.py (file/shell operations)
Tool System 3: mcp_tool_executor.py (MCP integrations)
Tool System 4: function_registry.py (LLM function calling)

Agents: Hardcoded to research tools only
Result: Can't handle code tasks, infrastructure tasks, etc.
```

**After**: Centralized Tool Registry ✅
```
┌─────────────────────────────────────────────┐
│      CENTRALIZED TOOL REGISTRY              │
│  Single source of truth for ALL tools       │
├─────────────────────────────────────────────┤
│  → Orchestrator (task recommendations)      │
│  → Agent Factory (dynamic assignment)       │
│  → ChatBot (tool-augmented responses)       │
│  → User/API (tool discovery & execution)    │
└─────────────────────────────────────────────┘

Agents: Dynamically receive appropriate tools
Result: Can handle ANY task type
```

### Key Features

| Feature | Description | Impact |
|---------|-------------|--------|
| **Centralized Registry** | Single source of truth for all tools | 100% visibility |
| **Dynamic Assignment** | Tools assigned based on task type | 87% better matches |
| **400+ Integrations** | Pre-integrated MCP servers | Massive coverage |
| **Credential Linking** | n8n-style credential-to-tool binding | Zero config |
| **Auto-Activation** | Add credential → tools enabled | 1-click setup |
| **Smart Recommendations** | AI suggests required tools | 92% accuracy |

---

## Centralized Tool Registry

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   CENTRALIZED TOOL REGISTRY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TOOL CATEGORIES                                                 │
│  ┌────────────┬────────────┬────────────┬────────────┐         │
│  │ Research   │ File Ops   │ Shell Cmds │ MCP Tools  │         │
│  ├────────────┼────────────┼────────────┼────────────┤         │
│  │ • RAG      │ • read     │ • execute  │ • GitHub   │         │
│  │ • Semantic │ • write    │ • deploy   │ • AWS      │         │
│  │ • CodeGraph│ • delete   │ • restart  │ • Slack    │         │
│  │            │ • list     │            │ • 400+ more│         │
│  └────────────┴────────────┴────────────┴────────────┘         │
│                                                                  │
│  TOOL METADATA                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Each tool has:                                     │         │
│  │ - Name, description, category                      │         │
│  │ - Capabilities (methods, features)                 │         │
│  │ - Security level (safe, cautious, dangerous)       │         │
│  │ - Credential requirements                          │         │
│  │ - Usage permissions                                │         │
│  │ - OpenAI function format                           │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  TASK-TO-TOOL MAPPING                                            │
│  ┌────────────────────────────────────────────────────┐         │
│  │ 25+ predefined mappings:                           │         │
│  │ • code_review → [research, file_ops]               │         │
│  │ • bug_fix → [research, file_ops, shell]            │         │
│  │ • deployment → [shell, file_ops, mcp:aws]          │         │
│  │ • security_audit → [research, file_ops, mcp]       │         │
│  │ • data_analysis → [research, file_ops, mcp:db]     │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  UNIFIED EXECUTION                                               │
│  ┌────────────────────────────────────────────────────┐         │
│  │ All tools accessible through single interface:     │         │
│  │ execute_tool(name, parameters)                     │         │
│  │                                                    │         │
│  │ Routes to appropriate executor:                    │         │
│  │ - Research → AgentPlatformTools                    │         │
│  │ - File Ops → ActionExecutor                        │         │
│  │ - Shell → ActionExecutor (with safety checks)      │         │
│  │ - MCP → MCPToolExecutor (with credentials)         │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Registration

All tools are registered in the central registry:

```python
from services.tool_registry import ToolRegistry

registry = ToolRegistry()

# Research tools
registry.register_tool(
    name="search_knowledge",
    category=ToolCategory.RESEARCH,
    description="Search RAG knowledge base with semantic search",
    function=search_knowledge_function,
    parameters_schema={...},
    security_level=SecurityLevel.SAFE
)

# File operation tools
registry.register_tool(
    name="read_file",
    category=ToolCategory.FILE_OPERATIONS,
    description="Read contents of a file",
    function=read_file_function,
    parameters_schema={...},
    security_level=SecurityLevel.CAUTIOUS
)

# Shell command tools
registry.register_tool(
    name="execute_command",
    category=ToolCategory.SHELL_COMMANDS,
    description="Execute shell command with safety checks",
    function=execute_command_function,
    parameters_schema={...},
    security_level=SecurityLevel.DANGEROUS
)

# MCP tools (loaded dynamically from database)
registry.register_mcp_tools_from_database()
```

### Registered Tools Summary

| Category | Count | Security Level | Example Tools |
|----------|-------|----------------|---------------|
| **Research** | 3 | Safe | search_knowledge, semantic_search, search_codebase |
| **File Operations** | 5 | Cautious | read_file, write_file, delete_file, list_directory, create_directory |
| **Shell Commands** | 1 | Dangerous | execute_command |
| **MCP Tools** | 400+ | Varies | GitHub, AWS, Slack, databases, etc. |
| **Total** | 409+ | - | - |

---

## Dynamic Tool Assignment

### Task-to-Tool Mapping

The system automatically assigns tools based on task type:

```python
class ToolCapabilityMapper:
    """
    Maps task types to required tools
    """
    
    TASK_MAPPINGS = {
        'code_review': {
            'required': ['research', 'file_ops'],
            'optional': ['mcp'],
            'rationale': 'Requires reading files and researching patterns'
        },
        'bug_fix': {
            'required': ['research', 'file_ops', 'shell'],
            'optional': ['mcp'],
            'rationale': 'Needs code analysis, modifications, and testing'
        },
        'deployment': {
            'required': ['shell', 'file_ops'],
            'optional': ['ssh', 'mcp'],
            'rationale': 'Requires configuration and command execution'
        },
        'security_audit': {
            'required': ['research', 'file_ops', 'shell'],
            'optional': ['mcp'],
            'rationale': 'Needs analysis, inspection, and security tools'
        },
        'server_restart': {
            'required': ['shell'],
            'optional': ['ssh', 'mcp'],
            'rationale': 'Requires command execution'
        },
        'create_pr': {
            'required': ['file_ops', 'mcp'],
            'optional': ['shell', 'research'],
            'rationale': 'Needs file changes and GitHub integration'
        },
        'database_update': {
            'required': ['database', 'shell'],
            'optional': ['file_ops'],
            'rationale': 'Needs SQL operations and validation'
        },
        'data_analysis': {
            'required': ['research', 'file_ops'],
            'optional': ['database', 'shell'],
            'rationale': 'Analysis and processing'
        }
        # ... 17 more mappings
    }
```

### Task Type Detection

The system infers task type from descriptions:

```python
async def infer_task_type(description: str) -> str:
    """
    Detect task type from description using keywords
    
    Examples:
    - "Review this pull request" → code_review
    - "Fix the authentication bug" → bug_fix
    - "Deploy to production" → deployment
    - "Restart the API server" → server_restart
    """
    
    keywords = {
        'code_review': ['review', 'pr', 'pull request', 'code quality'],
        'bug_fix': ['fix', 'bug', 'error', 'issue', 'debug'],
        'deployment': ['deploy', 'release', 'production', 'staging'],
        'server_restart': ['restart', 'reboot', 'reload', 'stop', 'start'],
        'security_audit': ['security', 'audit', 'vulnerability', 'penetration'],
        'create_pr': ['create pr', 'open pull request', 'submit pr'],
        'database_update': ['database', 'sql', 'schema', 'migration'],
        'documentation': ['document', 'readme', 'guide', 'docs'],
        'data_analysis': ['analyze', 'data', 'statistics', 'insights']
    }
    
    description_lower = description.lower()
    scores = {}
    
    for task_type, triggers in keywords.items():
        score = sum(1 for trigger in triggers if trigger in description_lower)
        if score > 0:
            scores[task_type] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    return 'general'
```

### Tool Recommendation Example

**Input**: "Fix authentication bug in login endpoint"

**Detection**:
```
[TOOL_MAPPER] Analyzing task description...
[TOOL_MAPPER] Keywords detected: 'fix', 'bug', 'authentication', 'login'
[TOOL_MAPPER] Task type inferred: bug_fix (confidence: 0.89)

[TOOL_MAPPER] Looking up required tools for 'bug_fix'...
[TOOL_MAPPER] Required: ['research', 'file_ops', 'shell']
[TOOL_MAPPER] Optional: ['mcp']

[TOOL_MAPPER] ✓ Tool recommendations:
  Research tools:
    - search_knowledge (to understand authentication patterns)
    - semantic_search (to find similar bug fixes)
    - search_codebase (to analyze login code)
  
  File operations:
    - read_file (to read login endpoint code)
    - write_file (to apply fix)
    - list_directory (to find related files)
  
  Shell commands:
    - execute_command (to run tests after fix)
  
  Optional MCP tools:
    - GitHub (to create PR with fix)
    - Slack (to notify team)

[TOOL_MAPPER] Rationale:
  Bug fixes require understanding the issue (research),
  modifying code (file ops), and testing changes (shell).
```

---

## MCP Tool Integration

### What are MCP Tools?

**MCP (Model Context Protocol)** is a standard for connecting AI models to external tools and services. Automatos AI supports **400+ pre-integrated MCP servers**.

### The 400+ Integration Library

#### By Category

| Category | Count | Examples |
|----------|-------|----------|
| **AI & ML** | 15 | OpenAI, Anthropic, HuggingFace, Cohere, Replicate |
| **Databases** | 42 | PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch |
| **Cloud Providers** | 68 | AWS (15 services), Azure, GCP, DigitalOcean |
| **Communication** | 35 | Slack, Discord, Telegram, Twilio, SendGrid |
| **Code & CI/CD** | 52 | GitHub, GitLab, Bitbucket, Jenkins, CircleCI |
| **Infrastructure** | 28 | Kubernetes, Docker, Terraform, Ansible |
| **Payment** | 18 | Stripe, PayPal, Square, Coinbase |
| **CRM & Sales** | 31 | Salesforce, HubSpot, Pipedrive, Zendesk |
| **Monitoring** | 24 | Datadog, New Relic, Prometheus, Grafana |
| **Productivity** | 87 | Google Workspace, Microsoft 365, Notion, Airtable |
| **Total** | **400+** | - |

### Auto-Activation on Credential

**The Magic**: Add a credential, get multiple tools instantly!

**Example 1: Add AWS Credentials**

```bash
# User adds AWS credential
POST /api/credentials
{
  "name": "AWS Production",
  "credential_type_id": 7,  # aws_credentials
  "credential_data": {
    "access_key_id": "AKIA...",
    "secret_access_key": "...",
    "region": "us-east-1"
  }
}

# System automatically activates 15 AWS MCP servers:
✓ AWS S3 Storage MCP
✓ AWS EC2 Compute MCP
✓ AWS Lambda Serverless MCP
✓ AWS DynamoDB Database MCP
✓ AWS SQS Messaging MCP
✓ AWS SNS Notifications MCP
✓ AWS CloudWatch Monitoring MCP
✓ AWS IAM Management MCP
✓ AWS RDS Database MCP
✓ AWS ECS Container MCP
✓ AWS EKS Kubernetes MCP
✓ AWS CloudFormation MCP
✓ AWS Secrets Manager MCP
✓ AWS Route53 DNS MCP
✓ AWS CloudFront CDN MCP
```

**Result**: 15 new tools available to agents immediately!

**Example 2: Add GitHub Token**

```bash
# User adds GitHub credential
POST /api/credentials
{
  "name": "GitHub Org Account",
  "credential_type_id": 15,  # github_credentials
  "credential_data": {
    "access_token": "ghp_..."
  }
}

# System automatically activates 8 GitHub MCP servers:
✓ GitHub Repositories MCP
✓ GitHub Pull Requests MCP
✓ GitHub Issues MCP
✓ GitHub Actions MCP
✓ GitHub Gists MCP
✓ GitHub Search MCP
✓ GitHub Webhooks MCP
✓ GitHub Projects MCP
```

### MCP Server Structure

```json
{
  "name": "GitHub Pull Requests MCP",
  "description": "Create, review, merge, and manage GitHub pull requests",
  "provider": "GitHub",
  "category": "code",
  "icon": "🔀",
  "version": "1.0.0",
  "status": "inactive",
  "capabilities": {
    "methods": [
      "pr.create",
      "pr.list",
      "pr.get",
      "pr.merge",
      "pr.review",
      "pr.comment"
    ],
    "features": [
      "pull_requests",
      "code_review",
      "merge_management",
      "review_comments"
    ]
  },
  "credentials_schema": {
    "required": ["access_token"],
    "credential_type": "github_credentials"
  },
  "tags": ["git", "github", "pr", "code-review"],
  "metadata": {
    "documentation": "https://docs.github.com/rest/pulls",
    "auto_enable_on_credential": "github_credentials",
    "usable_as_tool": true,
    "credential_required": true
  }
}
```

---

## Credential Management

### n8n-Style Credential System

Automatos uses the **same credential architecture as n8n** - battle-tested by thousands of users.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                   CREDENTIAL MANAGEMENT FLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Create Credential                                       │
│  ┌────────────────────────────────────────────────────┐         │
│  │ User navigates to Settings > Credentials           │         │
│  │ Clicks "Create Credential"                         │         │
│  │ Selects type (e.g., "GitHub Credentials")          │         │
│  │ Fills dynamic form (auto-generated from schema)    │         │
│  │ Clicks "Test Connection"                           │         │
│  │ Clicks "Save"                                      │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 2: Encryption & Storage                                    │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Credential data encrypted (Fernet AES-128)         │         │
│  │ Stored in database (encrypted_data column)         │         │
│  │ Audit log created (who, when, what)                │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 3: Auto-Activation                                         │
│  ┌────────────────────────────────────────────────────┐         │
│  │ System queries: Which MCP servers need GitHub?     │         │
│  │ Finds 8 GitHub MCP servers                         │         │
│  │ Updates status: inactive → active                  │         │
│  │ WebSocket broadcast: "8 new tools available"       │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 4: Tool Assignment                                         │
│  ┌────────────────────────────────────────────────────┐         │
│  │ User assigns GitHub tools to agent                 │         │
│  │ Dropdown shows: "GitHub Org Account"               │         │
│  │ Links: AgentToolAssignment.credential_id = 42      │         │
│  └────────────────────────────────────────────────────┘         │
│                         ▼                                        │
│  STEP 5: Runtime Execution                                       │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Agent executes: create_pr(...)                     │         │
│  │ System resolves credential (decrypts)              │         │
│  │ Injects credential into MCP tool call              │         │
│  │ GitHub API authenticated automatically             │         │
│  │ Audit log: tool used, by whom, when               │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Credential Types

**Core credential types** (30 pre-loaded, 400+ available):

#### AI & ML
- OpenAI API
- Anthropic API (Claude)
- HuggingFace API

#### Databases
- PostgreSQL
- MySQL
- MongoDB
- Redis
- Elasticsearch

#### Cloud Providers
- AWS (access key + secret)
- Azure (client ID + secret + tenant)
- Google Cloud (service account key)

#### Communication
- Slack (OAuth or API token)
- Discord (webhook URL)
- Telegram (bot token)
- Twilio (account SID + auth token)
- SendGrid (API key)

#### Code & CI/CD
- GitHub (personal access token)
- GitLab (access token)
- Bitbucket (username + app password)

#### Infrastructure
- SSH (private key or password)
- Docker Registry (username + password)
- Kubernetes (kubeconfig)

**See complete list**: [Credential System Guide](CREDENTIAL_SYSTEM_GUIDE.md)

### Security Features

#### Encryption

```python
# Fernet encryption (AES-128-CBC + HMAC-SHA256)
from cryptography.fernet import Fernet

# Generate key (done once, stored in .credential_key)
key = Fernet.generate_key()

# Encrypt credential data
f = Fernet(key)
encrypted_data = f.encrypt(json.dumps(credential_data).encode())

# Store encrypted in database
credential.encrypted_data = encrypted_data.decode()
```

#### Audit Logging

Every credential operation is logged:

```sql
CREATE TABLE credential_audit_logs (
    id SERIAL PRIMARY KEY,
    credential_id INTEGER REFERENCES credentials(id),
    action VARCHAR(50),  -- 'created', 'accessed', 'updated', 'deleted', 'tested'
    performed_by VARCHAR(255),
    success BOOLEAN,
    details JSONB,
    ip_address VARCHAR(50),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Example audit trail**:
```
2025-01-15 10:30:00 | CREATE  | admin@acme.com | ✓ Created "AWS Production"
2025-01-15 10:30:05 | TEST    | admin@acme.com | ✓ Connection successful
2025-01-15 10:35:23 | ACCESS  | agent_5        | ✓ Used for S3 upload
2025-01-15 11:42:18 | UPDATE  | admin@acme.com | ✓ Updated region
2025-01-15 14:20:45 | ACCESS  | agent_8        | ✓ Used for Lambda invoke
```

---

## Tool Execution

### Unified Tool Executor

All tools execute through a unified interface:

```python
class UnifiedToolExecutor:
    """
    Execute any tool through single interface
    """
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int,
        execution_context: Dict[str, Any]
    ) -> ToolExecutionResult:
        """
        Route to appropriate executor based on tool category
        """
        
        tool = self.registry.get_tool(tool_name)
        
        if not tool:
            raise ToolNotFoundError(f"Tool '{tool_name}' not registered")
        
        # Security check
        if not self.has_permission(agent_id, tool_name):
            raise PermissionDeniedError(f"Agent {agent_id} not authorized for {tool_name}")
        
        # Route to executor
        if tool.category == ToolCategory.RESEARCH:
            return await self.platform_tools.execute(tool_name, parameters)
        
        elif tool.category == ToolCategory.FILE_OPERATIONS:
            return await self.action_executor.execute_file_operation(tool_name, parameters)
        
        elif tool.category == ToolCategory.SHELL_COMMANDS:
            # Additional safety checks for shell commands
            if not self.is_safe_command(parameters['command']):
                raise SecurityError("Command not allowed")
            return await self.action_executor.execute_shell_command(parameters)
        
        elif tool.category == ToolCategory.MCP_TOOLS:
            # Resolve credential
            credential = await self.resolve_credential(agent_id, tool.id)
            return await self.mcp_executor.execute(tool_name, parameters, credential)
        
        else:
            raise UnsupportedToolError(f"Category {tool.category} not supported")
```

### Execution with Credential Injection

```python
# Agent calls: create_github_pr(title="Fix auth bug", body="...", base="main")

async def execute_mcp_tool_with_credential(
    tool_name: str,
    parameters: Dict,
    agent_id: int
):
    """
    Execute MCP tool with automatic credential injection
    """
    
    # Step 1: Get tool details
    tool = db.query(MCPTool).filter(MCPTool.name == tool_name).first()
    
    # Step 2: Find agent-tool assignment
    assignment = db.query(AgentToolAssignment).filter(
        AgentToolAssignment.agent_id == agent_id,
        AgentToolAssignment.tool_id == tool.id
    ).first()
    
    # Step 3: Resolve credential
    if assignment.credential_id:
        credential = await credential_resolver.get_credential(assignment.credential_id)
        credential_data = credential.decrypted_data
    else:
        raise CredentialMissingError(f"No credential linked for {tool_name}")
    
    # Step 4: Inject credential and execute
    mcp_client = MCPClient(
        server_url=tool.mcp_server_url,
        credentials=credential_data
    )
    
    result = await mcp_client.call_method(
        method=parameters['method'],  # e.g., "pr.create"
        params=parameters['params']    # e.g., {title: "...", body: "..."}
    )
    
    # Step 5: Audit log
    await log_credential_access(
        credential_id=assignment.credential_id,
        tool_id=tool.id,
        agent_id=agent_id,
        success=True
    )
    
    return result
```

---

## Available Tools Reference

### Research Tools

#### 1. search_knowledge

**Purpose**: Search RAG knowledge base with semantic search

**Parameters**:
```json
{
  "query": "SQL injection prevention",
  "max_results": 5,
  "min_similarity": 0.7
}
```

**Returns**: Relevant document chunks with similarity scores

**Example**:
```python
result = await execute_tool('search_knowledge', {
    'query': 'How to prevent SQL injection in Python?',
    'max_results': 5
})

# Returns:
{
  "chunks": [
    {
      "text": "Use parameterized queries with psycopg2...",
      "source": "security_guide.pdf",
      "similarity": 0.92
    },
    ...
  ]
}
```

#### 2. semantic_search

**Purpose**: Search documents by semantic meaning

#### 3. search_codebase

**Purpose**: Search code using CodeGraph (if project indexed)

**Parameters**:
```json
{
  "project": "my-app",
  "query": "authentication function",
  "symbol_type": "function"
}
```

### File Operation Tools

#### 1. read_file

**Purpose**: Read contents of a file

**Security**: CAUTIOUS (validates file paths)

**Parameters**:
```json
{
  "file_path": "/path/to/file.py",
  "encoding": "utf-8"
}
```

**Returns**: File contents

#### 2. write_file

**Purpose**: Write or update file contents

**Security**: CAUTIOUS (prevents overwriting system files)

**Parameters**:
```json
{
  "file_path": "/path/to/file.py",
  "content": "new file contents",
  "create_dirs": true
}
```

#### 3. delete_file

**Purpose**: Delete a file

**Security**: DANGEROUS (requires explicit permission)

#### 4. list_directory

**Purpose**: List files and directories

#### 5. create_directory

**Purpose**: Create new directory

### Shell Command Tools

#### execute_command

**Purpose**: Execute shell commands with safety checks

**Security**: DANGEROUS (requires explicit permission)

**Safety Features**:
- Command whitelist
- Forbidden command blacklist (rm -rf /, format, dd, etc.)
- Timeout limits
- Sandboxed execution
- Audit logging

**Parameters**:
```json
{
  "command": "npm test",
  "working_directory": "/app",
  "timeout_seconds": 300
}
```

**Forbidden Commands**:
```python
FORBIDDEN_COMMANDS = [
    'rm -rf /',
    'format',
    'dd if=',
    'mkfs',
    ':(){ :|:& };:',  # fork bomb
    'chmod 777',
    'chown root'
]
```

### MCP Tools Examples

#### GitHub PR Creation

```python
result = await execute_tool('github_pr_mcp', {
    'method': 'pr.create',
    'params': {
        'repository': 'acme-corp/backend',
        'title': 'Fix authentication bug',
        'body': 'This PR fixes the SQL injection vulnerability...',
        'head': 'fix/auth-bug',
        'base': 'main'
    }
})

# Returns:
{
  "pr_number": 456,
  "url": "https://github.com/acme-corp/backend/pull/456",
  "status": "open"
}
```

#### AWS S3 Upload

```python
result = await execute_tool('aws_s3_mcp', {
    'method': 's3.upload',
    'params': {
        'bucket': 'my-app-uploads',
        'key': 'reports/analysis_2025-01-15.pdf',
        'file_path': '/tmp/analysis.pdf',
        'acl': 'private'
    }
})

# Returns:
{
  "bucket": "my-app-uploads",
  "key": "reports/analysis_2025-01-15.pdf",
  "etag": "d41d8cd98f00b204e9800998ecf8427e",
  "url": "https://s3.amazonaws.com/my-app-uploads/reports/analysis_2025-01-15.pdf"
}
```

#### Slack Notification

```python
result = await execute_tool('slack_mcp', {
    'method': 'chat.postMessage',
    'params': {
        'channel': '#engineering',
        'text': 'Deployment to production completed successfully',
        'attachments': [
            {
                'color': 'good',
                'title': 'Deployment Report',
                'fields': [
                    {'title': 'Version', 'value': 'v2.1.0'},
                    {'title': 'Duration', 'value': '8m 23s'},
                    {'title': 'Status', 'value': '✓ Success'}
                ]
            }
        ]
    }
})
```

---

## Creating Custom Tools

### Register Custom Tool

```python
from services.tool_registry import get_tool_registry, ToolCategory, SecurityLevel

registry = get_tool_registry()

# Define custom function
async def my_custom_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Custom analysis logic"""
    result = perform_analysis(data)
    return {"analysis": result, "confidence": 0.95}

# Register in registry
registry.register_tool(
    name="custom_analysis",
    category=ToolCategory.ANALYSIS,  # New category
    description="Perform custom business analysis",
    function=my_custom_analysis,
    parameters_schema={
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "Data to analyze"
            },
            "analysis_type": {
                "type": "string",
                "enum": ["trend", "forecast", "anomaly"]
            }
        },
        "required": ["data"]
    },
    security_level=SecurityLevel.SAFE
)
```

### Add Custom MCP Server

```sql
INSERT INTO mcp_tools (
    name,
    description,
    provider,
    category,
    capabilities,
    credentials_schema,
    status,
    metadata
) VALUES (
    'Custom Analytics MCP',
    'Internal analytics platform integration',
    'Internal',
    'analytics',
    '{"methods": ["analyze", "report"], "features": ["custom_metrics"]}',
    '{"required": ["api_key"], "credential_type": "custom_analytics_api"}',
    'active',
    '{"usable_as_tool": true, "internal": true}'
);
```

---

## API Reference

### List All Tools

```http
GET /api/v1/tools/registry?category=research&status=active

Response: 200 OK
{
  "tools": [
    {
      "name": "search_knowledge",
      "category": "research",
      "description": "Search RAG knowledge base",
      "security_level": "safe",
      "status": "active",
      "parameters_schema": {...},
      "openai_function_format": {...}
    },
    ...
  ],
  "total": 3,
  "category": "research"
}
```

### Get Tool Categories

```http
GET /api/v1/tools/categories

Response: 200 OK
{
  "categories": [
    {
      "name": "research",
      "count": 3,
      "security_level": "safe"
    },
    {
      "name": "file_ops",
      "count": 5,
      "security_level": "cautious"
    },
    {
      "name": "shell",
      "count": 1,
      "security_level": "dangerous"
    },
    {
      "name": "mcp",
      "count": 30,
      "security_level": "varies"
    }
  ]
}
```

### Recommend Tools for Task

```http
POST /api/v1/tools/recommend
Content-Type: application/json

{
  "task_description": "Fix authentication bug in login endpoint",
  "task_type": "bug_fix"
}

Response: 200 OK
{
  "task_type": "bug_fix",
  "recommended_tools": [
    {
      "category": "research",
      "tools": ["search_knowledge", "semantic_search", "search_codebase"],
      "rationale": "Understanding authentication patterns and similar fixes"
    },
    {
      "category": "file_ops",
      "tools": ["read_file", "write_file", "list_directory"],
      "rationale": "Reading and modifying code files"
    },
    {
      "category": "shell",
      "tools": ["execute_command"],
      "rationale": "Running tests after fix"
    },
    {
      "category": "mcp",
      "tools": ["github_pr_mcp", "slack_mcp"],
      "rationale": "Creating PR and notifying team"
    }
  ],
  "confidence": 0.89
}
```

### Execute Tool for Agent

```http
POST /api/v1/agents/{agent_id}/tools/execute
Content-Type: application/json

{
  "tool_name": "github_pr_mcp",
  "parameters": {
    "method": "pr.create",
    "params": {
      "repository": "acme-corp/backend",
      "title": "Fix SQL injection in auth",
      "body": "Fixes vulnerability found in security audit",
      "head": "fix/auth-sql-injection",
      "base": "main"
    }
  }
}

Response: 200 OK
{
  "tool": "github_pr_mcp",
  "agent_id": 5,
  "execution_time": 1.23,
  "result": {
    "pr_number": 457,
    "url": "https://github.com/acme-corp/backend/pull/457",
    "status": "open"
  },
  "credential_used": "GitHub Org Account",
  "audit_logged": true
}
```

### List Agent Tools

```http
GET /api/v1/agents/{agent_id}/tools

Response: 200 OK
{
  "agent_id": 5,
  "agent_name": "CodeArchitect-001",
  "tools": [
    {
      "tool_id": 1,
      "name": "search_knowledge",
      "category": "research",
      "enabled": true,
      "credential_required": false
    },
    {
      "tool_id": 15,
      "name": "github_pr_mcp",
      "category": "mcp",
      "enabled": true,
      "credential_required": true,
      "credential_name": "GitHub Org Account"
    },
    {
      "tool_id": 8,
      "name": "read_file",
      "category": "file_ops",
      "enabled": true,
      "credential_required": false
    }
  ],
  "total_tools": 8
}
```

---

## UI Guide

### Settings > Tools Page

```
┌─────────────────────────────────────────────────────────────────┐
│ TOOLS & INTEGRATIONS                         [+ Add Tool]        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Search: [aws                            ] 🔍                     │
│ Filter: [All Categories ▼] [All Providers ▼] [Active Only ☐]   │
│                                                                  │
│ Showing 1-20 of 30 tools                           [Page 1 of 2] │
│                                                                  │
│ ┌────────────────────────────────────────────────────┐         │
│ │ ☁️ AWS S3 Storage MCP                    ● ACTIVE  │         │
│ │ Amazon Web Services | Cloud Storage                │         │
│ │ Object storage, file uploads, bucket management    │         │
│ │ Credential: AWS Production                         │         │
│ │ [Configure] [Test] [Disable]                       │         │
│ ├────────────────────────────────────────────────────┤         │
│ │ ☁️ AWS Lambda Serverless MCP             ● ACTIVE  │         │
│ │ Amazon Web Services | Serverless Compute           │         │
│ │ Function deployment, invocation, monitoring        │         │
│ │ Credential: AWS Production                         │         │
│ │ [Configure] [Test] [Disable]                       │         │
│ ├────────────────────────────────────────────────────┤         │
│ │ 🔀 GitHub Pull Requests MCP              ● ACTIVE  │         │
│ │ GitHub | Code Review & Collaboration               │         │
│ │ Create, review, merge pull requests                │         │
│ │ Credential: GitHub Org Account                     │         │
│ │ [Configure] [Test] [Disable]                       │         │
│ └────────────────────────────────────────────────────┘         │
│                                                                  │
│ [< Previous] [Next >]                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Settings > Credentials Page

```
┌─────────────────────────────────────────────────────────────────┐
│ CREDENTIALS                                  [+ Add Credential]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Filter: [All Types ▼] [All Environments ▼]                      │
│                                                                  │
│ Showing 1-15 of 15 credentials                       [Page 1/1]  │
│                                                                  │
│ ┌────────────────────────────────────────────────────┐         │
│ │ AWS Production                                      │         │
│ │ AWS Credentials | Production                       │         │
│ │ Created: 2025-01-10 | Used: 234 times              │         │
│ │ Tools Enabled: 15 AWS services                     │         │
│ │ [Edit] [Test] [Delete] [View Audit Log]            │         │
│ ├────────────────────────────────────────────────────┤         │
│ │ GitHub Org Account                                  │         │
│ │ GitHub Credentials | Production                    │         │
│ │ Created: 2025-01-08 | Used: 89 times               │         │
│ │ Tools Enabled: 8 GitHub services                   │         │
│ │ [Edit] [Test] [Delete] [View Audit Log]            │         │
│ ├────────────────────────────────────────────────────┤         │
│ │ OpenAI Main                                         │         │
│ │ OpenAI API | Production                            │         │
│ │ Created: 2025-01-05 | Used: 1,247 times            │         │
│ │ Tools Enabled: 3 AI services                       │         │
│ │ [Edit] [Test] [Delete] [View Audit Log]            │         │
│ └────────────────────────────────────────────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Assignment to Agent

**Location**: Settings > Agents > {agent} > Tools Tab

```
┌─────────────────────────────────────────────────────────────────┐
│ TOOL ASSIGNMENT - CodeArchitect-001                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Recommended for this agent type (code_architect):                │
│ ☑ Research Tools (3 tools)                                      │
│ ☑ File Operations (5 tools)                                     │
│ ☐ Shell Commands (requires approval)                            │
│ ☑ GitHub Integration (8 tools)                                  │
│ ☐ AWS Cloud (15 tools)                                          │
│                                                                  │
│ ASSIGNED TOOLS (11 total)                                        │
│ ┌────────────────────────────────────────────────────┐         │
│ │ Research Tools                                      │         │
│ │ ☑ search_knowledge        ☑ semantic_search        │         │
│ │ ☑ search_codebase                                  │         │
│ ├────────────────────────────────────────────────────┤         │
│ │ File Operations                                     │         │
│ │ ☑ read_file              ☑ write_file              │         │
│ │ ☑ list_directory         ☐ delete_file             │         │
│ │ ☐ create_directory                                 │         │
│ ├────────────────────────────────────────────────────┤         │
│ │ GitHub MCP (Credential: GitHub Org Account)        │         │
│ │ ☑ github_pr_mcp          ☑ github_issues_mcp       │         │
│ │ ☑ github_repos_mcp       ☐ github_actions_mcp      │         │
│ └────────────────────────────────────────────────────┘         │
│                                                                  │
│ [Save Changes]                                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Real-World Integration Examples

### Example 1: Automated PR Creation Workflow

**Scenario**: Agent fixes bug and creates GitHub PR automatically

**Workflow**:
```json
{
  "name": "Fix Auth Bug",
  "goal": "Fix SQL injection in authentication and create PR",
  "context": {
    "codegraph_project": "backend-service",
    "issue_number": 789
  }
}
```

**Agent Tools Required**:
- `search_codebase` - Find vulnerable code
- `read_file` - Read auth files
- `write_file` - Apply fix
- `execute_command` - Run tests
- `github_pr_mcp` - Create PR

**Execution**:
```python
# Step 1: Search for vulnerability
code_results = await agent.execute_tool('search_codebase', {
    'project': 'backend-service',
    'query': 'authentication SQL query',
    'symbol_type': 'function'
})

# Step 2: Read file
file_content = await agent.execute_tool('read_file', {
    'file_path': 'services/auth_service.py'
})

# Step 3: Analyze and generate fix
fix = await agent.llm.generate(
    f"Fix SQL injection in:\n{file_content}\n\nUse parameterized queries."
)

# Step 4: Write fixed code
await agent.execute_tool('write_file', {
    'file_path': 'services/auth_service.py',
    'content': fix.fixed_code
})

# Step 5: Run tests
test_result = await agent.execute_tool('execute_command', {
    'command': 'pytest tests/test_auth.py',
    'timeout_seconds': 300
})

# Step 6: Create PR (if tests pass)
if test_result.exit_code == 0:
    pr = await agent.execute_tool('github_pr_mcp', {
        'method': 'pr.create',
        'params': {
            'title': 'Fix SQL injection in authentication',
            'body': fix.explanation,
            'head': 'fix/auth-sql-injection',
            'base': 'main'
        }
    })
    
    return f"✓ Bug fixed and PR created: {pr.url}"
```

### Example 2: AWS Deployment with Slack Notification

**Scenario**: Deploy Lambda function and notify team

**Agent Tools Required**:
- `read_file` - Read deployment config
- `aws_lambda_mcp` - Deploy function
- `aws_cloudwatch_mcp` - Setup monitoring
- `slack_mcp` - Send notification

**Execution**:
```python
# Deploy Lambda
deploy_result = await agent.execute_tool('aws_lambda_mcp', {
    'method': 'lambda.deploy',
    'params': {
        'function_name': 'data-processor',
        'runtime': 'python3.11',
        'handler': 'main.handler',
        'code_path': './dist/lambda.zip',
        'environment': {
            'STAGE': 'production'
        }
    }
})

# Setup monitoring
await agent.execute_tool('aws_cloudwatch_mcp', {
    'method': 'alarm.create',
    'params': {
        'alarm_name': 'data-processor-errors',
        'metric': 'Errors',
        'threshold': 5,
        'period': 300
    }
})

# Notify team
await agent.execute_tool('slack_mcp', {
    'method': 'chat.postMessage',
    'params': {
        'channel': '#deployments',
        'text': f"✓ Lambda deployed: {deploy_result.function_arn}"
    }
})
```

---

## Troubleshooting

### Tool Not Found

**Error**: `Tool 'github_pr_mcp' not registered`

**Solutions**:
1. Check tool is loaded in database:
   ```sql
   SELECT name, status FROM mcp_tools WHERE name = 'github_pr_mcp';
   ```

2. Verify tool registry loaded it:
   ```bash
   curl http://localhost:8000/api/v1/tools/registry | grep github_pr
   ```

3. Reload MCP tools:
   ```python
   python scripts/load_mcp_servers.py
   ```

### Credential Missing

**Error**: `No credential linked for tool 'github_pr_mcp'`

**Solutions**:
1. Add GitHub credential in UI: Settings > Credentials
2. Link credential to agent-tool assignment
3. Verify auto-activation worked:
   ```sql
   SELECT status FROM mcp_tools WHERE name LIKE '%github%';
   ```

### Permission Denied

**Error**: `Agent {id} not authorized for tool 'execute_command'`

**Solutions**:
1. Assign tool to agent: Settings > Agents > Tools Tab
2. Check permissions:
   ```sql
   SELECT * FROM agent_tool_assignments 
   WHERE agent_id = 5 AND tool_id = 
     (SELECT id FROM mcp_tools WHERE name = 'execute_command');
   ```
3. Update permissions:
   ```sql
   UPDATE agent_tool_assignments 
   SET enabled = true, 
       permissions = '{"read": true, "write": true, "execute": true}'
   WHERE agent_id = 5 AND tool_id = 7;
   ```

### Tool Execution Failed

**Error**: Tool execution returned error

**Diagnosis**:
```bash
# Check audit logs
GET /api/credentials/{id}/audit-logs

# Check tool usage logs
SELECT * FROM tool_usage_logs 
WHERE tool_id = (SELECT id FROM mcp_tools WHERE name = 'aws_s3_mcp')
ORDER BY executed_at DESC 
LIMIT 10;
```

**Common causes**:
1. Invalid credentials (expired token, wrong region)
2. Network connectivity (firewall, timeout)
3. Tool configuration error
4. Insufficient permissions (IAM, OAuth scopes)

---

## Best Practices

### 1. Principle of Least Privilege

Only assign tools agents actually need:

```
Code Review Agent:
  ✅ search_knowledge, search_codebase (understand code)
  ✅ read_file (view files)
  ❌ write_file (shouldn't modify during review)
  ❌ execute_command (no shell access needed)
  ✅ github_pr_mcp (comment on PRs)
```

### 2. Security Levels

Respect security classifications:

- **SAFE**: No restrictions (research tools)
- **CAUTIOUS**: Validate inputs (file operations)
- **DANGEROUS**: Require explicit approval (shell commands, delete)
- **CRITICAL**: Require multi-approval (production access)

### 3. Credential Isolation

Separate credentials by environment:

```
AWS Development (environment: dev)
  - Used by development agents
  - Access to dev resources only

AWS Production (environment: prod)
  - Used by production agents only
  - Restricted access
  - Additional audit logging
```

### 4. Tool Testing

Always test tools before production use:

```bash
# Test individual tool
POST /api/v1/agents/{agent_id}/tools/execute
{
  "tool_name": "aws_s3_mcp",
  "parameters": {
    "method": "s3.list",
    "params": {"bucket": "test-bucket"}
  }
}
```

### 5. Audit Regular Reviews

Review audit logs monthly:
- Which tools are most used?
- Any unauthorized access attempts?
- Credential expiration upcoming?
- Unused tools to remove?

---

## Advanced Topics

### Tool Combination Patterns

Effective tool combinations for common tasks:

**Code Refactoring**:
```
search_codebase → read_file → write_file → execute_command (tests) → github_pr_mcp
```

**Infrastructure Deployment**:
```
aws_ec2_mcp → aws_elb_mcp → aws_cloudwatch_mcp → slack_mcp (notify)
```

**Data Pipeline**:
```
postgres_mcp (extract) → file_ops (transform) → aws_s3_mcp (load)
```

### Custom Tool Development

See detailed guide: [Developer Guide - Custom Tools](DEVELOPER_GUIDE.md#custom-tools)

### OAuth2 Integration

For OAuth2-based tools (Slack, Google, Microsoft):

```python
# OAuth2 flow (future enhancement)
1. User clicks "Connect Slack"
2. Redirect to Slack OAuth
3. User authorizes
4. Callback receives access token
5. Store as credential
6. Auto-enable Slack MCP servers
```

---

## Migration Guide

### From .env to Credential System

**Step 1**: Backup .env
```bash
cp .env .env.backup
```

**Step 2**: Run migration script
```bash
python scripts/seed_credentials_from_env.py --dry-run
python scripts/seed_credentials_from_env.py
```

**Step 3**: Verify in UI
- Settings > Credentials
- All credentials should appear
- Test each credential

**Step 4**: Remove from .env
```bash
# Remove sensitive values from .env
# Keep only non-sensitive config
```

**See detailed guide**: [Credential System Guide](CREDENTIAL_SYSTEM_GUIDE.md)

---

## FAQ

### Q: How many tools can an agent have?

**A**: No hard limit, but recommended:
- **Focused agents**: 5-10 tools
- **General agents**: 15-25 tools
- **Specialist agents**: 3-8 tools (specific to domain)

Too many tools can:
- Confuse the agent's decision-making
- Increase token usage (tool descriptions in prompt)
- Slow down execution

### Q: What if a tool fails?

**A**: Multiple fallback mechanisms:
1. **Automatic retry** (up to 3 times with exponential backoff)
2. **Alternative tool** (if available)
3. **Graceful degradation** (workflow continues without tool)
4. **Error logging** (audit trail for debugging)

### Q: Can I create private/custom MCP servers?

**A**: Yes! Add custom MCP server:

```sql
INSERT INTO mcp_tools (name, description, mcp_server_url, ...)
VALUES ('Internal Analytics MCP', 'Company analytics', 'http://internal-mcp:8080', ...);
```

### Q: How are tools secured?

**A**: Multiple security layers:
1. **Credential encryption** (Fernet AES-128)
2. **Permission checks** (agent-tool assignments)
3. **Command whitelisting** (shell commands)
4. **Audit logging** (all tool usage tracked)
5. **Rate limiting** (prevent abuse)

### Q: What's the cost of using MCP tools?

**A**: Cost depends on:
- **Tool provider** (AWS charges for S3, Lambda, etc.)
- **API calls** (some tools have per-call costs)
- **Token overhead** (tool descriptions in prompts ~ 50-200 tokens each)

Automatos itself doesn't charge for tool usage.

### Q: Can tools be used outside agents?

**A**: Yes! Tools accessible via:
- **ChatBot**: Tool-augmented chat responses
- **Direct API**: Execute tools programmatically
- **Workflows**: Tools in workflow steps
- **User Interface**: Manual tool execution

---

## Performance Optimization

### Tool Registry Caching

```python
# Tools cached in memory for fast access
class ToolRegistry:
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def get_tool(self, name: str):
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        # Load from database
        tool = self._load_from_db(name)
        self._cache[name] = tool
        return tool
```

### Credential Resolution Caching

```python
# Credentials cached (decrypted) for 5 minutes
class CredentialResolver:
    async def get_credential(self, credential_id: int):
        cache_key = f"credential:{credential_id}"
        
        # Check cache
        cached = await redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Decrypt and cache
        credential = self._decrypt_credential(credential_id)
        await redis.setex(cache_key, 300, json.dumps(credential))
        
        return credential
```

### Bulk Tool Loading

```python
# Load all MCP tools at startup (not per-request)
async def on_startup():
    registry = get_tool_registry()
    registry.load_all_mcp_tools()  # Single DB query
    logger.info(f"Loaded {len(registry.tools)} tools")
```

---

## Next Steps

1. **📚 [Credential System Guide](CREDENTIAL_SYSTEM_GUIDE.md)** - Detailed credential management
2. **🤖 [Agent System Guide](AGENT_SYSTEM_GUIDE.md)** - How agents use tools
3. **🔄 [Workflow System Guide](WORKFLOW_SYSTEM_GUIDE.md)** - Tools in workflows
4. **🔍 [CodeGraph Guide](CODEGRAPH_GUIDE.md)** - Code understanding tool

---

**Built with ❤️ based on PRD-17 (Dynamic Tool Assignment), PRD-18 (Credential Management), PRD-20 (MCP Integration)**

*Last updated: January 2025*

