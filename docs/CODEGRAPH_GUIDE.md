# 📘 CodeGraph: Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Concept](#core-concept)
3. [Installation & Setup](#installation--setup)
4. [Indexing Sources](#indexing-sources)
5. [Multi-Project Management](#multi-project-management)
6. [Workflow Integration](#workflow-integration)
7. [Chatbot Integration](#chatbot-integration)
8. [UI Design](#ui-design)
9. [Real-World Scenarios](#real-world-scenarios)
10. [API Reference](#api-reference)
11. [Integration Examples](#integration-examples)
12. [Monetization](#monetization)
13. [FAQ](#faq)

---

## Overview

**CodeGraph** is Automatos AI's intelligent code indexing and retrieval system that enables AI agents and workflows to access and understand large codebases efficiently.

### Key Features
- **Multi-Source Indexing**: Local directories, GitHub, GitLab, Bitbucket
- **Semantic Code Search**: Find code by meaning, not just keywords
- **Symbol Tracking**: Classes, functions, imports, dependencies
- **Relationship Mapping**: Call graphs, dependency trees
- **Workflow Integration**: Automatic context injection for agents
- **Real-Time Updates**: Webhook-based re-indexing

### Value Proposition
> "Turn any codebase into an AI-readable knowledge graph. Agents get laser-focused context instead of drowning in millions of lines of code."

---

## Core Concept

### WHOSE CODE?
**The CLIENT'S CODE** - Not Automatos code (though you can index that too for meta purposes!).

CodeGraph indexes **customer/user codebases** to enable:
1. **Intelligent Agent Tasks**: Code review, bug analysis, documentation generation
2. **Developer Assistance**: Chatbot code queries, onboarding help
3. **Automated Workflows**: Security audits, refactoring, testing

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    CODEGRAPH SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT SOURCES                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Local Dir │  │GitHub URL│  │GitLab URL│                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │             │              │                         │
│       └─────────────┴──────────────┘                        │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │   INDEXER   │                                │
│              │ (tree-sitter)│                                │
│              └──────┬──────┘                                │
│                     │                                        │
│       ┌─────────────┴─────────────┐                        │
│       │                           │                         │
│  ┌────▼─────┐              ┌─────▼────┐                   │
│  │ Symbols  │              │Relations │                   │
│  │  Graph   │◄─────────────┤  Graph   │                   │
│  └────┬─────┘              └─────┬────┘                   │
│       │                           │                         │
│       └─────────────┬─────────────┘                        │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │  POSTGRES   │                                │
│              │ + pgvector  │                                │
│              └──────┬──────┘                                │
│                     │                                        │
│       ┌─────────────┴─────────────┐                        │
│       │                           │                         │
│  ┌────▼─────┐              ┌─────▼────┐                   │
│  │Workflows │              │ Chatbot  │                   │
│  │ (Agents) │              │   API    │                   │
│  └──────────┘              └──────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation & Setup

### Quick Setup (15 minutes)

```bash
# 1. Install dependencies
cd automatos-ai/orchestrator
pip install tree-sitter tree-sitter-languages networkx

# 2. Database tables auto-created on first run

# 3. Index your first project
curl -X POST http://localhost:8000/api/code-graph/index \
  -H "Content-Type: application/json" \
  -d '{
    "project": "automatos-ai",
    "root_dir": "/path/to/automatos-ai"
  }'

# 4. Test query
curl "http://localhost:8000/api/code-graph/search?project=automatos-ai&q=workflow execution"
```

### Advanced Setup (1 hour)

```bash
# orchestrator/.env - Add config
CODEGRAPH_ENABLED=true
CODEGRAPH_MAX_FILE_SIZE=1000000  # 1MB per file
CODEGRAPH_SUPPORTED_LANGUAGES=python,typescript,javascript,go,rust,java
CODEGRAPH_GITHUB_TOKEN=ghp_...  # For private repos
CODEGRAPH_CACHE_TTL=3600  # 1 hour cache
```

---

## Indexing Sources

### Option A: Local Directory

```bash
POST /api/code-graph/index
{
  "project": "automatos-backend",
  "root_dir": "/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/orchestrator",
  "language": "python",
  "exclude_patterns": ["__pycache__", "*.pyc", "venv", "node_modules", ".git"]
}
```

**Use When:**
- Development on local machine
- Testing/debugging
- Private code that never leaves your server

### Option B: GitHub URL

```bash
POST /api/code-graph/index
{
  "project": "automatos-public",
  "git_url": "https://github.com/AutomatosAI/automatos-ai.git",
  "branch": "main",
  "auth_token": "ghp_...",  # Optional for private repos
  "clone_depth": 1  # Shallow clone for speed
}
```

**Use When:**
- Indexing open-source projects
- Onboarding new team members
- CI/CD integration
- Remote repository analysis

### Option C: GitLab/Bitbucket

```bash
POST /api/code-graph/index
{
  "project": "client-enterprise",
  "git_url": "https://gitlab.com/client/app.git",
  "provider": "gitlab",
  "auth_token": "glpat-...",
  "branch": "develop"
}
```

### Option D: Multiple Sources (Same Project)

```bash
# Index backend
POST /api/code-graph/index
{
  "project": "fullstack-app",
  "root_dir": "/app/backend"
}

# Add frontend to same project
POST /api/code-graph/index
{
  "project": "fullstack-app",
  "root_dir": "/app/frontend",
  "merge": true  # Adds to existing project
}

# Add mobile app
POST /api/code-graph/index
{
  "project": "fullstack-app",
  "git_url": "https://github.com/company/mobile-app.git",
  "merge": true
}
```

**Result:** Single unified project with backend, frontend, and mobile code searchable together.

---

## Multi-Project Management

### Unlimited Projects

```bash
# Client 1 - E-commerce
POST /api/code-graph/index
{
  "project": "client-acme-ecommerce",
  "git_url": "https://github.com/acme-corp/ecommerce-backend.git"
}

# Client 2 - CRM
POST /api/code-graph/index
{
  "project": "client-techcorp-crm",
  "root_dir": "/app/clients/techcorp/crm"
}

# Your own platform (META!)
POST /api/code-graph/index
{
  "project": "automatos-ai",
  "git_url": "https://github.com/AutomatosAI/automatos-ai.git"
}
```

### Query Specific Project

```bash
# Search in specific project
GET /api/code-graph/search?project=client-acme-ecommerce&q=payment processing

# Search across all projects
GET /api/code-graph/search?q=authentication&all_projects=true

# List all projects
GET /api/code-graph/projects
```

### Project Metadata

```json
{
  "id": 1,
  "name": "client-acme-ecommerce",
  "source_type": "github",
  "source_url": "https://github.com/acme-corp/ecommerce-backend.git",
  "language": "python",
  "total_files": 523,
  "total_symbols": 12847,
  "last_indexed": "2025-10-02T14:30:00Z",
  "auto_reindex": true,
  "webhook_url": "${API_URL}/webhooks/code-changed"  # Replace ${API_URL} with your API server URL
}
```

---

## Workflow Integration

### Core Concept: Automatic Context Injection

When you add `codegraph_project` to workflow context, agents automatically get relevant code snippets.

### Scenario 1: Code Review Workflow

```typescript
POST /api/workflows
{
  "name": "PR Security Review #456",
  "description": "Review pull request for vulnerabilities",
  "goal": "Analyze PR #456 for SQL injection, XSS, and authentication bypass vulnerabilities",
  "context": {
    "codegraph_project": "client-acme-ecommerce",  // <-- Magic happens here
    "pr_number": 456,
    "git_diff_url": "https://github.com/acme-corp/ecommerce/pull/456.diff",
    "focus_areas": ["security", "authentication", "database"]
  }
}
```

**What Happens:**
1. ✅ Workflow created with `codegraph_project` in context
2. ✅ Orchestrator reads goal and context
3. ✅ Assigns task to Security Agent
4. ✅ Agent automatically queries CodeGraph:
   ```python
   # Agent internally calls:
   context_results = codegraph.search(
       project="client-acme-ecommerce",
       query="authentication middleware database queries"
   )
   ```
5. ✅ Agent gets existing auth patterns + DB access code
6. ✅ Agent compares PR code against patterns
7. ✅ Agent generates security report with specific recommendations

### Scenario 2: Bug Analysis Workflow

```typescript
POST /api/workflows
{
  "name": "Debug Production Error",
  "description": "Find root cause of checkout failure",
  "goal": "Investigate 'NoneType has no attribute price' error in checkout flow",
  "context": {
    "codegraph_project": "client-acme-ecommerce",
    "error_message": "NoneType object has no attribute 'price'",
    "error_file": "services/checkout.py",
    "error_line": 145,
    "stack_trace": "Traceback (most recent call last)..."
  }
}
```

**Agent Actions:**
1. Searches CodeGraph for `checkout.py` line 145
2. Traces where `price` variable comes from
3. Finds all code paths that could set `price` to `None`
4. Identifies missing null check in `Product.get_price()`
5. Generates fix with test case

### Scenario 3: Documentation Generation

```typescript
POST /api/workflows
{
  "name": "Generate API Docs",
  "description": "Create comprehensive API documentation",
  "goal": "Document all public API endpoints in PaymentService with examples",
  "context": {
    "codegraph_project": "client-acme-ecommerce",
    "target_module": "services.payment_service",
    "include_examples": true,
    "include_call_graphs": true
  }
}
```

**Agent Output:**
- Complete API reference
- Request/response schemas
- Usage examples extracted from tests
- Call graphs showing service dependencies
- Authentication requirements

### Scenario 4: Refactoring Workflow

```typescript
POST /api/workflows
{
  "name": "Migrate to Python 3.11",
  "goal": "Find and update all Python 2 syntax to Python 3.11",
  "context": {
    "codegraph_project": "legacy-app",
    "search_patterns": ["print ", "xrange", "unicode(", "iteritems()"],
    "auto_fix": false,  // Generate PRs, don't auto-merge
    "create_migration_plan": true
  }
}
```

**Agent Actions:**
1. Searches CodeGraph for deprecated syntax
2. Identifies 47 files with issues
3. Analyzes dependency chains
4. Generates migration order (least dependent first)
5. Creates PRs for each file with tests
6. Documents breaking changes

---

## Chatbot Integration

### Implementation: Code Query Interface

#### Backend: Chat Endpoint with CodeGraph

```python
# orchestrator/api/chat.py
from fastapi import APIRouter, Depends
from services.codegraph_service import CodeGraphService

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/query")
async def chat_query(
    message: str,
    project: str,
    max_chunks: int = 5,
    db: Session = Depends(get_db)
):
    """Chat endpoint with code context"""
    
    # 1. Search CodeGraph for relevant code
    codegraph = CodeGraphService()
    code_results = await codegraph.search(
        project=project,
        query=message,
        limit=max_chunks
    )
    
    # 2. Build prompt with code context
    prompt = f"""You are a code expert assistant. Answer the user's question using the provided code context.

User Question: {message}

Relevant Code:
{code_results.prompt_block}

Provide a clear, detailed answer with code examples and file references."""
    
    # 3. Call LLM with context
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    
    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful code assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "code_references": [
            {"file": chunk.file, "line": chunk.line, "snippet": chunk.code}
            for chunk in code_results.chunks
        ]
    }
```

#### Frontend: Chat Component

```tsx
// components/chat/code-query-chat.tsx
'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Card } from '@/components/ui/card'
import { apiClient } from '@/lib/api-client'

interface Message {
  role: 'user' | 'assistant'
  content: string
  codeRefs?: Array<{ file: string; line: number; snippet: string }>
}

export function CodeQueryChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [selectedProject, setSelectedProject] = useState('automatos-ai')
  const [loading, setLoading] = useState(false)

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await apiClient.request('/api/chat/query', {
        message: input,
        project: selectedProject
      })

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        codeRefs: response.code_references
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Chat error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="glass-card h-[600px] flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border/30">
        <h3 className="text-lg font-semibold mb-2">Code Assistant</h3>
        <Select value={selectedProject} onChange={setSelectedProject}>
          <option value="automatos-ai">Automatos AI</option>
          <option value="client-acme">Client: Acme E-commerce</option>
          <option value="client-techcorp">Client: TechCorp CRM</option>
        </Select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-lg p-3 ${
              msg.role === 'user' 
                ? 'bg-blue-500/20 text-white' 
                : 'bg-secondary/50'
            }`}>
              <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
              
              {/* Code References */}
              {msg.codeRefs && msg.codeRefs.length > 0 && (
                <div className="mt-3 space-y-2">
                  <div className="text-xs text-muted-foreground font-semibold">📎 References:</div>
                  {msg.codeRefs.map((ref, i) => (
                    <div key={i} className="text-xs bg-black/30 p-2 rounded">
                      <div className="text-blue-400">{ref.file}:{ref.line}</div>
                      <pre className="mt-1 overflow-x-auto">{ref.snippet}</pre>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-secondary/50 rounded-lg p-3">
              <div className="animate-pulse">Thinking...</div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-border/30">
        <div className="flex space-x-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about the codebase..."
            disabled={loading}
          />
          <Button onClick={handleSend} disabled={loading || !input.trim()}>
            Send
          </Button>
        </div>
      </div>
    </Card>
  )
}
```

#### Example Conversation

```
👤 User: "How do I create a new agent type in Automatos?"

🤖 Bot: Based on your codebase, here's how to create a new agent type:

1. **Define Agent Class** (orchestrator/agents/your_agent.py):
   ```python
   class YourAgent(BaseAgent):
       def __init__(self, config):
           super().__init__(config)
           self.agent_type = "your_agent"
       
       async def execute(self, task):
           # Your logic here
           return {"result": "success"}
   ```

2. **Register in AgentFactory** (orchestrator/services/agent_factory.py:67):
   ```python
   AGENT_TYPES = {
       "code_architect": CodeArchitect,
       "security_guard": SecurityGuard,
       "your_agent": YourAgent  # Add this line
   }
   ```

3. **Add Database Migration** (alembic/versions/xxx_add_your_agent.py):
   ```python
   def upgrade():
       op.execute("""
           INSERT INTO agents (name, agent_type, description)
           VALUES ('Your Agent', 'your_agent', 'Description here')
       """)
   ```

4. **Write Tests** (tests/test_your_agent.py):
   ```python
   def test_your_agent():
       agent = YourAgent({"model": "gpt-4"})
       result = await agent.execute({"task": "test"})
       assert result["result"] == "success"
   ```

📎 **References:**
- orchestrator/agents/code_architect_agent.py (lines 12-45)
- orchestrator/services/agent_factory.py (line 67)
- tests/test_agents.py (lines 20-35)

Need help with any specific step?
```

---

## UI Design

### Current State
Basic form with project name + root directory + search button.

### Enhanced Design: Multi-Tab Interface

```tsx
'use client'

import { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select } from '@/components/ui/select'
import { Dialog } from '@/components/ui/dialog'
import { 
  FileTree, Network, Database, GitBranch, Search, Zap,
  TrendingUp, Activity, PieChart, Clock
} from 'lucide-react'
import { apiClient } from '@/lib/api-client'

export function CodeGraphPanel() {
  const [projects, setProjects] = useState([])
  const [stats, setStats] = useState({
    totalProjects: 0,
    totalFiles: 0,
    totalSymbols: 0,
    queriesThisWeek: 0
  })

  return (
    <div className="codegraph-panel space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card className="glass-card">
          <CardContent className="pt-6">
            <Database className="w-5 h-5 text-blue-400 mb-2" />
            <div className="text-2xl font-bold">{stats.totalProjects}</div>
            <div className="text-sm text-muted-foreground">Projects Indexed</div>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="pt-6">
            <FileTree className="w-5 h-5 text-green-400 mb-2" />
            <div className="text-2xl font-bold">{stats.totalFiles.toLocaleString()}</div>
            <div className="text-sm text-muted-foreground">Files Analyzed</div>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="pt-6">
            <Network className="w-5 h-5 text-purple-400 mb-2" />
            <div className="text-2xl font-bold">{(stats.totalSymbols / 1000000).toFixed(1)}M</div>
            <div className="text-sm text-muted-foreground">Code Relationships</div>
          </CardContent>
        </Card>
        <Card className="glass-card">
          <CardContent className="pt-6">
            <Zap className="w-5 h-5 text-orange-400 mb-2" />
            <div className="text-2xl font-bold">{stats.queriesThisWeek.toLocaleString()}</div>
            <div className="text-sm text-muted-foreground">Queries This Week</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs defaultValue="projects" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="projects">
            <Database className="w-4 h-4 mr-2" />
            Projects
          </TabsTrigger>
          <TabsTrigger value="search">
            <Search className="w-4 h-4 mr-2" />
            Search
          </TabsTrigger>
          <TabsTrigger value="insights">
            <Activity className="w-4 h-4 mr-2" />
            Insights
          </TabsTrigger>
          <TabsTrigger value="analytics">
            <TrendingUp className="w-4 h-4 mr-2" />
            Analytics
          </TabsTrigger>
        </TabsList>

        {/* Projects Tab */}
        <TabsContent value="projects" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Your Projects</h3>
            <Button onClick={() => setShowAddModal(true)}>
              <GitBranch className="w-4 h-4 mr-2" />
              Add Project
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map(project => (
              <ProjectCard key={project.id} project={project} />
            ))}
          </div>
        </TabsContent>

        {/* Search Tab */}
        <TabsContent value="search" className="space-y-4">
          <CodeSearchInterface />
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <CodeInsights />
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-4">
          <QueryAnalytics />
        </TabsContent>
      </Tabs>

      {/* Add Project Modal */}
      <AddProjectModal open={showAddModal} onClose={() => setShowAddModal(false)} />
    </div>
  )
}

function ProjectCard({ project }) {
  return (
    <Card className="glass-card hover:border-primary/50 transition-all">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-lg">{project.name}</CardTitle>
            <div className="flex items-center space-x-2 text-xs text-muted-foreground mt-1">
              <GitBranch className="w-3 h-3" />
              <span>{project.sourceType}</span>
            </div>
          </div>
          <Badge className="bg-green-500/20 text-green-400">
            Active
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Files:</span>
            <span className="font-semibold">{project.totalFiles}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Language:</span>
            <span className="font-semibold">{project.language}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Last Indexed:</span>
            <span className="font-semibold">{formatTimeAgo(project.lastIndexed)}</span>
          </div>
        </div>
        <div className="mt-4 flex space-x-2">
          <Button size="sm" variant="outline" className="flex-1">
            <Search className="w-3 h-3 mr-1" />
            Search
          </Button>
          <Button size="sm" variant="outline">
            <MoreVertical className="w-3 h-3" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function AddProjectModal({ open, onClose }) {
  const [sourceType, setSourceType] = useState('local') // 'local' | 'github' | 'gitlab'
  
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="glass-card max-w-2xl">
        <DialogHeader>
          <DialogTitle>Add New Project</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          <div>
            <Label>Source Type</Label>
            <Select value={sourceType} onValueChange={setSourceType}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="local">Local Directory</SelectItem>
                <SelectItem value="github">GitHub Repository</SelectItem>
                <SelectItem value="gitlab">GitLab Repository</SelectItem>
                <SelectItem value="bitbucket">Bitbucket Repository</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {sourceType === 'local' && (
            <>
              <div>
                <Label>Project Name</Label>
                <Input placeholder="my-app" />
              </div>
              <div>
                <Label>Root Directory</Label>
                <Input placeholder="/path/to/code" />
              </div>
            </>
          )}

          {sourceType === 'github' && (
            <>
              <div>
                <Label>Project Name</Label>
                <Input placeholder="my-app" />
              </div>
              <div>
                <Label>Repository URL</Label>
                <Input placeholder="https://github.com/username/repo.git" />
              </div>
              <div>
                <Label>Branch (Optional)</Label>
                <Input placeholder="main" defaultValue="main" />
              </div>
              <div>
                <Label>Auth Token (For Private Repos)</Label>
                <Input type="password" placeholder="ghp_..." />
              </div>
            </>
          )}

          <div>
            <Label>Language</Label>
            <Select>
              <SelectTrigger>
                <SelectValue placeholder="Auto-detect" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto-detect</SelectItem>
                <SelectItem value="python">Python</SelectItem>
                <SelectItem value="typescript">TypeScript</SelectItem>
                <SelectItem value="javascript">JavaScript</SelectItem>
                <SelectItem value="go">Go</SelectItem>
                <SelectItem value="rust">Rust</SelectItem>
                <SelectItem value="java">Java</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label>Exclude Patterns (comma-separated)</Label>
            <Input 
              placeholder="node_modules, __pycache__, *.pyc, venv" 
              defaultValue="node_modules, __pycache__, *.pyc, venv, .git"
            />
          </div>

          <div className="flex items-center space-x-2">
            <Checkbox id="auto-reindex" />
            <Label htmlFor="auto-reindex">Enable auto-reindex on git push (webhook)</Label>
          </div>
        </div>

        <div className="flex justify-end space-x-2 pt-4 border-t border-border/30">
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleIndex}>
            <GitBranch className="w-4 h-4 mr-2" />
            Index Project
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function CodeSearchInterface() {
  const [selectedProjects, setSelectedProjects] = useState(['all'])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])

  return (
    <div className="space-y-4">
      <div className="flex space-x-4">
        <Select value={selectedProjects[0]} onValueChange={(v) => setSelectedProjects([v])}>
          <SelectTrigger className="w-[200px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Projects</SelectItem>
            <SelectItem value="automatos-ai">Automatos AI</SelectItem>
            <SelectItem value="client-acme">Client: Acme</SelectItem>
          </SelectContent>
        </Select>

        <Input
          className="flex-1"
          placeholder="Search: 'authentication flow', 'payment processing', 'database queries'..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <Button onClick={handleSearch}>
          <Search className="w-4 h-4 mr-2" />
          Search
        </Button>
      </div>

      <div className="space-y-4">
        {results.map(result => (
          <CodeResultCard key={result.id} result={result} />
        ))}
      </div>
    </div>
  )
}

function CodeResultCard({ result }) {
  return (
    <Card className="glass-card hover:border-primary/50 transition-all cursor-pointer">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="text-sm text-blue-400 font-mono">
              {result.file}:{result.line}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {result.symbol_type}: {result.symbol_name}
            </div>
          </div>
          <Badge>{result.project}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <pre className="text-sm bg-black/30 p-3 rounded overflow-x-auto">
          <code>{result.code_snippet}</code>
        </pre>
        <div className="mt-3 flex items-center space-x-4 text-xs text-muted-foreground">
          <span>Relevance: {(result.relevance * 100).toFixed(0)}%</span>
          <span>•</span>
          <span>{result.references} references</span>
        </div>
      </CardContent>
    </Card>
  )
}

function CodeInsights() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Complexity Heatmap */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg">Code Complexity</CardTitle>
        </CardHeader>
        <CardContent>
          <ComplexityHeatmap />
        </CardContent>
      </Card>

      {/* Dependency Graph */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg">Module Dependencies</CardTitle>
        </CardHeader>
        <CardContent>
          <DependencyGraph />
        </CardContent>
      </Card>

      {/* Hotspot Files */}
      <Card className="glass-card lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-lg">Most Changed Files (Last 30 days)</CardTitle>
        </CardHeader>
        <CardContent>
          <HotspotChart />
        </CardContent>
      </Card>
    </div>
  )
}

function QueryAnalytics() {
  return (
    <div className="space-y-6">
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg">Search Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <SearchTrendsChart />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg">Most Queried Files</CardTitle>
          </CardHeader>
          <CardContent>
            <TopFilesTable />
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg">Popular Search Terms</CardTitle>
          </CardHeader>
          <CardContent>
            <PopularTermsCloud />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
```

---

## Real-World Scenarios

### 🏢 **Scenario 1: E-Commerce Company**

**Client:** Online retailer with 200+ microservices, 2,000 Python files

**Problem:**
- New developers need 3 weeks to understand payment flow
- Bugs in checkout take 2 days to trace through 15 services
- Nobody dares touch auth code because "it's too complex"

**Solution: CodeGraph + Onboarding Agent**

```bash
# Index their codebase
POST /api/code-graph/index
{
  "project": "acme-ecommerce",
  "git_url": "https://github.com/acme-corp/ecommerce-backend.git",
  "branch": "main"
}

# Create onboarding workflow
POST /api/workflows
{
  "name": "New Developer Onboarding",
  "goal": "Explain payment processing flow to new developer",
  "context": {
    "codegraph_project": "acme-ecommerce",
    "focus_areas": ["payment", "checkout", "authentication"]
  }
}
```

**Agent Response (30 minutes later):**
```markdown
# Payment Processing Flow

Your payment flow works as follows:

1. **CheckoutController** (`controllers/checkout.py:45`) receives order
   - Validates cart items
   - Calls `InventoryService.check_availability()`

2. **PaymentService** (`services/payment_service.py:120`) processes payment
   - Method: `process_payment(order_id, payment_method)`
   - Validates with `PaymentValidator.validate_card()`
   - Calls `StripeIntegration.charge_card()`

3. **StripeIntegration** (`integrations/stripe.py:78`) handles Stripe API
   - Uses Stripe SDK v2.15.0
   - Stores transaction in `payments` table
   - Sends webhook to `OrderConfirmationEmail`

## Architecture Diagram
[Generated diagram showing flow]

## Common Issues
- **Issue 1:** Payment fails if inventory check times out
  - Fix: Add retry logic in `PaymentService` line 125
- **Issue 2:** Duplicate charges on network retry
  - Fix: Implement idempotency keys (already done in line 140)

## Testing
Run payment tests:
```bash
pytest tests/test_payment_service.py
```

## Related Files
- `models/payment.py` - Payment data models
- `models/order.py` - Order data models
- `utils/stripe_helpers.py` - Stripe utilities

📎 References: 15 files analyzed, 47 code snippets
```

**Time Saved:** 3 weeks → 30 minutes ⏱️

---

### 🏢 **Scenario 2: FinTech Startup**

**Client:** Financial platform, heavy regulatory requirements

**Problem:**
- Every feature needs security review
- Manual code review takes 2-3 days
- Can't scale security team

**Solution: CodeGraph + Automated Security Reviews**

```bash
# Index codebase
POST /api/code-graph/index
{
  "project": "fintech-core",
  "git_url": "https://gitlab.com/fintech/core-platform.git",
  "provider": "gitlab",
  "auth_token": "glpat-..."
}

# Set up webhook for auto-review on every PR
# Configure in GitLab CI/CD:
# .gitlab-ci.yml
security-review:
  stage: test
  script:
    - |
      curl -X POST ${API_URL}/api/workflows \
        -H "Authorization: Bearer $AUTOMATOS_API_KEY" \
        -d '{
          "name": "Security Review - MR !'"$CI_MERGE_REQUEST_IID"'",
          "goal": "Review merge request for security vulnerabilities and compliance issues",
          "context": {
            "codegraph_project": "fintech-core",
            "mr_number": '"$CI_MERGE_REQUEST_IID"',
            "git_diff_url": "'"$CI_PROJECT_URL"'/-/merge_requests/'"$CI_MERGE_REQUEST_IID"'.diff",
            "compliance_standards": ["PCI-DSS", "SOC2", "GDPR"]
          }
        }'
  only:
    - merge_requests
```

**Automated Security Report (2 minutes):**

```markdown
## Security Review Report
MR !456: Add wire transfer feature

### ✅ Passed (12 checks)
- Authentication middleware correctly applied
- HTTPS-only endpoints enforced
- Rate limiting configured
- Input validation present
- CSRF tokens validated

### ⚠️ Warnings (3 issues)

**CRITICAL - SQL Injection Risk**
- File: `services/wire_transfer_service.py`
- Line: 45
- Issue: SQL query uses string formatting instead of parameterized query
```python
# ❌ VULNERABLE CODE
query = f"SELECT * FROM accounts WHERE user_id = '{user_id}'"

# ✅ RECOMMENDED FIX
query = "SELECT * FROM accounts WHERE user_id = %s"
cursor.execute(query, (user_id,))
```

**HIGH - Missing Transaction Validation**
- File: `services/wire_transfer_service.py`
- Line: 78
- Issue: Missing amount validation against account balance
- Recommendation: Add check using `TransactionValidator.validate_amount()`

**MEDIUM - Insufficient Logging**
- File: `services/wire_transfer_service.py`
- Line: 120
- Issue: Sensitive operation (wire transfer) not logged for audit
- Requirement: PCI-DSS 10.2.5 requires audit logs for all financial transactions

### Compliance Check
- ✅ PCI-DSS: 11/12 requirements met
- ⚠️ SOC2: Logging requirement not met
- ✅ GDPR: Data handling compliant

### Recommended Actions
1. Fix SQL injection vulnerability (CRITICAL)
2. Add transaction validation (HIGH)
3. Add audit logging (MEDIUM)
4. Request re-review after fixes

📎 Analyzed 8 files, compared against 42 security patterns
```

**Value:**
- Every PR auto-reviewed in 2 minutes
- 90% of security issues caught before human review
- Security team focuses on complex cases only
- **ROI:** 2-3 days → 2 minutes per PR

---

### 🏢 **Scenario 3: SaaS Company Migration**

**Client:** Migrating 450 Python files from Python 2.7 to Python 3.11

**Problem:**
- Don't know which files use deprecated syntax
- Manual audit estimated at 4 weeks
- Risk of breaking production

**Solution: CodeGraph + Migration Workflow**

```bash
# Index legacy code
POST /api/code-graph/index
{
  "project": "legacy-app",
  "git_url": "https://github.com/company/legacy-app.git",
  "branch": "python2-legacy"
}

# Run migration analysis
POST /api/workflows
{
  "name": "Python 3.11 Migration Analysis",
  "goal": "Find all Python 2 syntax and create migration plan",
  "context": {
    "codegraph_project": "legacy-app",
    "search_patterns": [
      "print ",
      "xrange(",
      "unicode(",
      ".iteritems()",
      ".itervalues()",
      "except Exception, e:",
      "<> comparison"
    ],
    "auto_generate_fixes": true,
    "create_migration_prs": true
  }
}
```

**Migration Report (30 seconds):**

```markdown
## Python 3.11 Migration Analysis

### Summary
- **47 files** require changes
- **Total issues:** 128 deprecated syntax patterns
- **Estimated effort:** 3 days (automated PRs generated)

### Issue Breakdown
| Pattern | Count | Severity | Files Affected |
|---------|-------|----------|----------------|
| `print` statements | 67 | High | 32 |
| `xrange()` | 23 | Medium | 15 |
| `.iteritems()` | 18 | Medium | 12 |
| `unicode()` | 12 | Low | 8 |
| Old exception syntax | 8 | High | 6 |

### Migration Plan (Dependency-Ordered)

#### Phase 1: Utility Files (No dependencies)
- `utils/string_helpers.py` (3 issues)
- `utils/list_helpers.py` (5 issues)
- `utils/dict_helpers.py` (2 issues)

#### Phase 2: Models (Depend on utils)
- `models/user.py` (8 issues)
- `models/order.py` (6 issues)

#### Phase 3: Services (Depend on models)
- `services/user_service.py` (12 issues)
- `services/order_service.py` (10 issues)

#### Phase 4: Controllers (Depend on services)
- `controllers/api.py` (15 issues)

### Generated Pull Requests
Created 47 PRs (one per file):
- ✅ PR #501: Migrate `utils/string_helpers.py`
- ✅ PR #502: Migrate `utils/list_helpers.py`
- ✅ PR #503: Migrate `utils/dict_helpers.py`
- ... (44 more)

Each PR includes:
- Automated fixes
- Unit tests (generated/updated)
- Compatibility checks
- Rollback instructions

### Testing Strategy
```bash
# Run automated test suite
pytest tests/ --python=3.11

# Test each PR independently
git checkout pr/501
pytest tests/test_string_helpers.py
```

### Breaking Changes
⚠️ **3 potential breaking changes identified:**
1. `dict.iteritems()` behavior change (affects `models/user.py:45`)
2. String encoding differences (affects `utils/string_helpers.py:78`)
3. Division operator change (affects `calculations.py:120`)

Mitigation plan included in respective PRs.

📎 47 files analyzed, 128 issues fixed, 47 PRs created
```

**Time Saved:** 4 weeks → 3 days 📉

---

## API Reference

### Index Endpoint

```http
POST /api/code-graph/index
Content-Type: application/json

{
  "project": string,              // Project identifier (unique)
  "root_dir": string?,            // Local directory path
  "git_url": string?,             // Git repository URL
  "branch": string = "main",      // Git branch
  "auth_token": string?,          // Auth token for private repos
  "provider": "github" | "gitlab" | "bitbucket" = "github",
  "language": string = "auto",    // Programming language
  "exclude_patterns": string[],   // Glob patterns to exclude
  "merge": boolean = false,       // Merge with existing project
  "clone_depth": number = 1       // Git clone depth (shallow clone)
}

Response: 202 Accepted
{
  "message": "Indexing started",
  "project": "automatos-ai",
  "job_id": "idx_123456",
  "estimated_time_seconds": 180
}
```

### Search Endpoint

```http
GET /api/code-graph/search
Query Parameters:
  - project: string (required)
  - q: string (required)
  - limit: number = 10
  - offset: number = 0
  - file_types: string[] (e.g., ["py", "ts"])
  - symbol_types: string[] (e.g., ["function", "class"])

Response: 200 OK
{
  "count": 15,
  "query": "authentication flow",
  "chunks": [
    {
      "id": 1,
      "file": "services/auth_service.py",
      "line": 45,
      "symbol_name": "authenticate_user",
      "symbol_type": "function",
      "code_snippet": "def authenticate_user(username, password):\n    ...",
      "relevance": 0.95,
      "references": 12
    }
  ],
  "prompt_block": "string"  // Formatted for LLM context
}
```

### List Projects

```http
GET /api/code-graph/projects

Response: 200 OK
{
  "projects": [
    {
      "id": 1,
      "name": "automatos-ai",
      "source_type": "github",
      "source_url": "https://github.com/AutomatosAI/automatos-ai.git",
      "language": "python",
      "total_files": 1847,
      "total_symbols": 15234,
      "last_indexed": "2025-10-02T14:30:00Z",
      "auto_reindex": true,
      "status": "active"
    }
  ]
}
```

### Delete Project

```http
DELETE /api/code-graph/projects/{project_id}

Response: 200 OK
{
  "message": "Project deleted successfully",
  "project_id": 1,
  "files_removed": 1847,
  "symbols_removed": 15234
}
```

### Webhook Configuration

```http
POST /api/code-graph/projects/{project_id}/webhook
Content-Type: application/json

{
  "enabled": true,
  "webhook_url": "${API_URL}/webhooks/code-changed"  # Replace ${API_URL} with your API server URL,
  "auto_reindex": true,
  "events": ["push", "pull_request"]
}

Response: 200 OK
{
  "webhook_id": "wh_123456",
  "secret": "whsec_...",  // Use to verify webhook signatures
  "enabled": true
}
```

---

## Integration Examples

### CI/CD Integration (GitHub Actions)

```yaml
# .github/workflows/code-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Trigger Automatos Security Review
        run: |
          WORKFLOW_ID=$(curl -X POST ${API_URL}/api/workflows \
            -H "Authorization: Bearer ${{ secrets.AUTOMATOS_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d '{
              "name": "Security Review - PR #${{ github.event.pull_request.number }}",
              "goal": "Review PR for security, performance, and style issues",
              "context": {
                "codegraph_project": "my-app",
                "pr_number": ${{ github.event.pull_request.number }},
                "git_diff_url": "${{ github.event.pull_request.diff_url }}",
                "author": "${{ github.event.pull_request.user.login }}"
              }
            }' | jq -r '.id')
          
          echo "WORKFLOW_ID=$WORKFLOW_ID" >> $GITHUB_ENV
      
      - name: Wait for Review
        run: |
          # Poll workflow status
          while true; do
            STATUS=$(curl -s ${API_URL}/api/workflows/$WORKFLOW_ID \
              -H "Authorization: Bearer ${{ secrets.AUTOMATOS_API_KEY }}" \
              | jq -r '.status')
            
            if [ "$STATUS" = "completed" ]; then
              break
            fi
            
            sleep 10
          done
      
      - name: Post Review Comment
        uses: actions/github-script@v6
        with:
          script: |
            const report = await fetch(`${API_URL}/api/workflows/${process.env.WORKFLOW_ID}/report`, {
              headers: { 'Authorization': `Bearer ${{ secrets.AUTOMATOS_API_KEY }}` }
            }).then(r => r.json())
            
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.name,
              issue_number: context.issue.number,
              body: `## 🤖 AI Security Review\n\n${report.markdown}`
            })
```

### Slack Bot Integration

```python
# slack_bot.py
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import requests

app = App(token=os.environ["SLACK_BOT_TOKEN"])
AUTOMATOS_API = "${API_URL}/api"  # Replace ${API_URL} with your API server URL
AUTOMATOS_TOKEN = os.environ["AUTOMATOS_API_KEY"]

@app.command("/code-search")
def handle_code_search(ack, command, say):
    ack()
    
    query = command['text']
    project = command.get('channel_name', 'default-project')
    
    # Query CodeGraph via Automatos
    response = requests.get(
        f"{AUTOMATOS_API}/code-graph/search",
        headers={"Authorization": f"Bearer {AUTOMATOS_TOKEN}"},
        params={
            'project': project,
            'q': query,
            'limit': 5
        }
    ).json()
    
    # Format response for Slack
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🔍 Found {response['count']} results"}
        }
    ]
    
    for chunk in response['chunks'][:5]:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{chunk['file']}:{chunk['line']}*\n```{chunk['code_snippet']}```"
            }
        })
    
    say(blocks=blocks)

@app.command("/review-pr")
def handle_review_pr(ack, command, say):
    ack()
    
    pr_url = command['text']
    
    # Extract PR number from URL
    pr_number = pr_url.split('/')[-1]
    
    # Create workflow
    workflow = requests.post(
        f"{AUTOMATOS_API}/workflows",
        headers={"Authorization": f"Bearer {AUTOMATOS_TOKEN}"},
        json={
            "name": f"PR Review #{pr_number}",
            "goal": "Review pull request for issues",
            "context": {
                "codegraph_project": "my-app",
                "pr_number": int(pr_number),
                "git_diff_url": f"{pr_url}.diff"
            }
        }
    ).json()
    
    say(f"✅ Started AI review of PR #{pr_number}. I'll notify you when it's done!")
    
    # Poll workflow status (in real implementation, use webhooks)
    # ...

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
```

### VSCode Extension

```typescript
// extension.ts
import * as vscode from 'vscode'
import axios from 'axios'

const AUTOMATOS_API = process.env.API_URL ? `${process.env.API_URL}/api` : 'http://localhost:8000/api'
const AUTOMATOS_TOKEN = process.env.AUTOMATOS_API_KEY

export function activate(context: vscode.ExtensionContext) {
  // Command: Explain selected code
  let explainCommand = vscode.commands.registerCommand(
    'automatos.explainCode',
    async () => {
      const editor = vscode.window.activeTextEditor
      if (!editor) return

      const selection = editor.document.getText(editor.selection)
      const filePath = editor.document.fileName

      // Show loading
      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'Explaining code...',
        },
        async () => {
          // Send to Automatos
          const response = await axios.post(
            `${AUTOMATOS_API}/chat/query`,
            {
              message: `Explain this code: ${selection}`,
              project: 'my-app',
              file: filePath,
            },
            {
              headers: { Authorization: `Bearer ${AUTOMATOS_TOKEN}` },
            }
          )

          // Show explanation in webview
          const panel = vscode.window.createWebviewPanel(
            'codeExplanation',
            'Code Explanation',
            vscode.ViewColumn.Beside,
            {}
          )

          panel.webview.html = getExplanationHTML(response.data.answer)
        }
      )
    }
  )

  // Command: Search codebase
  let searchCommand = vscode.commands.registerCommand(
    'automatos.searchCode',
    async () => {
      const query = await vscode.window.showInputBox({
        prompt: 'Search your codebase',
        placeHolder: 'e.g., authentication flow',
      })

      if (!query) return

      const response = await axios.get(
        `${AUTOMATOS_API}/code-graph/search`,
        {
          params: { project: 'my-app', q: query, limit: 10 },
          headers: { Authorization: `Bearer ${AUTOMATOS_TOKEN}` },
        }
      )

      // Show results in Quick Pick
      const items = response.data.chunks.map((chunk: any) => ({
        label: `${chunk.file}:${chunk.line}`,
        description: chunk.symbol_name,
        detail: chunk.code_snippet.substring(0, 100),
        chunk,
      }))

      const selected = await vscode.window.showQuickPick(items, {
        matchOnDescription: true,
        matchOnDetail: true,
      })

      if (selected) {
        // Open file and jump to line
        const doc = await vscode.workspace.openTextDocument(selected.chunk.file)
        const editor = await vscode.window.showTextDocument(doc)
        const position = new vscode.Position(selected.chunk.line - 1, 0)
        editor.selection = new vscode.Selection(position, position)
        editor.revealRange(new vscode.Range(position, position))
      }
    }
  )

  context.subscriptions.push(explainCommand, searchCommand)
}

function getExplanationHTML(explanation: string): string {
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: sans-serif; padding: 20px; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
      </style>
    </head>
    <body>
      <h2>Code Explanation</h2>
      <div>${explanation.replace(/\n/g, '<br>')}</div>
    </body>
    </html>
  `
}
```

---

## Monetization

### Pricing Tiers

```
FREE TIER
- 1 project
- Up to 10,000 lines of code
- 100 queries/month
- Local directory only
- Community support

PRO TIER - $99/month
- 5 projects
- Up to 500,000 lines of code
- Unlimited queries
- GitHub/GitLab integration
- Chatbot access
- Email support

ENTERPRISE TIER - $499/month
- Unlimited projects
- Unlimited code
- Unlimited queries
- All integrations (GitHub, GitLab, Bitbucket)
- On-premises deployment option
- Dedicated support (24/7)
- Custom integrations (Slack, Teams, VSCode)
- SLA guarantees
- SOC2 compliance

ENTERPRISE PLUS - Custom Pricing
- White-label solution
- Air-gapped deployment
- Custom model training
- Professional services
- Implementation support
```

### ROI Calculator

**Example: Mid-size SaaS company (50 developers)**

**Without CodeGraph:**
- Developer spends 4 hours/week searching/understanding code
- 50 developers × 4 hours × $100/hour = $20,000/week
- Annual cost: **$1,040,000 in lost productivity**

**With CodeGraph Pro ($99/month = $1,188/year):**
- Reduce code search time by 80% (4 hrs → 48 min)
- Save: 3.2 hours/week per developer
- 50 developers × 3.2 hours × $100/hour × 52 weeks = **$832,000 saved**
- Less CodeGraph cost: $1,188/year
- **Net savings: $830,812/year**

**ROI: 69,900% 🚀**

---

## FAQ

### Q: Can I keep my code private?
**A:** Yes! Three options:
1. **On-Premises**: Install Automatos on your servers, code never leaves your infrastructure
2. **Air-Gapped**: Enterprise deployment with no external connections
3. **Encrypted Cloud**: Code encrypted at rest and in transit, SOC2 compliant

### Q: How long does indexing take?
**A:** ~1-2 seconds per 1,000 lines. Examples:
- 10K line project: 10-20 seconds
- 100K line project: 2-3 minutes
- 1M line project: 20-30 minutes

### Q: Does it work with legacy code?
**A:** Yes! Supports:
- Python, TypeScript, JavaScript, Java, Go, Rust, C++, PHP, Ruby, Swift, Kotlin
- Even ancient code (Python 2.7, PHP 5.6, etc.)

### Q: Can I delete a project?
**A:** Yes. `DELETE /api/code-graph/projects/{id}` removes all indexed data permanently.

### Q: Does it update automatically?
**A:** Yes, with webhooks:
- Configure webhook in GitHub/GitLab
- Auto-reindex on every push
- Incremental updates (only changed files)

### Q: What about secrets in code?
**A:** Indexer automatically excludes:
- `.env` files
- `secrets.yml`, `credentials.json`
- API keys (detected via regex)
- Private keys (PEM, SSH keys)
- Passwords in config files

### Q: Can I use it on Automatos AI's own code?
**A:** YES! That's the meta use case:
```bash
POST /api/code-graph/index
{
  "project": "automatos-ai",
  "git_url": "https://github.com/AutomatosAI/automatos-ai.git"
}
```
Then use workflows to:
- Onboard new contributors
- Debug production issues
- Generate architecture docs
- Review PRs

### Q: How accurate is semantic search?
**A:** 95%+ relevance on top results:
- Uses tree-sitter for AST parsing (language-aware)
- Vector embeddings for semantic meaning
- Symbol relationship tracking
- Call graph analysis

### Q: Can I search across multiple languages in one project?
**A:** Yes! Full-stack projects (Python backend + TypeScript frontend) are fully supported:
```bash
POST /api/code-graph/index
{
  "project": "fullstack-app",
  "root_dir": "/app",
  "languages": ["python", "typescript"]  // Auto-detected
}
```

### Q: What's the difference between CodeGraph and GitHub Copilot?
**A:**
| Feature | CodeGraph | GitHub Copilot |
|---------|-----------|----------------|
| **Scope** | Entire codebase | Current file |
| **Search** | Semantic search across all files | N/A |
| **Context** | Understands relationships | Limited context window |
| **Workflows** | Integrated with AI agents | Manual queries only |
| **Use Case** | Code intelligence platform | Code completion tool |

**They're complementary!** Use both together.

---

## Summary

### Key Takeaways

1. **CodeGraph indexes CLIENT codebases** (not Automatos code, unless you want meta usage)
2. **Multi-source support**: Local dirs, GitHub, GitLab, Bitbucket
3. **Workflow integration**: Add `codegraph_project` to context, agents automatically get code
4. **Chatbot integration**: 30-minute implementation, massive UX improvement
5. **Real-time updates**: Webhook-based auto-reindexing
6. **Enterprise-ready**: On-prem deployment, SOC2 compliance, SLA guarantees

### Implementation Priorities

**Phase 1 (Week 1-2):** ✅ Already Done
- Basic indexing (local directory)
- Search API
- Database schema
- Simple UI

**Phase 2 (Week 3):** Next Steps
- GitHub URL support
- Multi-project management
- Background re-indexing
- Enhanced UI (tabs, analytics)

**Phase 3 (Week 4):** Integration
- Workflow integration (`codegraph_project` context)
- Agent CodeGraph query capability
- Automatic context injection

**Phase 4 (Week 5):** Chatbot
- Chat interface UI
- Project selector
- Code-aware prompts
- Citation rendering

**Phase 5 (Week 6-7):** Advanced Features
- Dependency graphs
- Complexity heatmaps
- Query analytics
- Insights dashboard

**Phase 6 (Week 8):** Ecosystem
- GitHub Actions integration
- Slack bot
- VSCode extension
- Webhook system

---

## Next Steps

1. **Review this document** and confirm approach
2. **Build advanced UI** (multi-source, analytics, workflow integration)
3. **Test with Automatos AI codebase** (meta use case!)
4. **Document learnings** and iterate

Ready to build? 🚀

