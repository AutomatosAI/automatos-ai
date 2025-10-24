# PRD-11: CodeGraph Implementation & Integration

**Status:** Draft  
**Priority:** High  
**Effort:** 12-16 hours  
**Dependencies:** None (standalone feature)

---

## 1. Overview

### 1.1 Purpose

CodeGraph transforms client codebases into AI-readable knowledge graphs, enabling agents to access precise code context instead of entire repositories. This is **NOT** for Automatos's own code (though we can index it for meta purposes) - it's for **client/user codebases**.

### 1.2 Vision Alignment

Following the Context Engineering paradigm:
- **Atoms**: Individual code symbols (functions, classes)
- **Molecules**: Symbol relationships (calls, imports, dependencies)
- **Cells**: Contextual code chunks with embeddings
- **Organs**: Multi-file analysis with call graphs
- **Organisms**: Complete codebase understanding

### 1.3 Value Proposition

> "Turn any codebase into an AI-readable knowledge graph. Instead of dumping 10,000 files into context, agents get laser-focused, relevant code snippets."

**Real-World Impact:**
- **3 weeks → 30 minutes**: Developer onboarding time
- **2-3 days → 2 minutes**: Security review automation
- **4 weeks → 3 days**: Legacy migration projects

---

## 2. Problem Statement

### 2.1 Current State

**What Exists:**
- Basic UI in `CodeGraphPanel.tsx` (69 lines)
- Backend API stubs exist but incomplete
- Database schema not fully implemented
- No multi-project support
- No GitHub integration
- No analytics

**What's Missing:**
- Multi-source indexing (GitHub, GitLab, local)
- Background re-indexing
- Call graph analysis
- Complexity heatmaps
- Query analytics
- Workflow integration
- Chatbot integration

### 2.2 Business Impact

Without CodeGraph:
- Agents can't access client code intelligently
- Manual code review takes days
- No automated legacy migration
- No intelligent refactoring
- No code-aware chatbot assistance

---

## 3. Success Criteria

### 3.1 Functional Requirements
- [ ] Index codebases from 3 sources (local, GitHub, GitLab)
- [ ] Support 6+ languages (Python, TypeScript, JavaScript, Go, Rust, Java)
- [ ] Multi-project management (unlimited projects)
- [ ] Search by semantic meaning or symbol name
- [ ] Generate call graphs and dependency trees
- [ ] Integrate with workflow system
- [ ] Enable chatbot code queries

### 3.2 Performance Requirements
- [ ] Index 10K lines in <10 seconds
- [ ] Index 100K lines in <2 minutes
- [ ] Search latency <500ms
- [ ] Support 50+ concurrent projects
- [ ] Handle repositories up to 1M lines

### 3.3 Quality Requirements
- [ ] Search relevance >85%
- [ ] Symbol extraction accuracy >95%
- [ ] Relationship mapping accuracy >90%
- [ ] Zero data loss during re-indexing
- [ ] Graceful degradation on parse errors

---

## 4. Functional Requirements

### 4.1 Code Indexing

#### 4.1.1 Multi-Source Support

**Local Directory:**
```python
POST /api/code-graph/index
{
  "project": "my-app",
  "source_type": "local",
  "root_dir": "/path/to/code",
  "language": "auto",  # Auto-detect
  "exclude_patterns": ["node_modules", "__pycache__", "*.pyc"]
}
```

**GitHub Repository:**
```python
POST /api/code-graph/index
{
  "project": "client-acme",
  "source_type": "github",
  "git_url": "https://github.com/acme-corp/backend.git",
  "branch": "main",
  "auth_token": "ghp_...",  # Optional for private repos
  "clone_depth": 1  # Shallow clone
}
```

**GitLab Repository:**
```python
POST /api/code-graph/index
{
  "project": "enterprise-app",
  "source_type": "gitlab",
  "git_url": "https://gitlab.com/company/app.git",
  "branch": "develop",
  "auth_token": "glpat-...",
  "provider": "gitlab"
}
```

#### 4.1.2 Language Support

| Language | Parser | Status | Features |
|----------|--------|--------|----------|
| Python | tree-sitter-python | ✅ Ready | Classes, functions, imports, decorators |
| TypeScript | tree-sitter-typescript | ✅ Ready | Interfaces, types, classes, exports |
| JavaScript | tree-sitter-javascript | ✅ Ready | Functions, classes, modules |
| Go | tree-sitter-go | ⚠️ Partial | Functions, structs, interfaces |
| Rust | tree-sitter-rust | ⚠️ Partial | Functions, structs, traits, impls |
| Java | tree-sitter-java | ⚠️ Partial | Classes, methods, interfaces |

#### 4.1.3 Symbol Extraction

Extract and index:
- **Functions/Methods**: Name, parameters, return type, docstring
- **Classes/Structs**: Name, inheritance, methods, properties
- **Imports**: Dependencies, source modules
- **Variables**: Global/class-level constants
- **Types**: Interfaces, enums, type aliases
- **Comments**: Documentation strings

#### 4.1.4 Relationship Mapping

Build graphs of:
- **Call Graph**: Function A calls Function B
- **Import Graph**: Module A imports Module B
- **Inheritance Graph**: Class A extends Class B
- **Dependency Graph**: Package A depends on Package B

### 4.2 Code Search

#### 4.2.1 Symbol Search

```python
GET /api/code-graph/search?project=my-app&q=authenticate_user&type=function
```

**Response:**
```json
{
  "count": 3,
  "results": [
    {
      "symbol_type": "function",
      "name": "authenticate_user",
      "file": "services/auth_service.py",
      "line": 45,
      "signature": "def authenticate_user(username: str, password: str) -> User",
      "docstring": "Authenticate user with username and password",
      "calls": ["hash_password", "check_user_exists"],
      "called_by": ["login_endpoint", "api_authenticate"]
    }
  ]
}
```

#### 4.2.2 Semantic Search

```python
POST /api/code-graph/search
{
  "project": "my-app",
  "query": "How do I validate user permissions?",
  "limit": 10,
  "file_types": ["py"]
}
```

**Response:** Returns semantically relevant code chunks with embeddings.

#### 4.2.3 Call Graph Queries

```python
GET /api/code-graph/call-graph?project=my-app&symbol=process_payment&depth=2
```

**Response:**
```json
{
  "root": "process_payment",
  "nodes": [
    {"id": "process_payment", "type": "function", "file": "payment.py"},
    {"id": "validate_card", "type": "function", "file": "validation.py"},
    {"id": "charge_stripe", "type": "function", "file": "stripe_api.py"}
  ],
  "edges": [
    {"from": "process_payment", "to": "validate_card", "type": "calls"},
    {"from": "process_payment", "to": "charge_stripe", "type": "calls"}
  ]
}
```

### 4.3 Project Management

#### 4.3.1 List Projects

```python
GET /api/code-graph/projects
```

**Response:**
```json
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
      "total_relationships": 42301,
      "last_indexed": "2025-10-02T14:30:00Z",
      "index_duration_seconds": 127,
      "status": "active",
      "auto_reindex": true
    }
  ]
}
```

#### 4.3.2 Delete Project

```python
DELETE /api/code-graph/projects/{project_id}
```

Removes all indexed data permanently.

#### 4.3.3 Re-index Project

```python
POST /api/code-graph/projects/{project_id}/reindex
{
  "incremental": true  # Only re-index changed files
}
```

### 4.4 Analytics

#### 4.4.1 Query Analytics

```python
GET /api/code-graph/analytics/queries?project=my-app&period=7d
```

Track:
- Most queried files
- Popular search terms
- Average query latency
- Search success rate

#### 4.4.2 Complexity Metrics

```python
GET /api/code-graph/analytics/complexity?project=my-app
```

**Response:**
```json
{
  "files": [
    {
      "file": "services/payment_service.py",
      "complexity": 87,  # Cyclomatic complexity
      "lines": 523,
      "functions": 23,
      "classes": 4,
      "dependencies": 12
    }
  ]
}
```

---

## 5. Technical Architecture

### 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CodeGraph System                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  INPUT SOURCES                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │Local Dir │  │GitHub URL│  │GitLab URL│              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │             │              │                     │
│       └─────────────┴──────────────┘                    │
│                     │                                    │
│              ┌──────▼──────┐                            │
│              │   INDEXER   │                            │
│              │(tree-sitter)│                            │
│              └──────┬──────┘                            │
│                     │                                    │
│       ┌─────────────┴─────────────┐                    │
│       │                           │                     │
│  ┌────▼─────┐              ┌─────▼────┐               │
│  │ Symbol   │              │Relations │               │
│  │  Index   │◄─────────────┤  Graph   │               │
│  └────┬─────┘              └─────┬────┘               │
│       │                           │                     │
│       └─────────────┬─────────────┘                    │
│                     │                                    │
│              ┌──────▼──────┐                            │
│              │  POSTGRES   │                            │
│              │ + pgvector  │                            │
│              │ + networkx  │                            │
│              └──────┬──────┘                            │
│                     │                                    │
│       ┌─────────────┴─────────────┐                    │
│       │                           │                     │
│  ┌────▼─────┐              ┌─────▼────┐               │
│  │Workflows │              │ Chatbot  │               │
│  │ (Agents) │              │   API    │               │
│  └──────────┘              └──────────┘               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Database Schema

```sql
-- Projects table
CREATE TABLE codegraph_projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- local, github, gitlab
    source_url TEXT,
    branch VARCHAR(255),
    language VARCHAR(50),
    total_files INTEGER DEFAULT 0,
    total_symbols INTEGER DEFAULT 0,
    total_relationships INTEGER DEFAULT 0,
    last_indexed TIMESTAMP,
    index_duration_seconds FLOAT,
    status VARCHAR(50) DEFAULT 'active',
    auto_reindex BOOLEAN DEFAULT false,
    exclude_patterns JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Code symbols (functions, classes, etc.)
CREATE TABLE codegraph_symbols (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    symbol_type VARCHAR(50) NOT NULL,  -- function, class, interface, etc.
    name VARCHAR(500) NOT NULL,
    qualified_name VARCHAR(1000),  -- Full path: module.Class.method
    file_path VARCHAR(1000) NOT NULL,
    line_number INTEGER NOT NULL,
    signature TEXT,
    docstring TEXT,
    code_snippet TEXT,
    embedding VECTOR(1536),  -- For semantic search
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_codegraph_symbols_project ON codegraph_symbols(project_id);
CREATE INDEX idx_codegraph_symbols_type ON codegraph_symbols(symbol_type);
CREATE INDEX idx_codegraph_symbols_name ON codegraph_symbols(name);
CREATE INDEX idx_codegraph_symbols_embedding ON codegraph_symbols 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Code relationships (calls, imports, inheritance)
CREATE TABLE codegraph_relationships (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    from_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
    to_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,  -- calls, imports, extends, implements
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_codegraph_relationships_project ON codegraph_relationships(project_id);
CREATE INDEX idx_codegraph_relationships_from ON codegraph_relationships(from_symbol_id);
CREATE INDEX idx_codegraph_relationships_to ON codegraph_relationships(to_symbol_id);

-- Query analytics
CREATE TABLE codegraph_query_logs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    query_type VARCHAR(50) NOT NULL,  -- symbol, semantic, call_graph
    query_text TEXT NOT NULL,
    results_count INTEGER,
    execution_time_ms FLOAT,
    user_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_codegraph_query_logs_project ON codegraph_query_logs(project_id);
CREATE INDEX idx_codegraph_query_logs_created ON codegraph_query_logs(created_at DESC);

-- File metadata
CREATE TABLE codegraph_files (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    file_path VARCHAR(1000) NOT NULL,
    file_hash VARCHAR(64),  -- SHA-256 for change detection
    file_size INTEGER,
    lines_of_code INTEGER,
    complexity_score INTEGER,
    language VARCHAR(50),
    last_modified TIMESTAMP,
    indexed_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    UNIQUE(project_id, file_path)
);

CREATE INDEX idx_codegraph_files_project ON codegraph_files(project_id);
CREATE INDEX idx_codegraph_files_path ON codegraph_files(file_path);
```

---

## 6. Implementation Details

### 6.1 Indexing Pipeline

```python
class CodeGraphIndexer:
    """
    Main indexing engine using tree-sitter
    """
    
    async def index_project(
        self,
        project_id: int,
        source_type: str,
        source_path: str,
        **options
    ):
        """
        1. Clone/load source code
        2. Discover files
        3. Parse each file
        4. Extract symbols
        5. Build relationships
        6. Generate embeddings
        7. Store in database
        """
        
        # Step 1: Source loading
        if source_type == "github":
            repo_path = await self.clone_repo(source_path, options)
        elif source_type == "local":
            repo_path = source_path
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Step 2: File discovery
        files = self.discover_files(
            repo_path,
            exclude=options.get("exclude_patterns", [])
        )
        
        logger.info(f"Found {len(files)} files to index")
        
        # Step 3: Parse files in parallel
        symbols_batch = []
        relationships_batch = []
        
        for file_path in files:
            try:
                result = await self.parse_file(file_path)
                symbols_batch.extend(result.symbols)
                relationships_batch.extend(result.relationships)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
        
        # Step 4: Generate embeddings
        await self.generate_embeddings(symbols_batch)
        
        # Step 5: Store in database
        await self.store_symbols(project_id, symbols_batch)
        await self.store_relationships(project_id, relationships_batch)
        
        # Step 6: Update project stats
        await self.update_project_stats(project_id)
        
        logger.info(f"Indexed {len(symbols_batch)} symbols")
    
    async def parse_file(self, file_path: str):
        """
        Parse single file using tree-sitter
        """
        # Detect language
        language = self.detect_language(file_path)
        parser = self.get_parser(language)
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Parse AST
        tree = parser.parse(bytes(code, 'utf8'))
        
        # Extract symbols
        extractor = SymbolExtractor(language)
        symbols = extractor.extract_symbols(tree, file_path)
        
        # Extract relationships
        relationships = extractor.extract_relationships(tree, symbols)
        
        return ParseResult(
            symbols=symbols,
            relationships=relationships,
            tree=tree
        )
```

### 6.2 Search Implementation

```python
class CodeGraphSearch:
    """
    Search engine for code queries
    """
    
    async def symbol_search(
        self,
        project_id: int,
        query: str,
        symbol_type: Optional[str] = None,
        limit: int = 10
    ):
        """
        Exact/fuzzy symbol name search
        """
        query_filter = text("""
            SELECT 
                id,
                symbol_type,
                name,
                qualified_name,
                file_path,
                line_number,
                signature,
                docstring,
                code_snippet
            FROM codegraph_symbols
            WHERE project_id = :project_id
                AND (:symbol_type IS NULL OR symbol_type = :symbol_type)
                AND (
                    name ILIKE :query 
                    OR qualified_name ILIKE :query
                )
            ORDER BY 
                CASE 
                    WHEN name = :exact_query THEN 1
                    WHEN name ILIKE :starts_with THEN 2
                    ELSE 3
                END,
                name
            LIMIT :limit
        """)
        
        results = db.execute(
            query_filter,
            {
                "project_id": project_id,
                "symbol_type": symbol_type,
                "query": f"%{query}%",
                "exact_query": query,
                "starts_with": f"{query}%",
                "limit": limit
            }
        ).fetchall()
        
        return [self._format_symbol(r) for r in results]
    
    async def semantic_search(
        self,
        project_id: int,
        query: str,
        limit: int = 10
    ):
        """
        Vector similarity search for semantic code queries
        """
        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        
        # Vector search
        similarity_query = text("""
            SELECT 
                cs.*,
                1 - (cs.embedding <=> :query_embedding::vector) as similarity
            FROM codegraph_symbols cs
            WHERE cs.project_id = :project_id
                AND cs.embedding IS NOT NULL
            ORDER BY cs.embedding <=> :query_embedding::vector
            LIMIT :limit
        """)
        
        results = db.execute(
            similarity_query,
            {
                "project_id": project_id,
                "query_embedding": query_embedding,
                "limit": limit
            }
        ).fetchall()
        
        return [self._format_symbol(r) for r in results]
    
    async def call_graph_search(
        self,
        project_id: int,
        symbol_name: str,
        depth: int = 2,
        direction: str = "both"  # "calls", "called_by", "both"
    ):
        """
        Build call graph starting from symbol
        """
        # Find root symbol
        root = await self.find_symbol(project_id, symbol_name)
        
        if not root:
            return {"error": "Symbol not found"}
        
        # Traverse graph
        nodes = [root]
        edges = []
        visited = {root.id}
        
        queue = [(root.id, 0)]
        
        while queue:
            symbol_id, current_depth = queue.pop(0)
            
            if current_depth >= depth:
                continue
            
            # Get relationships
            if direction in ["calls", "both"]:
                outgoing = await self.get_calls_from(symbol_id)
                for rel in outgoing:
                    if rel.to_symbol_id not in visited:
                        nodes.append(rel.to_symbol)
                        edges.append(rel)
                        visited.add(rel.to_symbol_id)
                        queue.append((rel.to_symbol_id, current_depth + 1))
            
            if direction in ["called_by", "both"]:
                incoming = await self.get_calls_to(symbol_id)
                for rel in incoming:
                    if rel.from_symbol_id not in visited:
                        nodes.append(rel.from_symbol)
                        edges.append(rel)
                        visited.add(rel.from_symbol_id)
                        queue.append((rel.from_symbol_id, current_depth + 1))
        
        return {
            "root": symbol_name,
            "nodes": [self._format_node(n) for n in nodes],
            "edges": [self._format_edge(e) for e in edges],
            "depth": depth
        }
```

---

## 7. Workflow Integration

### 7.1 Automatic Context Injection

When a workflow includes `codegraph_project` in its context, agents automatically get relevant code:

```python
# In agent execution
async def execute_with_codegraph(
    self,
    task: str,
    context: dict
):
    """
    Execute agent task with CodeGraph context
    """
    codegraph_project = context.get("codegraph_project")
    
    if codegraph_project:
        # Search for relevant code
        code_results = await codegraph_search.semantic_search(
            project=codegraph_project,
            query=task,
            limit=5
        )
        
        # Format code context
        code_context = self.format_code_context(code_results)
        
        # Augment agent prompt
        augmented_prompt = f"""
        {task}
        
        ## Relevant Code Context:
        {code_context}
        
        Use the above code as reference when completing this task.
        """
        
        # Execute with enhanced context
        result = await self.llm.generate(augmented_prompt)
        return result
```

### 7.2 Workflow Example

```typescript
POST /api/workflows
{
  "name": "Security Review - PR #456",
  "description": "Review pull request for security issues",
  "goal": "Analyze PR #456 for SQL injection, XSS, and auth bypass",
  "context": {
    "codegraph_project": "client-acme-ecommerce",  // <-- Enables code access
    "pr_number": 456,
    "git_diff_url": "https://github.com/acme/app/pull/456.diff"
  }
}
```

**Agent automatically receives:**
- Existing authentication patterns
- Database query examples
- Security middleware code
- Related test cases

---

## 8. Chatbot Integration

### 8.1 Chat Interface with Code Context

```typescript
// Frontend: Chat with code awareness
POST /api/chat/query
{
  "message": "How do I authenticate users in this codebase?",
  "project": "automatos-ai",
  "include_code": true,
  "max_results": 3
}
```

**Response:**
```json
{
  "answer": "To authenticate users in this codebase, use the `authenticate_user` function in `services/auth_service.py`. Here's how it works...",
  "code_references": [
    {
      "symbol": "authenticate_user",
      "file": "services/auth_service.py",
      "line": 45,
      "snippet": "def authenticate_user(username: str, password: str) -> User:\n    ...",
      "relevance": 0.95
    }
  ]
}
```

---

## 9. API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/code-graph/index` | POST | Index new project |
| `/api/code-graph/projects` | GET | List all projects |
| `/api/code-graph/projects/{id}` | GET | Get project details |
| `/api/code-graph/projects/{id}` | DELETE | Delete project |
| `/api/code-graph/projects/{id}/reindex` | POST | Re-index project |
| `/api/code-graph/search` | GET | Symbol search |
| `/api/code-graph/search` | POST | Semantic search |
| `/api/code-graph/call-graph` | GET | Generate call graph |
| `/api/code-graph/analytics/queries` | GET | Query analytics |
| `/api/code-graph/analytics/complexity` | GET | Code complexity metrics |
| `/api/code-graph/health` | GET | System health check |

---

## 10. Implementation Timeline

### Phase 1: Core Indexing (Week 1)
**Day 1-2:** Database schema + basic indexer
- Create tables
- Implement file discovery
- Basic tree-sitter integration

**Day 3-4:** Symbol extraction
- Python parser
- TypeScript parser
- Symbol storage

**Day 5:** Testing
- Index test projects
- Verify symbol extraction
- Performance testing

### Phase 2: Search & Relationships (Week 2)
**Day 1-2:** Search implementation
- Symbol search
- Semantic search
- Query API

**Day 3-4:** Relationship mapping
- Call graph builder
- Dependency tracking
- Graph queries

**Day 5:** Analytics
- Query tracking
- Complexity metrics
- Dashboard integration

### Phase 3: Integration (Week 3)
**Day 1-2:** Workflow integration
- Context injection
- Agent access
- Testing with workflows

**Day 3-4:** Multi-source support
- GitHub cloning
- GitLab support
- Credential management

**Day 5:** UI enhancement
- Advanced UI components
- Project management
- Analytics visualization

---

## 11. Success Metrics

### 11.1 Performance
- Index 10K lines: <10s
- Symbol search: <100ms
- Semantic search: <500ms
- Call graph generation: <1s

### 11.2 Quality
- Symbol extraction accuracy: >95%
- Search relevance: >85%
- Relationship accuracy: >90%

### 11.3 Adoption
- Projects indexed per user: >3
- Queries per day: >50
- Agent usage rate: >70%

---

## 12. Risk Mitigation

### 12.1 Technical Risks
- **Large repos**: Implement incremental indexing
- **Parse errors**: Graceful degradation, skip unparseable files
- **Storage costs**: Cleanup old/unused projects automatically
- **Performance**: Cache queries, optimize indexes

### 12.2 Quality Risks
- **Search relevance**: User feedback loops, tuning
- **Symbol accuracy**: Language-specific testing
- **Relationship mapping**: Validate with known codebases

---

## 13. Dependencies

- **tree-sitter**: v0.20+ (symbol parsing)
- **tree-sitter-languages**: Language grammar support
- **networkx**: v3.0+ (graph analysis)
- **pgvector**: PostgreSQL extension (vector search)
- **GitPython**: v3.1+ (Git repository handling)

---

## 14. Out of Scope (Future Enhancements)

- Bitbucket support (add later)
- Real-time file watching (webhook-based)
- Code diff analysis
- Historical code search (git blame integration)
- Multi-language projects in single query
- Custom language parsers
- AI-powered code suggestions
- Automated refactoring suggestions

---

## 15. Acceptance Criteria

### 15.1 Functional
- [ ] Can index local directory
- [ ] Can index GitHub repository
- [ ] Can search by symbol name
- [ ] Can search semantically
- [ ] Can generate call graphs
- [ ] Integrates with workflows
- [ ] Accessible via chatbot

### 15.2 Non-Functional
- [ ] All performance targets met
- [ ] No data loss during re-indexing
- [ ] Graceful error handling
- [ ] Comprehensive logging
- [ ] Security: No token leakage

### 15.3 Quality
- [ ] Unit tests: >80% coverage
- [ ] Integration tests pass
- [ ] Real-world testing with 3+ projects
- [ ] Documentation complete

---

**Total Effort:** 12-16 hours (2 weeks)  
**Priority:** High (enables code-aware agents)  
**ROI:** Massive (3 weeks → 30 min onboarding, 2-3 days → 2 min reviews)

This PRD enables the complete CodeGraph system from indexing to workflow integration, transforming how AI agents interact with code.

