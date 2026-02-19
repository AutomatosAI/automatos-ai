# PRD 62: CodeGraph v2 — Top-10 Competitive Upgrade

**Status:** Draft
**Priority:** High
**Effort:** 40-50 hours (phased)
**Dependencies:** PRD-11 (CodeGraph v1, completed), PRD-30 (Modular Architecture)
**Created:** 2026-02-18
**Research Base:** Deep analysis of 10+ leading code graph/code intelligence projects + Automatos codebase audit

---

## Executive Summary

Deep research across the top code graph/code intelligence open-source projects (Aider 30K+ stars, Tree-sitter 23.8K, Sourcetrail 16.4K, Semgrep 9.2K, CodeQL 8K, Joern 2.9K, Code-Graph-RAG 1.9K, Emerge 1K, CodeFuse-CGM 521, CodePrism/CodeGraph-Rust) reveals that **Automatos already has a surprisingly strong CodeGraph implementation** — ~4,500+ lines of working code across backend + frontend with full API, graph visualization (ReactFlow + D3), agent integration, and workflow hooks.

However, critical gaps remain: **no tree-sitter** (limits language coverage to 3), regex-based TypeScript parsing, no MCP exposure, no incremental indexing, and a database schema mismatch. This PRD closes those gaps across 8 phases.

**Verdict: KEEP and ENHANCE** — Your existing implementation is worth keeping. It already has multi-tenant isolation, workspace-based security, agent tool integration, and a working ReactFlow visualization that none of the open-source projects have in a SaaS context.

---

## Part 1: How the Top Projects Work

### Tier 1: Foundational Infrastructure

#### 1. Tree-sitter (23.8K stars) — The Universal Parser
- **Core Innovation:** Incremental parsing library that generates parsers from grammar definitions. Produces concrete syntax trees (CSTs) that are error-recovering and zero-copy.
- **Languages:** 100+ via community grammars
- **Performance:** Sub-millisecond incremental parses. Used by Neovim, Helix, Zed, Emacs as their parsing backbone.
- **Ecosystem:** `tree-sitter-graph` (official DSL for graph construction from trees), Graph-sitter (Codegen — semantic call graphs), IBM `tree-sitter-codeviews` (15+ code views: AST, CFG, DFG, PDG, CPG)
- **License:** MIT
- **Key Lesson:** Tree-sitter is the universal foundation — every new code graph project uses it. Automatos should too.

#### 2. Aider (30K+ stars) — LLM Context Optimization via Repo Map
- **Core Innovation:** Uses tree-sitter to parse every file, builds a NetworkX dependency graph (files as nodes, cross-references as edges), runs PageRank to rank symbols by importance, then fits the most important symbols into a token budget (default 1,024 tokens). This is sent to the LLM as context.
- **Pipeline:** tree-sitter parse → symbol extraction → dependency graph → PageRank → token budget optimization
- **Performance:** 4.3-6.5% context window utilization (vs 54-70% for iterative search)
- **UI:** Text-only repo map output (no graph visualization)
- **License:** Apache-2.0
- **Key Lesson:** PageRank-based importance ranking for LLM context is brilliant and directly applicable to Automatos's agent tools.

### Tier 2: Visual Exploration

#### 3. Sourcetrail (16.4K stars) — Gold Standard UI
- **Core Innovation:** Three synchronized views (Search, Graph, Code) with bidirectional navigation. Click anything in any view and all three update. This is the benchmark for code exploration UI.
- **Graph View:** Sugiyama-style layout, color-coded nodes (gray=types, yellow=functions, blue=variables), bundled edges with counts, expansion arrows for class members, striped hatching for external symbols
- **Code View:** Snippets grouped by file with 3 states (minimized/snippet/maximized), syntax highlighting, click-to-navigate
- **Search:** Fuzzy matching ("UsrMdl" → "UserModel"), autocompletion, full-text search
- **Storage:** SQLite for persistent symbol/relationship database
- **License:** GPL-3.0 (petermost fork actively maintained, 2025 releases)
- **Key Lesson:** Synchronized multi-view UI is the gold standard. Automatos's ReactFlow visualization is a start but lacks the code-view synchronization.

#### 4. Emerge (1K stars) — Best Web-Based Visualization
- **Core Innovation:** D3.js force-directed graph web app with Louvain modularity clustering (auto-detects tightly-coupled modules), heatmap overlays (SLOC + Fan-Out risk, git churn), keyboard shortcuts, dark mode, semantic TF-IDF search.
- **Languages:** C, C++, Groovy, Java, JavaScript, TypeScript, Kotlin, Objective-C, Ruby, Swift, Python, Go
- **Output:** Standalone interactive HTML app — open in any browser
- **License:** MIT
- **Key Lesson:** Louvain clustering + heatmap overlays for code quality metrics is a powerful pattern for architecture understanding. The standalone web app approach is useful for sharing.

### Tier 3: Security & Analysis

#### 5. Semgrep (9.2K stars) — AST Pattern Matching + Taint Analysis
- **Core Innovation:** tree-sitter → OCaml Generic AST → Intermediate Language pipeline. Patterns look like source code but match semantically. Taint analysis traces data from sources to sinks.
- **Languages:** 35+ for code analysis, 12 for supply chain
- **UI:** VS Code extension (inline findings), AppSec Platform (security dashboards, dependency graphs)
- **License:** LGPL-2.1
- **Key Lesson:** The Generic AST concept (language-agnostic unified representation) is powerful for cross-language analysis.

#### 6. CodeQL (8K stars) — Code as Relational Database
- **Core Innovation:** Represents code as a relational database with tables for expressions, statements, types. Custom QL query language (Datalog-inspired). Full AST + CFG + DFG in the database.
- **Languages:** 15 with deep framework support
- **UI:** VS Code extension (AST viewer, data-flow path expansion), GitHub Code Scanning web UI
- **License:** MIT (queries) / Proprietary (engine)
- **Key Lesson:** The "code as database" approach enables extremely powerful queries. The VS Code AST viewer with bidirectional navigation is excellent.

#### 7. Joern (2.9K stars) — Code Property Graph (Academic Gold Standard)
- **Core Innovation:** Unified Code Property Graph (CPG) = AST + CFG + PDG merged. CPGQL query language (Scala DSL). Seven exportable representations (AST, CFG, CDG, DDG, PDG, CPG14, ALL).
- **Languages:** C, C++, Java, JavaScript, TypeScript, Python, Kotlin, LLVM, x86 binaries
- **Storage:** FlatGraph (in-memory, 25-30% less memory than OverflowDB)
- **UI:** None (REPL + Graphviz export to external tools)
- **License:** Apache-2.0
- **Key Lesson:** CPG is the most complete graph representation but heavy. For most use cases, AST + call graph + import graph is sufficient.

### Tier 4: AI-Native Code Intelligence

#### 8. Code-Graph-RAG (1.9K stars) — Graph + RAG for Code
- **Core Innovation:** tree-sitter → Memgraph knowledge graph → LLM-generated Cypher queries. Natural language code Q&A. Real-time file watching for incremental updates.
- **Languages:** C++, Java, JavaScript, Lua, Python, Rust, TypeScript
- **MCP Server:** First-class MCP integration for Claude Code
- **License:** MIT
- **Key Lesson:** Graph + RAG combination for code understanding is the dominant 2026 pattern. MCP is the integration standard.

#### 9. CodeFuse-CGM (521 stars) — Graph-Aware LLM Attention
- **Core Innovation:** NeurIPS 2025. Feeds code graph structure directly into LLM attention mechanism (replaces causal mask with adjacency-derived mask). CodeT5+ encodes nodes → MLP adapter → LLM embedding space. 512x context compression.
- **Performance:** 44% SWE-Bench-Lite (#1 among open-weight models)
- **R4 Chain:** Rewriter → Retriever → Reranker → Reader (CGM)
- **Key Lesson:** Graph structure in LLM attention is cutting-edge research. The 7 node types / 5 edge types schema is a good reference model.

#### 10. CodePrism (18 stars) + CodeGraph-Rust (141 stars) — MCP-Native Tools
- **CodePrism:** Rust, 20 MCP tools, sub-50ms queries, MIT. AI-generated codebase.
- **CodeGraph-Rust:** Rust, SurrealDB + FAISS, 4 agentic MCP tools, 14 languages via tree-sitter, hybrid search (70% vector + 30% lexical + graph traversal)
- **Key Lesson:** MCP is becoming the standard integration pattern for code intelligence tools. Multiple tools per server is the norm.

### 2026 Trends

1. **MCP as integration standard** — Code-Graph-RAG, CodePrism, CodeGraph-Rust all ship MCP servers
2. **Graph + RAG is the dominant AI pattern** — Parse code into graph, query with NL, synthesize with LLM
3. **Tree-sitter is universal** — Every new project uses it as parsing foundation
4. **Louvain clustering** — Automatic module detection for architecture visualization
5. **PageRank for context selection** — Aider's approach is being adopted widely
6. **Graph-aware attention** — CodeFuse-CGM feeds graph into LLM (cutting edge)

---

## Part 2: Automatos Current State (Honest Assessment)

### What Already Works (Strengths)

Automatos has a **substantial, production-ready CodeGraph system** across backend + frontend:

#### Backend (~2,350 lines)

| Component | File | Lines | Status | Quality |
|-----------|------|-------|--------|---------|
| Core Service | `modules/codegraph/codegraph_service.py` | 1,504 | Working | Production |
| Project Context | `modules/codegraph/project_context.py` | 355 | Working | Good |
| REST API | `api/codegraph.py` | 436 | Working | Complete |
| Unit Tests | `modules/codegraph/tests/test_codegraph_service.py` | 191 | Working | Good |
| Integration Tests | `modules/codegraph/tests/test_codegraph_integration.py` | 147 | Working | Good |
| Test Fixtures | `modules/codegraph/tests/conftest.py` | 255 | Working | Comprehensive |
| DB Models (legacy) | `core/models/code_graph.py` | 55 | Outdated | Needs fix |

**Features implemented:**
- GitHub repository indexing (clone, parse, store) with auth token support
- Python AST parsing (full — using `ast` module)
- TypeScript/JavaScript parsing (regex-based — less accurate)
- Symbol extraction: functions, classes, methods
- Relationship tracking: calls, imports, extends, implements, references
- Semantic search via vector embeddings (EnhancedVectorStore)
- Fuzzy/exact symbol search
- Call graph generation (BFS traversal with configurable depth)
- Project lifecycle management (create, list, delete, reindex)
- Workspace-based multi-tenant isolation
- Query logging for analytics
- Background task support for long-running indexing

#### Frontend (~1,815 lines)

| Component | File | Lines | Status | Quality |
|-----------|------|-------|--------|---------|
| CodeGraph Panel | `components/knowledge/CodeGraphPanel.tsx` | 663 | Working | Production |
| Call Graph Viz | `components/knowledge/CodeGraphVisualization.tsx` | 265 | Working | Good |
| Knowledge Graph | `components/knowledge/KnowledgeGraphVisualizer.tsx` | 496 | Working | Good |
| Settings | `components/settings/CodeGraphSettingsTab.tsx` | 391 | Working | Complete |

**Visualization features:**
- ReactFlow-based interactive call graph with depth (1-5) and direction (in/out/both) controls
- Color-coded nodes: blue=functions, green=classes, purple=methods, orange=imports
- D3.js force-directed knowledge graph with entity type colors
- MiniMap overlay, zoom controls, pan/drag
- Graph type selector: Call Graph, Dependencies, Inheritance
- Node search with entry point selection
- Graph export (PNG from KnowledgeGraphVisualizer)

**Integration points:**
- Tab in `document-management.tsx` (main knowledge hub)
- `search_codebase` tool available to agents
- Jira bug triage recipe uses CodeGraph for symbol search
- Chat integration (CodeWidgetData source type)
- Workflow context (codegraph_project in workflow JSON)
- API client with 8 codegraph methods

**Total: ~4,500+ lines of working CodeGraph code**

### What's Missing (Gaps vs Top Projects)

| Gap | Impact | Who Has It | Priority |
|-----|--------|-----------|----------|
| **1. No tree-sitter parsing** | Limited to 3 languages, TS/JS regex unreliable | Every top project | Critical |
| **2. Database schema mismatch** | Migration creates old tables, service uses new ones via raw SQL | N/A (internal bug) | Critical |
| **3. No incremental indexing** | Full re-index on every change, slow for large repos | Code-Graph-RAG (file watcher), Pathway | High |
| **4. No MCP exposure** | CodeGraph not available to external AI assistants | Code-Graph-RAG, CodePrism, CodeGraph-Rust | High |
| **5. No PageRank context optimization** | Agent tool sends all symbols, not ranked by importance | Aider | High |
| **6. No architecture metrics** | Can't detect modules, coupling, complexity hotspots | Emerge (Louvain, heatmaps) | Medium |
| **7. Basic graph visualization** | ReactFlow call graph is good but lacks code-view sync | Sourcetrail (3 synchronized views) | Medium |
| **8. No graph-RAG integration** | Can't query code structure via natural language | Code-Graph-RAG, CodeGraph-Rust | Medium |

### 5 Bugs Found

| # | Bug | Severity | Location | Fix |
|---|-----|----------|----------|-----|
| 1 | **Schema mismatch** — migration creates `code_symbols`/`code_edges` but service uses `codegraph_projects`/`codegraph_symbols`/etc. | Critical | `alembic/.../add_code_graph.py` vs `codegraph_service.py` | Create proper migration for actual tables |
| 2 | **TypeScript/JavaScript parsing is regex-based** — misses nested functions, arrow functions, destructured imports | High | `codegraph_service.py` (TS/JS parser methods) | Replace with tree-sitter |
| 3 | **Empty placeholder directories** — `analysis/`, `graph/`, `search/` exist with only `__init__.py` | Low | `modules/codegraph/` | Delete or implement planned features |
| 4 | **Relationship matching uses fuzzy fallback** — external dependencies silently skipped | Medium | `codegraph_service.py` | Log warnings, store as "external" relationship type |
| 5 | **No cache invalidation** — re-index deletes everything and re-creates | Medium | `codegraph_service.py` | Add file hash checking for incremental updates |

---

## Part 3: Build vs. Adopt Analysis

### The Question
> "Do I keep or bin the existing CodeGraph module?"

### Verdict: **KEEP (Enhance, Don't Replace)**

#### Why NOT to adopt Code-Graph-RAG / CodePrism / CodeGraph-Rust:

| Concern | Detail |
|---------|--------|
| **Multi-tenant isolation** | None of the 10 projects support workspace-based multi-tenancy. |
| **Agent tool integration** | Your `search_codebase` tool is already wired into agents and workflows (Jira triage). No open-source project has this. |
| **Frontend UI** | You have 1,815 lines of working React components (ReactFlow + D3). Code-Graph-RAG has no UI. Emerge has a standalone HTML app that doesn't integrate. |
| **API completeness** | 9 REST endpoints with workspace isolation, background tasks, auth. Open-source projects are CLI/MCP only. |
| **Settings management** | CodeGraphSettingsTab (391 lines) with LLM provider, embedding model, performance tuning. Nothing comparable in open-source. |
| **Test coverage** | Integration + unit tests with realistic fixtures. Most open-source projects have minimal tests. |
| **Migration cost** | Estimated 60-80 hours to rip out + integrate + retrofit multi-tenancy + restore feature parity. |

#### What TO adopt (techniques, not codebases):

| Technique | Source | Effort | Impact |
|-----------|--------|--------|--------|
| tree-sitter parsing | Tree-sitter, Aider, Code-Graph-RAG | 8h | 14+ languages, accurate TS/JS |
| PageRank context ranking | Aider | 4h | Better agent context, less tokens |
| MCP tool exposure | Code-Graph-RAG, CodePrism | 4h | External AI assistant integration |
| Louvain clustering + heatmaps | Emerge | 6h | Architecture understanding |
| Natural language graph queries | Code-Graph-RAG | 4h | "What functions call the auth module?" |
| Incremental indexing (file hashing) | Code-Graph-RAG | 4h | Faster re-indexing |

**Bottom line:** Your existing implementation is ~4,500 lines of working, multi-tenant, production-ready code with a full React frontend. Adopting an open-source project would cost more than enhancing. Adopt the **techniques** (tree-sitter, PageRank, MCP, Louvain) not codebases.

---

## Existing Frontend Reality

The CodeGraph frontend is **already extensive and fully functional**:

| Component | Lines | What It Does |
|-----------|-------|-------------|
| `CodeGraphPanel.tsx` | 663 | Main container: project management, search (fuzzy + semantic), visualization tab |
| `CodeGraphVisualization.tsx` | 265 | ReactFlow call graph with depth/direction/type controls, color-coded nodes, MiniMap |
| `KnowledgeGraphVisualizer.tsx` | 496 | D3 force-directed entity graph with zoom/export/search, node importance sizing |
| `CodeGraphSettingsTab.tsx` | 391 | Full admin settings (LLM provider, embedding, analysis depth, performance limits) |
| `document-management.tsx` | — | CodeGraph is a tab in the main knowledge hub |

**Visualization libraries already installed:**
- `reactflow` (^11.11.4) — Node/edge flow diagrams
- `d3` (^7.9.0) — Force-directed graphs
- `recharts` (2.8.0) — Charts
- `plotly.js` (2.26.2) — Advanced charting

**Frontend work in this PRD is minimal** — mostly small enhancements to existing components, not new pages.

---

## Part 4: Implementation Plan

### Phase 1: Tree-sitter Integration (8h) — CRITICAL

**What:** Replace Python `ast` module and regex-based TS/JS parsing with tree-sitter for all languages. This is the single highest-impact improvement.

**Why:** Every top project uses tree-sitter. It gets you from 3 languages (Python good, TS/JS bad) to 14+ languages with accurate parsing.

#### Backend Changes

**Install dependency:**
```bash
pip install tree-sitter tree-sitter-language-pack
```

**New file: `orchestrator/modules/codegraph/parsers/treesitter_parser.py`**

```python
"""
Tree-sitter based multi-language code parser.
Replaces ast module (Python) and regex (TypeScript/JavaScript) with
tree-sitter for accurate, cross-language symbol extraction.
"""
from tree_sitter_language_pack import get_language, get_parser

class TreeSitterParser:
    """Parse source code using tree-sitter grammars."""

    SUPPORTED_LANGUAGES = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.tsx': 'tsx', '.jsx': 'javascript', '.java': 'java',
        '.go': 'go', '.rs': 'rust', '.cpp': 'cpp', '.c': 'c',
        '.rb': 'ruby', '.php': 'php', '.swift': 'swift',
        '.kt': 'kotlin', '.cs': 'c_sharp',
    }

    def parse_file(self, file_path: str, content: str) -> ParseResult:
        """Extract symbols and relationships from a source file."""
        ext = Path(file_path).suffix
        lang_name = self.SUPPORTED_LANGUAGES.get(ext)
        if not lang_name:
            return ParseResult(symbols=[], relationships=[])

        parser = get_parser(lang_name)
        tree = parser.parse(content.encode())

        symbols = self._extract_symbols(tree.root_node, file_path, content)
        relationships = self._extract_relationships(tree.root_node, file_path, content, symbols)

        return ParseResult(symbols=symbols, relationships=relationships)

    def _extract_symbols(self, node, file_path, content):
        """Walk AST and extract function/class/method definitions."""
        symbols = []
        # Language-agnostic symbol extraction using node types:
        # function_definition, class_definition, method_definition,
        # function_declaration, class_declaration, etc.
        symbol_node_types = {
            'function_definition', 'function_declaration',
            'class_definition', 'class_declaration',
            'method_definition', 'method_declaration',
            'arrow_function',  # JS/TS
            'impl_item',       # Rust
            'interface_declaration',  # TS/Java
        }
        self._walk_tree(node, file_path, content, symbols, symbol_node_types)
        return symbols

    def _extract_relationships(self, node, file_path, content, symbols):
        """Extract calls, imports, extends from AST."""
        relationships = []
        # call_expression → calls relationship
        # import_statement → imports relationship
        # class_heritage / extends_clause → extends relationship
        ...
        return relationships
```

**Modify: `orchestrator/modules/codegraph/codegraph_service.py`**

Replace `_parse_python_file()` and `_parse_typescript_file()` with unified tree-sitter parser:

```python
# BEFORE: Two separate methods with different quality
def _parse_python_file(self, ...):  # Python ast module
def _parse_typescript_file(self, ...):  # Regex-based

# AFTER: Single unified parser
from .parsers.treesitter_parser import TreeSitterParser

def _parse_file(self, file_path: str, content: str) -> ParseResult:
    return self.treesitter_parser.parse_file(file_path, content)
```

**Files to create:**
- `orchestrator/modules/codegraph/parsers/__init__.py`
- `orchestrator/modules/codegraph/parsers/treesitter_parser.py`

**Files to modify:**
- `orchestrator/modules/codegraph/codegraph_service.py` — replace parser methods
- `requirements.txt` — add `tree-sitter`, `tree-sitter-language-pack`

---

### Phase 2: Fix Schema + Incremental Indexing (4h) — CRITICAL

**What:** Fix the database schema mismatch and add file-hash-based incremental indexing.

#### 2.1 Fix Schema Mismatch

**Current problem:** The Alembic migration (`20250812_add_code_graph.py`) creates `code_symbols` and `code_edges`, but `codegraph_service.py` uses `codegraph_projects`, `codegraph_symbols`, `codegraph_files`, `codegraph_relationships`, `codegraph_query_logs` — created via raw SQL.

**Fix:** Create a proper migration for the actual tables:

```sql
-- New migration: replace old tables with correct schema
DROP TABLE IF EXISTS code_edges;
DROP TABLE IF EXISTS code_symbols;

CREATE TABLE IF NOT EXISTS codegraph_projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) DEFAULT 'github',
    source_url TEXT,
    branch VARCHAR(255) DEFAULT 'main',
    status VARCHAR(50) DEFAULT 'pending',
    total_files INTEGER DEFAULT 0,
    total_symbols INTEGER DEFAULT 0,
    total_relationships INTEGER DEFAULT 0,
    language VARCHAR(50),
    last_indexed TIMESTAMP,
    index_duration_seconds FLOAT,
    exclude_patterns TEXT[] DEFAULT '{}',
    auto_reindex BOOLEAN DEFAULT FALSE,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS codegraph_files (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64),  -- SHA-256 for incremental indexing
    file_size INTEGER,
    lines_of_code INTEGER,
    language VARCHAR(50),
    workspace_id UUID NOT NULL,
    indexed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS codegraph_symbols (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    symbol_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    qualified_name TEXT,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    signature TEXT,
    docstring TEXT,
    code_snippet TEXT,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}',
    workspace_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS codegraph_relationships (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES codegraph_projects(id) ON DELETE CASCADE,
    from_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
    to_symbol_id INTEGER REFERENCES codegraph_symbols(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    workspace_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_codegraph_symbols_project ON codegraph_symbols(project_id);
CREATE INDEX idx_codegraph_symbols_name ON codegraph_symbols(name);
CREATE INDEX idx_codegraph_files_project ON codegraph_files(project_id);
CREATE INDEX idx_codegraph_files_hash ON codegraph_files(file_hash);
```

#### 2.2 Incremental Indexing

**Modify: `orchestrator/modules/codegraph/codegraph_service.py`**

Add file hash checking to skip unchanged files during re-index:

```python
import hashlib

async def _should_reparse_file(self, project_id: int, file_path: str, content: str) -> bool:
    """Check if file has changed since last index via SHA-256 hash."""
    file_hash = hashlib.sha256(content.encode()).hexdigest()
    existing = await self._get_file_hash(project_id, file_path)
    return existing != file_hash

async def reindex_project(self, project_id: int):
    """Incremental re-index: only re-parse changed files."""
    for file_path, content in self._discover_files(repo_path):
        if await self._should_reparse_file(project_id, file_path, content):
            # Parse and update
            await self._delete_file_symbols(project_id, file_path)
            result = self._parse_file(file_path, content)
            await self._store_symbols(project_id, file_path, result)
            await self._update_file_hash(project_id, file_path, content)
        # else: skip unchanged file
```

**Files to modify:**
- `orchestrator/modules/codegraph/codegraph_service.py` — add incremental logic
- Create new Alembic migration

---

### Phase 3: MCP Tool Exposure (4h) — HIGH

**What:** Expose CodeGraph as MCP tools so external AI assistants (Claude Desktop, Cursor, etc.) can search and analyze indexed codebases through Automatos.

**Why:** MCP is the dominant 2026 integration pattern. Code-Graph-RAG, CodePrism, and CodeGraph-Rust all ship MCP servers.

#### MCP Tool Definitions

```python
# 4 MCP tools (following CodeGraph-Rust pattern)
tools = [
    {
        "name": "search_codebase",
        "description": "Search indexed codebase for symbols (functions, classes, methods) by name or semantic similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (symbol name or natural language)"},
                "project": {"type": "string", "description": "Project name"},
                "search_type": {"type": "string", "enum": ["fuzzy", "semantic"], "default": "fuzzy"},
                "symbol_type": {"type": "string", "enum": ["function", "class", "method", "all"], "default": "all"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query", "project"]
        }
    },
    {
        "name": "get_call_graph",
        "description": "Get the call graph for a symbol, showing what it calls and what calls it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name to get call graph for"},
                "project": {"type": "string", "description": "Project name"},
                "depth": {"type": "integer", "default": 2, "maximum": 5},
                "direction": {"type": "string", "enum": ["outgoing", "incoming", "both"], "default": "both"}
            },
            "required": ["symbol", "project"]
        }
    },
    {
        "name": "analyze_architecture",
        "description": "Get high-level architecture overview: modules, key classes, dependency patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "focus_path": {"type": "string", "description": "Optional: focus on a specific directory"}
            },
            "required": ["project"]
        }
    },
    {
        "name": "find_dependencies",
        "description": "Find all files/symbols that depend on a given symbol, or that a symbol depends on.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "project": {"type": "string"},
                "direction": {"type": "string", "enum": ["dependents", "dependencies", "both"], "default": "both"}
            },
            "required": ["symbol", "project"]
        }
    }
]
```

**Files to modify:**
- `orchestrator/modules/tools/services/database_tool_integration.py` (or MCP gateway) — add tool definitions
- `orchestrator/modules/codegraph/codegraph_service.py` — add `analyze_architecture()` and `find_dependencies()` methods

---

### Phase 4: PageRank Context Optimization (4h) — HIGH

**What:** When the `search_codebase` agent tool is invoked, use PageRank (like Aider's repo map) to rank symbols by importance and return only the most relevant ones within a token budget.

**Why:** Currently the agent tool returns raw search results. Aider proved that PageRank ranking improves LLM context quality dramatically (4-6% utilization vs 54-70%).

#### Backend Changes

**New file: `orchestrator/modules/codegraph/ranking/pagerank_ranker.py`**

```python
"""
PageRank-based symbol importance ranking.
Inspired by Aider's repo map — ranks symbols by how frequently
they are referenced across the codebase.
"""
import networkx as nx

class PageRankRanker:
    """Rank code symbols by structural importance using PageRank."""

    def rank_symbols(
        self,
        symbols: List[Dict],
        relationships: List[Dict],
        token_budget: int = 2048
    ) -> List[Dict]:
        """
        Build a dependency graph and rank symbols by PageRank.
        Returns symbols sorted by importance, fitting within token_budget.
        """
        G = nx.DiGraph()

        # Add symbols as nodes
        for sym in symbols:
            G.add_node(sym['id'], **sym)

        # Add relationships as edges
        for rel in relationships:
            G.add_edge(rel['from_symbol_id'], rel['to_symbol_id'],
                      type=rel['relationship_type'])

        # Run PageRank
        try:
            ranks = nx.pagerank(G, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            ranks = {n: 1.0 / len(G) for n in G.nodes()}

        # Sort by rank, fit within token budget
        ranked = sorted(symbols, key=lambda s: ranks.get(s['id'], 0), reverse=True)

        result = []
        tokens_used = 0
        for sym in ranked:
            sym_tokens = len(sym.get('signature', '').split()) * 2  # rough estimate
            if tokens_used + sym_tokens > token_budget:
                break
            sym['importance_rank'] = ranks.get(sym['id'], 0)
            result.append(sym)
            tokens_used += sym_tokens

        return result
```

**Modify: `orchestrator/modules/agents/services/agent_platform_tools.py`**

In the `search_codebase` tool, use PageRank ranking before returning results:

```python
# After getting search results, rank by importance
ranker = PageRankRanker()
ranked_results = ranker.rank_symbols(results, relationships, token_budget=2048)
```

**Files to create:**
- `orchestrator/modules/codegraph/ranking/__init__.py`
- `orchestrator/modules/codegraph/ranking/pagerank_ranker.py`

**Files to modify:**
- `orchestrator/modules/agents/services/agent_platform_tools.py` — use ranker
- `requirements.txt` — add `networkx` (may already be present)

---

### Phase 5: Architecture Metrics & Visualization (6h) — MEDIUM

**What:** Add Louvain modularity clustering, complexity metrics, and heatmap overlays to the existing graph visualization. Inspired by Emerge.

#### 5.1 Backend: Architecture Analysis

**New file: `orchestrator/modules/codegraph/analysis/architecture_analyzer.py`**

```python
"""
Architecture analysis — module detection, coupling metrics, hotspot identification.
Inspired by Emerge's Louvain clustering and heatmap approach.
"""

class ArchitectureAnalyzer:
    """Analyze codebase architecture from code graph."""

    async def analyze(self, project_id: int) -> ArchitectureReport:
        """Run full architecture analysis."""
        # Build networkx graph from codegraph relationships
        G = await self._build_graph(project_id)

        # Louvain community detection (auto-detect modules)
        communities = self._detect_communities(G)

        # Compute metrics per file/symbol
        metrics = {
            'fan_in': nx.in_degree_centrality(G),
            'fan_out': nx.out_degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'pagerank': nx.pagerank(G),
        }

        # Identify hotspots (high fan-out + high centrality)
        hotspots = self._identify_hotspots(metrics)

        # Detect circular dependencies
        cycles = list(nx.simple_cycles(G))

        return ArchitectureReport(
            communities=communities,
            metrics=metrics,
            hotspots=hotspots,
            cycles=cycles[:20],  # limit
            total_nodes=len(G.nodes()),
            total_edges=len(G.edges()),
        )
```

#### 5.2 API Endpoint

```python
@router.get("/projects/{project_id}/architecture")
async def get_architecture_analysis(project_id: int, ...):
    """Get architecture analysis: modules, hotspots, coupling metrics."""
```

#### 5.3 Frontend: Enhance Existing Visualization (small changes)

**File:** `frontend/components/knowledge/CodeGraphVisualization.tsx` (MODIFY, not create new)

Add to existing ReactFlow visualization:
- Louvain cluster colors on nodes (different color per detected module)
- Heatmap toggle (color nodes by complexity/coupling score)
- Hotspot badges on high-risk nodes
- Cycle highlight (red edges for circular dependencies)

**Files to create:**
- `orchestrator/modules/codegraph/analysis/architecture_analyzer.py`

**Files to modify:**
- `orchestrator/api/codegraph.py` — add architecture endpoint
- `frontend/components/knowledge/CodeGraphVisualization.tsx` — add cluster colors, heatmap toggle

---

### Phase 6: Natural Language Code Queries (4h) — MEDIUM

**What:** Let users ask natural language questions about their codebase. Translate questions to graph queries, execute, and return structured answers.

**Why:** Code-Graph-RAG proves this is the dominant pattern for AI-native code intelligence.

#### Backend Changes

**New file: `orchestrator/modules/codegraph/search/nl_code_search.py`**

```python
"""
Natural language code search — translates questions to
graph queries using LLM, executes, and returns answers.
"""

class NLCodeSearch:
    """Natural language interface to code graph."""

    async def query(self, question: str, project_id: int) -> Dict:
        """
        Answer natural language questions about code structure.

        Examples:
        - "What functions call the authenticate method?"
        - "Show me all classes that implement the BaseHandler interface"
        - "What are the dependencies of the user module?"
        """
        # Step 1: Classify query type
        query_type = await self._classify_query(question)

        # Step 2: Generate appropriate SQL/graph query
        if query_type == 'call_graph':
            results = await self._query_call_graph(question, project_id)
        elif query_type == 'dependency':
            results = await self._query_dependencies(question, project_id)
        elif query_type == 'search':
            results = await self._semantic_search(question, project_id)
        else:
            results = await self._general_query(question, project_id)

        # Step 3: Generate natural language answer
        answer = await self._generate_answer(question, results)

        return {
            "question": question,
            "answer": answer,
            "results": results,
            "query_type": query_type
        }
```

**API Endpoint:**
```python
@router.post("/projects/{project_id}/ask")
async def ask_code_question(project_id: int, body: CodeQuestionRequest, ...):
    """Ask a natural language question about the codebase."""
```

**Frontend:** Add a "Ask about code" input to `CodeGraphPanel.tsx` (small addition to the existing Search tab).

---

### Phase 7: Enhanced Graph Visualization (6h) — MEDIUM

**What:** Bring the existing ReactFlow visualization closer to Sourcetrail's synchronized views by adding a code snippet panel that syncs with graph selection.

#### 7.1 Frontend: Code Snippet Sync Panel

**File:** `frontend/components/knowledge/CodeGraphVisualization.tsx` (MODIFY existing)

Add a code panel below or beside the graph:
- When user clicks a node in the graph → code panel shows the symbol's source code
- Syntax-highlighted code snippet
- File path + line number
- "View full file" link
- Shows docstring if available

This brings the visualization closer to Sourcetrail's 2-view pattern (graph + code) without needing to build a full 3-view desktop app.

#### 7.2 Frontend: Minimap Enhancement

Add to the existing ReactFlow minimap:
- File tree sidebar showing indexed files
- Click a file → highlights all its symbols in the graph
- Shows file-level metrics (LOC, symbol count)

**Files to modify:**
- `frontend/components/knowledge/CodeGraphVisualization.tsx` — add code panel, file tree

---

### Phase 8: Bug Fixes + Cleanup (3h) — HIGH

Fix the 5 bugs identified during the code audit.

#### Bug 1: Schema Mismatch (Critical)
Addressed in Phase 2.

#### Bug 2: TS/JS Regex Parsing (High)
Addressed in Phase 1 (tree-sitter replaces regex).

#### Bug 3: Empty Placeholder Directories (Low)
Delete `analysis/`, `graph/`, `search/` empty directories (or use them in Phases 5-6).

#### Bug 4: Relationship Fuzzy Fallback (Medium)
**File:** `codegraph_service.py`
Add "external" relationship type and log warnings:
```python
if not found_symbol:
    relationships.append(CodeRelationship(
        from_symbol=current_symbol,
        to_symbol_name=ref_name,
        relationship_type='external_reference',
        metadata={'status': 'unresolved', 'reason': 'not_in_project'}
    ))
    logger.debug(f"Unresolved reference: {ref_name}")
```

#### Bug 5: No Cache Invalidation (Medium)
Addressed in Phase 2 (file hash-based incremental indexing).

---

## Priority Matrix

| Phase | Feature | Impact | Effort | Priority |
|-------|---------|--------|--------|----------|
| 1 | Tree-sitter Integration | Critical (3→14+ languages) | 8h | P0 — Do First |
| 2 | Fix Schema + Incremental Indexing | Critical (correctness + performance) | 4h | P0 — Do First |
| 8 | Bug Fixes + Cleanup | High (stability) | 3h | P0 — Do First |
| 3 | MCP Tool Exposure | High (integration channel) | 4h | P1 — Do Second |
| 4 | PageRank Context Optimization | High (agent quality) | 4h | P1 — Do Second |
| 5 | Architecture Metrics + Viz | Medium (understanding) | 6h | P2 — Do Third |
| 6 | Natural Language Code Queries | Medium (AI-native) | 4h | P2 — Do Third |
| 7 | Enhanced Graph Visualization | Medium (UX) | 6h | P3 — Future |

**MVP (Phases 1-2, 8):** 15 hours — Gets tree-sitter, fixes bugs, adds incremental indexing
**Core (+ Phases 3-4):** 23 hours — Adds MCP + PageRank for competitive parity
**Full (All phases):** 39 hours — Best-in-class for a multi-tenant code intelligence platform

---

## Competitive Comparison (After Implementation)

| Feature | Automatos (Current) | Automatos (After PRD-62) | Code-Graph-RAG | Emerge | Sourcetrail |
|---------|---------------------|--------------------------|----------------|--------|-------------|
| Parsing | Python AST + regex | tree-sitter (14+ languages) | tree-sitter | Custom parsers | Clang/JDT |
| Languages | 3 (Python, JS, TS) | 14+ | 7 | 12 | C/C++, Java |
| Graph Type | Call + Import | Call + Import + Architecture | Knowledge Graph | Dependency/Inheritance | Symbol Relationship |
| Storage | PostgreSQL (raw SQL) | PostgreSQL (proper migration) | Memgraph | In-memory → HTML | SQLite |
| Graph Viz | ReactFlow + D3 | ReactFlow + D3 + heatmaps + code sync | None (Memgraph Lab) | D3 force-directed | Sugiyama (desktop) |
| NL Queries | None | LLM-powered | Cypher via LLM | None | None |
| MCP Tools | None | 4 tools | Yes | None | None |
| Context Ranking | None | PageRank | None | None | None |
| Multi-tenant | Full isolation | Full isolation | None | None | None |
| Agent Integration | search_codebase tool | Enhanced + MCP | MCP only | None | IDE plugins |
| Incremental Index | No | File hash-based | File watcher | No | Changed files |
| Architecture Metrics | None | Louvain + coupling + hotspots | None | Louvain + heatmaps | None |
| Web UI | Full React app | Full React app + code panel | None | Standalone HTML | Desktop (Qt6) |
| License | Proprietary | Proprietary | MIT | MIT | GPL-3.0 |

---

## Files Summary

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `orchestrator/modules/codegraph/parsers/__init__.py` | 1 | Package init |
| `orchestrator/modules/codegraph/parsers/treesitter_parser.py` | 1 | tree-sitter multi-language parser |
| `orchestrator/modules/codegraph/ranking/__init__.py` | 4 | Package init |
| `orchestrator/modules/codegraph/ranking/pagerank_ranker.py` | 4 | PageRank importance ranking |
| `orchestrator/modules/codegraph/analysis/architecture_analyzer.py` | 5 | Louvain clustering + metrics |
| `orchestrator/modules/codegraph/search/nl_code_search.py` | 6 | Natural language code queries |
| Alembic migration (schema fix) | 2 | Proper codegraph tables |

### Modified Files
| File | Phases | Changes |
|------|--------|---------|
| `orchestrator/modules/codegraph/codegraph_service.py` | 1, 2, 8 | Replace parsers with tree-sitter, add incremental indexing, fix relationship handling |
| `orchestrator/api/codegraph.py` | 5, 6 | Add architecture and NL query endpoints |
| `orchestrator/modules/agents/services/agent_platform_tools.py` | 4 | Use PageRank ranker for search_codebase tool |
| `orchestrator/modules/tools/services/database_tool_integration.py` | 3 | Add MCP tool definitions |
| `frontend/components/knowledge/CodeGraphVisualization.tsx` | 5, 7 | Add cluster colors, heatmap toggle, code snippet panel |
| `frontend/components/knowledge/CodeGraphPanel.tsx` | 6 | Add "Ask about code" input |
| `requirements.txt` | 1, 4 | Add tree-sitter, tree-sitter-language-pack, networkx |

### Deleted Files
| File | Reason |
|------|--------|
| `orchestrator/modules/codegraph/analysis/__init__.py` (empty) | Replace with real implementation in Phase 5 |
| `orchestrator/modules/codegraph/graph/__init__.py` (empty) | Unused placeholder |
| `orchestrator/modules/codegraph/search/__init__.py` (empty) | Replace with real implementation in Phase 6 |

---

## Success Criteria

- [ ] Phase 1: Tree-sitter parses 14+ languages accurately, TypeScript arrow functions no longer missed
- [ ] Phase 2: Database uses proper migration-managed schema, re-indexing only processes changed files
- [ ] Phase 3: External MCP clients can search codebases and get call graphs through Automatos
- [ ] Phase 4: Agent tool returns PageRank-ranked results within token budget
- [ ] Phase 5: Architecture analysis detects modules, coupling hotspots, and circular dependencies
- [ ] Phase 6: Users can ask "What calls the authenticate function?" and get structured answers
- [ ] Phase 7: Clicking a graph node shows the source code snippet in a synced panel
- [ ] Phase 8: All 5 bugs verified fixed, empty directories removed

---

## Out of Scope (Future PRDs)

- Full Code Property Graph (AST + CFG + PDG unified, like Joern) — heavy, not needed for most use cases
- Graph-aware LLM attention (CodeFuse-CGM approach) — cutting-edge research, not production-ready
- IDE plugins (VS Code, IntelliJ) — would need separate extension
- Local file system indexing (currently GitHub-only) — could add for on-premise
- Multi-repo analysis (cross-repository relationships)
- Git history analysis (code churn, change coupling over time)
- Security-focused analysis (taint tracking, vulnerability detection)
- Fine-tuned code understanding model

---

**Estimated Total Effort:** 39-50 hours
**MVP (Phases 1-2, 8):** 15 hours
**Priority:** High
**Dependencies:** PRD-11 (completed)
