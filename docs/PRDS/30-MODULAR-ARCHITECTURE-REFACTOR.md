# PRD-30: Modular Architecture Refactoring
## Complete Codebase Restructuring for Standalone, Sellable Modules

**Version:** 1.0.0
**Status:** PLANNING
**Priority:** CRITICAL
**Estimated Effort:** 4-6 weeks
**Author:** Automatos AI Platform Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current State Analysis](#3-current-state-analysis)
4. [Target Architecture](#4-target-architecture)
5. [Module Specifications](#5-module-specifications)
   - [5.1 RAG Module](#51-rag-module)
   - [5.2 Memory Module](#52-memory-module)
   - [5.3 Agents Module](#53-agents-module)
   - [5.4 Tools Module](#54-tools-module)
   - [5.5 Reasoning Module](#55-reasoning-module)
   - [5.6 Evaluation Module](#56-evaluation-module)
6. [Shared Infrastructure](#6-shared-infrastructure)
7. [Migration Plan](#7-migration-plan)
8. [Files to Delete](#8-files-to-delete)
9. [Testing Strategy](#9-testing-strategy)
10. [Rollback Plan](#10-rollback-plan)
11. [Task Checklist](#11-task-checklist)

---

## 1. Executive Summary

### 1.1 Purpose

Transform the Automatos AI Platform from a monolithic, scattered codebase into a **modular architecture** where each core capability (RAG, Memory, Agents, Tools) is:

1. **Self-contained** - Can run independently
2. **Sellable** - Can be packaged as a standalone product
3. **Shared** - Used by all platform consumers (Chatbot, Workflows, Agents, Third-party)
4. **Testable** - Has clear boundaries and interfaces
5. **Maintainable** - Single source of truth, no duplication

### 1.2 Business Value

| Benefit | Impact |
|---------|--------|
| **Reduced Maintenance** | Fix bugs once, not in 5 places |
| **Faster Development** | Clear module boundaries |
| **Revenue Streams** | Sell RAG, Memory, Agents as products |
| **Third-party Integration** | Clean APIs for external use |
| **Team Scaling** | Different teams own different modules |

### 1.3 Success Metrics

- [ ] Zero duplicate implementations
- [ ] Each module has < 5 external dependencies
- [ ] Module can be imported in 1 line
- [ ] 90%+ test coverage per module
- [ ] API response time unchanged or improved

---

## 2. Problem Statement

### 2.1 Current Issues

#### Issue 1: Massive Code Duplication
```
CHUNKING: 4 different implementations (1,706 lines duplicated)
VECTOR STORE: 4 different implementations (2,402 lines duplicated)
MEMORY: 10 scattered files (4,000+ lines scattered)
RAG RETRIEVAL: 6 different implementations (3,000+ lines duplicated)
```

#### Issue 2: Scattered Functionality
```
"Where is RAG code?"
- services/rag_service.py
- services/semantic_chunker.py
- context_engineering/chunking.py
- context_engineering/chunking/semantic_chunker.py
- context_engineering/vector_store.py
- context_engineering/retrieval/vector_store_enhanced.py
- context_engineering/context_retriever.py
- context_engineering/retrieval/context_retrieval_engine.py
- context_engineering/context_optimizer.py
- utils/document_manager.py
- api/documents.py (embedded)
```

**Answer: EVERYWHERE. This is the problem.**

#### Issue 3: Tight Coupling
```python
# Current: Everything imports everything
from services.rag_service import get_rag_service
from context_engineering.chunking.semantic_chunker import SemanticChunker
from context_engineering.context_optimizer import ContextOptimizer
from utils.document_manager import DocumentManager
from context_engineering.retrieval.vector_store_enhanced import EnhancedVectorStore
# 5 different imports for one RAG operation!
```

#### Issue 4: No Clear Ownership
- Who owns RAG? services/? context_engineering/? utils/?
- Who owns Memory? services/? memory/? core/?
- Who owns Agents? services/? multi_agent/? core/?

---

## 3. Current State Analysis

### 3.1 Directory Structure (Current)

```
orchestrator/                          # 100+ files, no clear organization
├── api/                               # 25+ endpoints, mixed concerns
├── context_engineering/               # RAG + Context (scattered)
│   ├── chunking/                      # Chunking (good)
│   ├── mathematical_foundations/      # Math (good, keep here)
│   ├── retrieval/                     # Vector stores (duplicated)
│   ├── chunking.py                    # DUPLICATE of chunking/
│   ├── context_optimizer.py           # Good, keep
│   ├── context_retriever.py           # DUPLICATE of retrieval/
│   ├── embeddings.py                  # DUPLICATE of llm_provider/
│   ├── vector_store.py                # DUPLICATE of retrieval/
│   └── ... 10 more files
├── core/                              # Orchestration + helpers
│   ├── llm/                           # LLM orchestration
│   ├── _vector_store_helper.py        # DUPLICATE
│   ├── memory_prompt_injector.py      # Memory (scattered)
│   └── ... 15 more files
├── credentials/                       # Good, isolated
├── database/                          # Good, isolated
├── evaluation/                        # Good structure
├── memory/                            # Memory types (partial)
├── models/                            # Good, isolated
├── multi_agent/                       # Good structure
├── reasoning/                         # Good structure
├── seeds/                             # Good, isolated
├── services/                          # DUMPING GROUND - 50+ files!
│   ├── chat/                          # Chat services
│   ├── llm_provider/                  # LLM clients (good)
│   ├── rag_service.py                 # RAG (wrapper)
│   ├── semantic_chunker.py            # DUPLICATE
│   ├── memory_knowledge_system.py     # Memory (1362 lines!)
│   ├── agent_factory.py               # Agents (2142 lines!)
│   └── ... 40 more files
└── utils/                             # Utilities (mixed)
    ├── document_manager.py            # RAG (embedded chunking)
    └── ... 9 more files
```

### 3.2 Duplication Analysis

#### CHUNKING - 4 Implementations

| File | Lines | Has Math? | Status |
|------|-------|-----------|--------|
| `services/semantic_chunker.py` | 400 | ❌ No | DELETE |
| `context_engineering/chunking.py` | 429 | ❌ No | DELETE |
| `context_engineering/chunking/semantic_chunker.py` | 477 | ✅ Yes | **KEEP** |
| `utils/document_manager.py` (embedded) | ~150 | ❌ No | EXTRACT |

**Total duplicated lines: 979**

#### VECTOR STORE - 4 Implementations

| File | Lines | Features | Status |
|------|-------|----------|--------|
| `context_engineering/vector_store.py` | 529 | Basic pgvector | MERGE |
| `context_engineering/retrieval/vector_store_enhanced.py` | 672 | Hybrid, ranking, math | **KEEP** |
| `core/_vector_store_helper.py` | ~100 | Helper functions | DELETE |
| `api/documents.py` (embedded) | ~200 | Search logic | EXTRACT |

**Total duplicated lines: 829**

#### CONTEXT RETRIEVAL - 3 Implementations

| File | Lines | Features | Status |
|------|-------|----------|--------|
| `context_engineering/context_retriever.py` | 585 | Basic retrieval | MERGE |
| `context_engineering/retrieval/context_retrieval_engine.py` | 655 | Advanced, multi-strategy | **KEEP** |
| `context_engineering/context_optimizer.py` | 928 | Knapsack, MMR, entropy | **KEEP** |

#### EMBEDDINGS - 2 Implementations

| File | Lines | Features | Status |
|------|-------|----------|--------|
| `context_engineering/embeddings.py` | 364 | SentenceTransformer, OpenAI | DELETE |
| `services/llm_provider/embedding_manager.py` | 217 | Centralized, multi-provider | **KEEP** |

#### MEMORY - 10 Scattered Files

| File | Lines | Location | Status |
|------|-------|----------|--------|
| `memory/manager.py` | ~400 | memory/ | KEEP |
| `memory/augmentation.py` | ~300 | memory/ | KEEP |
| `memory/consolidation.py` | ~250 | memory/ | KEEP |
| `memory/access_patterns.py` | ~200 | memory/ | KEEP |
| `memory/memory_types.py` | ~150 | memory/ | KEEP |
| `services/memory_knowledge_system.py` | 1362 | services/ | MERGE |
| `core/memory_prompt_injector.py` | ~200 | core/ | MERGE |
| `core/workflow_memory_integrator.py` | ~300 | core/ | MERGE |
| `services/chat/memory_injector.py` | ~150 | chat/ | MERGE |

**Total memory-related lines: ~3,312 across 10 files**

---

## 4. Target Architecture

### 4.1 New Directory Structure

```
orchestrator/
│
├── modules/                           # 🎯 STANDALONE PRODUCTS
│   │
│   ├── search/                       # 🔥 CORE SEARCH ENGINE (Phase 1a)
│   │   ├── __init__.py               # Public API
│   │   ├── service.py                # SearchService class
│   │   ├── config.py                 # Search configuration
│   │   │
│   │   ├── vector_store/             # pgvector operations
│   │   │   ├── __init__.py
│   │   │   ├── store.py              # PgVectorStore
│   │   │   ├── indexing.py           # Index management
│   │   │   └── queries.py            # Query builders
│   │   │
│   │   ├── embeddings/               # Embedding generation
│   │   │   ├── __init__.py
│   │   │   ├── manager.py            # EmbeddingManager
│   │   │   └── providers.py          # OpenAI, HuggingFace, etc.
│   │   │
│   │   ├── retrieval/                # Search algorithms
│   │   │   ├── __init__.py
│   │   │   ├── vector_search.py      # Cosine similarity
│   │   │   ├── hybrid_search.py      # Vector + BM25
│   │   │   └── reranking.py          # Cross-encoder reranking
│   │   │
│   │   ├── optimization/             # Mathematical optimization
│   │   │   ├── __init__.py
│   │   │   ├── context_optimizer.py  # Main optimizer
│   │   │   ├── knapsack.py           # Token budget
│   │   │   ├── mmr.py                # Diversity (MMR)
│   │   │   └── entropy.py            # Information theory
│   │   │
│   │   └── tests/
│   │
│   ├── rag/                          # RAG MODULE (Phase 1b) - uses search/
│   │   ├── __init__.py               # Public API
│   │   ├── service.py                # RAGService class
│   │   ├── config.py                 # RAG configuration
│   │   │
│   │   ├── chunking/                 # Document chunking
│   │   │   ├── __init__.py
│   │   │   ├── semantic.py           # SemanticChunker
│   │   │   ├── strategies.py         # ChunkingStrategy enum
│   │   │   └── metadata.py           # ChunkMetadata
│   │   │
│   │   ├── ingestion/                # Document ingestion
│   │   │   ├── __init__.py
│   │   │   ├── processor.py          # DocumentProcessor
│   │   │   ├── handlers/             # File type handlers
│   │   │   │   ├── pdf.py
│   │   │   │   ├── markdown.py
│   │   │   │   ├── docx.py
│   │   │   │   └── text.py
│   │   │   └── pipeline.py           # Ingestion pipeline
│   │   │
│   │   └── tests/
│   │
│   ├── knowledge/                    # KNOWLEDGE BASE MODULE (Phase 1c) - uses search/
│   │   ├── __init__.py               # Public API
│   │   ├── service.py                # KnowledgeService class
│   │   ├── config.py
│   │   │
│   │   ├── graph/                    # Knowledge graph
│   │   │   ├── __init__.py
│   │   │   ├── builder.py            # Graph construction
│   │   │   ├── traversal.py          # Graph queries
│   │   │   └── entities.py           # Entity extraction
│   │   │
│   │   ├── storage/                  # Knowledge storage
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py
│   │   │   └── neo4j.py              # Future: graph DB
│   │   │
│   │   └── tests/
│   │
│   ├── nl_to_sql/                    # NL-TO-SQL MODULE (Phase 1d) - uses search/
│   │   ├── __init__.py               # Public API
│   │   ├── service.py                # NLToSQLService class
│   │   ├── config.py
│   │   │
│   │   ├── schema/                   # Schema awareness
│   │   │   ├── __init__.py
│   │   │   ├── introspection.py      # DB schema discovery
│   │   │   ├── embeddings.py         # Schema embeddings
│   │   │   └── provider.py           # SchemaProvider
│   │   │
│   │   ├── query/                    # Query generation
│   │   │   ├── __init__.py
│   │   │   ├── builder.py            # SQL builder
│   │   │   ├── validator.py          # SQL validation
│   │   │   └── executor.py           # Safe execution
│   │   │
│   │   └── tests/
│   │
│   ├── codegraph/                    # CODEGRAPH MODULE (Phase 1e) - uses search/
│   │   ├── __init__.py               # Public API
│   │   ├── service.py                # CodeGraphService class
│   │   ├── config.py
│   │   │
│   │   ├── analysis/                 # Code analysis
│   │   │   ├── __init__.py
│   │   │   ├── parser.py             # AST parsing
│   │   │   ├── dependencies.py       # Dependency analysis
│   │   │   └── metrics.py            # Code metrics
│   │   │
│   │   ├── graph/                    # Code graph
│   │   │   ├── __init__.py
│   │   │   ├── builder.py            # Graph construction
│   │   │   ├── queries.py            # Graph queries
│   │   │   └── visualizer.py         # Graph visualization
│   │   │
│   │   ├── search/                   # Code search
│   │   │   ├── __init__.py
│   │   │   ├── semantic.py           # Semantic code search
│   │   │   └── structural.py         # Structural search
│   │   │
│   │   └── tests/
│   │
│   ├── learning/                     # LEARNING MODULE (Phase 5.5) - cross-cutting
│   │   ├── __init__.py               # Public API
│   │   ├── service.py                # LearningService class
│   │   ├── config.py
│   │   │
│   │   ├── patterns/                 # Pattern recognition
│   │   │   ├── __init__.py
│   │   │   ├── detector.py           # Pattern detection
│   │   │   ├── extractor.py          # Pattern extraction
│   │   │   └── storage.py            # Pattern storage
│   │   │
│   │   ├── playbooks/                # Playbook mining
│   │   │   ├── __init__.py
│   │   │   ├── miner.py              # Workflow pattern mining
│   │   │   ├── templates.py          # Playbook templates
│   │   │   └── executor.py           # Playbook execution
│   │   │
│   │   ├── feedback/                 # Feedback learning
│   │   │   ├── __init__.py
│   │   │   ├── collector.py          # Feedback collection
│   │   │   ├── analyzer.py           # Feedback analysis
│   │   │   └── adapter.py            # Model adaptation
│   │   │
│   │   ├── engine/                   # Learning engine
│   │   │   ├── __init__.py
│   │   │   ├── reinforcement.py      # RL-based learning
│   │   │   └── continuous.py         # Continuous improvement
│   │   │
│   │   └── tests/
│   │
│   ├── memory/                       # MEMORY MODULE (Phase 2)
│   │   ├── __init__.py
│   │   ├── service.py                # MemoryService class
│   │   ├── config.py
│   │   │
│   │   ├── types/                    # Memory types
│   │   │   ├── __init__.py
│   │   │   ├── episodic.py           # Event-based
│   │   │   ├── semantic.py           # Factual knowledge
│   │   │   ├── procedural.py         # How-to
│   │   │   └── working.py            # Short-term
│   │   │
│   │   ├── storage/                  # Storage backends
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py
│   │   │   ├── knowledge_graph.py
│   │   │   └── cache.py
│   │   │
│   │   ├── operations/               # Memory operations
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py
│   │   │   ├── consolidation.py
│   │   │   ├── augmentation.py
│   │   │   └── injection.py          # Prompt injection
│   │   │
│   │   └── tests/
│   │
│   ├── agents/                       # AGENTS MODULE (Phase 3)
│   │   ├── __init__.py
│   │   ├── service.py                # AgentService class
│   │   ├── config.py
│   │   │
│   │   ├── factory/                  # Agent creation
│   │   │   ├── __init__.py
│   │   │   ├── builder.py            # AgentBuilder
│   │   │   ├── registry.py           # Agent registry
│   │   │   └── templates.py          # Agent templates
│   │   │
│   │   ├── execution/                # Agent execution
│   │   │   ├── __init__.py
│   │   │   ├── executor.py           # AgentExecutor
│   │   │   ├── runtime.py            # AgentRuntime
│   │   │   └── context.py            # Execution context
│   │   │
│   │   ├── skills/                   # Skill management
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   ├── registry.py
│   │   │   └── mapper.py
│   │   │
│   │   ├── communication/            # Inter-agent
│   │   │   ├── __init__.py
│   │   │   ├── messaging.py
│   │   │   ├── coordination.py
│   │   │   └── protocols.py
│   │   │
│   │   ├── selection/                # Agent selection
│   │   │   ├── __init__.py
│   │   │   ├── intelligent.py
│   │   │   ├── llm_based.py
│   │   │   └── scoring.py
│   │   │
│   │   └── tests/
│   │
│   ├── tools/                        # TOOLS MODULE (Phase 4)
│   │   ├── __init__.py
│   │   ├── service.py                # ToolService class
│   │   │
│   │   ├── registry/                 # Tool registration
│   │   │   ├── __init__.py
│   │   │   ├── registry.py
│   │   │   ├── discovery.py
│   │   │   └── validation.py
│   │   │
│   │   ├── execution/                # Tool execution
│   │   │   ├── __init__.py
│   │   │   ├── executor.py
│   │   │   ├── formatter.py          # Result formatting
│   │   │   └── sandbox.py
│   │   │
│   │   ├── mcp/                      # MCP integration
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   ├── executor.py
│   │   │   └── discovery.py
│   │   │
│   │   └── tests/
│   │
│   ├── reasoning/                    # REASONING MODULE (Phase 5)
│   │   └── (restructure existing)
│   │
│   └── evaluation/                   # EVALUATION MODULE (Phase 6)
│       └── (restructure existing)
│
├── shared/                           # SHARED INFRASTRUCTURE
│   │
│   ├── llm/                          # LLM providers
│   │   ├── __init__.py
│   │   ├── manager.py                # From services/llm_provider/
│   │   ├── embedding_manager.py
│   │   └── clients/
│   │       ├── openai.py
│   │       ├── anthropic.py
│   │       ├── huggingface.py
│   │       └── ...
│   │
│   ├── math/                         # Mathematical foundations
│   │   ├── __init__.py
│   │   ├── information_theory.py     # From context_engineering/
│   │   ├── vector_operations.py
│   │   ├── statistical_analysis.py
│   │   ├── graph_theory.py
│   │   ├── probability_theory.py
│   │   ├── distance_metrics.py
│   │   └── optimization_algorithms.py
│   │
│   ├── database/                     # Database utilities
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── models.py
│   │   └── migrations/
│   │
│   └── utils/                        # Common utilities
│       ├── __init__.py
│       ├── logging.py
│       ├── config.py
│       └── validation.py
│
├── core/                             # ORCHESTRATION ONLY (minimal)
│   ├── __init__.py
│   ├── orchestrator.py               # 9-stage workflow
│   ├── task_decomposer.py
│   ├── result_aggregator.py
│   └── workflow_executor.py
│
├── api/                              # API LAYER (thin)
│   ├── __init__.py
│   ├── rag.py                        # → modules.rag
│   ├── memory.py                     # → modules.memory
│   ├── agents.py                     # → modules.agents
│   ├── tools.py                      # → modules.tools
│   ├── workflows.py
│   └── system.py
│
├── consumers/                        # PLATFORM CONSUMERS
│   ├── chatbot/                      # Chatbot integration
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── streaming.py
│   │
│   ├── workflows/                    # Workflow integration
│   │   ├── __init__.py
│   │   └── integrator.py
│   │
│   └── external/                     # Third-party API
│       ├── __init__.py
│       └── client.py
│
└── deprecated/                       # TO DELETE AFTER MIGRATION
    └── (old files moved here first)
```

### 4.2 Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CONSUMERS (THIN)                               │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│    │ Chatbot  │    │ Workflows│    │ Third    │    │  CLI     │        │
│    │          │    │          │    │ Party    │    │          │        │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘        │
└─────────┼───────────────┼───────────────┼───────────────┼───────────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        API LAYER (THIN - existing routes)                │
│    /api/documents  /api/chat  /api/agents  /api/workflows  /api/code    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRODUCT MODULES                                  │
│                                                                          │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│   │    RAG     │ │ KNOWLEDGE  │ │  NL2SQL    │ │ CODEGRAPH  │          │
│   │  Document  │ │    Base    │ │  Database  │ │    Code    │          │
│   │  Retrieval │ │   Graph    │ │   Query    │ │  Analysis  │          │
│   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘          │
│         │              │              │              │                  │
│         └──────────────┴──────────────┴──────────────┘                  │
│                                 │                                        │
│                                 ▼                                        │
│                    ┌────────────────────────┐                           │
│                    │    🔥 SEARCH (CORE)    │                           │
│                    │                        │                           │
│                    │  • Vector Store        │                           │
│                    │  • Embeddings          │                           │
│                    │  • Retrieval           │                           │
│                    │  • Optimization        │                           │
│                    │    (Knapsack/MMR)      │                           │
│                    └────────────────────────┘                           │
│                                                                          │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│   │   MEMORY   │ │   AGENTS   │ │   TOOLS    │ │  LEARNING  │          │
│   │            │ │            │ │            │ │            │          │
│   │ • Episodic │ │ • Factory  │ │ • Registry │ │ • Patterns │          │
│   │ • Semantic │ │ • Executor │ │ • Executor │ │ • Playbooks│          │
│   │ • Procedur │ │ • Skills   │ │ • MCP      │ │ • Feedback │          │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
│         │              │              │              │                  │
│         └──────────────┴──────────────┴──────────────┘                  │
│                                 │                                        │
│                    (All can use SEARCH for retrieval)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SHARED INFRASTRUCTURE                            │
│                                                                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│   │     LLM      │  │     MATH     │  │   DATABASE   │  │   UTILS    │ │
│   │              │  │              │  │              │  │            │ │
│   │ • Manager    │  │ • Entropy    │  │ • Connection │  │ • Logging  │ │
│   │ • Embeddings │  │ • MMR        │  │ • Models     │  │ • Config   │ │
│   │ • Clients    │  │ • Knapsack   │  │ • Migrations │  │ • Validate │ │
│   └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Sellable Products

| Module | Standalone Product | Dependencies |
|--------|-------------------|--------------|
| `search/` | `automatos-search` | shared/ only |
| `rag/` | `automatos-rag` | search/ |
| `knowledge/` | `automatos-knowledge` | search/ |
| `nl_to_sql/` | `automatos-nl2sql` | search/ |
| `codegraph/` | `automatos-codegraph` | search/ |
| `memory/` | `automatos-memory` | search/ (optional) |
| `agents/` | `automatos-agents` | memory/, tools/, search/ |
| `tools/` | `automatos-tools` | shared/ only |
| `learning/` | `automatos-learning` | all modules (cross-cutting) |

---

## 5. Module Specifications

### 5.1 RAG Module

#### 5.1.1 Purpose
Provide complete Retrieval-Augmented Generation capabilities as a standalone, sellable product.

#### 5.1.2 Public API

```python
# modules/rag/__init__.py

from .service import RAGService
from .config import RAGConfig
from .chunking import SemanticChunker, ChunkingStrategy
from .retrieval import VectorStore, ContextRetriever
from .optimization import ContextOptimizer

__all__ = [
    'RAGService',
    'RAGConfig',
    'SemanticChunker',
    'ChunkingStrategy',
    'VectorStore',
    'ContextRetriever',
    'ContextOptimizer',
]

# Convenience function
async def retrieve(query: str, **config) -> RAGResult:
    """One-liner RAG retrieval"""
    service = RAGService()
    return await service.retrieve(query, **config)
```

#### 5.1.3 Service Class

```python
# modules/rag/service.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .config import RAGConfig
from .chunking import SemanticChunker
from .retrieval import VectorStore, ContextRetriever
from .optimization import ContextOptimizer
from .ingestion import DocumentProcessor

@dataclass
class RAGResult:
    """Result from RAG retrieval"""
    chunks: List[Dict[str, Any]]
    formatted_context: str
    total_tokens: int
    sources: List[str]
    metrics: Dict[str, float]  # diversity_score, information_gain, etc.

class RAGService:
    """
    Unified RAG Service
    
    Usage:
        # Simple usage
        from modules.rag import RAGService
        rag = RAGService()
        result = await rag.retrieve("How do agents work?")
        
        # With configuration
        from modules.rag import RAGService, RAGConfig
        config = RAGConfig(max_chunks=10, diversity=0.4)
        rag = RAGService(config)
        result = await rag.retrieve("query", context_type="agent")
        
        # Document ingestion
        await rag.ingest(document_path="/path/to/doc.pdf")
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize components
        self._chunker = None
        self._vector_store = None
        self._retriever = None
        self._optimizer = None
        self._processor = None
    
    @property
    def chunker(self) -> SemanticChunker:
        """Lazy initialization of chunker"""
        if self._chunker is None:
            self._chunker = SemanticChunker(
                strategy=self.config.chunking_strategy,
                target_size=self.config.chunk_size,
                min_size=self.config.min_chunk_size,
                max_size=self.config.max_chunk_size
            )
        return self._chunker
    
    @property
    def vector_store(self) -> VectorStore:
        """Lazy initialization of vector store"""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                connection_string=self.config.database_url,
                dimension=self.config.embedding_dimension
            )
        return self._vector_store
    
    @property
    def retriever(self) -> ContextRetriever:
        """Lazy initialization of retriever"""
        if self._retriever is None:
            self._retriever = ContextRetriever(
                vector_store=self.vector_store,
                config=self.config.retrieval_config
            )
        return self._retriever
    
    @property
    def optimizer(self) -> ContextOptimizer:
        """Lazy initialization of optimizer"""
        if self._optimizer is None:
            self._optimizer = ContextOptimizer()
        return self._optimizer
    
    async def retrieve(
        self,
        query: str,
        max_chunks: int = None,
        max_tokens: int = None,
        diversity: float = None,
        context_type: str = "default"
    ) -> RAGResult:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Search query
            max_chunks: Override max chunks
            max_tokens: Override max tokens
            diversity: Override diversity (0=relevance, 1=diversity)
            context_type: Type of context (chatbot, agent, workflow)
        
        Returns:
            RAGResult with chunks, formatted context, and metrics
        """
        # Get candidates
        candidates = await self.retriever.search(
            query=query,
            limit=self.config.candidate_multiplier * (max_chunks or self.config.max_chunks)
        )
        
        # Optimize selection using mathematical algorithms
        optimized = await self.optimizer.optimize_context(
            candidates=candidates,
            max_tokens=max_tokens or self.config.max_tokens,
            objective=self._get_objective(diversity or self.config.diversity)
        )
        
        # Format result
        return RAGResult(
            chunks=optimized.chunks,
            formatted_context=self._format_context(optimized.chunks, query),
            total_tokens=optimized.total_tokens,
            sources=list(set(c['source'] for c in optimized.chunks)),
            metrics={
                'diversity_score': optimized.diversity_score,
                'information_gain': optimized.information_gain,
                'retrieval_time_ms': optimized.retrieval_time_ms
            }
        )
    
    async def ingest(
        self,
        document_path: str = None,
        content: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            document_path: Path to document file
            content: Raw content (alternative to path)
            metadata: Document metadata
        
        Returns:
            Ingestion result with document_id, chunk_count, etc.
        """
        if self._processor is None:
            self._processor = DocumentProcessor(
                chunker=self.chunker,
                vector_store=self.vector_store
            )
        
        return await self._processor.process(
            path=document_path,
            content=content,
            metadata=metadata
        )
    
    def _format_context(self, chunks: List[Dict], query: str) -> str:
        """Format chunks into LLM-ready context"""
        parts = [f"## Retrieved Context for: {query}\n"]
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'unknown')
            relevance = chunk.get('relevance', 0)
            content = chunk.get('content', '')
            
            parts.append(f"\n### Source {i}: {source} (relevance: {relevance:.0%})")
            parts.append(content)
        
        return "\n".join(parts)
    
    def _get_objective(self, diversity: float) -> str:
        """Map diversity to optimization objective"""
        if diversity > 0.6:
            return "maximize_diversity"
        elif diversity < 0.3:
            return "maximize_information"
        return "balanced"
```

#### 5.1.4 RAG Module Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG MODULE FLOW                                  │
└─────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────┐
                           │   User Query    │
                           │ "How do agents  │
                           │     work?"      │
                           └────────┬────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG SERVICE                                    │
│                                                                          │
│   ┌──────────────┐                                                      │
│   │   retrieve() │                                                      │
│   └──────┬───────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    1. QUERY PROCESSING                        │     │
│   │                                                               │     │
│   │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │     │
│   │   │   Expand    │ ──▶  │  Generate   │ ──▶  │  Normalize  │ │     │
│   │   │   Query     │      │  Embedding  │      │  Vector     │ │     │
│   │   └─────────────┘      └─────────────┘      └─────────────┘ │     │
│   │                                                               │     │
│   │   Input: "How do agents work?"                               │     │
│   │   Output: [0.12, -0.45, 0.78, ...] (1024d)                  │     │
│   └──────────────────────────────────────────────────────────────┘     │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    2. CANDIDATE RETRIEVAL                     │     │
│   │                                                               │     │
│   │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │     │
│   │   │   Vector    │ ──▶  │   Hybrid    │ ──▶  │    Get      │ │     │
│   │   │   Search    │      │   (BM25)    │      │ Top 30      │ │     │
│   │   │  (pgvector) │      │             │      │ Candidates  │ │     │
│   │   └─────────────┘      └─────────────┘      └─────────────┘ │     │
│   │                                                               │     │
│   │   SQL: SELECT ... ORDER BY embedding <=> query_vec LIMIT 30  │     │
│   └──────────────────────────────────────────────────────────────┘     │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    3. OPTIMIZATION (Mathematical)             │     │
│   │                                                               │     │
│   │   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │     │
│   │   │  Calculate  │ ──▶  │  Knapsack   │ ──▶  │    MMR      │ │     │
│   │   │  Entropy    │      │  Selection  │      │  Diversity  │ │     │
│   │   └─────────────┘      └─────────────┘      └─────────────┘ │     │
│   │                                                               │     │
│   │   H(X) = -Σ p(x) * log2(p(x))                                │     │
│   │   Knapsack: max value within token budget                    │     │
│   │   MMR: λ*relevance - (1-λ)*max_sim_to_selected              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│          │                                                              │
│          ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    4. FORMAT RESULT                           │     │
│   │                                                               │     │
│   │   ┌─────────────────────────────────────────────────────────┐│     │
│   │   │  RAGResult:                                              ││     │
│   │   │    chunks: [chunk1, chunk2, ...]                        ││     │
│   │   │    formatted_context: "## Retrieved Context..."         ││     │
│   │   │    total_tokens: 1847                                   ││     │
│   │   │    sources: ["AGENT_GUIDE.md", "FACTORY.md"]           ││     │
│   │   │    metrics: {diversity: 0.82, info_gain: 0.76}         ││     │
│   │   └─────────────────────────────────────────────────────────┘│     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Return to      │
                           │  Consumer       │
                           │  (Chatbot/      │
                           │   Workflow/     │
                           │   Agent)        │
                           └─────────────────┘
```

#### 5.1.5 RAG Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RAG INGESTION FLOW                                  │
└─────────────────────────────────────────────────────────────────────────┘

         ┌──────────────┐
         │   Document   │
         │   Upload     │
         │  (PDF/MD/    │
         │   DOCX/TXT)  │
         └──────┬───────┘
                │
                ▼
┌──────────────────────────────────────┐
│      1. FILE HANDLER SELECTION       │
│                                      │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │   PDF   │ │Markdown │ │  Text  │ │
│  │ Handler │ │ Handler │ │Handler │ │
│  └────┬────┘ └────┬────┘ └───┬────┘ │
│       └──────────┼───────────┘      │
└──────────────────┼──────────────────┘
                   │
                   ▼ (raw text)
┌──────────────────────────────────────┐
│      2. SEMANTIC CHUNKING            │
│                                      │
│  Strategy: ADAPTIVE                  │
│                                      │
│  ┌────────────────────────────────┐ │
│  │ Input: "# Agent Factory\n\n    │ │
│  │ The AgentFactory class..."     │ │
│  └────────────────────────────────┘ │
│              │                       │
│              ▼                       │
│  ┌────────────────────────────────┐ │
│  │ Chunks:                        │ │
│  │ [                              │ │
│  │   {content: "# Agent Factory   │ │
│  │    \nThe AgentFactory...",     │ │
│  │    entropy: 4.2,               │ │
│  │    coherence: 0.89},           │ │
│  │   {...},                       │ │
│  │ ]                              │ │
│  └────────────────────────────────┘ │
└──────────────────┼──────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│      3. EMBEDDING GENERATION         │
│                                      │
│  Model: BAAI/bge-large-en-v1.5      │
│  Dimension: 1024                     │
│                                      │
│  For each chunk:                     │
│    embedding = model.encode(chunk)   │
│                                      │
└──────────────────┼──────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│      4. VECTOR STORE INSERT          │
│                                      │
│  INSERT INTO document_chunks (       │
│    document_id, content, embedding,  │
│    metadata, parent_content, headers │
│  ) VALUES (...)                      │
│                                      │
└──────────────────┼──────────────────┘
                   │
                   ▼
         ┌──────────────┐
         │   Document   │
         │   Ready for  │
         │   Retrieval  │
         └──────────────┘
```

#### 5.1.6 Files Migration for RAG Module

| Source File | Destination | Lines | Action |
|-------------|-------------|-------|--------|
| `context_engineering/chunking/semantic_chunker.py` | `modules/rag/chunking/semantic.py` | 477 | MOVE |
| `context_engineering/chunking/__init__.py` | `modules/rag/chunking/__init__.py` | 50 | MOVE |
| `context_engineering/retrieval/vector_store_enhanced.py` | `modules/rag/retrieval/vector_store.py` | 672 | MOVE |
| `context_engineering/retrieval/context_retrieval_engine.py` | `modules/rag/retrieval/context_retriever.py` | 655 | MOVE |
| `context_engineering/context_optimizer.py` | `modules/rag/optimization/context_optimizer.py` | 928 | MOVE |
| `utils/document_manager.py` (extract) | `modules/rag/ingestion/processor.py` | ~300 | EXTRACT |

**Files to DELETE after migration:**
| File | Lines | Reason |
|------|-------|--------|
| `services/semantic_chunker.py` | 400 | Duplicate |
| `context_engineering/chunking.py` | 429 | Duplicate |
| `context_engineering/vector_store.py` | 529 | Merged into enhanced |
| `context_engineering/context_retriever.py` | 585 | Merged into engine |
| `context_engineering/embeddings.py` | 364 | Use shared/llm |
| `core/_vector_store_helper.py` | ~100 | Duplicate |
| `services/rag_service.py` | 370 | Replaced by module |

---

### 5.2 Memory Module

#### 5.2.1 Purpose
Provide complete memory management (episodic, semantic, procedural, working) as a standalone product.

#### 5.2.2 Public API

```python
# modules/memory/__init__.py

from .service import MemoryService
from .types import EpisodicMemory, SemanticMemory, ProceduralMemory

__all__ = ['MemoryService', 'EpisodicMemory', 'SemanticMemory', 'ProceduralMemory']

async def store(content: str, memory_type: str = "episodic", **metadata):
    """Store a memory"""
    service = MemoryService()
    return await service.store(content, memory_type, **metadata)

async def recall(query: str, memory_types: List[str] = None, limit: int = 5):
    """Recall relevant memories"""
    service = MemoryService()
    return await service.recall(query, memory_types, limit)
```

#### 5.2.3 Memory Module Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEMORY MODULE FLOW                                │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         MEMORY SERVICE              │
                    │                                     │
                    │   store() / recall() / consolidate()│
                    └──────────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │    EPISODIC      │    │    SEMANTIC      │    │   PROCEDURAL     │
    │    MEMORY        │    │    MEMORY        │    │    MEMORY        │
    │                  │    │                  │    │                  │
    │  • Events        │    │  • Facts         │    │  • How-to        │
    │  • Conversations │    │  • Concepts      │    │  • Procedures    │
    │  • Experiences   │    │  • Relations     │    │  • Skills        │
    │                  │    │                  │    │                  │
    │  Time-ordered    │    │  Graph-based     │    │  Step-ordered    │
    └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │           STORAGE LAYER             │
                    │                                     │
                    │   ┌──────────┐    ┌──────────┐     │
                    │   │PostgreSQL│    │Knowledge │     │
                    │   │ (pgvector)│   │  Graph   │     │
                    │   └──────────┘    └──────────┘     │
                    └─────────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │         CONSOLIDATION               │
                    │                                     │
                    │   • Merge similar memories          │
                    │   • Update importance scores        │
                    │   • Decay old, irrelevant memories  │
                    └─────────────────────────────────────┘
```

#### 5.2.4 Files Migration for Memory Module

| Source File | Destination | Action |
|-------------|-------------|--------|
| `memory/manager.py` | `modules/memory/service.py` | MERGE |
| `memory/memory_types.py` | `modules/memory/types/__init__.py` | MOVE |
| `memory/augmentation.py` | `modules/memory/operations/augmentation.py` | MOVE |
| `memory/consolidation.py` | `modules/memory/operations/consolidation.py` | MOVE |
| `memory/access_patterns.py` | `modules/memory/operations/retrieval.py` | MERGE |
| `services/memory_knowledge_system.py` | `modules/memory/` (split) | EXTRACT |
| `core/memory_prompt_injector.py` | `modules/memory/operations/injection.py` | MOVE |
| `core/workflow_memory_integrator.py` | `modules/memory/integrations/workflow.py` | MOVE |
| `services/chat/memory_injector.py` | `modules/memory/integrations/chat.py` | MOVE |

---

### 5.3 Agents Module

#### 5.3.1 Purpose
Provide complete agent lifecycle management as a standalone product.

#### 5.3.2 Public API

```python
# modules/agents/__init__.py

from .service import AgentService
from .factory import AgentFactory, AgentBuilder
from .execution import AgentExecutor, AgentRuntime

__all__ = ['AgentService', 'AgentFactory', 'AgentBuilder', 'AgentExecutor']

async def create_agent(name: str, agent_type: str, skills: List[str] = None):
    """Create an agent"""
    service = AgentService()
    return await service.create(name, agent_type, skills)

async def execute_task(agent_id: str, task: str, context: Dict = None):
    """Execute a task with an agent"""
    service = AgentService()
    return await service.execute(agent_id, task, context)
```

#### 5.3.3 Agent Module Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENTS MODULE FLOW                                │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  Create Agent   │
                         │    Request      │
                         └────────┬────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          AGENT FACTORY                                   │
│                                                                          │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐       │
│   │  Load Skills   │───▶│  Build Agent   │───▶│   Register     │       │
│   │  from Registry │    │  Configuration │    │   in DB        │       │
│   └────────────────┘    └────────────────┘    └────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Agent Ready    │
                         │  (AgentRuntime) │
                         └────────┬────────┘
                                  │
                         ┌────────┴────────┐
                         │  Execute Task   │
                         └────────┬────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT EXECUTOR                                   │
│                                                                          │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐       │
│   │ Get Context    │───▶│  Execute with  │───▶│  Process       │       │
│   │ (uses RAG)     │    │  LLM + Tools   │    │  Result        │       │
│   └────────────────┘    └────────────────┘    └────────────────┘       │
│          │                      │                      │                │
│          ▼                      ▼                      ▼                │
│   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐       │
│   │ modules.rag    │    │ modules.tools  │    │ modules.memory │       │
│   │ .retrieve()    │    │ .execute()     │    │ .store()       │       │
│   └────────────────┘    └────────────────┘    └────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 5.3.4 Files Migration for Agents Module

| Source File | Destination | Lines | Action |
|-------------|-------------|-------|--------|
| `services/agent_factory.py` | `modules/agents/factory/` (split) | 2142 | SPLIT |
| `services/skill_loader.py` | `modules/agents/skills/loader.py` | 1212 | MOVE |
| `core/agent_execution_manager.py` | `modules/agents/execution/executor.py` | 1317 | MOVE |
| `core/intelligent_agent_selector.py` | `modules/agents/selection/intelligent.py` | 239 | MOVE |
| `core/llm/llm_agent_selector.py` | `modules/agents/selection/llm_based.py` | ~300 | MOVE |
| `services/inter_agent_communication.py` | `modules/agents/communication/messaging.py` | 1196 | MOVE |
| `multi_agent/coordination_manager.py` | `modules/agents/communication/coordination.py` | 878 | MOVE |

---

### 5.4 Tools Module

#### 5.4.1 Public API

```python
# modules/tools/__init__.py

from .service import ToolService
from .registry import ToolRegistry
from .execution import ToolExecutor

__all__ = ['ToolService', 'ToolRegistry', 'ToolExecutor']

async def execute_tool(tool_name: str, parameters: Dict = None):
    """Execute a tool by name"""
    service = ToolService()
    return await service.execute(tool_name, parameters)
```

#### 5.4.2 Files Migration for Tools Module

| Source File | Destination | Action |
|-------------|-------------|--------|
| `services/tool_registry.py` | `modules/tools/registry/registry.py` | MOVE |
| `services/unified_tool_executor.py` | `modules/tools/execution/executor.py` | MOVE |
| `services/tool_result_formatter.py` | `modules/tools/execution/formatter.py` | MOVE |
| `services/mcp_tool_executor.py` | `modules/tools/mcp/executor.py` | MOVE |
| `services/mcp_auto_activation.py` | `modules/tools/mcp/discovery.py` | MOVE |
| `services/tool_capability_mapper.py` | `modules/tools/registry/mapper.py` | MOVE |

---

## 6. Shared Infrastructure

### 6.1 LLM Providers (Keep from services/llm_provider)

```
shared/llm/
├── __init__.py
├── manager.py                    # From services/llm_provider/manager.py
├── embedding_manager.py          # From services/llm_provider/embedding_manager.py
└── clients/
    ├── __init__.py
    ├── base.py
    ├── openai.py
    ├── anthropic.py
    ├── huggingface.py
    └── ...
```

### 6.2 Mathematical Foundations (Keep from context_engineering)

```
shared/math/
├── __init__.py
├── information_theory.py         # Entropy, mutual information
├── vector_operations.py          # Cosine similarity, etc.
├── statistical_analysis.py       # Trends, confidence intervals
├── graph_theory.py               # Dependency graphs
├── probability_theory.py         # Bayesian, confidence
├── distance_metrics.py           # Distance calculations
└── optimization_algorithms.py    # Knapsack, etc.
```

### 6.3 Database (Keep from database/)

```
shared/database/
├── __init__.py
├── connection.py                 # From database/database.py
├── models.py                     # SQLAlchemy models
└── migrations/
```

---

## 7. Migration Plan

### 7.1 Phase Overview

| Phase | Module | Duration | Dependencies | Sellable As |
|-------|--------|----------|--------------|-------------|
| 0 | Preparation | 2 days | None | - |
| 1a | **Search (Core)** | 3 days | Shared infra | `automatos-search` |
| 1b | RAG Module | 3 days | Phase 1a | `automatos-rag` |
| 1c | Knowledge Module | 2 days | Phase 1a | `automatos-knowledge` |
| 1d | NL-to-SQL Module | 2 days | Phase 1a | `automatos-nl2sql` |
| 1e | CodeGraph Module | 2 days | Phase 1a | `automatos-codegraph` |
| 2 | Memory Module | 1 week | Phase 1a | `automatos-memory` |
| 3 | Agents Module | 1 week | Phase 1a, 2 | `automatos-agents` |
| 4 | Tools Module | 3 days | Shared infra | `automatos-tools` |
| 5 | Reasoning Module | 3 days | Phase 3 | - |
| 5.5 | **Learning Module** | 3 days | All modules | `automatos-learning` |
| 6 | Evaluation Module | 2 days | Phase 3 | - |
| 7 | Cleanup | 3 days | All | - |

**Total Estimated Duration: 5-6 weeks**

### 7.2 Phase 0: Preparation (2 days)

```
Tasks:
□ Create modules/ directory structure
□ Create shared/ directory structure
□ Move mathematical_foundations to shared/math/
□ Move llm_provider to shared/llm/
□ Update imports in affected files
□ Run tests to verify no breakage
```

### 7.3 Phase 1: RAG Module (1 week)

```
Day 1-2: Chunking
□ Create modules/rag/chunking/
□ Move semantic_chunker.py from context_engineering/chunking/
□ Delete services/semantic_chunker.py
□ Delete context_engineering/chunking.py
□ Update all imports
□ Test chunking

Day 3-4: Retrieval
□ Create modules/rag/retrieval/
□ Move vector_store_enhanced.py
□ Move context_retrieval_engine.py
□ Delete context_engineering/vector_store.py
□ Delete context_engineering/context_retriever.py
□ Delete core/_vector_store_helper.py
□ Update all imports
□ Test retrieval

Day 5: Optimization
□ Create modules/rag/optimization/
□ Move context_optimizer.py
□ Split into knapsack.py, mmr.py, entropy.py
□ Update all imports
□ Test optimization

Day 6: Ingestion
□ Create modules/rag/ingestion/
□ Extract document processing from document_manager.py
□ Create file handlers (pdf.py, markdown.py, etc.)
□ Update all imports
□ Test ingestion

Day 7: Integration
□ Create modules/rag/service.py
□ Create modules/rag/__init__.py
□ Update api/documents.py to use module
□ Update all consumers (chatbot, workflows)
□ Delete services/rag_service.py
□ Run full integration tests
```

### 7.4 Phase 2: Memory Module (1 week)

```
Day 1-2: Memory Types
□ Create modules/memory/types/
□ Move memory_types.py
□ Create episodic.py, semantic.py, procedural.py
□ Test types

Day 3-4: Storage
□ Create modules/memory/storage/
□ Extract from memory_knowledge_system.py
□ Create postgres.py, knowledge_graph.py
□ Test storage

Day 5-6: Operations
□ Create modules/memory/operations/
□ Move augmentation.py, consolidation.py
□ Move memory_prompt_injector.py → injection.py
□ Test operations

Day 7: Integration
□ Create modules/memory/service.py
□ Delete services/memory_knowledge_system.py
□ Delete scattered memory files
□ Update all consumers
□ Run full integration tests
```

### 7.5 Phase 3-6: (Similar detailed breakdown)

---

## 8. Files to Delete

### 8.1 After Phase 1 (RAG)

| File | Lines | Reason |
|------|-------|--------|
| `services/semantic_chunker.py` | 400 | Duplicate |
| `context_engineering/chunking.py` | 429 | Duplicate |
| `context_engineering/vector_store.py` | 529 | Merged |
| `context_engineering/context_retriever.py` | 585 | Merged |
| `context_engineering/embeddings.py` | 364 | Use shared |
| `core/_vector_store_helper.py` | 100 | Duplicate |
| `services/rag_service.py` | 370 | Replaced |

**Total deleted: 2,777 lines**

### 8.2 After Phase 2 (Memory)

| File | Lines | Reason |
|------|-------|--------|
| `core/memory_prompt_injector.py` | 200 | Moved |
| `core/workflow_memory_integrator.py` | 300 | Moved |
| `services/chat/memory_injector.py` | 150 | Moved |

**Total deleted: 650 lines**

### 8.3 After All Phases

**Estimated total lines deleted: 5,000+**
**Estimated duplicate code eliminated: 15,000+ lines**

---

## 9. Testing Strategy

### 9.1 Unit Tests per Module

```python
# modules/rag/tests/test_chunking.py
async def test_semantic_chunker_splits_by_headers():
    chunker = SemanticChunker(strategy=ChunkingStrategy.HIERARCHICAL)
    chunks = chunker.chunk("# Header\nContent...")
    assert len(chunks) > 0
    assert chunks[0].metadata.entropy > 0

# modules/rag/tests/test_retrieval.py
async def test_vector_store_search():
    store = VectorStore()
    results = await store.search("test query", limit=5)
    assert len(results) <= 5

# modules/rag/tests/test_optimization.py
async def test_knapsack_respects_budget():
    optimizer = ContextOptimizer()
    result = await optimizer.optimize(chunks, max_tokens=1000)
    assert result.total_tokens <= 1000
```

### 9.2 Integration Tests

```python
# tests/integration/test_rag_module.py
async def test_rag_end_to_end():
    from modules.rag import RAGService
    
    rag = RAGService()
    
    # Ingest document
    await rag.ingest(content="# Test\nTest content...")
    
    # Retrieve
    result = await rag.retrieve("test")
    
    assert result.chunks is not None
    assert len(result.sources) > 0
```

### 9.3 Consumer Tests

```python
# tests/integration/test_chatbot_uses_rag.py
async def test_chatbot_retrieves_context():
    from modules.rag import RAGService
    from consumers.chatbot import ChatService
    
    chat = ChatService()
    response = await chat.process("How do agents work?")
    
    # Verify RAG was called
    assert response.context_used is True

# tests/integration/test_workflow_uses_rag.py
async def test_workflow_has_context():
    from core.orchestrator import WorkflowOrchestrator
    
    result = await WorkflowOrchestrator().execute("Analyze agent code")
    assert result.rag_context is not None
```

### 9.4 Performance Tests

```python
# tests/performance/test_rag_performance.py
import time

async def test_retrieval_latency():
    from modules.rag import RAGService
    
    rag = RAGService()
    
    start = time.time()
    result = await rag.retrieve("complex technical query")
    duration = time.time() - start
    
    assert duration < 2.0  # Max 2 seconds
    assert result.metrics['retrieval_time_ms'] < 2000
```

---

## 10. Rollback Plan

### 10.1 Backup Strategy

Before each phase:
1. Git tag current state: `git tag pre-phase-{N}`
2. Document working state
3. Ensure all tests pass

### 10.2 Rollback Procedures

**If Phase 1 fails:**
```bash
# Restore RAG from backup
git checkout pre-phase-1 -- services/rag_service.py
git checkout pre-phase-1 -- services/semantic_chunker.py
git checkout pre-phase-1 -- context_engineering/

# Remove new module
rm -rf orchestrator/modules/rag/

# Restart services
docker-compose restart backend
```

**If specific component fails:**
```python
# Fallback in code
try:
    from modules.rag import RAGService
except ImportError:
    from services.rag_service import RAGService  # Legacy fallback
```

### 10.3 Feature Flags

```python
# config.py
FEATURE_FLAGS = {
    "use_rag_module": False,      # Set True when ready
    "use_memory_module": False,
    "use_agents_module": False,
}

# Usage
from config import FEATURE_FLAGS

if FEATURE_FLAGS["use_rag_module"]:
    from modules.rag import RAGService
else:
    from services.rag_service import RAGService
```

---

## 11. Task Checklist

### Phase 0: Preparation

- [ ] **P0-001**: Create `orchestrator/modules/` directory
- [ ] **P0-002**: Create `orchestrator/modules/rag/` structure
- [ ] **P0-003**: Create `orchestrator/modules/memory/` structure
- [ ] **P0-004**: Create `orchestrator/modules/agents/` structure
- [ ] **P0-005**: Create `orchestrator/modules/tools/` structure
- [ ] **P0-006**: Create `orchestrator/shared/` directory
- [ ] **P0-007**: Move `context_engineering/mathematical_foundations/` → `shared/math/`
- [ ] **P0-008**: Move `services/llm_provider/` → `shared/llm/`
- [ ] **P0-009**: Update imports for moved shared components
- [ ] **P0-010**: Verify all tests pass
- [ ] **P0-011**: Create git tag `pre-phase-1`

### Phase 1a: Search (Core) Module

- [ ] **P1a-001**: Create `modules/search/__init__.py`
- [ ] **P1a-002**: Create `modules/search/service.py` (SearchService)
- [ ] **P1a-003**: Create `modules/search/config.py`

#### Vector Store
- [ ] **P1a-004**: Create `modules/search/vector_store/__init__.py`
- [ ] **P1a-005**: Move `context_engineering/retrieval/vector_store_enhanced.py` → `modules/search/vector_store/store.py`
- [ ] **P1a-006**: Create `modules/search/vector_store/indexing.py`
- [ ] **P1a-007**: Create `modules/search/vector_store/queries.py`

#### Embeddings
- [ ] **P1a-008**: Create `modules/search/embeddings/__init__.py`
- [ ] **P1a-009**: Move embedding logic from `services/llm_provider/embedding_manager.py` → `modules/search/embeddings/manager.py`
- [ ] **P1a-010**: Create `modules/search/embeddings/providers.py`

#### Retrieval
- [ ] **P1a-011**: Create `modules/search/retrieval/__init__.py`
- [ ] **P1a-012**: Create `modules/search/retrieval/vector_search.py`
- [ ] **P1a-013**: Create `modules/search/retrieval/hybrid_search.py` (BM25)
- [ ] **P1a-014**: Create `modules/search/retrieval/reranking.py` (FlashRank)

#### Optimization
- [ ] **P1a-015**: Create `modules/search/optimization/__init__.py`
- [ ] **P1a-016**: Move `context_engineering/context_optimizer.py` → `modules/search/optimization/context_optimizer.py`
- [ ] **P1a-017**: Extract `modules/search/optimization/knapsack.py`
- [ ] **P1a-018**: Extract `modules/search/optimization/mmr.py`
- [ ] **P1a-019**: Extract `modules/search/optimization/entropy.py`

#### Testing
- [ ] **P1a-020**: Write unit tests for vector_store
- [ ] **P1a-021**: Write unit tests for retrieval
- [ ] **P1a-022**: Write unit tests for optimization
- [ ] **P1a-023**: Integration test: embed → store → search → optimize
- [ ] **P1a-024**: Create git tag `post-phase-1a`

---

### Phase 1b: RAG Module

#### Chunking
- [ ] **P1b-001**: Create `modules/rag/chunking/__init__.py`
- [ ] **P1b-002**: Copy `context_engineering/chunking/semantic_chunker.py` → `modules/rag/chunking/semantic.py`
- [ ] **P1b-003**: Update imports to use `modules.search`
- [ ] **P1b-004**: Create `modules/rag/chunking/strategies.py`
- [ ] **P1b-005**: Create `modules/rag/chunking/metadata.py`
- [ ] **P1b-006**: Delete `services/semantic_chunker.py` (400 lines)
- [ ] **P1b-007**: Delete `context_engineering/chunking.py` (429 lines)

#### Ingestion
- [ ] **P1b-008**: Create `modules/rag/ingestion/__init__.py`
- [ ] **P1b-009**: Create `modules/rag/ingestion/processor.py`
- [ ] **P1b-010**: Create `modules/rag/ingestion/handlers/markdown.py`
- [ ] **P1b-011**: Create `modules/rag/ingestion/handlers/pdf.py`
- [ ] **P1b-012**: Create `modules/rag/ingestion/handlers/text.py`
- [ ] **P1b-013**: Create `modules/rag/ingestion/pipeline.py`

#### Integration
- [ ] **P1b-014**: Create `modules/rag/service.py` (RAGService - wraps search/)
- [ ] **P1b-015**: Create `modules/rag/config.py`
- [ ] **P1b-016**: Create `modules/rag/__init__.py`
- [ ] **P1b-017**: Update `api/documents.py` to use `modules.rag`
- [ ] **P1b-018**: Delete `services/rag_service.py` (370 lines)
- [ ] **P1b-019**: Write tests
- [ ] **P1b-020**: Create git tag `post-phase-1b`

---

### Phase 1c: Knowledge Module

- [ ] **P1c-001**: Create `modules/knowledge/__init__.py`
- [ ] **P1c-002**: Create `modules/knowledge/service.py` (KnowledgeService)
- [ ] **P1c-003**: Create `modules/knowledge/config.py`

#### Graph
- [ ] **P1c-004**: Create `modules/knowledge/graph/__init__.py`
- [ ] **P1c-005**: Create `modules/knowledge/graph/builder.py`
- [ ] **P1c-006**: Create `modules/knowledge/graph/traversal.py`
- [ ] **P1c-007**: Move entity extraction from `services/entity_extractor.py` → `modules/knowledge/graph/entities.py`

#### Storage
- [ ] **P1c-008**: Create `modules/knowledge/storage/__init__.py`
- [ ] **P1c-009**: Create `modules/knowledge/storage/postgres.py`
- [ ] **P1c-010**: Extract from `services/database_knowledge_service.py`

#### Integration
- [ ] **P1c-011**: Update consumers to use `modules.knowledge`
- [ ] **P1c-012**: Delete `services/entity_extractor.py`
- [ ] **P1c-013**: Delete `services/database_knowledge_service.py`
- [ ] **P1c-014**: Write tests
- [ ] **P1c-015**: Create git tag `post-phase-1c`

---

### Phase 1d: NL-to-SQL Module

- [ ] **P1d-001**: Create `modules/nl_to_sql/__init__.py`
- [ ] **P1d-002**: Create `modules/nl_to_sql/service.py` (NLToSQLService)
- [ ] **P1d-003**: Create `modules/nl_to_sql/config.py`

#### Schema
- [ ] **P1d-004**: Create `modules/nl_to_sql/schema/__init__.py`
- [ ] **P1d-005**: Move `services/database_introspection.py` → `modules/nl_to_sql/schema/introspection.py`
- [ ] **P1d-006**: Move `services/schema_provider.py` → `modules/nl_to_sql/schema/provider.py`
- [ ] **P1d-007**: Create `modules/nl_to_sql/schema/embeddings.py`

#### Query
- [ ] **P1d-008**: Create `modules/nl_to_sql/query/__init__.py`
- [ ] **P1d-009**: Move `services/nl_to_sql_service.py` → `modules/nl_to_sql/query/builder.py`
- [ ] **P1d-010**: Move `services/sql_validator.py` → `modules/nl_to_sql/query/validator.py`
- [ ] **P1d-011**: Create `modules/nl_to_sql/query/executor.py`

#### Integration
- [ ] **P1d-012**: Update consumers to use `modules.nl_to_sql`
- [ ] **P1d-013**: Delete old service files
- [ ] **P1d-014**: Write tests
- [ ] **P1d-015**: Create git tag `post-phase-1d`

---

### Phase 1e: CodeGraph Module

- [ ] **P1e-001**: Create `modules/codegraph/__init__.py`
- [ ] **P1e-002**: Create `modules/codegraph/service.py` (CodeGraphService)
- [ ] **P1e-003**: Create `modules/codegraph/config.py`

#### Analysis
- [ ] **P1e-004**: Create `modules/codegraph/analysis/__init__.py`
- [ ] **P1e-005**: Extract from `services/codegraph_service.py` → `modules/codegraph/analysis/parser.py`
- [ ] **P1e-006**: Create `modules/codegraph/analysis/dependencies.py`
- [ ] **P1e-007**: Create `modules/codegraph/analysis/metrics.py`

#### Graph
- [ ] **P1e-008**: Create `modules/codegraph/graph/__init__.py`
- [ ] **P1e-009**: Extract from `services/codegraph_service.py` → `modules/codegraph/graph/builder.py`
- [ ] **P1e-010**: Create `modules/codegraph/graph/queries.py`

#### Search
- [ ] **P1e-011**: Create `modules/codegraph/search/__init__.py`
- [ ] **P1e-012**: Create `modules/codegraph/search/semantic.py` (uses modules.search)
- [ ] **P1e-013**: Create `modules/codegraph/search/structural.py`

#### Integration
- [ ] **P1e-014**: Update consumers to use `modules.codegraph`
- [ ] **P1e-015**: Delete `services/codegraph_service.py` (1383 lines)
- [ ] **P1e-016**: Write tests
- [ ] **P1e-017**: Create git tag `post-phase-1e`

### Phase 2: Memory Module

#### Day 1-2: Memory Types
- [ ] **P2-001**: Create `modules/memory/types/__init__.py`
- [ ] **P2-002**: Copy `memory/memory_types.py` content → `modules/memory/types/`
- [ ] **P2-003**: Create `modules/memory/types/episodic.py` (EpisodicMemory)
- [ ] **P2-004**: Create `modules/memory/types/semantic.py` (SemanticMemory)
- [ ] **P2-005**: Create `modules/memory/types/procedural.py` (ProceduralMemory)
- [ ] **P2-006**: Create `modules/memory/types/working.py` (WorkingMemory)
- [ ] **P2-007**: Write unit tests for memory types

#### Day 3-4: Storage
- [ ] **P2-008**: Create `modules/memory/storage/__init__.py`
- [ ] **P2-009**: Extract storage logic from `services/memory_knowledge_system.py`
- [ ] **P2-010**: Create `modules/memory/storage/postgres.py`
- [ ] **P2-011**: Create `modules/memory/storage/knowledge_graph.py`
- [ ] **P2-012**: Create `modules/memory/storage/cache.py`
- [ ] **P2-013**: Write unit tests for storage

#### Day 5-6: Operations
- [ ] **P2-014**: Create `modules/memory/operations/__init__.py`
- [ ] **P2-015**: Move `memory/augmentation.py` → `modules/memory/operations/`
- [ ] **P2-016**: Move `memory/consolidation.py` → `modules/memory/operations/`
- [ ] **P2-017**: Move `core/memory_prompt_injector.py` → `modules/memory/operations/injection.py`
- [ ] **P2-018**: Create `modules/memory/operations/retrieval.py`
- [ ] **P2-019**: Write unit tests for operations

#### Day 7: Integration
- [ ] **P2-020**: Create `modules/memory/service.py` (MemoryService)
- [ ] **P2-021**: Create `modules/memory/config.py`
- [ ] **P2-022**: Create `modules/memory/__init__.py`
- [ ] **P2-023**: Update chat service to use `modules.memory`
- [ ] **P2-024**: Delete `services/memory_knowledge_system.py` (1362 lines)
- [ ] **P2-025**: Delete `core/memory_prompt_injector.py`
- [ ] **P2-026**: Delete `core/workflow_memory_integrator.py`
- [ ] **P2-027**: Delete `services/chat/memory_injector.py`
- [ ] **P2-028**: Write integration tests
- [ ] **P2-029**: Full memory test: store → recall → verify
- [ ] **P2-030**: Create git tag `post-phase-2`

### Phase 3: Agents Module

- [ ] **P3-001**: Create `modules/agents/factory/`
- [ ] **P3-002**: Split `services/agent_factory.py` (2142 lines) into:
  - [ ] **P3-002a**: `modules/agents/factory/builder.py`
  - [ ] **P3-002b**: `modules/agents/factory/registry.py`
  - [ ] **P3-002c**: `modules/agents/factory/templates.py`
- [ ] **P3-003**: Create `modules/agents/execution/`
- [ ] **P3-004**: Move `core/agent_execution_manager.py` → `modules/agents/execution/executor.py`
- [ ] **P3-005**: Create `modules/agents/skills/`
- [ ] **P3-006**: Move `services/skill_loader.py` → `modules/agents/skills/loader.py`
- [ ] **P3-007**: Create `modules/agents/selection/`
- [ ] **P3-008**: Move `core/intelligent_agent_selector.py` → `modules/agents/selection/`
- [ ] **P3-009**: Create `modules/agents/communication/`
- [ ] **P3-010**: Move `services/inter_agent_communication.py` → `modules/agents/communication/`
- [ ] **P3-011**: Move `multi_agent/coordination_manager.py` → `modules/agents/communication/`
- [ ] **P3-012**: Create `modules/agents/service.py`
- [ ] **P3-013**: Create `modules/agents/__init__.py`
- [ ] **P3-014**: Delete `services/agent_factory.py`
- [ ] **P3-015**: Delete `services/skill_loader.py`
- [ ] **P3-016**: Write tests
- [ ] **P3-017**: Create git tag `post-phase-3`

### Phase 4: Tools Module

- [ ] **P4-001**: Create `modules/tools/registry/`
- [ ] **P4-002**: Move `services/tool_registry.py` → `modules/tools/registry/`
- [ ] **P4-003**: Create `modules/tools/execution/`
- [ ] **P4-004**: Move `services/unified_tool_executor.py` → `modules/tools/execution/executor.py`
- [ ] **P4-005**: Move `services/tool_result_formatter.py` → `modules/tools/execution/formatter.py`
- [ ] **P4-006**: Create `modules/tools/mcp/`
- [ ] **P4-007**: Move `services/mcp_tool_executor.py` → `modules/tools/mcp/`
- [ ] **P4-008**: Move `services/mcp_auto_activation.py` → `modules/tools/mcp/`
- [ ] **P4-009**: Create `modules/tools/service.py`
- [ ] **P4-010**: Create `modules/tools/__init__.py`
- [ ] **P4-011**: Delete old tool files from services/
- [ ] **P4-012**: Write tests
- [ ] **P4-013**: Create git tag `post-phase-4`

### Phase 5: Reasoning Module

- [ ] **P5-001**: Create `modules/reasoning/__init__.py`
- [ ] **P5-002**: Create `modules/reasoning/service.py`
- [ ] **P5-003**: Move `reasoning/reasoning_engine.py` → `modules/reasoning/engine.py`
- [ ] **P5-004**: Move `reasoning/tool_selection.py` → `modules/reasoning/tool_selection.py`
- [ ] **P5-005**: Move `reasoning/output_processing.py` → `modules/reasoning/output.py`
- [ ] **P5-006**: Move `reasoning/execution_orchestrator.py` → `modules/reasoning/orchestrator.py`
- [ ] **P5-007**: Update imports
- [ ] **P5-008**: Write tests
- [ ] **P5-009**: Create git tag `post-phase-5`

---

### Phase 5.5: Learning Module

- [ ] **P5.5-001**: Create `modules/learning/__init__.py`
- [ ] **P5.5-002**: Create `modules/learning/service.py` (LearningService)
- [ ] **P5.5-003**: Create `modules/learning/config.py`

#### Patterns
- [ ] **P5.5-004**: Create `modules/learning/patterns/__init__.py`
- [ ] **P5.5-005**: Create `modules/learning/patterns/detector.py`
- [ ] **P5.5-006**: Create `modules/learning/patterns/extractor.py`
- [ ] **P5.5-007**: Move `memory/access_patterns.py` → `modules/learning/patterns/access.py`

#### Playbooks
- [ ] **P5.5-008**: Create `modules/learning/playbooks/__init__.py`
- [ ] **P5.5-009**: Move `services/playbook_miner.py` → `modules/learning/playbooks/miner.py`
- [ ] **P5.5-010**: Create `modules/learning/playbooks/templates.py`
- [ ] **P5.5-011**: Create `modules/learning/playbooks/executor.py`

#### Feedback
- [ ] **P5.5-012**: Create `modules/learning/feedback/__init__.py`
- [ ] **P5.5-013**: Create `modules/learning/feedback/collector.py`
- [ ] **P5.5-014**: Create `modules/learning/feedback/analyzer.py`
- [ ] **P5.5-015**: Create `modules/learning/feedback/adapter.py`

#### Engine
- [ ] **P5.5-016**: Create `modules/learning/engine/__init__.py`
- [ ] **P5.5-017**: Move `context_engineering/learning_engine.py` → `modules/learning/engine/core.py`
- [ ] **P5.5-018**: Create `modules/learning/engine/reinforcement.py`
- [ ] **P5.5-019**: Create `modules/learning/engine/continuous.py`

#### Integration
- [ ] **P5.5-020**: Wire learning into RAG (feedback loop)
- [ ] **P5.5-021**: Wire learning into Agents (performance tracking)
- [ ] **P5.5-022**: Wire learning into Memory (pattern detection)
- [ ] **P5.5-023**: Delete old learning files
- [ ] **P5.5-024**: Write tests
- [ ] **P5.5-025**: Create git tag `post-phase-5.5`

---

### Phase 6: Evaluation Module

- [ ] **P6-001**: Create `modules/evaluation/__init__.py`
- [ ] **P6-002**: Create `modules/evaluation/service.py`
- [ ] **P6-003**: Move `evaluation/evaluation_engine.py` → `modules/evaluation/engine.py`
- [ ] **P6-004**: Move `evaluation/evaluation_service.py` → `modules/evaluation/service.py`
- [ ] **P6-005**: Move `evaluation/benchmark_design.py` → `modules/evaluation/benchmarks.py`
- [ ] **P6-006**: Move `evaluation/integration_evaluator.py` → `modules/evaluation/integration.py`
- [ ] **P6-007**: Move `evaluation/component_assessment.py` → `modules/evaluation/assessment.py`
- [ ] **P6-008**: Update imports
- [ ] **P6-009**: Write tests
- [ ] **P6-010**: Create git tag `post-phase-6`

### Phase 7: Consumers & API Cleanup

#### Consumers (Thin Wrappers)
- [ ] **P7-001**: Create `consumers/` directory
- [ ] **P7-002**: Create `consumers/chatbot/service.py` (thin - calls modules)
- [ ] **P7-003**: Create `consumers/chatbot/streaming.py`
- [ ] **P7-004**: Create `consumers/workflows/integrator.py`
- [ ] **P7-005**: Create `consumers/external/client.py` (third-party API)
- [ ] **P7-006**: Move logic from `services/chat/` to appropriate modules
- [ ] **P7-007**: Delete fat service files, keep thin consumers

#### API Cleanup
- [ ] **P7-008**: Audit `api/` folder - remove embedded business logic
- [ ] **P7-009**: Update `api/documents.py` → thin wrapper to `modules.rag`
- [ ] **P7-010**: Update `api/chat.py` → thin wrapper to `consumers.chatbot`
- [ ] **P7-011**: Update `api/agents.py` → thin wrapper to `modules.agents`
- [ ] **P7-012**: Update `api/workflows.py` → thin wrapper to `modules.tools`

#### Final Cleanup
- [ ] **P7-013**: Remove empty `context_engineering/` folder
- [ ] **P7-014**: Remove empty `services/` files (keep llm_provider in shared/)
- [ ] **P7-015**: Remove deprecated `core/` files
- [ ] **P7-016**: Remove `deprecated/` folder if created
- [ ] **P7-017**: Audit and remove unused imports across codebase
- [ ] **P7-018**: Run full test suite
- [ ] **P7-019**: Performance benchmarking (compare before/after)
- [ ] **P7-020**: Create final git tag `v2.0-modular`

---

## 12. Summary Metrics

### Before Refactoring

| Metric | Value |
|--------|-------|
| Duplicate code | ~15,000 lines |
| Files with RAG/Search logic | 12+ scattered |
| Files with Memory logic | 10 scattered |
| Files with Agent logic | 8+ scattered |
| Files in services/ | 50+ (dumping ground) |
| Module import depth | 3-5 imports per feature |
| Sellable products | 0 |

### After Refactoring

| Metric | Target |
|--------|--------|
| Duplicate code | 0 lines |
| Search logic | 1 module (`modules/search`) |
| RAG logic | 1 module (`modules/rag`) |
| Knowledge logic | 1 module (`modules/knowledge`) |
| NL-to-SQL logic | 1 module (`modules/nl_to_sql`) |
| CodeGraph logic | 1 module (`modules/codegraph`) |
| Memory logic | 1 module (`modules/memory`) |
| Agent logic | 1 module (`modules/agents`) |
| Tool logic | 1 module (`modules/tools`) |
| Learning logic | 1 module (`modules/learning`) |
| Files in services/ | 0 (moved to modules/) |
| Module import depth | 1 import per feature |
| Sellable products | 9 standalone modules |

### Module Summary

| Module | Purpose | Sellable As |
|--------|---------|-------------|
| `search/` | Core vector search engine | `automatos-search` |
| `rag/` | Document retrieval | `automatos-rag` |
| `knowledge/` | Knowledge graph | `automatos-knowledge` |
| `nl_to_sql/` | Natural language to SQL | `automatos-nl2sql` |
| `codegraph/` | Code analysis | `automatos-codegraph` |
| `memory/` | Multi-type memory system | `automatos-memory` |
| `agents/` | Agent lifecycle | `automatos-agents` |
| `tools/` | Tool execution | `automatos-tools` |
| `learning/` | Self-improvement | `automatos-learning` |

### Code Health Goals

```
✅ Single Source of Truth per feature
✅ One module = One concern
✅ Clear public APIs (1-line imports)
✅ Testable boundaries
✅ Sellable as standalone products
✅ Third-party consumable
✅ Thin consumers (chatbot, workflows)
✅ Thin API layer (routes only)
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2024-12-03 | AI | Initial comprehensive PRD |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Owner | | | |
| Tech Lead | | | |
| DevOps | | | |

---

**END OF PRD-30**
