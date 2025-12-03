# PRD 08: Universal RAG & Semantic Search System
**Version 2.0 - Supercharged with Kimia Context Engineering + LangChain**

## 1. Overview

### Purpose
Transform the RAG system from basic vector-only search into a **production-grade, universal retrieval system** using:
- **David Kimia's Context Engineering** principles (hierarchical chunking, cognitive formatting)
- **LangChain's advanced retrievers** (hybrid search, reranking)
- **IBM Zurich Cognitive Tools** research (structured context scaffolding)

### What Was Wrong (v1.0)
| Problem | Impact | Root Cause |
|---------|--------|------------|
| Empty header chunks ranked high | Useless results like `"### 3.1 Agent Flow"` | Fixed-size chunking broke semantic units |
| No keyword matching | Missed exact term matches | Vector-only search |
| No quality filtering | Garbage results returned | No reranking stage |
| Context without structure | Hard for LLM to reason | Chunks dumped without formatting |
| Single retrieval method | Limited coverage | No hybrid approach |

### What We're Building (v2.0)
```
┌─────────────────────────────────────────────────────────────────┐
│            UNIVERSAL RAG SERVICE v2.0                           │
│    (Chatbot, Agents, Search, Context Engineering, Workflows)    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │   INGEST    │    │   RETRIEVE   │    │    FORMAT       │   │
│  ├─────────────┤    ├──────────────┤    ├─────────────────┤   │
│  │ 1. Markdown │    │ 1. Query     │    │ 1. Cognitive    │   │
│  │    Header   │    │    Transform │    │    Structure    │   │
│  │    Split    │    │    (expand)  │    │                 │   │
│  │             │    │              │    │ 2. Source       │   │
│  │ 2. Parent/  │    │ 2. Hybrid    │    │    Citations    │   │
│  │    Child    │    │    Search    │    │                 │   │
│  │    Storage  │    │    (V+BM25)  │    │ 3. Token        │   │
│  │             │    │              │    │    Budget       │   │
│  │ 3. Quality  │    │ 3. Rerank    │    │                 │   │
│  │    Filter   │    │    (Cross-   │    │                 │   │
│  │             │    │     Encoder) │    │                 │   │
│  └─────────────┘    └──────────────┘    └─────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Research Foundation

### 2.1 David Kimia's Context Engineering
From [Context Engineering Repository](https://github.com/davidkimai/Context-Engineering):

> "Context is not just the single prompt users send to an LLM. Context is the complete information payload provided at inference time."

**Key Principles Applied:**
1. **Hierarchical Chunking** - Semantic boundaries with parent/child relationships
2. **Hybrid Search** - Vector similarity + keyword matching (BM25)
3. **Cognitive Tools** - Structured prompts that scaffold reasoning
4. **Query Transformation** - Expand and reformulate queries

### 2.2 IBM Zurich Cognitive Tools Research
From [Eliciting Reasoning in Language Models](https://arxiv.org/pdf/2506.12115):

> "Cognitive tools break down the problem by identifying main concepts, extracting relevant information, and highlighting meaningful properties."

**Applied to RAG:**
- Format retrieved context with structure (headers, sources, relevance)
- Don't just dump chunks - scaffold reasoning
- Use structured formats (Markdown, JSON) for better LLM parsing

### 2.3 LangChain Components
| Component | Purpose |
|-----------|---------|
| `MarkdownHeaderTextSplitter` | Keep headers WITH their content |
| `ParentDocumentRetriever` | Store small chunks, return parent context |
| `EnsembleRetriever` | Combine vector + BM25 search |
| `BM25Retriever` | Keyword/exact match search |
| `FlashrankRerank` | Cross-encoder reranking (FREE) |
| `ContextualCompressionRetriever` | Quality filtering after retrieval |

---

## 3. Technical Architecture

### 3.1 Chunking Pipeline (SmartChunker)

**Problem Solved:** Empty headers, broken semantic units, no context

```python
# OLD (Broken)
splitter = CharacterTextSplitter(chunk_size=500)
# Result: "### 3.1 Agent Creation Flow" (useless header alone)

# NEW (Smart)
class SmartChunker:
    """
    Hierarchical chunking following Kimia's research:
    1. Split by markdown headers (keeps headers WITH content)
    2. Create parent/child relationships
    3. Filter garbage chunks
    """
    
    def __init__(self):
        # Step 1: Markdown header splitting
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ]
        )
        
        # Step 2: Child chunk splitting (for precise matching)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, content: str, metadata: dict) -> List[Chunk]:
        # Split by headers first
        header_sections = self.header_splitter.split_text(content)
        
        chunks = []
        for section in header_sections:
            # Skip garbage
            if self._is_garbage(section.page_content):
                continue
            
            # Store parent content
            parent_content = section.page_content
            parent_headers = section.metadata  # {h1: "...", h2: "..."}
            
            # Create child chunks for precise matching
            children = self.child_splitter.split_text(parent_content)
            
            for i, child_content in enumerate(children):
                chunks.append(Chunk(
                    content=child_content,
                    parent_content=parent_content,  # CRITICAL: Keep parent
                    headers=parent_headers,
                    chunk_index=i,
                    source_metadata=metadata
                ))
        
        return chunks
    
    def _is_garbage(self, content: str) -> bool:
        """Reject empty headers, separators, etc."""
        stripped = content.strip()
        
        # Too short
        if len(stripped) < 50:
            return True
        
        # Just separators
        cleaned = stripped.replace('-', '').replace('=', '').replace('#', '').replace('`', '')
        if len(cleaned.strip()) < 20:
            return True
        
        # Just a header with no content
        lines = [l for l in stripped.split('\n') if l.strip()]
        if len(lines) <= 1 and lines[0].startswith('#'):
            return True
        
        return False
```

### 3.2 Hybrid Retriever

**Problem Solved:** Vector search misses exact keyword matches

```python
class HybridRetriever:
    """
    Combine vector similarity + BM25 keyword search
    Following Kimia's HybridRAG pattern
    """
    
    def __init__(
        self,
        vectorstore,  # pgvector
        documents: List[Document],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        # Vector retriever (semantic similarity)
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 20}
        )
        
        # BM25 retriever (keyword matching)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 20
        
        # Ensemble (combine both)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[vector_weight, keyword_weight]
        )
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Document]:
        """
        Hybrid search: Vector + BM25
        
        Why both?
        - Vector: "How do agents work?" → semantic similarity
        - BM25: "AgentFactory" → exact keyword match
        """
        return self.ensemble_retriever.get_relevant_documents(query)[:top_k]
```

### 3.3 Reranker

**Problem Solved:** Initial retrieval returns garbage that matches keywords but doesn't answer query

```python
class RAGReranker:
    """
    Cross-encoder reranking for quality filtering
    Uses FlashRank (FREE, runs locally)
    """
    
    def __init__(self, top_n: int = 8):
        self.reranker = FlashrankRerank(top_n=top_n)
    
    def rerank(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Rerank documents by how well they ANSWER the query
        Not just how similar they are
        
        Initial: 20 candidates
        After rerank: 8 best that actually answer the question
        """
        # FlashRank uses a cross-encoder model
        # It reads (query, document) pairs and scores them
        reranked = self.reranker.compress_documents(documents, query)
        return reranked
```

### 3.4 Cognitive Context Formatter

**Problem Solved:** Dumping chunks without structure makes it hard for LLM to reason

```python
class CognitiveContextFormatter:
    """
    Format retrieved chunks using IBM Zurich Cognitive Tools research
    
    Don't just dump chunks - scaffold reasoning:
    1. Group by source
    2. Include header hierarchy
    3. Show relevance indicators
    4. Add source citations
    """
    
    def format_context(
        self,
        chunks: List[Chunk],
        query: str,
        max_tokens: int = 2000
    ) -> str:
        # Group by source document
        by_source = self._group_by_source(chunks)
        
        parts = []
        
        # Cognitive scaffolding header
        parts.append(f"## Retrieved Knowledge\n")
        parts.append(f"**Query:** {query}\n")
        parts.append(f"**Sources:** {len(by_source)} documents, {len(chunks)} sections\n")
        parts.append("---\n")
        
        token_count = 0
        
        for source_file, source_chunks in by_source.items():
            source_header = f"\n### 📄 {source_file}\n"
            parts.append(source_header)
            
            for chunk in source_chunks:
                # Build chunk with header hierarchy
                chunk_parts = []
                
                # Show header path (h1 > h2 > h3)
                if chunk.headers:
                    header_path = " → ".join(
                        f"**{v}**" for v in chunk.headers.values() if v
                    )
                    chunk_parts.append(f"**Section:** {header_path}\n")
                
                # Content
                chunk_parts.append(chunk.content)
                
                # Relevance indicator
                if hasattr(chunk, 'relevance_score'):
                    chunk_parts.append(f"\n*Relevance: {chunk.relevance_score:.0%}*")
                
                chunk_text = "\n".join(chunk_parts)
                chunk_tokens = len(chunk_text) // 4  # Approximate
                
                if token_count + chunk_tokens > max_tokens:
                    break
                
                parts.append(chunk_text + "\n")
                token_count += chunk_tokens
        
        return "\n".join(parts)
```

### 3.5 Universal RAG Service

**The main service used by ALL components:**

```python
class UniversalRAGService:
    """
    Single RAG service for the entire platform:
    - Chatbot: Conversational, diverse results
    - Agents: Focused, task-specific
    - Document Search: User-facing, more options
    - Context Engineering: Research-oriented
    - Workflows: Deterministic, reproducible
    """
    
    def __init__(self):
        self.chunker = SmartChunker()
        self.hybrid_retriever = None  # Lazy init
        self.reranker = RAGReranker(top_n=10)
        self.formatter = CognitiveContextFormatter()
        self.embedding_manager = get_embedding_manager()
        
        # Context-specific configurations
        self.context_configs = {
            "chatbot": RAGConfig(
                expand_query=True,
                return_parents=True,
                diversity=0.3,
                max_chunks=8,
                max_tokens=2000
            ),
            "agent": RAGConfig(
                expand_query=False,
                return_parents=True,
                diversity=0.1,
                max_chunks=5,
                max_tokens=1500
            ),
            "search": RAGConfig(
                expand_query=True,
                return_parents=False,
                diversity=0.4,
                max_chunks=12,
                max_tokens=3000
            ),
            "workflow": RAGConfig(
                expand_query=False,
                return_parents=True,
                diversity=0.0,
                max_chunks=5,
                max_tokens=1500
            ),
            "context_engineering": RAGConfig(
                expand_query=True,
                return_parents=True,
                diversity=0.3,
                max_chunks=10,
                max_tokens=2500
            )
        }
    
    async def retrieve(
        self,
        query: str,
        context: Literal["chatbot", "agent", "search", "workflow", "context_engineering"] = "chatbot",
        **overrides
    ) -> RAGResult:
        """
        Universal retrieval with context-aware configuration
        """
        config = self.context_configs.get(context, self.context_configs["chatbot"])
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 1. Query transformation (optional)
        queries = [query]
        if config.expand_query:
            queries = self._expand_query(query)
        
        # 2. Hybrid search (vector + BM25)
        candidates = []
        for q in queries:
            results = await self._hybrid_search(q, top_k=20)
            candidates.extend(results)
        
        # 3. Deduplicate
        candidates = self._deduplicate(candidates)
        
        # 4. Rerank
        reranked = self.reranker.rerank(query, candidates)[:config.max_chunks]
        
        # 5. Expand to parents (if configured)
        if config.return_parents:
            reranked = self._expand_to_parents(reranked)
        
        # 6. Apply diversity (MMR)
        if config.diversity > 0:
            reranked = self._apply_mmr(reranked, config.diversity)
        
        # 7. Format with cognitive structure
        formatted_context = self.formatter.format_context(
            reranked, query, config.max_tokens
        )
        
        return RAGResult(
            chunks=reranked,
            formatted_context=formatted_context,
            total_tokens=self._count_tokens(formatted_context),
            sources=self._extract_sources(reranked),
            query=query,
            context_type=context
        )
    
    def _expand_query(self, query: str) -> List[str]:
        """Generate query variants for broader coverage"""
        # Simple expansion - can be enhanced with LLM
        variants = [query]
        
        # Add question form if not already
        if not query.endswith('?'):
            variants.append(f"What is {query}?")
            variants.append(f"How does {query} work?")
        
        return variants[:3]  # Limit to 3 variants
    
    def _expand_to_parents(self, chunks: List[Chunk]) -> List[Chunk]:
        """Replace child chunks with their parent content"""
        expanded = []
        seen_parents = set()
        
        for chunk in chunks:
            if chunk.parent_content:
                parent_id = hash(chunk.parent_content[:100])
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    # Create new chunk with parent content
                    expanded.append(Chunk(
                        content=chunk.parent_content,
                        headers=chunk.headers,
                        source_metadata=chunk.source_metadata,
                        relevance_score=chunk.relevance_score
                    ))
            else:
                expanded.append(chunk)
        
        return expanded

# Singleton
_universal_rag = None

def get_universal_rag() -> UniversalRAGService:
    global _universal_rag
    if _universal_rag is None:
        _universal_rag = UniversalRAGService()
    return _universal_rag
```

---

## 4. Database Schema Updates

### 4.1 Enhanced document_chunks Table

```sql
-- Add parent content and header hierarchy
ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS parent_content TEXT,
ADD COLUMN IF NOT EXISTS headers JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(20) DEFAULT 'child';

-- Create index for header search
CREATE INDEX IF NOT EXISTS idx_document_chunks_headers 
ON document_chunks USING gin(headers);

-- Add BM25 full-text search index
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_fts 
ON document_chunks USING gin(to_tsvector('english', content));
```

### 4.2 RAG Configuration Table

```sql
CREATE TABLE IF NOT EXISTS rag_configurations (
    id SERIAL PRIMARY KEY,
    context_type VARCHAR(50) UNIQUE NOT NULL,
    expand_query BOOLEAN DEFAULT true,
    return_parents BOOLEAN DEFAULT true,
    diversity FLOAT DEFAULT 0.3,
    max_chunks INTEGER DEFAULT 8,
    max_tokens INTEGER DEFAULT 2000,
    vector_weight FLOAT DEFAULT 0.7,
    keyword_weight FLOAT DEFAULT 0.3,
    rerank_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default configurations
INSERT INTO rag_configurations (context_type, expand_query, return_parents, diversity, max_chunks, max_tokens)
VALUES 
    ('chatbot', true, true, 0.3, 8, 2000),
    ('agent', false, true, 0.1, 5, 1500),
    ('search', true, false, 0.4, 12, 3000),
    ('workflow', false, true, 0.0, 5, 1500),
    ('context_engineering', true, true, 0.3, 10, 2500)
ON CONFLICT (context_type) DO NOTHING;
```

---

## 5. API Endpoints

### 5.1 Universal RAG Retrieve

```
POST /api/rag/retrieve
```

**Request:**
```json
{
    "query": "How does AgentFactory work?",
    "context": "chatbot",
    "max_chunks": 8,
    "max_tokens": 2000,
    "return_formatted": true
}
```

**Response:**
```json
{
    "success": true,
    "result": {
        "formatted_context": "## Retrieved Knowledge\n**Query:** How does AgentFactory work?...",
        "chunks": [
            {
                "content": "The AgentFactory class is responsible for...",
                "source": "AGENT_FLOW_GUIDE.md",
                "headers": {"h1": "Agent System", "h2": "Agent Factory"},
                "relevance_score": 0.89
            }
        ],
        "total_tokens": 1456,
        "sources": ["AGENT_FLOW_GUIDE.md", "02-AGENT-FACTORY-LIFECYCLE.md"]
    },
    "metrics": {
        "retrieval_time_ms": 234,
        "rerank_time_ms": 156,
        "total_candidates": 20,
        "after_rerank": 8
    }
}
```

### 5.2 Document Re-indexing

```
POST /api/documents/reindex-all
```

Triggers full re-chunking and re-embedding with new SmartChunker.

---

## 6. Implementation Phases

### Phase 1: Smart Chunking (3 hours)
- [ ] Create `SmartChunker` class
- [ ] Implement `MarkdownHeaderTextSplitter` integration
- [ ] Add parent/child relationship storage
- [ ] Add garbage chunk filtering
- [ ] Update `DocumentManager` to use new chunker

### Phase 2: Hybrid Search (2 hours)
- [ ] Create `HybridRetriever` class
- [ ] Integrate `BM25Retriever` from LangChain
- [ ] Create `EnsembleRetriever` wrapper
- [ ] Add full-text search index to database

### Phase 3: Reranking (1 hour)
- [ ] Install `flashrank` package
- [ ] Create `RAGReranker` class
- [ ] Integrate with retrieval pipeline

### Phase 4: Cognitive Formatting (1.5 hours)
- [ ] Create `CognitiveContextFormatter` class
- [ ] Implement source grouping
- [ ] Add header hierarchy formatting
- [ ] Add relevance indicators

### Phase 5: Universal Service (2 hours)
- [ ] Create `UniversalRAGService` class
- [ ] Implement context-specific configurations
- [ ] Add query expansion
- [ ] Add parent expansion
- [ ] Create singleton accessor

### Phase 6: Re-index & Test (1.5 hours)
- [ ] Delete all existing chunks
- [ ] Re-import all documents with new chunker
- [ ] Test with sample queries
- [ ] Verify improvement in results

**Total: ~11 hours**

---

## 7. Success Criteria

### 7.1 Chunking Quality
- [ ] NO empty header chunks in database
- [ ] ALL chunks have parent_content populated
- [ ] ALL chunks have headers JSON populated
- [ ] Garbage filter rejects < 50 char chunks

### 7.2 Retrieval Quality
- [ ] Query "AgentFactory" returns AgentFactory CODE, not just headers
- [ ] Relevance scores > 75% for top results
- [ ] Keyword matches appear in results (BM25 working)
- [ ] Reranking filters out irrelevant chunks

### 7.3 Performance
- [ ] Retrieval < 500ms for hybrid search
- [ ] Reranking < 300ms
- [ ] Total RAG pipeline < 1s

### 7.4 Universal Usage
- [ ] Chatbot uses `get_universal_rag().retrieve(..., context="chatbot")`
- [ ] Agents use `get_universal_rag().retrieve(..., context="agent")`
- [ ] Search page uses `get_universal_rag().retrieve(..., context="search")`
- [ ] Workflows use `get_universal_rag().retrieve(..., context="workflow")`

---

## 8. Testing Queries

After implementation, test with:

| Query | Expected Top Result |
|-------|---------------------|
| "How does AgentFactory work?" | Actual AgentFactory code with `create_agent()` method |
| "Show me agent creation flow" | Diagram + explanation from AGENT_FLOW_GUIDE.md |
| "database schema" | SQL schema definitions, not just mentions |
| "workflow execution" | WorkflowExecutor class code |
| "RAG retrieval" | This PRD or RAG service code |

---

## 9. Migration from v1.0

### Steps:
1. **Stop backend**
2. **Run database migrations** (add new columns)
3. **Delete all document_chunks** (will re-create)
4. **Deploy new code**
5. **Start backend**
6. **Trigger re-index** via API or upload all docs again

### Rollback:
If issues occur, revert to v1.0 by:
1. Dropping new columns
2. Re-deploying old code
3. Re-importing documents with old chunker

---

## 10. Future Enhancements

### v2.1 (Next iteration)
- [ ] LLM-powered query expansion
- [ ] Multi-modal RAG (images, tables)
- [ ] Feedback loop for relevance tuning
- [ ] A/B testing for retrieval strategies

### v2.2 (Future)
- [ ] Knowledge graph integration
- [ ] Cross-document reasoning
- [ ] Personalized retrieval per user/agent
- [ ] RAG caching layer

---

**This PRD transforms RAG from "barely working" to "production-grade" using proven research and battle-tested LangChain components.**
