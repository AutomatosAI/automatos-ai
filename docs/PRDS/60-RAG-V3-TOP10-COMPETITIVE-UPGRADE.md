# PRD 60: RAG v3 — Top-10 Competitive Upgrade

**Status:** Draft
**Priority:** Critical
**Effort:** 34-44 hours (phased, includes 4h prerequisite S3 migration fix + 4h PRD-19 multimodal resurrection)
**Dependencies:** PRD-08 (RAG v2, completed), PRD-19 (Multimodal Knowledge — partially completed, 5 bugs), PRD-42/46 (Cloud Doc Sync — caused DocumentWidget breakage), PRD-09 (Context Real Data)

---

## Executive Summary

Deep competitive research against the top 10 open-source RAG projects (Dify, LangChain, RAGFlow, Pathway, Flowise, LlamaIndex, Quivr, LightRAG, Haystack, txtai) revealed **8 critical gaps** in Automatos AI's RAG implementation. While our mathematical foundations (Knapsack DP, Shannon entropy, RRF fusion) are strong, the system lacks features that every top-10 project delivers: document return to users, proper hybrid search, advanced document parsing, evaluation feedback loops, parent-child chunk retrieval, and a dedicated search UI.

This PRD addresses all 8 gaps in priority order, transforming Automatos from a "good chunker with vector search" into a **complete RAG platform** competitive with Dify, RAGFlow, and LlamaIndex.

---

## Current State: What We Have (Strengths)

| Component | File | Status |
|-----------|------|--------|
| 5-strategy SemanticChunker | `orchestrator/modules/rag/chunking/semantic_chunker.py` | Working |
| 0/1 Knapsack DP optimizer | `orchestrator/modules/search/optimization/context_optimizer.py` | Working |
| RRF multi-query fusion | `orchestrator/modules/rag/service.py:299-349` | Working |
| HyDE + query decomposition | `orchestrator/modules/rag/query_enhancer.py` | Working |
| Iterative/Agentic RAG | `orchestrator/modules/rag/iterative_rag.py` | Working |
| pgvector store (4 search modes) | `orchestrator/modules/search/vector_store/store.py` | Working |
| Cross-encoder reranking | `orchestrator/modules/rag/service.py:351-387` | Working |
| Cloud file download (Composio) | `orchestrator/modules/rag/services/cloud_file_downloader.py` | Working |
| Document upload + ingestion pipeline | `orchestrator/modules/rag/ingestion/manager.py` | Working |
| Frontend document management | `frontend/components/documents/` | Working |
| Frontend semantic search UI | `frontend/components/documents/semantic-search.tsx` | Working |

---

## Gap Analysis: What's Missing

### Gap 1: Document Return to Users (CRITICAL)
**Impact:** Users search but only see chunks — never the actual document
**Every top-10 project does this:** Dify shows inline citations with source links. RAGFlow lets you hover citations to see original content with tables/charts. LlamaIndex returns source nodes with page numbers.

**Current behavior:** The chatbot tool router (`consumers/chatbot/tool_router.py:78-157`) groups chunks by source file and generates download links, but:
- Download links point to hardcoded paths (`/var/automatos/documents/{source}`)
- No document preview capability
- No page-level or section-level citations
- No way to navigate from a chunk back to its location in the original document
- The semantic search component (`frontend/components/documents/semantic-search.tsx`) shows chunks with similarity scores but no document context

### Gap 2: Hybrid Search (BM25 + Vector) Not Wired
**Impact:** Missing keyword matches that vector search alone misses
**Every top-10 project does this:** Dify has configurable semantic/keyword weight slider. RAGFlow uses native Elasticsearch hybrid. Pathway combines Splade + dense vectors.

**Current behavior:** `EnhancedVectorStore.search()` (`modules/search/vector_store/store.py`) has a `SearchMode.HYBRID` mode that combines `1 - (embedding <=> query) * 0.7 + ts_rank(...) * 0.3`, but:
- `RAGService._get_candidates()` (`modules/rag/service.py:668-685`) hardcodes `SearchMode.VECTOR_ONLY`
- The `ts_rank` function requires a `tsvector` column on `document_chunks` — this column likely doesn't exist
- No BM25 implementation; `ts_rank` is PostgreSQL full-text search (different algorithm)
- No configurable weight between semantic and keyword results

### Gap 3: Document Parsing Quality
**Impact:** Tables, images, and structured content lost during ingestion
**RAGFlow's DeepDoc** has OCR, table structure recognition, 14+ document-aware templates. Our parser is basic.

**Current behavior:** `DocumentManager` in `ingestion/manager.py` handles:
- PDF via `pdfplumber` (text extraction only — no table structure recognition)
- DOCX via `python-docx` (text only — images/charts dropped)
- Markdown, Text, Python, JSON (basic text extraction)
- No OCR capability for scanned PDFs
- No table extraction as structured data
- No image extraction or captioning

**However:** PRD-19 built multimodal processors at `orchestrator/modules/rag/ingestion/multimodal/processors.py` (728 lines) with Camelot table extraction, Tesseract OCR, GPT-4V image descriptions, and LaTeX formula parsing. **This code exists but has 5 bugs preventing it from working** (see Phase 4B). Dependencies are already installed (`camelot-py`, `pytesseract`, `pdfplumber`, `pillow`). Fix, don't rebuild.

### Gap 4: Parent-Child Chunk Retrieval
**Impact:** Small chunks retrieved for precision, but no way to expand to surrounding context
**LangChain's `ParentDocumentRetriever`** stores small chunks for matching but returns parent sections. RAGFlow recently added parent-child chunking.

**Current behavior:**
- `DocumentChunk` dataclass in `ingestion/manager.py:92-99` already has `parent_content` and `headers` fields
- These fields are populated during Markdown header-based splitting
- But `_get_candidates()` always returns `parent_content: None` and `headers: {}` (hardcoded at lines 715-716)
- The `document_chunks` table metadata JSONB could store parent references, but nothing reads them back
- `ContextRetrievalEngine._get_surrounding_chunks()` returns empty list (placeholder)

### Gap 5: Evaluation & Feedback Loop
**Impact:** No way to know if RAG answers are good — can't improve without measurement
**LlamaIndex** has built-in evaluation (faithfulness, relevancy, answer correctness). Haystack has `EvaluationResult`. txtai has built-in scoring.

**Current behavior:**
- No user feedback mechanism (thumbs up/down on RAG answers)
- No automated evaluation (faithfulness, relevancy, hallucination detection)
- `document_usage` table tracks queries and execution time but not answer quality
- No A/B testing between RAG configurations
- No ground truth dataset for regression testing

### Gap 6: Dedicated Search Results Page
**Impact:** Users have no standalone way to search their knowledge base
**Dify** has a retrieval test panel with similarity scores. **RAGFlow** has a full search interface. Even **Flowise** shows Document Store previews.

**Current behavior:**
- Semantic search exists as a component (`frontend/components/documents/semantic-search.tsx`) embedded in the documents page
- No standalone `/search` route
- No full-text search option (only vector similarity)
- No filters (by file type, date range, tags, source)
- No pagination or infinite scroll
- Search results show chunks but not document context

### Gap 7: Knowledge Graph / Entity Retrieval
**Impact:** Missing relationships between concepts, can't do multi-hop reasoning
**LightRAG** excels here with dual-level entity + thematic graph retrieval. **LlamaIndex** has `KnowledgeGraphIndex`.

**Current behavior:**
- `EntityExtractor` exists (`modules/search/services/entity_extractor.py`) — extracts entities and relationships
- `document_relationships` table is created by `EnhancedVectorStore` but never populated
- `ContextRetrievalEngine._get_related_documents()` returns empty list (placeholder at strategy `MULTI_HOP`)
- No graph storage, no graph traversal, no entity linking

### Gap 8: Real-Time Index Sync
**Impact:** Documents updated in cloud storage don't reflect in search until manual re-sync
**Pathway** is the gold standard — incremental updates via Differential Dataflow.

**Current behavior:**
- Cloud sync (`services/cloud_sync_service.py`) is batch-only, triggered manually
- No change detection or incremental update
- Full re-ingestion required for updated documents
- No webhook listeners for cloud storage change notifications

---

## Existing Frontend Reality (IMPORTANT)

Before detailing implementation, note the **massive existing frontend** (~307 .tsx files, ~85K lines):

| Area | Components | Key Files |
|------|-----------|-----------|
| **Document Management** | 28 components | `document-management.tsx` (orchestrator with 10+ tabs), `document-library.tsx`, `modern-file-manager.tsx` |
| **Cloud Storage** | Full 5-provider OAuth | `cloud-storage-panel.tsx`, `cloud-storage-connections.tsx` (Google Drive, Dropbox, OneDrive, Box, SharePoint via Composio) |
| **Chatbot Widgets** | 9 widget types | `DocumentWidget/index.tsx` (444 lines — Content/Chunks tabs), `DataWidget`, `CodeWidget`, `FileWidget`, etc. |
| **Semantic Search** | Already exists | `semantic-search.tsx` (AI search with similarity slider, 4 context modes, debounced) |
| **Knowledge Base** | 10 components | `DatabaseQueryExplorer.tsx`, `SemanticLayerBuilder`, `QueryTemplatesGrid` |
| **Context Engineering** | 7 components | `context-engineering.tsx`, `configure-rag-modal.tsx` |
| **Analytics** | 15 components | Dashboard, charts, metrics |

### Critical Broken Feature: DocumentWidget (S3 Migration)

The `DocumentWidget` (`frontend/components/widgets/DocumentWidget/index.tsx`, 444 lines) was a **working feature** that displayed RAG results in the chatbot with Content tab + Chunks tab (similarity scores, relevance %). **It broke after PRD-42/46 migrated to S3 Vectors:**

- Widget expects `data.content` and `data.chunks` from backend — these stopped being populated
- Download creates blob from `data.content` in memory — content is now in S3, not local
- The `semantic-search.tsx` hook may be querying pgvector endpoints that changed
- Tool router (`tool_router.py:78-157`) has hardcoded local paths (`/var/automatos/documents/`)

**This must be fixed FIRST before adding new features.**

---

## Detailed Implementation Plan

### Phase 0: Fix S3 Migration Breakage (4h) — PREREQUISITE

Fix the DocumentWidget and semantic search that broke when PRD-42/46 moved to S3 Vectors.

#### 0.1 Fix DocumentWidget Data Pipeline

**File:** `orchestrator/consumers/chatbot/tool_router.py`

The tool router builds document context for chatbot responses. After S3 migration, chunk content and document content must be fetched from the new storage layer:

```python
# Fix: Ensure RAG results include content from current storage
# The tool router groups chunks by source and creates download links
# Update to use the document content API instead of local file paths
```

**File:** `frontend/components/widgets/DocumentWidget/index.tsx`

The widget already has Content tab + Chunks tab. Fix the data flow:
- Ensure `data.content` is populated by fetching from the correct backend endpoint
- Ensure `data.chunks` array with `similarity`, `content`, `metadata` is populated
- Update download handler to use API endpoint instead of in-memory blob from `data.content`

#### 0.2 Fix Semantic Search Hook

**File:** `frontend/hooks/use-semantic-search-api.ts`

Verify the hook queries the correct backend endpoint after S3 migration. The semantic search component already has the UI — it just needs the data pipeline restored.

#### 0.3 Fix Tool Router Download Links

**File:** `orchestrator/consumers/chatbot/tool_router.py:78-157`

```python
# BEFORE (line ~92):
file_path = metadata.get('file_path', f"/var/automatos/documents/{source}")

# AFTER: Use document API endpoint
document_id = doc.get('document_id', metadata.get('document_id'))
```

---

### Phase 1: Document Return & Citations (6h) — HIGHEST PRIORITY

Enhance existing components to properly return documents to users. **Do NOT create new viewer components** — the DocumentWidget and artifact-viewer already exist.

#### 1.1 Backend: Document Content API

**File:** `orchestrator/api/documents.py`

Add endpoints for retrieving document content with chunk highlighting:

```python
@router.get("/{document_id}/content")
async def get_document_content(
    document_id: int,
    highlight_chunk_ids: Optional[List[int]] = Query(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Return document content with optional chunk highlighting.

    For text-based documents (md, txt, py, json): returns raw content.
    For PDF: returns extracted text with page markers.
    For DOCX: returns extracted text with section markers.

    If highlight_chunk_ids provided, wraps matching chunk text in
    <mark> tags so frontend can scroll to and highlight the relevant section.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.workspace_id == ctx.workspace_id
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get document chunks for this document
    chunks_query = text("""
        SELECT id, content, metadata
        FROM document_chunks
        WHERE document_id = :doc_id
        ORDER BY (metadata->>'chunk_index')::int NULLS LAST
    """)
    chunks = db.execute(chunks_query, {"doc_id": document_id}).fetchall()

    # Build full content from chunks (reconstructed)
    full_content = "\n\n".join(c.content for c in chunks)

    # Apply highlighting if requested
    highlighted_content = full_content
    if highlight_chunk_ids:
        highlight_set = set(highlight_chunk_ids)
        for chunk in chunks:
            if chunk.id in highlight_set:
                highlighted_content = highlighted_content.replace(
                    chunk.content,
                    f'<mark data-chunk-id="{chunk.id}">{chunk.content}</mark>',
                    1  # Replace first occurrence only
                )

    return {
        "document_id": document_id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "content": highlighted_content,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "preview": c.content[:200],
                "metadata": json.loads(c.metadata) if isinstance(c.metadata, str) else (c.metadata or {})
            }
            for c in chunks
        ] if highlight_chunk_ids else None
    }
```

#### 1.2 Backend: Enhanced Search Response

**File:** `orchestrator/modules/rag/service.py`

Modify `_get_candidates()` to include `document_id` in every result (already does this at line 711), and add a new method:

```python
async def search_with_document_context(
    self,
    query: str,
    max_results: int = 10,
    min_similarity: float = 0.5,
    include_document_info: bool = True,
    workspace_id: str = None
) -> Dict:
    """
    Search that returns results grouped by document with full metadata.
    Used by the frontend search page (not the LLM context builder).

    Returns:
        {
            "query": str,
            "total_results": int,
            "documents": [
                {
                    "document_id": int,
                    "filename": str,
                    "file_type": str,
                    "max_similarity": float,
                    "chunks": [
                        {
                            "chunk_id": int,
                            "content": str,
                            "similarity": float,
                            "metadata": dict
                        }
                    ]
                }
            ],
            "execution_time_ms": float
        }
    """
```

This method should:
1. Call `_get_candidates()` with the workspace_id
2. Group results by `document_id`
3. For each document, fetch filename/type from `documents` table
4. Return grouped results sorted by max similarity per document

#### 1.3 Backend: Search API Endpoint

**File:** `orchestrator/api/documents.py` (or create new `orchestrator/api/search.py`)

```python
@router.get("/search")
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
    min_similarity: float = Query(0.5, ge=0.0, le=1.0),
    file_types: Optional[List[str]] = Query(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Semantic search across all documents in workspace.
    Returns results grouped by document with chunk highlights.
    """
```

#### 1.4 Frontend: Enhance Existing DocumentWidget (NOT a new component)

**File:** `frontend/components/widgets/DocumentWidget/index.tsx` (already 444 lines)

This widget already has Content tab + Chunks tab with similarity scores, relevance percentages, and download. After Phase 0 restores data flow, enhance with:
- "View in Document" button that scrolls to the chunk in full document context
- Clickable chunk → opens document content with that chunk highlighted via `<mark>` tags
- Source document link with file type icon

**DO NOT create a new `document-viewer.tsx`** — the DocumentWidget + `artifact-viewer.tsx` already handle document display.

#### 1.5 Frontend: Enhance Existing Semantic Search (small changes)

**File:** `frontend/components/documents/semantic-search.tsx` (already exists with similarity slider, 4 context modes)

Small additions to existing component:
- Group results by document (accordion/collapsible per document)
- Show document title + max relevance score at document level
- "View in Document" button that opens DocumentWidget with chunk highlighted
- Add document download button per group

#### 1.6 Frontend: Citation Badges in Chat Messages

**File:** Chat message rendering component in `frontend/components/chatbot/`

When the LLM response includes document references (from `build_tool_context_message`):
- Render clickable citation badges inline (e.g., `[1]`, `[2]`)
- On click, open DocumentWidget with the relevant chunk
- The chatbot already has `artifact-viewer.tsx` for rendering artifacts — leverage that pattern

---

### Phase 2: Wire Up Hybrid Search (4h)

#### 2.1 Add tsvector Column

**Migration:** Create an Alembic migration (or SQL script)

```sql
-- Add tsvector column to document_chunks
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS
    search_vector tsvector;

-- Populate from existing content
UPDATE document_chunks
SET search_vector = to_tsvector('english', content)
WHERE search_vector IS NULL;

-- Create GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector
ON document_chunks USING gin(search_vector);

-- Create trigger to auto-update on insert/update
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trig_update_search_vector ON document_chunks;
CREATE TRIGGER trig_update_search_vector
    BEFORE INSERT OR UPDATE OF content ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_search_vector();
```

#### 2.2 Switch RAG Service to Hybrid Mode

**File:** `orchestrator/modules/rag/service.py`

In `_get_candidates()`, change the search mode from `VECTOR_ONLY` to `HYBRID`:

```python
# BEFORE (line ~681):
search_results = await self._vector_store.search(
    query_embedding=query_embedding,
    mode=SearchMode.VECTOR_ONLY,
    ...
)

# AFTER:
search_results = await self._vector_store.search(
    query_embedding=query_embedding,
    mode=SearchMode.HYBRID,
    ranking_strategy=RankingStrategy.HYBRID_SCORE,
    limit=limit,
    query_text=query  # Already passed! Needed for ts_rank
)
```

#### 2.3 Verify Hybrid Search in EnhancedVectorStore

**File:** `orchestrator/modules/search/vector_store/store.py`

Check that the `HYBRID` search mode SQL correctly references the `search_vector` column (not computed `to_tsvector` on the fly, which is slow):

```python
# Ensure the hybrid query uses the indexed column:
# WHERE search_vector @@ plainto_tsquery('english', $query)
# Not: WHERE to_tsvector('english', content) @@ ...
```

Read the full `search()` method in `store.py` to find the HYBRID case and verify/fix the SQL.

#### 2.4 Make Hybrid Weight Configurable

**File:** `orchestrator/modules/rag/service.py` — `RAGConfig` dataclass

Add:
```python
hybrid_vector_weight: float = 0.7  # Weight for vector similarity (0.0-1.0)
hybrid_keyword_weight: float = 0.3  # Weight for keyword match (0.0-1.0)
```

Wire this through to `EnhancedVectorStore.search()` so the weights are configurable via system settings.

---

### Phase 3: Parent-Child Chunk Retrieval (4h)

#### 3.1 Store Parent References During Ingestion

**File:** `orchestrator/modules/rag/ingestion/manager.py`

During chunking (especially for Markdown with headers), store the parent chunk ID in metadata:

```python
# When creating chunks from MarkdownHeaderTextSplitter:
for i, chunk in enumerate(chunks):
    metadata = {
        "chunk_index": i,
        "headers": chunk.metadata.get("headers", {}),
        "parent_chunk_index": None,  # Will be set for child chunks
        "section_title": chunk.metadata.get("Header 1", ""),
        "subsection_title": chunk.metadata.get("Header 2", ""),
    }

    # For hierarchical splitting: small chunks reference their parent section
    if parent_section_index is not None:
        metadata["parent_chunk_index"] = parent_section_index
```

#### 3.2 Context Expansion in RAG Service

**File:** `orchestrator/modules/rag/service.py`

Add a method to expand retrieved chunks to include surrounding context:

```python
async def _expand_to_parent_context(
    self,
    candidates: List[Dict],
    expand_window: int = 1
) -> List[Dict]:
    """
    For each retrieved chunk, fetch the surrounding chunks from the same document.
    This gives the LLM more context around the matched section.

    Args:
        candidates: List of candidate chunks from _get_candidates()
        expand_window: Number of chunks before/after to include

    Returns:
        Enriched candidates with 'expanded_content' field containing
        the matched chunk + surrounding chunks concatenated.
    """
    if not candidates:
        return candidates

    import asyncpg, os
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/automatos")
    conn = await asyncpg.connect(db_url)

    try:
        for candidate in candidates:
            doc_id = candidate.get("document_id")
            chunk_metadata = candidate.get("metadata", {})
            chunk_index = chunk_metadata.get("chunk_index")

            if doc_id is None or chunk_index is None:
                candidate["expanded_content"] = candidate["content"]
                continue

            # Fetch surrounding chunks
            surrounding = await conn.fetch("""
                SELECT content, metadata
                FROM document_chunks
                WHERE document_id = $1
                  AND (metadata->>'chunk_index')::int
                      BETWEEN $2 AND $3
                ORDER BY (metadata->>'chunk_index')::int
            """, doc_id, max(0, chunk_index - expand_window), chunk_index + expand_window)

            if surrounding:
                candidate["expanded_content"] = "\n\n".join(r["content"] for r in surrounding)
            else:
                candidate["expanded_content"] = candidate["content"]
    finally:
        await conn.close()

    return candidates
```

Call this in `retrieve()` before the Knapsack optimization step, and use `expanded_content` for the context formatting while keeping original `content` for token counting relevance scores.

#### 3.3 Store chunk_index Reliably

**File:** `orchestrator/modules/rag/ingestion/manager.py`

Verify that every chunk stored in `document_chunks` has `chunk_index` in its metadata JSONB. Audit the `_store_chunks_to_db()` method (or equivalent) and ensure:

```python
metadata = {
    "chunk_index": i,
    "document_id": document_id,
    "filename": filename,
    ...
}
```

This is critical for Phase 3.2 to work.

---

### Phase 4: Improved Document Parsing (6h)

#### 4.1 Table Extraction from PDFs

**File:** `orchestrator/modules/rag/ingestion/manager.py`

`pdfplumber` already supports table extraction. Add:

```python
def _extract_pdf_with_tables(self, file_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract text AND tables from PDF.
    Tables are converted to Markdown format for better LLM comprehension.
    """
    import pdfplumber

    text_parts = []
    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract regular text
            page_text = page.extract_text() or ""
            text_parts.append(f"[Page {page_num}]\n{page_text}")

            # Extract tables
            page_tables = page.extract_tables()
            for table_idx, table in enumerate(page_tables):
                if table and len(table) > 1:
                    # Convert to Markdown table
                    headers = table[0]
                    md_table = "| " + " | ".join(str(h or "") for h in headers) + " |\n"
                    md_table += "| " + " | ".join("---" for _ in headers) + " |\n"
                    for row in table[1:]:
                        md_table += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"

                    text_parts.append(f"\n[Table {table_idx + 1}, Page {page_num}]\n{md_table}")
                    tables.append({
                        "page": page_num,
                        "index": table_idx,
                        "markdown": md_table,
                        "raw": table
                    })

    return "\n\n".join(text_parts), tables
```

#### 4.2 Add Page Numbers to Chunk Metadata

When chunking PDF content, preserve the `[Page N]` markers and store the page number(s) in chunk metadata:

```python
metadata = {
    "chunk_index": i,
    "pages": [page_num],  # List of pages this chunk spans
    "has_table": any("[Table" in chunk_content),
    ...
}
```

This enables "View on Page 5" links in the frontend.

#### 4.3 OCR for Scanned PDFs (Optional — Lower Priority)

Consider adding `pytesseract` or `surya` for OCR:

```python
# In _extract_pdf_with_tables, after extracting text:
if not page_text.strip():
    # Page has no extractable text — likely scanned
    try:
        from PIL import Image
        import pytesseract

        image = page.to_image(resolution=300)
        ocr_text = pytesseract.image_to_string(image.original)
        if ocr_text.strip():
            text_parts.append(f"[Page {page_num} - OCR]\n{ocr_text}")
    except ImportError:
        logger.warning("pytesseract not available for OCR")
```

**Dependencies to add:** `pytesseract`, `Pillow` (optional, skip if not needed for MVP)

#### 4.4 Resurrect PRD-19 Multimodal Pipeline (4h)

> **Context:** PRD-19 built a multimodal document processing system with Table, Image, Formula, and orchestrator processors. The code exists at `orchestrator/modules/rag/ingestion/multimodal/processors.py` (728 lines) with a supporting API at `orchestrator/api/knowledge_multimodal.py`, database schema in `orchestrator/core/database/init_complete_schema.sql`, and a service at `orchestrator/modules/rag/services/multimodal_knowledge_tools.py`. **The bones are solid but 5 bugs prevent it from working.**

**What exists (working processors):**
- `TableProcessor` — Camelot-based table extraction from PDFs (lattice + stream methods)
- `ImageProcessor` — OCR via Tesseract + AI image descriptions
- `FormulaProcessor` — LaTeX detection, variable/operator extraction, domain classification
- `MultimodalDocumentProcessor` — Orchestrator that runs all processors on a document
- Database schema: `knowledge_items`, `kb_tables`, `kb_images`, `kb_formulas`, `knowledge_relationships`
- API endpoints: `/types`, `/items`, `/upload`, `/search`, `/items/{id}`, `/stats`
- Dependencies already installed: `camelot-py`, `pdfplumber`, `pytesseract`, `pandas`, `pillow`

**Bug 1: workspace_id Column Missing (CRITICAL)**

The API code references `workspace_id` in 46+ places, but the column doesn't exist in the `knowledge_items` table schema.

```sql
-- Fix: Add workspace_id to knowledge_items
ALTER TABLE knowledge_items
    ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_knowledge_items_workspace
    ON knowledge_items(workspace_id);
```

**Bug 2: Deprecated GPT-4V Model Name**

**File:** `orchestrator/modules/rag/ingestion/multimodal/processors.py` (line ~431-472)

```python
# BEFORE (broken):
response = self.openai_client.chat.completions.create(
    model="gpt-4-vision-preview",  # DEPRECATED
    ...
)

# AFTER:
response = self.openai_client.chat.completions.create(
    model="gpt-4o",  # Current model with vision
    ...
)
```

**Bug 3: Factory Ignores OpenAI Key**

**File:** `orchestrator/modules/rag/ingestion/multimodal/processors.py` (line ~717-727)

```python
# BEFORE (broken):
def create_multimodal_processor(openai_key=None):
    return MultimodalDocumentProcessor()  # Ignores openai_key!

# AFTER:
def create_multimodal_processor(openai_key=None):
    openai_client = None
    if openai_key:
        from openai import OpenAI
        openai_client = OpenAI(api_key=openai_key)
    return MultimodalDocumentProcessor(openai_client=openai_client)
```

**Bug 4: Extracted Content Never Embedded**

After multimodal extraction, the content is stored in `knowledge_items` but the `embedding` column is always NULL. Fix: After extraction, generate embeddings for extracted content and store them.

```python
# In knowledge_multimodal.py upload handler, after extraction:
from modules.rag.ingestion.manager import DocumentManager

# Generate embedding for the extracted content
embedding = await doc_manager.generate_embedding(item.content)

# Store embedding alongside the knowledge item
await db.execute(text("""
    UPDATE knowledge_items SET embedding = :embedding WHERE id = :id
"""), {"embedding": embedding, "id": item.id})
```

**Bug 5: Multimodal Not Connected to RAG Ingestion Pipeline**

The multimodal processors exist as standalone, but the main `DocumentManager.ingest_file()` pipeline doesn't call them. Wire it in:

```python
# In orchestrator/modules/rag/ingestion/manager.py, after text extraction:
if file_type == 'pdf' and self.enable_multimodal:
    from modules.rag.ingestion.multimodal.processors import create_multimodal_processor
    processor = create_multimodal_processor(openai_key=self.openai_key)
    multimodal_results = await processor.process(file_path)

    # Append extracted tables as additional chunks
    for table in multimodal_results.get('tables', []):
        chunks.append({
            "content": table['markdown'],
            "metadata": {"type": "table", "page": table.get('page'), **base_metadata}
        })

    # Append image descriptions as additional chunks
    for image in multimodal_results.get('images', []):
        if image.get('description'):
            chunks.append({
                "content": f"[Image: {image['description']}]",
                "metadata": {"type": "image", "page": image.get('page'), **base_metadata}
            })
```

**Diagram processor:** PRD-19 planned diagram analysis but never built it. Out of scope for this fix — can revisit when there's demand.

#### 4.5 XLSX/CSV as Markdown Tables (existing section)

**File:** `orchestrator/modules/rag/ingestion/manager.py`

Currently these formats may not be handled well. Add:

```python
def _extract_spreadsheet(self, file_path: str) -> str:
    """Extract spreadsheet data as Markdown tables for LLM consumption."""
    import openpyxl

    wb = openpyxl.load_workbook(file_path, data_only=True)
    parts = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        parts.append(f"## Sheet: {sheet_name}\n")

        # Build Markdown table
        headers = rows[0]
        md = "| " + " | ".join(str(h or "") for h in headers) + " |\n"
        md += "| " + " | ".join("---" for _ in headers) + " |\n"
        for row in rows[1:]:
            md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"

        parts.append(md)

    return "\n\n".join(parts)
```

---

### Phase 5: Evaluation & Feedback Loop (4h)

#### 5.1 Database: Feedback Table

Create migration:

```sql
CREATE TABLE IF NOT EXISTS rag_feedback (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    response_text TEXT,
    chunk_ids INTEGER[],          -- Which chunks were used
    document_ids INTEGER[],       -- Which documents were referenced
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),  -- 1-5 star rating
    feedback_type VARCHAR(50),    -- 'thumbs_up', 'thumbs_down', 'star_rating', 'correction'
    correction_text TEXT,         -- User's corrected answer (if provided)
    rag_config_id INTEGER,        -- Which RAG config was used
    execution_time_ms FLOAT,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_rag_feedback_workspace ON rag_feedback(workspace_id);
CREATE INDEX idx_rag_feedback_created ON rag_feedback(created_at);
```

#### 5.2 Backend: Feedback API

**File:** `orchestrator/api/context.py` (or new `orchestrator/api/rag_feedback.py`)

```python
@router.post("/feedback")
async def submit_rag_feedback(
    feedback: RAGFeedbackRequest,  # Pydantic model
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Submit feedback on a RAG-generated response.
    Used by chatbot UI when user clicks thumbs up/down or rates an answer.
    """

@router.get("/feedback/stats")
async def get_feedback_stats(
    time_range: str = Query("7d"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get aggregated feedback statistics.
    Returns: avg rating, thumbs up %, common correction patterns,
    worst-performing RAG configs, most-cited documents.
    """
```

#### 5.3 Frontend: Feedback UI in Chat

When a chatbot message uses RAG (detected by presence of document citations), show:
- Thumbs up / thumbs down buttons
- Optional 1-5 star rating on expand
- "Suggest correction" text area

**File:** Add to the chat message component in `frontend/components/chat/` or `frontend/components/chatbot/`

#### 5.4 Backend: Automated Relevance Scoring

**File:** `orchestrator/modules/rag/service.py`

Add post-retrieval relevance validation:

```python
async def _score_retrieval_quality(
    self,
    query: str,
    chunks: List[Dict],
    response: str = None
) -> Dict:
    """
    Score the quality of retrieved chunks using heuristics.
    Called after retrieval to log quality metrics.

    Returns:
        {
            "avg_similarity": float,
            "source_diversity": float,
            "coverage_score": float,    # Do chunks cover the query's concepts?
            "freshness_score": float,   # How recent are the source documents?
        }
    """
```

Log these scores to `document_usage` for trending analysis.

---

### Phase 6: Dedicated Search Route (2h) — Reduced Scope

**Note:** `semantic-search.tsx` already exists as a full component with AI-powered search, similarity slider, and 4 context modes ('documents', 'chatbot', 'agent', 'patterns'). It's currently embedded in the documents page. This phase just gives it a standalone route and adds filters.

#### 6.1 Frontend: Search Route

**File:** Create `frontend/app/search/page.tsx`

```tsx
// Simply wraps the existing semantic-search component with a layout
export default function SearchPage() {
    return (
        <MainLayout>
            <EnhancedSemanticSearch standalone={true} />
        </MainLayout>
    )
}
```

#### 6.2 Frontend: Enhance Existing Semantic Search for Standalone Mode

**File:** `frontend/components/documents/semantic-search.tsx` (MODIFY, not create new)

Add a `standalone` prop that, when true:
- Shows a larger search bar at top
- Adds filter sidebar: file type checkboxes, date range, tags
- Adds search mode toggle: "Semantic" vs "Keyword" vs "Hybrid" (once Phase 2 is done)
- Adds sort options: Relevance, Date, Name
- Shows document-grouped results (from Phase 1.5 enhancements)

**DO NOT create a separate `knowledge-search.tsx`** — enhance the existing component.

#### 6.3 Add Search to Navigation

**File:** The main sidebar/navigation component

Add a "Search" link to the sidebar navigation.

---

### Phase 7: Knowledge Graph Foundation (6h)

This is the largest phase and can be deferred if timeline is tight. However, it's what separates top-5 from top-10.

#### 7.1 Populate document_relationships Table

**File:** `orchestrator/modules/search/vector_store/store.py`

The `document_relationships` table already exists (created by `EnhancedVectorStore._ensure_tables()`). Currently never populated.

During document ingestion, after chunking:

```python
async def _extract_and_store_relationships(
    self,
    document_id: int,
    chunks: List[Dict],
    db_conn
):
    """
    Extract entity relationships between chunks using EntityExtractor,
    then store in document_relationships table.
    """
    from modules.search.services.entity_extractor import EntityExtractor

    extractor = EntityExtractor()

    for chunk in chunks:
        entities = await extractor.extract_entities(chunk["content"])

        for entity in entities:
            # Find other chunks that mention the same entity
            related_chunks = [
                c for c in chunks
                if c["id"] != chunk["id"]
                and entity.name.lower() in c["content"].lower()
            ]

            for related in related_chunks:
                await db_conn.execute("""
                    INSERT INTO document_relationships
                    (source_id, target_id, relationship_type, strength)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                """, chunk["id"], related["id"], entity.entity_type, 0.8)
```

#### 7.2 Implement Multi-Hop Retrieval

**File:** `orchestrator/modules/search/retrieval/context_retrieval_engine.py`

Replace the placeholder `_get_related_documents()`:

```python
async def _get_related_documents(
    self,
    initial_results: List[SearchResult],
    max_hops: int = 2
) -> List[SearchResult]:
    """
    Follow document_relationships to find related chunks.
    Implements breadth-first graph traversal up to max_hops.
    """
    visited = set(r.document.id for r in initial_results)
    current_ids = [r.document.id for r in initial_results]
    related = []

    for hop in range(max_hops):
        if not current_ids:
            break

        # Find related chunk IDs
        results = await self.vector_store.pool.fetch("""
            SELECT DISTINCT target_id, strength
            FROM document_relationships
            WHERE source_id = ANY($1)
              AND target_id != ALL($2)
            ORDER BY strength DESC
            LIMIT 10
        """, current_ids, list(visited))

        next_ids = []
        for r in results:
            if r["target_id"] not in visited:
                visited.add(r["target_id"])
                next_ids.append(r["target_id"])

        # Fetch the actual chunks for related IDs
        if next_ids:
            chunks = await self.vector_store.pool.fetch("""
                SELECT id, content, metadata
                FROM document_chunks
                WHERE id = ANY($1)
            """, next_ids)

            for chunk in chunks:
                related.append(...)  # Convert to SearchResult

        current_ids = next_ids

    return related
```

#### 7.3 Entity Extraction During Ingestion

**File:** `orchestrator/modules/rag/ingestion/pipeline.py`

Add a post-ingestion step that calls entity extraction:

```python
# After embedding generation in ingest_file():
if self.enable_entity_extraction:
    await self._extract_and_store_relationships(document_id, chunks, conn)
```

Make this configurable (off by default since it's LLM-intensive).

---

### Phase 8: Polish & Integration (4h)

#### 8.1 Update RAG Configuration UI

**File:** `frontend/components/context/configure-rag-modal.tsx`

Add toggles for:
- Hybrid search enable/disable
- Vector/keyword weight slider (when hybrid enabled)
- Parent-child expansion enable/disable
- Expansion window size (1-3 chunks)

#### 8.2 Update Context Engineering Dashboard

**File:** `frontend/components/context/context-engineering.tsx`

Add metrics cards for:
- Hybrid search hit rate (keyword matches vs vector matches)
- Average chunk expansion ratio
- Feedback score trends
- Entity relationship count

#### 8.3 Update System Settings

**File:** Relevant system settings components

Add RAG v3 settings:
- `hybrid_search_enabled` (boolean, default: true)
- `hybrid_vector_weight` (float, default: 0.7)
- `hybrid_keyword_weight` (float, default: 0.3)
- `parent_child_expansion` (boolean, default: true)
- `expansion_window` (int, default: 1)
- `entity_extraction_enabled` (boolean, default: false)
- `ocr_enabled` (boolean, default: false)

---

## File Change Summary

### Backend Files to Modify
| File | Phase | Changes |
|------|-------|---------|
| `orchestrator/consumers/chatbot/tool_router.py` | 0, 1 | Fix hardcoded local paths, use document API endpoints, populate DocumentWidget data |
| `orchestrator/modules/rag/service.py` | 1, 2, 3, 5 | Add `search_with_document_context()`, wire hybrid search, add `_expand_to_parent_context()`, add `_score_retrieval_quality()` |
| `orchestrator/modules/rag/ingestion/manager.py` | 3, 4 | Add `_extract_pdf_with_tables()`, `_extract_spreadsheet()`, page numbers, parent chunk refs, `chunk_index` |
| `orchestrator/modules/search/vector_store/store.py` | 2, 7 | Verify HYBRID SQL uses `search_vector` column, add relationship population |
| `orchestrator/modules/search/retrieval/context_retrieval_engine.py` | 3, 7 | Implement `_get_related_documents()` and `_get_surrounding_chunks()` |
| `orchestrator/api/documents.py` | 1 | Add `GET /{id}/content`, `GET /search` endpoints |
| `orchestrator/modules/rag/ingestion/multimodal/processors.py` | 4B | Fix GPT-4V model name, fix factory OpenAI key passthrough |
| `orchestrator/api/knowledge_multimodal.py` | 4B | Fix workspace_id references (46+ places) |
| `orchestrator/modules/rag/services/multimodal_knowledge_tools.py` | 4B | Fix SQL schema references |
| `orchestrator/modules/rag/ingestion/manager.py` | 4B | Wire multimodal processors into ingestion pipeline |

### Backend Files to Create
| File | Purpose |
|------|---------|
| `orchestrator/api/rag_feedback.py` | Feedback submission and stats endpoints |
| Database migration for `search_vector` column + `rag_feedback` table + `knowledge_items.workspace_id` | SQL migration |

### Frontend Files to Modify (Enhance Existing — NOT Create New)
| File | Phase | Changes |
|------|-------|---------|
| `frontend/components/widgets/DocumentWidget/index.tsx` | 0, 1 | **Fix broken data pipeline** (S3 migration), add "View in Document" link |
| `frontend/hooks/use-semantic-search-api.ts` | 0 | **Fix endpoint** after S3 migration |
| `frontend/components/documents/semantic-search.tsx` | 1, 6 | Group by document, add standalone mode with filters |
| `frontend/components/context/configure-rag-modal.tsx` | 8 | Add hybrid search, parent-child, expansion settings |
| `frontend/components/context/context-engineering.tsx` | 8 | Add new metric cards |
| Navigation/sidebar component | 6 | Add Search link |
| Chat message component | 1 | Add citation badges |

### Frontend Files to Create (Minimal — Only What Doesn't Exist)
| File | Purpose |
|------|---------|
| `frontend/app/search/page.tsx` | Search route (wraps existing `semantic-search.tsx`) |
| `frontend/components/chat/rag-feedback.tsx` | Thumbs up/down + rating UI |
| `frontend/components/chat/citation-badge.tsx` | Inline citation rendering |

**Removed from original plan (already exist):**
- ~~`frontend/components/documents/document-viewer.tsx`~~ → Use existing `DocumentWidget` + `artifact-viewer.tsx`
- ~~`frontend/components/search/knowledge-search.tsx`~~ → Enhance existing `semantic-search.tsx`

---

## Priority Matrix

| Phase | Feature | Impact | Effort | Priority |
|-------|---------|--------|--------|----------|
| **0** | **Fix S3 Migration Breakage** | **CRITICAL** | **4h** | **P0 — Do First (prerequisite)** |
| 1 | Document Return & Citations | CRITICAL | 6h (reduced — existing components) | P0 — Do First |
| 2 | Hybrid Search (BM25 + Vector) | HIGH | 4h | P0 — Do First |
| 3 | Parent-Child Chunk Retrieval | HIGH | 4h | P1 — Do Second |
| 4 | Document Parsing (tables, OCR) | MEDIUM | 6h | P1 — Do Second |
| 4B | **Resurrect PRD-19 Multimodal** (5 bug fixes) | **HIGH** | **4h** | **P1 — Do with Phase 4** |
| 5 | Evaluation & Feedback | HIGH | 4h | P1 — Do Second |
| 6 | Dedicated Search Route | MEDIUM | 2h (reduced — enhance existing) | P2 — Do Third |
| 7 | Knowledge Graph Foundation | LOW (now) | 6h | P3 — Future |
| 8 | Polish & Integration | LOW | 4h | P3 — Future |

**Recommended order:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 + 4B → Phase 5 → Phase 6 → Phase 7 → Phase 8

---

## Success Criteria

- [ ] Users can click a search result and view the full document with the relevant chunk highlighted
- [ ] Chatbot responses include clickable citations that link to source documents
- [ ] Hybrid search returns results that keyword-only and vector-only searches miss individually
- [ ] Short content (code snippets, config values, definitions) is not penalized in retrieval
- [ ] Parent context expansion gives LLM 2-3x more context around matched chunks
- [ ] PDF tables are extracted as Markdown and searchable
- [ ] Users can rate RAG answers with thumbs up/down
- [ ] A dedicated `/search` page exists with filters and document-grouped results
- [ ] All searches are workspace-isolated (multi-tenant security)
- [ ] Retrieval quality metrics are logged and visible in the context engineering dashboard

---

## Competitive Positioning After Implementation

| Feature | Dify | RAGFlow | LlamaIndex | **Automatos (After)** |
|---------|------|---------|------------|----------------------|
| Chunking Strategies | 2 (basic, custom) | 14+ templates | 20+ splitters | **5 semantic + LangChain** |
| Hybrid Search | Yes | Yes (native) | Yes | **Yes (pgvector + tsvector)** |
| Document Return | Citations + links | Hoverable refs | Source nodes | **Viewer + highlights** |
| Parent-Child | No | Yes (recent) | Yes | **Yes** |
| Table Extraction | Via plugins | DeepDoc (excellent) | Via LlamaParse | **pdfplumber tables** |
| Feedback Loop | Basic | No | Evaluation suite | **Ratings + auto-scoring** |
| Math Optimization | No | No | No | **Knapsack DP (unique)** |
| Multi-Query RRF | Agent Node | No | SubQuestionQuery | **Built-in** |
| Agentic RAG | Yes | Yes | Yes | **Iterative + cognitive tools** |
| Search UI | Retrieval test | Full UI | No (library) | **Full page + filters** |

---

## Technical Notes for Implementation

### Database Connection Patterns
- **ORM queries:** Use `db: Session = Depends(get_db)` for SQLAlchemy ORM operations
- **Raw SQL in API routes:** Use `db.execute(text("..."))` with SQLAlchemy text()
- **Raw SQL in services:** Use `asyncpg.connect(db_url)` for async operations (as in `_get_candidates`)
- **Workspace isolation:** Always filter by `workspace_id` from `RequestContext`

### Auth Pattern
```python
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

@router.get("/endpoint")
async def handler(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    workspace_id = ctx.workspace_id  # Always available
```

### Frontend API Pattern
```typescript
// Use SWR hooks (see existing patterns in frontend/hooks/)
const { data, isLoading, error } = useSWR(
    `/api/documents/search?q=${query}&limit=10`,
    fetcher
)
```

### Embedding Dimension
The system uses configurable embedding dimensions via `system_settings.vector_store_dimensions` (default: 1024). Don't hardcode dimensions.

### Existing Search Hook
`frontend/hooks/use-semantic-search-api.ts` — existing SWR hook for semantic search. Extend or create new hooks for the enhanced search endpoints.

---

**Estimated Total Effort:** 34-44 hours (all phases, including Phase 0 prerequisite + Phase 4B multimodal)
**MVP (Phases 0-3):** 18 hours (includes fixing S3 breakage)
**Core (Phases 0-5 + 4B):** 32 hours (includes multimodal resurrection)
**Priority:** Critical — Phase 0 (S3 fix) is a regression that must be fixed regardless
**Dependencies:** PRD-08 completed, PRD-19 (Multimodal Knowledge — partially completed, has 5 bugs), PRD-42/46 (Cloud Doc Sync — caused the breakage)
