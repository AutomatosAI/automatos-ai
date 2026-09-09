# Semantic Chunking Strategies

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/context/configure-rag-modal.tsx](frontend/components/context/configure-rag-modal.tsx)
- [frontend/components/documents/delete-confirmation-modal.tsx](frontend/components/documents/delete-confirmation-modal.tsx)
- [frontend/components/documents/document-details-modal.tsx](frontend/components/documents/document-details-modal.tsx)
- [frontend/components/documents/team-multi-select.tsx](frontend/components/documents/team-multi-select.tsx)
- [frontend/components/documents/upload-provider-modal.tsx](frontend/components/documents/upload-provider-modal.tsx)
- [frontend/hooks/use-context-management-api.ts](frontend/hooks/use-context-management-api.ts)
- [frontend/hooks/use-document-api.ts](frontend/hooks/use-document-api.ts)
- [frontend/hooks/use-notifications-api.ts](frontend/hooks/use-notifications-api.ts)
- [frontend/hooks/use-teams.ts](frontend/hooks/use-teams.ts)
- [frontend/hooks/use-template-api.ts](frontend/hooks/use-template-api.ts)
- [orchestrator/alembic/versions/prd158_cloud_default_team.py](orchestrator/alembic/versions/prd158_cloud_default_team.py)
- [orchestrator/api/context.py](orchestrator/api/context.py)
- [orchestrator/api/documents.py](orchestrator/api/documents.py)
- [orchestrator/api/github_webhooks.py](orchestrator/api/github_webhooks.py)
- [orchestrator/api/system.py](orchestrator/api/system.py)
- [orchestrator/core/models/cloud_sync.py](orchestrator/core/models/cloud_sync.py)
- [orchestrator/modules/rag/chunking/semantic_chunker.py](orchestrator/modules/rag/chunking/semantic_chunker.py)
- [orchestrator/modules/rag/ingestion/contextual_annotator.py](orchestrator/modules/rag/ingestion/contextual_annotator.py)
- [orchestrator/modules/rag/ingestion/manager.py](orchestrator/modules/rag/ingestion/manager.py)
- [orchestrator/modules/rag/service.py](orchestrator/modules/rag/service.py)
- [orchestrator/modules/rag/services/cloud_file_downloader.py](orchestrator/modules/rag/services/cloud_file_downloader.py)
- [orchestrator/modules/rag/services/cloud_sync_service.py](orchestrator/modules/rag/services/cloud_sync_service.py)
- [orchestrator/modules/search/services/entity_extractor.py](orchestrator/modules/search/services/entity_extractor.py)
- [orchestrator/tests/security/test_s5_closures.py](orchestrator/tests/security/test_s5_closures.py)
- [orchestrator/tests/test_entity_extractor_no_vendor_key.py](orchestrator/tests/test_entity_extractor_no_vendor_key.py)
- [orchestrator/tests/test_p2w1_contextual_annotations.py](orchestrator/tests/test_p2w1_contextual_annotations.py)

</details>



## Purpose and Scope

This page documents the semantic chunking strategies used in Automatos AI's document ingestion pipeline. Chunking is the process of splitting large documents into smaller, semantically meaningful units for vector embedding and retrieval. This page covers the `SemanticChunker` implementation, mathematical foundations for entropy-based splitting, multi-modal strategies for different file types (PDF, DOCX, MD, PY), and the parent-child expansion mechanism for context retrieval.

For information about the overall RAG retrieval system and similarity search, see [RAG Retrieval System](). For the end-to-end document ingestion flow, see [Document Ingestion Pipeline]().

---

## Chunking Architecture Overview

The ingestion pipeline transforms raw files into vectorized chunks using a layered architecture that transitions from physical file formats to semantic vector space. The `DocumentManager` coordinates this process per workspace `[orchestrator/api/documents.py:78-87]()`, utilizing `DocumentProcessor` for extraction and `SemanticChunker` for splitting `[orchestrator/modules/rag/ingestion/manager.py:44-48]()`.

### Document Ingestion Flow

```mermaid
graph TB
    subgraph "Natural Language Space"
        PDF["PDF (pdfplumber)"]
        DOCX["DOCX (python-docx)"]
        CODE["Python (PythonCodeTextSplitter)"]
        MD["Markdown (MarkdownTextSplitter)"]
    end

    subgraph "Code Entity Space"
        DocManager["DocumentManager<br/>(modules/rag/ingestion/manager.py)"]
        Proc["DocumentProcessor<br/>(modules/rag/ingestion/manager.py)"]
        SChunker["SemanticChunker<br/>(modules/rag/chunking/semantic_chunker.py)"]
    end
    
    subgraph "Semantic Space"
        Sim["Semantic Similarity"]
        Entropy["Information Density<br/>(Entropy)"]
        Hier["Hierarchical Split"]
    end

    subgraph "Vector Storage Layer"
        S3V["S3VectorsBackend<br/>(modules/search/vector_store/backends/s3_vectors_backend.py)"]
        PGV["EnhancedVectorStore<br/>(pgvector)"]
    end

    DocManager --> Proc
    Proc --> PDF
    Proc --> DOCX
    Proc --> CODE
    Proc --> MD
    
    PDF --> SChunker
    DOCX --> SChunker
    CODE --> SChunker
    MD --> SChunker
    
    SChunker --> Sim
    SChunker --> Entropy
    SChunker --> Hier
    
    Sim --> S3V
    Entropy --> S3V
    Hier --> PGV
```

**Sources:**
- `[orchestrator/modules/rag/ingestion/manager.py:113-130]()`
- `[orchestrator/modules/rag/ingestion/manager.py:44-48]()`
- `[orchestrator/modules/rag/service.py:5-10]()`

---

## Semantic Chunking Strategies

The `SemanticChunker` supports multiple advanced strategies defined in the `ChunkingStrategy` enum.

### 1. Semantic Similarity
This strategy splits text based on the cosine similarity between consecutive sentences. If the similarity falls below a defined threshold, a new chunk is started. The `RAGConfig` class allows tuning the `min_similarity` (default 0.5) and `diversity` (default 0.3) factors used during retrieval and chunking optimization `[orchestrator/modules/rag/service.py:169-172]()`.

### 2. Information Density (Entropy-based)
Utilizes Shannon entropy to calculate information density. Boundaries are placed where information density shifts significantly, ensuring that fact-dense sections are preserved as cohesive units `[orchestrator/modules/rag/service.py:5-10]()`.

### 3. Hierarchical Strategy
Creates a parent-child relationship between chunks. Large "parent" chunks provide broad context, while smaller "child" chunks allow for high-precision vector matches. This is reflected in the `DocumentChunk` dataclass which includes a `parent_content` field for context expansion `[orchestrator/modules/rag/ingestion/manager.py:100-101]()`.

### 4. Code-Aware Chunking
For Python files, the system uses the `PythonCodeTextSplitter` `[orchestrator/modules/rag/ingestion/manager.py:35-36]()`. This splitter respects class and function boundaries, preventing logic from being severed mid-definition `[orchestrator/modules/rag/ingestion/manager.py:126-129]()`.

### 5. Adaptive Strategy
The system balances chunk sizes dynamically. `RAGConfig` initializes with `chunk_size` (default 500), `min_chunk_size` (100), and `max_chunk_size` (1500) `[orchestrator/modules/rag/service.py:161-166]()`.

**Sources:**
- `[orchestrator/modules/rag/service.py:128-172]()`
- `[orchestrator/modules/rag/ingestion/manager.py:94-111]()`
- `[orchestrator/modules/rag/ingestion/manager.py:126-129]()`

---

## Multi-Modal Extraction Logic

The `DocumentProcessor` handles format-specific extraction before passing text to the chunker. It supports a wide array of MIME types including PDF, DOCX, Markdown, and various code formats `[orchestrator/api/documents.py:89-105]()`.

| Format | Library | Strategy |
| :--- | :--- | :--- |
| **PDF** | `pdfplumber` / `PyPDF2` | Primary extraction via `pdfplumber` with a `PyPDF2` fallback for robust text recovery `[orchestrator/modules/rag/ingestion/manager.py:157-194]()`. |
| **DOCX** | `python-docx` | Iterates through `doc.paragraphs` to maintain structural flow `[orchestrator/modules/rag/ingestion/manager.py:196-203]()`. |
| **Markdown** | `MarkdownTextSplitter` | Splits based on header hierarchy (h1, h2, h3) `[orchestrator/modules/rag/ingestion/manager.py:33-34]()`. |
| **Cloud Files** | `CloudFileDownloader` | Downloads from Composio (Dropbox, OneDrive, GDrive, Box) before local processing `[orchestrator/modules/rag/services/cloud_file_downloader.py:28-35]()`. |

### Extraction Data Flow

```mermaid
graph LR
    subgraph "Natural Language Space"
        File["Raw File (Cloud/Local)"]
        MIME["MIME Detection (magic)"]
    end
    
    subgraph "Code Entity Space"
        Proc["DocumentProcessor<br/>(manager.py)"]
        PDF_Ext["extract_text_from_pdf()"]
        DOCX_Ext["extract_text_from_docx()"]
    end
    
    subgraph "Semantic Space"
        SChunk["SemanticChunker<br/>(semantic_chunker.py)"]
        RAGS["RAGService<br/>(service.py)"]
    end

    File --> MIME
    MIME --> Proc
    Proc --> PDF_Ext
    Proc --> DOCX_Ext
    PDF_Ext --> SChunk
    DOCX_Ext --> SChunk
    SChunk --> RAGS
```

**Sources:**
- `[orchestrator/modules/rag/ingestion/manager.py:131-155]()`
- `[orchestrator/api/documents.py:131-149]()`
- `[orchestrator/modules/rag/services/cloud_file_downloader.py:71-84]()`
- `[orchestrator/modules/rag/service.py:1-10]()`

---

## Parent-Child Expansion & Metadata

The system uses a `DocumentChunk` dataclass to track context expansion data.

### Metadata Schema
Chunks include the following fields to facilitate retrieval expansion:
- `chunk_index`: The sequence number within the document `[orchestrator/modules/rag/ingestion/manager.py:96]()`.
- `parent_content`: Stores the text of the larger section for context injection `[orchestrator/modules/rag/ingestion/manager.py:100]()`.
- `headers`: A dictionary mapping the header hierarchy (h1, h2, h3) to the chunk `[orchestrator/modules/rag/ingestion/manager.py:101]()`.

### Context Retrieval Configuration
The `RAGConfig` allows enabling `parent_child_expansion` with a configurable `expansion_window` (default 1), which retrieves adjacent chunks to provide the LLM with surrounding context `[orchestrator/modules/rag/service.py:156-157]()`.

**Sources:**
- `[orchestrator/modules/rag/ingestion/manager.py:94-111]()`
- `[orchestrator/modules/rag/service.py:156-157]()`

---

## Entity Extraction and Knowledge Graph

The `EntityExtractor` service performs Named Entity Recognition (NER) and relationship mapping to build a knowledge graph, augmenting the basic chunking strategy `[orchestrator/modules/search/services/entity_extractor.py:40-41]()`.

1. **Regex-Based Extraction**: Fast extraction for technology names and acronyms `[orchestrator/modules/search/services/entity_extractor.py:90-121]()`.
2. **LLM Extraction**: Uses an LLM (e.g., `gpt-4o-mini`) to identify Technologies, Concepts, Organizations, People, and Products `[orchestrator/modules/search/services/entity_extractor.py:123-146]()`.
3. **Relationship Mapping**: Identifies links like `is_part_of`, `uses`, `created_by`, and `depends_on` `[orchestrator/modules/search/services/entity_extractor.py:31-37]()`.

### Graph Knowledge Architecture

```mermaid
graph TD
    subgraph "Natural Language Space"
        Doc["Document Text"]
        Query["User Question"]
    end

    subgraph "Code Entity Space"
        E_Ext["EntityExtractor<br/>(entity_extractor.py)"]
        LLM_Ext["_extract_with_llm()"]
        Rel_Ext["extract_relationships()"]
    end

    subgraph "Data Models"
        Ent["ExtractedEntity"]
        Rel["ExtractedRelationship"]
    end

    Doc --> E_Ext
    E_Ext --> LLM_Ext
    E_Ext --> Rel_Ext
    LLM_Ext --> Ent
    Rel_Ext --> Rel
```

**Sources:**
- `[orchestrator/modules/search/services/entity_extractor.py:40-63]()`
- `[orchestrator/modules/search/services/entity_extractor.py:123-155]()`
- `[orchestrator/modules/search/services/entity_extractor.py:185-190]()`

---