# PRD-42: Cloud Document Sync with S3 Vectors

**Version:** 1.0
**Status:** 🟡 Planning
**Date:** 2026-01-31
**Author:** System Architecture Team
**Prerequisites:** PRD-36 (Composio Integration), PRD-11 (Document Management)
**Blocks:** None

---

## Executive Summary

Transform the Automatos AI Platform document management system from a local upload-based model to a **hybrid cloud-native architecture** where users connect their existing cloud storage providers (Google Drive, Dropbox, OneDrive, Box, SharePoint, S3) via Composio-managed OAuth. Instead of storing duplicate copies of user documents, the system will:

1. **Sync document metadata** from cloud storage providers
2. **Download, chunk, and embed** documents using existing semantic chunking pipeline
3. **Store only vector embeddings** in AWS S3 Vectors (not original files)
4. **Enable RAG queries** across all connected storage providers
5. **Maintain workspace isolation** and GDPR compliance

**Key Benefits:**
- ✅ **90-95% cost reduction** - only pay for vector storage (~$6/100GB vs $730+/10TB for full storage)
- ✅ **Zero document liability** - users keep documents in their own cloud, we never store originals
- ✅ **GDPR/compliance friendly** - data stays in user's control
- ✅ **OAuth handled by Composio** - no custom authentication code needed
- ✅ **Unified RAG** - query across Google Drive, Dropbox, etc. from single interface
- ✅ **Scales to 2 billion vectors** per index with S3 Vectors

**Target Users:** Single users and small businesses (1-10 people) with 1-2 cloud storage providers

---

## Problem Statement

### Current System Limitations

The existing document management system (PRD-11) has several constraints that limit scalability and increase operational costs:

**Storage Costs:**
- System stores full document copies locally (filesystem or S3)
- Vector embeddings stored separately in PostgreSQL with pgvector
- For 1000 users with 10GB documents each: ~$730-1230/month storage costs
- Duplicate storage: original file + chunks + vectors

**User Experience:**
- Users must manually upload documents to Automatos
- No sync with existing document repositories (Google Drive, Dropbox, etc.)
- Documents live in isolated system, not user's primary workspace
- No automatic updates when cloud documents change

**Compliance & Liability:**
- We store and are responsible for user documents
- GDPR data retention requirements fall on us
- Security breach exposes user documents
- Users lose control over their own data

**Current Architecture:**
```
User → Upload File → Local Storage (/var/automatos/documents/)
                  ↓
            PostgreSQL (metadata + vectors via pgvector)
                  ↓
            RAG Query → Chunks + Original File
```

### What Doesn't Work

Currently, the system:
- ❌ Does NOT sync from cloud storage providers
- ❌ Does NOT use Composio's 500+ app integrations for documents
- ❌ Does NOT leverage AWS S3 Vectors (new 2026 service)
- ❌ Does NOT support folder navigation in cloud storage
- ❌ Does NOT handle real-time document updates via webhooks
- ❌ Stores 60 test documents that can be deleted (fresh start opportunity)

---

## Success Criteria

### Phase 1: MVP (Pilot Launch)
- [ ] Users can connect Google Drive, Dropbox, OneDrive, Box, SharePoint via Composio OAuth
- [ ] Users can select a root folder (e.g., "Automatos") for syncing
- [ ] System displays folder tree and file list from connected storage
- [ ] Users can manually trigger "Sync Now" for selected folders
- [ ] Documents are chunked using existing SemanticChunker (adaptive strategy)
- [ ] Embeddings stored in AWS S3 Vectors (not PostgreSQL pgvector)
- [ ] Original files NOT stored locally (only metadata references)
- [ ] RAG queries work across all connected storage providers (unified search)
- [ ] Users can view/download original files from cloud storage
- [ ] Disconnecting a storage app prompts: "Delete vectors or keep for queries?"
- [ ] All 60 existing test documents migrated or deleted (clean slate)
- [ ] Workspace isolation maintained (workspace_id scoping)

### Phase 2: Production (Auto-Sync)
- [ ] Real-time sync via Composio webhooks (file created/updated/deleted)
- [ ] Background job handles periodic sync fallback (every 30 mins)
- [ ] Document change detection (modified_at timestamp comparison)
- [ ] Incremental sync (only changed files re-processed)
- [ ] Sync status dashboard (last synced, pending files, errors)

### Phase 3: Scale (Future)
- [ ] Support 1000+ users with 100GB+ documents each
- [ ] Sub-100ms RAG query latency via S3 Vectors warm cache
- [ ] Advanced folder filtering (exclude patterns, file type filters)
- [ ] Document version history tracking
- [ ] Collaborative document annotations

---

## Goals

1. **Reduce Storage Costs by 90%+**
   - Eliminate local document storage
   - Store only vector embeddings (100x smaller than originals)
   - Target: $56/month for 1000 users vs $730+ current

2. **Enable Cloud-Native Document Access**
   - Users connect existing Google Drive, Dropbox, OneDrive, Box, SharePoint, S3
   - OAuth managed entirely by Composio (no custom auth code)
   - Folder navigation and file selection UI

3. **Maintain RAG Quality**
   - Preserve existing semantic chunking quality (SemanticChunker with adaptive strategy)
   - Same or better retrieval accuracy with S3 Vectors vs pgvector
   - Support 10+ chunks per query with context optimization

4. **GDPR Compliance & Data Sovereignty**
   - User documents never leave their cloud storage
   - We store only vector embeddings (not PII/document content)
   - Users maintain full control and ownership

5. **Workspace Isolation**
   - Each workspace has independent S3 vector bucket
   - Vectors scoped by workspace_id
   - No cross-workspace data leakage

---

## Functional Requirements

### FR-1: Cloud Storage Connection Management

**FR-1.1:** Users can connect cloud storage providers via Composio OAuth
- Supported providers: Google Drive, Dropbox, OneDrive, Box, SharePoint, Amazon S3
- OAuth flow handled by Composio (hosted auth UI)
- Connection status tracked in `composio_connections` table
- UI shows: app logo, connection status (active/pending/error), connected date

**FR-1.2:** Users can disconnect cloud storage providers
- Disconnect triggers confirmation modal: "Delete synced vectors or keep for RAG queries?"
- If "Delete": Remove all `cloud_documents` and S3 vector entries for that connection
- If "Keep": Mark connection as disconnected but preserve vectors (read-only)
- Update `composio_connections.status = 'disconnected'`

**FR-1.3:** System validates connection health on page load
- Check if Composio connection still active (token not expired)
- Display warning badge if connection needs re-authorization
- Auto-refresh expired tokens via Composio SDK

### FR-2: Folder Navigation & Selection

**FR-2.1:** After connecting a storage provider, users see folder tree interface
- Root folder displays top-level folders from cloud storage
- Click folder to expand children (lazy loading)
- Display folder metadata: name, path, file count, last modified

**FR-2.2:** Users select ONE root folder per app for syncing
- Suggested: Create shared "Automatos" folder in cloud storage
- UI shows: "Select a folder to sync. We recommend creating a dedicated 'Automatos' folder."
- Once selected, sync all files and subfolders recursively underneath
- Store selection in `cloud_sync_config` table: `(connection_id, root_folder_path)`

**FR-2.3:** Folder tree displays file type icons and sync status
- Icons: PDF, Word, Excel, Text, Markdown, etc.
- Sync badges: ✅ Synced, ⏳ Pending, ❌ Error, ⊘ Excluded
- Filter by file type (checkboxes: PDFs, Docs, Sheets, Text)

### FR-3: Manual Document Sync (Pilot)

**FR-3.1:** Users trigger sync via "Sync Now" button
- Button appears next to each connected storage provider
- Initiates background job: `sync_cloud_documents_task(connection_id, root_folder_path)`
- Shows progress modal: "Syncing 47 files from Google Drive..."
- Real-time progress updates via WebSocket or polling

**FR-3.2:** Sync process workflow
```python
1. List all files in root_folder recursively via Composio API
   - GOOGLEDRIVE_LIST_FILES (recursive)
   - DROPBOX_LIST_FOLDER (recursive)
   - ONEDRIVE_LIST_FILES (recursive)

2. For each file:
   a. Check if already synced (compare external_file_id in cloud_documents)
   b. If new or modified (cloud_modified_at > last_synced_at):
      - Download file content via Composio (GOOGLEDRIVE_DOWNLOAD_FILE, etc.)
      - Save to temp file
      - Extract text (existing DocumentProcessor logic)
      - Chunk using SemanticChunker (adaptive strategy, 500 target size)
      - Generate embeddings (BAAI/bge-large-en-v1.5, 1024 dimensions)
      - Store vectors in S3 Vectors (NOT local storage)
      - Create cloud_documents metadata record
      - Delete temp file
   c. If unchanged: skip

3. Update last_synced_at timestamp
4. Return sync summary: {synced: 23, skipped: 18, errors: 6}
```

**FR-3.3:** Sync respects file size and type limits
- Max file size: 50MB (configurable via system settings)
- Supported types: PDF, DOCX, TXT, MD, JSON, XLSX, PPTX
- Unsupported files logged with reason: "File type not supported"

### FR-4: AWS S3 Vectors Integration

**FR-4.1:** Create workspace-scoped S3 vector buckets
- Bucket naming: `automatos-vectors-{workspace_id}`
- Index naming: `documents-index`
- Dimension: 1024 (matches BAAI/bge-large-en-v1.5)
- Distance metric: COSINE

**FR-4.2:** Vector insertion flow
```python
import boto3

s3vectors = boto3.client('s3vectors')

# Ensure bucket + index exist
s3vectors.create_vector_bucket(Bucket=bucket_name)
s3vectors.create_vector_index(
    Bucket=bucket_name,
    VectorIndex={
        'IndexName': 'documents-index',
        'IndexDimension': 1024,
        'DistanceMetric': 'COSINE'
    }
)

# Insert vectors
vector_objects = []
for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vector_objects.append({
        'Key': f'doc_{external_file_id}_chunk_{idx}',
        'Vector': {'float32': embedding},  # 1024-dim array
        'Metadata': {
            'external_file_id': external_file_id,
            'chunk_index': idx,
            'chunk_text': chunk[:500],  # First 500 chars for preview
            'app_name': app_name,  # GOOGLEDRIVE, DROPBOX, etc.
            'workspace_id': str(workspace_id),
            'file_name': file_name,
            'file_path': file_path
        }
    })

s3vectors.put_vectors(
    Bucket=bucket_name,
    VectorIndex='documents-index',
    VectorObjects=vector_objects
)
```

**FR-4.3:** Vector querying flow
```python
# Generate query embedding
query_embedding = await embedding_manager.generate_embedding(query_text)

# Query S3 Vectors
results = s3vectors.query_vectors(
    Bucket=f'automatos-vectors-{workspace_id}',
    VectorIndex='documents-index',
    QueryVector={'float32': query_embedding},
    MaxResults=top_k,  # Default: 10
    MinScore=0.5       # Minimum similarity threshold
)

# Format results
formatted_results = []
for result in results.get('Matches', []):
    formatted_results.append({
        'score': result['Score'],
        'chunk_text': result['Metadata']['chunk_text'],
        'external_file_id': result['Metadata']['external_file_id'],
        'app_name': result['Metadata']['app_name'],
        'file_name': result['Metadata']['file_name']
    })
```

### FR-5: Document Viewing & Download

**FR-5.1:** Users can view document content
- Click document in file list → Opens preview modal
- Download original file from cloud storage via Composio API
- Display in browser using appropriate viewer:
  - PDF: Embedded PDF viewer
  - Text/Markdown: Syntax-highlighted text area
  - Images: Image preview
  - Office docs: Download prompt (no inline preview for MVP)

**FR-5.2:** Download flow
```python
# User clicks "Download" or "View"
# 1. Lookup cloud_documents by id
cloud_doc = db.query(CloudDocument).get(document_id)

# 2. Get Composio connection
connection = db.query(ComposioConnection).get(cloud_doc.connection_id)
entity_id = connection.composio_entity_id

# 3. Download via Composio
if app_name == "GOOGLEDRIVE":
    file_content = composio.execute_action(
        action="GOOGLEDRIVE_DOWNLOAD_FILE",
        params={"fileId": cloud_doc.external_file_id},
        entity_id=entity_id
    )
elif app_name == "DROPBOX":
    file_content = composio.execute_action(
        action="DROPBOX_DOWNLOAD_FILE",
        params={"path": cloud_doc.file_path},
        entity_id=entity_id
    )

# 4. Return file content with correct MIME type
return Response(
    content=file_content["data"]["content"],
    media_type=cloud_doc.mime_type,
    headers={"Content-Disposition": f'inline; filename="{cloud_doc.file_name}"'}
)
```

**FR-5.3:** Temporary caching for performance
- Cache downloaded files for 1 hour in `/tmp/automatos_previews/{hash}`
- Serve from cache if requested again within 1 hour
- Background job cleans up cache files older than 1 hour

### FR-6: Unified RAG Queries

**FR-6.1:** RAG endpoint queries all connected storage providers by default
- Endpoint: `POST /api/cloud-documents/rag/query`
- Parameters:
  ```json
  {
    "query": "Find documents about Q4 revenue projections",
    "top_k": 10,
    "min_similarity": 0.5,
    "max_tokens": 2000
  }
  ```

**FR-6.2:** Query process
```python
async def rag_query(query, workspace_id, top_k=10, min_similarity=0.5):
    # 1. Generate query embedding
    query_embedding = await embedding_manager.generate_embedding(query)

    # 2. Query S3 Vectors (searches across ALL apps automatically)
    results = s3vectors.query_vectors(
        Bucket=f'automatos-vectors-{workspace_id}',
        VectorIndex='documents-index',
        QueryVector={'float32': query_embedding},
        MaxResults=top_k * 2,  # Over-fetch for diversity filtering
        MinScore=min_similarity
    )

    # 3. Apply existing context optimization (Knapsack, MMR, diversity)
    optimized_chunks = context_optimizer.optimize(
        chunks=results['Matches'],
        max_tokens=max_tokens,
        diversity_factor=0.3
    )

    # 4. Format context for LLM
    formatted_context = format_context(optimized_chunks)

    return {
        "context": formatted_context,
        "chunks": optimized_chunks,
        "sources": extract_sources(optimized_chunks)
    }
```

**FR-6.3:** Result formatting includes source attribution
```
Context:
---
Source: Google Drive - /Automatos/Q4 Planning.docx (Similarity: 0.87)
Chunk: "Our Q4 revenue projections show 23% growth year-over-year..."

Source: Dropbox - /Business/Financial Reports/2026-Forecast.pdf (Similarity: 0.82)
Chunk: "Revenue breakdown by region: North America $4.2M, Europe $2.1M..."
---
```

### FR-7: Migration from Existing System

**FR-7.1:** Deprecate local document upload (phased approach)

**Phase 1: Add cloud sync (keep upload)**
- Add "Connect Cloud Storage" tab to document management UI
- Keep existing "Upload Files" tab functional
- Both systems work in parallel during pilot

**Phase 2: Migrate test documents**
- Delete 60 existing test documents from PostgreSQL + local storage
- Clean up `documents` and `document_chunks` tables
- Drop pgvector extension (optional, can keep for other features)

**Phase 3: Remove upload UI (post-pilot)**
- Remove "Upload Files" tab
- Redirect users to "Connect Cloud Storage"
- Update documentation and help text

**FR-7.2:** Database migration script
```sql
-- 1. Backup existing data
CREATE TABLE documents_backup AS SELECT * FROM documents;
CREATE TABLE document_chunks_backup AS SELECT * FROM document_chunks;

-- 2. Delete test documents
DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents);
DELETE FROM documents;

-- 3. Optional: Drop pgvector if not used elsewhere
-- DROP EXTENSION IF EXISTS vector;

-- 4. Add new cloud_documents table (see schema section)
```

### FR-8: Error Handling & Resilience

**FR-8.1:** OAuth token expiration
- Detect expired tokens on Composio API calls (401 errors)
- Display banner: "Your Google Drive connection needs re-authorization"
- Clicking banner triggers OAuth refresh flow
- Auto-retry failed sync after re-auth

**FR-8.2:** File download failures
- If Composio API fails (file deleted, permissions changed), log error
- Mark document as `sync_status: 'error'` in cloud_documents
- Display error in UI with actionable message: "File no longer accessible in Google Drive"
- Retry logic: 3 attempts with exponential backoff

**FR-8.3:** S3 Vectors quota limits
- Monitor S3 Vectors bucket size and index capacity
- Alert when approaching 2 billion vector limit per index
- Graceful degradation: If S3 Vectors unavailable, fallback to "View Document" only (no RAG)

---

## Technical Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER'S CLOUD STORAGE                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │  Google    │  │  Dropbox   │  │  OneDrive  │  │    Box     │       │
│  │   Drive    │  │            │  │            │  │            │       │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘       │
│        ↑               ↑               ↑               ↑                │
│        └───────────────┴───────────────┴───────────────┘                │
│                           OAuth + File Access                           │
│                    (Managed by Composio - no custom auth)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          COMPOSIO PLATFORM                              │
│  - OAuth token management                                               │
│  - GOOGLEDRIVE_LIST_FILES, GOOGLEDRIVE_DOWNLOAD_FILE                   │
│  - DROPBOX_LIST_FOLDER, DROPBOX_DOWNLOAD_FILE                          │
│  - ONEDRIVE_LIST_FILES, ONEDRIVE_DOWNLOAD_FILE                         │
│  - Webhook triggers (file created/updated/deleted)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      AUTOMATOS AI PLATFORM                              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  BACKEND (FastAPI)                                              │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │ Cloud Document Sync Service                              │  │   │
│  │  │  - list_cloud_files(connection_id, folder_path)         │  │   │
│  │  │  - sync_folder(connection_id, root_folder_path)         │  │   │
│  │  │  - download_and_process(connection_id, file_id)         │  │   │
│  │  │  - query_vectors(workspace_id, query, top_k)            │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │ Existing Document Processor (REUSE)                     │  │   │
│  │  │  - extract_text_from_pdf()                              │  │   │
│  │  │  - extract_text_from_docx()                             │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │ Existing SemanticChunker (REUSE)                        │  │   │
│  │  │  - Strategy: ADAPTIVE                                    │  │   │
│  │  │  - Target size: 500 tokens                              │  │   │
│  │  │  - Min: 100, Max: 1500                                  │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │
│  │  │ Existing EmbeddingManager (REUSE)                       │  │   │
│  │  │  - Model: BAAI/bge-large-en-v1.5                        │  │   │
│  │  │  - Dimension: 1024                                      │  │   │
│  │  │  - Batch processing                                     │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  FRONTEND (React/TypeScript)                                    │   │
│  │  - CloudStorageConnections.tsx (connect/disconnect apps)       │   │
│  │  - FolderNavigator.tsx (tree view with file selection)         │   │
│  │  - CloudDocumentList.tsx (synced documents table)              │   │
│  │  - SyncButton.tsx (manual "Sync Now" trigger)                  │   │
│  │  - DocumentPreview.tsx (view/download from cloud)              │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA STORAGE LAYER                             │
│                                                                          │
│  ┌──────────────────────────────┐  ┌─────────────────────────────────┐│
│  │  PostgreSQL                  │  │  AWS S3 Vectors                 ││
│  │  - cloud_documents           │  │  - Bucket: automatos-vectors-   ││
│  │    (metadata ONLY)           │  │    {workspace_id}               ││
│  │  - composio_connections      │  │  - Index: documents-index       ││
│  │  - cloud_sync_config         │  │  - Dimension: 1024              ││
│  │  - workspaces                │  │  - Metric: COSINE               ││
│  │                              │  │  - Capacity: 2 billion vectors  ││
│  │  NO pgvector!                │  │                                 ││
│  │  NO original files!          │  │  ONLY vector embeddings!        ││
│  └──────────────────────────────┘  └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Document Sync

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. USER CLICKS "SYNC NOW"                                               │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. FRONTEND → POST /api/cloud-documents/sync                            │
│    { connection_id: 123, root_folder_path: "/Automatos" }              │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. BACKEND: CloudDocumentSyncService.sync_folder()                      │
│    a. Get Composio connection & entity_id from database                │
│    b. Call Composio API: GOOGLEDRIVE_LIST_FILES (recursive)            │
│    c. Get list of files: [{id, name, path, mimeType, modifiedTime}]    │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. FOR EACH FILE:                                                       │
│    a. Check if exists in cloud_documents (by external_file_id)         │
│    b. If new OR modified (modifiedTime > last_synced_at):              │
│       - Download via Composio: GOOGLEDRIVE_DOWNLOAD_FILE               │
│       - Save to /tmp/automatos_sync/{hash}_{filename}                  │
│       - Extract text: DocumentProcessor.extract_text_from_pdf()        │
│       - Chunk: SemanticChunker.chunk_text() → 20 chunks                │
│       - Embed: EmbeddingManager.generate_embedding() → 20 x 1024-dim   │
│       - Store in S3 Vectors: s3vectors.put_vectors()                   │
│       - Create cloud_documents record (metadata only)                  │
│       - Delete temp file                                               │
│    c. If unchanged: skip (no re-processing)                            │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. UPDATE DATABASE                                                      │
│    - cloud_documents.last_synced_at = NOW()                            │
│    - cloud_documents.sync_status = 'synced'                            │
│    - cloud_documents.chunk_count = 20                                  │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. RETURN RESPONSE                                                      │
│    { synced: 15, skipped: 8, errors: 2, total_chunks: 300 }           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: RAG Query

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. USER ASKS QUESTION: "Find Q4 revenue projections"                   │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. FRONTEND → POST /api/cloud-documents/rag/query                       │
│    { query: "Find Q4 revenue projections", top_k: 10 }                 │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. GENERATE QUERY EMBEDDING                                             │
│    embedding = EmbeddingManager.generate_embedding(query)              │
│    → [0.234, -0.567, 0.891, ... ] (1024 dimensions)                   │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. QUERY S3 VECTORS                                                     │
│    results = s3vectors.query_vectors(                                  │
│        Bucket='automatos-vectors-{workspace_id}',                      │
│        VectorIndex='documents-index',                                  │
│        QueryVector={'float32': embedding},                             │
│        MaxResults=20,  # Over-fetch                                    │
│        MinScore=0.5                                                    │
│    )                                                                   │
│    → Returns 20 chunks sorted by similarity                            │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. APPLY CONTEXT OPTIMIZATION (REUSE EXISTING)                         │
│    optimized = ContextOptimizer.optimize(                              │
│        chunks=results,                                                 │
│        max_tokens=2000,                                                │
│        diversity_factor=0.3                                            │
│    )                                                                   │
│    - Knapsack algorithm selects best 10 chunks within 2000 tokens     │
│    - MMR diversity scoring prevents duplicate information             │
│    - Source diversity penalty (prefer different docs)                 │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. FORMAT CONTEXT FOR LLM                                               │
│    context = """                                                       │
│    Source: Google Drive - Q4 Planning.docx (Score: 0.87)              │
│    Our Q4 revenue projections show 23% growth...                       │
│                                                                         │
│    Source: Dropbox - Financial Report.pdf (Score: 0.82)               │
│    Revenue breakdown: North America $4.2M...                           │
│    """                                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. RETURN TO FRONTEND                                                   │
│    {                                                                   │
│      context: "...",                                                   │
│      chunks: [...],                                                    │
│      sources: [                                                        │
│        {file_name: "Q4 Planning.docx", app: "GOOGLEDRIVE", score: 0.87}│
│      ]                                                                 │
│    }                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### New Tables

```sql
-- Cloud storage sync configuration per connection
CREATE TABLE cloud_sync_config (
    id SERIAL PRIMARY KEY,
    connection_id INTEGER NOT NULL REFERENCES composio_connections(id) ON DELETE CASCADE,
    root_folder_path VARCHAR(1000) NOT NULL,  -- e.g., "/Automatos"
    sync_enabled BOOLEAN DEFAULT TRUE,
    last_sync_at TIMESTAMP,
    next_sync_at TIMESTAMP,
    sync_frequency_minutes INTEGER DEFAULT 30,  -- For auto-sync (Phase 2)

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(connection_id)
);

-- Cloud documents metadata (NO file content stored!)
CREATE TABLE cloud_documents (
    id SERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    connection_id INTEGER NOT NULL REFERENCES composio_connections(id) ON DELETE CASCADE,

    -- External cloud storage identifiers
    app_name VARCHAR(100) NOT NULL,  -- GOOGLEDRIVE, DROPBOX, ONEDRIVE, etc.
    external_file_id VARCHAR(255) NOT NULL,  -- Google Drive file ID, Dropbox path, etc.
    file_name VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000),  -- Full path in cloud storage
    mime_type VARCHAR(100),
    file_size BIGINT,

    -- Vector storage references (NO local file storage!)
    s3_vector_bucket VARCHAR(255) NOT NULL,  -- automatos-vectors-{workspace_id}
    s3_vector_index VARCHAR(255) NOT NULL DEFAULT 'documents-index',
    chunk_count INTEGER DEFAULT 0,

    -- Sync tracking
    cloud_modified_at TIMESTAMP,  -- Last modified in cloud storage
    last_synced_at TIMESTAMP,
    sync_status VARCHAR(50) DEFAULT 'pending',  -- pending, syncing, synced, error
    sync_error TEXT,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(connection_id, external_file_id),
    INDEX idx_cloud_documents_workspace (workspace_id),
    INDEX idx_cloud_documents_connection (connection_id),
    INDEX idx_cloud_documents_app (app_name),
    INDEX idx_cloud_documents_sync_status (sync_status)
);

-- Sync job tracking (for background tasks)
CREATE TABLE cloud_sync_jobs (
    id SERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    connection_id INTEGER NOT NULL REFERENCES composio_connections(id) ON DELETE CASCADE,

    status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Results
    files_synced INTEGER DEFAULT 0,
    files_skipped INTEGER DEFAULT 0,
    files_errored INTEGER DEFAULT 0,
    total_chunks_created INTEGER DEFAULT 0,

    error_message TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);
```

### Modified Tables

```sql
-- Extend composio_connections with sync metadata
ALTER TABLE composio_connections
ADD COLUMN sync_enabled BOOLEAN DEFAULT FALSE,
ADD COLUMN total_documents_synced INTEGER DEFAULT 0,
ADD COLUMN last_successful_sync TIMESTAMP;
```

### Indexes for Performance

```sql
-- Fast lookups for workspace queries
CREATE INDEX idx_cloud_documents_workspace_status
ON cloud_documents(workspace_id, sync_status);

-- Fast lookups for file change detection
CREATE INDEX idx_cloud_documents_modified
ON cloud_documents(connection_id, cloud_modified_at);

-- Fast lookups for sync jobs
CREATE INDEX idx_cloud_sync_jobs_workspace_status
ON cloud_sync_jobs(workspace_id, status);
```

### Migration Script

```sql
-- File: alembic/versions/20260131_add_cloud_document_sync.py

"""Add cloud document sync with S3 Vectors

Revision ID: 20260131_cloud_sync
Revises: 20260125_add_composio_action_metadata
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

def upgrade():
    # Create cloud_sync_config table
    op.create_table(
        'cloud_sync_config',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('connection_id', sa.Integer(), sa.ForeignKey('composio_connections.id', ondelete='CASCADE'), nullable=False),
        sa.Column('root_folder_path', sa.String(1000), nullable=False),
        sa.Column('sync_enabled', sa.Boolean(), server_default='true'),
        sa.Column('last_sync_at', sa.DateTime()),
        sa.Column('next_sync_at', sa.DateTime()),
        sa.Column('sync_frequency_minutes', sa.Integer(), server_default='30'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    op.create_unique_constraint('uq_cloud_sync_config_connection', 'cloud_sync_config', ['connection_id'])

    # Create cloud_documents table
    op.create_table(
        'cloud_documents',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('connection_id', sa.Integer(), sa.ForeignKey('composio_connections.id', ondelete='CASCADE'), nullable=False),
        sa.Column('app_name', sa.String(100), nullable=False),
        sa.Column('external_file_id', sa.String(255), nullable=False),
        sa.Column('file_name', sa.String(500), nullable=False),
        sa.Column('file_path', sa.String(1000)),
        sa.Column('mime_type', sa.String(100)),
        sa.Column('file_size', sa.BigInteger()),
        sa.Column('s3_vector_bucket', sa.String(255), nullable=False),
        sa.Column('s3_vector_index', sa.String(255), nullable=False, server_default='documents-index'),
        sa.Column('chunk_count', sa.Integer(), server_default='0'),
        sa.Column('cloud_modified_at', sa.DateTime()),
        sa.Column('last_synced_at', sa.DateTime()),
        sa.Column('sync_status', sa.String(50), server_default='pending'),
        sa.Column('sync_error', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    op.create_unique_constraint('uq_cloud_documents_connection_file', 'cloud_documents', ['connection_id', 'external_file_id'])
    op.create_index('idx_cloud_documents_workspace', 'cloud_documents', ['workspace_id'])
    op.create_index('idx_cloud_documents_connection', 'cloud_documents', ['connection_id'])
    op.create_index('idx_cloud_documents_app', 'cloud_documents', ['app_name'])
    op.create_index('idx_cloud_documents_sync_status', 'cloud_documents', ['sync_status'])
    op.create_index('idx_cloud_documents_workspace_status', 'cloud_documents', ['workspace_id', 'sync_status'])
    op.create_index('idx_cloud_documents_modified', 'cloud_documents', ['connection_id', 'cloud_modified_at'])

    # Create cloud_sync_jobs table
    op.create_table(
        'cloud_sync_jobs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('connection_id', sa.Integer(), sa.ForeignKey('composio_connections.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('started_at', sa.DateTime()),
        sa.Column('completed_at', sa.DateTime()),
        sa.Column('files_synced', sa.Integer(), server_default='0'),
        sa.Column('files_skipped', sa.Integer(), server_default='0'),
        sa.Column('files_errored', sa.Integer(), server_default='0'),
        sa.Column('total_chunks_created', sa.Integer(), server_default='0'),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_cloud_sync_jobs_workspace_status', 'cloud_sync_jobs', ['workspace_id', 'status'])

    # Extend composio_connections
    op.add_column('composio_connections', sa.Column('sync_enabled', sa.Boolean(), server_default='false'))
    op.add_column('composio_connections', sa.Column('total_documents_synced', sa.Integer(), server_default='0'))
    op.add_column('composio_connections', sa.Column('last_successful_sync', sa.DateTime()))

    # IMPORTANT: Delete existing test documents (only 60, can be removed)
    op.execute("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents)")
    op.execute("DELETE FROM documents")
    # Note: Keep documents and document_chunks tables for now (backward compatibility)
    # Can be dropped in Phase 2 if fully migrating away from local uploads

def downgrade():
    op.drop_column('composio_connections', 'last_successful_sync')
    op.drop_column('composio_connections', 'total_documents_synced')
    op.drop_column('composio_connections', 'sync_enabled')
    op.drop_table('cloud_sync_jobs')
    op.drop_table('cloud_documents')
    op.drop_table('cloud_sync_config')
```

---

## API Endpoints

### Cloud Storage Connections

#### GET /api/cloud-documents/connections
List all connected cloud storage providers for workspace.

**Response:**
```json
{
  "connections": [
    {
      "id": 123,
      "app_name": "GOOGLEDRIVE",
      "status": "active",
      "connected_at": "2026-01-15T10:30:00Z",
      "sync_enabled": true,
      "total_documents_synced": 47,
      "last_successful_sync": "2026-01-31T09:15:00Z",
      "root_folder_path": "/Automatos"
    },
    {
      "id": 124,
      "app_name": "DROPBOX",
      "status": "active",
      "connected_at": "2026-01-20T14:22:00Z",
      "sync_enabled": true,
      "total_documents_synced": 23,
      "last_successful_sync": "2026-01-31T09:20:00Z",
      "root_folder_path": "/Apps/Automatos"
    }
  ]
}
```

#### POST /api/cloud-documents/connect/{app_name}
Initiate OAuth connection to cloud storage provider.

**Parameters:**
- `app_name`: GOOGLEDRIVE | DROPBOX | ONEDRIVE | BOX | SHAREPOINT | AMAZONS3

**Request:**
```json
{
  "callback_url": "https://app.automatos.ai/documents?connected=GOOGLEDRIVE"
}
```

**Response:**
```json
{
  "redirect_url": "https://auth.composio.dev/oauth/authorize?client_id=...",
  "app_name": "GOOGLEDRIVE"
}
```

#### DELETE /api/cloud-documents/connections/{connection_id}
Disconnect cloud storage provider (with confirmation).

**Query Params:**
- `delete_vectors`: boolean (if true, delete all vectors; if false, mark disconnected but keep vectors)

**Response:**
```json
{
  "status": "disconnected",
  "vectors_deleted": true,
  "documents_affected": 47
}
```

### Folder Navigation & Selection

#### GET /api/cloud-documents/connections/{connection_id}/folders
List folders in cloud storage (lazy loading for tree navigation).

**Query Params:**
- `path`: string (optional, default: root) - e.g., "/Automatos/Projects"

**Response:**
```json
{
  "folders": [
    {
      "name": "Automatos",
      "path": "/Automatos",
      "file_count": 12,
      "last_modified": "2026-01-30T18:45:00Z",
      "has_children": true
    },
    {
      "name": "Personal",
      "path": "/Personal",
      "file_count": 5,
      "last_modified": "2026-01-25T10:00:00Z",
      "has_children": true
    }
  ],
  "current_path": "/"
}
```

#### GET /api/cloud-documents/connections/{connection_id}/files
List files in a specific folder.

**Query Params:**
- `path`: string (optional, default: root)
- `file_types`: string[] (optional) - e.g., ["pdf", "docx", "txt"]
- `recursive`: boolean (optional, default: false)

**Response:**
```json
{
  "files": [
    {
      "external_file_id": "1a2b3c4d5e6f",
      "name": "Q4 Planning.docx",
      "path": "/Automatos/Q4 Planning.docx",
      "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "size": 245760,
      "modified_at": "2026-01-28T14:30:00Z",
      "is_synced": true,
      "sync_status": "synced",
      "chunk_count": 18
    },
    {
      "external_file_id": "7h8i9j0k1l2m",
      "name": "Meeting Notes.pdf",
      "path": "/Automatos/Meeting Notes.pdf",
      "mime_type": "application/pdf",
      "size": 512000,
      "modified_at": "2026-01-30T09:15:00Z",
      "is_synced": false,
      "sync_status": "pending",
      "chunk_count": 0
    }
  ],
  "total": 2,
  "path": "/Automatos"
}
```

#### POST /api/cloud-documents/connections/{connection_id}/select-folder
Set root folder for syncing.

**Request:**
```json
{
  "root_folder_path": "/Automatos"
}
```

**Response:**
```json
{
  "status": "success",
  "root_folder_path": "/Automatos",
  "sync_enabled": true
}
```

### Document Sync

#### POST /api/cloud-documents/connections/{connection_id}/sync
Manually trigger sync for a connection.

**Request:**
```json
{
  "force": false  // If true, re-process all files even if unchanged
}
```

**Response:**
```json
{
  "job_id": 456,
  "status": "running",
  "message": "Sync job started. Processing files from /Automatos"
}
```

#### GET /api/cloud-documents/sync-jobs/{job_id}
Get sync job status.

**Response:**
```json
{
  "id": 456,
  "status": "completed",
  "started_at": "2026-01-31T10:00:00Z",
  "completed_at": "2026-01-31T10:05:23Z",
  "files_synced": 15,
  "files_skipped": 8,
  "files_errored": 2,
  "total_chunks_created": 287,
  "errors": [
    {
      "file_name": "Corrupted.pdf",
      "error": "Failed to extract text: PDF is encrypted"
    },
    {
      "file_name": "TooLarge.zip",
      "error": "File exceeds 50MB limit"
    }
  ]
}
```

#### GET /api/cloud-documents/connections/{connection_id}/sync-status
Get current sync status summary.

**Response:**
```json
{
  "connection_id": 123,
  "app_name": "GOOGLEDRIVE",
  "sync_enabled": true,
  "last_sync_at": "2026-01-31T10:05:23Z",
  "total_documents": 47,
  "synced": 45,
  "pending": 0,
  "errors": 2,
  "total_chunks": 892
}
```

### Document Operations

#### GET /api/cloud-documents/{document_id}
Get cloud document metadata.

**Response:**
```json
{
  "id": 789,
  "app_name": "GOOGLEDRIVE",
  "file_name": "Q4 Planning.docx",
  "file_path": "/Automatos/Q4 Planning.docx",
  "file_size": 245760,
  "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "chunk_count": 18,
  "cloud_modified_at": "2026-01-28T14:30:00Z",
  "last_synced_at": "2026-01-31T10:02:15Z",
  "sync_status": "synced",
  "s3_vector_bucket": "automatos-vectors-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "s3_vector_index": "documents-index"
}
```

#### GET /api/cloud-documents/{document_id}/download
Download original file from cloud storage.

**Response:**
- Headers: `Content-Type: {mime_type}`, `Content-Disposition: inline; filename="{file_name}"`
- Body: File content (binary)

#### GET /api/cloud-documents/{document_id}/preview
Get document preview (first 5 chunks).

**Response:**
```json
{
  "document_id": 789,
  "file_name": "Q4 Planning.docx",
  "preview_chunks": [
    {
      "chunk_index": 0,
      "content": "Q4 Revenue Projections\n\nOur Q4 revenue projections show 23% growth year-over-year driven by enterprise expansion..."
    },
    {
      "chunk_index": 1,
      "content": "Regional Breakdown\n\nNorth America: $4.2M (+18%)\nEurope: $2.1M (+35%)\nAsia Pacific: $1.8M (+42%)..."
    }
  ],
  "total_chunks": 18,
  "showing": 2
}
```

#### DELETE /api/cloud-documents/{document_id}
Delete cloud document (removes vectors, keeps file in cloud storage).

**Response:**
```json
{
  "status": "deleted",
  "vectors_removed": 18,
  "message": "Document metadata and vectors deleted. Original file remains in Google Drive."
}
```

### RAG Queries

#### POST /api/cloud-documents/rag/query
Query documents using RAG across all connected storage providers.

**Request:**
```json
{
  "query": "Find documents about Q4 revenue projections",
  "top_k": 10,
  "min_similarity": 0.5,
  "max_tokens": 2000,
  "diversity_factor": 0.3
}
```

**Response:**
```json
{
  "query": "Find documents about Q4 revenue projections",
  "context": "Source: Google Drive - /Automatos/Q4 Planning.docx (Score: 0.87)\nOur Q4 revenue projections show 23% growth...\n\nSource: Dropbox - /Business/Financial Report.pdf (Score: 0.82)\nRevenue breakdown by region...",
  "chunks": [
    {
      "document_id": 789,
      "file_name": "Q4 Planning.docx",
      "app_name": "GOOGLEDRIVE",
      "file_path": "/Automatos/Q4 Planning.docx",
      "chunk_index": 0,
      "content": "Our Q4 revenue projections show 23% growth year-over-year driven by enterprise expansion...",
      "similarity_score": 0.87,
      "chunk_tokens": 142
    },
    {
      "document_id": 801,
      "file_name": "Financial Report.pdf",
      "app_name": "DROPBOX",
      "file_path": "/Business/Financial Report.pdf",
      "chunk_index": 3,
      "content": "Revenue breakdown by region: North America $4.2M, Europe $2.1M...",
      "similarity_score": 0.82,
      "chunk_tokens": 98
    }
  ],
  "sources": [
    {
      "document_id": 789,
      "file_name": "Q4 Planning.docx",
      "app_name": "GOOGLEDRIVE",
      "file_path": "/Automatos/Q4 Planning.docx",
      "chunks_used": 1
    },
    {
      "document_id": 801,
      "file_name": "Financial Report.pdf",
      "app_name": "DROPBOX",
      "file_path": "/Business/Financial Report.pdf",
      "chunks_used": 1
    }
  ],
  "total_tokens": 240,
  "diversity_score": 0.73
}
```

#### POST /api/cloud-documents/search
Semantic search across cloud documents (simpler than RAG).

**Request:**
```json
{
  "query": "Q4 revenue",
  "limit": 20,
  "min_similarity": 0.7,
  "filter_apps": ["GOOGLEDRIVE", "DROPBOX"]  // Optional
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": 789,
      "file_name": "Q4 Planning.docx",
      "app_name": "GOOGLEDRIVE",
      "file_path": "/Automatos/Q4 Planning.docx",
      "similarity_score": 0.87,
      "snippet": "...Q4 revenue projections show 23% growth year-over-year..."
    }
  ],
  "total": 1
}
```

---

## Backend Implementation

### File Structure (UPDATED - Uses Existing Infrastructure)

```
orchestrator/
├── api/
│   ├── documents.py                # EXISTING: Extend with cloud endpoints
│   └── composio.py                 # EXISTING: Already has OAuth (NO CHANGES)
│
├── modules/
│   ├── rag/
│   │   ├── services/
│   │   │   ├── cloud_sync_service.py         # NEW: Cloud file sync orchestrator
│   │   │   └── cloud_file_downloader.py      # NEW: Composio file download wrapper
│   │   │
│   │   ├── ingestion/
│   │   │   ├── manager.py                    # EXISTING: Reuse DocumentManager
│   │   │   ├── pipeline.py                   # EXTEND: Add ingest_from_cloud()
│   │   │   └── cloud_source.py               # NEW: Cloud file source adapter
│   │   │
│   │   ├── chunking/
│   │   │   └── semantic_chunker.py           # EXISTING: NO CHANGES
│   │   │
│   │   └── service.py                        # EXTEND: Add cloud query support
│   │
│   └── search/
│       ├── vector_store/
│       │   ├── store.py                      # EXISTING: EnhancedVectorStore base
│       │   ├── backends/
│       │   │   ├── pgvector_backend.py       # EXTRACT: Current pgvector logic
│       │   │   └── s3_vectors_backend.py     # NEW: S3 Vectors backend
│       │   └── __init__.py                   # Factory pattern for backends
│       │
│       └── optimization/
│           └── context_optimizer.py          # EXISTING: NO CHANGES
│
├── core/
│   ├── models/
│   │   ├── composio.py                       # EXISTING: Already has connections
│   │   ├── core.py                           # EXTEND: Add cloud_documents table
│   │   └── sync_state.py                     # NEW: Cloud sync state models
│   │
│   ├── composio/
│   │   ├── client.py                         # EXISTING: Use execute_action()
│   │   ├── entity_manager.py                 # EXISTING: Use for workspace entities
│   │   └── tool_executor.py                  # EXISTING: Use for file downloads
│   │
│   ├── llm/
│   │   └── embedding_manager.py              # EXISTING: NO CHANGES
│   │
│   └── services/
│       └── sync_scheduler.py                 # NEW: Background sync orchestration
│
└── alembic/
    └── versions/
        └── 20260131_add_cloud_sync.py        # NEW: Migration (minimal)
```

### Architecture Principles

**REUSE > EXTEND > CREATE**

1. ✅ **REUSE** existing `ComposioClient.execute_action()` for file downloads
2. ✅ **REUSE** existing `IngestionPipeline` for document processing
3. ✅ **REUSE** existing `SemanticChunker` for chunking
4. ✅ **REUSE** existing `EntityManager` for workspace isolation
5. ✅ **EXTEND** `EnhancedVectorStore` with backend abstraction
6. ✅ **EXTEND** `IngestionPipeline` with cloud source support
7. ✅ **CREATE** only new cloud-specific adapters

---

## Integration with Existing Infrastructure

### 1. Use Existing ComposioToolExecutor for File Downloads

**DON'T CREATE**: New Composio client wrapper
**DO REUSE**: Existing `core/composio/tool_executor.py`

```python
# orchestrator/modules/rag/services/cloud_file_downloader.py
"""Uses existing ToolExecutor - no new Composio client!"""

from core.composio.tool_executor import ComposioToolExecutor
from typing import BinaryIO
import tempfile

class CloudFileDownloader:
    """Thin wrapper around existing ToolExecutor for file downloads."""

    def __init__(self, db: Session):
        self.executor = ComposioToolExecutor(db)  # EXISTING

    async def download_file(
        self,
        app_name: str,
        external_file_id: str,
        workspace_id: str,
        agent_id: int = None  # Optional: for access control
    ) -> str:
        """
        Downloads file from cloud storage using EXISTING ToolExecutor.
        Returns path to temporary file.
        """

        # Map app to Composio action (existing pattern)
        action_map = {
            "GOOGLEDRIVE": "GOOGLEDRIVE_DOWNLOAD_FILE",
            "DROPBOX": "DROPBOX_DOWNLOAD_FILE",
            "ONEDRIVE": "ONEDRIVE_DOWNLOAD_FILE",
            "BOX": "BOX_DOWNLOAD_FILE"
        }

        action = action_map.get(app_name.upper())
        if not action:
            raise ValueError(f"Unsupported app: {app_name}")

        # Build params based on app
        if app_name == "GOOGLEDRIVE":
            params = {"fileId": external_file_id}
        elif app_name in ["DROPBOX", "ONEDRIVE"]:
            params = {"path": external_file_id}
        else:
            params = {"id": external_file_id}

        # REUSE existing ToolExecutor.execute() - already handles:
        # - Entity resolution
        # - Connection validation
        # - Action mapping
        # - Error handling
        result = await self.executor.execute(
            action=action,
            params=params,
            agent_id=agent_id,  # Uses existing access control
            workspace_id=workspace_id,
            app_name=app_name
        )

        if not result.get("success"):
            raise Exception(f"Download failed: {result.get('error')}")

        # Save to temp file
        file_content = result["data"]["content"]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        tmp.write(file_content)
        tmp.close()

        return tmp.name
```

### 2. Extend Existing IngestionPipeline

**DON'T CREATE**: New document processor
**DO EXTEND**: Existing `modules/rag/ingestion/pipeline.py`

```python
# orchestrator/modules/rag/ingestion/pipeline.py
# EXISTING FILE - ADD new method

class IngestionPipeline:
    # ... existing methods ...

    async def ingest_from_cloud(
        self,
        app_name: str,
        external_file_id: str,
        file_name: str,
        workspace_id: str,
        metadata: Dict = None
    ) -> IngestionResult:
        """
        NEW METHOD - Ingests file from cloud storage.
        REUSES all existing chunking, embedding, storage logic.
        """

        # 1. Download using EXISTING CloudFileDownloader
        downloader = CloudFileDownloader(self.db)
        tmp_path = await downloader.download_file(
            app_name=app_name,
            external_file_id=external_file_id,
            workspace_id=workspace_id
        )

        try:
            # 2. REUSE existing ingest_file() - already handles:
            #    - Text extraction (PDF, DOCX, etc.)
            #    - Chunking via SemanticChunker
            #    - Embedding generation
            #    - Vector storage
            result = self.ingest_file(
                file_path=tmp_path,
                metadata={
                    **metadata,
                    "source": "cloud",
                    "app_name": app_name,
                    "external_file_id": external_file_id,
                    "file_name": file_name
                }
            )

            return result

        finally:
            # 3. Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
```

### 3. Extend Existing EnhancedVectorStore with Backend Abstraction

**DON'T CREATE**: Separate S3 Vectors service
**DO EXTEND**: Add backend pattern to `modules/search/vector_store/store.py`

```python
# orchestrator/modules/search/vector_store/__init__.py
"""Factory pattern for vector store backends."""

from typing import Literal
from .store import EnhancedVectorStore
from .backends.pgvector_backend import PgVectorBackend
from .backends.s3_vectors_backend import S3VectorsBackend

VectorBackend = Literal["pgvector", "s3_vectors"]

def get_vector_store(
    backend: VectorBackend = "pgvector",
    workspace_id: str = None,
    **kwargs
) -> EnhancedVectorStore:
    """
    Factory for vector stores with pluggable backends.

    Args:
        backend: "pgvector" (existing) or "s3_vectors" (new)
        workspace_id: Required for s3_vectors (bucket scoping)
    """

    if backend == "pgvector":
        # EXISTING - uses PostgreSQL pgvector
        backend_impl = PgVectorBackend(**kwargs)

    elif backend == "s3_vectors":
        # NEW - uses AWS S3 Vectors
        if not workspace_id:
            raise ValueError("workspace_id required for s3_vectors backend")
        backend_impl = S3VectorsBackend(workspace_id=workspace_id, **kwargs)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    return EnhancedVectorStore(backend=backend_impl)


# orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py
"""S3 Vectors backend implementation."""

import boto3
from typing import List, Dict
from .base import VectorBackend

class S3VectorsBackend(VectorBackend):
    """AWS S3 Vectors backend for EnhancedVectorStore."""

    def __init__(self, workspace_id: str, region: str = "us-east-1"):
        self.workspace_id = workspace_id
        self.bucket_name = f"automatos-vectors-{workspace_id}"
        self.index_name = "documents-index"
        self.client = boto3.client('s3vectors', region_name=region)

        # Ensure bucket + index exist
        self._ensure_setup()

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5,
        filters: Dict = None
    ) -> List[Dict]:
        """Search vectors - implements VectorBackend interface."""

        results = self.client.query_vectors(
            Bucket=self.bucket_name,
            VectorIndex=self.index_name,
            QueryVector={'float32': query_embedding},
            MaxResults=limit,
            MinScore=min_score
        )

        # Normalize to EnhancedVectorStore format
        return self._normalize_results(results)

    def add_documents(
        self,
        documents: List[Dict],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Add documents - implements VectorBackend interface."""

        vector_objects = []
        for doc, embedding in zip(documents, embeddings):
            vector_objects.append({
                'Key': f"doc_{doc['id']}_chunk_{doc['chunk_index']}",
                'Vector': {'float32': embedding},
                'Metadata': doc.get('metadata', {})
            })

        self.client.put_vectors(
            Bucket=self.bucket_name,
            VectorIndex=self.index_name,
            VectorObjects=vector_objects
        )

        return [obj['Key'] for obj in vector_objects]
```

### 4. Use Existing EntityManager (NO CHANGES NEEDED)

**DON'T CREATE**: New entity management
**DO USE**: Existing `core/composio/entity_manager.py` as-is

```python
# In your cloud sync service
from core.composio.entity_manager import EntityManager

class CloudSyncService:
    def __init__(self, db: Session):
        self.entity_manager = EntityManager(db)  # EXISTING

    async def sync_workspace(self, workspace_id: UUID):
        # Get entity (REUSES existing logic)
        entity = self.entity_manager.get_or_create_entity(workspace_id)

        # Get connected apps (REUSES existing logic)
        connected_apps = self.entity_manager.get_connected_apps(workspace_id)

        # Entity manager ALREADY handles:
        # - Workspace isolation
        # - Connection tracking
        # - Status management
```

### 5. Use Existing RAGService (MINIMAL CHANGES)

**DON'T CREATE**: New RAG query service
**DO EXTEND**: Existing `modules/rag/service.py` to support S3 backend

```python
# orchestrator/modules/rag/service.py
# EXISTING FILE - MINIMAL CHANGE

class RAGService:
    def __init__(self, db: Session, vector_backend: str = "pgvector"):
        self.db = db

        # CHANGE: Use factory instead of direct instantiation
        # OLD: self.vector_store = EnhancedVectorStore(...)
        # NEW: Use backend factory
        from modules.search.vector_store import get_vector_store
        self.vector_store = get_vector_store(backend=vector_backend)

        # Everything else stays the same!
        # - retrieve()
        # - chunk_document()
        # - All optimization logic

    # All existing methods work unchanged!
    # The backend abstraction makes them work with S3 Vectors automatically.
```

---

## Summary: What You're Actually Building

### NEW Files (Minimal):
1. `/orchestrator/modules/rag/services/cloud_sync_service.py` - Orchestration only
2. `/orchestrator/modules/rag/services/cloud_file_downloader.py` - Thin wrapper around ToolExecutor
3. `/orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py` - S3 implementation
4. `/orchestrator/core/models/sync_state.py` - Sync tracking models
5. `/orchestrator/api/documents.py` - Add cloud endpoints to EXISTING file

### EXTENDED Files (Small changes):
1. `/orchestrator/modules/rag/ingestion/pipeline.py` - Add `ingest_from_cloud()` method
2. `/orchestrator/modules/rag/service.py` - Add backend parameter to `__init__`
3. `/orchestrator/modules/search/vector_store/__init__.py` - Add factory function
4. `/orchestrator/core/models/core.py` - Add cloud_documents table

### REUSED As-Is (NO changes):
1. ✅ `/orchestrator/core/composio/client.py` - Use execute_action()
2. ✅ `/orchestrator/core/composio/entity_manager.py` - Use get_or_create_entity()
3. ✅ `/orchestrator/core/composio/tool_executor.py` - Use execute()
4. ✅ `/orchestrator/modules/rag/chunking/semantic_chunker.py` - Perfect as-is
5. ✅ `/orchestrator/modules/rag/ingestion/manager.py` - Use DocumentManager
6. ✅ `/orchestrator/modules/search/optimization/context_optimizer.py` - Use as-is
7. ✅ `/orchestrator/core/llm/embedding_manager.py` - Use create_embedding_manager()

### Total New Code: ~800 lines (vs 3000+ in original PRD)
### Leverage Existing: ~6000 lines of battle-tested code

### Key Services (UPDATED - Extends Existing)

#### 1. CloudSyncService (Orchestrator)

```python
# orchestrator/modules/rag/services/cloud_sync_service.py
"""
Cloud Document Sync Service - Orchestrates cloud file syncing.
EXTENDS existing infrastructure - does NOT duplicate.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

# REUSE existing infrastructure
from core.composio.entity_manager import EntityManager
from core.composio.tool_executor import ComposioToolExecutor
from modules.rag.ingestion.pipeline import IngestionPipeline
from modules.rag.services.cloud_file_downloader import CloudFileDownloader
from modules.search.vector_store import get_vector_store  # Factory pattern
from core.models.core import Document
from core.models.sync_state import CloudSyncState, CloudSyncJob

class CloudSyncService:
    """
    Syncs documents from user's cloud storage to S3 Vectors.

    Key responsibilities:
    - List files from cloud storage via Composio
    - Download files temporarily
    - Chunk and embed using existing pipeline
    - Store vectors in S3 Vectors (NOT local storage)
    - Track metadata in PostgreSQL
    """

    def __init__(self, db: Session, workspace_id: str):
        self.db = db
        self.workspace_id = workspace_id
        self.composio = get_composio_client()
        self.s3vectors = S3VectorsClient(workspace_id)
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = create_embedding_manager()

    async def list_folders(
        self,
        connection_id: int,
        path: str = None
    ) -> List[Dict]:
        """List folders from cloud storage for tree navigation."""

        connection = self._get_connection(connection_id)
        entity_id = connection.composio_entity_id
        app_name = connection.app_name.upper()

        # Map to Composio actions
        if app_name == "GOOGLEDRIVE":
            action = "GOOGLEDRIVE_LIST_FILES"
            params = {
                "q": f"mimeType='application/vnd.google-apps.folder' and '{path or 'root'}' in parents",
                "fields": "files(id,name,mimeType,modifiedTime)"
            }
        elif app_name == "DROPBOX":
            action = "DROPBOX_LIST_FOLDER"
            params = {"path": path or ""}
        elif app_name == "ONEDRIVE":
            action = "ONEDRIVE_LIST_FILES"
            params = {"folder": path or "root", "filter": "folder"}
        else:
            raise ValueError(f"Unsupported app: {app_name}")

        # Execute via Composio
        result = self.composio.execute_action(
            action=action,
            params=params,
            entity_id=entity_id
        )

        # Normalize response
        folders = []
        for item in result.get("data", {}).get("files", []):
            folders.append({
                "name": item.get("name"),
                "path": item.get("id") if app_name == "GOOGLEDRIVE" else item.get("path"),
                "last_modified": item.get("modifiedTime"),
                "has_children": True  # Assume yes, lazy load on expand
            })

        return folders

    async def list_files(
        self,
        connection_id: int,
        path: str = None,
        file_types: List[str] = None,
        recursive: bool = False
    ) -> List[Dict]:
        """List files in a folder from cloud storage."""

        connection = self._get_connection(connection_id)
        entity_id = connection.composio_entity_id
        app_name = connection.app_name.upper()

        # Build query based on app
        if app_name == "GOOGLEDRIVE":
            action = "GOOGLEDRIVE_LIST_FILES"

            # Build MIME type filter
            mime_filter = ""
            if file_types:
                mime_map = {
                    "pdf": "application/pdf",
                    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "txt": "text/plain",
                    "md": "text/markdown"
                }
                mime_types = [mime_map.get(ft) for ft in file_types if ft in mime_map]
                if mime_types:
                    mime_filter = " and (" + " or ".join([f"mimeType='{mt}'" for mt in mime_types]) + ")"

            params = {
                "q": f"'{path or 'root'}' in parents{mime_filter}",
                "fields": "files(id,name,mimeType,size,modifiedTime)",
                "pageSize": 100
            }
        elif app_name == "DROPBOX":
            action = "DROPBOX_LIST_FOLDER"
            params = {"path": path or "", "recursive": recursive}
        elif app_name == "ONEDRIVE":
            action = "ONEDRIVE_LIST_FILES"
            params = {"folder": path or "root"}

        result = self.composio.execute_action(
            action=action,
            params=params,
            entity_id=entity_id
        )

        # Normalize and check sync status
        files = []
        for item in result.get("data", {}).get("files", []):
            external_file_id = item.get("id")

            # Check if already synced
            cloud_doc = self.db.query(CloudDocument).filter(
                CloudDocument.connection_id == connection_id,
                CloudDocument.external_file_id == external_file_id
            ).first()

            files.append({
                "external_file_id": external_file_id,
                "name": item.get("name"),
                "path": item.get("path", item.get("id")),
                "mime_type": item.get("mimeType"),
                "size": item.get("size"),
                "modified_at": item.get("modifiedTime"),
                "is_synced": cloud_doc is not None,
                "sync_status": cloud_doc.sync_status if cloud_doc else "pending",
                "chunk_count": cloud_doc.chunk_count if cloud_doc else 0
            })

        return files

    async def sync_folder(
        self,
        connection_id: int,
        root_folder_path: str,
        force: bool = False
    ) -> Dict:
        """
        Main sync method - downloads, chunks, embeds, stores in S3 Vectors.

        Returns sync job summary.
        """

        # Create sync job
        job = CloudSyncJob(
            workspace_id=self.workspace_id,
            connection_id=connection_id,
            status="running",
            started_at=datetime.utcnow()
        )
        self.db.add(job)
        self.db.commit()

        try:
            # List all files recursively
            files = await self.list_files(
                connection_id=connection_id,
                path=root_folder_path,
                recursive=True
            )

            synced = 0
            skipped = 0
            errored = 0
            total_chunks = 0
            errors = []

            for file_info in files:
                try:
                    # Skip if already synced and not forcing
                    if file_info["is_synced"] and not force:
                        # Check if modified
                        cloud_doc = self.db.query(CloudDocument).filter(
                            CloudDocument.connection_id == connection_id,
                            CloudDocument.external_file_id == file_info["external_file_id"]
                        ).first()

                        file_modified = datetime.fromisoformat(file_info["modified_at"].replace("Z", "+00:00"))
                        if cloud_doc.cloud_modified_at and file_modified <= cloud_doc.cloud_modified_at:
                            skipped += 1
                            continue

                    # Download and process
                    result = await self._download_and_process_file(
                        connection_id=connection_id,
                        external_file_id=file_info["external_file_id"],
                        file_name=file_info["name"],
                        file_path=file_info["path"],
                        mime_type=file_info["mime_type"],
                        file_size=file_info["size"],
                        cloud_modified_at=file_info["modified_at"]
                    )

                    synced += 1
                    total_chunks += result["chunk_count"]

                except Exception as e:
                    errored += 1
                    errors.append({
                        "file_name": file_info["name"],
                        "error": str(e)
                    })

            # Update job
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.files_synced = synced
            job.files_skipped = skipped
            job.files_errored = errored
            job.total_chunks_created = total_chunks
            self.db.commit()

            # Update connection stats
            connection = self._get_connection(connection_id)
            connection.total_documents_synced = synced + skipped
            connection.last_successful_sync = datetime.utcnow()
            self.db.commit()

            return {
                "job_id": job.id,
                "status": "completed",
                "files_synced": synced,
                "files_skipped": skipped,
                "files_errored": errored,
                "total_chunks_created": total_chunks,
                "errors": errors
            }

        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            self.db.commit()
            raise

    async def _download_and_process_file(
        self,
        connection_id: int,
        external_file_id: str,
        file_name: str,
        file_path: str,
        mime_type: str,
        file_size: int,
        cloud_modified_at: str
    ) -> Dict:
        """
        Download file from cloud, chunk, embed, store in S3 Vectors.
        CRITICAL: Original file is NOT stored locally - only vectors!
        """

        connection = self._get_connection(connection_id)
        app_name = connection.app_name.upper()
        entity_id = connection.composio_entity_id

        # 1. Download file via Composio
        if app_name == "GOOGLEDRIVE":
            download_action = "GOOGLEDRIVE_DOWNLOAD_FILE"
            params = {"fileId": external_file_id}
        elif app_name == "DROPBOX":
            download_action = "DROPBOX_DOWNLOAD_FILE"
            params = {"path": file_path}
        elif app_name == "ONEDRIVE":
            download_action = "ONEDRIVE_DOWNLOAD_FILE"
            params = {"item_id": external_file_id}

        file_content = self.composio.execute_action(
            action=download_action,
            params=params,
            entity_id=entity_id
        )

        # 2. Save to temp file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            tmp.write(file_content["data"]["content"])
            tmp_path = tmp.name

        try:
            # 3. Extract text (REUSE existing DocumentProcessor)
            file_type = self.doc_processor.detect_file_type(tmp_path)

            if file_type.value == "pdf":
                text = self.doc_processor.extract_text_from_pdf(tmp_path)
            elif file_type.value == "docx":
                text = self.doc_processor.extract_text_from_docx(tmp_path)
            elif file_type.value in ["text", "markdown"]:
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # 4. Chunk (REUSE existing SemanticChunker)
            from modules.rag.chunking.semantic_chunker import SemanticChunker, ChunkingStrategy

            chunker = SemanticChunker(
                strategy=ChunkingStrategy.ADAPTIVE,
                target_size=500,
                min_size=100,
                max_size=1500
            )
            chunks = chunker.chunk_text(text)

            # 5. Generate embeddings (REUSE existing EmbeddingManager)
            embeddings = []
            for chunk in chunks:
                embedding = await self.embedding_manager.generate_embedding(chunk.content)
                embeddings.append(embedding)

            # 6. Store in S3 Vectors (NEW - replaces pgvector)
            await self.s3vectors.put_vectors(
                external_file_id=external_file_id,
                file_name=file_name,
                file_path=file_path,
                app_name=app_name,
                chunks=chunks,
                embeddings=embeddings
            )

            # 7. Create/update cloud_documents metadata
            cloud_doc = self.db.query(CloudDocument).filter(
                CloudDocument.connection_id == connection_id,
                CloudDocument.external_file_id == external_file_id
            ).first()

            if not cloud_doc:
                cloud_doc = CloudDocument(
                    workspace_id=self.workspace_id,
                    connection_id=connection_id,
                    app_name=app_name,
                    external_file_id=external_file_id
                )
                self.db.add(cloud_doc)

            cloud_doc.file_name = file_name
            cloud_doc.file_path = file_path
            cloud_doc.mime_type = mime_type
            cloud_doc.file_size = file_size
            cloud_doc.s3_vector_bucket = self.s3vectors.bucket_name
            cloud_doc.s3_vector_index = self.s3vectors.index_name
            cloud_doc.chunk_count = len(chunks)
            cloud_doc.cloud_modified_at = datetime.fromisoformat(cloud_modified_at.replace("Z", "+00:00"))
            cloud_doc.last_synced_at = datetime.utcnow()
            cloud_doc.sync_status = "synced"
            cloud_doc.sync_error = None

            self.db.commit()

            return {
                "document_id": cloud_doc.id,
                "chunk_count": len(chunks),
                "status": "synced"
            }

        finally:
            # 8. CRITICAL: Delete temp file (we NEVER store original!)
            os.unlink(tmp_path)

    async def query_rag(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
        max_tokens: int = 2000,
        diversity_factor: float = 0.3
    ) -> Dict:
        """
        Query documents using RAG across all connected storage providers.
        REUSES existing context optimization logic.
        """

        # 1. Generate query embedding
        query_embedding = await self.embedding_manager.generate_embedding(query)

        # 2. Query S3 Vectors
        results = await self.s3vectors.query_vectors(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch for diversity filtering
            min_score=min_similarity
        )

        # 3. Apply existing context optimization (REUSE)
        from modules.search.optimization.context_optimizer import ContextOptimizer

        optimizer = ContextOptimizer()
        optimized_chunks = optimizer.optimize(
            chunks=results,
            max_tokens=max_tokens,
            diversity_factor=diversity_factor
        )

        # 4. Format context
        context_parts = []
        sources = []

        for chunk in optimized_chunks:
            context_parts.append(
                f"Source: {chunk['app_name']} - {chunk['file_path']} (Score: {chunk['score']:.2f})\n"
                f"{chunk['content']}\n"
            )

            sources.append({
                "document_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "app_name": chunk["app_name"],
                "file_path": chunk["file_path"],
                "similarity_score": chunk["score"]
            })

        return {
            "context": "\n".join(context_parts),
            "chunks": optimized_chunks,
            "sources": sources,
            "total_tokens": sum(chunk.get("tokens", 0) for chunk in optimized_chunks),
            "diversity_score": optimizer.calculate_diversity_score(optimized_chunks)
        }

    def _get_connection(self, connection_id: int) -> ComposioConnection:
        """Get and validate connection."""
        connection = self.db.query(ComposioConnection).get(connection_id)
        if not connection:
            raise ValueError(f"Connection {connection_id} not found")
        if connection.status != "active":
            raise ValueError(f"Connection {connection_id} is not active")
        return connection
```

#### S3VectorsClient

```python
# orchestrator/services/s3_vectors_client.py

import boto3
from typing import List, Dict, Any
from datetime import datetime

class S3VectorsClient:
    """
    Wrapper around AWS S3 Vectors API.
    Handles bucket/index creation and vector operations.
    """

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.bucket_name = f"automatos-vectors-{workspace_id}"
        self.index_name = "documents-index"
        self.client = boto3.client('s3vectors')

        # Ensure bucket and index exist
        self._ensure_setup()

    def _ensure_setup(self):
        """Create S3 vector bucket and index if they don't exist."""

        # Create bucket
        try:
            self.client.create_vector_bucket(Bucket=self.bucket_name)
        except Exception as e:
            if "BucketAlreadyExists" not in str(e) and "BucketAlreadyOwnedByYou" not in str(e):
                raise

        # Create index
        try:
            self.client.create_vector_index(
                Bucket=self.bucket_name,
                VectorIndex={
                    'IndexName': self.index_name,
                    'IndexDimension': 1024,  # BAAI/bge-large-en-v1.5
                    'DistanceMetric': 'COSINE'
                }
            )
        except Exception as e:
            if "IndexAlreadyExists" not in str(e):
                raise

    async def put_vectors(
        self,
        external_file_id: str,
        file_name: str,
        file_path: str,
        app_name: str,
        chunks: List[Any],
        embeddings: List[List[float]]
    ):
        """Insert vectors for a document."""

        vector_objects = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_objects.append({
                'Key': f'doc_{external_file_id}_chunk_{idx}',
                'Vector': {'float32': embedding},
                'Metadata': {
                    'external_file_id': external_file_id,
                    'chunk_index': str(idx),
                    'chunk_text': chunk.content[:500],  # First 500 chars
                    'app_name': app_name,
                    'workspace_id': self.workspace_id,
                    'file_name': file_name,
                    'file_path': file_path
                }
            })

        # Batch insert (max 1000 per request)
        batch_size = 1000
        for i in range(0, len(vector_objects), batch_size):
            batch = vector_objects[i:i + batch_size]
            self.client.put_vectors(
                Bucket=self.bucket_name,
                VectorIndex=self.index_name,
                VectorObjects=batch
            )

    async def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_score: float = 0.5
    ) -> List[Dict]:
        """Query vectors by similarity."""

        results = self.client.query_vectors(
            Bucket=self.bucket_name,
            VectorIndex=self.index_name,
            QueryVector={'float32': query_embedding},
            MaxResults=top_k,
            MinScore=min_score
        )

        # Format results
        formatted = []
        for match in results.get('Matches', []):
            formatted.append({
                'score': match['Score'],
                'content': match['Metadata']['chunk_text'],
                'external_file_id': match['Metadata']['external_file_id'],
                'app_name': match['Metadata']['app_name'],
                'file_name': match['Metadata']['file_name'],
                'file_path': match['Metadata']['file_path'],
                'chunk_index': int(match['Metadata']['chunk_index']),
                'document_id': None  # Will be populated from database lookup
            })

        return formatted

    async def delete_vectors(self, external_file_id: str):
        """Delete all vectors for a document."""

        # S3 Vectors doesn't have batch delete by prefix
        # Need to query first, then delete individually
        # For MVP, mark as deleted in DB and clean up via background job
        pass
```

---

## Frontend Implementation

### Component Structure

```
frontend/
├── components/
│   └── documents/
│       ├── cloud-storage/                    # NEW
│       │   ├── CloudStorageConnections.tsx   # Connect/disconnect apps
│       │   ├── FolderNavigator.tsx           # Tree view for folder selection
│       │   ├── CloudDocumentList.tsx         # List of synced docs
│       │   ├── SyncButton.tsx                # Manual "Sync Now" trigger
│       │   ├── SyncProgressModal.tsx         # Real-time sync progress
│       │   └── DocumentPreviewModal.tsx      # View/download from cloud
│       │
│       ├── document-management.tsx           # MODIFY: Add cloud storage tab
│       ├── document-upload.tsx               # KEEP: Backward compat (Phase 1)
│       └── semantic-search.tsx               # KEEP: Works with S3 Vectors
│
├── hooks/
│   ├── use-cloud-documents.ts                # NEW: Cloud document hooks
│   ├── use-composio-api.ts                   # EXISTING: Reuse
│   └── use-document-api.ts                   # EXISTING: Keep for upload
│
└── lib/
    └── api-client.ts                          # MODIFY: Add cloud endpoints
```

### Key Components

#### CloudStorageConnections.tsx

```typescript
// frontend/components/documents/cloud-storage/CloudStorageConnections.tsx

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

interface CloudConnection {
  id: number;
  app_name: string;
  status: string;
  connected_at: string;
  sync_enabled: boolean;
  total_documents_synced: number;
  last_successful_sync: string | null;
  root_folder_path: string | null;
}

const SUPPORTED_APPS = [
  { name: 'GOOGLEDRIVE', displayName: 'Google Drive', icon: '/icons/google-drive.svg' },
  { name: 'DROPBOX', displayName: 'Dropbox', icon: '/icons/dropbox.svg' },
  { name: 'ONEDRIVE', displayName: 'OneDrive', icon: '/icons/onedrive.svg' },
  { name: 'BOX', displayName: 'Box', icon: '/icons/box.svg' },
  { name: 'SHAREPOINT', displayName: 'SharePoint', icon: '/icons/sharepoint.svg' },
];

export function CloudStorageConnections() {
  const queryClient = useQueryClient();
  const [connectingApp, setConnectingApp] = useState<string | null>(null);

  // Fetch connections
  const { data: connections, isLoading } = useQuery({
    queryKey: ['cloud-documents', 'connections'],
    queryFn: async () => {
      const response = await apiClient.get('/api/cloud-documents/connections');
      return response.data.connections as CloudConnection[];
    },
  });

  // Connect mutation
  const connectMutation = useMutation({
    mutationFn: async (appName: string) => {
      const callbackUrl = `${window.location.origin}/documents?connected=${appName}`;
      const response = await apiClient.post(`/api/cloud-documents/connect/${appName}`, {
        callback_url: callbackUrl,
      });
      return response.data;
    },
    onSuccess: (data, appName) => {
      // Open OAuth popup
      const popup = window.open(
        data.redirect_url,
        'composio-oauth',
        'width=600,height=700,scrollbars=yes'
      );

      // Poll for popup close
      const checkClosed = setInterval(() => {
        if (popup?.closed) {
          clearInterval(checkClosed);
          setConnectingApp(null);
          queryClient.invalidateQueries({ queryKey: ['cloud-documents', 'connections'] });
        }
      }, 1000);
    },
  });

  // Disconnect mutation
  const disconnectMutation = useMutation({
    mutationFn: async ({ connectionId, deleteVectors }: { connectionId: number; deleteVectors: boolean }) => {
      const response = await apiClient.delete(
        `/api/cloud-documents/connections/${connectionId}?delete_vectors=${deleteVectors}`
      );
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cloud-documents', 'connections'] });
    },
  });

  const handleConnect = (appName: string) => {
    setConnectingApp(appName);
    connectMutation.mutate(appName);
  };

  const handleDisconnect = (connection: CloudConnection) => {
    const deleteVectors = confirm(
      `Do you want to delete all synced vectors for ${connection.app_name}?\n\n` +
      `Click OK to delete (removes ${connection.total_documents_synced} documents from RAG).\n` +
      `Click Cancel to keep vectors (you can still query them, but won't sync new files).`
    );

    disconnectMutation.mutate({
      connectionId: connection.id,
      deleteVectors,
    });
  };

  const getConnectionStatus = (connection: CloudConnection) => {
    if (connection.status === 'active' && connection.sync_enabled) {
      return <Badge variant="success">✓ Connected</Badge>;
    } else if (connection.status === 'active') {
      return <Badge variant="warning">⚠ Connected (Sync Disabled)</Badge>;
    } else {
      return <Badge variant="destructive">✗ Disconnected</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Cloud Storage</h2>
        <p className="text-muted-foreground">
          Connect your cloud storage providers to sync documents for RAG queries.
          We only store vector embeddings - your files stay in your cloud.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {SUPPORTED_APPS.map((app) => {
          const connection = connections?.find((c) => c.app_name === app.name);

          return (
            <Card key={app.name}>
              <CardHeader className="flex flex-row items-center gap-4">
                <img src={app.icon} alt={app.displayName} className="w-10 h-10" />
                <div className="flex-1">
                  <CardTitle>{app.displayName}</CardTitle>
                  {connection && getConnectionStatus(connection)}
                </div>
              </CardHeader>
              <CardContent>
                {connection ? (
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      Connected {new Date(connection.connected_at).toLocaleDateString()}
                    </p>
                    <p className="text-sm">
                      <strong>{connection.total_documents_synced}</strong> documents synced
                    </p>
                    {connection.last_successful_sync && (
                      <p className="text-sm text-muted-foreground">
                        Last sync: {new Date(connection.last_successful_sync).toLocaleString()}
                      </p>
                    )}
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleDisconnect(connection)}
                      disabled={disconnectMutation.isPending}
                    >
                      Disconnect
                    </Button>
                  </div>
                ) : (
                  <Button
                    onClick={() => handleConnect(app.name)}
                    disabled={connectingApp === app.name}
                  >
                    {connectingApp === app.name ? 'Connecting...' : 'Connect'}
                  </Button>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
```

#### FolderNavigator.tsx

```typescript
// frontend/components/documents/cloud-storage/FolderNavigator.tsx

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { Button } from '@/components/ui/button';
import { ChevronRight, ChevronDown, Folder, FileIcon } from 'lucide-react';

interface FolderItem {
  name: string;
  path: string;
  has_children: boolean;
  last_modified: string;
}

interface FolderNavigatorProps {
  connectionId: number;
  onFolderSelect: (path: string) => void;
  selectedPath: string | null;
}

export function FolderNavigator({ connectionId, onFolderSelect, selectedPath }: FolderNavigatorProps) {
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set(['/']));
  const queryClient = useQueryClient();

  // Fetch folders for a path
  const useFolders = (path: string) => {
    return useQuery({
      queryKey: ['cloud-documents', 'folders', connectionId, path],
      queryFn: async () => {
        const response = await apiClient.get(
          `/api/cloud-documents/connections/${connectionId}/folders`,
          { params: { path: path === '/' ? undefined : path } }
        );
        return response.data.folders as FolderItem[];
      },
      enabled: expandedPaths.has(path),
    });
  };

  const toggleExpand = (path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const FolderTree = ({ path, level = 0 }: { path: string; level?: number }) => {
    const { data: folders, isLoading } = useFolders(path);
    const isExpanded = expandedPaths.has(path);

    if (!isExpanded && path !== '/') return null;

    return (
      <div style={{ paddingLeft: level * 20 }}>
        {isLoading && <div className="text-sm text-muted-foreground">Loading...</div>}
        {folders?.map((folder) => (
          <div key={folder.path}>
            <div
              className={`flex items-center gap-2 py-1 px-2 hover:bg-accent rounded cursor-pointer ${
                selectedPath === folder.path ? 'bg-accent' : ''
              }`}
            >
              <button
                onClick={() => toggleExpand(folder.path)}
                className="p-0 hover:bg-transparent"
              >
                {folder.has_children ? (
                  expandedPaths.has(folder.path) ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )
                ) : (
                  <div className="w-4" />
                )}
              </button>
              <Folder className="w-4 h-4 text-yellow-500" />
              <span
                className="flex-1 text-sm"
                onClick={() => onFolderSelect(folder.path)}
              >
                {folder.name}
              </span>
            </div>
            {folder.has_children && <FolderTree path={folder.path} level={level + 1} />}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="border rounded-lg p-4 max-h-96 overflow-y-auto">
      <div className="mb-2">
        <h3 className="font-semibold">Select Folder to Sync</h3>
        <p className="text-sm text-muted-foreground">
          We recommend creating a dedicated "Automatos" folder in your cloud storage.
        </p>
      </div>
      <FolderTree path="/" />
    </div>
  );
}
```

---

## Implementation Roadmap

### Phase 1: MVP - Manual Sync (Weeks 1-3)

**Week 1: Backend Foundation**
- [ ] Create database migration script (cloud_documents, cloud_sync_config, cloud_sync_jobs tables)
- [ ] Run migration: `alembic upgrade head`
- [ ] Delete existing 60 test documents: `DELETE FROM document_chunks; DELETE FROM documents;`
- [ ] Implement S3VectorsClient service
  - [ ] Bucket/index creation
  - [ ] put_vectors() method
  - [ ] query_vectors() method
- [ ] Test S3 Vectors integration with sample data
- [ ] Create CloudDocument, CloudSyncConfig SQLAlchemy models

**Week 2: Sync Service**
- [ ] Implement CloudDocumentSyncService
  - [ ] list_folders() - Composio folder listing
  - [ ] list_files() - Composio file listing with sync status
  - [ ] sync_folder() - Main sync orchestration
  - [ ] _download_and_process_file() - Download, chunk, embed, store
- [ ] Create API endpoints in /api/cloud_documents.py
  - [ ] GET /connections
  - [ ] POST /connect/{app_name}
  - [ ] DELETE /connections/{connection_id}
  - [ ] GET /connections/{id}/folders
  - [ ] GET /connections/{id}/files
  - [ ] POST /connections/{id}/select-folder
  - [ ] POST /connections/{id}/sync
  - [ ] GET /sync-jobs/{job_id}
- [ ] Test end-to-end sync with Google Drive (10 sample PDFs)
- [ ] Verify vectors in S3 Vectors bucket via AWS Console

**Week 3: Frontend + RAG**
- [ ] Create CloudStorageConnections.tsx component
  - [ ] Display supported apps grid
  - [ ] Connect/disconnect buttons
  - [ ] OAuth popup flow
- [ ] Create FolderNavigator.tsx component
  - [ ] Tree view with lazy loading
  - [ ] Folder selection
- [ ] Create SyncButton.tsx + SyncProgressModal.tsx
  - [ ] Manual "Sync Now" trigger
  - [ ] Real-time progress display
- [ ] Implement query_rag() in CloudDocumentSyncService
  - [ ] Reuse EmbeddingManager for query embedding
  - [ ] Call S3 Vectors query API
  - [ ] Reuse ContextOptimizer for chunk selection
- [ ] Create POST /api/cloud-documents/rag/query endpoint
- [ ] Test RAG queries across Google Drive + Dropbox
- [ ] Add cloud storage tab to DocumentManagement.tsx

### Phase 2: Auto-Sync via Webhooks (Weeks 4-5)

**Week 4: Webhook Integration**
- [ ] Review Composio webhook documentation
- [ ] Implement webhook signature verification (HMAC-SHA256)
- [ ] Create POST /api/cloud-documents/webhook endpoint
- [ ] Handle trigger types:
  - [ ] GOOGLEDRIVE_FILE_CREATED
  - [ ] GOOGLEDRIVE_FILE_UPDATED
  - [ ] GOOGLEDRIVE_FILE_DELETED
  - [ ] DROPBOX_FILE_ADDED, DROPBOX_FILE_MODIFIED, DROPBOX_FILE_DELETED
  - [ ] ONEDRIVE_FILE_CREATED, ONEDRIVE_FILE_UPDATED
- [ ] Subscribe to triggers when user connects app
- [ ] Unsubscribe when user disconnects app
- [ ] Test webhook flow with ngrok (local dev)
- [ ] Deploy webhook endpoint to production

**Week 5: Background Sync Jobs**
- [ ] Set up Celery or RQ for background tasks
- [ ] Create cloud_sync_task.py with @task decorator
- [ ] Implement periodic sync (every 30 mins) as fallback
  - [ ] Cron job or Celery beat schedule
  - [ ] Query cloud_sync_config for enabled connections
  - [ ] Trigger sync_folder() for each
- [ ] Add sync status dashboard to frontend
  - [ ] Last sync timestamp
  - [ ] Next scheduled sync
  - [ ] Sync history (last 10 jobs)
- [ ] Error notification system
  - [ ] Email alerts for failed syncs
  - [ ] In-app notifications
- [ ] Test auto-sync: upload file to Google Drive, verify appears in Automatos within 30 seconds

### Phase 3: Polish & Scale (Week 6)

**Week 6: Production Readiness**
- [ ] Add document preview modal
  - [ ] Download file from cloud via Composio
  - [ ] Display PDF/text/image preview
  - [ ] Cache for 1 hour
- [ ] Implement document download endpoint
- [ ] Add file type filters to folder navigator
- [ ] Implement exclude patterns (e.g., skip .tmp files)
- [ ] Performance optimization:
  - [ ] Batch embedding generation (20 chunks at a time)
  - [ ] Parallel file processing (max 5 concurrent)
- [ ] Error handling improvements:
  - [ ] Retry logic with exponential backoff
  - [ ] Token expiration handling
  - [ ] File size validation
- [ ] Monitoring & logging:
  - [ ] CloudWatch metrics for S3 Vectors usage
  - [ ] Sync job success/failure rates
  - [ ] Average sync duration
- [ ] Documentation:
  - [ ] User guide: "How to connect Google Drive"
  - [ ] Admin guide: S3 Vectors bucket management
  - [ ] API documentation (Swagger/OpenAPI)
- [ ] Load testing:
  - [ ] 100 concurrent sync jobs
  - [ ] 1000 documents per workspace
  - [ ] RAG query latency under load

---

## Migration Strategy

### Step 1: Backup Existing Data (Day 1)

```bash
# Backup PostgreSQL
pg_dump automatos_db > backup_$(date +%Y%m%d).sql

# Backup documents table
psql -d automatos_db -c "COPY (SELECT * FROM documents) TO '/tmp/documents_backup.csv' CSV HEADER"
psql -d automatos_db -c "COPY (SELECT * FROM document_chunks) TO '/tmp/chunks_backup.csv' CSV HEADER"
```

### Step 2: Run Database Migration (Day 1)

```bash
# Generate migration
alembic revision --autogenerate -m "Add cloud document sync with S3 Vectors"

# Review migration file
cat alembic/versions/20260131_add_cloud_document_sync.py

# Run migration
alembic upgrade head

# Verify tables created
psql -d automatos_db -c "\dt cloud_*"
```

### Step 3: Delete Test Documents (Day 1)

```sql
-- Count existing documents (should be ~60)
SELECT COUNT(*) FROM documents;

-- Delete all (only test data, safe to remove)
DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents);
DELETE FROM documents;

-- Verify deletion
SELECT COUNT(*) FROM documents;  -- Should be 0
SELECT COUNT(*) FROM document_chunks;  -- Should be 0
```

### Step 4: Deploy Backend Changes (Week 1)

```bash
# Deploy to staging
git checkout -b feature/cloud-document-sync
git add orchestrator/services/cloud_document_sync.py
git add orchestrator/services/s3_vectors_client.py
git add orchestrator/api/cloud_documents.py
git commit -m "Add cloud document sync with S3 Vectors"
git push origin feature/cloud-document-sync

# Run tests
pytest orchestrator/services/tests/test_cloud_document_sync.py

# Deploy to production (via CI/CD)
```

### Step 5: Deploy Frontend Changes (Week 3)

```bash
# Build frontend
cd frontend
npm run build

# Deploy to Vercel/production
vercel --prod
```

### Step 6: User Migration Plan (Post-Launch)

**For Existing Users (if any beyond test):**
1. Send email: "We've upgraded document management - connect your cloud storage"
2. Keep old upload flow available for 30 days (parallel mode)
3. Provide migration tool: "Upload your existing documents to Google Drive, then sync"
4. After 30 days, remove upload UI, redirect to cloud storage

**For New Users:**
- Onboarding flow shows cloud storage connection first
- Guided setup: "Create an 'Automatos' folder in Google Drive"
- Sample data: Pre-populate with demo documents for testing

---

## Success Metrics

### Quantitative Metrics

| Metric | Current (Upload-Based) | Target (Cloud Sync) | Measurement Method |
|--------|------------------------|---------------------|-------------------|
| **Storage Cost** | $730/month (1000 users, 10GB each) | $56/month (90%+ reduction) | AWS billing dashboard |
| **Documents Synced** | 0 (manual upload only) | 500+ per workspace | Database count |
| **RAG Query Latency** | <200ms (pgvector) | <500ms (S3 Vectors cold), <100ms (warm) | Application metrics |
| **Sync Success Rate** | N/A | >95% | cloud_sync_jobs.status = 'completed' / total |
| **User Adoption** | N/A (no cloud sync) | 70% of active users connect at least 1 app | composio_connections count |
| **OAuth Success Rate** | N/A | >99% | composio_connections.status = 'active' / total |
| **Vector Storage Size** | ~10GB per 1000 docs | ~100MB per 1000 docs (100x smaller) | S3 Vectors bucket size |
| **Chunk Quality (Similarity)** | 0.75 avg similarity | 0.75+ avg similarity (maintain quality) | RAG query results |

### Qualitative Metrics

- [ ] Users can connect Google Drive in <60 seconds
- [ ] Users understand that "we only store vectors, not files" (onboarding messaging)
- [ ] Sync errors are clear and actionable (e.g., "File deleted from Google Drive")
- [ ] RAG queries return relevant results from cloud documents
- [ ] Support tickets related to document storage decrease by 50%
- [ ] Users report satisfaction with "always in sync" experience (webhooks)

---

## Dependencies

### Internal Dependencies
- **PRD-36: Composio Integration** (COMPLETE) - OAuth and app connections
- **PRD-11: Document Management** (COMPLETE) - Reuse DocumentProcessor, SemanticChunker
- Existing RAGService and ContextOptimizer

### External Dependencies
- **AWS S3 Vectors** - Service availability (GA as of Jan 2026)
- **Composio Platform** - API stability for 500+ apps
- **Cloud Provider APIs** - Google Drive, Dropbox, OneDrive uptime

### Technical Dependencies
- boto3 >= 1.34.0 (S3 Vectors support)
- composio-python >= 0.5.0
- PostgreSQL >= 14 (for new tables)
- React >= 18, TypeScript >= 5

---

## Testing Strategy

### Unit Tests

```python
# orchestrator/services/tests/test_cloud_document_sync.py

import pytest
from unittest.mock import Mock, patch
from services.cloud_document_sync import CloudDocumentSyncService

@pytest.fixture
def sync_service(db_session, workspace_id):
    return CloudDocumentSyncService(db_session, workspace_id)

def test_list_folders_google_drive(sync_service, mock_composio):
    """Test folder listing from Google Drive"""
    mock_composio.execute_action.return_value = {
        "data": {
            "files": [
                {"id": "abc123", "name": "Automatos", "mimeType": "application/vnd.google-apps.folder"}
            ]
        }
    }

    folders = await sync_service.list_folders(connection_id=1, path=None)

    assert len(folders) == 1
    assert folders[0]["name"] == "Automatos"
    mock_composio.execute_action.assert_called_once_with(
        action="GOOGLEDRIVE_LIST_FILES",
        params={"q": "mimeType='application/vnd.google-apps.folder' and 'root' in parents"},
        entity_id="workspace-uuid"
    )

def test_sync_folder_creates_cloud_documents(sync_service, mock_composio, mock_s3vectors):
    """Test that sync creates cloud_documents records"""
    mock_composio.execute_action.side_effect = [
        # List files
        {"data": {"files": [{"id": "file1", "name": "Doc.pdf", "size": 10000}]}},
        # Download file
        {"data": {"content": b"PDF content..."}}
    ]

    result = await sync_service.sync_folder(connection_id=1, root_folder_path="/Automatos")

    assert result["files_synced"] == 1
    assert result["total_chunks_created"] > 0

    # Verify cloud_documents record created
    cloud_doc = db_session.query(CloudDocument).filter(
        CloudDocument.external_file_id == "file1"
    ).first()
    assert cloud_doc is not None
    assert cloud_doc.sync_status == "synced"

def test_query_rag_returns_formatted_context(sync_service, mock_s3vectors, mock_embedding_manager):
    """Test RAG query formatting"""
    mock_s3vectors.query_vectors.return_value = [
        {
            "score": 0.87,
            "content": "Q4 revenue projections...",
            "file_name": "Planning.docx",
            "app_name": "GOOGLEDRIVE"
        }
    ]

    result = await sync_service.query_rag(query="Q4 revenue", top_k=10)

    assert "Q4 revenue projections" in result["context"]
    assert "GOOGLEDRIVE" in result["context"]
    assert len(result["chunks"]) > 0
    assert result["chunks"][0]["score"] == 0.87
```

### Integration Tests

```python
# orchestrator/tests/integration/test_cloud_sync_e2e.py

@pytest.mark.integration
def test_end_to_end_google_drive_sync(test_client, test_db, mock_google_drive_api):
    """Test full sync flow: connect → list → sync → query"""

    # 1. Connect Google Drive
    response = test_client.post("/api/cloud-documents/connect/GOOGLEDRIVE", json={
        "callback_url": "http://localhost:3000/documents"
    })
    assert response.status_code == 200
    redirect_url = response.json()["redirect_url"]
    assert "auth.composio.dev" in redirect_url

    # Simulate OAuth callback (mocked)
    test_client.post("/api/composio/connect/GOOGLEDRIVE/callback", json={
        "connection_id": "conn123",
        "status": "active"
    })

    # 2. List folders
    response = test_client.get("/api/cloud-documents/connections/1/folders")
    assert response.status_code == 200
    folders = response.json()["folders"]
    assert any(f["name"] == "Automatos" for f in folders)

    # 3. Select folder
    response = test_client.post("/api/cloud-documents/connections/1/select-folder", json={
        "root_folder_path": "/Automatos"
    })
    assert response.status_code == 200

    # 4. Trigger sync
    response = test_client.post("/api/cloud-documents/connections/1/sync")
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    # 5. Wait for sync completion (poll)
    import time
    for _ in range(30):  # Max 30 seconds
        response = test_client.get(f"/api/cloud-documents/sync-jobs/{job_id}")
        if response.json()["status"] == "completed":
            break
        time.sleep(1)

    assert response.json()["status"] == "completed"
    assert response.json()["files_synced"] > 0

    # 6. Query RAG
    response = test_client.post("/api/cloud-documents/rag/query", json={
        "query": "Find Q4 revenue",
        "top_k": 10
    })
    assert response.status_code == 200
    assert len(response.json()["chunks"]) > 0
    assert "Q4" in response.json()["context"]
```

### Frontend Tests

```typescript
// frontend/components/documents/cloud-storage/__tests__/CloudStorageConnections.test.tsx

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CloudStorageConnections } from '../CloudStorageConnections';

test('displays supported cloud storage apps', () => {
  render(<CloudStorageConnections />);

  expect(screen.getByText('Google Drive')).toBeInTheDocument();
  expect(screen.getByText('Dropbox')).toBeInTheDocument();
  expect(screen.getByText('OneDrive')).toBeInTheDocument();
  expect(screen.getByText('Box')).toBeInTheDocument();
});

test('initiates OAuth flow when connect button clicked', async () => {
  const mockOpen = jest.spyOn(window, 'open').mockReturnValue({} as Window);

  render(<CloudStorageConnections />);

  const connectButton = screen.getByRole('button', { name: /connect/i });
  await userEvent.click(connectButton);

  await waitFor(() => {
    expect(mockOpen).toHaveBeenCalledWith(
      expect.stringContaining('auth.composio.dev'),
      'composio-oauth',
      expect.any(String)
    );
  });
});

test('shows connected status for active connections', async () => {
  // Mock API response with active connection
  fetchMock.mockResponseOnce(JSON.stringify({
    connections: [
      { id: 1, app_name: 'GOOGLEDRIVE', status: 'active', total_documents_synced: 47 }
    ]
  }));

  render(<CloudStorageConnections />);

  await waitFor(() => {
    expect(screen.getByText('✓ Connected')).toBeInTheDocument();
    expect(screen.getByText('47 documents synced')).toBeInTheDocument();
  });
});
```

---

## Timeline

| Phase | Duration | Effort | Start Date | End Date |
|-------|----------|--------|------------|----------|
| **Phase 1: MVP (Manual Sync)** | 3 weeks | 120 hours | Week 1 | Week 3 |
| - Backend foundation | 1 week | 40 hours | Week 1 | Week 1 |
| - Sync service + API | 1 week | 40 hours | Week 2 | Week 2 |
| - Frontend + RAG | 1 week | 40 hours | Week 3 | Week 3 |
| **Phase 2: Auto-Sync (Webhooks)** | 2 weeks | 80 hours | Week 4 | Week 5 |
| - Webhook integration | 1 week | 40 hours | Week 4 | Week 4 |
| - Background jobs | 1 week | 40 hours | Week 5 | Week 5 |
| **Phase 3: Polish & Scale** | 1 week | 40 hours | Week 6 | Week 6 |
| **Total** | **6 weeks** | **240 hours** | | |

**Milestones:**
- ✅ Week 1: Database migration complete, S3 Vectors integration working
- ✅ Week 2: Manual sync working end-to-end for Google Drive
- ✅ Week 3: Frontend complete, RAG queries working across cloud storage
- ✅ Week 4: Webhooks live, real-time sync operational
- ✅ Week 5: Background jobs running, periodic sync as fallback
- ✅ Week 6: Production-ready, monitoring dashboard live

---

## Appendix

### Environment Variables

```bash
# AWS S3 Vectors
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Composio (already configured)
COMPOSIO_API_KEY=your_composio_api_key
COMPOSIO_WEBHOOK_SECRET=webhook_signing_secret

# Backend URL for webhooks
BACKEND_URL=https://api.automatos.ai

# Frontend URL for OAuth callbacks
FRONTEND_URL=https://app.automatos.ai
```

### Composio Actions Reference

**Google Drive:**
- `GOOGLEDRIVE_LIST_FILES` - List files in a folder
- `GOOGLEDRIVE_DOWNLOAD_FILE` - Download file content
- `GOOGLEDRIVE_GET_FILE_METADATA` - Get file metadata
- `GOOGLEDRIVE_SEARCH_FILES` - Search files by query

**Dropbox:**
- `DROPBOX_LIST_FOLDER` - List files in a folder
- `DROPBOX_DOWNLOAD_FILE` - Download file content
- `DROPBOX_GET_FILE_METADATA` - Get file metadata

**OneDrive:**
- `ONEDRIVE_LIST_FILES` - List files
- `ONEDRIVE_DOWNLOAD_FILE` - Download file
- `ONEDRIVE_GET_FILE_METADATA` - Get metadata

**Webhooks/Triggers:**
- `GOOGLEDRIVE_FILE_CREATED`
- `GOOGLEDRIVE_FILE_UPDATED`
- `GOOGLEDRIVE_FILE_DELETED`
- `DROPBOX_FILE_ADDED`
- `DROPBOX_FILE_MODIFIED`
- `DROPBOX_FILE_DELETED`
- `ONEDRIVE_FILE_CREATED`
- `ONEDRIVE_FILE_UPDATED`

### AWS S3 Vectors API Reference

```python
import boto3

client = boto3.client('s3vectors', region_name='us-east-1')

# Create bucket
client.create_vector_bucket(Bucket='automatos-vectors-{workspace_id}')

# Create index
client.create_vector_index(
    Bucket='automatos-vectors-{workspace_id}',
    VectorIndex={
        'IndexName': 'documents-index',
        'IndexDimension': 1024,
        'DistanceMetric': 'COSINE'  # or EUCLIDEAN, DOT_PRODUCT
    }
)

# Insert vectors
client.put_vectors(
    Bucket='automatos-vectors-{workspace_id}',
    VectorIndex='documents-index',
    VectorObjects=[
        {
            'Key': 'doc_abc123_chunk_0',
            'Vector': {'float32': [0.1, 0.2, ...]},  # 1024 dimensions
            'Metadata': {'file_name': 'Doc.pdf', 'chunk_index': '0'}
        }
    ]
)

# Query vectors
response = client.query_vectors(
    Bucket='automatos-vectors-{workspace_id}',
    VectorIndex='documents-index',
    QueryVector={'float32': [0.1, 0.2, ...]},
    MaxResults=10,
    MinScore=0.5
)

# Response format
{
    'Matches': [
        {
            'Key': 'doc_abc123_chunk_0',
            'Score': 0.87,
            'Metadata': {'file_name': 'Doc.pdf', 'chunk_index': '0'}
        }
    ]
}
```

### Cost Calculator

**Assumptions:**
- 1000 active workspaces
- Average 10GB documents per workspace (100 documents @ 100MB each)
- Embedding dimension: 1024 (BAAI/bge-large-en-v1.5)
- Average chunks per document: 20
- Vector size: 1024 floats × 4 bytes = 4KB per vector

**Storage:**
- Total documents: 1000 workspaces × 100 docs = 100,000 docs
- Total chunks: 100,000 docs × 20 chunks = 2,000,000 chunks
- Vector storage: 2M chunks × 4KB = 8GB
- S3 Vectors cost: 8GB × $0.06/GB = **$0.48/month**
- Metadata storage (PostgreSQL): ~100MB = **$5/month**

**Operations:**
- Ingestion (one-time): 8GB × $0.20/GB = $1.60
- Queries: 1M queries/month × $0.0025 = **$2.50/month**

**Total: ~$8/month for 1000 workspaces** (vs $730+ with local storage!)

---

## Glossary

- **S3 Vectors**: AWS service for storing and querying vector embeddings (GA Jan 2026)
- **Composio**: Integration platform managing OAuth for 500+ apps (Google Drive, Dropbox, etc.)
- **RAG**: Retrieval-Augmented Generation - using document chunks to augment LLM prompts
- **Semantic Chunking**: Intelligent text splitting based on semantic meaning (vs fixed-size)
- **Vector Embedding**: Numerical representation of text (1024-dimensional array)
- **Cosine Similarity**: Distance metric for comparing vector embeddings (0-1 scale)
- **Entity**: Composio's workspace-scoped authentication container
- **External File ID**: Cloud provider's unique identifier for a file (Google Drive file ID, Dropbox path, etc.)
- **Sync Status**: pending → syncing → synced → error
- **Knapsack Algorithm**: Dynamic programming optimization for selecting chunks within token budget
- **MMR**: Maximal Marginal Relevance - diversity scoring algorithm
- **Context Optimizer**: System for selecting optimal chunks for LLM context window

---

**END OF PRD**
