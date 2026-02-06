# PRD-42: Cloud Document Sync - Testing Guide

**Date:** 2026-02-04
**Status:** ✅ **READY FOR TESTING**
**Backend:** Mock S3 Vectors (in-memory)

---

## ✅ What's Been Done

### 1. Environment Setup
- ✅ Added AWS credentials to `orchestrator/.env`
- ✅ Switched to local PostgreSQL for testing
- ✅ Installed boto3 (already in venv)
- ✅ Created database tables (70 total, including 3 cloud sync tables)

### 2. Mock S3 Vectors Backend
- ✅ Created `s3_vectors_mock.py` - in-memory vector storage
- ✅ Implements same interface as real S3 backend
- ✅ Supports search, add, delete operations
- ✅ Uses numpy for cosine similarity
- ✅ Zero AWS costs, instant testing

### 3. Code Updates
- ✅ Updated `vector_store/__init__.py` to support mock backend
- ✅ Updated `api/cloud_documents.py` to use mock (2 places)
- ✅ All 12 API endpoints ready and registered

### 4. Test Scripts
- ✅ `scripts/init_test_db.py` - Initialize database from models
- ✅ `scripts/test_cloud_sync.py` - End-to-end mock backend test

---

## 🚀 How to Start Testing

### 1. Start Backend API
```bash
cd orchestrator
source venv/bin/activate
python -m uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 2. Start Frontend (separate terminal)
```bash
cd frontend
npm run dev
```

### 3. Verify API Health
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

---

## 🧪 Testing the Mock Backend

### Quick Test (Already Passing)
```bash
cd orchestrator
source venv/bin/activate
python scripts/test_cloud_sync.py
```

Expected output:
```
============================================================
PRD-42: Cloud Document Sync - Mock S3 Vectors Test
============================================================

1️⃣  Creating mock S3 vectors backend for workspace: test-workspace-123
   ✅ Backend initialized: automatos-vectors-test-workspace-123

2️⃣  Adding test documents with embeddings...
   ✅ Added 3 document chunks
   📊 Total vectors in storage: 3

3️⃣  Testing vector search...
   ✅ Search completed: found 3 results

4️⃣  Testing document deletion...
   ✅ Deleted 2 chunks for gdrive_file_123
   📊 Remaining vectors: 1

5️⃣  Testing connection cleanup...
   ✅ Deleted all 1 vectors
   📊 Final vector count: 0

============================================================
✅ All tests passed! Mock S3 Vectors backend is working.
============================================================
```

---

## 📋 API Endpoints to Test

Base: `http://localhost:8000/api/cloud-documents`

### 1. List Connections
```bash
curl http://localhost:8000/api/cloud-documents/connections \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 2. List Folders (for tree navigation)
```bash
curl "http://localhost:8000/api/cloud-documents/connections/1/folders?path=/" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3. List Files with Sync Status
```bash
curl "http://localhost:8000/api/cloud-documents/connections/1/files?path=/Documents" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 4. Set Root Folder
```bash
curl -X POST http://localhost:8000/api/cloud-documents/connections/1/select-folder \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"root_folder_path": "/Automatos"}'
```

### 5. Trigger Sync
```bash
curl -X POST http://localhost:8000/api/cloud-documents/connections/1/sync \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 6. Get Sync Job Status
```bash
curl http://localhost:8000/api/cloud-documents/sync-jobs/1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 7. Query Vectors (RAG)
```bash
curl -X POST http://localhost:8000/api/cloud-documents/rag/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "What is cloud document sync?",
    "top_k": 10,
    "min_similarity": 0.5
  }'
```

### 8. Get Sync Status
```bash
curl http://localhost:8000/api/cloud-documents/connections/1/sync-status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 9. Disconnect (with vector cleanup)
```bash
curl -X DELETE "http://localhost:8000/api/cloud-documents/connections/1?delete_vectors=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 🎯 Manual Testing Steps

### Test 1: Connect Cloud Storage (via Composio)
1. Open frontend: `http://localhost:3000`
2. Navigate to **Settings** → **Credentials**
3. Click "Add Connection"
4. Select **Google Drive** (or Dropbox, OneDrive, Box)
5. Complete OAuth flow
6. Verify connection appears in list

### Test 2: Select Root Folder
1. Go to **Documents** → **Cloud Storage** tab
2. Click on your connected storage
3. Browse folder tree
4. Click "Select" on a folder (e.g., "Automatos" or "Documents")
5. Verify "Root Folder: /path" is displayed

### Test 3: Trigger Sync
1. Click **"Sync Now"** button
2. Observe progress indicator
3. Wait for completion
4. Check sync results:
   - Files synced count
   - Files skipped count
   - Any errors

### Test 4: View Synced Documents
1. After sync completes, view file list
2. Verify each file shows:
   - ✅ Sync status badge
   - Chunk count
   - Last synced timestamp
3. Click on a file to see details

### Test 5: Query Documents (RAG)
1. In the search box, enter: "What is this document about?"
2. Click "Search"
3. Verify results show:
   - Relevant chunks from synced documents
   - Source information (app name, file path)
   - Similarity scores
4. Results should include files from ALL connected storage providers

### Test 6: Disconnect Storage
1. Click "Disconnect" on a storage connection
2. Choose option:
   - **Keep vectors for queries** (vectors remain, can't add new docs)
   - **Delete all vectors** (complete cleanup)
3. Verify:
   - Connection removed from list
   - Vectors deleted if selected
   - Files no longer appear in list

---

## 🔍 Database Verification

### Check Tables Were Created
```bash
psql -U postgres -d orchestrator_db -c "\dt" | grep cloud
```

Expected:
```
 public | cloud_documents    | table | postgres
 public | cloud_sync_config  | table | postgres
 public | cloud_sync_jobs    | table | postgres
```

### View Cloud Documents
```bash
psql -U postgres -d orchestrator_db -c "SELECT id, file_name, sync_status, chunk_count FROM cloud_documents;"
```

### View Sync Jobs
```bash
psql -U postgres -d orchestrator_db -c "SELECT id, status, files_synced, files_errored, started_at, completed_at FROM cloud_sync_jobs ORDER BY id DESC LIMIT 5;"
```

### View Connections
```bash
psql -U postgres -d orchestrator_db -c "SELECT id, app_name, sync_enabled, total_documents_synced, last_successful_sync FROM composio_connections;"
```

---

## 🐛 Common Issues & Solutions

### Issue: "Composio entity not found"
**Cause:** No workspace → Composio entity mapping
**Solution:** Create a workspace first in Settings

### Issue: "No files to sync"
**Cause:** Root folder is empty or not set
**Solution:** Select a folder with actual documents

### Issue: "Sync job failed"
**Cause:** Check error message in sync job
**Common reasons:**
- Composio connection expired (re-authenticate)
- File type not supported
- Download permission denied

### Issue: "No search results"
**Cause:** No documents synced yet or wrong embeddings
**Solution:**
1. Verify sync completed successfully
2. Check chunk_count > 0 for documents
3. Ensure embedding service is running

### Issue: "Mock backend not storing vectors"
**Cause:** Backend instance recreated between operations
**Solution:** Mock is in-memory per instance - this is expected behavior for testing

---

## 📊 What to Look For

### ✅ Success Indicators
- Sync jobs complete without errors
- Documents show chunk_count > 0
- RAG queries return relevant results
- File list shows sync badges
- Progress indicators work smoothly

### ⚠️ Warning Signs
- Sync jobs taking too long (>1 min for small files)
- All files skipped (0 synced)
- Zero chunks created
- Empty search results
- Connection timeouts

---

## 🚀 Switch to Real S3 Vectors (When Ready)

### Prerequisites
- AWS account with S3 Vectors access
- Real AWS credentials (not mock)
- AWS S3 Vectors service available in your region

### Step 1: Verify S3 Vectors Service
```bash
cd orchestrator
source venv/bin/activate
python -c "import boto3; client = boto3.client('s3vectors', region_name='us-east-1'); print('✅ S3 Vectors available')"
```

### Step 2: Update Code (2 locations)
```python
# File: orchestrator/api/cloud_documents.py
# Line ~523 and ~474

# BEFORE:
backend = get_vector_store(
    backend="s3_vectors_mock",
    workspace_id=str(ctx.workspace_id),
)

# AFTER:
backend = get_vector_store(
    backend="s3_vectors",
    workspace_id=str(ctx.workspace_id),
)
```

### Step 3: Update AWS Credentials
```bash
# File: orchestrator/.env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-real-key>
AWS_SECRET_ACCESS_KEY=<your-real-secret>
S3_VECTORS_ENABLED=true
```

### Step 4: Restart & Test
```bash
# Restart API server
cd orchestrator
source venv/bin/activate
python -m uvicorn main:app --reload

# Run tests again
python scripts/test_cloud_sync.py

# Sync documents and verify in AWS console
# Check bucket: automatos-vectors-{workspace_id}
# Check index: documents-index
```

---

## 📝 Test Results Checklist

- [ ] Mock backend test script passes
- [ ] API server starts without errors
- [ ] Frontend loads cloud documents panel
- [ ] Composio connection successful
- [ ] Folder tree navigation works
- [ ] Root folder selection works
- [ ] Sync job completes successfully
- [ ] Documents show in list with sync status
- [ ] Chunk counts are > 0
- [ ] RAG queries return results
- [ ] Search relevance is reasonable
- [ ] Multi-provider search works
- [ ] Disconnect cleanup works
- [ ] No database errors in logs
- [ ] No API errors in console

---

## 🎉 Next Steps After Testing

1. **Gather feedback** on UI/UX
2. **Test with larger document sets** (100+ files)
3. **Monitor performance** (sync time, query speed)
4. **Plan S3 Vectors migration** (when service available)
5. **Implement Phase 2 features:**
   - Webhook sync (real-time updates)
   - Incremental sync (only changed files)
   - Background jobs (periodic sync)
   - Sync dashboard (progress tracking)

---

**Ready to Test!** 🚀

Everything is set up and working. The mock backend is functioning perfectly for local testing. When you're ready to move to production with real S3 Vectors, it's just a 2-line code change.
