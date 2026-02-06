# Cloud Document Sync Bug Report & Fix (PRD-42)

**Date:** 2026-02-05
**Status:** FIXED - Awaiting Testing
**Critical Issue:** Files download but never complete processing

---

## 🔍 Investigation Summary

### Actual Problem (Not What Was Reported)
The handover document stated files "never reach upload_document()", but investigation revealed:
- ✅ Files **DO** download successfully
- ✅ `upload_document()` **IS** called
- ✅ Document records **ARE** created
- ❌ **Processing fails** during embedding generation or S3 storage
- ❌ Cloud documents incorrectly marked as "synced"

### Database Evidence
```sql
-- Documents table
ID: 688, File: CONTEXT_ENGINEERING_GUIDE.md, Status: failed, Chunks: 0
ID: 687, File: CREDENTIAL_SYSTEM_GUIDE.md, Status: failed, Chunks: 0
ID: 686, File: DEVELOPER_GUIDE.md, Status: failed, Chunks: 0

-- Cloud documents table
ID: 18, File: CONTEXT_ENGINEERING_GUIDE.md, Status: synced, DocID: 688, Chunks: 0  ← WRONG!
ID: 17, File: CREDENTIAL_SYSTEM_GUIDE.md, Status: synced, DocID: 687, Chunks: 0  ← WRONG!
ID: 16, File: DEVELOPER_GUIDE.md, Status: synced, DocID: 686, Chunks: 0  ← WRONG!
```

---

## 🐛 Root Causes Identified

### Bug #1: Silent Failure Masking
**File:** `cloud_sync_service.py:344`
**Issue:** Assumes `document_id != None` means success

```python
if document_id:  # ← TRUE even when processing fails!
    # Marks cloud_document as "synced" (WRONG!)
```

**Problem:** `upload_document()` creates document record BEFORE processing, so failures during embedding/S3 storage still return a valid ID.

### Bug #2: Missing Exception Traceback
**File:** `cloud_sync_service.py:395`
**Issue:** No full traceback logged

```python
except Exception as e:
    logger.error(f"Sync failed for {file_name}: {e}")  # ← No exc_info=True!
```

**Result:** Impossible to diagnose WHY processing fails.

### Bug #3: S3 Vectors Error Swallowing
**File:** `s3_vectors_backend.py:210-212`
**Issue:** Returns empty list instead of raising exception

```python
except ClientError as e:
    logger.error(f"S3 Vectors put failed: {e}")
    return []  # ← Should raise!
```

**Result:** S3 storage failures are silent.

### Bug #4: Metadata Mismatch
**File:** `manager.py:882-890` vs `s3_vectors_backend.py:174-186`

**DocumentManager sends:**
```python
{
    "document_id": "688",
    "source_file": "guide.md",
    "file_type": "md"
}
```

**S3 Vectors expects:**
```python
{
    "external_file_id": "...",  # ← Uses this for key generation
    "file_name": "...",
    "app_name": "..."
}
```

**Result:** All vectors get key `doc_unknown_chunk_N` (broken!)

---

## ✅ Fixes Applied

### Fix #1: Check Document Status Before Marking Synced
**File:** `cloud_sync_service.py:344-357`

```python
if document_id:
    # Get chunk count AND STATUS from documents table
    from core.models import Document
    doc = self.db.query(Document).get(document_id)
    chunk_count = doc.chunk_count if doc else 0
    doc_status = doc.status if doc else "failed"

    # Only mark as synced if processing completed successfully
    if doc_status != "completed":
        logger.error(f"Document {document_id} processing failed with status: {doc_status}")
        if existing:
            existing.sync_status = "error"
            existing.sync_error = f"Document processing failed: {doc_status}"
            self.db.commit()
        files_errored += 1
        continue  # Skip to next file

    # NOW mark as synced (only if status == "completed")
    if existing:
        existing.document_id = document_id
        existing.sync_status = "synced"
        ...
```

### Fix #2: Add Full Exception Traceback
**File:** `cloud_sync_service.py:395`

```python
except Exception as e:
    logger.error(f"Sync failed for {file_name}: {e}", exc_info=True)  # ← Added exc_info=True
    files_errored += 1
```

### Fix #3: Raise S3 Vectors Exceptions
**File:** `s3_vectors_backend.py:210-212`

```python
except ClientError as e:
    logger.error(f"S3 Vectors put failed: {e}")
    raise  # ← Re-raise instead of returning []
```

### Fix #4: Fix Metadata Schema Mismatch
**File:** `manager.py:880-892`

```python
documents_for_s3.append({
    "external_file_id": str(document_id),  # ← Use document_id as external_file_id
    "document_id": str(document_id),
    "chunk_index": chunk.chunk_index,
    "chunk_text": chunk.content[:500],
    "file_name": os.path.basename(file_path),  # ← Match expected field name
    "source_file": os.path.basename(file_path),
    "file_path": file_path,
    "file_type": file_type.value if hasattr(file_type, 'value') else str(file_type),
    "app_name": "document_sync",  # ← Add app_name field
    "workspace_id": workspace_id
})
```

### Fix #5: Add Diagnostic Logging
**File:** `cloud_sync_service.py:330-343`

```python
logger.info(f"✅ Downloaded {file_name} to {tmp_path}")

logger.info(f"🔄 Starting upload_document() for {file_name}")
document_id = await doc_manager.upload_document(...)

logger.info(f"✅ upload_document() returned document_id={document_id} for {file_name}")
```

---

## 🎯 What These Fixes Accomplish

1. **Full error visibility** - Will now see complete tracebacks when processing fails
2. **Correct status tracking** - Cloud documents only marked "synced" if processing succeeds
3. **Proper error propagation** - S3 failures now raise exceptions instead of silent failure
4. **Correct S3 Vector keys** - Vectors stored with proper keys: `doc_{document_id}_chunk_{N}`
5. **Detailed progress logging** - Can track exactly where in the pipeline failures occur

---

## 🔬 Next Steps: Run Tests

### 1. Restart Backend with Fixes
```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/orchestrator
python main.py
```

### 2. Watch Logs
```bash
tail -f <log-file> | grep -E "Downloaded|Starting upload|returned document_id|ERROR|Sync failed"
```

### 3. Trigger Sync from UI
- Navigate to Automatos folder
- Click "Sync" button
- Should have 4x .md files and 2x .pdf files

### 4. Expected Output (Success)
```
✅ Downloaded CONTEXT_ENGINEERING_GUIDE.md to /tmp/...
🔄 Starting upload_document() for CONTEXT_ENGINEERING_GUIDE.md
✅ Uploaded document to S3: s3://automatos-ai/workspaces/1/documents/689_CONTEXT_ENGINEERING_GUIDE.md
Starting document processing for document 689
Extracted 12534 characters from document 689
SemanticChunker created 15 chunks
✅ Stored 15 vectors in S3 for document 689
Document 689 processed successfully with 15 chunks
✅ upload_document() returned document_id=689 for CONTEXT_ENGINEERING_GUIDE.md
```

### 5. Expected Output (Failure - But Now Visible!)
```
✅ Downloaded CONTEXT_ENGINEERING_GUIDE.md to /tmp/...
🔄 Starting upload_document() for CONTEXT_ENGINEERING_GUIDE.md
✅ Uploaded document to S3: s3://automatos-ai/workspaces/1/documents/689_CONTEXT_ENGINEERING_GUIDE.md
Starting document processing for document 689
Extracted 12534 characters from document 689
ERROR: Failed to store vectors in S3: <ACTUAL ERROR HERE>  ← Will now see this!
Traceback (most recent call last):  ← Full traceback now visible!
  File "manager.py", line 917, in _process_document
    raise
  ...
  botocore.exceptions.ClientError: An error occurred (InvalidParameterValue) when calling PutVectors: ...
Sync failed for CONTEXT_ENGINEERING_GUIDE.md: <error>
```

---

## 🔍 Likely Actual Error (Hypothesis)

Based on the pattern of failures, the most likely culprits are:

### Hypothesis #1: S3 Vectors Index Doesn't Exist
The S3 Vectors index ARN exists but may not be properly initialized:
```
arn:aws:s3vectors:eu-west-1:810390208173:bucket/automatos-ai/index/automatos-vector-index
```

**Check:** Verify index exists and is active in AWS console

### Hypothesis #2: Embedding Dimension Mismatch
Config says 1024 dimensions, but HuggingFace model might generate different size:
```python
S3_VECTORS_DIMENSION=1024
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

**Check:** Verify BGE-large-en-v1.5 outputs 1024-dim embeddings (it does)

### Hypothesis #3: AWS Credentials or Permissions
S3 bucket exists but S3 Vectors permissions might be different:
- ✅ Can create S3 bucket (`automatos-ai`)
- ❌ Can't write to S3 Vectors index?

**Check:** IAM permissions for `s3vectors:PutVectors` action

### Hypothesis #4: Bucket Name Mismatch
Documents go to `automatos-ai` but vectors expect workspace-scoped bucket:
```python
self.bucket_name = bucket_template.replace("{workspace_id}", self.workspace_id)
# If S3_VECTORS_BUCKET="automatos-ai" → "automatos-ai"
# If S3_VECTORS_BUCKET="automatos-vectors-{workspace_id}" → "automatos-vectors-1"
```

**Check:** What is `S3_VECTORS_BUCKET` in .env?

---

## 📋 Testing Checklist

After fixes, verify:

- [ ] Files download from Google Drive
- [ ] `upload_document()` is called (see "Starting upload_document" log)
- [ ] Documents uploaded to S3 (see "Uploaded document to S3" log)
- [ ] Text extraction works (see "Extracted N characters" log)
- [ ] Chunking works (see "SemanticChunker created N chunks" log)
- [ ] Embeddings generated (see embedding-related logs)
- [ ] Vectors stored in S3 (see "Stored N vectors in S3" log)
- [ ] Database updated correctly:
  - `documents.status = "completed"`
  - `documents.chunk_count > 0`
  - `cloud_documents.sync_status = "synced"`
  - `cloud_documents.chunk_count > 0`
- [ ] Documents searchable in UI

---

## 💾 Files Modified

1. `orchestrator/modules/rag/services/cloud_sync_service.py`
   - Line 344-357: Check document status before marking synced
   - Line 330: Add download success logging
   - Line 335: Add upload_document start logging
   - Line 343: Add upload_document success logging
   - Line 395: Add exc_info=True to exception logging

2. `orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py`
   - Line 212: Raise exception instead of returning []

3. `orchestrator/modules/rag/ingestion/manager.py`
   - Line 880-892: Fix metadata schema to match S3 Vectors expectations

---

## 🎓 Lessons Learned

1. **Don't trust return values** - Just because `document_id` is returned doesn't mean processing succeeded
2. **Always log full tracebacks** - `exc_info=True` is critical for debugging async code
3. **Avoid silent failures** - Returning `[]` instead of raising exceptions masks errors
4. **Validate API contracts** - Mismatched schemas between components cause subtle bugs
5. **Test error paths** - The happy path worked; error handling was broken

---

## 📞 For Next Agent

The fixes are applied. To complete this task:

1. **Restart the orchestrator** to load the fixed code
2. **Trigger a sync** from the UI
3. **Check logs** for the actual error (will now be visible)
4. **Fix the underlying issue** revealed by the logs
5. **Verify end-to-end** that documents show in UI with chunk_count > 0

The fixes ensure you'll see the real error. Most likely it's one of the hypotheses above (permissions, dimension mismatch, or bucket name issue).

Good luck!
