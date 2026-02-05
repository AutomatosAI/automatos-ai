# PRD-42 Cloud Document Sync - Testing Guide

## ✅ What's Been Built

### **1. AWS Setup Guide**
**File:** `AWS_S3_VECTORS_SETUP.md`
- Complete instructions for S3 Vectors setup
- IAM policy configuration
- Connection testing
- Alternative options (Pinecone, pgvector)
- Troubleshooting guide

### **2. Frontend Components**

#### **Provider Cards** (`frontend/components/documents/provider-cards.tsx`)
- Grid view of all storage providers
- Shows stats: document count, chunk count, sync status
- "Manual Upload" card for direct uploads
- Cloud provider cards (Google Drive, Dropbox, etc.)
- Click to browse/manage each provider

#### **Provider Browser** (`frontend/components/documents/provider-browser.tsx`)
- Folder tree navigation
- File list with sync status
- Select root folder for syncing
- "Sync Now" button
- Real-time sync progress
- Search and filter files
- Stats dashboard (total, synced, pending, chunks)

### **3. API Hooks** (`frontend/hooks/use-cloud-storage.ts`)
- `useCloudConnections()` - Get all connected providers
- `useCloudFolders()` - Browse folder tree
- `useCloudFiles()` - List files with sync status
- `useSelectRootFolder()` - Set sync root folder
- `useTriggerSync()` - Start sync job
- `useSyncJob()` - Monitor sync progress
- `useDisconnectProvider()` - Remove connection

### **4. Backend** (Already Complete)
- ✅ 12 API endpoints (`orchestrator/api/cloud_documents.py`)
- ✅ CloudSyncService (folder nav, file listing, sync)
- ✅ Mock S3 Vectors backend (in-memory testing)
- ✅ Database tables (cloud_sync_config, cloud_documents, cloud_sync_jobs)
- ✅ Composio integration for Google Drive/Dropbox

### **5. Architecture**
```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
┌──────▼────────────────────────────────────────┐
│  Frontend (React/Next.js)                      │
│  - Provider Cards                              │
│  - Folder Browser                              │
│  - File List with Sync Status                 │
└──────┬────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────┐
│  API (FastAPI)                                 │
│  - /api/cloud-documents/connections            │
│  - /api/cloud-documents/connections/{id}/sync  │
│  - /api/cloud-documents/rag/query              │
└──────┬────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────┐
│  Composio (OAuth + File Operations)           │
│  - GOOGLEDRIVE_LIST_FILES                     │
│  - DROPBOX_DOWNLOAD_FILE                      │
└──────┬────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────┐
│  Storage                                       │
│  - PostgreSQL (metadata only)                 │
│  - Mock S3 Vectors (embeddings)               │
│  - User's Cloud (original files)              │
└───────────────────────────────────────────────┘
```

---

## 🧪 Testing Instructions

### **Step 1: Start Services**

```bash
# Terminal 1: Backend API
cd orchestrator
source venv/bin/activate
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

### **Step 2: Verify Connections**

1. Go to `http://localhost:3000/tools`
2. Verify Google Drive and Dropbox are connected
3. Check status shows "Active" or "Connected"

### **Step 3: Navigate to Documents**

1. Go to `http://localhost:3000/documents`
2. Click on "Library" tab
3. You should see **Provider Cards**:
   - "Uploaded Documents" (manual upload)
   - "Google Drive" (your connected account)
   - "Dropbox" (your connected account)

### **Step 4: Test Google Drive Sync**

1. **Click on "Google Drive" card**
   - Should open Provider Browser view
   - Shows folder tree and stats

2. **Browse Folders**
   - Click on folders to navigate
   - Click "Up" to go back
   - Current path shown at top

3. **Select Root Folder**
   - Navigate to folder you want to sync (e.g., `/Automatos` or `/Work Documents`)
   - Click "Set as Root"
   - Confirmation should appear

4. **Trigger Sync**
   - Click "Sync Now" button
   - Watch progress:
     - Files change from "Pending" to "Synced"
     - Chunk counts update
     - Stats update in real-time

5. **Verify Sync Results**
   - Check "Synced" badge on files
   - Verify chunk counts > 0
   - Check "Total Chunks" stat

### **Step 5: Test Dropbox Sync**

1. Click "Back" to return to provider cards
2. Click on "Dropbox" card
3. Repeat steps 2-4 for Dropbox

### **Step 6: Test RAG Queries**

```bash
# Via API
curl -X POST http://localhost:8000/api/cloud-documents/rag/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "What projects are mentioned in the documents?",
    "top_k": 10,
    "min_similarity": 0.5
  }'
```

**Expected Response:**
- Results from BOTH Google Drive and Dropbox
- Each result shows source (app_name, file_path)
- Similarity scores
- Chunk content

### **Step 7: Test Manual Upload**

1. Click on "Uploaded Documents" card (or use Upload button)
2. Upload a test PDF or document
3. Verify it appears in the manual upload section
4. Vectors should be stored in mock S3 backend

---

## 🔍 What to Check

### **Frontend Checks**
- [ ] Provider cards display correctly
- [ ] Stats are accurate (document count, chunks)
- [ ] Folder navigation works
- [ ] File list loads without errors
- [ ] Sync button triggers sync
- [ ] Progress updates in real-time
- [ ] Search filters work
- [ ] "Set as Root" updates configuration

### **Backend Checks**
```bash
# Check database
psql "postgresql://postgres:alrckxcy2fgvy7zxzhhv0wa37gtc690w@shortline.proxy.rlwy.net:47906/railway" -c "
SELECT
  app_name,
  COUNT(*) as doc_count,
  SUM(chunk_count) as total_chunks
FROM cloud_documents
GROUP BY app_name;
"

# Check sync jobs
psql "..." -c "
SELECT
  id,
  status,
  files_synced,
  files_errored,
  total_chunks_created
FROM cloud_sync_jobs
ORDER BY id DESC
LIMIT 5;
"
```

### **Mock S3 Backend Check**
```bash
cd orchestrator
source venv/bin/activate
python scripts/test_cloud_sync.py
```

---

## 🐛 Common Issues

### Issue: Provider cards not showing
**Fix:** Check `useCloudConnections()` hook is fetching data
```bash
# Check API endpoint
curl http://localhost:8000/api/cloud-documents/connections \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Issue: Folders not loading
**Fix:** Verify Composio actions are enabled
- Go to `/tools`
- Check Google Drive has `GOOGLEDRIVE_LIST_FILES` enabled
- Check Dropbox has `DROPBOX_LIST_FOLDER` enabled

### Issue: Sync fails
**Check logs:**
```bash
tail -f logs/orchestrator.log
```

**Common causes:**
- Missing Composio action permissions
- File too large (>10MB limit)
- Unsupported file type
- Rate limiting from cloud provider

### Issue: No search results in RAG
**Fix:**
1. Verify chunks were created (check `chunk_count` in database)
2. Check embeddings were generated
3. Verify mock S3 backend has vectors

---

## 📊 Performance Testing

### **Small Scale (10-50 files)**
- Sync should complete in < 2 minutes
- No errors expected
- All files should have chunks

### **Medium Scale (50-200 files)**
- Sync may take 5-10 minutes
- Watch for rate limiting from cloud provider
- Some large files may be skipped (check sync_status='error')

### **Large Scale (200+ files)**
- Consider batch processing
- Monitor memory usage (mock backend is in-memory)
- May need to switch to real S3 Vectors

---

## 🚀 Next Steps: Email Upload Trigger

### **Architecture**
```
Email arrives
  ↓
Trigger/Recipe detects attachment
  ↓
Download attachment
  ↓
Choose destination:
  - Upload to Google Drive → /Invoices
  - Upload to Dropbox → /Reports
  ↓
Auto-trigger sync for that folder
  ↓
Document embedded and searchable
```

### **What We Need**

1. **Composio Actions for Upload**
   ```
   GOOGLEDRIVE_UPLOAD_FILE
   DROPBOX_UPLOAD_FILE
   ONEDRIVE_UPLOAD_FILE
   ```

2. **Recipe Integration**
   - Email trigger (via Composio: GMAIL_NEW_EMAIL)
   - Extract attachments
   - Upload to cloud storage
   - Call sync endpoint

3. **API Enhancement**
   - Add endpoint: `POST /api/cloud-documents/upload-to-provider`
   - Parameters: file, connectionId, targetPath
   - Returns: uploadedFileId

### **Testing Email Upload**

1. **Set up email trigger** (via Composio)
2. **Send test email** to yourself with PDF attachment
3. **Recipe should:**
   - Detect new email
   - Extract attachment
   - Upload to Google Drive → `/Automatos/Emails`
   - Trigger sync
   - Document becomes searchable

### **Use Cases**
```yaml
# Invoice Processing
Trigger: Email with "invoice" in subject
Action:
  - Extract PDF attachment
  - Upload to Google Drive → /Invoices/{YYYY-MM}
  - Auto-sync folder
  - Tag with "invoice" + sender

# Daily Reports
Trigger: Cron (daily 9 AM)
Action:
  - Generate report PDF
  - Upload to Dropbox → /Reports/{date}
  - Auto-sync folder
  - Send Slack notification

# Support Tickets
Trigger: Jira ticket closed
Action:
  - Fetch ticket + comments
  - Generate markdown
  - Upload to Google Drive → /Support/{ticket_id}
  - Auto-sync
  - Build knowledge base
```

---

## ✅ Success Criteria

### **Phase 1: Manual Testing (Now)**
- [ ] Provider cards display
- [ ] Folder navigation works
- [ ] Sync completes successfully
- [ ] Files show sync status
- [ ] RAG queries return results from cloud docs

### **Phase 2: AWS S3 Vectors (When Available)**
- [ ] Switch from mock to real S3 backend
- [ ] Verify vectors persist across restarts
- [ ] Test at scale (1000+ documents)
- [ ] Monitor AWS costs

### **Phase 3: Automation (Next)**
- [ ] Email trigger working
- [ ] Auto-upload to cloud storage
- [ ] Auto-sync after upload
- [ ] Notifications working

---

## 📝 Testing Checklist

Copy this checklist for your testing session:

```
## PRD-42 Testing Session - [DATE]

### Setup
- [ ] Backend API running (port 8000)
- [ ] Frontend running (port 3000)
- [ ] Google Drive connected
- [ ] Dropbox connected
- [ ] Database accessible

### Provider Cards
- [ ] Manual upload card shows
- [ ] Google Drive card shows
- [ ] Dropbox card shows
- [ ] Stats are correct
- [ ] Click opens browser

### Google Drive
- [ ] Folder tree loads
- [ ] Files list loads
- [ ] Set root folder works
- [ ] Sync button works
- [ ] Files show "Synced" status
- [ ] Chunk counts > 0

### Dropbox
- [ ] Same checks as Google Drive

### RAG Queries
- [ ] Query returns results
- [ ] Results from multiple providers
- [ ] Relevance is good
- [ ] Source attribution correct

### Database
- [ ] cloud_documents table populated
- [ ] cloud_sync_jobs table has entries
- [ ] Metadata correct

### Errors
- [ ] Check orchestrator.log for errors
- [ ] Check frontend console for errors
- [ ] Check browser network tab for 500s

### Notes
- Performance: ___
- Issues found: ___
- Next steps: ___
```

---

**Ready to test!** Start with Step 1 and work through the checklist. Report any issues you find.
