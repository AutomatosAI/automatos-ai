# AWS S3 Vectors Setup Guide
## For Automatos AI - Cloud Document Sync (PRD-42)

---

## ⚠️ **Important Note**

As of January 2025, **AWS S3 Vectors is not yet publicly available**. This service was mentioned in the PRD as a "new 2026 service".

**Current Status:**
- ✅ Mock S3 Vectors backend working (in-memory, for testing)
- ⏳ Real AWS S3 Vectors - waiting for service launch
- 🔄 Alternative: Can use other vector databases (Pinecone, Weaviate, etc.)

---

## 🔍 **Check if S3 Vectors is Available**

### Option 1: AWS Console
1. Go to https://console.aws.amazon.com/
2. Search for "S3 Vectors" in the services search bar
3. If it appears → Service is available
4. If not → Service not yet launched

### Option 2: AWS CLI
```bash
# List available services
aws service-quotas list-services | grep -i vector

# Try to list S3 Vectors quotas (will fail if not available)
aws service-quotas list-service-quotas --service-code s3-vectors
```

### Option 3: Boto3 (Python)
```bash
cd orchestrator
source venv/bin/activate
python << EOF
import boto3
try:
    client = boto3.client('s3vectors', region_name='us-east-1')
    print("✅ S3 Vectors service is available!")
except Exception as e:
    print(f"❌ S3 Vectors not available: {e}")
EOF
```

---

## 🚀 **If S3 Vectors IS Available**

### Step 1: Create IAM User for Automatos

1. **Go to IAM Console:**
   - https://console.aws.amazon.com/iam/

2. **Create New User:**
   - Click "Users" → "Add users"
   - Username: `automatos-s3-vectors`
   - Access type: ✅ Programmatic access
   - Click "Next: Permissions"

3. **Create Custom Policy:**
   - Click "Attach policies directly"
   - Click "Create policy"
   - Choose JSON tab
   - Paste this policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3VectorsFullAccess",
            "Effect": "Allow",
            "Action": [
                "s3vectors:CreateVectorBucket",
                "s3vectors:DeleteVectorBucket",
                "s3vectors:ListVectorBuckets",
                "s3vectors:CreateIndex",
                "s3vectors:DeleteIndex",
                "s3vectors:ListIndexes",
                "s3vectors:DescribeIndex",
                "s3vectors:PutVectors",
                "s3vectors:GetVectors",
                "s3vectors:DeleteVectors",
                "s3vectors:QueryVectors",
                "s3vectors:ListVectors"
            ],
            "Resource": "*"
        }
    ]
}
```

   - Name: `AutomatosS3VectorsPolicy`
   - Click "Create policy"

4. **Attach Policy to User:**
   - Go back to user creation
   - Refresh policies list
   - Search for `AutomatosS3VectorsPolicy`
   - Check the box
   - Click "Next: Tags" → "Next: Review" → "Create user"

5. **Save Credentials:**
   - **IMPORTANT:** Copy these immediately (won't be shown again)
   - Access Key ID: `AKIA...`
   - Secret Access Key: `wJalr...`

---

### Step 2: Configure Orchestrator Environment

1. **Update `.env` file:**

```bash
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/orchestrator

# Edit .env file
nano .env
```

2. **Update AWS section:**

```bash
# AWS S3 Vectors (PRD-42: Cloud Document Sync)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...  # From Step 1.5
AWS_SECRET_ACCESS_KEY=wJalr...  # From Step 1.5
S3_VECTORS_ENABLED=true
```

3. **Save and exit** (Ctrl+X, Y, Enter)

---

### Step 3: Test Connection

```bash
cd orchestrator
source venv/bin/activate

# Test AWS credentials
python << EOF
import boto3
from config import config

print(f"Region: {config.AWS_REGION}")
print(f"Access Key: {config.AWS_ACCESS_KEY_ID[:10]}...")
print(f"S3 Vectors Enabled: {config.S3_VECTORS_ENABLED}")

try:
    client = boto3.client(
        's3vectors',
        region_name=config.AWS_REGION,
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
    )

    # Try to list buckets
    response = client.list_vector_buckets()
    print(f"✅ Connected! Existing buckets: {len(response.get('buckets', []))}")

except Exception as e:
    print(f"❌ Connection failed: {e}")
EOF
```

---

### Step 4: Switch from Mock to Real S3 Vectors

**Update 2 files:**

#### File 1: `orchestrator/api/cloud_documents.py` (line ~523)

```python
# BEFORE (mock):
backend = get_vector_store(
    backend="s3_vectors_mock",
    workspace_id=str(ctx.workspace_id),
)

# AFTER (real):
backend = get_vector_store(
    backend="s3_vectors",
    workspace_id=str(ctx.workspace_id),
)
```

#### File 2: `orchestrator/api/cloud_documents.py` (line ~474)

```python
# BEFORE (mock):
backend = get_vector_store(
    backend="s3_vectors_mock",
    workspace_id=str(ctx.workspace_id)
)

# AFTER (real):
backend = get_vector_store(
    backend="s3_vectors",
    workspace_id=str(ctx.workspace_id)
)
```

---

### Step 5: Restart & Verify

```bash
# Restart API server
# (Kill existing process: Ctrl+C)
cd orchestrator
source venv/bin/activate
python -m uvicorn main:app --reload --port 8000
```

**Check logs for:**
```
✅ S3 vector bucket created: automatos-vectors-{workspace_id}
✅ S3 vector index created: documents-index
```

---

## 🔄 **If S3 Vectors is NOT Available Yet**

### Alternative Option 1: Continue with Mock (Recommended for now)

**Pros:**
- ✅ Works immediately
- ✅ No AWS costs
- ✅ Perfect for development/testing
- ✅ Easy to switch later (2 line code change)

**Cons:**
- ❌ Vectors stored in RAM (lost on restart)
- ❌ Not scalable to production

**Setup:** Nothing! Already working.

---

### Alternative Option 2: Use Pinecone

**Pinecone is a production-ready vector database with similar features to S3 Vectors.**

#### Setup:

1. **Sign up:** https://app.pinecone.io/
2. **Create API Key**
3. **Install:** `pip install pinecone-client`
4. **Create Backend:** Similar to `s3_vectors_backend.py`

**Pros:**
- ✅ Production-ready now
- ✅ Free tier (100k vectors)
- ✅ Managed service (no infrastructure)
- ✅ Good documentation

**Cons:**
- ❌ Another service to manage
- ❌ Migration needed if switching to S3 Vectors later

---

### Alternative Option 3: Use pgvector (Keep Current)

**Continue using PostgreSQL with pgvector extension.**

**Pros:**
- ✅ Already working
- ✅ No additional setup
- ✅ Integrated with existing DB

**Cons:**
- ❌ Not optimized for large-scale vector search
- ❌ Limited to 2M vectors (vs 2B for S3 Vectors)
- ❌ Slower for high-volume queries

---

## 📊 **Cost Estimates**

### S3 Vectors (When Available)
- **Storage:** ~$0.06/GB/month for vectors
- **Queries:** ~$0.004 per 1000 queries
- **Example:** 100GB vectors + 1M queries/month = ~$10/month

### Pinecone
- **Free tier:** 100k vectors, 1 index
- **Starter:** $70/month (5M vectors)
- **Scale:** Custom pricing

### pgvector (Current)
- **Cost:** $0 (part of PostgreSQL)
- **Scales to:** ~2M vectors

---

## 🧪 **Testing Checklist**

Once AWS is set up:

- [ ] IAM user created with S3 Vectors policy
- [ ] Credentials added to `.env`
- [ ] Connection test passes
- [ ] Code switched from mock to real backend
- [ ] API server restarted
- [ ] Sync a test document
- [ ] Verify vector bucket created in AWS console
- [ ] Verify index created
- [ ] Test RAG query returns results
- [ ] Check CloudWatch logs for errors

---

## 📞 **Troubleshooting**

### Error: "Module 's3vectors' not found"
**Cause:** Service not available in boto3 yet
**Fix:** Wait for AWS to launch service or use mock/alternative

### Error: "Access Denied"
**Cause:** IAM policy missing permissions
**Fix:** Review policy in Step 1.3, ensure all actions are included

### Error: "Region not supported"
**Cause:** S3 Vectors may not be in all regions initially
**Fix:** Try different region (us-east-1, us-west-2, eu-west-1)

### Error: "Bucket already exists"
**Cause:** Bucket name conflict (unlikely with UUID workspace IDs)
**Fix:** Check AWS console, delete old bucket

---

## 🎯 **Next Steps**

1. **Check if S3 Vectors is available** (see top of guide)
2. **If YES:** Follow Steps 1-5
3. **If NO:** Continue with mock, revisit monthly
4. **Test:** Sync documents from Google Drive/Dropbox
5. **Monitor:** Check AWS console for bucket creation
6. **Scale:** Add more workspaces, test performance

---

## 📝 **Useful AWS Console Links**

- IAM Users: https://console.aws.amazon.com/iam/home#/users
- S3 Vectors Console: https://console.aws.amazon.com/s3vectors/ (when available)
- CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/
- Billing: https://console.aws.amazon.com/billing/

---

**Questions?** Check AWS S3 Vectors documentation (when released) or use mock backend for now.
