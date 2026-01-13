# Railway Deployment Checklist

## ✅ Fixes Applied

### 1. Startup Script (`start-railway.sh`)
- ✅ Handles libmagic for both Railpack and Nixpacks
- ✅ Sets LD_LIBRARY_PATH correctly
- ✅ Uses Railway's PORT environment variable
- ✅ Production mode with 4 workers (faster than reload)

### 2. Procfile
- ✅ Uses startup script instead of direct uvicorn
- ✅ Ensures libmagic is available before app starts

### 3. Railway Configuration
- ✅ Set to use RAILPACK (better caching)
- ✅ Nixpacks config updated as fallback

## 🔧 Railway Dashboard Settings

### Backend Service

**Build Settings:**
- Builder: **Railpack** (or Nixpacks if Railpack not available)
- Root Directory: `orchestrator`

**Environment Variables:**
```bash
# Database (auto-provided)
DATABASE_URL=${DATABASE_URL}

# Redis (auto-provided)
REDIS_URL=${REDIS_URL}

# Encryption Key (CRITICAL)
CREDENTIAL_ENCRYPTION_KEY=Q9XenEwnmUC92ssMABH1VEXjgX6obqs_ZFlJP4JCB_s=

# API Config
ENVIRONMENT=production
LOG_LEVEL=INFO
API_KEY=your-secure-api-key

# CORS
CORS_ALLOW_ORIGINS=https://ui.automatos.app,http://localhost:3000

# NLTK
NLTK_DATA=/usr/local/nltk_data

# For Railpack: System packages (if needed)
RAILPACK_BUILD_APT_PACKAGES=libmagic1,tesseract-ocr,ghostscript,postgresql-client,gcc,g++
```

## 🧪 Testing After Deploy

```bash
# Test health endpoint
curl https://api.automatos.app/health
curl https://automatos-ai-production.up.railway.app/health

# Should return:
# {"status":"ok","timestamp":"..."}
# NOT: {"status":"error","code":502}
```

## 📊 Expected Results

**First Deploy:**
- Build time: 5-10 minutes (installing all packages)
- Image size: ~4GB (expected for ML stack)

**Subsequent Deploys (if requirements.txt unchanged):**
- Build time: 1-3 minutes (using cache)
- Image size: Same (~4GB)

**If caching works, you'll see:**
- "Using cached" messages in build logs
- Faster pip install step

## 🐛 Troubleshooting

**502 Error:**
- Check Railway logs for startup errors
- Verify libmagic is installed (check logs)
- Verify PORT environment variable is set

**libmagic Error:**
- Startup script should handle this automatically
- Check Railway logs to see if symlink was created

**Caching Not Working:**
- Verify requirements.txt hash hasn't changed
- Check if using Railpack (better caching)
- Look for "Using cached" in build logs
