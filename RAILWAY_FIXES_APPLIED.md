# Railway Fixes Applied

## Changes Made

### 1. **Startup Script** (`start-railway.sh`)
- Handles libmagic setup for both Railpack and Nixpacks
- Automatically finds and links libmagic library
- Sets proper LD_LIBRARY_PATH
- Uses Railway's PORT environment variable
- Starts with 4 workers (production mode)

### 2. **Procfile Updated**
- Changed from direct uvicorn command to use `start-railway.sh`
- Ensures libmagic is available before app starts

### 3. **Nixpacks Configuration**
- Removed libmagic symlink from install (moved to startup script)
- Added chmod for startup script
- Kept caching optimizations

### 4. **Railway.json**
- Set to use RAILPACK (better caching, smaller builds)

## For Railpack (Current Builder)

Railpack uses environment variables for system packages. Set in Railway dashboard:

**Environment Variable:**
```
RAILPACK_BUILD_APT_PACKAGES=libmagic1,tesseract-ocr,ghostscript,postgresql-client,gcc,g++
```

Or Railpack will auto-install based on Python package requirements.

## Testing

After deploying, test:
```bash
curl https://api.automatos.app/health
curl https://automatos-ai-production.up.railway.app/health
```

Both should return JSON with status "ok" instead of 502.

## Caching

- **Railpack**: Automatic pip caching via BuildKit
- **Nixpacks**: Caches based on requirements.txt hash
- Next deploy should be faster if requirements.txt hasn't changed
