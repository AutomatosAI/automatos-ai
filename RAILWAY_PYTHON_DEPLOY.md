# Railway Python Deployment Guide

## Overview
This guide covers deploying Automatos AI backend using Railway's native Python buildpack (Nixpacks) instead of Docker. This is **much faster** (no 20+ minute builds) and more efficient.

## How Railway Builds It

Railway **automatically** detects and builds your Python app when it finds:
- `requirements.txt` - Python dependencies
- `Procfile` - How to start the app
- `runtime.txt` - Python version (optional)
- `nixpacks.toml` - System dependencies (optional)

**No manual build needed!** Railway does it automatically on deploy.

## Railway Setup

### Service Templates

**Backend Service**: No template needed - Railway auto-detects Python  
**PostgreSQL**: Use pgvector template - https://railway.com/template/pgvector-pg18  
**Redis**: Use Redis template - Search "Redis" in Railway templates  
**Frontend**: No template needed - Railway auto-detects Next.js

### 1. Backend Service Configuration

**Create Backend Service:**
1. In Railway dashboard: Click **"New"** → **"GitHub Repo"** → Select your repository
2. Railway will create a new service

**Configure Backend:**
1. **Root Directory**: Set to `orchestrator`
2. **Dockerfile Path**: Leave **empty** (or remove it if already set)
3. **Build Command**: Leave empty (Railway auto-detects)
4. **Start Command**: Leave empty (uses `Procfile`)

Railway will automatically detect Python from `requirements.txt` and `Procfile`.

Railway will automatically:
- Detect Python from `requirements.txt`
- Install system dependencies from `nixpacks.toml`
- Install Python packages
- Download NLTK data
- Start with `Procfile` command

### 2. Environment Variables

Set these in Railway Backend Service:

```bash
# Database (auto-provided by Railway PostgreSQL service)
DATABASE_URL=${DATABASE_URL}

# Redis (auto-provided by Railway Redis service)
REDIS_URL=${REDIS_URL}

# Encryption Key (CRITICAL - use your existing key)
CREDENTIAL_ENCRYPTION_KEY=Q9XenEwnmUC92ssMABH1VEXjgX6obqs_ZFlJP4JCB_s=

# API Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
API_KEY=your-secure-api-key-here

# CORS (your frontend domain)
CORS_ALLOW_ORIGINS=https://ui.automatos.app,http://localhost:3000

# NLTK Data (optional, defaults to /usr/local/nltk_data)
NLTK_DATA=/usr/local/nltk_data

# Port (Railway provides this automatically)
PORT=${PORT}
```

### 3. Deploy

1. **Push to GitHub** (if connected to Railway)
2. Railway **auto-detects** changes and starts building
3. Build uses **Nixpacks** (fast Python buildpack)
4. App starts automatically with `Procfile` command

## Local Development

### Option 1: Use Start Script (Recommended)

```bash
cd automatos-ai/orchestrator
./start.sh
```

Or specify a port:
```bash
./start.sh 8000
```

### Option 2: Manual Start

```bash
cd automatos-ai/orchestrator

# Create virtual environment (first time only)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set NLTK data path
export NLTK_DATA=/usr/local/nltk_data

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Start with hot reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Docker Compose (Still Works)

The existing `docker-compose.yml` still works for local development:

```bash
cd automatos-ai
docker-compose up --build
```

## File Structure

```
automatos-ai/orchestrator/
├── Procfile              # Railway start command (--reload for dev mode)
├── runtime.txt           # Python version
├── nixpacks.toml         # System dependencies (tesseract, ghostscript, etc.)
├── requirements.txt      # Python packages
├── start.sh              # Local development script
├── main.py               # FastAPI app entry point
└── ...
```

## Build Process (Railway)

When you push to Railway, it automatically:

1. **Detects** Python app from `requirements.txt`
2. **Installs** system packages from `nixpacks.toml`:
   - Python 3.11
   - gcc, g++ (for compiling Python packages)
   - postgresql-client
   - tesseract-ocr
   - ghostscript
   - etc.
3. **Installs** Python packages from `requirements.txt`
4. **Downloads** NLTK data (punkt, stopwords)
5. **Starts** app using `Procfile`: `uvicorn main:app --host 0.0.0.0 --port $PORT --reload`

## Benefits Over Docker

✅ **Much faster builds** (2-5 minutes vs 20+ minutes)  
✅ **Better caching** (Railway caches Python packages)  
✅ **Smaller images** (optimized buildpack)  
✅ **Easier debugging** (standard Python environment)  
✅ **Auto-detection** (no Dockerfile needed)

## Troubleshooting

### Build Fails
- Check `requirements.txt` syntax
- Verify `Procfile` format
- Check Railway logs for specific errors

### App Won't Start
- Verify `DATABASE_URL` and `REDIS_URL` are set
- Check `CREDENTIAL_ENCRYPTION_KEY` is correct (no quotes)
- Review Railway logs for startup errors

### NLTK Data Missing
- Check `NLTK_DATA` environment variable
- Verify `nixpacks.toml` install phase includes NLTK download

## Switching Back to Docker

If you need to switch back to Docker:

1. In Railway: Set **Dockerfile Path** to `orchestrator/Dockerfile`
2. Set **Root Directory** to `automatos-ai`
3. Redeploy

The Dockerfile still works for local `docker-compose` usage.
