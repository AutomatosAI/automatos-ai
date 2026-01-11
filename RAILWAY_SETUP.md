# Railway Deployment Guide

## Overview
This guide covers deploying Automatos AI to Railway with PostgreSQL, Redis, Backend, and Frontend services.

## Prerequisites
- Railway account (https://railway.app)
- GitHub repository connected to Railway
- Custom domains configured (api.automatos.app, ui.automatos.app)

## Service Setup

### 1. PostgreSQL Service
- **Template**: Use Railway's pgvector template (PostgreSQL 18 with pgvector)
- **Template URL**: https://railway.com/deploy/pgvector-pg18
- **Connection**: Railway automatically provides `DATABASE_URL` environment variable

### 2. Redis Service
- **Template**: Use Railway's Redis template
- **Connection**: Railway automatically provides `REDIS_URL` environment variable

### 3. Backend Service (Orchestrator)

#### Build Configuration
- **Dockerfile Path**: `automatos-ai/orchestrator/Dockerfile`
- **Build Stage**: `production` (Railway will use this automatically)
- **Root Directory**: Set to `automatos-ai` in Railway service settings

#### Environment Variables
Set these in Railway's backend service:

```bash
# Database (from PostgreSQL service)
DATABASE_URL=${DATABASE_URL}  # Auto-provided by Railway

# Redis (from Redis service)  
REDIS_URL=${REDIS_URL}  # Auto-provided by Railway

# Or individual Redis vars if REDIS_URL not available:
REDIS_HOST=<from-redis-service>
REDIS_PORT=<from-redis-service>
REDIS_PASSWORD=<from-redis-service>

# API Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
API_KEY=<your-secure-api-key>

# LLM API Keys (optional, can be set in UI later)
OPENAI_API_KEY=<optional>
ANTHROPIC_API_KEY=<optional>
```

#### Port Configuration
- Railway automatically detects port 8000 from EXPOSE directive
- No manual port configuration needed

#### Custom Domain
- Add custom domain: `api.automatos.app`
- Railway handles HTTPS automatically

### 4. Frontend Service

#### Build Configuration
- **Dockerfile Path**: `automatos-ai/frontend/Dockerfile`
- **Build Stage**: `production` (Railway will use this automatically)
- **Root Directory**: Set to `automatos-ai` in Railway service settings

#### Environment Variables
Set these in Railway's frontend service:

```bash
# Backend API URL (use Railway's backend URL or custom domain)
NEXT_PUBLIC_API_URL=https://api.automatos.app
# OR use Railway's generated URL:
# NEXT_PUBLIC_API_URL=${RAILWAY_PUBLIC_DOMAIN}  # Backend service URL

# WebSocket URL (if using WebSockets)
NEXT_PUBLIC_WS_URL=wss://api.automatos.app/ws

# Node Environment
NODE_ENV=production
```

#### Port Configuration
- Railway automatically detects port 3000 from EXPOSE directive
- No manual port configuration needed

#### Custom Domain
- Add custom domain: `ui.automatos.app`
- Railway handles HTTPS automatically

## Railway Service Configuration Steps

### Step 1: Create Services
1. Create new project in Railway
2. Add PostgreSQL service (use pgvector template)
3. Add Redis service
4. Add Backend service (connect to GitHub repo)
5. Add Frontend service (connect to GitHub repo)

### Step 2: Configure Backend Service
1. Go to Backend service → Settings
2. Set **Root Directory** to: `automatos-ai`
3. Set **Dockerfile Path** to: `orchestrator/Dockerfile`
4. Add environment variables (see above)
5. Connect PostgreSQL service (Railway will auto-inject `DATABASE_URL`)
6. Connect Redis service (Railway will auto-inject `REDIS_URL`)

### Step 3: Configure Frontend Service
1. Go to Frontend service → Settings
2. Set **Root Directory** to: `automatos-ai`
3. Set **Dockerfile Path** to: `frontend/Dockerfile`
4. Add environment variables (see above)
5. Set `NEXT_PUBLIC_API_URL` to your backend URL

### Step 4: Configure Custom Domains
1. **Backend Domain**:
   - Go to Backend service → Settings → Domains
   - Add custom domain: `api.automatos.app`
   - Update DNS: Add CNAME record pointing to Railway's domain

2. **Frontend Domain**:
   - Go to Frontend service → Settings → Domains
   - Add custom domain: `ui.automatos.app`
   - Update DNS: Add CNAME record pointing to Railway's domain

### Step 5: Connect Services
Railway automatically provides connection strings when services are in the same project:
- PostgreSQL → Backend: `DATABASE_URL` auto-injected
- Redis → Backend: `REDIS_URL` auto-injected

## Environment Variables Reference

### Backend Required Variables
```bash
DATABASE_URL=postgresql://...  # Auto from PostgreSQL service
REDIS_URL=redis://...          # Auto from Redis service
ENVIRONMENT=production
```

### Backend Optional Variables
```bash
API_KEY=<secure-key>
OPENAI_API_KEY=<optional>
ANTHROPIC_API_KEY=<optional>
LOG_LEVEL=INFO
```

### Frontend Required Variables
```bash
NEXT_PUBLIC_API_URL=https://api.automatos.app
NODE_ENV=production
```

### Frontend Optional Variables
```bash
NEXT_PUBLIC_WS_URL=wss://api.automatos.app/ws
```

## DNS Configuration

### For api.automatos.app
```
Type: CNAME
Name: api
Value: <railway-backend-domain>.railway.app
```

### For ui.automatos.app
```
Type: CNAME
Name: ui
Value: <railway-frontend-domain>.railway.app
```

## Verification

### Check Backend
```bash
curl https://api.automatos.app/health
```

### Check Frontend
```bash
curl https://ui.automatos.app
```

### Check Database Connection
Backend logs should show:
```
✅ PostgreSQL is ready!
✅ Database connected!
```

## Troubleshooting

### Backend Issues
- **Database connection fails**: Check `DATABASE_URL` is set correctly
- **Redis connection fails**: Check `REDIS_URL` is set correctly
- **Port conflicts**: Railway handles ports automatically, no action needed

### Frontend Issues
- **API calls fail**: Verify `NEXT_PUBLIC_API_URL` points to correct backend URL
- **CORS errors**: Backend should allow requests from `ui.automatos.app`
- **Build fails**: Check Node version (should be 18) and dependencies

### Build Issues
- **Dockerfile not found**: Verify Root Directory is set to `automatos-ai`
- **Stage not found**: Railway uses `production` stage by default
- **Missing files**: Ensure all files are committed to GitHub

## Notes

1. **Railway automatically**:
   - Detects Dockerfiles
   - Uses production stage
   - Handles HTTPS/SSL
   - Provides service URLs

2. **Manual configuration needed**:
   - Root directory (must be `automatos-ai`)
   - Environment variables
   - Custom domains
   - DNS records

3. **Service dependencies**:
   - Backend depends on PostgreSQL and Redis
   - Frontend depends on Backend (via API URL)
