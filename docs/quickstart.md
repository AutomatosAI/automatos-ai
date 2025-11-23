---
title: Quick Start Guide
description: Get Automatos AI running in 10 minutes - comprehensive setup guide
---

# ⚡ Quick Start Guide

*Get Automatos AI running locally in under 10 minutes*

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [First Run](#first-run)
4. [Accessing Services](#accessing-services)
5. [Adding API Keys](#adding-api-keys)
6. [Verification](#verification)
7. [Next Steps](#next-steps)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Minimum Version | Check Command |
|----------|----------------|---------------|
| **Docker** | 24.0+ | `docker --version` |
| **Docker Compose** | 2.20+ | `docker-compose --version` |
| **Git** | 2.0+ | `git --version` |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **RAM** | 4 GB | 8+ GB |
| **Storage** | 10 GB free | 20+ GB free |
| **OS** | Linux, macOS, Windows (with WSL2) | - |

### Optional (But Recommended)

- **OpenAI API Key** ([get one here](https://platform.openai.com/api-keys))
  - Required for AI agent features
  - Platform works without it for testing
- **Anthropic API Key** ([get one here](https://console.anthropic.com/settings/keys))
  - Optional, for Claude models

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai
```

**Expected output:**
```
Cloning into 'automatos-ai'...
remote: Counting objects: 1234, done.
...
```

### Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env
```

**Review and customize** (optional):
```bash
# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

**Key settings to review:**

```bash
# Database passwords (change for production!)
POSTGRES_PASSWORD=automatos_dev_pass
REDIS_PASSWORD=automatos_redis_dev

# API Keys (add when ready)
OPENAI_API_KEY=     # Add your key here
ANTHROPIC_API_KEY=  # Optional

# Ports (change if needed)
API_PORT=8000       # Backend API port
FRONTEND_PORT=3000  # Frontend port
```

---

## First Run

### Start All Services

```bash
docker-compose up
```

**What happens:**
1. 🏗️ **Building images** (first run only, ~3-5 minutes)
   - Backend Python image
   - Frontend Next.js image
   - Pulling PostgreSQL, Redis images

2. 🗄️ **Initializing database**
   - Creating 71 tables
   - Loading 416 credential types
   - Loading 416 MCP tools

3. 🚀 **Starting services**
   - PostgreSQL database
   - Redis cache
   - Backend API (FastAPI)
   - Frontend (Next.js)

**Expected output (abbreviated):**
```
[+] Building 45.2s (23/23) FINISHED
...
automatos_postgres   | database system is ready to accept connections
automatos_redis      | Ready to accept connections
automatos_backend    | ✅ PostgreSQL is ready!
automatos_backend    | 📦 Checking seed data...
automatos_backend    | 📥 Loading seed data...
automatos_backend    | ✅ Seed data loaded successfully!
automatos_backend    | 🚀 Starting Backend Application
automatos_backend    | INFO: Uvicorn running on http://0.0.0.0:8000
automatos_frontend   | ready - started server on 0.0.0.0:3000
```

### Run in Background (Detached Mode)

```bash
# Stop with Ctrl+C first, then:
docker-compose up -d
```

**Check status:**
```bash
docker-compose ps
```

**Expected output:**
```
NAME                  STATUS              PORTS
automatos_postgres    Up (healthy)        0.0.0.0:5432->5432/tcp
automatos_redis       Up (healthy)        0.0.0.0:6379->6379/tcp
automatos_backend     Up (healthy)        0.0.0.0:8000->8000/tcp
automatos_frontend    Up                  0.0.0.0:3000->3000/tcp
```

---

## Accessing Services

### 🌐 Frontend Dashboard

**URL:** http://localhost:3000

**Features:**
- Agent management
- Workflow orchestration
- Document processing
- Real-time analytics
- System monitoring

### 📚 API Documentation

**URL:** http://localhost:8000/docs

**Interactive API:**
- Browse 373+ endpoints
- Test API calls directly
- View request/response schemas
- Download OpenAPI spec

### ❤️ Health Check

**URL:** http://localhost:8000/health

**Expected response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "database": "connected",
  "redis": "connected",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Optional Services

#### Admin Tools (Database Management)

```bash
# Start with all tools
docker-compose --profile all up -d
```

**Access:**
- **Adminer**: http://localhost:8080
  - System: PostgreSQL
  - Server: postgres
  - Username: postgres
  - Password: automatos_dev_pass
  - Database: orchestrator_db

---

## Adding API Keys

### Option 1: Via Environment File (Recommended)

```bash
# Edit .env file
nano .env

# Add your keys:
OPENAI_API_KEY=sk-your-actual-key-here
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here

# Restart backend
docker-compose restart backend
```

### Option 2: Via Web UI

1. Navigate to http://localhost:3000/settings
2. Click **Credentials** tab
3. Add **OpenAI** credential:
   - Name: `development_openai`
   - Type: `openai`
   - API Key: `sk-your-actual-key-here`
4. Add **Anthropic** credential (optional):
   - Name: `development_anthropic`
   - Type: `anthropic`
   - API Key: `sk-ant-your-actual-key-here`

---

## Verification

### 1. Check All Services Running

```bash
docker-compose ps
```

All services should show `Up` or `Up (healthy)`.

### 2. Test Backend API

```bash
# Health check
curl http://localhost:8000/health

# List agents
curl http://localhost:8000/api/agents | jq

# System metrics
curl http://localhost:8000/api/system/metrics | jq
```

### 3. Test Frontend

```bash
# Should return HTML
curl http://localhost:3000

# Or open in browser:
open http://localhost:3000  # macOS
# or visit in your browser
```

### 4. Check Database

```bash
# Connect to database
docker exec -it automatos_postgres psql -U postgres -d orchestrator_db

# Run query
SELECT COUNT(*) FROM credential_types;
# Should show: 416

SELECT COUNT(*) FROM mcp_tools;
# Should show: 416

# Exit
\q
```

### 5. Check Logs

```bash
# View all logs
docker-compose logs

# Follow backend logs
docker-compose logs -f backend

# Check for errors
docker-compose logs | grep -i error
```

---

## Next Steps

### 🤖 Create Your First Agent

#### Via API:
```bash
curl -X POST http://localhost:8000/api/v1/agents/create-specialized \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First Agent",
    "type": "code_architect",
    "model_config": {
      "provider": "openai",
      "model_id": "gpt-4-turbo-preview",
      "temperature": 0.7
    }
  }'
```

#### Via Web UI:
1. Go to http://localhost:3000/agents
2. Click **Create Agent**
3. Choose agent type
4. Configure settings
5. Click **Create**

### 📚 Upload Documents

1. Navigate to http://localhost:3000/documents
2. Click **Upload**
3. Select PDF, DOCX, or TXT files
4. Wait for processing (vectors generated automatically)
5. Search your documents semantically

### 🔄 Create a Workflow

1. Go to http://localhost:3000/workflows
2. Click **New Workflow**
3. Add workflow steps
4. Connect agents
5. Execute and monitor

---

## Troubleshooting

### Services Won't Start

**Issue:** `docker-compose up` fails

**Solutions:**
```bash
# Check Docker is running
docker info

# Check ports aren't in use
lsof -i:8000  # Backend
lsof -i:3000  # Frontend
lsof -i:5432  # PostgreSQL
lsof -i:6379  # Redis

# Kill processes if needed
kill <PID>

# Clean and restart
docker-compose down -v
docker-compose up
```

### Database Connection Errors

**Issue:** Backend can't connect to PostgreSQL

**Solutions:**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Verify connection
docker exec automatos_postgres pg_isready -U postgres

# Restart database
docker-compose restart postgres backend
```

### Frontend Not Loading

**Issue:** http://localhost:3000 shows error

**Solutions:**
```bash
# Check frontend logs
docker-compose logs frontend

# Common issues:
# 1. Node modules issue - rebuild
docker-compose build --no-cache frontend
docker-compose up -d frontend

# 2. Port in use
lsof -i:3000
kill <PID>
docker-compose restart frontend
```

### Seed Data Not Loading

**Issue:** No credential types or tools in database

**Solutions:**
```bash
# Check if seed data files exist
ls -lh orchestrator/database/*.json

# Manually load seed data
docker exec automatos_backend python database/load_seed_data.py

# Check database
docker exec -it automatos_postgres psql -U postgres -d orchestrator_db -c "SELECT COUNT(*) FROM credential_types;"
```

### API Keys Not Working

**Issue:** AI features failing with authentication errors

**Solutions:**
```bash
# Verify keys are set
docker exec automatos_backend env | grep API_KEY

# Test OpenAI key directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_KEY_HERE"

# Restart backend after adding keys
docker-compose restart backend
```

### Out of Memory

**Issue:** Services crashing with OOM errors

**Solutions:**
```bash
# Check memory usage
docker stats

# Increase Docker memory (Docker Desktop)
# Settings > Resources > Memory > 8GB

# Or reduce services
docker-compose up -d postgres redis backend
# Run frontend locally: cd frontend && npm run dev
```

### Port Already in Use

**Issue:** `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solutions:**
```bash
# Find what's using the port
lsof -i:8000

# Kill the process
kill -9 <PID>

# Or change port in .env
echo "API_PORT=8001" >> .env
docker-compose up -d
```

---

## Useful Commands

### Docker Compose

```bash
# Start services
docker-compose up                    # Foreground (see logs)
docker-compose up -d                 # Background (detached)

# Stop services
docker-compose down                  # Stop and remove containers
docker-compose down -v               # Also remove volumes (fresh start)

# View logs
docker-compose logs                  # All services
docker-compose logs -f backend       # Follow backend logs
docker-compose logs --tail=100       # Last 100 lines

# Restart services
docker-compose restart               # All services
docker-compose restart backend       # Specific service

# Rebuild
docker-compose build                 # Rebuild all images
docker-compose build --no-cache      # Clean rebuild
```

### Database Operations

```bash
# Connect to database
docker exec -it automatos_postgres psql -U postgres -d orchestrator_db

# Run SQL file
docker exec -i automatos_postgres psql -U postgres -d orchestrator_db < script.sql

# Backup database
docker exec automatos_postgres pg_dump -U postgres orchestrator_db > backup.sql

# Restore database
cat backup.sql | docker exec -i automatos_postgres psql -U postgres orchestrator_db
```

### Cleanup

```bash
# Remove all data (fresh start)
docker-compose down -v

# Remove Docker images
docker-compose down --rmi all

# Full cleanup
docker system prune -a --volumes
```

---

## Production Deployment

For production deployments, see:
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production setup
- **[Security Guide](SECURITY.md)** - Security hardening

---

## Getting Help

- **📖 Documentation**: https://docs.automatos.ai
- **💬 GitHub Issues**: https://github.com/AutomatosAI/automatos-ai/issues
- **🤝 Community**: https://discord.gg/automatos-ai
- **📧 Email**: support@automatos.ai

---

**Built with ❤️ to get you started fast**

*Last updated: January 2025*
