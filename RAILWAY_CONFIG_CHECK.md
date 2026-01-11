# Railway Configuration Check

## Problem
Railway can't find `automatos-ai/orchestrator/` directory. This happens because Railway's **Root Directory** setting determines the build context.

## Root Directory Setting Impact

### If Root Directory = `automatos-ai` (build context is `automatos-ai/`)
- Dockerfile paths should be: `orchestrator/Dockerfile`, `frontend/Dockerfile`
- COPY paths should be: `orchestrator/requirements.txt` (NOT `automatos-ai/orchestrator/requirements.txt`)
- Build context = `automatos-ai/` directory

### If Root Directory = NOT SET (build context is repo root)
- Dockerfile paths should be: `automatos-ai/orchestrator/Dockerfile`, `automatos-ai/frontend/Dockerfile`
- COPY paths should be: `automatos-ai/orchestrator/requirements.txt`
- Build context = repository root (`Automatos-AI-Platform/`)

## Current Dockerfile Issue

The Dockerfiles use paths like `COPY automatos-ai/orchestrator/requirements.txt` which assumes:
- Build context = repository root
- Root Directory = NOT SET (or repo root)

## Solution: Check Your Railway Settings

**In Railway Dashboard:**

### Backend Service:
1. Settings → **Root Directory**: Check what it's set to
   - If `automatos-ai`: Change Dockerfile paths to use `orchestrator/` (no `automatos-ai/` prefix)
   - If NOT SET or repo root: Current Dockerfiles are correct

2. Settings → **Dockerfile Path**: Should be:
   - `automatos-ai/orchestrator/Dockerfile` (if Root Directory not set)
   - `orchestrator/Dockerfile` (if Root Directory = `automatos-ai`)

### Frontend Service:
1. Settings → **Root Directory**: Same as backend
2. Settings → **Dockerfile Path**: Should be:
   - `automatos-ai/frontend/Dockerfile` (if Root Directory not set)
   - `frontend/Dockerfile` (if Root Directory = `automatos-ai`)

## Recommendation

**Option A**: Remove Root Directory setting (use repo root as build context)
- Current Dockerfiles will work as-is
- Dockerfile paths: `automatos-ai/orchestrator/Dockerfile`, `automatos-ai/frontend/Dockerfile`

**Option B**: Set Root Directory to `automatos-ai`
- Need to update Dockerfiles to remove `automatos-ai/` prefix from all COPY paths
- Dockerfile paths: `orchestrator/Dockerfile`, `frontend/Dockerfile`
