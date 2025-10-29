# Automatos AI Platform - Bootstrap Guide

## Overview

This guide explains how to bootstrap the Automatos AI platform and migrate from `.env` file configuration to secure database-stored credentials.

## Bootstrap Strategy

The platform uses a **smart bootstrap strategy** that solves the "chicken-and-egg" problem:
- 🥚 **Problem**: Database needs credentials, but credentials are stored in database!
- ✅ **Solution**: Start with `.env` file → Setup credentials in UI → Optional: Delete `.env`

## Initial Setup (Bootstrap Phase)

### Step 1: Create `.env` File

Create a `.env` file in the `automatos-ai` directory with minimal infrastructure credentials:

```bash
# Infrastructure (REQUIRED for bootstrap)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=automatos_dev_pass

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=automatos_redis_dev

# LLM Keys (OPTIONAL - can be added via UI later)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2: Start Infrastructure

```bash
cd automatos-ai
docker compose up -d postgres redis
```

### Step 3: Start Backend

```bash
cd orchestrator
uvicorn main:app --host 0.0.0.0 --port 8000
```

✅ **Platform starts successfully** without any LLM keys!

### Step 4: Configure Credentials in UI

1. Open browser: `http://localhost:3000`
2. Navigate to **Settings** → **Credentials**
3. Add your LLM credentials:
   - **development_openai**: OpenAI API key
   - **development_anthropic**: Anthropic API key
   - **development_db**: PostgreSQL credentials (optional)
   - **development_redis**: Redis credentials (optional)

### Step 5: (Optional) Remove `.env` File

Once credentials are in the database, you can delete the `.env` file:

```bash
rm automatos-ai/.env
```

The platform will now use database-stored credentials!

## Credential Resolution Order

### Infrastructure Credentials (Postgres, Redis)

1. ✅ Try database credential store
2. ✅ Fallback to `.env` file (ALWAYS available for bootstrap)
3. ✅ Fallback to environment variables
4. ✅ Fallback to safe defaults

**Result**: Platform always starts, even with empty database!

### LLM Credentials (OpenAI, Anthropic)

1. ✅ Try database credential store
2. ✅ Fallback to `.env` file
3. ✅ Fallback to environment variables
4. ⚠️ Return `None` (don't block startup)
5. ❌ Fail ONLY when LLM features are actually used

**Result**: Platform starts without LLM keys, fails gracefully when features are used!

## Docker Compose Setup

The `docker-compose.yml` file uses `.env` variables for service setup:

```yaml
postgres:
  environment:
    POSTGRES_DB: ${POSTGRES_DB:-orchestrator_db}
    POSTGRES_USER: ${POSTGRES_USER:-postgres}
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-automatos_dev_pass}

redis:
  command: redis-server --requirepass ${REDIS_PASSWORD:-automatos_redis_dev}
```

This ensures infrastructure starts correctly during bootstrap.

## Production Deployment

### Recommended Approach

1. **Start with `.env`**: Deploy platform with infrastructure credentials in `.env`
2. **Setup via UI**: Configure all credentials through the UI
3. **Keep `.env` minimal**: Only infrastructure credentials in `.env` (for disaster recovery)
4. **Rotate secrets**: Use UI to rotate LLM keys, database passwords, etc.

### Security Best Practices

- ✅ **Database credentials encrypted** at rest using Fernet encryption
- ✅ **Audit logging** for all credential access
- ✅ **Role-based access** to credential management
- ✅ **API keys never logged** in plaintext
- ✅ **Environment separation** (development/staging/production)

## Troubleshooting

### Platform won't start

**Error**: `OperationalError: could not connect to server`

**Solution**: 
```bash
# Check if postgres is running
docker compose ps postgres

# Check .env file has correct POSTGRES_* variables
cat automatos-ai/.env | grep POSTGRES
```

### LLM features not working

**Error**: `OpenAI API key not configured`

**Solution**:
1. Go to UI → Settings → Credentials
2. Add `development_openai` credential with your API key
3. Or set `OPENAI_API_KEY` in `.env` file

### Credentials not loading from database

**Check logs**:
```bash
tail -f automatos-ai/backend.log | grep credential
```

Expected output:
```
INFO - Using .env file for PostgreSQL (credential 'development_db' not in database yet)
INFO - OpenAI API key not configured (will fail if LLM features are used)
```

## Migration Guide

### From Hardcoded `.env` to Database Credentials

1. **Audit current `.env` file**:
   ```bash
   cat automatos-ai/.env
   ```

2. **Add each credential via UI**:
   - OpenAI: `development_openai`
   - Anthropic: `development_anthropic`
   - GitHub: `github_main`
   - Other services as needed

3. **Test each service** after adding credentials

4. **Remove from `.env`** (keep only infrastructure):
   ```bash
   # Edit .env to remove LLM keys
   nano automatos-ai/.env
   ```

5. **Restart platform**:
   ```bash
   # Platform should still work using database credentials
   docker compose restart orchestrator
   ```

## Advanced: Custom Credentials

### Adding New Credentials

Via UI:
```
Settings → Credentials → Add New
  Name: my_service_api
  Environment: development
  Type: api_key
  Fields:
    - api_key: your-secret-key
    - api_url: https://api.service.com
```

Via API:
```bash
curl -X POST http://localhost:8000/api/credentials \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_service_api",
    "environment": "development",
    "type": "api_key",
    "decrypted_data": {
      "api_key": "your-secret-key",
      "api_url": "https://api.service.com"
    }
  }'
```

### Using in Code

```python
from services.credential_resolver import get_credential_resolver

resolver = get_credential_resolver()

# Get API key
api_key = resolver.get_credential_field(
    credential_name="my_service_api",
    field_name="api_key",
    fallback_env="MY_SERVICE_API_KEY"
)

# Get all fields as dict
credentials = resolver.get_dict("my_service_api")
api_url = credentials["api_url"]
```

## Summary

✅ **Bootstrap with `.env`** - Platform starts with minimal config
✅ **Configure via UI** - Add LLM keys and other credentials  
✅ **Optional cleanup** - Remove `.env` after database setup
✅ **Graceful failures** - LLM features fail only when used
✅ **Production ready** - Encrypted storage, audit logs, access control

The bootstrap strategy ensures **new users can start immediately** without complex credential setup, while maintaining **security and flexibility** for production deployments!

