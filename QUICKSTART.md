# Automatos AI - Quick Start

## 🚀 Get Started in 3 Steps

### 1. Start the Platform
```bash
docker-compose up --build
```

**That's it!** No `.env` file needed. Infrastructure uses secure defaults.

### 2. Access the Platform
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Add Your API Keys (Optional)

Go to **Settings > Credentials** and add:

#### OpenAI API Key
- **Type**: OpenAI API
- **Name**: `development_openai` (or any name)
- **Environment**: development
- **Fields**:
  - API Key: `sk-...`
  - Organization ID: (optional)
  - Base URL: (optional, defaults to https://api.openai.com/v1)

#### Anthropic API Key
- **Type**: Anthropic API
- **Name**: `development_anthropic` (or any name)
- **Environment**: development
- **Fields**:
  - API Key: `sk-ant-...`
  - Base URL: (optional)

## 📦 What You Get

- **PostgreSQL**: Database with pgvector for embeddings
- **Redis**: Cache and session store
- **FastAPI**: Backend API (Python 3.11)
- **Next.js**: Frontend UI (React 18)
- **416 Credential Types**: Pre-configured credential types for all major services
- **Settings UI**: Manage all credentials, API keys, and system settings

## 🔧 Infrastructure Defaults

The platform uses these defaults (no configuration needed):

| Service | Default | Port |
|---------|---------|------|
| PostgreSQL | `orchestrator_db` / `postgres` / `automatos_dev_pass` | 5432 |
| Redis | Password: `automatos_redis_dev` | 6379 |
| Backend | API Key: `dev_api_key_change_in_production` | 8000 |
| Frontend | - | 3000 |

⚠️ **Production**: Change these defaults via Settings UI or environment variables.

## 🎯 Optional: Monitoring & Admin Tools

### Start with Monitoring (Prometheus + Grafana)
```bash
docker-compose --profile monitoring up
```
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090

### Start with Admin Tools (Adminer)
```bash
docker-compose --profile all up
```
- **Adminer**: http://localhost:8080 (Database GUI)

## 🛑 Stop Everything
```bash
docker-compose down
```

## 🗑️ Clean Up (Remove all data)
```bash
docker-compose down -v --rmi all
```

## 📖 Next Steps

1. **Create an Agent**: Go to Agents tab
2. **Upload Documents**: Go to Knowledge tab
3. **Build a Workflow**: Go to Workflows tab
4. **Chat with Your Data**: Go to Chatbot tab

## 🆘 Need Help?

- **Logs**: `docker-compose logs -f backend`
- **Database**: Use Adminer at http://localhost:8080
- **API Reference**: http://localhost:8000/docs

All credentials are encrypted in the database. API keys added via Settings UI are immediately available to the platform.

