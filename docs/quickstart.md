# ⚡ Quick Start Guide

> **"From Zero to Multi-Agent Orchestration in 5 Minutes"**

---

## 🏁 Prerequisites

- **Docker Desktop** (running)
- **Git**
- **OpenAI API Key** (or Anthropic/etc.)

That's it. We handle the rest.

---

## 🚀 1. Clone & Configure

```bash
# Clone the repository
git clone https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai

# Set up environment
cp .env.example .env

# Add your API key (essential!)
# Open .env and set:
# OPENAI_API_KEY=sk-...
```

---

## 🐳 2. Start the Platform

We use Docker Compose to spin up the entire stack (API, Database, Redis, Frontend).

```bash
# Start everything in detached mode
docker-compose up -d
```

**What's happening?**
- 🟢 **PostgreSQL** starts (port 5432)
- 🔴 **Redis** starts (port 6379)
- 🔵 **Orchestrator API** starts (port 8000)
- 🟣 **Frontend** starts (port 3000)

*Wait about 30 seconds for the database to initialize.*

---

## ✅ 3. Verify Installation

Check if the system is breathing:

```bash
curl http://localhost:8000/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "services": {
    "database": "connected",
    "redis": "connected"
  }
}
```

---

## 🤖 4. Create Your First Agent

Let's create a "Research Agent" that can search the web.

```bash
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Specialist",
    "agent_type": "custom",
    "provider": "openai",
    "model": "gpt-4",
    "system_prompt": "You are an expert researcher. Be concise.",
    "skills": ["web_search", "summarization"]
  }'
```

---

## 💬 5. Chat With Your Agent

Now, let's ask it something.

**Option A: Use the UI**
Open [http://localhost:3000](http://localhost:3000) in your browser.

**Option B: Use the API (Streaming)**

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the latest breakthroughs in fusion energy?",
    "stream": true
  }'
```

*Watch the tokens stream in real-time!* 🌊

---

## 🛠️ Troubleshooting

**"Database connection failed"**
- Wait 10 more seconds. Postgres takes a moment to wake up.
- Check logs: `docker-compose logs -f db`

**"API Key missing"**
- Did you edit `.env`?
- Restart the container: `docker-compose restart orchestrator`

**"Port already in use"**
- Stop other services on 8000/3000/5432.
- Or change ports in `.env`.

---

## 📚 Next Steps

- **[Developer Guide](DEVELOPER_GUIDE.md)** - Build your own modules
- **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** - How it works
- **[API Reference](API_REFERENCE.md)** - Full endpoint list

**Welcome to the future of AI orchestration!** 🎉
