# PRD-04 Environment Setup Guide

## ✅ Code Updates Complete

### 1. **Fixed agent_factory.py test code**
- Changed from `AgentType.CODE_ARCHITECT` enum to string `"code_architect"`
- Fixed all test agent creation to use `AgentMetadata` objects
- Test code now works properly

### 2. **Updated database models** ✅ FIXED UUID ISSUES
- Added all PRD-04 tables to `database/models.py`:
  - `agent_messages` - Inter-agent message tracking
  - `shared_contexts` - Team shared memory (UUID primary key)
  - `context_permissions` - Access control
  - `collaboration_sessions` - Team problem solving records (UUID primary key)
  - `collaboration_proposals` - Agent proposals (UUID foreign keys)
  - `consensus_votes` - Voting records (UUID foreign keys)
  - `message_broadcasts` - Team message tracking
- **FIXED**: All session_id and shared_context_id fields now use UUID type instead of String(255)
- **FIXED**: Added PostgreSQL UUID import to models.py

### 3. **Environment variable usage**
- `inter_agent_communication.py` now properly loads from .env
- Redis connection builds from `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
- Database URL from `DATABASE_URL`
- **ADDED**: REDIS_HOST=127.0.0.1 to .env configuration

## 📋 Your .env Configuration

Your current .env looks **GOOD**! You have all the required variables:

### ✅ Required for PRD-04:
```bash
# Database (PostgreSQL)
DATABASE_URL=postgresql://postgres:secure_password_123@127.0.0.1:5432/orchestrator_db  ✓

# Redis (for messaging)
REDIS_PASSWORD=redis_password_123  ✓
REDIS_PORT=6379  ✓
# Add this one:
REDIS_HOST=127.0.0.1  # or localhost

# LLM (for agents)
OPENAI_API_KEY=sk-proj-...  ✓
LLM_PROVIDER=openai  ✓
LLM_MODEL=gpt-4  ✓
```

### 🔧 Add to your .env:
```bash
# Redis host (missing from your config)
REDIS_HOST=127.0.0.1
```

## 🗄️ Database Setup

Run these commands to create the new tables:

```bash
# Connect to your PostgreSQL
psql -U postgres -d orchestrator_db

# The tables will be created automatically when you run the app
# Or you can manually create them using SQLAlchemy:
python -c "
from orchestrator.database.models import Base
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:secure_password_123@127.0.0.1:5432/orchestrator_db')
Base.metadata.create_all(engine)
print('✓ All tables created')
"
```

## 🚀 Testing the System

1. **Ensure services are running:**
```bash
# Start PostgreSQL
sudo systemctl start postgresql  # or: brew services start postgresql

# Start Redis
redis-server  # or: brew services start redis
```

2. **Test the implementation:**
```bash
cd automatos-ai
python test_inter_agent_communication.py
```

## ⚠️ Security Note

Your .env contains **real API keys** that are exposed:
- `OPENAI_API_KEY` - This is a real key, consider rotating it
- `ANTHROPIC_API_KEY` - This is also real
- `GITHUB_TOKEN` and `FRONTEND_GIT_PAT` - These are real tokens

**Recommendation:** 
1. Rotate these keys/tokens
2. Never commit .env to git
3. Use `.env.example` for documentation

## ✅ Summary

Your environment is **almost ready**! Just:
1. Add `REDIS_HOST=127.0.0.1` to your .env
2. Ensure PostgreSQL and Redis are running
3. Run the database table creation
4. Test with `python test_inter_agent_communication.py`

The inter-agent communication system will then be fully operational with:
- Real Redis pub/sub messaging
- PostgreSQL storage for collaboration data
- Real GPT-4 connections for each agent
- Full audit trail of all agent interactions
