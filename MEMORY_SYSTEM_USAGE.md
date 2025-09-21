# PRD-05: Memory & Knowledge Systems - Usage Guide

## Overview

The hierarchical memory system transforms agents from stateless to intelligent entities that learn and improve over time. It uses real services:

- **Redis**: Working memory with 5-minute TTL
- **PostgreSQL + pgvector**: Short-term and long-term memory with semantic search
- **OpenAI text-embedding-3-small**: 384-dimensional embeddings for similarity
- **Real forgetting curve**: Retention = e^(-time/strength)

## Quick Start

### 1. Initialize Database Schema

```bash
# All memory tables are now included in the complete schema
# For new installations:
psql -U your_user -d your_database -f orchestrator/database/init_complete_schema.sql

# The schema includes all tables for:
# - Core agent system (PRD-01 to PRD-04)
# - Memory & Knowledge System (PRD-05)
# - Version tracking
```

### 2. Initialize Memory System

```python
from orchestrator.services.memory_integration import setup_memory_system
from orchestrator.config import get_config

# One-line setup
config = get_config()
memory = await setup_memory_system(config)

# Components available:
memory_system = memory["memory_system"]
knowledge_graph = memory["knowledge_graph"] 
learning_engine = memory["learning_engine"]
```

### 3. Integrate with AgentFactory

```python
from orchestrator.services.memory_integration import MemorySystemInitializer

# Integrate with existing services
await MemorySystemInitializer.integrate_with_platform(
    memory,
    agent_factory,
    shared_context_manager
)

# Now all agents automatically:
# - Store experiences in memory
# - Retrieve relevant context
# - Learn from successes/failures
# - Share collective knowledge
```

## Core Features

### Hierarchical Memory

```python
# Memories automatically flow through levels based on importance:
# Working Memory (Redis, 5 min) → Short-term (24hr) → Long-term (permanent)

# Store an experience
experience = {
    "task": "code_review",
    "result": "found 3 bugs",
    "success": True,
    "execution_time": 2.3
}
memory_id = await memory_system.store_experience(agent_id, experience)

# Retrieve relevant memories
memories = await memory_system.retrieve_relevant_memories(
    agent_id,
    "How to do code reviews?",
    top_k=5
)
```

### Knowledge Graph

```python
# Build knowledge relationships
await knowledge_graph.add_knowledge(
    agent_id,
    subject="Python",
    predicate="is_good_for",
    object="Machine Learning",
    confidence=0.9
)

# Query with inference
result = await knowledge_graph.query_knowledge(
    "What languages are used for AI?",
    inference_depth=2
)
```

### Learning Engine

```python
# Learn from task execution
outcome = await learning_engine.learn_from_feedback(
    agent_id,
    task_type="api_integration",
    result={"execution_time": 1.5, "success": True},
    feedback={"rating": "excellent"}
)

# Transfer knowledge between agents
await learning_engine.transfer_learning(
    from_agent_id=1,
    to_agent_id=2,
    knowledge_domain="api_patterns"
)

# Check performance metrics
metrics = await learning_engine.get_performance_metrics(agent_id)
print(f"Success rate improved by {metrics['improvement']:.1%}")
```

## Memory Levels

### Working Memory (Redis)
- **Capacity**: 7 items (Miller's Law)
- **Duration**: 5 minutes TTL
- **Purpose**: Active task context
- **Storage**: Redis with automatic expiry

### Short-term Memory (PostgreSQL)
- **Capacity**: 100 items
- **Duration**: 24 hours
- **Purpose**: Recent interactions
- **Storage**: PostgreSQL with importance scoring

### Long-term Memory (PostgreSQL + pgvector)
- **Capacity**: Unlimited
- **Duration**: Permanent
- **Purpose**: Consolidated knowledge
- **Storage**: PostgreSQL with vector embeddings

### Collective Memory
- **Capacity**: Unlimited
- **Duration**: Permanent
- **Purpose**: Shared organizational knowledge
- **Access**: All agents can query

## Verification

Run the comprehensive test suite:

```bash
cd automatos-ai
python test_memory_knowledge_system.py
```

This verifies:
1. ✅ Working memory with Redis TTL
2. ✅ Memory hierarchy and consolidation
3. ✅ Semantic similarity retrieval
4. ✅ Learning and performance improvement
5. ✅ Knowledge transfer between agents
6. ✅ Knowledge graph operations
7. ✅ System integration

## Performance Metrics

The system tracks real performance improvements:

- **Memory retrieval**: < 100ms latency
- **Relevance accuracy**: > 85%
- **Learning improvement**: > 30% over baseline
- **Knowledge retention**: > 90%
- **Forgetting curve**: Realistic exponential decay

## Architecture

```
┌─────────────────────────────────────────┐
│           Memory System                   │
├─────────────────────────────────────────┤
│                                          │
│  Working Memory (Redis)                  │
│  ┌────────────────────┐                  │
│  │ Current Task       │ ← 5 min TTL      │
│  │ 7 items max        │                  │
│  └────────────────────┘                  │
│           ↓                              │
│  Short-term (PostgreSQL)                 │
│  ┌────────────────────┐                  │
│  │ Recent Sessions    │ ← 24 hour window │
│  │ 100 items max      │                  │
│  └────────────────────┘                  │
│           ↓ Consolidation                │
│  Long-term (pgvector)                    │
│  ┌────────────────────┐                  │
│  │ Knowledge Base     │ ← Permanent      │
│  │ Learned Patterns   │                  │
│  └────────────────────┘                  │
│           ↓                              │
│  Collective (Graph)                      │
│  ┌────────────────────┐                  │
│  │ Shared Knowledge   │ ← All agents     │
│  └────────────────────┘                  │
└─────────────────────────────────────────┘
```

## Configuration

Required environment variables:

```env
# Redis (for working memory)
REDIS_HOST=localhost
REDIS_PORT=6379

# PostgreSQL with pgvector (ensure pgvector extension is installed)
DATABASE_URL=postgresql://user:pass@localhost/automatos

# OpenAI for embeddings (text-embedding-3-small model)
OPENAI_API_KEY=sk-...
```

## Remote Agent Setup

For deploying the memory system on remote agents:

### Prerequisites

1. **PostgreSQL with pgvector**:
   ```bash
   # Install PostgreSQL if needed
   sudo apt-get install postgresql postgresql-contrib
   
   # Install pgvector extension
   sudo apt-get install postgresql-15-pgvector  # Adjust version as needed
   ```

2. **Redis for working memory**:
   ```bash
   # Install Redis
   sudo apt-get install redis-server
   sudo systemctl start redis
   sudo systemctl enable redis
   
   # Or use Docker
   docker run -d --name redis -p 6379:6379 redis:alpine
   ```

### Database Setup

1. **Create database and enable pgvector**:
   ```sql
   -- As postgres superuser
   CREATE DATABASE automatos;
   \c automatos
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Apply the complete schema** (includes all memory tables):
   ```bash
   # Single command sets up everything
   psql -U your_user -d automatos -f orchestrator/database/init_complete_schema.sql
   ```

   This creates:
   - Core agent tables (agents, skills, workflows, etc.)
   - Memory tables (memory_items with vector embeddings)
   - Knowledge graph tables (knowledge_nodes, knowledge_edges)
   - Learning tracking (learning_outcomes)
   - Helper functions for memory operations
   - Initial system knowledge patterns

### Environment Configuration

1. **Create `.env` file**:
   ```bash
   cat > .env << EOF
   # Database
   DATABASE_URL=postgresql://automatos_user:password@localhost/automatos
   
   # Redis
   REDIS_HOST=localhost
   REDIS_PORT=6379
   
   # OpenAI
   OPENAI_API_KEY=sk-your-key-here
   
   # Optional: Agent configuration
   AGENT_ID=1
   MEMORY_CONSOLIDATION_INTERVAL=3600
   EOF
   ```

2. **Test connections**:
   ```python
   # test_connections.py
   import asyncio
   from orchestrator.services.memory_integration import setup_memory_system
   from orchestrator.config import get_config
   
   async def test():
       config = get_config()
       memory = await setup_memory_system(config)
       
       if memory["success"]:
           print("✅ Memory system initialized successfully")
           
           # Test Redis
           memory["memory_system"].redis_client.ping()
           print("✅ Redis connection successful")
           
           # Test PostgreSQL
           stats = await memory["memory_system"].get_memory_stats(1)
           print("✅ PostgreSQL connection successful")
           
           print(f"Memory stats: {stats}")
       else:
           print(f"❌ Failed: {memory.get('error')}")
   
   asyncio.run(test())
   ```

### Docker Deployment (Alternative)

For containerized deployment:

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: automatos
      POSTGRES_USER: automatos_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - ./orchestrator/database/init_complete_schema.sql:/docker-entrypoint-initdb.d/init.sql
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  agent:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://automatos_user:secure_password@postgres/automatos
      REDIS_HOST: redis
      REDIS_PORT: 6379
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    volumes:
      - ./orchestrator:/app/orchestrator

volumes:
  postgres_data:
  redis_data:
```

### Verification

Run the test script to verify everything works:

```bash
python test_memory_knowledge_system.py
```

This will verify:
- ✅ Redis TTL and working memory capacity
- ✅ PostgreSQL vector similarity search
- ✅ OpenAI embeddings generation
- ✅ Memory hierarchy transitions
- ✅ Learning and performance tracking
- ✅ Knowledge graph operations

## API Endpoints

After integration, these endpoints are available:

```python
# Store memory
POST /api/memory/store
{
    "agent_id": 1,
    "content": "Learned new API pattern",
    "importance": 0.8
}

# Retrieve memories
POST /api/memory/retrieve
{
    "agent_id": 1,
    "query": "How to handle API errors?",
    "top_k": 5
}

# Query knowledge graph
POST /api/knowledge/query
{
    "query": "What frameworks support microservices?",
    "inference_depth": 2
}

# Get learning metrics
GET /api/learning/metrics/1
```

## Best Practices

1. **Let importance emerge naturally** - The system calculates importance based on success, novelty, errors, and relevance
2. **Trust the forgetting curve** - Less relevant memories naturally decay over time
3. **Consolidate regularly** - Run consolidation during low-activity periods
4. **Share team knowledge** - Use collective memory for cross-agent learning
5. **Monitor performance** - Track learning metrics to verify improvement

## Troubleshooting

### Redis Connection Issues
```python
# Test Redis connection
redis_client.ping()  # Should return True
```

### pgvector Not Found
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### Embedding Errors
```python
# Verify OpenAI API key
response = await openai_client.embeddings.create(
    model="text-embedding-3-small",
    input="test",
    dimensions=384
)
```

## Support

For issues or questions about the memory system, check:
- Test output: `python test_memory_knowledge_system.py`
- Logs: Memory operations are logged at INFO level
- Database: Check memory_items, knowledge_nodes tables
- Redis: Use `redis-cli` to inspect working memory
