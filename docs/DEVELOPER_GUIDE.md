---
title: Developer Onboarding Guide
---

# Developer Onboarding Guide - Automatos AI

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Git
- PostgreSQL client (optional, for debugging)

### 1. Clone & Setup
```bash
git clone <repository-url>
cd automatos-ai/orchestrator
cp .env.example .env  # Configure your environment
```

### 2. Environment Configuration
Edit `.env` file:
```env
# Database
DATABASE_URL=postgresql://automatos_user:automatos_pass@localhost:5432/automatos_ai

# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Service Configuration
MCP_PORT=8001
REDIS_PASSWORD=redis_password_123
```

### 3. Start Development Environment
```bash
# Start all services
docker compose up -d

# Check service health
docker compose ps
curl http://localhost:8000/api/system/health
```

### 4. Verify Installation
```bash
# Test core endpoints
curl http://localhost:8000/api/agents/types
curl http://localhost:8000/api/agents/stats
curl http://localhost:8000/api/system/metrics
```

## 📁 Code Organization

### Adding New API Endpoints

1. **Create Router** (`orchestrator/api/your_module.py`)
```python
from fastapi import APIRouter, Depends
from database.database import get_db
from database.models import YourModel

router = APIRouter(prefix=/api/your-module, tags=[your-module])

@router.get(/)
async def list_items(db: Session = Depends(get_db)):
    return db.query(YourModel).all()
```

2. **Register Router** (`main.py`)
```python
from api.your_module import router as your_router
app.include_router(your_router)
```

### Adding Database Models

1. **Define Model** (`orchestrator/database/models.py`)
```python
class YourModel(Base):
    __tablename__ = your_table
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
```

2. **Create Migration**
```bash
docker compose exec backend_api alembic revision --autogenerate -m Add your_table
docker compose exec backend_api alembic upgrade head
```

### Adding Services

1. **Create Service** (`orchestrator/services/your_service.py`)
```python
import logging

logger = logging.getLogger(__name__)

class YourService:
    def __init__(self):
        self.config = {}
    
    async def process(self, data):
        logger.info(Processing data)
        return {processed: data}
```

2. **Use in API**
```python
from services.your_service import YourService

service = YourService()
result = await service.process(input_data)
```

## 🔧 Development Workflow

### Local Development
```bash
# Make code changes
# Rebuild container
docker compose build backend_api

# Restart service
docker compose restart backend_api

# View logs
docker compose logs -f backend_api
```

### Database Operations
```bash
# Connect to database
docker compose exec postgres psql -U automatos_user -d automatos_ai

# Run migrations
docker compose exec backend_api alembic upgrade head

# Create new migration
docker compose exec backend_api alembic revision --autogenerate -m Description
```

### Testing
```bash
# Run unit tests
docker compose exec backend_api python -m pytest tests/

# Test specific endpoint
curl -X POST http://localhost:8000/api/agents/ \
  -H Content-Type: application/json \
  -d '{name: Test Agent, agent_type: code_architect}'
```

## 📊 Key Development Patterns

### 1. Error Handling
```python
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

@router.post(/)
async def create_item(item_data: ItemCreate, db: Session = Depends(get_db)):
    try:
        # Business logic
        result = process_item(item_data)
        return result
    except ValueError as e:
        logger.error(fValidation error: {e})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(fUnexpected error: {e})
        raise HTTPException(status_code=500, detail=Internal server error)
```

### 2. Database Sessions
```python
@router.post(/)
async def create_item(item_data: ItemCreate, db: Session = Depends(get_db)):
    try:
        item = Item(**item_data.dict())
        db.add(item)
        db.commit()
        db.refresh(item)
        return item
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. WebSocket Events
```python
from services.websocket_manager import manager

async def notify_clients(event_type: str, data: dict):
    await manager.broadcast({
        type: event_type,
        data: data,
        timestamp: datetime.utcnow().isoformat()
    })
```

### 4. Logging Standards
```python
import logging

logger = logging.getLogger(__name__)

# Info for normal operations
logger.info(fProcessing request for agent {agent_id})

# Warning for recoverable issues
logger.warning(fRetrying failed operation: {operation})

# Error for exceptions
logger.error(fFailed to process: {error}, exc_info=True)
```

## 🧪 Testing Guidelines

### Unit Test Structure
```python
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

def test_create_agent_success():
    # Arrange
    agent_data = {name: Test, agent_type: code_architect}
    
    # Act
    response = client.post(/api/agents/, json=agent_data)
    
    # Assert
    assert response.status_code == 200
    assert response.json()[name] == Test
```

### Integration Tests
```python
def test_agent_workflow_integration():
    # Create agent
    agent_response = client.post(/api/agents/, json=agent_data)
    agent_id = agent_response.json()[id]
    
    # Execute agent
    exec_response = client.post(f/api/agents/{agent_id}/execute)
    assert exec_response.status_code == 200
```

## 🐛 Debugging Tips

### Common Issues

1. **Import Errors**
   - Check file paths in `src/` directory
   - Verify `__init__.py` files exist
   - Use absolute imports: `from database.models import Agent`

2. **Database Connection**
   - Verify PostgreSQL container is running
   - Check `DATABASE_URL` in `.env`
   - Run `docker compose logs postgres`

3. **API Key Issues**
   - Ensure keys are set in `.env`
   - Don't commit keys to Git
   - Check key format and permissions

### Debugging Commands
```bash
# Check container status
docker compose ps

# View application logs
docker compose logs -f backend_api

# Database connection test
docker compose exec postgres pg_isready

# Container shell access
docker compose exec backend_api /bin/bash

# Python REPL in container
docker compose exec backend_api python
```

## 📈 Performance Guidelines

### Database Optimization
```python
# Use eager loading for relationships
agents = db.query(Agent).options(joinedload(Agent.skills)).all()

# Implement pagination
query = db.query(Agent).offset(skip).limit(limit)

# Use database functions for aggregation
total = db.query(func.count(Agent.id)).scalar()
```

### Async Best Practices
```python
# Use async for I/O operations
async def fetch_external_data():
    async with httpx.AsyncClient() as client:
        response = await client.get(https://api.example.com)
        return response.json()

# Avoid blocking operations in async functions
# BAD: time.sleep(1)
# GOOD: await asyncio.sleep(1)
```

## 🔄 Git Workflow

### Branch Strategy
```bash
# Feature development
git checkout -b feature/your-feature-name
git commit -m feat: add new endpoint for X
git push origin feature/your-feature-name

# Bug fixes
git checkout -b fix/issue-description
git commit -m fix: resolve database connection timeout
```

### Commit Message Format
```
type(scope): description

Types: feat, fix, docs, style, refactor, test, chore
Examples:
- feat(api): add bulk agent creation endpoint
- fix(database): resolve connection timeout issue
- docs(readme): update installation instructions
```

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **SQLAlchemy Guide**: https://docs.sqlalchemy.org/
- **Docker Compose Reference**: https://docs.docker.com/compose/
- **PostgreSQL pgvector**: https://github.com/pgvector/pgvector

## 🆘 Getting Help

1. **Check Logs**: Always start with `docker compose logs`
2. **Review Tests**: Look at existing tests for patterns
3. **Documentation**: Refer to ARCHITECTURE.md and FLOW_DIAGRAMS.md
4. **Issues**: Create detailed GitHub issues with reproduction steps

Happy coding! 🎉
