# 👨‍💻 Developer Guide

> **"Build the future of AI orchestration."**

---

## 🏗️ Development Philosophy

Automatos AI is built on **Modular Domain-Driven Design**.

- **Everything is a Module**: Features live in `modules/`.
- **Everything is Typed**: Python type hints are mandatory.
- **Everything is Async**: Blocking code is the enemy.
- **Everything is Tested**: If it's not tested, it doesn't exist.

---

## 🛠️ Environment Setup

### Option A: Docker (Recommended for running)
See [Quick Start](quickstart.md). Great for running the stack, harder for active development.

### Option B: Local Python (Recommended for coding)

**1. Prerequisites**
- Python 3.11+
- Redis (running locally on 6379)
- PostgreSQL (running locally on 5432)

**2. Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev tools
pip install pytest ruff black
```

**3. Configuration**
```bash
# Copy env file
cp .env.example .env

# Edit .env to point to local DB/Redis
# DATABASE_URL=postgresql://user:pass@localhost:5432/automatos
# REDIS_URL=redis://localhost:6379/0
```

**4. Run Migrations**
```bash
alembic upgrade head
```

**5. Start Server**
```bash
uvicorn orchestrator.main:app --reload --port 8000
```

---

## 🧩 Module Anatomy

A well-behaved module in `orchestrator/modules/` looks like this:

```
modules/my_feature/
├── __init__.py          # Exports public interface
├── service.py           # Core business logic
├── models.py            # Pydantic models (internal)
├── utils.py             # Helpers
└── README.md            # Documentation
```

**Rules:**
1. **Encapsulation**: Don't import private submodules from other modules. Use the public interface.
2. **Stateless**: Services should be stateless. Use DB/Redis for state.
3. **Async**: All I/O must be `async def`.

---

## 🎓 Tutorials

### Tutorial 1: Adding a New Tool

**Goal**: Create a tool that reverses a string.

1. **Create file**: `modules/tools/implementations/utils/reverse.py`

2. **Implement**:
```python
from modules.tools import tool_registry, ToolCategory

@tool_registry.register(
    category=ToolCategory.CODE_OPS,
    name="reverse_string",
    description="Reverses a given string"
)
async def reverse_string(text: str) -> dict:
    return {"reversed": text[::-1]}
```

3. **That's it.** Restart server. It's live.

### Tutorial 2: Adding an API Endpoint

**Goal**: Add `GET /api/hello`.

1. **Create file**: `orchestrator/api/hello.py`

2. **Implement**:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/hello", tags=["hello"])

@router.get("/")
async def say_hello():
    return {"message": "Hello World"}
```

3. **Register**: Edit `orchestrator/main.py`
```python
from orchestrator.api.hello import router as hello_router
app.include_router(hello_router)
```

### Tutorial 3: Adding a Background Consumer

**Goal**: Process "Heavy Tasks".

1. **Create Service**: `consumers/heavy_task/processor.py`
```python
async def process_heavy_task(task_id: int):
    # Do work...
    pass
```

2. **Trigger it**: In your API endpoint
```python
from fastapi import BackgroundTasks

@router.post("/heavy")
async def trigger(bg_tasks: BackgroundTasks):
    bg_tasks.add_task(process_heavy_task, 123)
```

---

## 🧪 Testing

We use **pytest**.

```bash
# Run all tests
pytest

# Run specific test
pytest tests/api/test_agents.py

# Run with coverage
pytest --cov=orchestrator
```

**Writing a Test:**
```python
@pytest.mark.asyncio
async def test_reverse_tool():
    result = await reverse_string("abc")
    assert result["reversed"] == "cba"
```

---

## 🎨 Code Style

We use **Ruff** for linting and formatting.

```bash
# Check for issues
ruff check .

# Fix issues
ruff check --fix .

# Format code
ruff format .
```

**Pre-commit Hook:**
We recommend setting up pre-commit to run these automatically.

---

## 📚 Resources

- **[FastAPI Docs](https://fastapi.tiangolo.com/)** - Our web framework
- **[SQLAlchemy 2.0 Docs](https://docs.sqlalchemy.org/en/20/)** - Our ORM
- **[Pydantic Docs](https://docs.pydantic.dev/)** - Data validation
- **[Redis-py Docs](https://redis-py.readthedocs.io/)** - Redis client

---

**Happy Coding!** 🚀
Questions? Ask in Discord.
