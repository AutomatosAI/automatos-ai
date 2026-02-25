# PRD-56: Infrastructure Scaling, Physical Workspaces & Ephemeral Agent Compute

**Version:** 2.0
**Status:** Planning Phase → Phase 2 Implementation
**Date:** February 15, 2026 (v1.0) · February 25, 2026 (v2.0 — Physical Workspaces)
**Author:** Automatos Core Team
**Prerequisites:** PRD-37 (SaaS Foundation), PRD-54 (LLM Marketplace)
**Blocks:** None (Foundation for enterprise scaling)

### Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-15 | Initial 4-phase roadmap: TaskRunner → ARQ Workers → K8s → Enterprise |
| 2.0 | 2026-02-25 | **Major extension:** Physical Workspace architecture for pilot launch. Added: persistent workspace volumes, workspace filesystem model, tool execution routing (API vs Worker), command sandboxing & whitelist, storage quotas (5GB default), credential injection, Railway Volume integration, `services/workspace-worker/` layout, security model for 15-user pilot. Extended Phase 2 from ephemeral `/tmp/` to persistent `/workspaces/{id}/` with repo caching. |

---

## Executive Summary

This PRD defines the infrastructure evolution path for Automatos AI — from the current Railway-hosted pilot to a fully scalable, workspace-isolated, enterprise-grade compute platform. The core architectural change: **agent tasks execute in isolated compute environments with persistent physical workspaces** rather than in-process with the API server.

### The Problem

Today, all agent execution (workflows, subtasks, tool calls) runs inside the FastAPI process via `asyncio.create_task()`. This means:

- **No isolation** — One workspace's heavy agent task starves all others
- **No persistence** — Tasks are lost if the server restarts
- **No physical workspace** — Agents can't clone repos, run tests, or persist build artifacts between tasks
- **No resource limits** — Can't enforce plan-tier CPU/memory caps
- **No security boundary** — Agent code execution (shell tools, file ops) shares the API server's filesystem and network
- **No horizontal scaling** — Everything runs in one process on one container
- **No auditability** — No infrastructure-level task lifecycle tracking

### The Solution

A **4-phase migration** introducing a `TaskRunner` abstraction that decouples task dispatch from task execution, with **physical workspaces** as the foundation for agent compute:

| Phase | Infrastructure | Timeline | User Scale |
|-------|---------------|----------|------------|
| **Phase 1** (Now) | Railway + `LocalTaskRunner` | Week 1 | Pilot (<50 users) |
| **Phase 2** (Soft Launch) | Railway + ARQ Workers + **Physical Workspaces** + Persistent Volume | Weeks 2-6 | Pilot (15 users) |
| **Phase 3** (Scale) | Managed Kubernetes + Ephemeral Pods | Months 3-6 | Growth (500+ workspaces) |
| **Phase 4** (Enterprise) | Multi-cluster / Bring-Your-Own-Cloud | Month 6+ | Enterprise tenants |

### Key Architecture Decisions

**1. TaskRunner Interface:** Abstract boundary between task dispatch and execution. Swap implementations without touching business logic:

```
LocalTaskRunner     → asyncio (current behavior, Railway-compatible)
QueuedTaskRunner    → Redis queue + worker containers (Phase 2)
KubernetesTaskRunner → K8s Jobs with ephemeral pods (Phase 3)
```

**2. Physical Workspaces (NEW in v2.0):** Each workspace gets a persistent filesystem on the worker volume. Repos clone once and persist. Test results, build artifacts, and data survive between tasks. Agents work in a real development environment — not a throwaway `/tmp/` dir.

```
/workspaces/{workspace_id}/
├── repos/          ← Cloned repos persist (git pull, not re-clone)
├── tasks/          ← Ephemeral per-task execution dirs (cleaned up)
├── artifacts/      ← Test reports, coverage, build outputs (kept)
├── .ssh/           ← Deploy keys (injected from credential store)
└── .gitconfig      ← Per-workspace git config
```

**3. Tool Execution Routing:** Tools split between API (instant, stateless) and Worker (filesystem, subprocess). Agent code is unaware of which backend runs which tool.

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Phase 1: TaskRunner Abstraction](#3-phase-1-taskrunner-abstraction)
4. [Phase 2: Queue-Based Worker + Physical Workspaces](#4-phase-2-queue-based-worker--physical-workspaces)
5. [Physical Workspace Architecture](#5-physical-workspace-architecture) **(NEW v2.0)**
6. [Tool Execution Routing](#6-tool-execution-routing) **(NEW v2.0)**
7. [Workspace Worker Service](#7-workspace-worker-service) **(NEW v2.0)**
8. [Pilot Security Model](#8-pilot-security-model) **(NEW v2.0)**
9. [Phase 3: Kubernetes Ephemeral Pods](#9-phase-3-kubernetes-ephemeral-pods)
10. [Phase 4: Enterprise Multi-Tenant](#10-phase-4-enterprise-multi-tenant)
11. [Data Models & Schema](#11-data-models--schema)
12. [API Changes](#12-api-changes)
13. [Security Model (Full)](#13-security-model-full)
14. [Cost Analysis](#14-cost-analysis)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Risk Assessment](#16-risk-assessment)

---

## 1. Current Architecture Analysis

### Execution Flow (As-Is)

```
POST /api/workflows/{id}/execute
  │
  ├─ Create WorkflowExecution (status: pending)
  ├─ asyncio.create_task(execute_workflow_with_progress())  ← Fire-and-forget
  │    │
  │    ├─ OrchestratorService.execute_workflow()
  │    │    ├─ Stage 1-3: Decompose → Select → Enhance context
  │    │    ├─ Stage 4: AgentExecutionManager.execute_workflow_subtasks()
  │    │    │    ├─ AgentFactory.execute_with_prompt(agent, prompt, tools)
  │    │    │    │    ├─ LLM call (OpenAI/Anthropic/OpenRouter)
  │    │    │    │    ├─ Tool execution (shell, file_ops, research)
  │    │    │    │    └─ Return result dict
  │    │    │    ├─ SSE broadcast (subtask_update)
  │    │    │    └─ DB write (WorkflowExecution.output_data)
  │    │    ├─ Stage 5-9: Aggregate → Learn → Assess → Store → Synthesize
  │    │    └─ Return final result
  │    └─ Update WorkflowExecution (status: completed)
  │
  └─ Return 202 Accepted + execution_id
```

### Current Limitations

| Limitation | Impact | Risk Level |
|-----------|--------|------------|
| In-process execution (`asyncio.create_task`) | Tasks lost on restart/deploy | **High** |
| No resource isolation between workspaces | Noisy neighbor, DoS risk | **High** |
| Shared filesystem for tool execution | Cross-tenant data leakage | **Critical** (enterprise blocker) |
| No task queue persistence | Cannot retry failed tasks | **Medium** |
| Single-process concurrency limit | ~50 concurrent agent tasks max | **Medium** |
| No per-workspace resource quotas | Can't enforce plan limits | **Medium** |
| No task priority system | Free-tier tasks block paid | **Low** (pilot only) |

### Key Files Affected

| File | Role | Lines |
|------|------|-------|
| `modules/agents/execution/execution_manager.py` | Agent task dispatch & tracking | 1,309 |
| `modules/agents/factory/agent_factory.py` | Agent runtime & LLM calls | 2,499 |
| `modules/orchestrator/service.py` | 9-stage workflow pipeline | ~800 |
| `api/workflows.py` | Workflow execution endpoints | ~1,100 |
| `api/workflow_recipes.py` | Recipe execution endpoints | ~800 |
| `consumers/chatbot/service.py` | Chat-triggered agent execution | ~1,300 |

---

## 2. Target Architecture

### Control Plane / Data Plane Separation

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONTROL PLANE                               │
│                   (Always Running)                                │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Next.js  │  │   FastAPI    │  │    Task Dispatcher        │  │
│  │ Frontend │→ │   API Server │→ │  (TaskRunner interface)   │  │
│  └──────────┘  └──────────────┘  └──────────────────────────┘  │
│                       │                      │                   │
│  ┌──────────┐  ┌──────┴─────┐  ┌────────────┴───────────┐     │
│  │PostgreSQL│  │   Redis    │  │   S3 / Object Storage  │     │
│  │+pgvector │  │ Cache/Queue│  │   (artifacts, repos)   │     │
│  └──────────┘  └────────────┘  └────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
                           │
                    Task Dispatch
                    (Phase-dependent)
                           │
┌──────────────────────────────────────────────────────────────────┐
│                       DATA PLANE                                  │
│                (Ephemeral Compute)                                 │
│                                                                   │
│  Phase 2: Worker Containers (Docker)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Worker 1    │  │ Worker 2    │  │ Worker N    │             │
│  │ (ws: abc)   │  │ (ws: def)   │  │ (ws: ghi)   │             │
│  │ task: bugfix│  │ task: docs  │  │ task: review│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                   │
│  Phase 3: Kubernetes Jobs                                         │
│  ┌─ Namespace: ws-abc123 (Pro: 4CPU/8GB) ───────────────┐      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │      │
│  │  │ Pod: fix  │  │ Pod: docs│  │ Pod: test│           │      │
│  │  │ TTL: 30m  │  │ TTL: 15m │  │ TTL: 10m │           │      │
│  │  └──────────┘  └──────────┘  └──────────┘           │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                   │
│  ┌─ Namespace: ws-def456 (Starter: 1CPU/2GB) ───────────┐      │
│  │  ┌──────────┐                                         │      │
│  │  │ Pod: chat │                                         │      │
│  │  │ TTL: 5m   │                                         │      │
│  │  └──────────┘                                         │      │
│  └───────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### TaskRunner Interface (Core Abstraction)

```python
class TaskRunner(ABC):
    """Abstract interface for agent task execution.

    Implementations:
    - LocalTaskRunner:      asyncio in-process (Phase 1)
    - QueuedTaskRunner:     Redis queue + workers (Phase 2)
    - KubernetesTaskRunner: K8s Jobs (Phase 3)
    """

    @abstractmethod
    async def submit_task(self, task: AgentTask) -> TaskHandle:
        """Submit a task for execution. Returns immediately with a handle."""

    @abstractmethod
    async def get_status(self, handle: TaskHandle) -> TaskStatus:
        """Poll task status (pending, running, completed, failed, cancelled)."""

    @abstractmethod
    async def get_result(self, handle: TaskHandle) -> TaskResult:
        """Retrieve completed task result. Blocks until done or timeout."""

    @abstractmethod
    async def cancel_task(self, handle: TaskHandle) -> bool:
        """Request task cancellation. Returns True if cancellation was accepted."""

    @abstractmethod
    async def stream_updates(self, handle: TaskHandle) -> AsyncIterator[TaskEvent]:
        """Stream real-time task progress events."""
```

---

## 3. Phase 1: TaskRunner Abstraction (This Week)

### Goal

Introduce the `TaskRunner` interface and `LocalTaskRunner` implementation without changing any runtime behavior. All existing agent execution paths route through the new abstraction.

### User Stories

#### US-01: TaskRunner Abstract Interface
**Description:** Define the core `TaskRunner` ABC with data models for `AgentTask`, `TaskHandle`, `TaskResult`, `TaskStatus`, and `TaskEvent`.

**Acceptance Criteria:**
- [ ] `core/task_runner/base.py` — Abstract base class with 5 methods
- [ ] `core/task_runner/models.py` — Pydantic models for task lifecycle
- [ ] `core/task_runner/__init__.py` — Clean exports
- [ ] All models workspace-scoped (carry `workspace_id`)
- [ ] Task types: `workflow_subtask`, `chat_agent`, `recipe_step`, `background_job`
- [ ] Priority levels: `low`, `normal`, `high`, `critical`
- [ ] Resource requirements model: CPU, memory, disk, timeout

#### US-02: LocalTaskRunner Implementation
**Description:** Implement `LocalTaskRunner` that wraps current `asyncio.create_task()` behavior behind the `TaskRunner` interface. Zero behavior change.

**Acceptance Criteria:**
- [ ] `core/task_runner/local.py` — Full implementation
- [ ] Uses `asyncio.create_task()` internally (same as today)
- [ ] In-memory task tracking with `Dict[str, TaskHandle]`
- [ ] Status transitions: `pending → running → completed/failed`
- [ ] Timeout enforcement via `asyncio.wait_for()`
- [ ] Cancellation via `asyncio.Task.cancel()`
- [ ] Stream updates via `asyncio.Queue` (maps to existing SSE pattern)
- [ ] Passes existing tests (no behavior change)

#### US-03: TaskRunner Factory & Configuration
**Description:** Factory function that returns the correct `TaskRunner` based on environment configuration.

**Acceptance Criteria:**
- [ ] `core/task_runner/factory.py` — `get_task_runner()` factory
- [ ] Configuration via `TASK_RUNNER_BACKEND` env var: `local` (default), `queued`, `kubernetes`
- [ ] Singleton pattern (one runner per process)
- [ ] FastAPI dependency injection compatible

#### US-04: Integration Points
**Description:** Identify (but don't yet modify) all call sites that will route through `TaskRunner` in Phase 2.

**Acceptance Criteria:**
- [ ] Document all `asyncio.create_task()` call sites for agent work
- [ ] Document `AgentFactory.execute_with_prompt()` callers
- [ ] Document `AgentExecutionManager.execute_workflow_subtasks()` callers
- [ ] Create integration plan for Phase 2 wiring

### Phase 1 File Structure

```
orchestrator/core/task_runner/
├── __init__.py          # Public exports
├── base.py              # TaskRunner ABC
├── models.py            # AgentTask, TaskHandle, TaskResult, TaskStatus, TaskEvent
├── local.py             # LocalTaskRunner (asyncio-based)
└── factory.py           # get_task_runner() factory
```

### Phase 1 Data Models

```python
class AgentTask(BaseModel):
    """A unit of agent work to be executed."""
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: Literal["workflow_subtask", "chat_agent", "recipe_step", "background_job"]
    workspace_id: UUID

    # What to execute
    agent_id: int
    prompt: str
    system_prompt: Optional[str] = None
    tools: List[str] = []
    context: Dict[str, Any] = {}

    # Execution parameters
    priority: Literal["low", "normal", "high", "critical"] = "normal"
    timeout_seconds: int = 300  # 5 min default
    max_retries: int = 2

    # Resource requirements (enforced in Phase 2+)
    resources: TaskResources = TaskResources()

    # Tracing
    parent_execution_id: Optional[int] = None
    correlation_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskResources(BaseModel):
    """Resource requirements for task execution."""
    cpu_millicores: int = 500       # 0.5 CPU
    memory_mb: int = 512            # 512 MB
    disk_mb: int = 1024             # 1 GB scratch space
    gpu: bool = False
    network_access: bool = True     # Can task make external calls?
    repo_clone: bool = False        # Needs git clone capability?


class TaskHandle(BaseModel):
    """Reference to a submitted task."""
    task_id: str
    workspace_id: UUID
    status: TaskStatusEnum
    submitted_at: datetime
    runner_backend: str  # "local", "queued", "kubernetes"


class TaskResult(BaseModel):
    """Result of a completed task."""
    task_id: str
    status: TaskStatusEnum
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0
    execution_time_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskStatusEnum(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"          # Phase 2: in Redis queue
    SCHEDULED = "scheduled"     # Phase 3: K8s pod scheduling
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
```

---

## 4. Phase 2: Queue-Based Worker + Physical Workspaces (Weeks 2-6)

### Goal

Move agent task execution from the API process to a dedicated **workspace worker** container connected via a Redis task queue. Each workspace gets a **persistent physical filesystem** on a Railway Volume. Agents can clone repos, run tests, save artifacts, and work in a real development environment.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAILWAY PROJECT                         │
│                                                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────────────────┐ │
│  │ Frontend  │────▶│ API      │────▶│ Redis                │ │
│  │ (Next.js) │◀────│ (FastAPI)│◀────│ Queue + Pub/Sub      │ │
│  └──────────┘     └────┬─────┘     └──────────┬───────────┘ │
│       ▲                │                       │             │
│       │                │                       ▼             │
│       │                │              ┌──────────────────┐   │
│       │                │              │ Workspace Worker │   │
│       │                │              │ (ARQ Consumer)   │   │
│       │                │              │                  │   │
│       │                │              │  /workspaces/    │   │
│       │                ▼              │    ├─ ws_abc/    │   │
│       │         ┌────────────┐        │    ├─ ws_def/    │   │
│       │         │ PostgreSQL │◀───────│    └─ ws_ghi/    │   │
│       └─────────│  tasks     │        │                  │   │
│                 │  results   │        └────────┬─────────┘   │
│                 └────────────┘                 │             │
│                                        ┌───────┴──────┐     │
│                                        │ Railway Vol  │     │
│                                        │ (persistent) │     │
│                                        └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

**Key difference from v1.0:** The worker mounts a **persistent Railway Volume** instead of using throwaway `/tmp/` dirs. Repos survive between tasks. Each workspace gets its own directory tree.

### Technology Choice: ARQ (Async Redis Queue)

**Why ARQ over Celery:**

| Factor | ARQ | Celery |
|--------|-----|--------|
| Async native | Yes (asyncio) | No (sync workers, needs eventlet/gevent) |
| Dependencies | Just `redis` | Heavy (kombu, billiard, vine, amqp) |
| FastAPI compatibility | Native (same event loop) | Requires adapter |
| Memory footprint | ~30MB per worker | ~80MB per worker |
| Configuration | Minimal | Complex (broker, backend, serializer) |
| Our stack | Already using Redis | Would need Redis anyway |

### Task Lifecycle

```
 User asks: "Clone my repo, run pytest, fix the failing test"

 ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────────┐
 │ SUBMIT  │───▶│ QUEUED  │───▶│ RUNNING │───▶│  COMPLETED   │
 └─────────┘    └─────────┘    └─────────┘    └──────────────┘
     │               │              │                │
     ▼               ▼              ▼                ▼
  API creates    Redis queue    Worker picks     Results written
  task record    receives job   up job           to DB + Redis
  in Postgres    (priority)     creates dir      pub/sub notifies
                                executes         frontend via WS
```

### Progress Streaming

```
Worker                    Redis Pub/Sub              Frontend
  │                           │                         │
  ├─ "Cloning repo..."  ────▶│────▶ WS channel ──────▶│  Cloning...
  ├─ "Running pytest..." ───▶│────▶ WS channel ──────▶│  Running tests...
  ├─ "3 passed, 1 failed" ─▶│────▶ WS channel ──────▶│  1 failure found
  ├─ "Fixing test_auth..." ─▶│────▶ WS channel ──────▶│  Fixing...
  └─ "All 4 passed"     ───▶│────▶ WS channel ──────▶│  Done
```

### User Stories

#### US-05: QueuedTaskRunner Implementation
**Description:** `TaskRunner` implementation that enqueues tasks to Redis and returns results via Redis pub/sub.

**Acceptance Criteria:**
- [ ] `core/task_runner/queued.py` — Full implementation
- [ ] Tasks serialized to Redis queue (JSON, workspace-scoped key)
- [ ] Results stored in Redis with TTL (1 hour)
- [ ] Status polling via Redis hash
- [ ] Event streaming via Redis pub/sub channel per task
- [ ] Priority queues: `tasks:critical`, `tasks:high`, `tasks:normal`, `tasks:low`
- [ ] Dead letter queue for failed tasks (`tasks:dead`)
- [ ] Configurable worker concurrency per queue

#### US-06: Workspace Worker Service
**Description:** Standalone worker process that consumes tasks from Redis queue and executes agent work within persistent physical workspaces.

**Acceptance Criteria:**
- [ ] `services/workspace-worker/main.py` — ARQ worker entry point
- [ ] Reuses `AgentFactory`, `UnifiedToolExecutor`, LLM clients
- [ ] Mounts Railway Volume at `/workspaces/`
- [ ] Each workspace gets persistent dir: `/workspaces/{workspace_id}/`
- [ ] Each task gets ephemeral dir: `/workspaces/{workspace_id}/tasks/task_{id}/`
- [ ] Task dir cleaned after completion; workspace dir persists
- [ ] Worker reports health to Redis (heartbeat)
- [ ] Graceful shutdown (finish current task, don't accept new)
- [ ] Storage quota enforcement before task execution

#### US-07: Workspace Worker Dockerfile
**Description:** Docker image for the workspace worker with full DevOps toolchain.

**Acceptance Criteria:**
- [ ] `services/workspace-worker/Dockerfile` — Multi-stage build
- [ ] Includes: Python 3.12, git, Node.js, npm/pnpm
- [ ] Pre-installed tools: pytest, pytest-cov, ruff, black, mypy, vitest
- [ ] Non-root user execution (UID 1000)
- [ ] Configurable via env vars: `WORKER_CONCURRENCY`, `WORKER_QUEUES`
- [ ] Health check endpoint or heartbeat
- [ ] Volume mount point: `/workspaces/`

#### US-08: Task Persistence & Recovery
**Description:** Tasks survive API server restarts and worker crashes.

**Acceptance Criteria:**
- [ ] Queued tasks persist in Redis (AOF/RDB)
- [ ] Running tasks have heartbeat timeout (60s)
- [ ] Orphaned tasks (no heartbeat) re-queued automatically
- [ ] Failed tasks stored with error info for retry/debugging
- [ ] Task history queryable via API

#### US-09: Per-Workspace Queue & Storage Limits
**Description:** Enforce concurrent task limits and storage quotas based on workspace plan tier.

**Acceptance Criteria:**
- [ ] Pilot plan: 1 concurrent task, 5GB storage
- [ ] Business plan: 3 concurrent tasks, 10GB storage
- [ ] Enterprise plan: 10 concurrent tasks, 50GB storage (configurable)
- [ ] Tasks exceeding concurrency limit are queued (not rejected)
- [ ] Workspace usage tracked in Redis counter
- [ ] Storage quota checked via `du -sh` before each task; task rejected if over limit
- [ ] Storage quotas configurable in `config.py` / environment variable

#### US-10: Docker Compose Worker Profile
**Description:** Add workspace worker to docker-compose for local development.

**Acceptance Criteria:**
- [ ] `docker-compose.yml` updated with workspace-worker service
- [ ] `--profile workers` to enable worker containers
- [ ] Default: 1 worker replica (pilot scale)
- [ ] Environment: shares backend env vars + worker-specific config
- [ ] Volume mount: `./workspaces:/workspaces` for local dev
- [ ] Railway: persistent volume mounted at `/workspaces`

### Phase 2 Infrastructure

```yaml
# docker-compose.yml additions
services:
  workspace-worker:
    build:
      context: .
      dockerfile: services/workspace-worker/Dockerfile
    deploy:
      replicas: ${WORKER_REPLICAS:-1}
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    environment:
      WORKER_CONCURRENCY: ${WORKER_CONCURRENCY:-3}
      WORKER_QUEUES: "critical,high,normal,low"
      DATABASE_URL: ${DATABASE_URL}
      REDIS_URL: ${REDIS_URL}
      WORKSPACE_VOLUME_PATH: /workspaces
      WORKSPACE_DEFAULT_QUOTA_GB: 5
      # LLM keys inherited from backend
    volumes:
      - workspace-data:/workspaces
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    profiles: ["workers"]

volumes:
  workspace-data:
    driver: local
```

### Phase 2 on Railway

Railway supports multiple services per project. The workspace worker deploys as a separate service with a persistent volume:

```
railway.json (or Railway Dashboard)
│
├── Service: api              ← Existing FastAPI
│   ├── Dockerfile: orchestrator/Dockerfile
│   └── Variables: DATABASE_URL, REDIS_URL, ...
│
├── Service: workspace-worker ← NEW
│   ├── Dockerfile: services/workspace-worker/Dockerfile
│   ├── Variables: DATABASE_URL, REDIS_URL, WORKSPACE_VOLUME_PATH, ...
│   └── Volume: /workspaces (Railway persistent volume, 100GB)
│
├── Service: agent-opt-worker ← Existing (FutureAGI scoring)
│   └── Dockerfile: services/agent-opt-worker/Dockerfile
│
├── Service: postgres         ← Existing (shared)
├── Service: redis            ← Existing (shared)
└── Service: frontend         ← Existing Next.js
```

**Isolation guarantee:** The API, workspace-worker, agent-opt-worker, and frontend are **separate Railway containers**. They share Postgres and Redis via internal networking, but their filesystems are completely isolated. A user in the workspace-worker cannot access the API container's filesystem, and vice versa.

Cost impact: ~$10-20/mo additional for 1 workspace-worker replica + ~$2/mo for persistent volume.

---

## 5. Physical Workspace Architecture (NEW in v2.0)

### Goal

Give each workspace a **persistent, isolated filesystem** on the worker volume. Agents work in a real development environment — they can clone repos, run tests, build projects, and persist results between tasks. This is the foundation for Automatos AI's DevOps capabilities.

### Workspace Filesystem Layout

```
/workspaces/                              ← Railway Volume (mounted on worker)
│
├── ws_abc123/                            ← Workspace A (user's workspace)
│   ├── repos/                            ← Cloned repositories (PERSISTENT)
│   │   ├── automatos-ai/                 ← Full repo checkout
│   │   │   ├── .git/
│   │   │   ├── orchestrator/
│   │   │   ├── frontend/
│   │   │   └── ...
│   │   └── another-project/
│   │       └── ...
│   ├── tasks/                            ← Per-task execution dirs (EPHEMERAL)
│   │   ├── task_7f3a/                    ← Active task workspace
│   │   │   ├── .task_env                 ← Injected credentials (cleaned up)
│   │   │   ├── .task_log                 ← Execution log
│   │   │   └── scratch/                  ← Temp files for this task
│   │   └── (cleaned up after completion)
│   ├── artifacts/                        ← Test reports, build outputs (PERSISTENT)
│   │   ├── 2026-02-25_pytest_report.html
│   │   ├── coverage.xml
│   │   └── build_output.tar.gz
│   ├── .ssh/                             ← Deploy keys (injected from credential store)
│   │   └── id_ed25519
│   ├── .gitconfig                        ← Per-workspace git identity
│   └── .workspace_meta.json              ← Workspace metadata (quota, created_at, etc.)
│
├── ws_def456/                            ← Workspace B (different user — ISOLATED)
│   ├── repos/
│   ├── tasks/
│   └── artifacts/
│
└── ws_ghi789/                            ← Workspace C
    └── ...
```

### Persistence Model

| Directory | Lifecycle | Purpose | Size Impact |
|-----------|-----------|---------|-------------|
| `repos/` | **Persistent** — survives across all tasks | Cloned repos; `git pull` instead of re-clone | High (biggest consumer) |
| `tasks/` | **Ephemeral** — cleaned after each task completes | Scratch space for active execution | Low (auto-cleaned) |
| `artifacts/` | **Persistent** — kept until workspace cleanup | Test reports, coverage, build outputs | Medium (user-managed) |
| `.ssh/` | **Persistent** — injected from credential store | Deploy keys for private repo access | Negligible |
| `.gitconfig` | **Persistent** — set once | Git author identity for commits | Negligible |

### Repo Caching (Key Performance Win)

```
Task 1: "Clone automatos-ai and run tests"
  → git clone https://github.com/.../automatos-ai.git
  → /workspaces/ws_abc/repos/automatos-ai/  [CREATED — 3.5GB]
  → pytest runs, results saved to artifacts/
  → Task dir cleaned up

  3 hours later...

Task 2: "Run tests again on automatos-ai"
  → /workspaces/ws_abc/repos/automatos-ai/  [ALREADY EXISTS]
  → git fetch && git pull  (fast — delta only)
  → pytest runs immediately
  → NO 3.5GB re-clone

  Next day...

Task 3: "Clone my-other-project too"
  → /workspaces/ws_abc/repos/my-other-project/  [CREATED]
  → automatos-ai STILL THERE from before
```

### Storage Quotas

Each workspace has a configurable storage limit. Enforced before task execution starts:

```python
# Workspace storage limits (configurable per plan tier)
WORKSPACE_STORAGE_LIMITS = {
    "pilot":      5 * 1024**3,    # 5GB  — soft launch default
    "starter":    2 * 1024**3,    # 2GB  — free tier (future)
    "business":  10 * 1024**3,    # 10GB
    "enterprise": 50 * 1024**3,   # 50GB
}
```

**Enforcement flow:**

```
Task arrives at worker
  │
  ├── 1. Resolve workspace dir: /workspaces/{workspace_id}/
  ├── 2. Check storage: du -sh /workspaces/{workspace_id}/
  ├── 3. Compare against workspace quota
  │
  ├── UNDER LIMIT → Execute task normally
  │
  └── OVER LIMIT → Reject task with error:
      "Workspace storage at 4.8GB / 5.0GB. Free space by deleting
       old repos (workspace cleanup tool) or upgrade your plan."
```

**For the 15-user pilot at 5GB each:** 75GB max. Railway persistent volumes support up to 100GB on Pro plan. Plenty of headroom.

### Workspace Metadata

Each workspace stores metadata for tracking and quota enforcement:

```json
// /workspaces/{workspace_id}/.workspace_meta.json
{
  "workspace_id": "ws_abc123",
  "created_at": "2026-02-25T10:00:00Z",
  "plan_tier": "pilot",
  "storage_quota_bytes": 5368709120,
  "last_task_at": "2026-02-25T14:30:00Z",
  "repos_cached": ["automatos-ai", "client-project"],
  "total_tasks_run": 47
}
```

---

## 6. Tool Execution Routing (NEW in v2.0)

### Goal

Define which tools execute in the API process (instant, stateless) vs. the workspace worker (filesystem, subprocess). Agents are unaware of the routing — the `TaskRunner` handles dispatch transparently.

### Routing Matrix

```
┌─────────────────────────────────────────────────────────────┐
│              TOOL EXECUTION ROUTING                          │
│                                                              │
│  ┌─────────────────────────┐  ┌────────────────────────────┐│
│  │  RUNS IN API (instant)  │  │  RUNS IN WORKER (queued)   ││
│  │                         │  │                            ││
│  │  search_codebase    ✓   │  │  execute_command       ✓   ││
│  │  semantic_search    ✓   │  │  read_file (workspace) ✓   ││
│  │  search_documents   ✓   │  │  write_file            ✓   ││
│  │  search_images      ✓   │  │  create_directory      ✓   ││
│  │  search_tables      ✓   │  │  list_directory        ✓   ││
│  │  database_query     ✓   │  │                            ││
│  │  composio_execute   ✓   │  │  (anything touching the    ││
│  │  http_request       ✓   │  │   filesystem or running     ││
│  │                         │  │   subprocesses)            ││
│  │  (reads from DB/index,  │  │                            ││
│  │   no filesystem needed) │  │                            ││
│  └─────────────────────────┘  └────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Detailed Tool Classification

| Tool | Location | Security Level | Rationale |
|------|----------|---------------|-----------|
| `search_codebase` | API | SAFE | Reads from CodeGraph index in Postgres, no filesystem |
| `semantic_search` | API | SAFE | Reads from pgvector, no filesystem |
| `search_documents` | API | SAFE | Reads from document index in Postgres |
| `search_images` | API | SAFE | Reads from image index in Postgres |
| `search_tables` | API | SAFE | Reads structured data from Postgres |
| `database_query` | API | CAUTIOUS | NL2SQL against Postgres (read-only) |
| `composio_execute` | API | CAUTIOUS | Calls external APIs (Jira, Slack, GitHub) |
| `http_request` | API | CAUTIOUS | Whitelisted HTTP calls to internal/platform URLs |
| `read_file` | **Worker** | CAUTIOUS | Reads files from workspace filesystem |
| `write_file` | **Worker** | CAUTIOUS | Writes files to workspace filesystem |
| `create_directory` | **Worker** | CAUTIOUS | Creates dirs in workspace filesystem |
| `list_directory` | **Worker** | SAFE | Lists workspace directory contents |
| `execute_command` | **Worker** | DANGEROUS | Runs shell commands (git, pytest, npm, etc.) |
| `ssh_execute` | **Disabled for pilot** | DANGEROUS | See notes below |

### SSH Execute — Pilot Decision

`ssh_execute` lets agents SSH into arbitrary hosts. For the 15-user pilot:

**Decision: DISABLE for pilot.** Agents use `execute_command` locally in their workspace instead. Re-enable in Phase 3 with per-workspace allowed-host configuration.

**Future (post-pilot):** Each workspace registers allowed SSH targets via credentials. Agents can only SSH to hosts that workspace has credentials for.

### Tool Routing Implementation

The `WorkspaceToolExecutor` wraps all worker-side tools with path validation and sandboxing:

```python
class WorkspaceToolExecutor:
    """All tool calls in worker are sandboxed to workspace dir."""

    def __init__(self, workspace_id: str, volume_path: str = "/workspaces"):
        self.workspace_id = workspace_id
        self.workspace_root = Path(volume_path) / workspace_id
        self.task_dir: Optional[Path] = None

    def resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve a path and verify it stays within the workspace."""
        resolved = (self.workspace_root / relative_path).resolve()
        if not str(resolved).startswith(str(self.workspace_root.resolve())):
            raise SecurityError(f"Path traversal blocked: {relative_path}")
        return resolved

    async def execute_command(self, command: str, timeout: int = 60) -> dict:
        """Execute a shell command, sandboxed to workspace directory."""
        binary = command.split()[0]
        if binary not in ALLOWED_COMMANDS:
            return {"error": f"Command '{binary}' not allowed", "allowed": list(ALLOWED_COMMANDS)}

        result = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.workspace_root),        # JAILED to workspace
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._sandboxed_env(),            # Stripped-down PATH
        )
        stdout, stderr = await asyncio.wait_for(
            result.communicate(), timeout=timeout
        )
        return {
            "exit_code": result.returncode,
            "stdout": stdout.decode()[:100_000],  # Cap output at 100KB
            "stderr": stderr.decode()[:50_000],
        }
```

### Command Whitelist

```python
# Commands agents are allowed to execute in their workspace
ALLOWED_COMMANDS = {
    # Version control
    "git",

    # Python ecosystem
    "python", "python3", "pip", "pip3", "uv",
    "pytest", "ruff", "black", "mypy", "isort", "flake8",

    # Node.js ecosystem
    "node", "npm", "npx", "pnpm", "yarn", "vitest", "jest",
    "tsc", "eslint", "prettier",

    # General tools
    "ls", "cat", "grep", "find", "tree", "wc", "sort", "head", "tail",
    "diff", "patch", "jq",
    "curl", "wget",       # For API testing
    "make",               # Build automation
    "tar", "gzip", "zip", "unzip",

    # Language runtimes (polyglot repos)
    "cargo", "go", "ruby", "java", "javac", "mvn", "gradle",
}

# Explicitly blocked — never allowed regardless of whitelist
BLOCKED_PATTERNS = [
    "rm -rf /",           # Filesystem destruction
    "sudo",               # Privilege escalation
    "su ",                # User switching
    "chmod 777",          # Dangerous permissions
    "docker",             # Container escape risk (pilot)
    "kubectl",            # K8s access (pilot)
    "ssh ",               # Use ssh_execute tool instead (disabled)
    "> /dev/",            # Device access
    "mkfs",               # Filesystem formatting
    "dd if=",             # Raw disk operations
]
```

---

## 7. Workspace Worker Service (NEW in v2.0)

### Service Location

Follows existing pattern alongside `services/agent-opt-worker/`:

```
services/
├── agent-opt-worker/              ← Existing (FutureAGI scoring — HTTP service)
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
│
└── workspace-worker/              ← NEW (code execution — ARQ queue consumer)
    ├── Dockerfile
    ├── main.py                    ← ARQ consumer entry point
    ├── executor.py                ← WorkspaceToolExecutor (sandboxed commands)
    ├── workspace_manager.py       ← Workspace dir provisioning, quotas, cleanup
    ├── requirements.txt
    └── README.md
```

**Key difference from agent-opt-worker:** The agent-opt-worker is a **FastAPI HTTP service** (request/response). The workspace-worker is an **ARQ queue consumer** (pull-based, long-running tasks).

### Dockerfile

```dockerfile
# services/workspace-worker/Dockerfile
FROM python:3.12-slim AS base

# System dependencies — DevOps toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget openssh-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Node.js (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm vitest jest

# Python dev/test tools (pre-installed so agents don't need to pip install)
RUN pip install --no-cache-dir \
    pytest pytest-cov pytest-asyncio \
    ruff black mypy isort flake8 \
    uv

# Application code
WORKDIR /app
COPY services/workspace-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator/ /app/orchestrator/
COPY services/workspace-worker/ /app/worker/

# Non-root user
RUN useradd -m -u 1000 worker \
    && mkdir -p /workspaces \
    && chown -R worker:worker /app /workspaces
USER worker

# Volume mount point
VOLUME /workspaces

# Entry point — ARQ worker consumer
CMD ["python", "-m", "worker.main"]
```

### Worker Main (ARQ Consumer)

```python
# services/workspace-worker/main.py
"""
Workspace Worker — ARQ Consumer
Pulls tasks from Redis queue, executes in isolated workspace directories.
"""
import asyncio
import logging
from arq import create_pool
from arq.connections import RedisSettings

from worker.executor import WorkspaceToolExecutor
from worker.workspace_manager import WorkspaceManager

logger = logging.getLogger("workspace-worker")

async def execute_workspace_task(ctx, task_payload: dict):
    """Main task handler — called by ARQ for each queued task."""
    workspace_id = task_payload["workspace_id"]
    task_id = task_payload["task_id"]
    redis = ctx["redis"]

    # 1. Provision workspace if first use
    ws_manager = WorkspaceManager(workspace_id)
    ws_manager.ensure_workspace_exists()

    # 2. Check storage quota
    if not ws_manager.check_quota():
        await _publish_error(redis, task_id, workspace_id,
            f"Workspace storage exceeds quota ({ws_manager.usage_human}/{ws_manager.quota_human})")
        return

    # 3. Create ephemeral task directory
    task_dir = ws_manager.create_task_dir(task_id)

    # 4. Inject credentials (SSH keys, tokens)
    ws_manager.inject_credentials(task_id, task_payload.get("credentials", {}))

    try:
        # 5. Execute task steps
        executor = WorkspaceToolExecutor(workspace_id)
        for step in task_payload.get("steps", []):
            await _publish_progress(redis, task_id, workspace_id, step)
            result = await executor.execute_step(step)

            if result.get("error"):
                await _publish_error(redis, task_id, workspace_id, result["error"])
                return result

        # 6. Publish completion
        await _publish_complete(redis, task_id, workspace_id, result)
        return result

    finally:
        # 7. Cleanup: task dir + credentials (workspace persists)
        ws_manager.cleanup_task(task_id)


class WorkerSettings:
    """ARQ worker configuration."""
    functions = [execute_workspace_task]
    redis_settings = RedisSettings.from_dsn(os.environ["REDIS_URL"])
    max_jobs = int(os.environ.get("WORKER_CONCURRENCY", 3))
    job_timeout = 600    # 10 min max per task
    keep_result = 3600   # Keep results for 1 hour
    health_check_interval = 30
```

### Workspace Manager

```python
# services/workspace-worker/workspace_manager.py
"""
Manages physical workspace directories on the persistent volume.
Handles provisioning, quota enforcement, and cleanup.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

VOLUME_PATH = os.environ.get("WORKSPACE_VOLUME_PATH", "/workspaces")
DEFAULT_QUOTA_GB = int(os.environ.get("WORKSPACE_DEFAULT_QUOTA_GB", 5))

class WorkspaceManager:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.root = Path(VOLUME_PATH) / workspace_id
        self.quota_bytes = DEFAULT_QUOTA_GB * 1024**3

    def ensure_workspace_exists(self):
        """Create workspace directory tree if first use."""
        for subdir in ["repos", "tasks", "artifacts"]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

        meta_path = self.root / ".workspace_meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "workspace_id": self.workspace_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "plan_tier": "pilot",
                "storage_quota_bytes": self.quota_bytes,
                "total_tasks_run": 0,
            }))

    def check_quota(self) -> bool:
        """Check if workspace is under storage quota."""
        usage = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        self._current_usage = usage
        return usage < self.quota_bytes

    @property
    def usage_human(self) -> str:
        return f"{self._current_usage / 1024**3:.1f}GB"

    @property
    def quota_human(self) -> str:
        return f"{self.quota_bytes / 1024**3:.0f}GB"

    def create_task_dir(self, task_id: str) -> Path:
        """Create ephemeral task execution directory."""
        task_dir = self.root / "tasks" / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def cleanup_task(self, task_id: str):
        """Remove ephemeral task directory + injected credentials."""
        task_dir = self.root / "tasks" / f"task_{task_id}"
        if task_dir.exists():
            shutil.rmtree(task_dir)
        # Clean credential files
        ssh_dir = self.root / ".ssh"
        task_env = self.root / f".task_env_{task_id}"
        if task_env.exists():
            task_env.unlink()

    def inject_credentials(self, task_id: str, credentials: dict):
        """Inject SSH keys and tokens for this task."""
        if "ssh_private_key" in credentials:
            ssh_dir = self.root / ".ssh"
            ssh_dir.mkdir(exist_ok=True)
            key_file = ssh_dir / "id_ed25519"
            key_file.write_text(credentials["ssh_private_key"])
            key_file.chmod(0o600)
        # Git config for commits
        if "git_name" in credentials:
            gitconfig = self.root / ".gitconfig"
            gitconfig.write_text(
                f'[user]\n  name = {credentials["git_name"]}\n'
                f'  email = {credentials.get("git_email", "agent@automatos.app")}\n'
            )
```

---

## 8. Pilot Security Model (NEW in v2.0)

### Threat Model (15 Trusted Beta Users)

Not bulletproof, but reasonable for trusted pilot. Hardens progressively in Phase 3.

```
┌──────────────────────────────────────────────────────────────┐
│               PILOT SECURITY LAYERS (Phase 2)                 │
│                                                               │
│  Layer 1: Container Isolation (Railway)                       │
│  ├── API and Worker are separate Railway containers           │
│  ├── Each has its own filesystem, process space, network      │
│  ├── Worker can't access API filesystem and vice versa        │
│  └── Worker connects to Postgres/Redis via internal network   │
│                                                               │
│  Layer 2: Workspace Directory Isolation (Code)                │
│  ├── All paths resolved via resolve_safe_path()               │
│  ├── symlink-resolved, must start with /workspaces/{ws_id}    │
│  ├── Path traversal (../../) blocked before execution         │
│  └── One workspace cannot access another's directory          │
│                                                               │
│  Layer 3: Command Sandboxing (Whitelist)                      │
│  ├── Only whitelisted binaries can execute                    │
│  ├── Blocked patterns: sudo, rm -rf /, docker, etc.          │
│  ├── All commands run with cwd pinned to workspace dir        │
│  └── Stripped-down PATH (no access to system binaries)        │
│                                                               │
│  Layer 4: Credential Isolation                                │
│  ├── SSH keys loaded per-task from encrypted credential store │
│  ├── Injected into workspace .ssh/ dir                        │
│  ├── Cleaned up after task completion                         │
│  ├── LLM API keys loaded per-workspace (Fernet-encrypted)    │
│  └── Never written to repo directories or task logs           │
│                                                               │
│  Layer 5: Resource Limits                                     │
│  ├── Storage quota: du -sh check before each task             │
│  ├── Task timeout: subprocess.run(timeout=300) default        │
│  ├── Output cap: stdout/stderr truncated at 100KB             │
│  ├── Concurrent tasks: 1 per workspace (pilot)                │
│  └── Railway container limits: 2 CPU, 2GB RAM                 │
│                                                               │
│  Layer 6: Network Restrictions                                │
│  ├── Worker connects to: Redis, Postgres (internal)           │
│  ├── Outbound: github.com, pypi.org, npmjs.org (for installs)│
│  ├── ssh_execute: DISABLED for pilot                          │
│  └── http_request: whitelisted domains only (existing)        │
└──────────────────────────────────────────────────────────────┘
```

### Path Traversal Prevention (Critical)

```python
def resolve_safe_path(workspace_id: str, relative_path: str) -> Path:
    """
    Resolve a path and guarantee it stays within the workspace.
    Blocks: ../../, symlink escape, absolute paths, null bytes.
    """
    base = Path(f"/workspaces/{workspace_id}").resolve()

    # Block null bytes (bypass attempt)
    if "\x00" in relative_path:
        raise SecurityError("Null byte in path")

    # Block absolute paths
    if relative_path.startswith("/"):
        raise SecurityError(f"Absolute path not allowed: {relative_path}")

    # Resolve and check containment
    resolved = (base / relative_path).resolve()
    if not str(resolved).startswith(str(base)):
        raise SecurityError(f"Path traversal blocked: {relative_path}")

    return resolved
```

### What Users CAN Do (Within Their Workspace)

- Clone any public or credentialed-private repo
- Run test suites (pytest, vitest, jest, go test, cargo test)
- Install dependencies (pip install, npm install — into workspace)
- Read and write any file in their workspace
- Run linters, formatters, type checkers
- Build projects (make, npm run build, cargo build)
- Save test reports, coverage data, build artifacts
- Use curl/wget for API testing

### What Users CANNOT Do

- Access another workspace's files (path validation)
- Run privileged commands (sudo, docker, kubectl)
- Access system files outside /workspaces/{their_id}
- SSH to external servers (disabled for pilot)
- Send HTTP requests to non-whitelisted domains
- Exceed their storage quota
- Run tasks longer than the timeout (killed)
- Access Postgres/Redis connection strings (not in subprocess env)

---

## 9. Phase 3: Kubernetes Ephemeral Pods (Months 3-6)

### Goal

Replace static worker containers with dynamically scheduled Kubernetes Jobs. Each agent task runs in its own pod with workspace-scoped resource limits, network policies, and ephemeral storage.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CONTROL PLANE (K8s Namespace: automatos-control)            │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────────┐   │
│  │ Frontend │  │ API Pod  │  │ Task Controller Pod     │   │
│  │ Deploy   │  │ Deploy   │  │ (watches Redis queue,   │   │
│  │ (2 rep)  │  │ (3 rep)  │  │  creates K8s Jobs)      │   │
│  └──────────┘  └──────────┘  └─────────────────────────┘   │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Postgres  │  │ Redis    │  │ S3/Minio │                  │
│  │StatefulSet│  │ Deploy   │  │          │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  DATA PLANE (Dynamic Namespaces)                              │
│                                                               │
│  Namespace: ws-{workspace_id}                                 │
│  ├── ResourceQuota (from plan_limits)                         │
│  ├── LimitRange (per-pod defaults)                            │
│  ├── NetworkPolicy (egress: LLM APIs only)                    │
│  ├── ServiceAccount (minimal RBAC)                            │
│  └── Jobs:                                                    │
│      ├── task-{id}-bugfix    [Running]  TTL: 30m             │
│      ├── task-{id}-docs      [Running]  TTL: 15m             │
│      └── task-{id}-review    [Completed → auto-cleanup]      │
└──────────────────────────────────────────────────────────────┘
```

### K8s Primitives Mapping

| Automatos Concept | K8s Primitive | Purpose |
|-------------------|--------------|---------|
| Agent task | **Job** | Run-to-completion workload |
| Task workspace | **Pod** with `emptyDir` volume | Isolated filesystem |
| Workspace isolation | **Namespace** per workspace | Resource & network boundary |
| Plan limits | **ResourceQuota** | CPU/memory caps per workspace |
| Per-task limits | **LimitRange** | Default CPU/memory per pod |
| Task timeout | `activeDeadlineSeconds` | Kill runaway tasks |
| Auto-cleanup | `ttlSecondsAfterFinished` | Remove completed job pods |
| Security boundary | **NetworkPolicy** | Restrict pod network access |
| Repo cloning | `emptyDir` with `sizeLimit` | Temp disk for git clone |
| Inter-agent comms | Redis pub/sub (existing) | Cross-pod messaging |
| Task scaling | **KEDA** `ScaledJob` | Scale from zero on queue depth |

### User Stories

#### US-11: KubernetesTaskRunner Implementation
**Description:** `TaskRunner` that creates K8s Jobs for agent tasks.

**Acceptance Criteria:**
- [ ] `core/task_runner/kubernetes.py` — Full implementation
- [ ] Creates K8s Job manifest from `AgentTask`
- [ ] Pod spec includes: agent image, env vars, resource limits, volumes
- [ ] Job namespace = `ws-{workspace_id}` (auto-created if missing)
- [ ] `activeDeadlineSeconds` from `task.timeout_seconds`
- [ ] `ttlSecondsAfterFinished: 300` (cleanup after 5 min)
- [ ] Status polling via K8s API (Job status)
- [ ] Log streaming via K8s API (pod logs)
- [ ] Result retrieval from Redis (pod writes result to Redis on completion)

#### US-12: Task Controller
**Description:** Long-running controller that watches the Redis queue and creates K8s Jobs.

**Acceptance Criteria:**
- [ ] `worker/controller.py` — Task Controller process
- [ ] Watches Redis queue for new tasks
- [ ] Creates K8s Job per task via `KubernetesTaskRunner`
- [ ] Handles namespace provisioning (lazy creation)
- [ ] Enforces workspace ResourceQuota before scheduling
- [ ] Reconciliation loop: detect orphaned jobs, re-queue failed
- [ ] Metrics: task queue depth, scheduling latency, pod startup time

#### US-13: Workspace Namespace Provisioning
**Description:** Automatic K8s namespace creation and configuration per workspace.

**Acceptance Criteria:**
- [ ] Namespace created on first task submission for workspace
- [ ] ResourceQuota set from `workspace.plan_limits`:
  - Starter: `cpu: 2, memory: 2Gi, pods: 3`
  - Pro: `cpu: 8, memory: 16Gi, pods: 10`
  - Enterprise: `cpu: 32, memory: 64Gi, pods: 50` (configurable)
- [ ] LimitRange defaults: `cpu: 500m, memory: 512Mi` per pod
- [ ] NetworkPolicy: allow egress to LLM APIs, Redis, Postgres; deny all else
- [ ] ServiceAccount with minimal RBAC (no cluster access)
- [ ] Namespace labels: `workspace-id`, `plan`, `owner`

#### US-14: Agent Task Pod Spec
**Description:** Pod template for agent task execution.

**Acceptance Criteria:**
- [ ] Base image: `automatos/agent-worker:latest` (same as Phase 2 worker)
- [ ] Volumes:
  - `emptyDir` at `/workspace` (scratch space, sizeLimit from task.resources.disk_mb)
  - ConfigMap for task definition
  - Secret for LLM API keys (from workspace credentials)
- [ ] Environment: `TASK_ID`, `WORKSPACE_ID`, `REDIS_URL`, `DATABASE_URL`
- [ ] Entrypoint: execute single task, write result to Redis, exit
- [ ] Security context: `runAsNonRoot`, `readOnlyRootFilesystem` (except /workspace)
- [ ] No service account token mounted (prevent K8s API access from pod)

#### US-15: KEDA Auto-Scaling
**Description:** Scale agent pods from zero based on queue depth.

**Acceptance Criteria:**
- [ ] KEDA `ScaledJob` per priority queue
- [ ] Scale trigger: Redis list length (`tasks:critical`, etc.)
- [ ] Min replicas: 0 (scale to zero when idle)
- [ ] Max replicas: configurable per environment
- [ ] Cooldown period: 30 seconds
- [ ] Pod startup target: <10 seconds (pre-pulled images)

#### US-16: Agent-to-Agent Communication
**Description:** Enable pods to communicate with other agent tasks in the same workspace.

**Acceptance Criteria:**
- [ ] Redis pub/sub channels scoped to workspace: `ws:{workspace_id}:agent_events`
- [ ] Message types: `task_completed`, `data_available`, `request_help`
- [ ] Pods can read other task results from Redis (same workspace only)
- [ ] NetworkPolicy allows intra-namespace communication
- [ ] Future: service mesh for direct pod-to-pod gRPC

### K8s Job Manifest Template

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: task-${task_id}
  namespace: ws-${workspace_id}
  labels:
    app: automatos-agent
    workspace: ${workspace_id}
    task-type: ${task_type}
    priority: ${priority}
spec:
  activeDeadlineSeconds: ${timeout_seconds}
  ttlSecondsAfterFinished: 300
  backoffLimit: ${max_retries}
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: agent-minimal
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: agent
        image: automatos/agent-worker:${version}
        command: ["python", "-m", "worker.execute_task"]
        args: ["--task-id", "${task_id}"]
        env:
        - name: TASK_ID
          value: "${task_id}"
        - name: WORKSPACE_ID
          value: "${workspace_id}"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: automatos-infra
              key: redis-url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: automatos-infra
              key: database-url
        resources:
          requests:
            cpu: "${cpu_request}m"
            memory: "${memory_request}Mi"
          limits:
            cpu: "${cpu_limit}m"
            memory: "${memory_limit}Mi"
        volumeMounts:
        - name: workspace
          mountPath: /workspace
      volumes:
      - name: workspace
        emptyDir:
          sizeLimit: "${disk_mb}Mi"
```

---

## 10. Phase 4: Enterprise Multi-Tenant (Month 6+)

### Goal

Support enterprise customers with dedicated compute, compliance requirements, and optional bring-your-own-cloud deployments.

### Capabilities

#### Dedicated Clusters
- Enterprise tenants get their own K8s cluster (or dedicated node pool)
- Full network isolation from other tenants
- Custom retention, compliance, and audit policies
- SOC 2 / ISO 27001 scope per cluster

#### Bring-Your-Own-Cloud (BYOC)
- Deploy agent worker pods into customer's cloud account
- Customer provides K8s cluster credentials
- Automatos control plane remains hosted
- Agent tasks execute within customer's network perimeter
- Data never leaves customer's environment

#### Air-Gapped Deployments
- Full Automatos stack as Helm chart
- Runs entirely within customer infrastructure
- Offline LLM support (local models via Ollama/vLLM)
- Manual update distribution

### Enterprise Features Matrix

| Feature | Pro | Enterprise | Enterprise+ (BYOC) |
|---------|-----|-----------|-------------------|
| Workspace namespaces | Shared cluster | Dedicated node pool | Customer cluster |
| Data residency | Multi-region | Specific region | Customer-controlled |
| Network isolation | NetworkPolicy | VPC peering | Customer VPC |
| Compliance | SOC 2 shared | SOC 2 dedicated | Customer-audited |
| SLA | 99.5% | 99.9% | Customer-managed |
| Agent image customization | No | Base + extensions | Full control |
| Max concurrent tasks | 10 | 50 | Unlimited |

---

## 11. Data Models & Schema

### New Database Table: `task_executions`

```sql
-- Task execution tracking (Phase 1+)
CREATE TABLE task_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),

    -- Task definition
    task_type VARCHAR(50) NOT NULL,  -- workflow_subtask, chat_agent, recipe_step, background_job
    agent_id INTEGER REFERENCES agents(id),
    prompt TEXT,
    configuration JSONB DEFAULT '{}',

    -- Execution metadata
    priority VARCHAR(20) DEFAULT 'normal',
    runner_backend VARCHAR(20) NOT NULL,  -- local, queued, kubernetes

    -- Resource tracking
    resources_requested JSONB DEFAULT '{}',  -- {cpu, memory, disk}
    resources_used JSONB DEFAULT '{}',       -- Actual consumption

    -- Lifecycle
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Results
    result JSONB,
    error_message TEXT,
    tokens_used INTEGER DEFAULT 0,
    execution_time_ms INTEGER DEFAULT 0,

    -- Tracing
    parent_execution_id UUID REFERENCES task_executions(id),
    correlation_id VARCHAR(255),

    -- K8s metadata (Phase 3)
    k8s_namespace VARCHAR(255),
    k8s_job_name VARCHAR(255),
    k8s_pod_name VARCHAR(255),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_task_executions_workspace ON task_executions(workspace_id);
CREATE INDEX idx_task_executions_status ON task_executions(status);
CREATE INDEX idx_task_executions_correlation ON task_executions(correlation_id);
CREATE INDEX idx_task_executions_submitted ON task_executions(submitted_at DESC);
```

### Redis Key Structure (Phase 2+)

```
# Task queues (priority-based)
tasks:critical          → List (FIFO)
tasks:high              → List (FIFO)
tasks:normal            → List (FIFO)
tasks:low               → List (FIFO)
tasks:dead              → List (failed tasks)

# Task state
task:{task_id}:status   → Hash {status, started_at, worker_id}
task:{task_id}:result   → JSON (TTL: 1 hour)

# Workspace concurrency tracking
ws:{workspace_id}:active_tasks → Set of task_ids
ws:{workspace_id}:task_count   → Integer (current active)

# Worker health
worker:{worker_id}:heartbeat   → Timestamp (TTL: 60s)
worker:{worker_id}:tasks       → Set of task_ids being processed

# Events (pub/sub)
task:{task_id}:events          → Pub/Sub channel
ws:{workspace_id}:agent_events → Pub/Sub channel
```

---

## 12. API Changes

### New Endpoints (Phase 2+)

```
# Task management
GET    /api/tasks                        → List tasks (filtered by workspace)
GET    /api/tasks/{task_id}              → Get task details + status
POST   /api/tasks/{task_id}/cancel       → Cancel running task
GET    /api/tasks/{task_id}/logs         → Stream task logs (SSE)
GET    /api/tasks/{task_id}/events       → Stream task events (SSE)

# Worker status (admin)
GET    /api/admin/workers                → List active workers
GET    /api/admin/workers/{id}/health    → Worker health check
GET    /api/admin/queues                 → Queue depths per priority

# Workspace compute (admin)
GET    /api/admin/workspaces/{id}/usage  → Compute usage per workspace
```

### Existing Endpoint Changes

No breaking changes. The `POST /api/workflows/{id}/execute` endpoint continues to work identically — it calls `TaskRunner.submit_task()` instead of `asyncio.create_task()` internally. The execution ID and SSE streaming continue to work.

---

## 13. Security Model (Full)

### Phase 2 Security (Workers)

| Concern | Mitigation |
|---------|-----------|
| Cross-workspace data | Each task gets isolated temp dir, cleaned after completion |
| Credential leakage | LLM keys loaded per-task from workspace credentials (Fernet-encrypted) |
| Resource exhaustion | Docker resource limits per worker container |
| Network access | Workers connect to Redis + Postgres + LLM APIs only |

### Phase 3 Security (K8s)

| Concern | Mitigation |
|---------|-----------|
| Cross-workspace data | Namespace isolation + NetworkPolicy |
| Pod escape | `runAsNonRoot`, `readOnlyRootFilesystem`, no privileged containers |
| K8s API access | No service account token mounted, RBAC minimal |
| Network lateral movement | NetworkPolicy: deny all ingress, egress allow-list only |
| Secret management | K8s Secrets + External Secrets Operator (AWS SM / Vault) |
| Image supply chain | Signed images, vulnerability scanning (Trivy) |
| DDoS via task submission | Per-workspace rate limits + queue depth limits |

### Compliance Alignment

| Standard | Phase 2 Coverage | Phase 3 Coverage |
|----------|-----------------|-----------------|
| SOC 2 Type II | Partial (audit logs, encryption) | Full (isolation, access controls) |
| GDPR | Data residency via region selection | Per-namespace data isolation |
| ISO 27001 | Encryption at rest/transit | Full security controls |
| HIPAA | Not applicable yet | Dedicated clusters (Phase 4) |

---

## 14. Cost Analysis

### Phase 1: No Change
- Railway: ~$20-40/mo (current pilot)
- No additional infra cost

### Phase 2: Railway + Physical Workspaces (Pilot — 15 Users)
- Backend service (API): ~$10/mo
- Workspace worker (1 replica): ~$10-15/mo
- Agent-opt worker (existing): ~$5/mo
- Persistent volume (100GB): ~$2/mo
- Postgres: ~$10/mo
- Redis: ~$5/mo
- **Total: ~$45-55/mo**
- Storage per workspace: 5GB default (15 users × 5GB = 75GB max capacity)
- Per-workspace cost: ~$3/mo (amortized across pilot users)

### Phase 3: Managed Kubernetes
- **GKE Autopilot** (recommended):
  - Control plane: $72/mo (free tier available)
  - Pods: $0.0445/vCPU-hour + $0.0049/GB-hour
  - Estimated for 100 workspaces (avg 2 tasks/day, 10 min each):
    - ~$150-250/mo compute
    - ~$50/mo networking
    - **Total: ~$300-400/mo**
- **AWS EKS + Karpenter**:
  - Control plane: $72/mo
  - Spot instances for workers: ~$200/mo
  - **Total: ~$350-500/mo**

### Cost Per Workspace (Phase 3)

| Plan | Est. Monthly Compute | Charge to Customer |
|------|---------------------|-------------------|
| Starter (2 tasks/day) | ~$1.50 | $29/mo |
| Pro (20 tasks/day) | ~$15 | $99/mo |
| Enterprise (100 tasks/day) | ~$75 | $499/mo |

Healthy margins at scale. The ephemeral model means idle workspaces cost $0.

---

## 15. Implementation Roadmap

### Phase 1: TaskRunner Abstraction (Week 1)
**Effort: 2-3 days**

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Models + ABC + LocalTaskRunner | `core/task_runner/` package |
| 2 | Factory + Configuration | `get_task_runner()`, env config |
| 2 | Integration point documentation | Call site inventory |
| 3 | Tests | Unit tests for LocalTaskRunner |

### Phase 2: Physical Workspaces + Queue Workers (Weeks 2-6)
**Effort: 3-4 weeks**

| Week | Task | Deliverable |
|------|------|-------------|
| 2 | ARQ integration + QueuedTaskRunner | `core/task_runner/queued.py` |
| 2 | WorkspaceManager (dir provisioning, quotas) | `services/workspace-worker/workspace_manager.py` |
| 2-3 | WorkspaceToolExecutor (sandboxed commands) | `services/workspace-worker/executor.py` |
| 3 | Worker Dockerfile + DevOps toolchain | `services/workspace-worker/Dockerfile` |
| 3 | ARQ consumer entry point | `services/workspace-worker/main.py` |
| 3-4 | Wire TaskRunner into execution pipeline | Replace `asyncio.create_task()` calls |
| 4 | Tool routing (API vs Worker split) | Tool registry update + dispatcher |
| 4 | Storage quota enforcement + command whitelist | Security layer |
| 5 | Credential injection (SSH keys, git config) | Per-workspace credential flow |
| 5 | Docker Compose + Railway deployment | Multi-service with persistent volume |
| 5-6 | Path traversal hardening + security testing | Penetration test workspace isolation |
| 6 | End-to-end testing (clone → test → fix → push) | DevOps workflow validation |

### Phase 3: Kubernetes (Months 3-6)
**Effort: 4-6 weeks**

| Month | Task | Deliverable |
|-------|------|-------------|
| 3 | KubernetesTaskRunner | `core/task_runner/kubernetes.py` |
| 3 | Task Controller | `worker/controller.py` |
| 3-4 | Namespace provisioning | Auto-namespace per workspace |
| 3-4 | Migrate workspace volumes to PersistentVolumeClaims | Per-workspace PVCs |
| 4 | NetworkPolicy + RBAC | Security boundaries |
| 4-5 | KEDA autoscaling | Scale from zero |
| 5 | Helm chart | Deployment package |
| 5-6 | Load testing + hardening | Production readiness |

### Phase 4: Enterprise (Month 6+)
**Effort: Ongoing**

| Quarter | Task | Deliverable |
|---------|------|-------------|
| Q3 2026 | Dedicated node pools | Enterprise isolation |
| Q3 2026 | External Secrets Operator | Vault/AWS SM integration |
| Q3 2026 | Per-workspace PersistentVolumeClaims | True storage isolation |
| Q4 2026 | BYOC agent deployment | Customer-cluster support |
| Q4 2026 | Helm chart for air-gap | Self-hosted package |

---

## 16. Risk Assessment

| Risk | Likelihood | Impact | Phase | Mitigation |
|------|-----------|--------|-------|-----------|
| Phase 2 introduces latency (queue overhead) | Medium | Low | 2 | Queue adds ~50-100ms; acceptable for agent tasks (seconds-long) |
| Worker container crashes during task | Medium | Medium | 2 | Task heartbeat + auto-requeue; result idempotency |
| **Path traversal escape from workspace** | Low | **Critical** | 2 | `resolve_safe_path()` with symlink resolution + null byte check. Security testing before pilot launch |
| **Storage exhaustion (large repo clones)** | Medium | Medium | 2 | Quota enforcement before each task; cleanup tooling for old repos; monitoring alerts at 80% |
| **Cross-workspace data leak (shared worker)** | Low | High | 2 | All paths validated per-request; subprocess `cwd` pinned; credentials cleaned per-task |
| **Command injection via agent tool calls** | Low | High | 2 | Whitelist enforcement; blocked patterns list; no shell expansion on user input |
| **Railway volume data loss** | Low | High | 2 | Railway volumes persist across deploys. Backup strategy: periodic `tar` to S3 for critical workspaces |
| **Single worker bottleneck (15 users)** | Medium | Low | 2 | 1 worker handles ~3 concurrent tasks; pilot users unlikely to saturate. Scale to 2 replicas if needed |
| K8s complexity slows feature development | Medium | High | 3 | Phase 3 only when revenue justifies; managed K8s (Autopilot) reduces ops |
| Pod startup latency (cold start) | Medium | Medium | 3 | Pre-pull images on nodes; KEDA warm pool |
| Redis as task queue: message loss | Low | High | 2-3 | Redis AOF persistence; critical tasks also written to Postgres |
| Namespace proliferation (1000+ workspaces) | Low | Medium | 3 | Lazy provisioning; cleanup inactive namespaces after 30 days |
| Cost overrun on K8s | Medium | Medium | 3 | KEDA scale-to-zero; spot instances; per-workspace billing |

---

## Appendix A: Technology Decisions

### Why GKE Autopilot (Recommended for Phase 3)

| Factor | GKE Autopilot | EKS + Karpenter | AKS |
|--------|---------------|-----------------|-----|
| Node management | Fully managed | Self-managed (Karpenter helps) | Mostly managed |
| Pay-per-pod | Yes | No (pay per node) | No |
| Scale to zero | Yes | Yes (with Karpenter) | Partial |
| Setup complexity | Low | Medium | Medium |
| Cost (small scale) | Lowest | Higher (min node) | Medium |
| GPU support | Yes | Yes | Yes |
| Banking compliance | GCP FedRAMP | AWS GovCloud | Azure Gov |

Given your banking IT background and that Azure/AWS are likely familiar, either GKE Autopilot (lowest ops) or EKS + Karpenter (most flexible) are strong choices.

### Why ARQ over Celery (Phase 2)

- Native asyncio (matches FastAPI)
- Minimal dependencies (just `arq` + `redis`)
- Result backend built-in
- Simple configuration
- Lower memory footprint
- We already depend on Redis

### Why Not Serverless Functions (Lambda/Cloud Functions)

- 15-minute timeout limit (agent tasks can run longer)
- Cold start latency (3-10s)
- No persistent filesystem (can't clone repos)
- Limited to 10GB memory
- No GPU access
- Vendor lock-in

---

## Appendix B: Monitoring & Observability

### Metrics to Track

```
# Task lifecycle
automatos_task_submitted_total{workspace, task_type, priority}
automatos_task_completed_total{workspace, task_type, status}
automatos_task_duration_seconds{workspace, task_type}
automatos_task_queue_depth{queue, priority}

# Worker health
automatos_worker_active_tasks{worker_id}
automatos_worker_heartbeat_age{worker_id}

# Resource usage
automatos_workspace_cpu_usage{workspace, plan}
automatos_workspace_memory_usage{workspace, plan}
automatos_workspace_active_tasks{workspace}

# K8s specific (Phase 3)
automatos_pod_startup_seconds{namespace}
automatos_pod_scheduling_delay_seconds{namespace}
automatos_namespace_quota_usage{namespace, resource}
```

### Dashboard (Grafana)

- Task throughput (tasks/min by workspace and type)
- Queue depth over time (P2)
- Pod scheduling latency (P3)
- Per-workspace resource consumption
- Error rates and failure reasons
- Cost attribution per workspace

---

*This PRD is a living document. Update as phases progress.*
