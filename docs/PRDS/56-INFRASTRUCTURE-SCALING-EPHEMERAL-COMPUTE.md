# PRD-56: Infrastructure Scaling & Ephemeral Agent Compute

**Version:** 1.0
**Status:** Planning Phase
**Date:** February 15, 2026
**Author:** Automatos Core Team
**Prerequisites:** PRD-37 (SaaS Foundation), PRD-54 (LLM Marketplace)
**Blocks:** None (Foundation for enterprise scaling)

---

## Executive Summary

This PRD defines the infrastructure evolution path for Automatos AI — from the current Railway-hosted pilot to a fully scalable, workspace-isolated, enterprise-grade compute platform. The core architectural change: **agent tasks execute in ephemeral, isolated compute environments** rather than in-process with the API server.

### The Problem

Today, all agent execution (workflows, subtasks, tool calls) runs inside the FastAPI process via `asyncio.create_task()`. This means:

- **No isolation** — One workspace's heavy agent task starves all others
- **No persistence** — Tasks are lost if the server restarts
- **No resource limits** — Can't enforce plan-tier CPU/memory caps
- **No security boundary** — Agent code execution (shell tools, file ops) shares the API server's filesystem and network
- **No horizontal scaling** — Everything runs in one process on one container
- **No auditability** — No infrastructure-level task lifecycle tracking

### The Solution

A **3-phase migration** introducing a `TaskRunner` abstraction that decouples task dispatch from task execution:

| Phase | Infrastructure | Timeline | User Scale |
|-------|---------------|----------|------------|
| **Phase 1** (Now) | Railway + `LocalTaskRunner` | Week 1 | Pilot (<50 users) |
| **Phase 2** (Soft Launch) | Railway/VPS + Redis Queue + Worker Containers | Weeks 2-6 | Early adopters (50-500 users) |
| **Phase 3** (Scale) | Managed Kubernetes + Ephemeral Pods | Months 3-6 | Growth (500+ workspaces) |
| **Phase 4** (Enterprise) | Multi-cluster / Bring-Your-Own-Cloud | Month 6+ | Enterprise tenants |

### Key Architecture Decision

Introduce a **`TaskRunner` interface** at the boundary between task dispatch and task execution. All agent work flows through this interface. Swap implementations without touching business logic:

```
LocalTaskRunner     → asyncio (current behavior, Railway-compatible)
QueuedTaskRunner    → Redis queue + worker containers (Phase 2)
KubernetesTaskRunner → K8s Jobs with ephemeral pods (Phase 3)
```

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Phase 1: TaskRunner Abstraction](#3-phase-1-taskrunner-abstraction)
4. [Phase 2: Queue-Based Worker Isolation](#4-phase-2-queue-based-worker-isolation)
5. [Phase 3: Kubernetes Ephemeral Pods](#5-phase-3-kubernetes-ephemeral-pods)
6. [Phase 4: Enterprise Multi-Tenant](#6-phase-4-enterprise-multi-tenant)
7. [Data Models & Schema](#7-data-models--schema)
8. [API Changes](#8-api-changes)
9. [Security Model](#9-security-model)
10. [Cost Analysis](#10-cost-analysis)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Risk Assessment](#12-risk-assessment)

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

## 4. Phase 2: Queue-Based Worker Isolation (Weeks 2-6)

### Goal

Move agent task execution from the API process to dedicated worker containers connected via a Redis task queue. Each task runs in an isolated worker with its own filesystem and resource limits.

### Architecture

```
┌─────────────────┐    Redis Queue     ┌──────────────────┐
│   FastAPI API    │ ──── tasks ──────→ │  Worker Pool     │
│   (Control)      │ ←── results ────── │  (2-8 workers)   │
│                  │ ←── events ──────  │                  │
│ QueuedTaskRunner │    (pub/sub)       │  ARQ Consumer    │
└─────────────────┘                     │  AgentFactory    │
                                        │  Tool Executor   │
                                        └──────────────────┘
```

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

#### US-06: Worker Container
**Description:** Standalone worker process that consumes tasks from Redis queue and executes agent work.

**Acceptance Criteria:**
- [ ] `worker/main.py` — ARQ worker entry point
- [ ] Reuses `AgentFactory`, `UnifiedToolExecutor`, LLM clients
- [ ] Each task gets isolated temp directory (`/tmp/task_{id}/`)
- [ ] Temp directory cleaned after task completion
- [ ] Worker reports health to Redis (heartbeat)
- [ ] Graceful shutdown (finish current task, don't accept new)
- [ ] Docker container: `Dockerfile.worker` (same base as backend)

#### US-07: Worker Dockerfile
**Description:** Docker image for worker containers with all agent dependencies.

**Acceptance Criteria:**
- [ ] `Dockerfile.worker` — Multi-stage build
- [ ] Includes: Python 3.11, git, tesseract, ghostscript (same as backend)
- [ ] Non-root user execution
- [ ] Configurable via env vars: `WORKER_CONCURRENCY`, `WORKER_QUEUES`
- [ ] Health check endpoint
- [ ] Resource limits in docker-compose: `deploy.resources.limits`

#### US-08: Task Persistence & Recovery
**Description:** Tasks survive API server restarts and worker crashes.

**Acceptance Criteria:**
- [ ] Queued tasks persist in Redis (AOF/RDB)
- [ ] Running tasks have heartbeat timeout (60s)
- [ ] Orphaned tasks (no heartbeat) re-queued automatically
- [ ] Failed tasks stored with error info for retry/debugging
- [ ] Task history queryable via API

#### US-09: Per-Workspace Queue Limits
**Description:** Enforce concurrent task limits based on workspace plan tier.

**Acceptance Criteria:**
- [ ] Starter plan: 2 concurrent tasks max
- [ ] Pro plan: 10 concurrent tasks max
- [ ] Enterprise plan: 50 concurrent tasks max (configurable)
- [ ] Tasks exceeding limit are queued (not rejected)
- [ ] Workspace usage tracked in Redis counter

#### US-10: Docker Compose Worker Profile
**Description:** Add worker containers to docker-compose for local development.

**Acceptance Criteria:**
- [ ] `docker-compose.yml` updated with worker service
- [ ] `--profile workers` to enable worker containers
- [ ] Default: 2 worker replicas
- [ ] Environment: shares backend env vars + worker-specific config
- [ ] Volume mount for development hot-reload

### Phase 2 Infrastructure

```yaml
# docker-compose.yml additions
services:
  worker:
    build:
      context: ./orchestrator
      dockerfile: Dockerfile.worker
    deploy:
      replicas: ${WORKER_REPLICAS:-2}
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
    environment:
      WORKER_CONCURRENCY: ${WORKER_CONCURRENCY:-5}
      WORKER_QUEUES: "critical,high,normal,low"
      DATABASE_URL: ${DATABASE_URL}
      REDIS_URL: ${REDIS_URL}
      # LLM keys inherited from backend
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    profiles: ["workers"]
```

### Phase 2 on Railway

Railway supports multiple services per project. Workers deploy as separate services:
- **backend** service: API server (no agent execution)
- **worker** service: 2+ replicas consuming Redis queue
- **postgres**, **redis**: existing managed services

Cost impact: ~$10-20/mo additional for 2 worker replicas on Railway.

---

## 5. Phase 3: Kubernetes Ephemeral Pods (Months 3-6)

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

## 6. Phase 4: Enterprise Multi-Tenant (Month 6+)

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

## 7. Data Models & Schema

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

## 8. API Changes

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

## 9. Security Model

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

## 10. Cost Analysis

### Phase 1: No Change
- Railway: ~$20-40/mo (current pilot)
- No additional infra cost

### Phase 2: Railway + Workers
- Backend service: ~$10/mo
- 2 Worker replicas: ~$20/mo
- Postgres: ~$10/mo
- Redis: ~$5/mo
- **Total: ~$45-60/mo**
- Per-workspace cost: negligible (shared workers)

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

## 11. Implementation Roadmap

### Phase 1: TaskRunner Abstraction (Week 1)
**Effort: 2-3 days**

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Models + ABC + LocalTaskRunner | `core/task_runner/` package |
| 2 | Factory + Configuration | `get_task_runner()`, env config |
| 2 | Integration point documentation | Call site inventory |
| 3 | Tests | Unit tests for LocalTaskRunner |

### Phase 2: Queue + Workers (Weeks 2-6)
**Effort: 2-3 weeks**

| Week | Task | Deliverable |
|------|------|-------------|
| 2 | ARQ integration + QueuedTaskRunner | `core/task_runner/queued.py` |
| 2-3 | Worker container + Dockerfile | `worker/`, `Dockerfile.worker` |
| 3 | Wire TaskRunner into execution pipeline | Replace `asyncio.create_task()` calls |
| 4 | Workspace queue limits + priority | Per-plan enforcement |
| 4-5 | Task persistence + recovery | Restart resilience |
| 5 | Docker Compose + Railway deployment | Multi-service deployment |
| 6 | Testing + monitoring | Load testing, metrics |

### Phase 3: Kubernetes (Months 3-6)
**Effort: 4-6 weeks**

| Month | Task | Deliverable |
|-------|------|-------------|
| 3 | KubernetesTaskRunner | `core/task_runner/kubernetes.py` |
| 3 | Task Controller | `worker/controller.py` |
| 3-4 | Namespace provisioning | Auto-namespace per workspace |
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
| Q4 2026 | BYOC agent deployment | Customer-cluster support |
| Q4 2026 | Helm chart for air-gap | Self-hosted package |

---

## 12. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Phase 2 introduces latency (queue overhead) | Medium | Low | Queue adds ~50-100ms; acceptable for agent tasks (seconds-long) |
| Worker container crashes during task | Medium | Medium | Task heartbeat + auto-requeue; result idempotency |
| K8s complexity slows feature development | Medium | High | Phase 3 only when revenue justifies; managed K8s (Autopilot) reduces ops |
| Pod startup latency (cold start) | Medium | Medium | Pre-pull images on nodes; KEDA warm pool |
| Redis as task queue: message loss | Low | High | Redis AOF persistence; critical tasks also written to Postgres |
| Namespace proliferation (1000+ workspaces) | Low | Medium | Lazy provisioning; cleanup inactive namespaces after 30 days |
| Cost overrun on K8s | Medium | Medium | KEDA scale-to-zero; spot instances; per-workspace billing |

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
