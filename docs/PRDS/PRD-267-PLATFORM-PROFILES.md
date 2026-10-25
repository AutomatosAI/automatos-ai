# PRD-267: Platform profiles: one codebase on containers or on Cloudflare

**Status:** Draft for Gerard's approval, 24 Oct 2026. Nothing here is approved; every scope choice is in §11 with a recommendation. · **Owner:** Gerard (scope and the §11 answers) · **Proposed by:** daarthur · **Written:** 24 Oct 2026, grounded on `origin/main @ 0c0afb02f` and on Cloudflare's docs, changelog, blog and trust hub as published on 24 Oct 2026.
**Source:** two write-ups from the same session: *Automatos on Cloudflare*, a reference stack for the Orderly partnership (claude.ai/artifact/5kyr2xMnFrhkdoLx9qb2hW), and *All-Cloudflare Automatos*, a capability-by-capability parity study (claude.ai/artifact/9hBKPbunj9ANfXqotfZcjk). A read-only survey of `origin/main` mapped every platform dependency for this PRD.
**Type:** Refactor and consolidation first (W0–W2: today's code moves behind ports, with no new vendor), then extension (W3–W8: a second set of adapters for Cloudflare). No existing behaviour is removed from the container model.
**Scope:** automatos-ai. The `edge/` TypeScript project it adds lives in this repo. This draft proposes leaving the Orderly partnership's own work (organisations, webhooks, the approvals API for server keys, cost export) to its own PRD; §4.11 says where they meet, and §11-Q13 asks Gerard to confirm the boundary.
**Needs Gerard's approval before code (AGENTS.md "stop and ask first"):**
- a new shared package used by the API and the services, and the build-context change it needs (§4.2, §11-Q12);
- tenancy (§4.6, workspace cells);
- authentication (§4.10, a trusted-proxy edition);
- quotas, which guard spend (US-007 moves them behind a port);
- removing a user-data store if Qdrant and mem0 retire (§11-Q5);
- the embedding-model change and re-embedding (§4.8), which touches user data;
- new dependencies (§8.3).

Labels: **[T]** read in code on `origin/main`, 24 Oct; **[C]** from Cloudflare's published docs, 24 Oct; **[I]** inferred.

## 1. Introduction

Orderly runs on Cloudflare and wants its agent layer there too. Automatos is built for containers: compose for the local edition, the Helm chart for platform teams, Railway for hosted Studio. Those deployments must keep working on the same code.

Cloudflare can run the web app, the API and an isolated sandbox per workspace. Its stateful services are not drop-in replacements for what Automatos uses, and the code has no seam where a replacement could go:

| Dependency | Today on `origin/main` [T] | Seam |
|---|---|---|
| Postgres | One sync SQLAlchemy engine in `orchestrator/core/database/database.py`, imported by about 280 non-test modules. JSONB in 117 files (29 of 53 model files); raw SQL strings in `text()` in 120 files (about 175 counting every `text(` call); `SKIP LOCKED` in 8; advisory locks in 11; LISTEN/NOTIFY threads in the board dispatcher. 16 raw `psycopg2.connect` calls in 6 files bypass the engine, plus one in the workspace worker | none |
| Redis | `orchestrator/core/redis/client.py` hands out the raw client. Three queue protocols, three pub/sub feeds, seven private `_get_redis` helpers, ten direct imports. The workspace worker polls `workspace:tasks:*` with `RPOP` (`services/workspace-worker/main.py:195`) and keeps task status, results and events under `workspace:task:{id}:*`; `orchestrator/api/workspace_github.py:294` is the only producer | none |
| Scheduling | APScheduler behind an `fcntl` lock on `/tmp/automatos_scheduler.lock` (`orchestrator/main.py:410`). The lock is per host, so every replica runs a scheduler; the Helm chart pins `api.replicas: 1` for that reason; six modules use `RedisJobStore` | partial |
| Background work | `create_task` in 44 files, threads in 12; the task runner factory (`core/task_runner`, `TASK_RUNNER_BACKEND`) has 2 callers | partial |
| Live events | Six SSE endpoints over four transports: pg LISTEN, Redis pub/sub, in-process queues, polling | none |
| API rate limits | slowapi `Limiter` with no storage (`orchestrator/main.py:1042`), so limits are per process | none |
| Files | `core/storage/s3.py`, with `S3_ENDPOINT_URL` selecting MinIO or AWS | clean |
| Vectors | `modules/search/vector_store` backends (`pgvector`, `s3_vectors`), but raw pgvector SQL in five other modules and `S3_VECTORS_ENABLED` checked in four places; Qdrant memory is a separate client | partial |
| Credentials | Fernet in `core/credentials/encryption.py`, one global key | clean |
| Shared code | `automatos_logging.py` exists in five places (three services, `services/shared/`, `orchestrator/core/monitoring/`) as three different versions; nothing else is shared between the API and `services/*`. Each image builds from its own folder (`./orchestrator`, `./services/*`, `./frontend`) | none |

The pattern this PRD needs already exists in five places [T]: the vector-store backend factory, the task runner factory (`local`, `queued`, `kubernetes`), `core/ports/context.py` with its Redis and vector-field adapters (`SHARED_CONTEXT_BACKEND`), the S3 endpoint switch, and the OTel tracer seam (`core/observability/tracer.py`). This PRD makes it the rule.

Most of the first three waves fix problems the container model has today, Cloudflare or not: a scheduler per replica, per-process rate limits, four streaming transports, a worker that polls.

## 2. Goals

- One codebase runs on the **containers** profile (today's behaviour, the default) or the **cloudflare** profile, selected by one setting.
- The container model loses nothing. Compose, the Helm chart and Railway keep working at every wave, and the local edition stays one `docker compose up`.
- The Helm chart can run more than one API replica without a duplicate scheduler or per-process rate limits (closes the K1 blocker in the platform-engineering notes).
- On the cloudflare profile, every workspace's agent code runs in its own sandbox, with egress allowlisted and credentials added outside it.
- On the cloudflare profile, Redis, Qdrant and the singleton scheduler are gone.
- Every port has one contract test suite that passes against both profiles in CI.
- The records database decision (§4.7) is made on a measured prototype, not on this document.

## 3. Principles

1. **Ports, not branches.** Code depends on a port (a `Protocol`). Only `automatos_platform/profile.py` chooses an adapter. No `if PLATFORM_PROFILE` anywhere else.
2. **Profile is separate from edition.** `PLATFORM_PROFILE` (`containers`, `cloudflare`) and `AUTH_EDITION` (`local`, `saas`) are independent. Dozens of files already branch on edition (46 read `AUTH_EDITION` directly, more through `IS_LOCAL_EDITION` and `IS_SAAS`) [T]; a profile must not multiply them.
3. **The containers adapter is today's code, moved.** W0–W2 move code behind ports without changing behaviour, except the changes FR-2 lists, each proved by a test.
4. **Replace, don't shim** (AGENTS.md). When a port lands, every direct use of what it wraps moves behind it in the same wave, and a lint rule stops new ones.
5. **Every adapter passes the port's contract tests.** The containers lane runs from W0; the cloudflare lane (`wrangler dev`, Miniflare) from W3, when the first cloudflare adapter lands.
6. **Cloudflare adapters speak HTTP to one bridge.** A container reaches Cloudflare's services only through outbound handlers, which see HTTP on port 80 and HTTPS on 443 and nothing else [C]. So every cloudflare adapter is a thin HTTP client, and the bridge (§4.5) is the one place that holds bindings.

## 4. Design

### 4.1 The setting

`PLATFORM_PROFILE` in `config.py`: `containers` (default) or `cloudflare`; anything else fails at boot. It goes in `orchestrator/reports/config-surface.json`. The workspace worker and the other services read it through their own config modules, as they do today.

### 4.2 The shared package

```
packages/automatos-platform/          # installed into the API image and every service image
  automatos_platform/
    ports/          # Protocols only, one module per port (§4.3)
    adapters/
      containers/   # today's code, moved: Postgres, Redis, S3, pgvector, APScheduler, Fernet…
      cloudflare/   # HTTP clients for the bridge
    profile.py      # the only module that imports adapters
    logging.py      # replaces the five copies of automatos_logging.py
    metrics.py      # replaces services/shared/automatos_metrics.py
  contract_tests/   # one suite per port, parametrised by profile

edge/               # TypeScript; deployed only on the cloudflare profile
  bridge/           # outbound handlers, one virtual host per port (§4.5)
  cell/             # the workspace Durable Object (§4.6)
  router/  vault/  ingress/
  wrangler.jsonc
```

- **Build contexts.** Today each image builds from its own folder, so none can see `packages/` [T]. Two ways in, chosen in §11-Q12:
  - *Repo-root contexts (recommended):* compose, `images.yml` and Railway build from the repo root with the same Dockerfiles, and each Dockerfile gets its own `<Dockerfile>.dockerignore` (BuildKit) so the context stays as small as today. The package installs with `pip install --no-deps ./packages/automatos-platform`, so the images' hashed locks don't change.
  - *A synced copy:* a script copies the package into each build context and CI fails if a copy differs. Simpler to land, but it is the copying this PRD removes.
- Its dependencies are a subset of what the images already install; the cloudflare adapters need only `httpx` [T]. The package is pure Python, so images stay within PRD-262's budgets.

### 4.3 The ports

| Port | containers adapter (today) | cloudflare adapter | Seam today [T] |
|---|---|---|---|
| Records (the engine and session) | SQLAlchemy on Postgres | §4.7: PlanetScale Postgres (Path 1) or the workspace cell (Path 2) | none |
| Cache | Redis via `core/cache/service.py` | KV for shared caches; cell memory for workspace caches | partial |
| Queue | Redis lists (three protocols today, one after W1) | Queues | none |
| Event stream | Redis pub/sub | The cell's event stream | none |
| Locks and quotas | Postgres advisory locks, Redis `SETNX`, sorted-set windows | Cell methods; the Rate Limiting binding for abuse limits | none |
| Schedules | APScheduler with a database leader (W1) | Cell alarms and Cron Triggers | partial |
| Durable runs | The `local` task runner | Workflows | partial |
| Objects | `core/storage/s3.py` | R2 through the same S3 client (endpoint only) | clean |
| Vectors | pgvector (and S3 Vectors while it exists) | Vectorize, a namespace per workspace | partial |
| Keyword search | `tsvector` | `tsvector` on Path 1; FTS5 in the cell on Path 2 | partial |
| Sandbox | The workspace worker's HTTP API (`core/workspace_client.py`) | Sandbox SDK, driven by the cell | partial |
| Vault | Fernet, key from env or file | The Vault Worker (§4.9) | clean |
| Browser | Playwright in the worker | Browser Run | clean |
| Telemetry | OTel and Prometheus | OTel and Analytics Engine | clean |

Each port module carries its contract: what a call guarantees (ordering, delivery at least once or at most once, consistency), what it may raise, and the limits an adapter must document. Example:

```python
class EventStream(Protocol):
    """Publish and follow a workspace's live events (chat tokens, canvas, board).

    Ordered per (workspace, channel). A follower that passes `after` resumes
    after that event id; events older than the retention window may be gone.
    """

    def publish(self, workspace_id: UUID, channel: str, event: Mapping[str, Any]) -> str: ...

    def follow(
        self, workspace_id: UUID, channel: str, after: str | None
    ) -> AsyncIterator[Event]: ...
```

### 4.4 Enforcement

- **import-linter, not ruff.** `orchestrator/.importlinter` has `root_package = modules` and one independence contract, and `ruff.toml` exempts `services/**` from `TID251` entirely (each service reads its own environment) [T]. So the bans live in a second import-linter config, `.importlinter-platform` at the repo root, run once per code root (the orchestrator and each service) with that root's top-level packages and `include_external_packages = True`. Two `forbidden` contracts:
  - nothing imports `automatos_platform.adapters` except `automatos_platform.profile`;
  - nothing imports `redis`, `boto3`, `psycopg2`, `asyncpg` or `apscheduler` (the adapters live in the package, outside these roots).
- **A ratchet, like PRD-211's.** Today's imports are carried as `ignore_imports` so the contracts are green from W0; each wave deletes the entries for what it moved, and new imports fail CI. Bedrock's `boto3` client stays as a named, permanent entry.
- **Contract tests:** `packages/automatos-platform/contract_tests/`, one suite per port. The containers lane (Postgres and Redis service containers) starts in W0; the cloudflare lane (`wrangler dev`) starts in W3.

### 4.5 The platform bridge (cloudflare profile)

A Cloudflare container reaches bindings (KV, Queues, Durable Objects, Vectorize, Secrets Store) only through outbound handlers: it sends HTTP to a virtual hostname and a handler in the Workers runtime makes the binding call [C]. `edge/bridge` defines one virtual host per port: `records.cf`, `events.cf`, `cache.cf`, `queue.cf`, `vectors.cf`, `vault.cf`, `telemetry.cf`.

What outbound handlers cover [C]:
- **Ports 80 and 443 only.** Other ports never reach a handler. The API container connects to PlanetScale (5432) and model providers directly, so it runs with internet enabled.
- **HTTPS only when asked.** With `interceptHttps = true`, Cloudflare injects a CA at `/etc/cloudflare/certs/cloudflare-containers-ca.crt`; the image must add it to its trust store in the entrypoint (and point `REQUESTS_CA_BUNDLE` and `NODE_EXTRA_CA_CERTS` at it).
- **Allowlists hold only with internet off.** While `enableInternet` is true, connections on other ports bypass `deniedHosts`. With it false, other ports are denied and DNS is restricted, and only `allowedHosts` or a handler lets traffic out.

- **Identity comes from the caller, not the request.** The handler knows which container is calling. A workspace's sandbox can reach only its own cell, its own vector namespace and its own vault keys, whatever the request says.
- **The API container** is a shared pool, so its calls carry the workspace in the path, and the bridge checks it against a short-lived token the router issued for that request [I].

**The caller contract.** Before the bridge calls any binding:
- **A sandbox** is identified by the container the handler intercepted, which is the cell's own container. It carries no token, and the bridge uses no workspace id from the request.
- **An API container** presents the router's token. The token is signed by a key only the router and bridge hold, and binds:
  - the workspace;
  - the request id;
  - the calling container's id;
  - the audience (the bridge virtual host it may call);
  - an expiry no longer than the request's own timeout, five minutes at most.
- **Replay.** The token is reusable across the bridge calls of one request, because one request makes many. It is refused from any other container, for any other workspace or virtual host, and after it expires.
- **Rejections.** A missing, invalid, expired, replayed or mismatched token is rejected with 401 or 403, and the bridge logs it, without the token.
- **Latency is unmeasured.** W4's acceptance includes a measurement (§10).

### 4.6 Workspace cells (cloudflare profile)

The platform-engineering notes call the workspace the isolation unit; today it is a filter on shared tables. On the cloudflare profile, each workspace is a Durable Object, its **cell**, created on first use in a jurisdiction (`eu`, `us` or `fedramp`) [C]:

| The cell holds | Replaces [T] |
|---|---|
| The workspace's schedule, in its own SQLite, with one alarm set for the earliest job | APScheduler per replica, `RedisJobStore` |
| The workspace's event stream (WebSockets with hibernation), with a resume cursor | pg LISTEN, Redis pub/sub, in-process queues, polling |
| Locks, dedup markers and exact quotas, as methods: an object handles one request at a time | advisory locks, `SETNX`, sorted-set windows |
| The board's next-task claim | `SKIP LOCKED` and `pg_notify` wake-ups |
| The workspace's sandbox container (Sandbox SDK 1.0 lets the object drive `ctx.container`) [C] | the shared worker polling `workspace:tasks:*` |
| Only if Path 2 is chosen after the W7 prototype: the workspace's records (SQLite, 10 GB, 30-day point-in-time recovery) [C] | Postgres rows filtered by `workspace_id` |

The cell lands **without records first** (W4). Records stay in Postgres unless Gerard chooses Path 2 at the W7 gate.

Costs, each answered in a story: cross-workspace reads (org dashboards, admin, billing) need a fan-out or a read model; the API cannot hold a transaction open on a cell across calls, so writes are single atomic batches or move into the cell; Python Durable Objects are documented but their coverage by the Python Workers GA is not stated, and no memory limit is published [C]; schema upgrades run per cell on wake.

### 4.7 Records: two paths, decided at a gate

**Path 1, Postgres billed through Cloudflare.** PlanetScale Postgres (17 or 18) bought through the Cloudflare account [C]. No rewrite. Not operated by Cloudflare, and outside its compliance lists and data-localisation controls. This is the cloudflare profile's records adapter from W3.

**Path 2, cells plus a D1 control plane.** Workspace records in each cell; organisations, users, the workspace registry and the marketplace in D1. It needs a dialect-neutral data layer. The order puts the evidence before the cost:

*Before the gate (useful on Postgres alone, approved with W0–W2):*
1. **Coordination behind ports:** `SKIP LOCKED`, advisory locks and LISTEN/NOTIFY move to the Queue, Locks and Event stream ports (US-004, US-006, US-007).
2. **One way in:** the raw `psycopg2.connect` calls and the `asyncpg` paths go through the engine (US-011b).

These two are what multiple replicas need.

*The gate (W7, US-018):* one workspace runs on a prototype cell with records, with only the tables that chat, the board and schedules touch made dialect-neutral, and is measured against §10's bars. Gerard decides Path 2 then (§11-Q3).

*After the gate, only if Path 2 is chosen (US-019):*
3. **Models:** JSONB becomes `JSON().with_variant(JSONB, "postgresql")` (117 files, 29 of 53 model files); Postgres `UUID` becomes `sa.Uuid`; `ARRAY` becomes JSON.
4. **Raw SQL into repositories:** raw SQL in `text()` in 120–175 files. ORM `.ilike()` already compiles for SQLite, and SQLAlchemy's SQLite `insert` supports `on_conflict_do_update`.
5. **Workspaces move one at a time:** export from Postgres, import into the workspace's cell.

**Schema on Postgres is unchanged.** Migrations cannot replay from an empty database, so a fresh database is built by `scripts/init_fresh_db` and stamped, then Alembic runs (`orchestrator/docker-entrypoint.sh`), and `create_tables()` also runs at boot (`orchestrator/main.py:283`) [T]. This PRD changes none of that. Cells build their schema from the same models and upgrade it on wake.

### 4.8 Vectors and embeddings

Vectorize allows at most 1536 dimensions [C]; Automatos uses 2048 (`VECTOR_STORE_DIMENSIONS`, `FIELD_EMBEDDING_DIM`, `S3_VECTORS_DIMENSION`) [T]. The cloudflare profile needs a model at or below 1536: Workers AI offers bge-m3 and qwen3-embedding-0.6b (1024 per their model cards) [C]. Changing the model means re-embedding every document and memory, so it is §11-Q4. Vectorize has no keyword search [C]; the existing BM25 + vector merge (`modules/rag/bm25_leg.py`) stays. Its keyword leg stays on Postgres `tsvector` on Path 1, and moves to FTS5 in the cell only on Path 2.

### 4.9 Vault

Cloudflare has no general KMS or HSM; its HSM support covers TLS keys only, and Secrets Store is beta with 100 secrets per account [C]. On the cloudflare profile, a **Vault Worker** is the only code holding the master key (from Secrets Store). Each cell keeps its workspace's data key, wrapped by the master key. Containers ask the vault (through the bridge) to decrypt for their own workspace, and every use is logged to an R2 bucket with a bucket lock. Customer-held keys (BYOK, an external key manager) need an outside KMS behind the same port: §11-Q8.

### 4.10 Identity

Clerk stays for hosted self-serve sign-up: Cloudflare has nothing equivalent [C]. For enterprise users, a **trusted-proxy** lane accepts an identity asserted by a proxy in front of Automatos: Cloudflare Access (`Cf-Access-Jwt-Assertion`, RS256, keys from the team's certs endpoint) [C], and by the same code GCP IAP or oauth2-proxy. It reuses PRD-266's credential registry (`core/auth/bearer_registry.py`) and does not choose PRD-257's open question 4 (a third sign-in path or generic OIDC); it is one more resolver either way. It is an authentication change, so §11-Q7.

### 4.11 Where this meets the Orderly work

The partnership asks for organisations above workspaces, per-tenant keys, run-completion webhooks, an approvals API for server keys and cost export. They are separate work. This PRD gives them homes: per-tenant keys are the Vault port (§4.9); webhooks out go through the Queue port, delivered by `edge/ingress` on the cloudflare profile; cost export reads from Basin (Cloudflare's pipelines, Iceberg tables and SQL, GA 1 Oct 2026) [C] on the cloudflare profile and from Postgres on the containers profile.

### 4.12 Observability

OTel stays the instrumentation on both profiles. On the cloudflare profile: container logs go to Workers Logs (7 days) and Logpush; Workers traces are beta; Monitors (SQL alert rules) and custom dashboards are beta; there is no OTLP metrics export yet, and Cloudflare documents no OTLP intake for containers' own traces [C]. So the Telemetry port's cloudflare adapter writes metrics to Analytics Engine, and container traces still go to an OTLP backend the operator chooses (§11-Q9).

## 5. User stories

Every story: tests go in with the story, and each one fails without the change; CI is the gate; route-manifest and config-surface updated when routes or settings change; functions and components within the AGENTS.md limits; commits signed off (`git commit -s`).

### Wave W0: The library

#### US-001: Package skeleton and the setting
**Description:** As a developer, I need the shared package and the profile setting, so ports have a home.

**Acceptance Criteria:**
- [ ] `packages/automatos-platform` exists with `ports/`, `adapters/containers/`, `adapters/cloudflare/`, `profile.py` and `contract_tests/`, installed by path in the API, workspace-worker, agent-opt-worker and media-render images
- [ ] `PLATFORM_PROFILE` in `config.py` (default `containers`; an unknown value fails at boot with the allowed values) and in `config-surface.json`
- [ ] A test proves an unknown value fails and the default is `containers`

#### US-001b: Build contexts
**Description:** As an operator, I need every image to build with the shared package, without changing how compose, Railway or the Helm chart deploy.

**Acceptance Criteria:**
- [ ] The option chosen in §11-Q12 (§4.2) is applied to `docker-compose.yml`, `.github/workflows/images.yml` and the Railway services' build settings (the Railway change is human-gated)
- [ ] Each image's compressed size stays within 1% of its size before the change (recorded in the PR), and its hashed lock is unchanged
- [ ] `docker compose up`, the kind end-to-end test and the smoke-images lane pass

#### US-002: One logging and metrics module
**Description:** As a maintainer, I want one copy of the shared logging and metrics code instead of five.

**Acceptance Criteria:**
- [ ] The five copies (three different versions today) are diffed first, and the PR lists every difference and which version wins
- [ ] `automatos_platform.logging` and `.metrics` replace `services/shared/automatos_{logging,metrics}.py`, the per-service copies and `orchestrator/core/monitoring/automatos_logging.py`; the copies are deleted
- [ ] Each service's log lines are unchanged except for the differences the PR lists (a test compares a record's JSON per service)

#### US-003: Enforcement and the contract-test lane
**Description:** As a reviewer, I want CI to stop new direct uses of a wrapped dependency.

**Acceptance Criteria:**
- [ ] `.importlinter-platform` with the two contracts in §4.4 runs in CI for the orchestrator and each service, green, with today's imports carried as `ignore_imports`
- [ ] A CI lane runs `contract_tests/` against the containers profile with Postgres and Redis service containers
- [ ] The package README explains the ratchet: each wave deletes the entries for what it moved

### Wave W1: Coordination ports (containers adapters)

#### US-004: Queue port
**Description:** As a developer, I need one queue contract instead of three Redis list protocols.

**Acceptance Criteria:**
- [ ] `ports/queue.py` (enqueue with priority and delay, claim with a visibility timeout, ack, retry, dead letter); the containers adapter on Redis
- [ ] `core/task_runner/queued.py`, the intake progress lists and the CLI host lists use it
- [ ] The port has a "claim the next due row" operation; the `SKIP LOCKED` claims in the 8 files that use them (including `services/board_dispatcher.py`, `api/board_tasks.py`, `services/cli_host_service.py`, `services/heartbeat_service.py` and the watch service and ticker) go through it, and the containers adapter keeps `SKIP LOCKED`
- [ ] `redis` list commands no longer appear outside the adapter

#### US-005: The worker is handed tasks
**Description:** As an operator, I want the API to hand the workspace worker its tasks, so the worker doesn't poll and a sleeping sandbox can be woken.

**Acceptance Criteria:**
- [ ] `ports/sandbox.py` (start a task, follow its status, cancel); the containers adapter calls the worker over HTTP through `WorkspaceClient`
- [ ] `api/workspace_github.py` uses the port instead of `lpush("workspace:tasks:normal")` through `runner._aredis`
- [ ] A new route on the worker's existing HTTP layer (`worker_http.py`) accepts a task; the `RPOP` loop and the `workspace:tasks:*` queues are deleted
- [ ] Task status and results (`workspace:task:{id}:status`, `:result`) are read through the port's follow operation, and task events go through the Event stream port (US-006)
- [ ] The Redis queue buffered tasks while the worker was down; the API now keeps an undelivered task and retries it, and a test proves a task sent while the worker is down runs when it returns
- [ ] A compose smoke test runs a GitHub task end to end

#### US-006: Event stream port
**Description:** As a user watching chat, the canvas or the board, I want one event stream that I can resume after a reconnect.

**Acceptance Criteria:**
- [ ] `ports/events.py` as in §4.3, with a resume cursor; the containers adapter on Redis Streams (so `after` works), trimmed by length
- [ ] The six SSE endpoints read from it: `api/workspace_files.py` `/canvas/events`, the board feed (`services/board_events.py`), workflows, `consumers/workflows/streaming`, the wizard, widgets chat
- [ ] The pg LISTEN threads in `services/board_events.py` and `services/board_dispatcher.py` are replaced by the port
- [ ] A test reconnects with `after` and receives only the missed events

#### US-007: Locks and quotas port
**Description:** As a developer, I need one lock and quota contract instead of advisory locks, `SETNX` and sorted sets in many files.

**Acceptance Criteria:**
- [ ] `ports/locks.py` (try-lock with TTL, dedup marker, sliding-window and exact-count quotas); the containers adapter keeps Postgres advisory locks where they guard database work and Redis elsewhere
- [ ] The 11 advisory-lock files, `services/webhook_dedup.py`, the `ai_notice` markers, `services/chat_turns.py` markers, `core/security/rate_limiter.py` and `api/widgets/rate_limit.py` use it
- [ ] slowapi gets shared storage, so API limits hold across replicas (behaviour change; a two-process test proves it)

#### US-008: Schedules port and one leader
**Description:** As an operator running two API replicas, I want scheduled work to run once.

**Acceptance Criteria:**
- [ ] `ports/schedules.py` (add, replace, remove and list jobs by id; the platform-wide jobs registered at boot)
- [ ] The containers adapter elects one leader with a Postgres advisory lock and renews it; the `fcntl` lock in `main.py` is deleted; the four fallback `AsyncIOScheduler`s in coordinator, harness, heartbeat and playbook_scheduler are deleted
- [ ] Two API processes against one database run each job once (test), and the second takes over within one lease when the first stops (test)

#### US-009: Cache port
**Description:** As a developer, I need one cache contract, so caches can move to KV later.

**Acceptance Criteria:**
- [ ] `ports/cache.py`; `core/cache/service.py` becomes the containers adapter, keeping its no-op fallback
- [ ] The cache users listed in the survey (routing, plugin, composio tool executor, digest, memory, graph router and the rest) go through it; the seven private `_get_redis` helpers are deleted
- [ ] The `redis` entries in `.importlinter-platform`'s `ignore_imports` are gone

### Wave W2: Data-side ports (containers adapters)

#### US-010: Objects, vectors and keyword search
**Description:** As a developer, I want every file, vector and keyword query through a port.

**Acceptance Criteria:**
- [ ] `ports/objects.py` wraps `core/storage/s3.py`, including presigned URLs (seven files call `generate_presigned_url` on the client today)
- [ ] `ports/vectors.py` generalises `modules/search/vector_store`; the raw pgvector SQL in codegraph, nl2sql introspection, RAG ingestion, multimodal knowledge tools and watch ticker moves behind it; `S3_VECTORS_ENABLED` is read only in the factory
- [ ] `ports/search.py` wraps the `tsvector` leg in `modules/rag/bm25_leg.py`
- [ ] The `boto3` entries in `.importlinter-platform` are gone, except Bedrock's named entry

#### US-011: Durable runs, browser, vault and telemetry ports
**Description:** As a developer, I want long runs, browsing, credential encryption and telemetry behind ports.

**Acceptance Criteria:**
- [ ] `ports/runs.py` generalises `core/task_runner` (start, follow, cancel, resume after restart); missions and Playbook runs start through it
- [ ] `ports/browser.py` wraps the worker's Playwright use
- [ ] `ports/vault.py` wraps `EncryptionService`; the containers adapter keeps the env key and the generated key file the local edition relies on
- [ ] `ports/telemetry.py` wraps the tracer seam and `metrics.py`

#### US-011b: One way into the database
**Description:** As a developer, I want every database connection to come from one engine, so connection settings, pooling and tracing apply everywhere.

**Acceptance Criteria:**
- [ ] `ports/records.py` provides the engine and sessions; the containers adapter is today's `core/database/database.py`
- [ ] The 16 raw `psycopg2.connect` calls in 6 files (`modules/rag/ingestion/manager.py`, `api/documents.py`, `core/credentials/service.py` and the database init scripts) and the `asyncpg` paths (`modules/rag/bm25_leg.py`, `modules/rag/service.py`, `core/credentials/tester.py`) go through it, or are listed in the PR with the reason they can't (the fresh-database bootstrap may need a raw connection)
- [ ] The `psycopg2` and `asyncpg` entries in `.importlinter-platform` are gone except those listed

### Wave W3: Cloudflare compute (records on Path 1)

#### US-012: The `edge/` project and the router
**Description:** As an operator, I want a Cloudflare deployment of the web app and the API with state outside.

**Acceptance Criteria:**
- [ ] `edge/` with `wrangler.jsonc`, the router Worker, the API container class (the same API image) and the scheduler running as one named instance
- [ ] The web app builds with `@opennextjs/cloudflare`; `frontend/app/api/chat/route.ts` and `frontend/app/api/workflows/stream/route.ts` move from the Edge runtime to Node
- [ ] Records on PlanetScale Postgres (Path 1), files on R2 through `S3_ENDPOINT_URL`, Redis on a managed provider until W4
- [ ] A staging deployment is human-gated; the PR records the measured cold start and request latency

#### US-013: The bridge and model calls
**Description:** As an operator, I want containers to reach Cloudflare's services through one bridge, and every model call through AI Gateway.

**Acceptance Criteria:**
- [ ] `edge/bridge` serves the virtual hosts in §4.5 with the caller contract; ports not yet implemented answer 501
- [ ] Bridge tests reject missing, invalid, expired, replayed (from another container) and cross-workspace or cross-host requests, for both API containers and sandboxes
- [ ] The API image trusts Cloudflare's interception CA when it is present (§4.5), and `interceptHttps` is on for the bridge's hosts
- [ ] The cloudflare Telemetry adapter writes metrics to Analytics Engine through `telemetry.cf`; traces still export over OTLP (§4.12)
- [ ] The cloudflare contract-test lane runs under `wrangler dev`, and every cloudflare adapter that exists passes it
- [ ] A platform-level base URL setting for every provider (Anthropic and OpenAI today have none; OpenRouter, NVIDIA and DeepSeek do), so calls route through AI Gateway with the workspace and run as metadata

### Wave W4: The cell, without records

#### US-014: The workspace cell
**Description:** As an operator on Cloudflare, I want each workspace's events, locks, quotas, schedules and sandbox in one Durable Object.

**Acceptance Criteria:**
- [ ] `edge/cell` with the event stream, locks and quotas, the schedule table with one alarm, and the board claim; Cron Triggers for platform-wide jobs
- [ ] The cloudflare adapters for Event stream, Locks, Schedules, Queue (Queues) and Cache (KV) pass the contract tests
- [ ] Each cell is created in the jurisdiction set on its workspace (default from config)
- [ ] A workspace that requires PCI DSS or FedRAMP High is refused a cell while §11-Q11 hasn't confirmed Containers and Queues, with a clear message (FR-7; test)

#### US-015: A sandbox per workspace
**Description:** As a workspace owner, I want my agents' code to run in a sandbox no other workspace can reach.

**Acceptance Criteria:**
- [ ] The cloudflare Sandbox adapter starts tasks in the cell's container (Sandbox SDK 1.0), built from the workspace-worker image
- [ ] Workspace files persist to R2 and are restored on wake; restore time for a 1 GB workspace is recorded
- [ ] Sandboxes run with `enableInternet = false` and `interceptHttps = true`; the worker image trusts the injected CA in its entrypoint (§4.5); git, package installs and API calls leave only through `allowedHosts` or the outbound handler
- [ ] The allowlist is per workspace, and Orderly-style API keys are added by the outbound handler, never placed in the container
- [ ] A test proves one workspace's sandbox cannot read another's files, reach a host off its list, or open a connection on a port other than 80 and 443

### Wave W5: Vectors and memory

#### US-016: Vectorize and the embedding model
**Description:** As an operator on Cloudflare, I want vector search without Postgres or Qdrant.

**Acceptance Criteria:**
- [ ] The embedding model chosen in §11-Q4, with a re-embed job that runs per workspace and reports progress
- [ ] The cloudflare Vectors adapter (a namespace per workspace) passes the contract tests
- [ ] Qdrant memory moves behind the Vectors port on both profiles (pgvector on containers), if §11-Q5 says retire

### Wave W6: Durable runs

#### US-017: Missions and Playbooks on Workflows
**Description:** As a user, I want a long mission to survive the container that runs it being moved.

**Acceptance Criteria:**
- [ ] The Workflows are TypeScript, in `edge/`. Each step calls the API through the router with the §4.5 caller contract, and the step's work runs in the API's Python code. Python Workflows are used only after Cloudflare confirms they're GA (§11-Q11)
- [ ] The cloudflare Runs adapter starts a Workflow per run; each step is durable and retried
- [ ] Every step carries an idempotency key derived from the run id and the step's name. A step's side effects use it: database writes upsert on it, outbound calls that support an idempotency header send it, and a step that finds its key already recorded returns the recorded result instead of acting again
- [ ] A test fails a step after its side effect but before the Workflow records success, and proves the retry doesn't repeat the effect
- [ ] A run survives a forced container restart mid-step (test on staging)

### Wave W7: The records gate

#### US-018: The records prototype
**Description:** As Gerard, I want to decide on Path 2 from measurements, before the large rewrite is spent.

**Acceptance Criteria:**
- [ ] Only the tables that chat, the board and schedules touch are made dialect-neutral (§4.7), and the full backend suite still passes on Postgres
- [ ] One workspace runs with those records in its cell, in a Python Durable Object if it fits the memory limit, else TypeScript
- [ ] The PR reports §10's records bars; Gerard records the decision in §11-Q3

#### US-019: Records into cells (only if §11-Q3 chooses Path 2)
**Description:** As an operator on Cloudflare, I want every workspace's records in its own cell.

**Acceptance Criteria:**
- [ ] §4.7 steps 3 and 4 done; the full backend suite passes on Postgres, and the suite's SQLite-marked subset passes against a cell
- [ ] A migration job moves workspaces one at a time, verifies row counts, and can roll a workspace back to Postgres
- [ ] Cross-workspace reads (org dashboards, admin, billing) have a read model or a fan-out, each listed in the PR

### Wave W8: Edges

#### US-020: Vault Worker, trusted-proxy edition and channels
**Description:** As an enterprise operator, I want per-workspace keys, SSO through my proxy, and email, voice and browsing on Cloudflare.

**Acceptance Criteria:**
- [ ] The Vault Worker (§4.9) with audit to a locked R2 bucket
- [ ] The trusted-proxy lane (§4.10), approved under §11-Q7
- [ ] Cloudflare adapters for Browser (Browser Run), inbound email (Email Workers into the Queue port) and outbound email; voice through Agents SDK voice if §11-Q10 says so

## 6. Functional requirements

- FR-1: `PLATFORM_PROFILE` selects adapters in one module; no other module branches on it.
- FR-2: The containers profile behaves as today, except these changes, each named and tested in its story:
  - US-005: the worker is handed tasks over HTTP instead of polling Redis, and the API holds undelivered tasks;
  - US-006: live streams move to Redis Streams and can be resumed;
  - US-007: API rate limits are shared across replicas, and quotas move behind a port;
  - US-008: one scheduler leader across replicas;
  - US-012: the two Edge-runtime frontend routes move to Node, on every profile;
  - US-016: Qdrant and mem0 retire on the containers profile too, only if §11-Q5 says so.
- FR-3: Every port has a contract test suite. The containers adapter passes it in the wave that adds the port; the cloudflare adapter passes it in the wave that adds that adapter.
- FR-4: After each wave, the dependencies it wrapped are forbidden outside adapters by `.importlinter-platform`.
- FR-5: On the cloudflare profile, a workspace's sandbox, vector namespace, vault keys and (on Path 2) records are reachable only by that workspace's own requests, enforced by the bridge.
- FR-6: Compose, the Helm chart and Railway deploy and pass their existing smoke tests at every wave.
- FR-7: The cloudflare profile refuses to create a cell for a workspace that needs a compliance regime not confirmed for every product the cell uses (§8.2). The workspace's required regime is a workspace setting.

## 7. Proposed non-goals (Gerard confirms or changes them in §11-Q13)

- Moving the local edition off Postgres or Redis.
- Choosing a sign-in design beyond the trusted-proxy lane (PRD-257's open question 4 is already Gerard's).
- The Orderly partnership's organisations, webhooks, approvals API and cost export (§4.11).
- A Cloudflare deployment of the marketing sites.
- Replacing Composio or the model providers; both stay outside, reached through the gateway or as today.

## 8. Technical considerations

### 8.1 Cloudflare limits that shape the design [C]

| Product | Limit or status that matters |
|---|---|
| Containers | GA (Apr 2026); up to 4 vCPU, 12 GiB, 20 GB disk; no autoscaling (planned); disk is ephemeral; outbound handlers see ports 80 and 443 only, HTTPS only with `interceptHttps` and the injected CA trusted; allowlists hold only with internet off; placement in `eu` or `fedramp` |
| Sandbox SDK | GA Apr 2026, 1.0 on 30 Sep 2026; a security notice on 22 Oct 2026 required image rebuilds |
| Durable Objects | SQLite storage GA, 10 GB per object, 30-day point-in-time recovery, one alarm per object, jurisdictions `eu`, `us`, `fedramp`; CPU 30 s per request by default, up to 5 min |
| D1 | 10 GB per database (fixed), one query at a time per database, 100 bound parameters; bound statically in config |
| Python Workers | GA 21 Sep 2026; FastAPI over ASGI; 128 MB memory, 64 MiB bundle; no threads |
| Workflows | Up to 25,000 steps, unlimited wall time, 1 GB state; TypeScript and Python (Python announced as beta, Nov 2025) |
| Queues | 128 KB messages, 5,000 messages/s per queue, delay up to 24 h, dead-letter queues |
| KV | Eventually consistent (up to 60 s); 1 write/s per key |
| Vectorize | 1536 dimensions, 20M vectors per index, 50,000 namespaces; no keyword search |
| Secrets Store | Beta; 100 secrets per account |

### 8.2 Compliance [C]

From Cloudflare's trust hub and data-localisation page (15 Oct 2026). "Not listed" means absent from the published list.

| Product | SOC 2 | PCI DSS | FedRAMP High | Data localisation |
|---|---|---|---|---|
| Workers | listed | listed | listed | logs US-only; Logpush instead |
| Durable Objects | listed | listed | listed | jurisdiction per object |
| R2 | listed | listed | listed | jurisdiction per bucket |
| D1 | listed | not listed | not listed | the D1 docs and the localisation page disagree |
| Queues | listed | not listed | not listed | not listed |
| Containers | not listed | not listed | not listed | placement only |

Containers run the API and every sandbox, so their scope is §11-Q11. The lists favour Durable Objects over D1, which supports cells for Path 2.

**Regulated workspaces stay on the containers profile until coverage is confirmed.** Containers and Queues aren't on the PCI DSS or FedRAMP High lists. A `fedramp` placement or jurisdiction is not the same thing as product coverage. So a workspace whose owner needs PCI DSS or FedRAMP High coverage can't be created on the cloudflare profile. That holds until Cloudflare confirms both products in writing (§11-Q11), or until compliant replacements are chosen for them (FR-7).

### 8.3 New dependencies

- **Python:** none. The cloudflare adapters use `httpx`, already installed.
- **npm, in `edge/` only:** `wrangler`, `@cloudflare/containers` or `@cloudflare/sandbox` (Sandbox SDK 1.0), and `@opennextjs/cloudflare` in `frontend/`. Each needs Gerard's approval (AGENTS.md).

### 8.4 Risks

- The cloudflare adapters' latency through the bridge is unmeasured (W4 measures it).
- Several Cloudflare products used here are beta (Secrets Store, Python Workflows, Email Service, Agents voice, sandbox snapshots).
- Path 2 is a large rewrite; the gate exists so it is chosen on evidence.

## 9. Waves and sequencing

| Wave | Content | Depends on | Changes the container model |
|---|---|---|---|
| W0 | Library, setting, build contexts, logging consolidation, enforcement | §11-Q1, §11-Q12 | Build contexts only; images and deploys otherwise unchanged |
| W1 | Queue (with row claims), worker hand-off, events, locks and quotas, schedules, cache | W0 | Yes: one scheduler leader, shared API limits, resumable streams, no worker polling |
| W2 | Objects, vectors, search, runs, browser, vault, telemetry; one way into the database | W0 | No |
| W3 | `edge/`, router, bridge, web app on Workers, Path 1 records, the cloudflare contract lane | W1, W2 | Yes: two frontend routes move from Edge to Node |
| W4 | The cell without records; a sandbox per workspace | W3 | No |
| W5 | Vectorize and the embedding model | W2, W3 (the bridge), §11-Q4 | Only if Qdrant is retired (§11-Q5) |
| W6 | Durable runs on Workflows | W2, W3 | No |
| W7 | The records prototype and gate; records into cells only if Path 2 is chosen | W1, W2, W4, §11-Q3 | No |
| W8 | Vault Worker, trusted-proxy lane, channels | W3, W4, §11-Q7 | The trusted-proxy lane works on containers too |

W0–W2 are useful on their own. Dates are Gerard's call; nothing here is proposed before WebSummit (10–13 Nov).

## 10. Success metrics

- Two API replicas on the Helm chart run each scheduled job once (W1).
- No direct `redis`, `boto3`, `psycopg2`, `asyncpg` or `apscheduler` import outside adapters, except the named entries in `.importlinter-platform` (W2).
- Contract tests green on both profiles for every port that has a cloudflare adapter.
- On the cloudflare profile (W4): a cell call through the bridge adds under 20 ms at p95 inside one location; a sleeping workspace's sandbox is ready in under 10 s with a 1 GB workspace restored.
- Records gate (W7): chat time-to-first-token and board operations within 10% of Postgres at p95, and the cell under its memory limit with a large workspace.

These bars are proposals; Gerard confirms or moves them.

## 11. Open questions for Gerard (each with a recommendation)

1. **Adopt the platform library and W0–W2?** They change shared code across the API and services. *Recommendation:* yes; they pay off on Helm and Railway even if Cloudflare never ships.
2. **Is the Cloudflare profile in scope now, or only W0–W2?** *Recommendation:* approve W0–W6 and the W7 prototype (US-018) as the plan, and review it again after W2. US-019, the records rewrite, waits for Q3.
3. **Records: Path 1 or Path 2?** *Recommendation:* Path 1 from W3; decide Path 2 at the W7 gate, after the prototype and before the rewrite.
4. **Embedding model at or below 1536 dimensions, and a re-embed?** *Recommendation:* a 1024-dimension model on both profiles, so vectors stay portable; re-embed per workspace.
5. **Retire Qdrant and mem0 in favour of the Vectors port on both profiles?** *Recommendation:* yes; two fewer containers on the container model.
6. **Tenancy: cells on the cloudflare profile.** This is a tenancy change (AGENTS.md). *Recommendation:* approve cells without records (W4) now; records only through the W7 gate.
7. **Authentication: the trusted-proxy lane.** *Recommendation:* approve as one resolver in PRD-266's registry, leaving PRD-257's question 4 open.
8. **Customer-held keys:** accept an outside KMS behind the Vault port for enterprise BYOK? *Recommendation:* yes; Cloudflare has no KMS.
9. **Container traces on Cloudflare:** an OTLP backend the operator chooses, until Cloudflare takes OTLP metrics and traces? *Recommendation:* yes.
10. **Voice:** keep Retell, or add an Agents SDK voice adapter (beta)? *Recommendation:* keep Retell; revisit when Agents voice leaves beta.
11. **Ask Cloudflare's account team:** Containers' and Queues' compliance scope (PCI DSS, FedRAMP High; FR-7 holds regulated workspaces back until confirmed); D1 residency (its docs disagree); autoscaling timing; whether Python Durable Objects and Python Workflows are GA; outbound-handler latency. *Recommendation:* before W3.
12. **Build contexts: repo-root contexts or a synced copy of the package (§4.2)?** It changes how compose, CI and Railway build the images. *Recommendation:* repo-root contexts with a `.dockerignore` per Dockerfile; the synced copy only if Railway can't build from the root.
13. **The proposed non-goals in §7, and the boundary with the Orderly work (§4.11):** confirm, or move any of them into this PRD. *Recommendation:* confirm; the Orderly pieces build on these ports but are a separate decision.

## Related

- PRD-262 (container diet): image budgets the package must fit.
- PRD-266 (Automatos Go backend): the credential registry the trusted-proxy lane joins.
- PRD-257 (draft, enterprise estate agents): its open question 4 is not decided here.
- PRD-256 (OpenTelemetry): the instrumentation both profiles keep.
- The platform-engineering notes (daarthur/automatos-platform-engineering): the seven contracts, the workspace as the isolation unit, and the K1 scheduler blocker.
