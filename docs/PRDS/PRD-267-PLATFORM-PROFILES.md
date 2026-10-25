# PRD-267: Platform profiles: fix the container model, then earn Cloudflare

**Status:** Draft for Gerard's approval, 25 Oct 2026. Nothing here is approved; every scope choice is in §11 with a recommendation. · **Owner:** Gerard (scope and the §11 answers) · **Proposed by:** daarthur · **Written:** 24 Oct 2026, grounded on `origin/main @ 0c0afb02f` (re-checked against `d8b2aa07e`) and on Cloudflare's docs, changelog, blog and trust hub as published on 24 Oct 2026. **Revised** 25 Oct 2026 after three reviews (a Sonnet fact-check, CodeRabbit, and an Opus due-diligence review); the shape changed to the gated plan below.
**Source:** two write-ups from the same session: *Automatos on Cloudflare*, a reference stack for the Orderly partnership (claude.ai/artifact/5kyr2xMnFrhkdoLx9qb2hW), and *All-Cloudflare Automatos*, a capability-by-capability parity study (claude.ai/artifact/9hBKPbunj9ANfXqotfZcjk). A read-only survey of `origin/main` mapped every platform dependency.
**Shape:** three parts, each gated by the one before.
- **Part A** fixes the container model. It is useful on its own, and is the part proposed for approval now.
- **The spike** runs today's stack on Cloudflare, with one new piece: a sandbox per workspace. It lasts two weeks and produces numbers.
- **Part B** is the full Cloudflare profile. It is written out so the direction is clear, but no Part B story starts until Gerard decides on the spike's numbers and a partner commitment (§11-Q7).

**Type:** Part A is refactor and consolidation, with tested behaviour changes listed in FR-2. The spike and Part B are extensions.
**Scope:** automatos-ai, including a new `edge/` TypeScript project from the spike on. The Orderly partnership's own features (organisations, webhooks, the approvals API for server keys, cost export) are proposed for their own PRD (§11, question 16). Throughout, "§11-Qn" means question n in §11.
**Needs Gerard's approval before code (AGENTS.md "stop and ask first"), each with a tracking issue as PRD-266 did:**

| Item | Where | Why it's on this list |
|---|---|---|
| A shared package used by the API and the services, and the build-context change it needs | §4.1, §11-Q2 | New cross-cutting code; changes how compose, CI and Railway build |
| The credit-meter jobs and the Composio sync job move under a new scheduler lease | §4.2, §11-Q5 | Billing, and a guardrail (the Composio deny list is fail-closed) |
| Shared API rate limits while the policy plane fails closed | §4.3, §11-Q3 | A guardrail: a Redis outage would become an API outage |
| A sandbox per workspace (spike) and cells (Part B) | §5, §6.2 | Tenancy |
| The trusted-proxy sign-in lane | §6.7 | Authentication |
| The worker's model calls through an API proxy that adds the key | §4.4, §11-Q17 | Authentication of Claude Code sessions |
| Re-embedding at 1024 dimensions; exporting and retiring Qdrant memory; a compliance-regime column | §6.3, §6.8 | Migrations that change user data |
| New npm dependencies in `edge/` | §8.3 | New dependencies |

Labels: **[T]** read in code on `origin/main`; **[C]** from Cloudflare's published docs, 24 Oct 2026; **[I]** inferred.

## 1. Introduction

Orderly runs on Cloudflare and wants its agent layer there too. Automatos is built for containers: compose for the local edition, the Helm chart for platform teams, Railway for hosted Studio. Those deployments must keep working on the same code.

Working out what Cloudflare would need showed that the container model has problems of its own, Cloudflare or not:

| Problem | Today on `origin/main` [T] |
|---|---|
| Every replica runs a scheduler | APScheduler behind an `fcntl` lock on `/tmp/automatos_scheduler.lock` (`orchestrator/main.py:410`). The lock is per host, so the Helm chart pins `api.replicas: 1`. Three modules use `RedisJobStore`; four services keep a fallback `AsyncIOScheduler` of their own |
| Rate limits are per process | slowapi `Limiter` with no storage (`orchestrator/main.py:1042`); it fails closed when `POLICY_PLANE_ENABLED` |
| Four streaming transports | Six SSE endpoints over pg LISTEN, Redis pub/sub, in-process queues and polling. Chat streams directly from `api/chat.py:399` |
| The worker polls, and holds the keys | The workspace worker polls `workspace:tasks:*` with `RPOP` (`services/workspace-worker/main.py:195`; the only producer is `orchestrator/api/workspace_github.py:294`). It keeps status and results under `workspace:task:{id}:*` and writes `task_executions` straight into Postgres. Its environment carries the Postgres superuser URL, `REDIS_URL` and the Anthropic key (`docker-compose.yml`), in the same process that runs agent code |
| Durable keys sit in an evicting Redis | Compose runs Redis with `--maxmemory 256mb --maxmemory-policy allkeys-lru`, so job-store entries and queue items can be evicted like cache entries |
| Five copies of the shared logging code | `automatos_logging.py` in five places, as three different versions; `automatos_metrics.py` in four |
| Dead plumbing | The `taskrunner:queue:*` queues in `core/task_runner/queued.py` have no consumer (`dequeue_task_sync` has no callers); the `kubernetes` task-runner backend named in `TASK_RUNNER_BACKEND` has no module |

Part A fixes these directly. It does not build a generic port for every dependency. A port is added only where a second implementation is real: the sandbox (containers and Cloudflare) and the event stream. Everything transaction-shaped stays with the database.

### 1.1 What Cloudflare would need, and why it's gated

On Cloudflare, a container reaches Cloudflare's own services only through outbound handlers: plain HTTP on ports 80 and 443 [C]. Its stateful services aren't drop-in replacements for Postgres or Redis. And the code leans on Postgres in ways no adapter can hide [T]:
- **Transaction-scoped locks:** first-login provisioning (`core/auth/hybrid.py:309`), both quotas (`services/agent_quota.py:72`, `core/media_render_quota.py:83`), the built-in skills seed, and two socials helpers. All use `pg_advisory_xact_lock`.
- **Row claims:** the board claim (`services/board_dispatcher.py`) is `FOR UPDATE SKIP LOCKED` over `board_tasks` joined to agents.
- **Notifications on commit:** `pg_notify` fires only when the transaction commits.

Moving any of those off Postgres while the records stay on Postgres would lose atomicity. That is why Part B keeps them on Postgres unless the records move too, and why records only move after a measured prototype (§6.5).

## 2. Goals

- **Part A:**
  - the Helm chart runs more than one API replica with one scheduler, shared rate limits and resumable streams;
  - the worker is handed its tasks and holds no database credentials;
  - durable Redis keys can't be evicted;
  - all of it without changing how compose, the Helm chart or Railway deploy, beyond what FR-2 lists.
- **The spike:**
  - measured answers for today's stack on Cloudflare Containers: database round trips per chat turn, pool size, always-on loops, cost;
  - a working sandbox per workspace that holds no secrets.
- **Part B:** a plan precise enough that, if Gerard approves it after the spike, it can start without another design round.

## 3. Principles

1. **Fix first, abstract second.** Part A changes the code directly. A port exists only where two implementations will exist (the sandbox; the event stream).
2. **Transactions stay with the database.** Transaction-scoped locks, row claims and notify-on-commit stay on Postgres for as long as the records do. Cross-process locks and fan-out may move; anything published about a database change is published after commit, through an outbox.
3. **Profile is separate from edition.** `PLATFORM_PROFILE` (`containers`, `cloudflare`), when it arrives, is independent of `AUTH_EDITION` (`local`, `saas`). Dozens of files already branch on edition [T]; a profile must not multiply them.
4. **Replace, don't shim** (AGENTS.md). When Part A replaces a mechanism, the old one is deleted in the same story, and the import-linter ratchet (§4.7) stops new uses.
5. **Least privilege for agent code.** Code that runs agents' commands holds no database, Redis or model-provider credentials. It reaches state through the API, and model calls get their key added outside the process.
6. **Evidence before cost.** Each gate comes before the expensive part it guards.

## 4. Part A: the container model

### 4.1 The shared package and build contexts

```
packages/automatos-platform/
  automatos_platform/
    logging.py      # replaces the five copies of automatos_logging.py
    metrics.py      # replaces the four copies of automatos_metrics.py
    ports/          # Part A: sandbox.py, events.py; more only when a second implementation is real
    adapters/containers/
    profile.py      # the only module that imports adapters
  contract_tests/   # one suite per port
```

**Build contexts.** Each image builds from its own folder (`./orchestrator`, `./services/*`, `./frontend`), so none can see `packages/` [T]. Two options, §11-Q2:
- *Repo-root contexts (recommended):*
  - compose, `images.yml` and Railway build from the repo root with the same Dockerfiles, each with its own `<Dockerfile>.dockerignore` (BuildKit) so the context stays as small as today;
  - the package installs with `pip install --no-deps`, so hashed locks don't change;
  - PRD-262's image tags hash `git rev-parse HEAD:<context>` (`scripts/stack_images.py:429`, `:604`), so the tag becomes a hash of the context tree and the package tree together;
  - the dev override mounts the package beside `./orchestrator:/app` so edits reload.
- *A synced copy:* a script copies the package into each context and CI fails if a copy differs. Simpler, but it is the copying this PRD removes.

### 4.2 One scheduler, with a fenced lease

- The `fcntl` lock and the four fallback `AsyncIOScheduler`s are deleted.
- **The lease.** The API process that holds a Postgres lease runs the scheduler. The lease reuses `core/database/boot_lock.py`'s advisory-lock pattern, with a lease row carrying a generation number.
- **Fencing.** A process that loses its connection or fails to renew the lease stops its scheduler before the lease can pass to another. Jobs that write check the generation they started under. So two schedulers never share the `RedisJobStore`.
- **Idempotent jobs.** Every job is idempotent per run window. This matters for billing: the credit-meter jobs (`services/credits/jobs.py`: meter tick, nightly overhead, hourly expiry) and the Composio sync job, which backs the fail-closed deny list. They move under the lease unchanged, but their idempotency is tested (§11-Q5).

### 4.3 Shared API rate limits

slowapi gets Redis storage, so limits hold across replicas. Today it fails closed when `POLICY_PLANE_ENABLED` (`swallow_errors=not _policy_plane_on`) [T]. With shared storage, a Redis outage would turn every rate-limited route into an error. §11-Q3 asks whether a storage error should fail open (logged) or closed when the plane is on.

### 4.4 The worker is handed its tasks

- **The Sandbox port** (`ports/sandbox.py`): start a task, follow its status, cancel. The containers adapter calls the worker over HTTP through `WorkspaceClient`, on a new route in the worker's existing HTTP layer (`worker_http.py`).
- **A durable hand-off.** The API writes the task to `task_executions` (which exists; the worker already updates it) before sending it. A relay re-sends undelivered tasks, so a task sent while the worker is down, or while the API restarts, still runs. If `task_executions` lacks a delivery-state column, that's one additive migration.
- **Retries never run a task twice.** Before running a task, the worker claims it through the API. The claim moves the row from pending to running in one conditional update on `task_id`, and succeeds once. A re-sent task that is already claimed is acknowledged without running. One that already finished gets its recorded result back. A worker that dies mid-task leaves the claim to expire after a lease, and only then can the task be claimed again.
- **A re-run after a crash doesn't repeat side effects.** The worker can die after an external effect but before the API records completion: a commit pushed, a file written, a call to Orderly's Hub API. So each side-effecting step carries an idempotency key made from the `task_id` and the step's name.
  - The worker records the key and the step's result through the API as the step completes.
  - A re-claimed task asks for its recorded steps first, and replays their results instead of acting again.
  - Outbound calls that accept an idempotency header (Orderly's Hub API included) send the key.

  Model calls can't be deduplicated, so a re-run may spend again. That cost is accepted, and capped by the per-task budget.
- **The worker reports through the API.** Status, results and events go over an authenticated callback instead of the worker's direct Postgres write (`task_executions_db.py`) and its Redis keys. The worker then needs neither `DATABASE_URL` nor `REDIS_URL`, and compose and the chart drop them from its environment.
- **Model calls get their key outside the worker (US-A09, §11-Q17).** Today the worker also carries `ANTHROPIC_API_KEY` and `CLAUDE_CODE_OAUTH_TOKEN` [T]. The proposal:
  - the worker's model clients, the Claude Code CLI included, point their base URL at an internal model proxy on the API;
  - each task carries a short-lived token scoped to its workspace, run and `task_id`, and the worker presents it (a run can hold several tasks, so a sibling task's token is refused);
  - the proxy checks the token, swaps in the right provider credential and forwards the call;
  - the worker's environment then holds no provider key.

  That a proxy can swap in a subscription OAuth token as well as an API key is inferred [I]. US-A09 proves it first, or keeps that credential as a named, temporary exception.
- **Deleted:** the `RPOP` loop, `workspace:tasks:*`, `workspace:task:{id}:*`, the dead `taskrunner:queue:*` queues and the `kubernetes` backend name.

### 4.5 One resumable event transport

- **The Event stream port** (`ports/events.py`): publish to a (workspace, channel), follow with a resume cursor.
- **The containers adapter** uses Redis Streams, trimmed by length.
- **The six SSE endpoints read from it:** canvas events, the board feed, workflows, `consumers/workflows/streaming`, the wizard and widgets chat. The pg LISTEN threads in `services/board_events.py` and `services/board_dispatcher.py` go.
- **Publish after commit.**
  - Events about database changes (board, task and run state) are written to an `event_outbox` table in the same transaction and relayed to the stream. That keeps today's notify-on-commit guarantee, and makes it durable.
  - Ephemeral events (canvas progress, typing) publish from a session `after_commit` hook.
- **The frontend resumes.** SSE responses carry `id:`, and clients reconnect with `Last-Event-ID`.
- **Chat stays direct.** Chat tokens keep streaming straight from `api/chat.py`; they don't go through the port.

### 4.6 Redis durability

Compose's `allkeys-lru` can evict job-store entries, stream entries and outbox relay state [T]. The option recommended in §11-Q4: `volatile-lru`, so only keys with a TTL are evicted, plus a test that every cache write sets a TTL. The alternative is a second, `noeviction` Redis for durable keys. The Helm chart and Railway get the same setting.

### 4.7 Enforcement

- **A second import-linter config.** `orchestrator/.importlinter` has `root_package = modules` and one independence contract, and `ruff.toml` exempts `services/**` from `TID251` [T]. So the bans live in `.importlinter-platform` at the repo root, run once per code root (the orchestrator and each service) with `include_external_packages = True`.
- **What's forbidden:** importing `automatos_platform.adapters` outside `automatos_platform.profile`; and new imports of `redis`, of `core.redis.client` (26 files use the raw client it returns) and of `apscheduler`. The external packages alone would miss most of the leak.
- **A ratchet, like PRD-211's.** Today's imports are `ignore_imports`, and each story deletes the entries for what it moved.
- **Contract tests** for each port run in a CI lane against Postgres and Redis service containers. One test runs under memory pressure, to prove durable keys survive.

## 5. The spike: today's stack on Cloudflare, plus sandboxes

Two weeks, human-gated deploys, a written report. Today's code runs on Cloudflare with as little change as possible:
- **The stack:**
  - the API image in Cloudflare Containers, as a fixed pool behind a router Worker;
  - PlanetScale Postgres billed through Cloudflare (pgvector and LISTEN/NOTIFY supported [C]);
  - a managed Redis;
  - R2 through `S3_ENDPOINT_URL`;
  - model calls through AI Gateway;
  - the web app on Workers with OpenNext, after the two Edge-runtime routes move to Node (`frontend/app/api/chat/route.ts`, `frontend/app/api/workflows/stream/route.ts`).
- **One new piece, the Sandbox port's Cloudflare adapter.** A sandbox per workspace (Sandbox SDK 1.0, driven by a Durable Object keyed by workspace).
  - **The image:** built from the worker image, but holding no database, Redis or model credentials. It reports through the API, as Part A makes the worker do.
  - **Network:** it runs with `enableInternet = false` and `interceptHttps = true`; the image trusts the injected CA at `/etc/cloudflare/certs/cloudflare-containers-ca.crt` in its entrypoint [C]. Git, package installs and model calls leave only through `allowedHosts` or the outbound handler, which adds keys (model, Orderly's Hub API) so they never enter the container.
  - **Identity:** the handler knows which container is calling (`ctx.containerId`) [C], so a sandbox can reach only its own workspace's routes.

**The spike measures, and the report states:**
1. **Database round trips:** PlanetScale latency from Containers, times queries per chat turn, for a real chat turn with tools. The ORM is synchronous and chatty, so this decides whether the API can live away from its database.
2. **Pool size:** the fixed API pool needed for today's Studio traffic (Containers have no autoscaling [C]), and its monthly cost.
3. **Always-on loops:** where the board dispatch loop (`main.py:634`), ChannelManager, the task reconciler and the scheduler live when pool containers sleep (`sleepAfter`). For example, a pinned instance that never sleeps.
4. **Workspace files:** the API reads `/workspaces` directly in a few places (`services/cli_host_service.py`, the board dispatcher). The report lists them and proposes a workspace-files route on the Sandbox port.
5. **Sandboxes:** wake time with a 1 GB workspace restored from R2; isolation tests (no other workspace's files, no off-list host, no port but 80 and 443); environment holds no secret.
6. **Cost:** per active workspace and per idle workspace.

## 6. Part B: the full Cloudflare profile (gated)

None of this starts until Gerard decides on the spike's report and a partner commitment (§11-Q7). It's written out so that decision is about a known plan.

### 6.1 The bridge and the caller contract

- **The bridge.** Containers reach Cloudflare bindings through `edge/bridge`, with one virtual host per port (`events.cf`, `cache.cf`, `queue.cf`, `vectors.cf`, `vault.cf`, `telemetry.cf`). The handlers see ports 80 and 443 only [C].
- **Callers:**
  - **A sandbox** is identified by its intercepted container; it carries no token.
  - **An API container** presents a router token. The token is signed with a key only the router and bridge hold, and binds:
    - the workspace;
    - the request id;
    - the calling container;
    - the audience (one virtual host);
    - an expiry.
    It is reusable within its request, and refused from any other container, workspace or host, or after it expires.
  - **Work no request starts** (alarms, queue consumers, Workflow steps, platform-wide jobs such as the credit meter and memory jobs) runs as a **system principal**: narrow scopes per job, every call audited.
  - **Long work** (streams and agent loops past the expiry) refreshes its token through the router. The token travels through `contextvars` into tasks it spawns.
- **Rejections.** Missing, invalid, expired, replayed or mismatched tokens are rejected (401 or 403) and logged without the token.
- **The limit, stated plainly.** On Path 1 the API container holds a database credential for every tenant, so the token protects Cloudflare bindings, not the database.

### 6.2 Workspace cells, without records

A Durable Object per workspace, created in a jurisdiction (`eu`, `us`, `fedramp`) [C].

**What the cell holds:**
- **Cross-process locks and dedup markers.** Not the transaction-scoped locks or the board claim (§1.1).
- **The workspace's schedule.** Its own SQLite, with one alarm set for the earliest job. Alarms fire at least once, with up to 6 retries [C], so jobs are idempotent.
- **Fan-out to browsers.** Over hibernatable WebSockets through the router, so an idle viewer costs nothing. The outbox relay feeds it. Chat tokens still stream from the API directly: one object handles one request at a time, with a soft limit around 1,000 requests a second [C], so it must not sit in every chat's path.
- **The workspace's sandbox.**

### 6.3 Vectors

- **Embeddings.** Vectorize allows 1536 dimensions at most [C], and Automatos uses 2048 [T]. Moving to a 1024-dimension model (bge-m3 or qwen3-embedding on Workers AI) is a migration with three steps:
  1. dual-write both embeddings;
  2. backfill per workspace;
  3. switch reads, keeping a rollback path until the old column is dropped.

  It applies to both profiles, so vectors stay portable (§11-Q9).
- **Index layout.** Limits: 50,000 namespaces and 20M vectors per index, and topK 50 with metadata [C]. A namespace per workspace in one index runs into both, so the layout is an index per organisation, or per shard of workspaces.
- **Qdrant.** Qdrant is the opt-in `memory` compose profile, and mem0 is a library [T]. Retiring them means exporting memories first (§11-Q10).

### 6.4 Durable runs on Workflows

- **Shape.** TypeScript Workflows in `edge/`. Each step calls the API through the router as a system principal, and the work runs in Python. Python Workflows are used only after Cloudflare confirms they're GA.
- **Idempotency key.** Each step's key comes from the run id and the step name. Recording the key happens in the same database transaction as the step's database effect, so a retry finds it and returns the recorded result.
- **Outbound calls.** Calls that take an idempotency header send it.
- **LLM calls** take no idempotency key, so a retried step can spend again. Retries are capped, and spend per run is bounded by AI Gateway budgets.
- **Step results are capped at 1 MiB** [C]. Large results go to R2 by reference.

### 6.5 Records: two paths, decided at a gate

**Path 1 (default for Part B):** PlanetScale Postgres through Cloudflare. No rewrite.

**Path 2:** each workspace's records in its cell, with a D1 control plane. This is not an adapter swap. Durable Object SQL has no SQLAlchemy dialect over HTTP, and the API can't hold a transaction open on a cell across calls. So domain logic that writes moves into the cell, and the transaction-scoped locks and claims (§1.1) move with it.

**The gate (US-B05).** A prototype puts one workspace's chat, board and schedules in a cell, built from only the tables they touch. It reports:
- latency against Path 1;
- lines of domain logic moved into the cell;
- memory headroom. A Durable Object is billed against a 128 MB allocation [C], and Pyodide plus SQLAlchemy in 128 MB is the main risk, so the prototype may be TypeScript.

The rewrite (US-B06) starts only if Gerard chooses Path 2 (§11-Q8).

### 6.6 Vault

Cloudflare has no general KMS or HSM; its HSM support covers TLS keys only, and Secrets Store is beta with 100 secrets per account [C]. A Vault Worker holds the master key; each workspace's data key is stored wrapped; every use is logged to an R2 bucket with a bucket lock. Customer-held keys need an outside KMS behind the same port (§11-Q12).

### 6.7 Identity

Clerk stays for self-serve sign-up. A trusted-proxy lane accepts identity from a proxy in front of Automatos, as one resolver in PRD-266's registry (`core/auth/bearer_registry.py`):
- **Cloudflare Access:** the `Cf-Access-Jwt-Assertion` JWT, checked for signature, issuer and audience (AUD) [C].
- **GCP IAP:** its signed header, checked the same way.
- **Unsigned headers** (oauth2-proxy's) only behind a proxy that strips client-sent copies, on a network the API accepts from no one else.

It leaves PRD-257's open question 4 alone (§11-Q11).

### 6.8 Compliance and observability

- **Coverage lists.** Containers aren't on Cloudflare's SOC 2, PCI DSS or FedRAMP High lists; Queues aren't on the PCI DSS or FedRAMP High lists [C].
- **Regulated workspaces.** A workspace that needs PCI DSS or FedRAMP High stays on the containers profile until Cloudflare confirms both products (FR-7). The required regime is a new workspace column, added by an additive migration.
- **Observability.** OTel stays the instrumentation. Container logs go to Workers Logs (7 days) and Logpush; metrics go to Analytics Engine; container traces go to an OTLP backend the operator chooses (§11-Q13).

## 7. User stories

Every story: tests go in with it and fail without it; CI is the gate; route-manifest and config-surface updated when routes or settings change; within the AGENTS.md limits; commits signed off. Estimated PR counts are in §9.

### Part A

#### US-A01: The shared package and one logging module
**Description:** As a maintainer, I want one copy of the shared logging and metrics code.

**Acceptance Criteria:**
- [ ] `packages/automatos-platform` exists as in §4.1
- [ ] The five logging copies and four metrics copies are diffed first; the PR lists every difference and which version wins
- [ ] `automatos_platform.logging` and `.metrics` replace every copy, and the copies are deleted
- [ ] Each service's log lines are unchanged except for the listed differences (a test compares a record's JSON per service)

#### US-A02: Build contexts
**Description:** As an operator, I want every image to build with the shared package, with deploys otherwise unchanged.

**Acceptance Criteria:**
- [ ] The option chosen in §11-Q2 is applied to `docker-compose.yml`, `docker-compose.dev.yml` (the package mounted for reload), `.github/workflows/images.yml` and the Railway services (the Railway change is human-gated)
- [ ] `scripts/stack_images.py` tags images from the context tree and the package tree together; a test proves a package change retags every image and an unrelated commit retags none
- [ ] Each image's compressed size stays within 1% of before, and its hashed lock is unchanged
- [ ] `docker compose up`, `make up`, the kind end-to-end test and the smoke-images lane pass

#### US-A03: One scheduler, fenced
**Description:** As an operator running two API replicas, I want scheduled work to run once, and billing jobs never twice.

**Acceptance Criteria:**
- [ ] The lease in §4.2, built on `core/database/boot_lock.py`'s pattern, with a generation number
- [ ] The `fcntl` lock and the four fallback `AsyncIOScheduler`s are deleted
- [ ] Two API processes against one database run each job once; when the leader stops or loses its connection, it stops its scheduler, and the other takes over within one lease period (tests)
- [ ] The credit-meter and Composio sync jobs are proven idempotent for a repeated run window (tests); the PR links the billing approval (§11-Q5)
- [ ] The Helm chart's `api.replicas: 1` pin and its comment go

#### US-A04: Shared API rate limits
**Description:** As an operator, I want rate limits to hold across replicas.

**Acceptance Criteria:**
- [ ] slowapi uses Redis storage; a two-process test proves one limit across both
- [ ] A storage error behaves as §11-Q3 decides, with the policy plane on and off (tests)

#### US-A05: The worker is handed its tasks
**Description:** As an operator, I want the worker to receive tasks without polling and without database or Redis credentials.

**Acceptance Criteria:**
- [ ] `ports/sandbox.py` with the containers adapter (§4.4); `api/workspace_github.py` uses it
- [ ] The API records the task in `task_executions` before sending, and a relay re-sends undelivered tasks. A task sent while the worker is down runs when it returns; a task survives an API restart (tests). Any migration is additive and moves both Alembic head pins
- [ ] The worker claims each task through the API with one conditional update on `task_id` (§4.4). A test fails the acknowledgement after a task ran, re-sends it, and proves it doesn't run again and the recorded result comes back. A test proves an expired claim lets a dead worker's task run once more
- [ ] Each side-effecting step records its idempotency key and result through the API (§4.4). A test kills the worker after a step's external effect but before completion is recorded, lets the claim expire, and proves the re-run replays the recorded result instead of repeating the effect
- [ ] The worker reports status, results and events through an authenticated API callback. `task_executions_db.py`, its Redis keys and the `RPOP` loop are deleted
- [ ] `DATABASE_URL`, `POSTGRES_*` and `REDIS_URL` are removed from the worker's environment in compose and the chart
- [ ] The dead `taskrunner:queue:*` code and the `kubernetes` backend name are deleted
- [ ] A compose smoke test runs a GitHub task end to end

#### US-A09: The worker holds no model-provider key
**Description:** As a workspace owner, I want my agents' code unable to read the platform's model credentials.

**Acceptance Criteria:**
- [ ] An internal model proxy on the API (§4.4) checks a per-task token scoped to the workspace, run and `task_id`, adds the provider credential, and forwards; it serves the providers the worker uses today
- [ ] The worker's model clients and the Claude Code CLI use the proxy's base URL; a Claude Code session with tools runs end to end through it
- [ ] `ANTHROPIC_API_KEY` and `CLAUDE_CODE_OAUTH_TOKEN` are removed from the worker's environment in compose and the chart. If the OAuth token can't be proxied, the PR says so and keeps it as a named exception, with §11-Q17 reopened
- [ ] A test reads the running worker's environment and finds no provider key. Negative tests: a token for one workspace is refused for another, and a token for one task is refused for a sibling task in the same run

#### US-A06: One resumable event transport
**Description:** As a user watching the board, the canvas or a run, I want updates that survive a reconnect, and never arrive before the change they describe.

**Acceptance Criteria:**
- [ ] `ports/events.py` with the Redis Streams adapter, and the `event_outbox` table with its relay. The migration is additive and moves both head pins
- [ ] The six SSE endpoints read from the port; the pg LISTEN threads are deleted
- [ ] State events are written to the outbox in the change's transaction; a test proves a rolled-back change publishes nothing
- [ ] SSE responses carry `id:`; the frontend reconnects with `Last-Event-ID` and receives only missed events (backend and vitest tests)
- [ ] Chat's direct stream is unchanged

#### US-A07: Redis durability
**Description:** As an operator, I want durable keys to survive memory pressure.

**Acceptance Criteria:**
- [ ] Compose, the chart and the Railway notes use the policy chosen in §11-Q4
- [ ] A test proves every cache write sets a TTL (if `volatile-lru`)
- [ ] A contract test fills Redis past its limit and proves streams, the job store and outbox state survive

#### US-A08: The ratchet
**Description:** As a reviewer, I want CI to stop new uses of what Part A replaced.

**Acceptance Criteria:**
- [ ] `.importlinter-platform` with the contracts in §4.7 runs for the orchestrator and each service, green, with today's imports as `ignore_imports`
- [ ] Each Part A story deletes its entries; the README explains the ratchet
- [ ] The contract-test lane runs in CI

### The spike

#### US-S01: Today's stack on Cloudflare
**Description:** As Gerard, I want to see Automatos running on Cloudflare with as little change as possible, and what it costs.

**Acceptance Criteria:**
- [ ] `edge/` with `wrangler.jsonc`, the router Worker and the API container class (the same API image), on PlanetScale, a managed Redis, R2 and AI Gateway; staging only, human-gated
- [ ] The web app on Workers via `@opennextjs/cloudflare`; the two Edge-runtime routes on Node, with the web app still working on every profile
- [ ] Measurements 1–4 and 6 from §5 in the report

#### US-S02: A sandbox per workspace
**Description:** As a workspace owner, I want my agents' code to run in a sandbox that holds no secrets and can't reach another workspace.

**Acceptance Criteria:**
- [ ] The Sandbox port's Cloudflare adapter (§5); the sandbox image has no database, Redis or model credential (a test reads its environment)
- [ ] Internet off, HTTPS intercepted, the CA trusted; model and Hub API keys added by the outbound handler
- [ ] Isolation tests: no other workspace's files; no off-list host; no port but 80 and 443
- [ ] Measurement 5 from §5 in the report

#### US-S03: The report
**Description:** As Gerard, I want one document to decide Part B on.

**Acceptance Criteria:**
- [ ] The report gives §5's six measurements, the spike's cost, what broke, and a recommendation on each Part B section
- [ ] It's added to this PRD as an appendix, and §11-Q7 is put to Gerard with it

### Part B (only after §11-Q7)

Each Part B story gets its full acceptance criteria when Part B is approved; the design they implement is §6.

- **US-B01:** the bridge and the caller contract (§6.1), with negative tests for every rejection, system-principal audit, and token refresh.
- **US-B02:** cells without records (§6.2). Locks, schedules (idempotent alarm jobs), browser fan-out over hibernatable WebSockets, sandbox ownership, and FR-7.
- **US-B03:** the embedding migration and Vectorize (§6.3). Dual-write, backfill, switch and rollback; the index layout; Qdrant export if §11-Q10 says retire.
- **US-B04:** Workflows (§6.4). Atomic idempotency records, capped retries, results by reference over 1 MiB, and a fail-after-effect test.
- **US-B05:** the records prototype and gate (§6.5). Latency, logic moved and memory headroom reported; Gerard records §11-Q8.
- **US-B06:** records into cells, only if Path 2. Workspaces move one at a time, with verification and rollback; cross-workspace reads get a read model.
- **US-B07:** the vault, the trusted-proxy lane, channels (Browser Run, inbound email) and the Analytics Engine metrics adapter (§6.6–§6.8).

## 8. Functional requirements and technical considerations

### 8.1 Functional requirements

- FR-1: Only `automatos_platform/profile.py` chooses an adapter; no other module branches on `PLATFORM_PROFILE`.
- FR-2: Part A changes the container model only in these ways, each tested in its story:
  - US-A03: one scheduler across replicas, which stops when its lease is lost;
  - US-A04: rate limits shared across replicas;
  - US-A05: the worker is handed tasks over HTTP, claims each one once, an undelivered task is re-sent, and the worker drops its database and Redis credentials;
  - US-A09: the worker's model calls go through the API's model proxy, and the worker drops its provider keys;
  - US-A06: streams resume, and state events publish after commit through an outbox;
  - US-A07: Redis eviction policy.
- FR-3: The spike changes the container model only by moving two frontend routes from the Edge runtime to Node (US-S01).
- FR-4: Each port's contract tests pass for every adapter that exists.
- FR-5: Agent code holds no database, Redis or model-provider credentials: Part A for the worker (US-A05 and US-A09), the spike and Part B for sandboxes.
- FR-6: Compose, the Helm chart and Railway deploy and pass their existing smoke tests at every story.
- FR-7 (Part B): the cloudflare profile refuses a cell for a workspace whose required compliance regime isn't confirmed for every product the cell uses.

### 8.2 Cloudflare limits that shape the design [C]

| Product | Limit or status that matters |
|---|---|
| Containers | GA (Apr 2026); up to 4 vCPU, 12 GiB, 20 GB disk; scaled manually (no autoscaling yet); ephemeral disk; outbound handlers on ports 80 and 443 only, HTTPS only with `interceptHttps` and the injected CA trusted; `allowedHosts` is deny-by-default, and while `enableInternet` is on, other ports bypass `deniedHosts` |
| Sandbox SDK | GA Apr 2026, 1.0 on 30 Sep 2026; a security notice on 22 Oct 2026 required image rebuilds |
| Durable Objects | SQLite storage GA, 10 GB per object, 30-day point-in-time recovery, billed against 128 MB; one alarm per object, at least once with up to 6 retries; soft limit around 1,000 requests/s per object |
| Workflows | Up to 25,000 steps, unlimited wall time, 1 GB state, step results up to 1 MiB; TypeScript, and Python (beta per secondary sources) |
| Vectorize | 1536 dimensions, 20M vectors per index, 50,000 namespaces, topK 50 with metadata; no keyword search |
| D1 | 10 GB per database (fixed), one query at a time per database |
| Queues | 128 KB messages, 5,000 messages/s per queue, no priority or ordering guarantee |
| Secrets Store | Beta; 100 secrets per account |

### 8.3 New dependencies

- **Python:** none.
- **npm, `edge/` only (spike on):** `wrangler`, `@cloudflare/sandbox`, and `@opennextjs/cloudflare` in `frontend/`. Each needs Gerard's approval, and §11-Q15 asks who maintains `edge/` and its CI lane.

### 8.4 Risks

- **Part A touches billing jobs, a guardrail and streaming at once.** That's why each is its own story with its own approval.
- **The spike may show the API can't live away from its database.** That is a valid result: Part B's Path 1 then needs a different database placement, or Part B stops.
- **Several Part B products are beta:** Secrets Store, Python Workflows, sandbox snapshots and Email Service.

## 9. Sequencing and size

| Step | Content | Depends on | Estimated PRs |
|---|---|---|---|
| A01 | Package, logging and metrics | §11-Q1 | 1 |
| A02 | Build contexts | A01, §11-Q2 | 1–2 |
| A03 | Fenced scheduler lease | A01, §11-Q5 | 2 |
| A04 | Shared rate limits | §11-Q3 | 1 |
| A05 | Worker hand-off and credentials | A01 | 2–3 |
| A06 | Event transport, outbox, frontend resume | A01 | 3 |
| A07 | Redis durability | §11-Q4 | 1 |
| A09 | The worker's model proxy | A05, §11-Q17 | 2 |
| A08 | Ratchet and contract lane | A01 | 1 |
| Spike | US-S01–S03 | A05 (the sandbox reports through the API), §11-Q6 | 3–4, over two weeks |
| Part B | US-B01–B07 | §11-Q7 | Sized when approved |

Part A is about 14–16 PRs, several of which can run in parallel after A01. Dates are Gerard's call; nothing is proposed before WebSummit (10–13 Nov).

## 10. Success metrics

- **Part A:**
  - two API replicas on the Helm chart run each scheduled job once, and billing jobs never double-count in a lease handover test;
  - a reconnecting viewer misses no board event;
  - the worker's environment holds no database or Redis credential;
  - no new import of `redis`, `core.redis.client` or `apscheduler` outside the allowlist.
- **The spike:** its report answers §5's six measurements. There are no pass bars; the numbers inform §11-Q7.
- **Part B bars,** proposed for Gerard to confirm or move at §11-Q7:
  - a bridge call adds under 20 ms at p95 inside one location;
  - a sandbox wakes in under 10 s with 1 GB restored;
  - the records prototype is within 10% of Path 1 at p95, with memory headroom above 25%.

## 11. Open questions for Gerard (each with a recommendation)

1. **Approve Part A?** *Recommendation:* yes. It fixes real problems on Helm and Railway and makes the worker safer, whether or not Cloudflare happens.
2. **Build contexts: repo-root contexts or a synced copy (§4.1)?** *Recommendation:* repo-root, with a `.dockerignore` per Dockerfile and tags from context plus package trees.
3. **Rate-limit storage errors with the policy plane on: fail open (logged) or closed?** *Recommendation:* fail open for storage errors only, logged and alerted, so a Redis outage doesn't take the API down; denials still fail closed.
4. **Redis eviction: `volatile-lru` with a TTL on every cache write, or a second `noeviction` Redis?** *Recommendation:* `volatile-lru`; one Redis is simpler for self-hosters.
5. **Billing:** approve moving the credit-meter and Composio sync jobs under the fenced lease, with idempotency tests? *Recommendation:* yes; it's what makes multiple replicas safe for billing.
6. **Run the spike?** It needs a Cloudflare account for staging (Workers Paid, Containers), a PlanetScale database and a managed Redis. *Recommendation:* yes, after A05, on an Automatos staging account.
7. **After the spike: start Part B, and which parts?** Decide on the report and on what Orderly commits to (the whole stack in their account, or only agent execution there). *Recommendation:* decide then, not now.
8. **Records: Path 1 or Path 2 (Part B)?** *Recommendation:* Path 1; Path 2 only on US-B05's numbers.
9. **The embedding model at 1024 dimensions on both profiles, with a dual-write migration (Part B)?** *Recommendation:* yes, if Part B goes ahead.
10. **Export and retire Qdrant memory (Part B)?** *Recommendation:* decide with Q9; it's an opt-in profile today, so fewer users are affected.
11. **The trusted-proxy lane (Part B), leaving PRD-257's question 4 open?** *Recommendation:* yes.
12. **An outside KMS for customer-held keys (Part B)?** *Recommendation:* yes; Cloudflare has none.
13. **Container traces to an operator-chosen OTLP backend on Cloudflare (Part B)?** *Recommendation:* yes.
14. **Ask Cloudflare's account team:**
    - Containers' and Queues' compliance scope;
    - D1 residency (its docs disagree);
    - autoscaling timing;
    - Python Workflows' status.

    *Recommendation:* during the spike.
15. **Who maintains `edge/` (TypeScript) and its CI lane?** *Recommendation:* Dave, through the spike; revisit at Q7.
16. **Confirm the proposed non-goals:**
    - moving the local edition off Postgres or Redis;
    - choosing a sign-in design beyond the trusted-proxy lane;
    - the Orderly partnership's own features;
    - Cloudflare hosting for the marketing sites;
    - replacing Composio or the model providers.

    *Recommendation:* confirm.
17. **The worker's model proxy (US-A09):** route the worker's model calls, the Claude Code CLI included, through an API proxy that adds the key, so the worker holds no provider credential? This changes how Claude Code sessions authenticate. *Recommendation:* yes. It's the container-model version of "the agent never holds a secret", and the spike's sandboxes need the same thing.
18. **KEDA in the Helm chart?** KEDA would scale the workspace worker and background consumers on their backlog, down to zero when idle, as an optional, off-by-default chart setting. Part A creates the backlog it needs: pending tasks in `task_executions` (US-A05) for KEDA's Postgres scaler, and the event streams (US-A06) for its Redis Streams scaler. Today's polling worker gives it nothing clean to scale on. On the reference workload in the hosting comparison (claude.ai/artifact/TLLxQmuFx9Fto7pXdoWmtU), it saves about $70 a month on GKE; the worker holds `/workspaces` on a volume, so scaling it past one replica needs the per-workspace sandbox work. It's a new chart dependency (the KEDA operator), so it needs approval. *Recommendation:* yes, as a follow-on story after US-A05 and US-A06, scaling to zero but capped at one worker replica until sandboxes are per workspace.

## Related

- PRD-262 (container diet): the image budgets and tree tags A02 must keep.
- PRD-266 (Automatos Go backend): the credential registry the trusted-proxy lane joins.
- PRD-257 (draft, enterprise estate agents): its open question 4 is not decided here.
- PRD-256 (OpenTelemetry): the instrumentation both profiles keep.
- PRD-211 (topology discipline): the import-linter ratchet A08 follows.
- The platform-engineering notes (daarthur/automatos-platform-engineering): the seven contracts (C3: the agent never holds a secret), the workspace as the isolation unit, and the K1 scheduler blocker.
