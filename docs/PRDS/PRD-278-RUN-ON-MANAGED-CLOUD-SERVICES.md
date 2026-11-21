# PRD-278: Run on a cloud account's own services — storage by the pod's role, TLS to the cache and the database, uploads in object storage, embeddings on Bedrock

**Status:** Draft for Dave's review. Written 20 Nov 2026. No code has been written.
**Owner:** Gerard · **Reviewer:** Dave
**Grounded on:** `automatos-ai` `main @ 97c727136`, read on 20 Nov 2026. 278 is the next free number: 277 is the highest on main.
**Labels:** **[T]** read in code at that commit; **[I]** inferred; **[G]** Gerard's decision; **[?]** not known.
**Relates to:** PRD-277 (the `enterprise` edition, whose installs this PRD makes runnable on managed services). PRD-279 to PRD-282 are its siblings: releases, the chart, the credential key, and operations.

**Decisions already taken [G] (Gerard, 19 Nov 2026):**

| | Decision |
|---|---|
| G1 | A self-hosted install for an organisation runs in a cloud account that organisation owns. |
| G2 | Models run on Bedrock in that same account by default. Another provider is something the operator switches on. |
| G3 | "It could be AWS, Azure, Google." AWS is first. Anything specific to one cloud sits behind a seam that already exists. |

---

## 1. Introduction

An organisation that runs Automatos in its own cloud account wants three things from the data layer that core cannot give today:

- **No stored cloud key.** A pod gets short-lived credentials from its own role. There is no access key to put in a secret, rotate or leak.
- **Encrypted connections that are checked.** Managed PostgreSQL and Redis require TLS, and a reviewer expects the client to check who it is talking to.
- **Nothing leaving the account by default.** That includes the text of documents, which today goes to an outside provider to be embedded.

Core is close. Its storage client already uses the pod's role when no key is set, and then a separate check refuses to use it. Its Bedrock client already works from a role. Its Redis client is handed a TLS address and drops the TLS. This PRD closes those gaps, moves uploaded files off the API's own disk, and adds embeddings on Bedrock.

## 2. What exists today

| Area | Today | Where | What it means |
|---|---|---|---|
| Storage client | Passes a key pair only when both parts are set. Otherwise it leaves credentials to the SDK's own chain. | `orchestrator/core/storage/s3.py:126-135` **[T]** | It would already work from a pod's role. |
| Storage gate | `is_storage_configured()` is true only with a custom endpoint or a full key pair. Its docstring says the SDK's own chain "does not" count. `_get_client` raises `StorageNotConfigured` otherwise. | `core/storage/s3.py:93-100`, `:143-144` **[T]** | A pod with a role and no key is told storage is not configured. |
| Who asks the gate | A dozen modules: document generation, thumbnails, brand assets, downloads, socials media, workspace purge and move. Most fall back to local disk or skip the step. | `modules/documents/generation_service.py:294`, `services/document_downloads.py:63`, `modules/documents/thumbnails/store.py:64` and others **[T]** | With a role and no key, each silently takes the local path. |
| Search index client | Passes the two key settings as they are. When they are unset the SDK's chain is used. | `modules/search/vector_store/backends/s3_vectors_backend.py:77-81` **[T]** | Works from a role today. |
| Search index creation | Not at start-up: on first use, and again on every search, add and delete. Each of them calls `_ensure_setup`, which tries to create the vector bucket and the index and ignores only the answer "already exists". Any other answer fails that call. It refuses to run against an index of a different size, and never deletes one. | `s3_vectors_backend.py:92-128`, `:130-161`; called from `:187`, `:272`, `:335`, `:403` **[T]** | An operator who wants the index encrypted with their own key must create it first. A role that may not create one fails every search, add and delete, even when both exist. |
| Bedrock chat | `bedrock-runtime`, with three ways to authenticate. With no key it uses the SDK's chain: "environment, profile, instance role". | `core/llm/clients/bedrock_client.py:60-72` **[T]** | The client works from a role. |
| Bedrock, further up | The provider is registered with an access key as its "env key", and the agent factory passes the key settings through. | `core/llm/providers.py:167-169`, `modules/agents/factory/agent_factory.py:511-513` **[T]** | Whether a Bedrock model is offered at all with no key is not known **[?]**. |
| Bedrock, provider and region | Agents store Bedrock as `bedrock` or `aws_bedrock`. Activation checks the stored name against the provider enum and sends any name the enum does not hold to OpenRouter. The client takes its region from `base_url`: agent creation sets `AWS_REGION`; activation and the spoken model set none, so the client's default applies; the settings path takes the stored credential's region. | `modules/agents/factory/agent_factory.py:513`, `:936-944`, `:993`; `consumers/chatbot/spoken_model.py:139-144`; `core/llm/manager.py:560-561`; `core/llm/clients/bedrock_client.py:63` **[T]** | US-105 makes both names reach Bedrock on every path, in `AWS_REGION` on the install's role. |
| SDK services in the image | The image keeps six: `s3`, `s3vectors`, `bedrock-runtime`, `sts`, `sso`, `sso-oidc`. A test requires every client to name one of them. | `orchestrator/tests/test_prd262_botocore_services.py` **[T]** | Embeddings through `bedrock-runtime` need nothing new. |
| Redis, main client | Built from a host, port, password and database number. `get_redis_client()` parses `REDIS_URL` and keeps only those four. The scheme is never read. | `core/redis/client.py:26-44`, `:180-192` **[T]** | `rediss://` is treated as `redis://`. The connection is plain. A user name in the address is dropped. |
| Redis, other clients | The async publish and subscribe client and the chat-turns client are built from the same four fields. The task queue builds two more from `REDIS_HOST`, `REDIS_PORT` and `REDIS_PASSWORD`, and ignores `REDIS_URL`. Two clients pass the whole address to `from_url` and do honour `rediss://`. A third does the same and falls back to the host fields. | `core/redis/client.py:62-82`, `services/chat_turns.py:93-110`, `core/task_runner/queued.py:109-135`, `modules/context/adapters/redis_context.py:37-50`, `modules/intake/progress.py:103`, `services/workspace-worker/main.py:101` **[T]** | TLS works only where the whole address is passed through. |
| Redis commands | `KEYS` and `SCAN`, `MGET`, a blocking pop over several keys, publish and subscribe. No cluster client anywhere. | `core/cache/service.py:103`, `:221`; `core/task_runner/queued.py:348`, `:249` **[T]** | One non-cluster instance, as now. |
| Database TLS | In production, `sslmode=require` is added to the address when it names none. That encrypts and does not check the server's certificate. | `core/database/database.py:70-80` **[T]** | No check of who answered. |
| Database, other paths | The scheduler's lock connection, two `asyncpg` connections, the documents routes' and the ingestion manager's own `psycopg2` connections, the start-up scripts, Alembic and the worker each build their own connection, and none gets that suffix. | `core/jobs/leadership.py:48`, `:144-148`; `modules/rag/bm25_leg.py:54`; `modules/rag/service.py:1327`; `api/documents.py:82-88`; `modules/rag/ingestion/manager.py:706`; `core/database/init_database.py:49`; `alembic/env.py:30-35`; `services/workspace-worker/task_executions_db.py:87` **[T]** | TLS depends on which path a query takes. |
| Database CA | No setting for a CA bundle. | not found in `config.py` **[T]** | A certificate check can only be set by hand, inside the address. |
| Uploads | `POST /upload` writes the file to `DOCUMENT_UPLOAD_DIR` and stores that local path in the database. | `api/documents.py:153-156`, `api/document_upload_steps.py:108-111` **[T]** | The original lives on the API's disk. |
| Uploads, the other path | Mission uploads already go to object storage, and the row's `file_path` becomes `s3://bucket/key`. Downloads and reprocessing already understand that form. | `modules/rag/ingestion/manager.py:622-645`, `:715-717`; `api/missions.py:1006`; `services/document_downloads.py:50-63`; `api/documents.py:827-841` **[T]** | The path exists. The main upload route does not use it. |
| The API's disk in the chart | A 10 GB volume that one pod can mount, by default. The chart refuses a second API replica unless the volume's access mode lets several pods mount it. | `charts/automatos/values.yaml:66-79`, `templates/_helpers.tpl:118` **[T]** | With the default, the API is tied to one node's zone. |
| Embedding providers | OpenAI, OpenRouter and a local Hugging Face model. Others in the enum are not built. No Bedrock. | `core/llm/embedding_manager.py:169-181`, `core/llm/clients/base.py:60-68` **[T]** | Document text leaves the account to be embedded, or a model is downloaded into the pod. |
| Embeddings with no key | A missing key or an unset provider gives a deterministic stand-in, quietly. | `embedding_manager.py:114`, `:145` **[T]** | An index can be filled with vectors that mean nothing. `test_p2w0_embeddings_fail_loud.py` covers some of this **[?]**. |
| Sizes | Three settings default to 2,048: `VECTOR_STORE_DIMENSIONS`, `S3_VECTORS_DIMENSION`, `FIELD_EMBEDDING_DIM`. A longer embedding is cut to fit. Nothing checks that the model can reach the index's size. | `config.py:1604`, `:1757`, `:1808`; `modules/rag/ingestion/manager.py:1411-1420` **[T]** | A model that produces fewer dimensions than the index expects is found out at the first write. |

## 3. Goals

- An install runs with no cloud key anywhere: object storage, the search index and Bedrock all work from the pod's own role.
- The connection to the cache is encrypted, and the connection to the database is encrypted and checked, on every code path, from one setting each.
- An uploaded file's original lives in object storage whenever object storage is configured, so the API can run with no disk of its own.
- Embeddings can be made on Bedrock.
- An install whose embedding model cannot fill its index refuses to start, and says why.
- The `local` and `saas` editions keep working. Where their behaviour changes, section 8 says so and asks.

## 4. Non-goals

- **A second API replica.** Removing the disk is necessary for it and not enough. PRD-267 and PRD-276 carry the rest.
- **Other object stores.** Storage stays the S3 API: S3 itself or a compatible endpoint.
- **Redis in cluster mode.** Core stays written for one instance.
- **Cloud permissions.** Which role may do what is the operator's to define.
- **Re-embedding what is already indexed.** Changing the embedding model on a live install means indexing again, and that is not built here.
- **Streaming large uploads.** The route still reads up to 50 MB into memory.

## 5. Design

### 5.1 Credentials from the pod's role

One new setting, `AWS_USE_WORKLOAD_IDENTITY`, off by default.

- **Why a setting and not a guess.** The gate excludes the SDK's own chain on purpose: a developer's laptop often has a profile lying around, and core should not start writing to a bucket because of it. So the operator says so in words.
- **With it on,** object storage counts as configured, and no key pair is passed to any client.
- **With it on and a key pair also set,** the install refuses to start. Two sources of credentials is a mistake waiting to be found.
- **At start-up the API proves it can reach the documents bucket** with one cheap call, and fails with a message that names the bucket and the role's problem. A wrong role is then found at install, not at the first upload.
- **Today the search index is not checked at start-up.** Core tries to create it on first use and on every search, add and delete. With `S3_VECTORS_CREATE_IF_MISSING` off, this wave adds a real start-up check of the index in a shared vector bucket. A vector bucket named per workspace is known only once a workspace uses it, so its index is still checked on first use.
- **A Bedrock model is offered with no key.** US-103 finds whatever further up treats a missing key as "provider not set up" and makes the setting answer it.
- **The search index can be left alone.** A second setting, `S3_VECTORS_CREATE_IF_MISSING`, on by default as today. With it off, a missing index is an error with a clear message, and core never tries to create one. An install whose role may not create an index needs it off, or none of its searches, adds and deletes can run.
- **A Bedrock model runs on Bedrock, in the install's region.** US-105: whichever name an agent stores for Bedrock, and whichever path builds its client, the call goes to Bedrock. A call on the install's role goes to `AWS_REGION`. A configuration with its own key keeps the region it names, and one that names no region uses `AWS_REGION`. A workspace's own Bedrock key is found under either name.

### 5.2 TLS to Redis

One function turns the Redis settings into connection arguments, and every client in the orchestrator is built from it.

- `rediss://` means TLS with the server's certificate checked. `redis://` means plain, as today.
- A user name in the address is kept.
- When the address is put together from `REDIS_HOST`, `REDIS_PORT` and `REDIS_PASSWORD`, a new `REDIS_TLS` setting chooses the scheme.
- The task queue stops reading the three parts itself and uses the same function.
- A test fails if anything in the orchestrator builds a Redis client another way, as the storage module's test already does for its clients.

### 5.3 TLS to the database, checked

Two new settings: `DATABASE_SSLMODE` and `DATABASE_SSLROOTCERT`. One function turns them into what each driver needs, and every connection path in section 2 uses it.

- With neither set, behaviour is as today: `require` in production.
- With `DATABASE_SSLMODE=verify-full` and a CA file, the server's certificate and name are checked.
- The CA file is the operator's to supply. Core ships no cloud's bundle. The chart mounts one from a ConfigMap (US-304).
- The unused `Config.get_database_url()` helper, which skips TLS whenever "db", "postgres" or "localhost" appears anywhere in the address, is deleted (`config.py:62-72` **[T]**).
- **The default stays as it is,** so that no existing install stops. D-6 asks whether the `enterprise` edition should refuse to start without a checked connection.

### 5.4 Uploads in object storage

When object storage is configured, the upload route stores the original the way mission uploads already do, and records `s3://bucket/key`. The file touches local disk only while it is being read and split, in a temporary folder that is removed afterwards.

When object storage is not configured, nothing changes: the local path, as in the `local` edition today.

Uploads are not the only thing the API keeps on disk. Generated documents, thumbnails, brand assets and deliverables are written under `DOCUMENT_STORAGE_DIR`, some with a copy in object storage and some without **[T]**. US-401 lists every one before anything is changed, and US-403 makes object storage the home for each when it is configured. The aim is one sentence: **with object storage configured, the API runs with no volume of its own.**

Files already on an install's disk are moved by a durable job (PRD-276), which can be run again safely.

### 5.5 Embeddings on Bedrock

A fourth provider, `bedrock`, beside the three that exist, behind the same `EmbeddingManager`.

- It calls `bedrock-runtime` with the same credential handling as the chat client, so it works from a role.
- The first model is Amazon Titan Text Embeddings V2, which produces up to 1,024 dimensions **[I]**. The model is a setting, so a second can follow.
- **Its size is smaller than core's default.** The three size settings default to 2,048 **[T]**. An install that chooses this model sets all three to 1,024 before its first start, and section 5.6's check says so if it has not. An install that already has an index at another size cannot switch without indexing again, which section 4 leaves out.
- A model that takes one text at a time is called with a bounded number of calls in flight, inside the existing time limits (`EMBEDDING_BATCH_TIMEOUT_S`).
- No call blocks the event loop.

### 5.6 Sizes that agree

At start-up, when a real embedding provider is set:

- If the model cannot produce at least as many dimensions as the index expects, the install refuses to start and names the two settings.
- Cutting a longer embedding down to the index's size stays as it is. That is how the current default model is used **[T]**.
- The same check covers the memory store's size (`FIELD_EMBEDDING_DIM`).

And one rule for a real index: when a provider is set and cannot be used, embedding fails. The deterministic stand-in is for an install with no provider at all, never a fallback for a broken one.

### 5.7 Settings

| Setting | Default | Meaning |
|---|---|---|
| `AWS_USE_WORKLOAD_IDENTITY` | off | Use the pod's own role. No key pair may be set with it. |
| `S3_VECTORS_CREATE_IF_MISSING` | on | Off: never create the vector bucket or index. |
| `REDIS_TLS` | off | Used only when the address is built from host, port and password. |
| `DATABASE_SSLMODE` | unset | Unset keeps today's behaviour. |
| `DATABASE_SSLROOTCERT` | unset | Path to a CA file. |
| `EMBEDDING_PROVIDER=bedrock` | | With `EMBEDDING_MODEL` naming the Bedrock model. |

Each is read through `config.py` and added to `orchestrator/reports/config-surface.json`.

## 6. Waves and user stories

Each wave is one pull request or a short series. CI is the only gate: nothing here is run on a developer's machine.

### Wave 1: Credentials from the pod's role

#### US-101: The setting and the gate · S
**Description:** As an operator, I want to tell core to use the pod's own role, so that I store no cloud key.

**Acceptance Criteria:**
- [x] `AWS_USE_WORKLOAD_IDENTITY` exists in `config.py` and in `config-surface.json`.
- [x] With it on and no key pair, `is_storage_configured()` is true and a client is built with no key arguments.
- [x] With it on and a key pair set, start-up fails with a message naming both.
- [x] With it off, every existing shape in `test_is_storage_configured_shapes` behaves as today.
- [ ] `ruff check` and the backend test lanes pass in CI. Open until CI shows it.

#### US-102: Proving the role at start-up · S
**Description:** As an operator, I want a wrong role found when the install starts, so that the first upload is not how I learn about it.

**Acceptance Criteria:**
- [x] With the setting on, start-up makes one cheap call to the documents bucket.
- [x] A refusal fails start-up with a message that names the bucket and the kind of failure, and quotes no credential.
- [x] The call runs off the event loop and has a short time limit.
- [x] With the setting off, no call is made.

#### US-103: Bedrock with no key · M
**Description:** As an operator, I want Bedrock models offered and usable from the pod's role, so that the default model path needs no key either.

**Acceptance Criteria:**
- [ ] The pull request states what, above the client, treats a missing key as "not set up", with file and line. Open until the pull request is written.
- [x] With the setting on and no key, a Bedrock model resolves and the client is built on the role branch.
- [x] With the setting off, behaviour is unchanged.
- [ ] `test_1170_bedrock_converse.py` passes, with a new case for the role path. The case is added; open until CI shows it passing.

#### US-104: An index core does not create · S
**Description:** As an operator who creates the search index myself, with my own key, I want core never to create one, so that a mistake in my setup is an error and not a second, unencrypted index.

**Acceptance Criteria:**
- [x] `S3_VECTORS_CREATE_IF_MISSING` exists, on by default.
- [ ] With it off and no index, start-up fails with a message naming the bucket and index it looked for. Delivered for a shared vector bucket. Open for a bucket named per workspace: its index is checked on that workspace's first use, and a missing one fails that call with the same message.
- [x] With it off, no create call is ever made. A test asserts it.
- [x] With it off and a role that is refused both create calls, core starts against a bucket and an index that exist.
- [x] The existing size check is unchanged.

#### US-105: A Bedrock model is called on Bedrock, in the install's region · M
**Description:** As an operator who runs models on Bedrock in my own account, I want every Bedrock agent called on Bedrock, in the install's region, so that its calls stay in that account and region whichever path makes them.

**Acceptance Criteria:**
- [x] With `AWS_USE_WORKLOAD_IDENTITY` on, an agent stored as `bedrock` or `aws_bedrock` is called on Bedrock in chat, in agent creation and in the spoken model.
- [x] On every path, a call on the install's role is in `AWS_REGION`; a configuration with its own key keeps the region it names, and uses `AWS_REGION` when it names none.
- [x] A Bedrock tier with no key uses the role and is never rerouted to another provider.
- [x] With the setting off, today's behaviour is unchanged, and a test pins it.
- [ ] The pull request states what main does today, with file and line. Open until the pull request is written.

### Wave 2: TLS to Redis

#### US-201: One way to build a Redis connection · M
**Description:** As a developer, I want one function that turns the settings into connection arguments, so that TLS is decided in one place.

**Acceptance Criteria:**
- [ ] One function returns the arguments for a sync client, an async client and a pool.
- [ ] `rediss://` gives TLS with certificate checking on. `redis://` gives none.
- [ ] A user name in the address is kept.
- [ ] `REDIS_TLS` exists and chooses the scheme when the address is built from its parts.
- [ ] Unit tests cover each scheme, with and without a user name and a password.

#### US-202: Every client through it · M
**Description:** As an operator, I want every Redis connection core opens to honour TLS, so that a managed Redis that requires it works.

**Acceptance Criteria:**
- [ ] The main pool, the async publish and subscribe client, `chat_turns` and the task queue are all built from US-201's function.
- [ ] The task queue no longer reads `REDIS_HOST`, `REDIS_PORT` or `REDIS_PASSWORD` itself.
- [ ] A test fails if any file in the orchestrator outside that module constructs a Redis client or pool directly. A connection made to test a user's own saved Redis credential is not covered.
- [ ] `test_redis_client_timeouts.py` passes with its timeouts unchanged.
- [ ] The chart's documented example address is unchanged for `local`, and a TLS example is added.

### Wave 3: TLS to the database, checked

#### US-301: One place that knows the database's TLS · M
**Description:** As a developer, I want one function to say how a database connection is secured, so that no path can differ.

**Acceptance Criteria:**
- [x] `DATABASE_SSLMODE` and `DATABASE_SSLROOTCERT` exist in `config.py` and `config-surface.json`.
- [x] One function returns what SQLAlchemy with `psycopg2`, a raw `psycopg2` connection and `asyncpg` each need.
- [x] With neither setting, the main engine behaves as today, and a test pins that.
- [x] If D-6 is accepted: in the `enterprise` edition, start-up fails unless `DATABASE_SSLMODE` is `verify-full` (`core/edition_start_up.py`, through the `requires_verified_database_tls` capability), and the chart does not render it otherwise. The API enforces the rule itself; the worker and the migration step get it from the chart's values, so an enterprise install made without the chart sets both settings on all three. In every edition, `verify-ca` or `verify-full` without a CA file is refused at start (the API, the worker, the entrypoint and the chart).
- [x] `Config.get_database_url()` is deleted.

#### US-302: Every connection path · L
**Description:** As a security reviewer, I want the same TLS on every connection core opens to its own database, so that the answer does not depend on which query ran.

**Acceptance Criteria:**
- [x] The main engine, the scheduler's lock connection, both `asyncpg` connections, the documents routes' own connections and the ingestion manager's all use US-301's function.
- [x] Alembic's `env.py` uses it, so the migration job connects the same way.
- [x] The start-up scripts that open their own connection use it.
- [x] A test fails if a file in the orchestrator opens a connection to the application's database another way. Connections to a user's own data source are not covered.
- [ ] The pull request lists every path it changed, with file and line.

#### US-303: The worker · S
**Description:** As an operator, I want the workspace worker's database connection secured the same way, for as long as it has one.

**Acceptance Criteria:**
- [x] The worker reads the same two settings and applies them to its one connection.
- [x] A test covers `verify-full` with a CA path.

#### US-304: A CA bundle in the chart · S
**Description:** As an operator, I want to hand the chart my database's CA, so that the certificate check has something to check against.

**Acceptance Criteria:**
- [x] A value names a ConfigMap holding a CA file. The chart mounts it read-only in the API, the migration job and the worker, and sets `DATABASE_SSLROOTCERT`.
- [x] Without the value, nothing is mounted and nothing changes.
- [x] The chart's unit tests cover both.

### Wave 4: Uploads in object storage

#### US-401: A table of everything the API writes to disk · M · analysis only
**Description:** As a maintainer, I want every place the API keeps a file listed, so that "no disk of its own" is checked and not assumed.

**Acceptance Criteria:**
- [ ] A table lists each writer: file, line, what it writes, whether it must survive a restart, and whether an object-storage copy exists today.
- [ ] Temporary files that need no home are marked as such.
- [ ] Anything that cannot move to object storage is listed for Gerard, not guessed.

#### US-402: The upload route stores the original in object storage · M
**Description:** As an operator, I want an uploaded file's original in object storage, so that it survives the pod and its zone.

**Acceptance Criteria:**
- [ ] With object storage configured, `POST /upload` records `s3://bucket/key` in `file_path`, under the same key layout mission uploads use.
- [ ] The local copy made for reading is removed when ingestion ends, whether it succeeded or not.
- [ ] Download and reprocess work for the new row with no local file present.
- [ ] With object storage not configured, the route behaves as today, and a test pins that.
- [ ] The size and type checks are unchanged.

#### US-403: The other writers · L
**Description:** As an operator, I want generated documents, thumbnails, brand assets and deliverables kept in object storage too, so that the API needs no volume.

**Acceptance Criteria:**
- [ ] Each row of US-401's table marked "must survive" has object storage as its home when it is configured.
- [ ] Each reader finds the file there with no local file present.
- [ ] Each keeps its local behaviour when object storage is not configured.
- [ ] The series is split by writer, one pull request each.

#### US-404: Moving what is already on disk · M
**Description:** As an operator of an existing install, I want the files already on the API's disk moved, so that switching does not lose them.

**Acceptance Criteria:**
- [ ] A durable job moves each original to object storage and rewrites its row.
- [ ] It can be stopped and run again, and a file already moved is skipped.
- [ ] A file missing from disk is counted and reported, and does not stop the run.
- [ ] It reports how many it moved, skipped and could not find.

#### US-405: The API without a volume · S
**Description:** As an operator, I want to switch the API's data volume off when object storage holds everything, so that the pod is not tied to one node's zone.

**Acceptance Criteria:**
- [ ] With `api.data.enabled=false` and object storage configured, the rendered chart has no volume claim for the API.
- [ ] The chart's README says when it is safe to switch off, and names US-404.
- [ ] The chart's unit tests cover it.

### Wave 5: Embeddings on Bedrock

#### US-501: The Bedrock embedding provider · M
**Description:** As an operator, I want embeddings made on Bedrock, so that document text does not leave my account.

**Acceptance Criteria:**
- [ ] `EMBEDDING_PROVIDER=bedrock` selects a provider that calls `bedrock-runtime`.
- [ ] It works from the pod's role and from a key, through the chat client's existing credential handling.
- [ ] It reports its dimension, and `get_dimension()` returns it.
- [ ] A page lists, for each supported Bedrock model, the value the three size settings must have.
- [ ] A batch is embedded with a bounded number of calls in flight, inside the existing time limits.
- [ ] No call runs on the event loop.
- [ ] `test_prd262_botocore_services.py` passes unchanged.

#### US-502: Sizes that agree, or no start · M
**Description:** As an operator, I want a mismatch between my embedding model and my index found at start-up, so that I do not find it in a failed write.

**Acceptance Criteria:**
- [ ] With a real provider set, start-up fails when the model's dimension is smaller than `S3_VECTORS_DIMENSION` or `FIELD_EMBEDDING_DIM`, and the message names the settings, both numbers and the value to set.
- [ ] A model that produces more dimensions than the index still starts, and is cut to fit as today.
- [ ] The current default configuration starts unchanged. A test pins it.

#### US-503: No stand-in vectors in a real index · S
**Description:** As an operator, I want embedding to fail when my provider is broken, so that my index is never filled with vectors that mean nothing.

**Acceptance Criteria:**
- [ ] With a provider set and unusable, an embedding call raises. It does not return the deterministic stand-in.
- [ ] With no provider set at all, the stand-in behaves as today.
- [ ] The pull request says what `test_p2w0_embeddings_fail_loud.py` already covered and what this adds.

## 7. Functional requirements

- FR-1: With `AWS_USE_WORKLOAD_IDENTITY` on, no client may be given a key pair, and object storage must count as configured.
- FR-2: The install must refuse to start when that setting and a key pair are both set.
- FR-3: With the setting on, start-up must prove the documents bucket can be reached, and fail clearly when it cannot.
- FR-4: A Bedrock model must be usable with no key when the setting is on.
- FR-5: With `S3_VECTORS_CREATE_IF_MISSING` off, core must never create a vector bucket or index.
- FR-6: Every Redis client in the orchestrator must be built by one function, and `rediss://` must give a TLS connection with the certificate checked.
- FR-7: Every connection to the application's database, in the orchestrator, in Alembic and in the worker, must be secured by the same two settings.
- FR-8: With object storage configured, an upload's original must be stored there and its row must say so.
- FR-9: With object storage configured, the API must keep nothing on local disk that has to survive a restart.
- FR-10: `EMBEDDING_PROVIDER=bedrock` must produce embeddings through `bedrock-runtime`.
- FR-11: An install must refuse to start when its embedding model produces fewer dimensions than an index expects.
- FR-12: A provider that is set and unusable must raise, never fall back to stand-in vectors.
- FR-13: With none of the new settings set, the `local` edition must behave as it does today.
- FR-14: With `AWS_USE_WORKLOAD_IDENTITY` on, an agent whose provider is Bedrock, under either stored name, must be called on Bedrock on every path, with the workspace's own Bedrock key found under either name. A call on the install's role must go to `AWS_REGION`; a configuration with its own key must keep the region it names, and use `AWS_REGION` when it names none. A Bedrock tier with no key must use the role, never another provider, and a Bedrock agent with no usable key must fail with a clear error rather than run on the role or on another provider.

## 8. Decisions for Gerard

Each has a recommendation, which stands unless he changes it.

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D-1 | Use the pod's role when asked, or whenever no key is set | When asked: an explicit setting | The gate excludes the SDK's own chain on purpose, and a stray profile on a laptop should not switch storage on |
| D-2 | Who supplies the database's CA | The operator, through the chart | Core stays free of any one cloud's files, and a bundle baked into an image goes stale |
| D-3 | Which installs store uploads in object storage | Every install where object storage is configured, the hosted one included | Today the hosted service's main upload route also writes to the pod's disk. This changes where its new uploads live. |
| D-4 | The first Bedrock embedding model | Amazon Titan Text Embeddings V2 | It is Amazon's own, so an AWS install adds no second supplier |
| D-5 | A model too small for the index | Refuse to start | The alternative is a failed write some time later |
| D-6 | Whether the `enterprise` edition may run without checking the database's certificate | No: it refuses to start unless `DATABASE_SSLMODE` is `verify-full` | AGENTS.md says TLS verification stays on. Today's unchecked `require` is kept only so that existing installs keep starting. |

## 9. Technical considerations

- **House rules that apply:** settings only through `config.py`, and each new name in `config-surface.json`; replace, don't shim; no blocking work on the event loop; pull requests under about 400 lines; signed-off commits.
- **Tests that pin today's shape and must be changed on purpose:** `test_prd233_s4_storage_factory.py` (`test_is_storage_configured_shapes`), `test_redis_client_timeouts.py`, `test_prd262_botocore_services.py`, `test_1170_bedrock_converse.py`, `test_p2w0_embeddings_fail_loud.py`.
- **Ask first.** US-404 rewrites existing rows, and D-3 changes where the hosted service keeps new uploads. AGENTS.md asks for an issue before a change to existing data, so Wave 4 opens with one.
- **A new dependency is not expected.** `boto3`, `redis` and both PostgreSQL drivers are already used.
- **PRD-277.** The `enterprise` edition is where these settings are first switched on together. Nothing here depends on its sign-in.
- **PRD-280.** The chart's service accounts are split there. This PRD adds only the CA mount and the volume switch.
- **The worker's database connection** is removed altogether by PRD-280. US-303 secures it until then.

## 10. Success metrics

- An install starts and serves chat, an upload, a search and a download with no cloud key in any secret.
- A packet capture is not needed to answer "is it encrypted": one setting each says so, and a test proves every path uses it.
- The API's pod can be deleted and replaced on another node with no file lost.
- A document is indexed with no request leaving the cloud account.

## 11. Risks

| Risk | What limits it |
|---|---|
| A connection path is missed, and one query stays unencrypted | The tests in US-202 and US-302 fail on any client or connection built outside the one function |
| The hosted service's uploads change home | D-3 asks first. US-404's job moves what exists, and can be run again. |
| The start-up check blocks a start during a short outage of object storage | A short time limit, and a clear message. The alternative is failing later, per request. |
| A Bedrock model that embeds one text at a time is slow for a large document | Bounded calls in flight, inside the existing limits. Measured before the default is chosen. |
| The size check stops an install that used to start | It applies only when the model is smaller than the index, which could not have worked |

## 12. Open questions

1. **What else, above the Bedrock client, needs a key to be present?** US-103 finds out.
2. **Does every caller reach embeddings through `EmbeddingManager`?** `services/ai_providers.py:241-245` reads the provider setting on its own **[T]**.
3. **How long does moving an install's existing files take,** and does it need a quiet period?
4. **Which Bedrock embedding models are offered in the regions the first installs will use?**
