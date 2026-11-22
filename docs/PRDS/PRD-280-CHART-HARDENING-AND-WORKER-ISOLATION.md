# PRD-280: A chart for a locked-down cluster — one identity per component, the worker on its own, rules between pods, and the memory store included

**Status:** Draft for Dave's review. Written 20 Nov 2026. No code has been written.
**Owner:** Gerard · **Reviewer:** Dave
**Grounded on:** `automatos-ai` `main @ 97c727136`, read on 20 Nov 2026.
**Labels:** **[T]** read in code at that commit; **[I]** inferred; **[G]** Gerard's decision; **[?]** not known.
**Relates to:** PRD-277 (the `enterprise` edition). PRD-278 (credentials from the pod's role, and the API without a volume). PRD-279 (how the chart is released). PRD-276 (durable background work, used in Wave 3).

**Decisions already taken [G] (Gerard, 19 Nov 2026):**

| | Decision |
|---|---|
| G1 | The first install for an organisation is a plain `helm install` of this chart. |
| G2 | Each workload has its own identity, and the worker, which runs agent code, has no cloud permissions at all. |
| G3 | Inside the cluster, nothing may talk to anything unless a written rule allows it. |

---

## 1. Introduction

The chart installs Automatos on Kubernetes, and it was written for someone trying the local edition. A regulated organisation's cluster asks more of it:

- its namespaces refuse a pod that starts as root or asks for extra privileges;
- its network denies every connection that no rule allows;
- its cloud gives each service account its own role, so a shared account means shared permissions.

Today the chart has one service account for everything, no network rules, a worker that starts as root, and no memory store. The worker is the one component that runs code an agent wrote, and it is also given the application's database address, which it uses for one statement.

This PRD makes the chart fit such a cluster without changing what a local user gets by default. It also closes one gap that only shows up there: in an install where models are reached with the API's own role, the worker has no way to reach a model at all.

## 2. What exists today

| Area | Today | Where | What it means |
|---|---|---|---|
| Service account | One, shared by the API, the web app and the worker. | `charts/automatos/templates/serviceaccount.yaml:1-9`; `api-deployment.yaml:28`; `frontend.yaml:19`; `worker.yaml:23` **[T]** | A role given to that account is given to all three. |
| Cluster token | Not mounted in the worker, the web app or the migration job. Mounted in the API by default. | `worker.yaml:24`; `frontend.yaml:20`; `migrate-job.yaml:31` **[T]** | The API holds a token for the cluster's own API. Nothing in the orchestrator uses a Kubernetes client **[T]**. |
| Migration job's account | The namespace's default account, unless the operator turns account creation off and names one. The chart's own account does not exist yet when the job runs. | `migrate-job.yaml:31-33` **[T]** | Workable. It should be stated, not accidental. |
| API, web app, migration job | Non-root, no privilege escalation, every capability dropped. No system-call filter. Root file system writable. | `api-deployment.yaml:29-33`, `:92-95`; `frontend.yaml:21-23`, `:48-51`; `migrate-job.yaml:35-37`, `:46-49` **[T]** | Each fails the "restricted" level on the missing filter. |
| Worker | No non-root setting. Starts as root, repairs the ownership of its volume, then drops to user 1000. Four capabilities added back for that. A system-call filter is set. | `worker.yaml:25-28`, `:84-104`; `services/workspace-worker/entrypoint.sh:68-93`; `Dockerfile:196` and no `USER` line **[T]** | "This passes PodSecurity baseline, not restricted", in the chart's own words. |
| Worker's capabilities | The template adds `DAC_OVERRIDE`. The README lists `DAC_READ_SEARCH`. | `worker.yaml:95-104`; `charts/automatos/README.md:21` **[T]** | The two disagree. |
| What the worker is given | The application's full database address, the Redis address, its internal token, and optionally a model key. | `worker.yaml:46-60` **[T]** | The worker holds a credential it needs for one statement. |
| What the worker does with the database | One statement: it updates the status of its own task. | `services/workspace-worker/task_executions_db.py:43-50`; `main.py:403-413` **[T]** | The whole database is reachable for the sake of one row. |
| What the worker does with Redis | Takes tasks from a queue, writes each task's status and result, publishes events, keeps a heartbeat. | `services/workspace-worker/main.py:195`, `:397-453` **[T]** | It shares one Redis, and one Redis user, with the API. |
| The worker's model credential | The coding agent it runs needs `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`. With neither, a canvas session fails with "no model credential". | `services/workspace-worker/worker_config.py:32-48`; `canvas_session_service.py:395-407` **[T]** | An install with no model key has no working canvas. |
| How the API reaches the worker | `WORKER_INTERNAL_URL`, set by the chart to the worker's service in the same namespace. Requests carry a shared token. | `orchestrator/config.py:868`; `templates/_helpers.tpl:155`; `services/workspace-worker/worker_http.py:121-125` **[T]** | The address is already a setting. |
| Placement | The API has node selector, toleration and affinity values. The worker, the web app and the migration job have none. | `api-deployment.yaml:107-116`; none in `worker.yaml`, `frontend.yaml`, `migrate-job.yaml` **[T]** | The worker cannot be sent to nodes of its own. |
| Namespace | Everything goes in the release's namespace. | all templates **[T]** | The worker cannot be given a different pod security level or its own network rules by namespace. |
| Network rules | None. Also no disruption budget and no namespace labels. | not found under `charts/` **[T]** | In a cluster that denies by default, nothing works until the operator writes the rules. |
| Volumes | Agent files: 10 GB, mountable by many pods, kept on uninstall. The API's data: 10 GB, one pod. Each takes a storage class. Neither can use a claim that already exists. | `workspaces-pvc.yaml`; `api-data-pvc.yaml`; `values.yaml:74-79`, `:156-160` **[T]** | An operator who supplies the storage cannot hand it in. |
| Memory store | Not in the chart. `QDRANT_URL` is an empty value, and the secret has no key for a Qdrant API key. | `values.yaml:45`; `_helpers.tpl:161-175` **[T]** | Agent memory needs a Qdrant the operator runs separately. |
| Memory store elsewhere | The compose file runs `qdrant/qdrant` pinned by digest, with one volume and no API key. Core's client sends an `api-key` header when `QDRANT_API_KEY` is set. | `docker-compose.yml:249-258`; `orchestrator/core/qdrant/client.py:121`; `config.py:1749-1750` **[T]** | The image and the client are known. Only the template is missing. |
| Chart checks | Lint, a render checked against one Kubernetes version, and unit tests. Installing into a real cluster is a script run by hand. | `.github/workflows/chart.yml:66-76` **[T]** | Nothing in CI proves the chart starts under a strict namespace. |

## 3. Goals

- Every component has its own service account, and none carries a cluster token it does not use.
- Every pod the chart creates passes the "restricted" pod security level, the worker included.
- The worker can be installed in its own namespace, on its own nodes, with its own secret.
- The worker is given no database address and no database password.
- An agent's coding session works in an install that has no model key, without giving the worker a cloud role.
- With one value, the chart installs the rules that allow exactly the connections it needs, for a cluster that denies the rest.
- The chart can install the memory store.
- With no new value set, a local user's install behaves as it does today, except that its pods are locked down further.

## 4. Non-goals

- **A second API replica.** PRD-267 and PRD-276.
- **Encrypting traffic between pods.** The chart serves plain HTTP inside the cluster, as now.
- **Creating namespaces or labelling them.** Those are the operator's, because the labels are what enforce the rules on the chart.
- **Cloud roles.** The chart gives each component a named account. Tying a role to it is the operator's.
- **More than one worker.** It stays one copy.
- **A highly available memory store.** One copy with its own disk, as in the compose file.

## 5. Design

### 5.1 One service account for each component

`api`, `frontend`, `worker`, `migrate` and, when it is enabled, `qdrant`. Each has a fixed, documented name and takes its own annotations, for clouds that tie a role to an account by annotation.

No pod mounts the cluster token. Nothing in the orchestrator calls the cluster's API **[T]**, so the API loses it too.

The migration job gets its own account, created as part of the same hook so it exists before the job runs.

### 5.2 Every pod at the "restricted" level

Each container gets what it lacks today: a system-call filter, and a read-only root file system.

A read-only root needs to know where each component writes. The API alone writes to a dozen temporary paths **[T]**. US-103 lists every writable path for every component before anything is switched, and each gets a scratch volume. Nothing that has to survive a restart may be on one (PRD-278 moves those to object storage).

### 5.3 The worker starts as itself

The worker starts as root for one reason: to repair the ownership of its volume when the storage handed it over owned by someone else **[T]**.

- **The image runs as user 1000 by default.**
- **The chart asks the storage to hand the volume over correctly,** with the pod's group setting, as it already requires for the API's volume (`values.yaml:73` **[T]**).
- **When it is not root, the worker does not try to repair anything.** It checks that it can write to its volume, and if it cannot, it stops with a message that says what the storage must do.
- **The four added capabilities go.** The worker then passes "restricted", and the disagreement between the README and the template disappears with them.

Running the image as root is still possible for the compose path, where the worker changes its own user id to match a folder on the developer's machine. The compose file says so explicitly. The chart no longer can.

### 5.4 The worker in a place of its own

Three values, each defaulting to today's behaviour:

| Value | Default | With it set |
|---|---|---|
| `worker.namespace` | The release's namespace | The worker's deployment, service, account, secret reference and volume claim are created there. The API's `WORKER_INTERNAL_URL` follows. |
| `worker.existingSecret` | The chart's `existingSecret` | The worker reads its own secret, which holds only what the worker needs. |
| Placement for every component | None | Node selector, tolerations, affinity and spread, for the worker, the web app, the migration job and the memory store as the API has today. |

**The agent-files volume across two namespaces.** A claim belongs to one namespace. So each side can be given a claim that already exists: one for the API, which mounts it read-only, and one for the worker. The operator binds both to the same file system. With one namespace the chart creates one claim, as today.

### 5.5 The worker without the database

The worker's only use of the database is to update its own task's status **[T]**. It already writes the same facts to Redis: a status, a result, and an event for each change **[T]**.

- **The API records task state.** It reads what the worker already publishes and writes the row itself.
- **A durable job checks for anything missed.** Published events can be lost, so a recurring job (PRD-276) compares running tasks with their status in Redis and corrects the row.
- **The worker's database code is deleted,** and the chart stops giving it the address. This is a replacement, not a second path.

US-301 confirms how task state is read today before this is built.

### 5.6 The worker's own Redis user

The worker still needs Redis, and today it uses the API's address and password. So it can read whatever the API keeps there.

The keys and channels the worker uses are listed and put under one prefix, so that a Redis user limited to that prefix can be made for it. PRD-278 already keeps a user name in a Redis address. Making the user is the operator's. Core's part is that one is enough.

### 5.7 A model for the worker's agent, with no key

In an install where the API reaches models with its own role, there is no model key to give the worker, and the worker must not get a role (G2). Its coding agent then has nothing to call.

- **The API relays.** One internal endpoint accepts the agent's model requests and forwards them to the install's configured provider, with the API's own credentials.
- **Only the worker can reach it,** by a network rule and the internal token.
- **Each session has its own short-lived token,** made by the API when the session starts and bound to that workspace, agent and session. The relay refuses a request without one, and a token is useless after its session ends.
- **The relay allows only the models the install allows,** and records each call's usage against the session's workspace, as any model call is recorded.
- **The worker points its agent at the relay** when it has no model key. With a key, it behaves as today.

Whether the agent's SDK can be pointed at such a relay, and in which request format, is checked against its documentation first **[I]** (US-401).

### 5.8 Rules between pods

`networkPolicy.enabled`, off by default. With it on, the chart creates a rule that denies everything for its own pods, then one rule for each connection it needs:

| From | To | Port |
|---|---|---|
| Whatever fronts the install (the operator names it) | Web app | 3000 |
| Whatever fronts the install | API | 8000 |
| Web app | API | 8000 |
| API | Worker | 8081 |
| API | Memory store | 6333 |
| API | The database and Redis, by the address ranges the operator gives | 5432, 6379 |
| API | Outside the cluster | 443 |
| Worker | Redis | 6379 |
| Worker | API, for the model relay | 8000 |
| Worker | Outside the cluster, only if the operator allows agent builds to fetch | 443 |
| Migration job | The database, by the same address ranges | 5432 |
| Every pod | Cluster DNS | 53 |
| Every pod, when telemetry is on | The collector | 4318 |

The web app and the memory store get no rule that lets them open a connection out. A test fails if a rule exists that this table does not list.

### 5.9 The memory store

`qdrant.enabled`, off by default. With it on:

- one copy, as a stateful set with its own disk, from the image the compose file already pins by digest;
- an API key, read from the secret under a new optional key `QDRANT_API_KEY`;
- `QDRANT_URL` set for the API without the operator writing it;
- the same "restricted" settings as every other pod, its own service account, and no way out.

Core's existing job already snapshots it to object storage daily **[T]** (`orchestrator/services/qdrant_snapshots.py`).

## 6. Waves and user stories

Each wave is one pull request or a short series. CI is the only gate.

### Wave 1: Identities and pod security

#### US-100: The chart installs under a strict namespace, in CI · M
**Description:** As a maintainer, I want CI to install the chart into a namespace that enforces "restricted", so that every later story in this PRD is proven by a real start and not only by a rendered file.

**Acceptance Criteria:**
- [ ] A CI lane, run only when the chart or an image's start-up changes, installs the chart into a namespace that enforces the "restricted" level.
- [ ] It enables every optional component the chart has at the time. Each later wave adds what it builds: the worker's namespace, the network rules, the memory store.
- [ ] Every pod becomes ready, and the lane fails if the namespace refuses one.
- [ ] The lane uses images built in the same run, not published ones.
- [ ] Until Wave 2 lands, the worker is the one pod the lane lets run at the "baseline" level, and the lane says so in its output.

#### US-101: A service account for each component · M
**Description:** As an operator, I want each component to have its own account, so that a cloud role I give the API is not also given to the worker.

**Acceptance Criteria:**
- [x] `api`, `frontend`, `worker` and `migrate` each have their own account, with a documented name and their own annotations.
- [x] No pod mounts the cluster token, the API's included.
- [ ] The migration job's account exists before the job runs, on a first install and on an upgrade.
- [x] `serviceAccount.create=false` with named accounts still works.
- [ ] The chart's unit tests cover each, and `helm lint` and the render check pass in CI.

#### US-102: A system-call filter on every pod · S
**Description:** As an operator, I want every pod to pass my cluster's strictest level, so that I do not have to weaken a namespace to install Automatos.

**Acceptance Criteria:**
- [ ] Every container has the default system-call filter, no privilege escalation and no capability.
- [x] A unit test renders the chart with every optional component on and fails if any container lacks one.

#### US-103: A table of everywhere a component writes · M · analysis only
**Description:** As a maintainer, I want every path each component writes to listed, so that a read-only root file system breaks nothing.

**Acceptance Criteria:**
- [ ] A table lists, for the API, the web app, the worker, the migration job and the memory store: each path written, by what, and whether it must survive a restart.
- [ ] Each row is marked as scratch or as needing a home.
- [ ] A row that needs a home and has none is listed for Gerard, not guessed.

#### US-104: A read-only root file system · M
**Description:** As an operator, I want no container able to change its own image at run time, so that what runs stays what was released.

**Acceptance Criteria:**
- [ ] Every container has a read-only root file system, with a scratch volume for each scratch row in US-103's table.
- [ ] Each component starts and passes its health check in CI's install test (US-100).
- [ ] No scratch volume holds anything US-103 marked as needing a home.

### Wave 2: The worker on its own

#### US-201: The worker starts as itself · M
**Description:** As an operator, I want the worker never to run as root, so that the one component that runs agent code has the least it can have.

**Acceptance Criteria:**
- [ ] The worker's image runs as user 1000 by default.
- [ ] Started as that user, the entrypoint repairs nothing. It checks the volume is writable and otherwise stops with a message that says what the storage must provide.
- [ ] The chart sets the worker non-root, with the pod's group setting on its volume, and adds no capability.
- [ ] The compose file states the user it starts the worker as, and the local development path works as today.
- [ ] The chart's README no longer lists capabilities for the worker.
- [ ] The worker's own tests pass, with new cases for both start modes.

#### US-202: The worker in its own namespace · M
**Description:** As an operator, I want to put the worker in a namespace of its own, so that I can give it its own security level and its own rules.

**Acceptance Criteria:**
- [ ] `worker.namespace` moves the worker's deployment, service and account, and the API's `WORKER_INTERNAL_URL` names the new address.
- [ ] `worker.existingSecret` names the worker's own secret, and the worker reads no other.
- [ ] The worker's internal token is a required key of that secret, not an optional one.
- [ ] Each side of the agent-files volume can be given a claim that already exists.
- [ ] With none of the three set, the rendered chart is the same as before this story. A test pins it.

#### US-203: Placement for every component · S
**Description:** As an operator, I want to choose the nodes each component runs on, so that the worker has nodes to itself.

**Acceptance Criteria:**
- [ ] The worker, the web app, the migration job and the memory store take a node selector, tolerations, affinity and spread, as the API does.
- [ ] Unit tests cover each.

### Wave 3: The worker without the database

#### US-301: How task state is read today · S · analysis only
**Description:** As a maintainer, I want to know everything that reads or writes a task's status, so that moving the write does not lose an update.

**Acceptance Criteria:**
- [ ] A page lists each place a task's status is written and read: the worker's database update, its Redis status and result, its events, and whatever in the API consumes them, with file and line.
- [ ] It says which of them the API already has and which the worker alone knows.

#### US-302: The API records task state · M
**Description:** As a security reviewer, I want the worker to hold no database password, so that code an agent runs cannot reach the system of record.

**Acceptance Criteria:**
- [ ] The API writes each task's status from what the worker publishes.
- [ ] A recurring durable job corrects any task whose row and Redis status disagree.
- [ ] `task_executions_db.py` and every call to it are deleted, and the worker has no database driver.
- [ ] The chart no longer gives the worker `DATABASE_URL`, and a test fails if it does.
- [ ] A task's status reaches the database when the worker's event is lost, shown by a test that drops it.
- [ ] The backend and worker test lanes pass in CI.

#### US-303: The worker's own Redis user · M
**Description:** As a security reviewer, I want the worker limited to its own keys in Redis, so that it cannot read what the API keeps there.

**Acceptance Criteria:**
- [ ] Every key and channel the worker reads or writes is under one documented prefix.
- [ ] A page gives the Redis permissions a user for the worker needs, and nothing wider works. A test with a Redis user limited to that prefix runs the worker's task cycle.
- [ ] The API reads and writes the same keys under the new names, and the old names are gone.
- [ ] A task in flight during an upgrade is handled: the pull request says how.

### Wave 4: A model for the worker's agent

#### US-401: Can the agent be pointed at a relay · S · analysis only
**Description:** As a maintainer, I want to know how the coding agent's SDK can be sent through a relay that holds the credentials, so that the design rests on its documentation and not on a guess.

**Acceptance Criteria:**
- [ ] A page quotes the SDK's documentation on using a gateway: the settings, the request format, and what the gateway must add.
- [ ] It says whether one relay can serve every provider the install might use, or only some.
- [ ] If it cannot be done, the page says so and section 5.7 is sent back to Gerard.

#### US-402: The relay · L
**Description:** As an operator of an install with no model key, I want the worker's agent to reach a model through the API, so that the coding canvas works and the worker still has no cloud permissions.

**Acceptance Criteria:**
- [ ] One internal route on the API forwards an agent's model request to the install's provider with the API's own credentials.
- [ ] It is not in the public route manifest and is not reachable through the install's public address. A test asserts both.
- [ ] It refuses a request without the worker's internal token.
- [ ] It refuses a model the install does not allow.
- [ ] Each call's usage is recorded against the workspace, as other model calls are.
- [ ] Streaming answers pass through as they arrive.

#### US-403: A token for each session · M
**Description:** As a security reviewer, I want each coding session to carry its own short-lived permission to use the relay, so that a task cannot spend as another workspace or after it has ended.

**Acceptance Criteria:**
- [ ] The API makes a token when a session starts, bound to its workspace, agent and session, with an expiry.
- [ ] The relay refuses a request with no token, an expired one, or one whose session has ended.
- [ ] Usage is recorded against the token's workspace, never one named in the request.
- [ ] Only a hash of the token is stored.

#### US-404: The worker uses the relay when it has no key · M
**Description:** As an operator, I want this to need no setting, so that an install with no model key simply works.

**Acceptance Criteria:**
- [ ] With no model key, the worker points its agent at the relay with the session's token.
- [ ] With a model key, the worker behaves as today. A test pins it.
- [ ] The "no model credential" failure remains only when there is neither a key nor a relay.
- [ ] A canvas session completes in CI against a stand-in provider.

### Wave 5: Rules between pods

#### US-501: The rules · M
**Description:** As an operator of a cluster that denies by default, I want the chart to bring the rules it needs, so that I do not have to work them out.

**Acceptance Criteria:**
- [ ] `networkPolicy.enabled` creates a deny-all rule for the chart's pods and one rule for each row of section 5.8.
- [ ] The operator supplies what fronts the install and the address ranges of the database and Redis, and the chart refuses to render with the value on and those missing.
- [ ] A unit test fails if a rendered rule has no row in section 5.8, or a row has no rule.
- [ ] The web app and the memory store have no rule that allows a connection out.
- [ ] With the value off, no rule is rendered.

### Wave 6: The memory store

#### US-601: Qdrant in the chart · M
**Description:** As an operator, I want the chart to install the memory store, so that agent memory works without a second install.

**Acceptance Criteria:**
- [ ] `qdrant.enabled` renders one stateful set with its own disk, from an image pinned by digest.
- [ ] It runs non-root with a read-only root file system and its own service account.
- [ ] `QDRANT_API_KEY` is an optional key in the secret, and the store requires it when it is set.
- [ ] The API's `QDRANT_URL` is set by the chart, and the chart refuses to render with both `qdrant.enabled` and a hand-written `QDRANT_URL`.
- [ ] The disk is kept when the chart is uninstalled.
- [ ] With the value off, nothing is rendered and `QDRANT_URL` is as today.

### Wave 7: What the chart guarantees

#### US-701: The chart's README says what it guarantees · S
**Description:** As a security reviewer, I want one page that says what each pod can and cannot do, so that I can review the chart without reading every template.

**Acceptance Criteria:**
- [ ] The README has a table: each component, its account, the user it runs as, what it mounts, what it may connect to, and which secrets it reads.
- [ ] A test fails if the table and the rendered chart disagree on the account, the user or the secret.

## 7. Functional requirements

- FR-1: Every component must have its own service account, and no pod may mount the cluster token.
- FR-2: Every pod the chart creates must pass the "restricted" pod security level.
- FR-3: Every container must have a read-only root file system.
- FR-4: The worker's image must run as a non-root user by default, and the chart must add no capability to it.
- FR-5: The worker must be installable in its own namespace, with its own secret and its own nodes.
- FR-6: The worker must be given no database address, and its code must contain no database connection.
- FR-7: Every key and channel the worker uses in Redis must be under one prefix.
- FR-8: With no model key, the worker's agent must reach a model through the API, with a token bound to its session.
- FR-9: The relay must be unreachable from outside the cluster, and must refuse a model the install does not allow.
- FR-10: With `networkPolicy.enabled`, the chart must deny every connection to and from its pods except the rows of section 5.8.
- FR-11: With `qdrant.enabled`, the chart must install the memory store and point the API at it.
- FR-12: With no new value set, an existing install's rendered chart must differ only by the changes in Wave 1 and US-201.

## 8. Decisions for Gerard

Each has a recommendation, which stands unless he changes it.

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D-1 | Whether the chart can still start the worker as root | No | The chart already requires storage that honours the pod's group for the API. One rule for both. |
| D-2 | How task state reaches the database once the worker cannot write it | The API records it from what the worker already publishes, with a durable job to catch a lost event | No new channel, and the worker's database code can be deleted outright |
| D-3 | How the worker's agent gets a model with no key | A relay on the API, with a token for each session | The worker keeps no cloud identity. The alternative is a long-lived model key in the one pod that runs agent code. |
| D-4 | Network rules on or off by default | Off | A cluster with no rule enforcement ignores them, and one that enforces them needs the operator's own values first |
| D-5 | Whether the chart or the operator creates namespaces | The operator | The namespace's labels are what hold the chart to its security level |
| D-6 | When the install test runs in CI | Only when the chart or an image's start-up changes | It is the slowest lane the chart has |

## 9. Technical considerations

- **House rules that apply:** settings only through `config.py` and `config-surface.json`; every route in `route-manifest.json` unless it is internal, in which case a test says so; replace, don't shim; no blocking work on the event loop; signed-off commits; pull requests under about 400 lines.
- **Ask first.** The relay touches how model calls are authorised and what spends money. AGENTS.md asks for an issue before changing either, so Wave 4 opens with one.
- **The relay is authenticated and authorised like any route.** A session's token resolves through the bearer registry to a request context for its workspace, so the route uses the existing request-context dependency. The worker's internal token is a second check, not the only one.
- **Two start modes are two supported environments, not a fallback.** The compose development path and storage that mounts as root both need the worker to start as root. The chart stops being a third.
- **The worker is its own package** with its own config module. Its changes do not go through `orchestrator/config.py`.
- **PRD-278.** A read-only root assumes nothing that must survive lives on the API's disk. Where PRD-278 has not landed, the API's data volume covers it.
- **PRD-277.** A token bound to a session follows the pattern that PRD sets for sessions and device tokens: random, stored hashed, checked through the bearer registry.
- **Tests that pin today's shape and must be changed on purpose:** the chart's `worker_frontend_test.yaml`, `api_test.yaml` and `migrate_job_test.yaml`; `test_prd262_smoke_worker_health.py`; `test_prd262_worker_starlette.py`.
- **The hosted service** runs the worker outside this chart, on storage that mounts as root **[T]** (`entrypoint.sh:66`). The image still starts as root there when told to. That path is unchanged.

## 10. Success metrics

- The chart installs into a namespace that enforces "restricted", in CI, with every component on.
- A person with a shell in the worker finds no database address, no cloud credential and no model key.
- A coding session completes in an install whose only model credential is the API's own role.
- An operator of a default-deny cluster installs with one extra value and no rules of their own.

## 11. Risks

| Risk | What limits it |
|---|---|
| A read-only root breaks a path nobody listed | US-103 lists them first, and US-100 starts every component in CI |
| Storage that cannot hand a volume over correctly stops the worker | The worker stops with a message that says what is needed. The chart already requires the same of the API's volume. |
| A lost event leaves a task's status wrong | The durable job corrects it, and a test drops an event on purpose |
| The relay becomes a way to spend another workspace's budget | A token for each session, usage recorded against the token's workspace, and only allowed models |
| Renaming the worker's Redis keys strands a task in flight | US-303 says how an upgrade handles it before it merges |
| The SDK cannot be pointed at a relay as assumed | US-401 checks first, and the design goes back to Gerard if not |

## 12. Open questions

1. **Does anything besides the worker's own update write a task's status to the database?** US-301.
2. **Can the memory store's image run as a non-root user with a read-only root as published,** or does the chart need a differently built one?
3. **Should the migration job also run under "restricted" in a namespace of its own?** It needs the database password and nothing else.
