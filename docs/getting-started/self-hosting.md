# Self-hosting Automatos AI — the local edition

This is the reference for running the platform on your own machine with
`docker compose`. [QUICKSTART.md](../../QUICKSTART.md) is the short path; this
page is the long one — what runs, where the dials are, what the local edition
does and does not include, and how to update, reset and debug it.

Everything here is read from the committed sources: `docker-compose.yml`,
`envs/api.defaults`, `envs/frontend.defaults`, `.env.example`,
`docker-entrypoint.sh`, `orchestrator/config.py` and
`services/workspace-worker/worker_config.py`. When a statement and the file
disagree, the file wins — open an issue.

---

## 1. Prerequisites

| Need | Notes |
|---|---|
| Docker Desktop (macOS / Windows) or Docker Engine (Linux) | with the Compose v2 plugin — the `docker compose` subcommand. The compose file uses v2 syntax (`${VAR:?…}` required variables, optional `env_file` entries), so the old `docker-compose` v1 binary is not supported. |
| Disk | About 3.7 GB of images plus your data volumes. Backend 1.6 GB, workspace-worker 1.3 GB, Postgres 460 MB, frontend 240 MB, MinIO 175 MB, Redis 40 MB. |
| Git | to clone and to pull updates. |
| Free ports | 3000, 8000, 5432, 6379, 9000, 9001 by default — every one is overridable (§4). |

Nothing else. No cloud account, no identity provider, no AWS.

## 2. The three secrets and the one key

Compose refuses to start until three variables are set in a `.env` file next
to `docker-compose.yml`. They are declared as `${VAR:?message}`, so an unset or
empty value stops `docker compose up` with the message shown:

| Variable | Error when missing | Used by |
|---|---|---|
| `POSTGRES_PASSWORD` | `POSTGRES_PASSWORD is required - set in .env file` | postgres, backend, workspace-worker |
| `REDIS_PASSWORD` | `REDIS_PASSWORD is required - set in .env file` | redis, backend, workspace-worker |
| `API_KEY` | `API_KEY is required - set in .env file` | backend (its own API-key principal) |

Any non-empty values work on a private machine. They have no defaults on
purpose: a public repository must not ship known passwords.

```bash
git clone https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai
cp .env.example .env      # fill in section 1 (the three secrets)
docker compose up
```

**One LLM key** makes the agents think. The platform boots and serves without
one, but chat, agents and embeddings have no model to call until you set
exactly one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY`
in `.env` — or add a key later in the UI under **Settings → API Keys**
(providers: OpenAI, Anthropic, OpenRouter, Azure OpenAI; stored encrypted in
your local database). While no key is stored the chat page shows a banner,
*"Add an LLM key to bring Auto to life"*, linking to that tab. The key stored
there is also what embeddings use (`PLATFORM_KEY_WORKSPACE_ID` in
`envs/api.defaults` points the platform key slot at the local workspace), so
document uploads index with it; without any embedding provider an upload is
accepted but its processing ends `failed`.

## 3. What runs where

`docker compose up` starts the default profile. Nothing in it needs the
internet after the images are built, except the LLM provider you chose (and
Composio, if you add a key).

| Service | Container | Image / build | Host port (variable) | What it does |
|---|---|---|---|---|
| `postgres` | `automatos_postgres` | `pgvector/pgvector:pg16` | 5432 (`POSTGRES_PORT`) | Relational data and the pgvector chunk store the local RAG leg searches (`S3_VECTORS_ENABLED=false`). |
| `redis` | `automatos_redis` | `redis:7-alpine` | 6379 (`REDIS_PORT`) | Cache, pub/sub, queues. `FLUSHDB`, `FLUSHALL` and `DEBUG` are disabled; 256 MB `allkeys-lru`. |
| `minio` | `automatos_minio` | `minio/minio` | 9000 API (`MINIO_PORT`), 9001 console (`MINIO_CONSOLE_PORT`) | S3-compatible object store for documents, generated outputs, plugin packages and images. |
| `minio-init` | `automatos_minio_init` | `minio/mc` | — | One-shot: creates the documents bucket (`S3_DOCUMENTS_BUCKET`, default `automatos-ai`) and exits. |
| `backend` | `automatos_backend` | built from `./orchestrator` (`development` target) | 8000 (`API_PORT`) | The FastAPI API. Source is bind-mounted and served by `uvicorn --reload`. |
| `frontend` | `automatos_frontend` | built from `./frontend` (`development` target) | 3000 (`FRONTEND_PORT`) | The Next.js UI (`npm run dev`, source bind-mounted). Starts once the backend health check passes. |
| `workspace-worker` | `automatos_workspace_worker` | built from `./services/workspace-worker` | none — port 8081 stays inside the compose network | The Code Canvas runtime: the container in which agents read, write and run things, confined to your workspace directory (§5). |

`docker compose --profile all up` adds two more:

| Service | Host port (variable) | What it does |
|---|---|---|
| `adminer` | 8080 (`ADMINER_PORT`) | Database GUI, pre-pointed at `postgres`. |
| `gotenberg` | 3001 (`GOTENBERG_PORT`) | DOCX/XLSX → PDF conversion. The backend reaches it at `http://gotenberg:3000`; without the profile, document conversions that need it fail, everything else is unaffected. |

Named volumes: `automatos_postgres_data`, `automatos_redis_data`,
`automatos_minio_data`, `automatos_backend_logs`, and `backend_data`, which
holds the auto-generated credential-encryption key (`CREDENTIAL_KEY_FILE` in
`envs/api.defaults`). The workspace directory is a **bind mount** of a host
folder, not a volume (§5).

### Where configuration lives

| File | Role |
|---|---|
| `.env` (from `.env.example`) | Your secrets and the few values only you can choose. Compose reads it for variable substitution **only** — a variable reaches a container only if `docker-compose.yml` references it. |
| `envs/api.defaults` | The committed local topology for the backend: `AUTH_EDITION=local`, `DEFAULT_WORKSPACE_ID`, MinIO wiring, worker URL, observability off. No secrets. |
| `envs/frontend.defaults` | The frontend's local wiring: `NEXT_PUBLIC_AUTH_EDITION=local`, localhost API URLs. |
| `envs/api.local`, `envs/frontend.local` (gitignored, optional) | Personal overrides for **any** backend/frontend variable — the deep-override lane. Listed after the defaults, so they win over them; values set explicitly in the compose `environment:` block still win over both. |
| `orchestrator/config.py` | Code defaults for every remaining dial. The only module that reads the environment. |

You should not need to open the last three for a standard install.

## 4. First boot — what happens and how long it takes

1. **Image builds.** Backend (Python dependencies plus system libraries for
   OCR and PDF rendering), frontend (`npm install`), workspace-worker (Node,
   Claude Code CLI, Python tooling, Chromium). This is the slow part and only
   happens once per dependency change.
2. **Data services.** Postgres initialises an empty cluster with
   `POSTGRES_PASSWORD` — this password is baked into the volume at that moment
   (see §10). MinIO starts and `minio-init` creates the documents bucket.
3. **Backend entrypoint** (`docker-entrypoint.sh`), in order, each step
   fail-closed unless noted:
   - wait for Postgres, verify the connection;
   - **empty database?** run `python -m scripts.init_fresh_db` — builds the
     CI-proven schema (the SQLAlchemy models plus a tolerant replay of the
     migration history) and stamps Alembic at heads. Nothing is restored from
     a committed SQL snapshot; the generator is the fresh path;
   - `alembic upgrade heads` — a no-op on a fresh database, incremental on an
     existing one;
   - `python -m core.database.load_seed_data` — idempotent seeds: credential
     types, model catalogue, skills, personas, plugin categories, the
     marketplace catalogue (agents, packages) and, in the local edition, the
     first-run content of §8 (this step logs a warning and continues on
     failure);
   - ensure the local workspace (`DEFAULT_WORKSPACE_ID`, named *Local
     Workspace*) and the operator user exist;
   - start `uvicorn`. The application then runs its own boot stages,
     including the Composio catalogue bootstrap (§7).
4. **Frontend** starts after the backend's health check passes (the check
   allows 40 s of start-up and probes every 30 s).

Readiness, not liveness, is the signal that it is usable:

| URL | Meaning |
|---|---|
| `http://localhost:8000/health` | 200 as soon as the API process answers, with per-component status (database, config, resources). |
| `http://localhost:8000/health/ready` | 503 `{"status": "starting"}` until the full boot finished, then 200 `{"status": "ready"}`. |
| `http://localhost:8000/health/bootstrap` | The per-stage boot report with timings. |
| `http://localhost:8000/docs` | The OpenAPI reference. |

Then open **http://localhost:3000**. There is no login: the local edition has
exactly one operator and lands straight in the local workspace.

### The operator profile — "Hello, \<your name\>"

The anonymous local session is bound to one `users` row (email
`LOCAL_OPERATOR_EMAIL`, default `local@automatos.local`, set in
`envs/api.defaults`; seeded name *Local Operator*). That session is the
instance's `super_admin` — there is no platform above the operator of a
self-hosted instance, so system settings, credentials and admin analytics are
all yours.

Set your name under **Settings → Profile** (or the avatar in the header, which
links to `/settings/profile`): display name, username and avatar URL. The
email is read-only — it is the key the session resolves your row by. The name
you set is what Auto greets you by and what appears as the author of the
agents, Playbooks and Deliverables you create. Nothing here is an account:
no password, no login, no second user.

## 5. The workspace-worker and the host-access dial

The worker is the runtime behind **Code Canvas** — the surface where an agent
edits files and runs commands. In the local edition it runs in the default
profile and its files live **on your machine**:

- `AUTOMATOS_WORKSPACE_DIR` in `.env` (default `./workspaces`, next to
  `docker-compose.yml`, created on first boot, gitignored) is bind-mounted at
  `/workspaces` — read-write in the worker, read-only in the backend. It is
  the one dial that decides what the agents can touch: point it at a
  different folder to hand them that folder.
- Each workspace gets its own subdirectory, `/workspaces/<workspace_id>/`
  (the local workspace id is the `DEFAULT_WORKSPACE_ID` value in
  `envs/api.defaults`). Every Canvas tool call is re-bound to that root
  before it runs: `..` traversal, symlink escapes, null bytes and shell
  commands that reference paths outside it are rejected
  (`canvas_confinement.py`).
- Mutations still need you: a file edit or shell command pauses, the UI
  shows a permission request (a diff for edits), and the tool runs only after
  you approve (`canvas_approvals.py`). Read-only navigation is not gated.
  Local is not unguarded.
- **Model credential.** Canvas sessions run a headless Claude Agent SDK
  subprocess inside the worker. It reads only `ANTHROPIC_API_KEY` or
  `CLAUDE_CODE_OAUTH_TOKEN` (a Claude subscription token) from `.env` — keys
  stored through the UI, and OpenAI/OpenRouter keys, do not reach it. With
  neither set, starting a session fails immediately instead of idling.
- **Linux ownership note.** The worker process runs as uid 1000 (`worker`).
  Its entrypoint `chown -R`s the mounted directory to that uid whenever it is
  owned by anyone else, so on a Linux host the files under
  `AUTOMATOS_WORKSPACE_DIR` end up owned by uid 1000. If that is not your
  user, expect to need `sudo` to edit or delete them, or pre-create the
  directory owned by uid 1000.
- Limits and knobs: the service is capped at 2 CPUs and 2 GB;
  `WORKER_CONCURRENCY` (default 3) and `WORKSPACE_DEFAULT_QUOTA_GB` (default
  5) are overridable in `.env`. `WORKER_INTERNAL_TOKEN` is an optional shared
  secret between backend and worker; the worker port is never published, so
  it is off by default.

## 6. Object storage — MinIO, and the real-S3 option

The backend talks to **one** S3 code path (`orchestrator/core/storage/s3.py`
builds every client) and the compose stack points it at MinIO:

- `S3_ENDPOINT_URL=http://minio:9000`; path-style addressing; buckets other
  than the documents bucket (marketplace packages, recipe logs) self-create
  on first use against MinIO — never against AWS.
- The store's credentials are mapped through **`S3_*` names on purpose**:
  compose sets the backend's `AWS_ACCESS_KEY_ID` from `S3_ACCESS_KEY_ID`
  (default: `MINIO_ROOT_USER`, `minioadmin`), `AWS_SECRET_ACCESS_KEY` from
  `S3_SECRET_ACCESS_KEY` (default: `MINIO_ROOT_PASSWORD`, `minioadmin`) and
  `AWS_REGION` from `S3_REGION` (default `us-east-1`). A developer's `AWS_*`
  variables in `.env` or the shell are deliberately **not** read by the local
  store — a real AWS key there used to make every MinIO call fail with
  `InvalidAccessKeyId`.
- Links handed to the browser are presigned against
  `S3_PUBLIC_ENDPOINT_URL=http://localhost:9000` (`envs/api.defaults`). If you
  change `MINIO_PORT`, set `S3_PUBLIC_ENDPOINT_URL` to the new host port in
  `envs/api.local`, or those links will not resolve.
- The MinIO console is at **http://localhost:9001** (`MINIO_ROOT_USER` /
  `MINIO_ROOT_PASSWORD`). Change both in `.env` before exposing the machine
  to anyone else.

**Real S3 instead of MinIO.** In `.env` set `S3_ENDPOINT_URL=` (empty — compose
treats an empty value as "no custom endpoint") plus `S3_ACCESS_KEY_ID`,
`S3_SECRET_ACCESS_KEY` and `S3_REGION`; then create `envs/api.local` with
`S3_PUBLIC_ENDPOINT_URL=` and `S3_USE_PATH_STYLE=false` (the MinIO defaults live in
`envs/api.defaults`, and `api.local` wins). Buckets are not auto-created on AWS —
create them first. Your shell's `AWS_*` variables are never read by the store.

## 7. Bring your own Composio key (optional)

Third-party app integrations (Gmail, Slack, GitHub, Shopify and the rest of
the Composio catalogue) run through Composio. In the local edition that is a
**bring-your-own key, env-only** setting:

1. Put `COMPOSIO_API_KEY=…` in `.env` (free tier at app.composio.dev).
2. Apply it: `docker compose up -d backend` (the container must be recreated;
   the key is read from the environment, not from the UI).
3. On that boot, if the local catalogue (`composio_apps_cache`) is empty, the
   backend syncs the full Composio catalogue in a background thread and then
   re-binds the seeded marketplace agents to the apps they were designed for.
   A later boot with a populated catalogue only re-binds anything still
   unbound. The Tools page shows *"Integration catalogue is syncing"* while
   this runs.

Without a key the platform is honest rather than empty:

- the Tools page shows *"Integrations are disabled — no Composio API key is
  configured."* with the fix;
- Composio tools are not offered to agents at all (excluded from discovery),
  and a direct call returns an explicit `integrations_unavailable` error —
  never a silent success;
- native platform tools (the `platform_*` actions, Playbooks, documents,
  Deliverables, Code Canvas) keep working.

`GET /api/tools/integrations/status` reports `available`, `reason`,
`key_configured`, `apps_cached`, `last_sync` and `sync_status` — the same
predicate the router uses.

## 8. What a fresh instance contains

Every boot in the local edition runs an idempotent first-run seed
(`orchestrator/core/seeds/seed_local_first_run.py`), scoped to the local
workspace:

- **Auto**, the assistant, through the same seeder the hosted edition uses.
- **A starter roster** on the Agents page — *Researcher* (Research Analyst),
  *Writer* (Content Writer) and *Analyst* (Business Analyst). They use native
  platform tools only, so they run with nothing but your LLM key.
- **One Playbook** — *Two-minute brief*: Researcher → Writer → Analyst produce
  a reviewed one-page brief on a topic you choose. Run it from the Playbooks
  page and follow the execution log.
- **One Deliverable** — *Welcome to Automatos (local edition)*, under
  **Deliverables → Blogs**, authored by Auto. It repeats these steps inside
  the product.

The seed refreshes its own content across upgrades but never overwrites
something you edited, and does not resurrect something you deleted (the
workspace keeps a ledger of what was seeded). The marketplace catalogue
(agents, packages, personas, plugin categories) is seeded the same way and is
fully usable offline.

## 9. What is not in the local edition

One codebase, two shipped editions (§12). These are hosted-edition surfaces,
hidden locally by an explicit list (`frontend/lib/auth-edition.ts`,
`SAAS_ONLY_ROUTES`) — not by role, because the local operator is deliberately
`super_admin`:

| Not present locally | Why |
|---|---|
| Accounts, sign-in/up, password reset, SSO | No identity provider. `AUTH_EDITION=local` — one operator, one workspace. |
| Team, invitations, Workspace Admin, plan/trial pills, plugin moderation | Multi-tenant and commercial machinery of the hosted edition. |
| Settings → Webhooks, Channels, Widget SDK | Inbound webhooks and channel callbacks need a public URL; the widget embed needs the hosted loader. |
| The community hub | **Planned, not shipped.** The seeded catalogue is local and offline. The hub — sharing and pulling other people's agents, packages and Playbooks — is a network service: pulling will need no account and nothing will be paywalled; publishing will need a free account and pass moderation. |
| Durable memory (mem0) and field memory (Qdrant) | Neither ships in the default stack. `MEM0_API_URL` / `QDRANT_URL` in `envs/api.defaults` point at names that do not resolve, and the backend degrades cleanly. Point them at your own instances via `envs/api.local` and the features light up. |
| S3 Vectors RAG | Hosted-only; local RAG runs on pgvector (`S3_VECTORS_ENABLED=false`). Never enable it against MinIO. |
| Auto Live voice (Retell) | The switch and the Retell credentials live in the hosted edition's system settings; nothing in the local stack configures them. |
| Loki / Prometheus / log relay, the agent-opt worker | Hosted telemetry and prompt-optimisation services — silenced by empty URLs in `envs/api.defaults`. |

Product capability is not gated: every agent, tool, Playbook, Mission and
Deliverable feature that exists in the code runs locally.

## 10. Updating

```bash
git pull
docker compose up -d --build
```

Source directories are bind-mounted, so most code changes are picked up by
the running containers; `--build` matters when dependencies or Dockerfiles
changed. Database migrations run on every backend boot
(`alembic upgrade heads`, fail-closed — a failing migration stops the backend
rather than serving a half-built schema), and the seeds are idempotent.
Changing a value in `.env` or `envs/*` needs the container recreated
(`docker compose up -d`), not just restarted.

## 11. Stopping and resetting

```bash
docker compose down          # stop; data stays in the named volumes
docker compose down -v       # stop and delete every named volume
```

`down -v` removes Postgres, Redis, MinIO, backend logs and `backend_data`
(the credential-encryption key). It does **not** touch the bind-mounted
`AUTOMATOS_WORKSPACE_DIR` — delete that folder yourself if you want the
agents' files gone too. Removing `backend_data` alone, while keeping the
database, makes every API key stored through the UI undecryptable; keep the
two together.

## 12. Troubleshooting

**`POSTGRES_PASSWORD is required - set in .env file`** (or the same for
`REDIS_PASSWORD` / `API_KEY`). Compose refused to start: the variable is
missing or empty. `.env` must sit next to `docker-compose.yml`; a value of
nothing counts as unset.

**Logs.** `docker compose logs -f backend` (boot steps, migrations, seeds),
`docker compose logs -f workspace-worker`, `docker compose logs -f frontend`.
`docker compose ps` shows health.

**`password authentication failed for user "postgres"` after changing
`POSTGRES_PASSWORD`.** Postgres applies the password only when it initialises
an empty volume; an existing `automatos_postgres_data` keeps the old one, so
the backend and worker now present a password the database does not know.
Either reset the data (`docker compose down -v`) or change the stored password
to match `.env` — the command runs inside the container over its local
socket:

```bash
docker compose exec postgres psql -U postgres -d orchestrator_db \
  -c "ALTER USER postgres PASSWORD '<the value now in .env>';"
docker compose up -d backend workspace-worker
```

**Backend shows `unhealthy` or `starting` for a long time.** A first boot
builds the schema, replays migrations and seeds the catalogue before serving;
`/health/ready` stays 503 the whole time. Watch the backend log for the
`Starting Backend Application` banner; a red `❌` line names the step that
failed (a bad migration or a missing local workspace stops the boot on
purpose).

**Chat answers nothing; banner "Add an LLM key to bring Auto to life".** No
model key is stored. Settings → API Keys, or one of the keys in §2.

**Tools page: "Integrations are disabled".** No `COMPOSIO_API_KEY` (§7). Native
tools keep working.

**Code Canvas session fails as soon as it starts.** The worker has no model
credential: set `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in `.env`
and `docker compose up -d workspace-worker`.

**Port already in use.** Override the `*_PORT` variable in `.env` (§3). After
changing `MINIO_PORT`, also set `S3_PUBLIC_ENDPOINT_URL` (§6).

**Files under `./workspaces` owned by another user (Linux).** Expected — the
worker runs as uid 1000 (§5).

**Stored API keys stopped decrypting.** The `backend_data` volume (the
encryption key) was removed or replaced while the database was kept (§11).

**A saas-mode boot says `AUTH_EDITION=saas requires Clerk to be configured`.**
You flipped the edition without the identity provider. The local edition is
`AUTH_EDITION=local` (`envs/api.defaults`); leave it unless you are running
your own Clerk instance.

## 13. How the editions relate

There is **one codebase**. The edition is a runtime flag:

| | Local edition (this guide) | Hosted edition (automatos.app) | Enterprise |
|---|---|---|---|
| Flag | `AUTH_EDITION=local` (backend) / `NEXT_PUBLIC_AUTH_EDITION=local` (frontend), set by `envs/*.defaults` | `AUTH_EDITION=saas` — the code default; the boot guard then **requires** Clerk (`CLERK_JWKS_URL`, `CLERK_SECRET_KEY`) and fails fast without it, so a hosted boot can never fall through to the anonymous local identity | Parked. No `ee/` directory, no license keys, nothing built. |
| Identity | one operator, no login | Clerk accounts, workspaces, teams, plans | — |
| Configuration | `docker-compose.yml` + `envs/*.defaults` + `.env` | Railway sets each service's environment itself and **never reads** the compose file or `envs/*` — local defaults cost the hosted edition nothing | — |
| Storage / RAG | MinIO + pgvector | AWS S3 + S3 Vectors | — |
| Tools | native tools; Composio with your own key | native tools; Composio with the platform key | — |

The same CI gates both: every change is compose-only, fresh-clone-only, or
guarded by `AUTH_EDITION`, and the fresh-clone smoke lane boots this exact
stack from an empty checkout. Contributions land in every edition under the
repository's Apache-2.0 licence with a DCO sign-off — see
### Keeping it small

Rebuilding leaves the previous image untagged and grows the build cache, and
Docker keeps both indefinitely — that is what turns 3.7 GB of images into 12 GB
of disk. `make up` and `make dev` clean up after themselves, so this stays flat
on its own. To reclaim at any time:

```bash
make clean     # or: docker image prune -f && docker builder prune -f
```

`make clean` removes only untagged images and unused build cache. It never
prunes volumes: a stopped stack's database volume looks "unused" to Docker, so
`docker volume prune` would delete your data. Only `make reset` removes data,
and it asks first.

**Docker Desktop's own cache limit.** Docker garbage-collects the build cache
only once it passes `defaultKeepStorage` — 20 GB by default, so in practice it
never fires. If you build often, lower it in Docker Desktop → Settings →
Docker Engine:

```json
{ "builder": { "gc": { "enabled": true, "defaultKeepStorage": "3GB" } } }
```

### Working on the code instead of with it

The default stack runs the shipped production images — the Next.js standalone
build and a compiler-free Python runtime — because that is what a user installs.
For hot reload while editing:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

That swaps the frontend and backend to their development stages and mounts your
source. It is heavier on purpose (the dev server alone holds ~3 GB of RAM).

Two things are opt-in to keep the default install small:

```bash
# headless Chromium for the workspace_html_to_png tool (+1.2 GB)
docker compose build --build-arg INSTALL_BROWSER=true workspace-worker

# Leiden clustering for the Knowledge Graph (+660 MB; without it the graph
# still clusters, using networkx's Louvain — lower partition quality)
docker compose build --build-arg INSTALL_GRAPH_EXTRAS=true backend
```

[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Network exposure (`BIND_ADDRESS`)

The API (8000), Postgres, Redis and MinIO publish on `127.0.0.1` by default, so
they are reachable from this machine only. To reach them from another device on
your network, set `BIND_ADDRESS=0.0.0.0` in `.env` deliberately. In the local
edition an anonymous request is the operator, and with session mode (PRD-234)
the API can start Claude Code sessions that run commands on this machine — do
not expose it to a network you do not control.
