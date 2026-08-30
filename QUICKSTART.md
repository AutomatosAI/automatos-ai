# Automatos AI — Quick Start (local edition)

Clone the repo and bring up the full stack with Docker. The local edition runs
with **no login** and a single default workspace — no Clerk, no cloud accounts.
This is the short path; the full reference (every service, every dial,
troubleshooting, how the editions relate) is
[docs/getting-started/self-hosting.md](docs/getting-started/self-hosting.md).

## 1. Set the three required secrets

Compose refuses to start until these three are set (they have no built-in
defaults, on purpose — a public image must not ship with known passwords):

- `POSTGRES_PASSWORD` — the Postgres password
- `REDIS_PASSWORD` — the Redis password
- `API_KEY` — the backend's own API key

Copy the example env file and fill them in (any non-empty values work locally):

```bash
cp .env.example .env
# then edit .env and set:
#   POSTGRES_PASSWORD=<choose-any-value>
#   REDIS_PASSWORD=<choose-any-value>
#   API_KEY=<choose-any-value>
```

That is the whole requirement — everything else (edition, workspace id, storage,
ports) already has a working default in the compose file and `envs/*.defaults`.

### Optional: one LLM key for AI features (bring your own key)

The platform boots and serves without any LLM key, but agents, chat, and
embeddings need one to actually think. Add **one** of these to `.env` when you
want AI features:

```bash
OPENAI_API_KEY=sk-...          # or
ANTHROPIC_API_KEY=sk-ant-...   # or an OpenRouter key for 300+ models
```

You can also add keys later through **Settings → API Keys** in the UI (until
you do, the chat page shows *"Add an LLM key to bring Auto to life"*).

## 2. Start the platform

```bash
make up
```

That builds what changed, starts the stack, and clears the images and build
cache the rebuild superseded — Docker keeps those forever otherwise, which is
what quietly turns a 3.7 GB stack into 12 GB of disk. `make status` shows what
is running and what it costs; `make down` stops it; `make clean` reclaims space
at any time (it never touches your data). Plain `docker compose up` still works
if you prefer it.

First run builds the images, builds the database schema, runs the seeds and
then serves. `http://localhost:8000/health` answers as soon as the API process
is up; `http://localhost:8000/health/ready` returns 503 until the full boot has
finished and 200 once the instance is usable.

## 3. Open it

| Surface | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MinIO console (object storage) | http://localhost:9001 |

## What you get in the local edition

- **No login.** `AUTH_EDITION=local` — you land straight in a single default
  workspace, no accounts to create. The one operator is you: set your name
  under **Settings → Profile** and Auto greets you by it.
- **Something to run on the first boot.** The local edition seeds Auto, a
  starter roster (Researcher, Writer, Analyst), one Playbook — *Two-minute
  brief* — and a welcome Deliverable under **Deliverables → Blogs**. Run the
  Playbook from the Playbooks page with a topic of your own.
- **Local RAG on pgvector.** Documents are chunked, embedded, and searched in
  Postgres (`S3_VECTORS_ENABLED=false`) — no AWS needed.
- **MinIO object storage.** An S3-compatible store (ports 9000 / 9001) holds
  generated outputs so nothing is lost between runs.
- **The core stack:** Postgres (5432), Redis (6379), backend API (8000),
  frontend (3000), MinIO (9000/9001) and the **workspace worker** — the Code
  Canvas runtime that lets agents act on files on *your* machine. It keeps
  those files in `./workspaces` next to `docker-compose.yml`
  (`AUTOMATOS_WORKSPACE_DIR` in `.env` points it elsewhere); every tool call
  is confined to that directory and mutations still need your approval.
  Canvas sessions need `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in
  `.env` (the SDK subprocess reads env only, not Settings → API Keys). On a
  Linux host the files there end up owned by uid 1000, the worker's user.
  `docker compose --profile all up` adds Gotenberg document rendering (3001)
  and Adminer (8080).

## What does *not* work out of the box

- **AI features need an LLM key** (above) — without one, agents and chat have no
  model to call.
- **Composio-powered integrations** (Gmail, Slack, GitHub, Shopify and the rest
  of the third-party app catalogue) need your own Composio key in `.env`
  (`COMPOSIO_API_KEY=…`, free tier at app.composio.dev; env-only, there is no
  UI field), then `docker compose up -d backend` to apply it. On that boot the
  backend syncs the catalogue itself and re-binds the seeded agents to their
  apps. Without a key the Tools page says *"Integrations are disabled — no
  Composio API key is configured."*, Composio tools are not offered to agents,
  and the native platform tools keep working (PRD-233 S2).
- **Durable memory (mem0) and field memory (Qdrant)** are not in the default
  stack; the backend degrades cleanly without them.

## Optional: database GUI

```bash
docker compose --profile all up
```

Adds **Adminer** at http://localhost:8080.

## Stop / clean up

```bash
docker compose down            # stop
docker compose down -v         # stop and delete all data volumes
```

`down -v` does not remove the bind-mounted `./workspaces` folder — delete it
yourself if you want the agents' files gone too.

## Updating

```bash
git pull && docker compose up -d --build
```

Database migrations run on every backend boot.

## Troubleshooting

- **Backend logs:** `docker compose logs -f backend`
- **Is it up?** `curl http://localhost:8000/health` (liveness) and
  `curl http://localhost:8000/health/ready` (readiness)
- **"POSTGRES_PASSWORD is required" / "REDIS_PASSWORD is required" /
  "API_KEY is required":** one of the three required secrets is missing from
  `.env` — see step 1.
- **"password authentication failed" after changing `POSTGRES_PASSWORD`:** the
  existing Postgres volume keeps the password it was initialised with — reset
  with `docker compose down -v` or `ALTER USER` it; see the guide.
- **Database GUI:** Adminer at http://localhost:8080 (with `--profile all`).
- **API reference:** http://localhost:8000/docs

All keys you add through Settings → API Keys are encrypted in the database and
available to the platform immediately. More in
[docs/getting-started/self-hosting.md](docs/getting-started/self-hosting.md).
