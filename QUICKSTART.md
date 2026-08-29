# Automatos AI — Quick Start (local edition)

Clone the repo and bring up the full stack with Docker. The local edition runs
with **no login** and a single default workspace — no Clerk, no cloud accounts.

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
ports) already has a working default baked into the compose file.

### Optional: one LLM key for AI features (bring your own key)

The platform boots and serves without any LLM key, but agents, chat, and
embeddings need one to actually think. Add **one** of these to `.env` when you
want AI features:

```bash
OPENAI_API_KEY=sk-...          # or
ANTHROPIC_API_KEY=sk-ant-...   # or an OpenRouter key for 300+ models
```

You can also add keys later through **Settings → Credentials** in the UI.

## 2. Start the platform

```bash
docker compose up
```

First run builds the images and initialises the database. When the backend is
healthy it serves `http://localhost:8000/health` (liveness) and
`http://localhost:8000/health/ready` (readiness — true once the local RAG
backend has constructed).

## 3. Open it

| Surface | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| MinIO console (object storage) | http://localhost:9001 |

## What you get in the local edition

- **No login.** `AUTH_EDITION=local` — you land straight in a single default
  workspace, no accounts to create.
- **Local RAG on pgvector.** Documents are chunked, embedded, and searched in
  Postgres (`S3_VECTORS_ENABLED=false`) — no AWS needed.
- **MinIO object storage.** An S3-compatible store (ports 9000 / 9001) holds
  generated outputs so nothing is lost between runs.
- **The core stack:** Postgres (5432), Redis (6379), backend API (8000),
  frontend (3000), and MinIO (9000/9001). Optional profiles add more:
  `docker compose --profile workers up` includes the workspace worker, and
  `--profile all` adds Gotenberg document rendering (3001) and Adminer (8080).

## What does *not* work out of the box

- **AI features need an LLM key** (above) — without one, agents and chat have no
  model to call.
- **Composio-powered integrations** (the 1,000+ external-tool marketplace) need a
  Composio API key in `.env` (`COMPOSIO_API_KEY=…`, free tier at app.composio.dev),
  then `docker compose up -d backend` to apply it, then run the catalogue sync
  once — the **Sync** action on the Tools page, or
  `curl -X POST http://localhost:8000/api/tools/sync`. Nothing syncs on boot.
  Without a key the integrations surface stays empty — an honest
  "integrations disabled" state and auto-sync-on-first-key are **PRD-233 S2**.

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

## Troubleshooting

- **Backend logs:** `docker compose logs -f backend`
- **Is it up?** `curl http://localhost:8000/health` (liveness) and
  `curl http://localhost:8000/health/ready` (readiness)
- **"POSTGRES_PASSWORD is required" / "REDIS_PASSWORD is required" /
  "API_KEY is required":** one of the three required secrets is missing from
  `.env` — see step 1.
- **Database GUI:** Adminer at http://localhost:8080 (with `--profile all`).
- **API reference:** http://localhost:8000/docs

All keys you add through Settings → Credentials are encrypted in the database and
available to the platform immediately.
