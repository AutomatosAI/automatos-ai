# Automatos AI — Quick Start (local edition)

Clone the repo and bring up the full stack with Docker. The local edition runs
with **no login** and a single default workspace — no Clerk, no cloud accounts.
This is the short path; the full reference (every service, every dial,
troubleshooting, how the editions relate) is
[docs/getting-started/self-hosting.md](docs/getting-started/self-hosting.md).

```bash
git clone --depth 1 https://github.com/AutomatosAI/automatos-ai.git   # the current commit only: under 80 MB, not 700 MB of history
cd automatos-ai
```

You need Docker Engine 25 or later with Docker Compose 2.24.4 or later (a
current Docker Desktop on macOS and Windows); `make up` checks both first and
stops, naming that floor, on an older Docker.

**On Windows, or setting up with an AI coding agent?** Use [SETUP.md](SETUP.md):
the step-by-step install for macOS, Linux and Windows (WSL2), with a check
after each step.

## 1. The three required secrets

Compose refuses to start until these three are set (they have no built-in
defaults, on purpose — a public image must not ship with known passwords):

- `POSTGRES_PASSWORD` — the Postgres password
- `REDIS_PASSWORD` — the Redis password
- `API_KEY` — the backend's own API key

`make up` writes them for you: its first step, `make env`, creates `.env` from
`.env.example` and gives each one a random value (`openssl rand -hex 24`) while
it is empty or still the `CHANGE_ME` placeholder, before the first start only.
`git clone` and `make up` (step 2) are the whole install. To choose your own
values, run `make env` (or `cp .env.example .env`) and set them in `.env` before
the first `make up`.

The stack never runs on the `CHANGE_ME` placeholders: the backend refuses to
start on one, and `make env` stops on an install that already started with one.
Both say how to rotate it.

That is the whole requirement — everything else (edition, workspace id, storage,
ports) already has a working default in the compose file and `envs/*.defaults`.

### Optional: one LLM key for AI features (bring your own key)

The platform boots and serves without any LLM key, but agents, chat, and
embeddings need one to actually think.

**Recommended: OpenRouter.** One key, 400+ models, and — since PRD-240 — web
search for every model your agents run on. It is already the default route
(Auto starts on `google/gemini-2.5-flash` via OpenRouter). Get a key at
openrouter.ai/keys and add it under **Settings → API Keys**. Add **NVIDIA**
beside it for free open models (Kimi, DeepSeek, Nemotron — see below).

**Every provider is added in the app** — **Settings → API Keys** lists OpenAI,
Anthropic, OpenRouter, NVIDIA, DeepSeek, Google, Grok / xAI, Cohere, Azure
OpenAI, AWS Bedrock and HuggingFace, and validates the key on save. Only two
keys ever need to be in `.env`, and only for the Code Canvas worker, which
reads the environment rather than Settings:

```bash
OPENAI_API_KEY=sk-...          # optional
ANTHROPIC_API_KEY=sk-ant-...   # optional — the Canvas's Auto session engine
```

(Until a key is stored the chat page shows *"Add an LLM key to bring Auto to
life"*.)

**About the NVIDIA key.** NVIDIA's hosted endpoint is a trial: its terms allow
internal testing and evaluation, not production, and no personal, financial or
health data (NVIDIA API Trial Terms §1.2, §1.4, §4.3); the free tier allows about
40 requests per minute per key. The key is your own agreement with NVIDIA —
Automatos only routes to it. A model run on NVIDIA is recorded at zero cost and
is never silently rerouted to a paid provider when the limit is hit. In
**Marketplace → LLMs** open the *NVIDIA* tab, press *Sync NVIDIA* once, and add
the models you want — "Kimi K3 · NVIDIA" is a different route from
"Kimi K3 · OpenRouter" and installs with its own (zero) price. Then pick the
route in **Settings → Orchestrator** (or on any agent): provider *NVIDIA*, model
*Kimi K3*. (PRD-236)

**Skills.** A fresh install has none. **Marketplace → Capabilities → Import from
GitHub**, then *Use baseline repo* to import the free library at
`https://github.com/AutomatosAI/automatos-skills`.

**Your files.** Everything your agents write lives in one folder on your
machine: `AUTOMATOS_WORKSPACE_DIR` in `.env` (default `./workspaces` next to
`docker-compose.yml`), mounted as the workspace root — you see `artifacts/`,
`reports/`, `sessions/`… directly, no workspace-id folder. Put it next to your
projects folder and the Deliverables Explorer, the chat's Code mode and your
Claude Code sessions all share the one place:

```
LOCAL_PROJECTS_DIR=/Users/you/Development
AUTOMATOS_WORKSPACE_DIR=/Users/you/Development/deliverables
```

`make up` moves an older `./workspaces/<workspace id>/` layout up one level
for you, once.

## 2. Start the platform

```bash
make up
```

`make up` downloads the images CI builds, scans and publishes for your
checkout's commit (`ghcr.io/automatosai/automatos-api`, `-frontend`,
`-workspace-worker`; multi-arch and signed) instead of building them, then
starts the stack. It says which tag it picked and why (`sha-<7>` of your commit,
or `edge` when that is a rebuild of it), and writes it to `.env` as
`AUTOMATOS_IMAGE_TAG=…`, so later `docker compose` commands run the same images.
On a checkout with your own changes it builds only the images whose folder you
changed and pulls the rest.

The first run downloads about 1.3 GB (the whole stack, compressed, amd64;
1.28 GB on arm64): about a minute at 200 Mbit/s, three to four at 50 Mbit/s,
plus unpacking and the first boot. CI times `make up` to a working UI on every
main publish against a 150-second bar. Updates download only the layers that
changed.

**OS security fixes.** Every Monday CI rebuilds the images with fresh OS
packages and moves `edge` to the rebuild once it boots. A checkout keeps the OS
packages its `sha-<7>` was built with; `git pull && make up` picks up the
weekly fixes. `AUTOMATOS_IMAGE_TAG=edge make up` follows `edge` and pulls on
every call; `AUTOMATOS_IMAGE_TAG=sha-1bc4c07 make up` pins one build.

The published web app calls the API at `http://localhost:8000` and MinIO at
`http://localhost:9000`, fixed when it was built. If `.env` changes those
(`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL`, `NEXT_PUBLIC_MEDIA_ORIGINS`),
`make up` builds the web app on your machine instead, a few minutes, and says
why. `make build` builds all three images from your checkout; `make dev` is for
editing the code with hot reload. `make status` shows what is running and what
it costs; `make down` stops it; `make clean` reclaims space at any time (it
never touches your data). Plain `docker compose up` works too, once `.env` has
the three secrets (`make env`): it runs the tag in `.env`, or `edge`.

The first boot builds the database schema, runs the seeds and then serves.
`http://localhost:8000/health` answers once the API's core is up (the schema, the
seeds and the database check); `http://localhost:8000/health/ready` returns 503
while the rest of the boot (the starters, the integrations catalogue, the scheduler
and channels) runs in the background, and 200 once the instance is fully up.

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
- **Web access for your agents (PRD-240).** Reading a page works out of the
  box — every agent has `platform_web_fetch`, no key, no app to connect, nothing to
  assign. Private and internal addresses are always refused. Searching the
  web (`platform_web_search`) uses the first engine you have, in this order:
  1. **Your OpenRouter key** — search on any model, including free NVIDIA
     ones; the model on an OpenRouter route also searches inside its own
     turn and cites its sources. Billed per search on your OpenRouter account
     (about a cent).
  2. **Composio** — if `COMPOSIO_KEY` is set for integrations, Composio Search
     is used automatically. Free tier, no extra setup.
  3. **SearXNG** — a self-hosted search container, no key at all:
     `docker compose --profile search up -d`, then
     `SEARXNG_URL=http://searxng:8080` in `.env`.

  Switch it off with `WEB_ACCESS=off`; block hosts with
  `WEB_ACCESS_DENY=example.com,corp.internal`; pin an engine with
  `WEB_SEARCH_PROVIDER=openrouter|composio|searxng`. Restart the backend after
  changing any of them. Session-mode agents (below) already have the web
  through your own Claude subscription.

- **Local RAG on pgvector.** Documents are chunked, embedded, and searched in
  Postgres (`S3_VECTORS_ENABLED=false`) — no AWS needed.
- **MinIO object storage.** An S3-compatible store (ports 9000 / 9001) holds
  generated outputs so nothing is lost between runs.
- **Every provider, one router.** Settings → API Keys lists every registered
  provider (OpenAI, Anthropic, Google, OpenRouter, NVIDIA, DeepSeek, Azure
  OpenAI, AWS Bedrock, Grok / xAI, Cohere, HuggingFace) and validates a key on
  save. Marketplace → LLMs shows one card per *route* — the same model served
  by NVIDIA (free) and by OpenRouter (paid) is two cards with two prices — and
  an installed model is bound to the route you picked.
- **Cost analytics that tag everything.** Analytics → LLM & Costs records every
  call with the provider that served it and how it bills: paid API, free
  route, or Claude Code subscription; cost by provider, by lane (chat, board
  tickets, missions, heartbeats, retrieval), by agent and by route, with
  cache reads and failed calls.
- **The core stack:** Postgres (5432), Redis (6379), backend API (8000),
  frontend (3000), MinIO (9000/9001) and the **workspace worker** — the Code
  Canvas runtime that lets agents act on files on *your* machine. It keeps
  those files in `./workspaces` next to `docker-compose.yml`
  (`AUTOMATOS_WORKSPACE_DIR` in `.env` points it elsewhere); every tool call
  is confined to that directory and mutations still need your approval.
  The Canvas's *Auto session* engine (a headless Claude Agent SDK subprocess,
  billed to an API key) needs `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`
  in `.env` — it reads env only, not Settings → API Keys. The **Runtime
  Canvas** for session agents (below) needs no key at all: it is your own
  Claude Code, launched by the host on your machine. On a Linux host the files
  under `./workspaces` end up owned by uid 1000, the worker's user.
  `docker compose --profile all up` adds Adminer, a database GUI (8080).

## Optional: your own Claude Code as an agent (session mode)

Session mode lets an agent's tickets run as **your own Claude Code sessions on
your machine**, under your own Claude login — no API key, and inside
Anthropic's terms (the host runs your unmodified `claude`, never touches a
token, never uses `-p`). Auto assigns the ticket, the board tracks it, the
session's files become Deliverables, and you can open any session in the
Canvas and type alongside it.

1. On this machine, run `claude` once and log in — that login is what every
   session uses.
2. In `.env` set `CLI_RUNTIME_ENABLED=true` and
   `LOCAL_PROJECTS_DIR=/path/to/your/projects` (one parent folder for the
   repositories your agents may work in; `LOCAL_PROJECTS_MOUNT=rw` lets the
   Canvas editor save into it), then `make up`.
3. **Settings → Session mode → Get a pairing code**, then from the repository
   root run the command it shows:

   ```bash
   make cli-host PAIR=XXXX-XXXX      # pairs and serves; Ctrl-C after "paired"
   make cli-host-install             # installs the host as a login service
   ```

   The host shows as *connected* on that page and restarts itself whenever
   the backend's contract changes.
4. Give an agent the runtime **Claude Code session** (Agent → Model →
   Runtime; pick its workspace folder inside your projects folder) and
   assign it a ticket — or pick it in the chat to open the Runtime Canvas:
   the explorer on its folder and a terminal with its Claude Code session
   already running.

Tokens per session are recorded on the ticket and in Analytics as
*Claude Code · Subscription* at $0 — your plan pays, there is no dollar
figure to invent. The full reference is the
[self-hosting guide](docs/getting-started/self-hosting.md#session-mode--your-own-claude-code-sessions-managed-prd-234).

## What does *not* work out of the box

- **AI features need a model key** (above) — without one, agents and chat have
  no model to call. A free NVIDIA key covers chat and agents; document search
  needs an embedding provider too (an OpenAI or OpenRouter key, or the local
  HuggingFace provider under Settings → System Settings → Embeddings).
- **Composio-powered integrations** (Gmail, Slack, GitHub, Shopify and the rest
  of the third-party app catalogue) need your own Composio key. Put
  `COMPOSIO_KEY=…` in `.env` (free tier at app.composio.dev; env-only, there is
  no UI field for it — the line is in `.env.example`), then
  `docker compose up -d backend` to apply it. On that boot the backend syncs
  the catalogue itself; you can also pull it on demand from **Marketplace →
  Tools → Sync**, which fetches every toolkit and its actions. Without a key
  the Tools page and Marketplace → Tools say *"Integrations are disabled — no
  Composio API key is configured."*, Composio tools are not offered to agents,
  and the native platform tools keep working (PRD-233 S2). Web access is
  **not** one of these: `platform_web_fetch` needs no key, and `platform_web_search` works with
  your OpenRouter key or a SearXNG container as well as with Composio.
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
- **"Placeholder secrets: … Each still starts with CHANGE_ME":** `.env` still
  holds an `.env.example` placeholder on an install that has already started.
  Rotate it as the message shows ([guide §2](docs/getting-started/self-hosting.md#2-the-three-secrets-and-the-one-key)),
  or, with no data to keep, `docker compose down -v` and `make up` again.
- **"password authentication failed" after changing `POSTGRES_PASSWORD`:** the
  existing Postgres volume keeps the password it was initialised with — reset
  with `docker compose down -v` or `ALTER USER` it; see the guide.
- **Database GUI:** Adminer at http://localhost:8080 (with `--profile all`).
- **API reference:** http://localhost:8000/docs

All keys you add through Settings → API Keys are encrypted in the database and
available to the platform immediately. More in
[docs/getting-started/self-hosting.md](docs/getting-started/self-hosting.md).
