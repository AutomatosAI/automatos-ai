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
| Docker Engine 25 or later with Docker Compose 2.24.4 or later: Docker Desktop (macOS / Windows) or Docker Engine (Linux) | Check with `docker version --format '{{.Server.Version}}'` and `docker compose version --short`. The compose file probes each service every second while it starts (healthcheck `start_interval`), which Compose refuses on an engine older than 25, and it uses Compose v2 syntax (`${VAR:?…}` required variables, optional `env_file` entries), so the old `docker-compose` v1 binary is not supported. `make up` checks both versions first and stops, naming this floor, on an older Docker. A Linux distribution's own `docker.io` package can be older: install from Docker's repository (docs.docker.com/engine/install). |
| Disk | The first `make up` downloads about 1.33 GB of images on amd64 and 1.28 GB on arm64, compressed: the API 521 MB, the workspace worker 507 MB and the web app 94 MB (CI's measurement, 16 Oct 2026), Postgres 159 MB, MinIO 64 MB and Redis 16 MB (the digests `docker-compose.yml` pins), less about 30 MB of base layers the API and the worker share. Docker unpacks what it downloads, so the images take more disk than that; `make status` shows the figure on your machine. `make up CORE=1` leaves out the worker and its image (§1b). Your data volumes come on top and grow with use. |
| Git | to clone and to pull updates. `git clone --depth 1` fetches only the current commit, under 80 MB, where the full history is about 700 MB; `git pull` updates a shallow clone as usual, and `git fetch --unshallow` fetches the history if you ever want it. |
| Free ports | 3000, 8000, 5432, 6379, 9000, 9001 by default — every one is overridable (§4). |
| Windows | The stack runs under Docker Desktop. `make` is not installed on Windows: §1b lists the commands behind every `make` target. The smooth path is everything inside WSL2: §1a has the recipe, and [SETUP.md](../../SETUP.md) walks through it step by step. Session mode runs inside WSL2, or natively on Windows 10 1809 or later (§1a). |

Nothing else. No cloud account, no identity provider, no AWS.

## 1a. Windows

There are two ways to run the local edition on Windows. The WSL2 recipe below
is the one [R-Capone](https://github.com/R-Capone) worked out and reported in
[issue #818](https://github.com/AutomatosAI/automatos-ai/issues/818), tested
on Windows 11 with WSL 2.7.8.

**Docker Desktop on Windows: the stack only.** Clone with Git for Windows as
usual. The repository's `.gitattributes` keeps every file the containers run
at LF line endings, even with Git's default `core.autocrlf=true`, and the
*Checkout line endings* CI lane checks that whenever one of them changes.
`make` is not installed, so run the commands in §1b from PowerShell. Session
mode runs natively too: see *Session mode on native Windows* below.

**Session mode on native Windows.** The session host also runs on Windows 10
version 1809 or later (Windows 11 included) without WSL2. Older Windows gets a
message pointing at WSL2. What it needs, and what differs from macOS and Linux:

- **Python 3.9 or later** from python.org, with the `py` launcher. Nothing else
  to install.
- **Claude Code installed natively** (`claude.exe`), or **GitHub Copilot CLI's
  standalone `copilot.exe`** (`winget install GitHub.Copilot`). A CLI installed
  with npm runs through a `.cmd` file, which the host refuses: `cmd.exe` would
  read the session's prompt as commands. **Codex** is not served on native
  Windows; for Codex agents, run the host inside WSL2.
- **No session sandbox.** Claude Code sandboxes its commands on macOS and Linux
  only. Start the host with `--no-session-sandbox` on a machine you accept as
  isolated. Without it, every session is refused with that reason.
- **The login task.** `--install` registers a Task Scheduler task, *Automatos
  CLI host*, that starts at your logon without elevation. It runs the host under
  a small supervisor that restarts it 15 s after a non-zero exit and logs to
  `%USERPROFILE%\.automatos\cli-host\host.log`. `--uninstall` first asks the
  host to stop and let go of its tickets. `--nudge` leaves a request the host
  picks up, since Windows has no SIGHUP.
- **Under the hood.** Sessions run on a ConPTY pseudo console, each in a job
  object that ends with it. Hooks reach the host over a named pipe that both ends
  authenticate, instead of `hooks.sock`. The Canvas terminal opens PowerShell.

```powershell
cd services\cli-host
py -m automatos_cli_host --allow C:/Users/you/automatos-deliverables --default-root C:/Users/you/automatos-deliverables --no-session-sandbox --pair XXXX-XXXX
py -m automatos_cli_host --install --allow C:/Users/you/automatos-deliverables --default-root C:/Users/you/automatos-deliverables --no-session-sandbox
```

The deliverables folder is the same one `AUTOMATOS_WORKSPACE_DIR` gives Docker
Desktop (the *Docker Desktop, native Windows* row in the table below).

**Everything inside WSL2: the stack and session mode.**

1. **Give Automatos a distro of its own.** In PowerShell:

   ```powershell
   wsl --install Ubuntu-24.04 --name automatos
   ```

   `--name` needs a recent WSL (`wsl --update`). Without it the distro is
   called `Ubuntu-24.04`: use that name wherever this section says
   `automatos`. A separate distro keeps systemd, Docker and the keep-alive
   task (step 7) away from any distro you already use.
2. **Turn on systemd.** Inside the distro, add this to `/etc/wsl.conf`:

   ```ini
   [boot]
   systemd=true
   ```

   Then run `wsl --shutdown` in PowerShell and reopen the distro with
   `wsl -d automatos`. *Check:* `systemctl --user status` prints a status, not
   an error.
3. **Install Docker Engine inside the distro.** Follow Docker's *Install
   Docker Engine on Ubuntu* guide in the distro, then:

   ```bash
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER     # then close the distro and open it again
   ```

   Docker Desktop with *Settings → Resources → WSL integration* turned on for
   this distro works too. With Docker Engine inside the distro, the containers
   stop whenever the distro stops (step 7).
4. **Clone inside the distro,** in your Linux home: `cd ~` and then
   `git clone --depth 1 https://github.com/AutomatosAI/automatos-ai.git`. Not under
   `/mnt/c`: file access across the Windows drive is slow, and session file
   paths stop lining up with Deliverables → Explorer.
5. **Create `.env`** with the three secrets (§2), then set the paths and ports
   below.
6. **Start it:** `sudo apt install -y make`, then `make up` (or the commands
   in §1b). Open http://localhost:3000 in your Windows browser: WSL forwards
   the ports.
7. **Keep the distro running.** WSL shuts a distro down a few seconds after its
   last `wsl.exe` session exits. That stops the stack (with Docker Engine in
   the distro) and the session host. A Windows Task Scheduler task that starts
   at logon and holds a session open keeps it up. Task Scheduler stops a task
   after 72 hours by default, so turn that limit off. In PowerShell:

   ```powershell
   $action   = New-ScheduledTaskAction -Execute "conhost.exe" -Argument "--headless wsl.exe -d automatos -u root -- sleep infinity"
   $trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
   $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
   Register-ScheduledTask -TaskName "Automatos WSL keep-alive" -Action $action -Trigger $trigger -Settings $settings
   ```

   Or in the Task Scheduler app: trigger *At log on*, action `conhost.exe`
   with the arguments `--headless wsl.exe -d automatos -u root -- sleep infinity`,
   and on the *Settings* tab clear *Stop the task if it runs longer than*. In
   the reporter's setup, the stack and the host were back and healthy about
   105 seconds after a Windows restart.
8. **Session mode** (optional): follow *Session mode* below, inside the
   distro. Two WSL details. Claude Code has no browser in WSL, so open the
   login link it prints in your Windows browser. And run
   `sudo loginctl enable-linger $USER` before `make cli-host-install`: without
   it, the `systemd --user` service (and so the host) runs only while you are
   logged in to the distro.

**Paths in `.env`.** `AUTOMATOS_WORKSPACE_DIR` (the deliverables root, §5) and
`LOCAL_PROJECTS_DIR` (your own projects, read-only) take an absolute path on
the machine that runs Docker:

| Where Docker runs | `AUTOMATOS_WORKSPACE_DIR` | `LOCAL_PROJECTS_DIR` |
|---|---|---|
| Inside the WSL2 distro | `/home/you/automatos-deliverables` | `/home/you/code` |
| Inside the WSL2 distro, projects on the Windows drive | `/home/you/automatos-deliverables` | `/mnt/c/Users/you/source` (works, but slower) |
| Docker Desktop, native Windows | `C:/Users/you/automatos-deliverables` | `C:/Users/you/source` |

On native Windows, write the paths with forward slashes. Under WSL2, keep
`AUTOMATOS_WORKSPACE_DIR` inside the distro, never under `/mnt/c`.

**Ports.** If something on the machine already holds a port (often a local
Postgres on 5432), `docker compose up` fails with `port is already allocated`.
Set the matching variable in `.env`, for example `POSTGRES_PORT=5433`. Only the
port on your machine changes; inside the compose network the containers still
reach Postgres on 5432. The other `*_PORT` variables are in §3.

## 1b. The commands behind `make`

`make` is a convenience; plain `docker compose` does the same work. Here is
what each target runs, from the repository root. On native Windows, run the
`docker` commands in PowerShell. `<deliverables>` is the absolute path in
`AUTOMATOS_WORKSPACE_DIR`, or the absolute path of the `workspaces` folder in
the repository when it is not set.

```bash
# make env: the Docker floor, then .env with the three secrets (section 2). make up, make build and
# make dev run it first.
sh scripts/stack_docker_floor.sh     # stops unless Docker Engine is 25 or later and Compose 2.24.4 or later
sh scripts/stack_env.sh              # creates .env from .env.example; fills the three while empty or CHANGE_ME,
                                     # before the first start only

# make up: the published images for this checkout, then start and tidy up
mkdir -p <deliverables>/projects     # PowerShell: New-Item -ItemType Directory -Force <deliverables>\projects
python3 scripts/stack_images.py up   # picks the tag (below), gets the images, writes AUTOMATOS_IMAGE_TAG to .env
docker compose up -d --no-build --remove-orphans
docker image prune -f
docker builder prune -f --filter until=168h

# make up CORE=1: the core five (postgres, redis, minio, backend, frontend), without the
# workspace worker. Every compose command reads docker-compose.core.yml too, through COMPOSE_FILE.
export COMPOSE_FILE=docker-compose.yml:docker-compose.core.yml   # PowerShell: $env:COMPOSE_FILE="docker-compose.yml;docker-compose.core.yml"
python3 scripts/stack_images.py up   # the same tag; gets the backend and frontend images only
docker compose -f docker-compose.yml rm --stop --force workspace-worker   # one an earlier make up started
docker compose up -d --no-build --remove-orphans
# The plain command, without make or the tag picker:
docker compose -f docker-compose.yml -f docker-compose.core.yml up -d

# make build: the three app images built from your checkout instead, as local-<sha7>
python3 scripts/stack_images.py build   # writes AUTOMATOS_IMAGE_TAG=local-<sha7> to .env
docker compose up -d postgres redis minio   # first, so they start while the images build
docker compose build backend frontend workspace-worker
docker compose up -d --no-build --remove-orphans

# make dev: for editing the code (the development stages, your source mounted, hot reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build --remove-orphans

# make down: stop, keep your data
docker compose down --remove-orphans

# make clean: dangling images and build cache unused for a week, never your data
docker image prune -f
docker builder prune -f --filter until=168h

# make status: what is running, and the disk it uses
docker compose ps
docker system df

# make logs: follow the logs; make logs s=backend for one service ($(s) is that name, or nothing for all)
docker compose logs -f --tail=200 $(s)

# make reset: DESTRUCTIVE, deletes the database and every named volume.
# make asks you to type 'reset' first; this command does not ask.
docker compose down -v --remove-orphans
```

`make env` first runs `scripts/stack_docker_floor.sh` (POSIX sh and the `docker`
CLI), so it is the first thing `make up`, `make build` and `make dev` check. It
stops when Docker does not answer, or when it is older than the floor in §1:
the engine's API older than 1.44 (Docker Engine 25), or Compose older than
2.24.4. Without it, check the two by hand:
`docker version --format '{{.Server.Version}}'` prints 25 or later, and
`docker compose version --short` prints 2.24.4 or later.

Then `make env` runs `scripts/stack_env.sh` (POSIX sh, `openssl` and the `docker`
CLI). Without it, the same by hand, from the repository root:

```bash
[ -f .env ] || { cp .env.example .env && chmod 600 .env; }
# Only before the first start: once the automatos_postgres_data volume exists, skip this
# (Postgres keeps its first password; section 2 says how to rotate one).
docker volume inspect automatos_postgres_data > /dev/null 2>&1 || for key in POSTGRES_PASSWORD REDIS_PASSWORD API_KEY; do
  if grep -qE "^${key}=(CHANGE_ME.*)?$" .env; then
    sed -i.bak "s|^${key}=.*|${key}=$(openssl rand -hex 24)|" .env && rm -f .env.bak
  fi
done
```

In PowerShell, without a POSIX shell: `Copy-Item .env.example .env`, then
replace each of the three `CHANGE_ME_…` values in `.env` with the output of
`$b = New-Object byte[] 24; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); -join ($b | ForEach-Object { $_.ToString('x2') })`,
one run per secret. A key missing from `.env` gets a line of its own.

`make build` starts `postgres`, `redis` and `minio` before it builds, so their
image pull happens first and their first-run initialisation and health checks
pass while the app images build; the backend still waits for all three to be
healthy. If the build fails, those three keep running until `make down`.

`make up` and `make build` do three more things. They move a
deliverables folder laid out before 2026-09-09 up one level, once (`make dev`
does this too); a fresh install never needs it. They print `make status`. And
they tell a running session host to pick up the new code
(`make cli-host-nudge`, below). `make up`, `make build` and `make dev` also pass
`AUTOMATOS_REVISION` (`git rev-parse HEAD`, with `-dirty` when `orchestrator`,
`frontend` or `services/workspace-worker` has uncommitted changes) to compose,
which stamps it on every image it builds as `org.opencontainers.image.revision`,
the label CI's images carry too. A build without it is stamped `unknown`.

**The core stack, five containers.** `make up CORE=1` runs postgres, redis,
minio, the backend and the frontend, and leaves out the workspace worker and
its image: `docker-compose.core.yml` puts the worker behind a profile of its
own, and `make` passes that file to every compose command through
`COMPOSE_FILE`. Without the worker, agents' workspace tools, the Deliverables
file browser and Code Canvas answer *Workspace worker not running*; the rest
works as before. A worker an earlier `make up` started is stopped and removed,
and its files stay in the `automatos_workspaces_data` volume. To start it
later, name it with the same two files:
`docker compose -f docker-compose.yml -f docker-compose.core.yml up -d workspace-worker`,
or run `make up` without `CORE=1`, which brings back all six. Plain
`docker compose up` always runs six, whatever your `.env` holds: no key turns the
worker off. `make build CORE=1` still builds all three images and then runs
five; `make dev` always runs six.

### The tag `make up` picks

The three app images run one tag, `AUTOMATOS_IMAGE_TAG`: `backend` is
`ghcr.io/automatosai/automatos-api`, `frontend` is
`ghcr.io/automatosai/automatos-frontend` and `workspace-worker` is
`ghcr.io/automatosai/automatos-workspace-worker`. `make up` prints the tag it
picked and why, by this rule (`scripts/stack_images.py`, Python 3.9 or later,
nothing to install):

1. **A tag you set** in the shell or on the command line
   (`AUTOMATOS_IMAGE_TAG=sha-1a2b3c4 make up`) is used as given. An official tag
   that is neither on this machine nor on GHCR stops `make up`; a `local-*` tag
   that is not on this machine is built. `AUTOMATOS_IMAGE_TAG=edge make up`
   follows `edge` and pulls it on every call. The line `make up` writes to `.env`
   is not a pin: it is rewritten on every run.
2. **On a clean checkout** (`git status --porcelain -- orchestrator frontend
   services/workspace-worker` prints nothing): `edge`, pulled first because
   compose never re-pulls a tag it already has, when `edge`'s images are a build
   of this very commit (their `org.opencontainers.image.revision` label is
   `git rev-parse HEAD`), such as the weekly rebuild with fresh OS packages.
   Otherwise `sha-<7>`, the first seven characters of `git rev-parse HEAD`, when
   CI has published it. A `sha-*` or `tree-*` image already on this machine
   counts as published, so a machine that has its images starts offline.
3. **Otherwise** `local-<sha7>`, or `local-<sha7>-dirty` when one of those
   folders has uncommitted changes. Each image whose folder is unchanged and
   whose `tree-<12>` CI has published (the first twelve characters of
   `git rev-parse HEAD:<folder>`; CI publishes one per tree) is pulled and
   re-tagged; the others are built here, and `make up` names them. That covers a
   `[skip ci]` merge and the few minutes before CI publishes a new commit.

The web app has its addresses built in: the published one is the local edition,
calling the API at `http://localhost:8000` and loading media from
`http://localhost:9000`. When `.env` sets `NEXT_PUBLIC_API_URL`,
`NEXT_PUBLIC_WS_URL`, `NEXT_PUBLIC_MEDIA_ORIGINS`, `NEXT_PUBLIC_AUTH_EDITION` or
`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` to anything else, `make up` builds the
frontend here (under step 3's tag, with the other two re-tagged) and says why.
A moved `API_PORT` or `MINIO_PORT` whose `NEXT_PUBLIC_*` values still point at
8000 or 9000 stops `make up` before it starts anything: set them as SETUP.md
step 3 shows.

Without `make` or Python, do the same by hand. For a clean checkout of a main
commit CI has published:

```bash
tag=sha-$(git rev-parse HEAD | cut -c1-7)
docker manifest inspect ghcr.io/automatosai/automatos-api:$tag > /dev/null && echo published
# make keeps exactly one such line at the end of .env, and replaces it each run
echo "AUTOMATOS_IMAGE_TAG=$tag" >> .env
docker compose up -d --no-build --remove-orphans
```

For `edge`, run `docker compose pull backend frontend workspace-worker` before
`up`. For a `local-<sha7>` tag, re-tag each published tree and build the rest:

```bash
docker pull ghcr.io/automatosai/automatos-api:tree-<first 12 of git rev-parse HEAD:orchestrator>
docker tag ghcr.io/automatosai/automatos-api:tree-<…> ghcr.io/automatosai/automatos-api:local-<sha7>
AUTOMATOS_IMAGE_TAG=local-<sha7> AUTOMATOS_REVISION=$(git rev-parse HEAD) docker compose build frontend
```

Plain `docker compose up` without `make` runs the tag in `.env`, or `edge` when
`.env` has none, and builds an image only when it cannot pull it. Running your
own unmerged code next to it, as a retest stack does:
`AUTOMATOS_IMAGE_TAG=local-<ref> docker compose up -d --build` builds the
production stages under a tag no pull ever replaces (no `local-*` tag is
published). Image names are shared by every stack on one Docker engine, so a
second stack sets its own `local-<ref>` in its own `.env`, not only in the shell.
A plain `docker compose up --build` with no tag at all names its build `:edge`,
which the next `make up` that picks `edge` pulls over.

The session-host targets run Python 3.9 or later, with nothing to install, on
macOS, Linux, WSL2 or Windows 10 1809 and later (there, type `py` for `python3`
and see *Session mode on native Windows* in §1a). Run them from
`services/cli-host`. Add
`--allow <your projects>` when `LOCAL_PROJECTS_DIR` is set:

```bash
cd services/cli-host

# make cli-host PAIR=XXXX-XXXX: pair once, then serve (Ctrl-C to stop)
python3 -m automatos_cli_host --allow <deliverables> --default-root <deliverables> --pair XXXX-XXXX

# make cli-host-install: the host as a login service
python3 -m automatos_cli_host --install --allow <deliverables> --default-root <deliverables>

python3 -m automatos_cli_host --service-status     # make cli-host-status
python3 -m automatos_cli_host --restart-service    # make cli-host-restart
python3 -m automatos_cli_host --uninstall          # make cli-host-uninstall

# make cli-host-nudge: a running host reloads on the new code
python3 -m automatos_cli_host --nudge
```

## 2. The three secrets and the one key

Compose refuses to start until three variables are set in a `.env` file next
to `docker-compose.yml`. They are declared as `${VAR:?message}`, so an unset or
empty value stops `docker compose up` with the message shown:

| Variable | Error when missing | Used by |
|---|---|---|
| `POSTGRES_PASSWORD` | `POSTGRES_PASSWORD is required - set in .env file` | postgres, backend, workspace-worker |
| `REDIS_PASSWORD` | `REDIS_PASSWORD is required - set in .env file` | redis, backend, workspace-worker |
| `API_KEY` | `API_KEY is required - set in .env file` | backend (its own API-key principal) |

**Channels (Settings → Channels).** Available in the local edition. Telegram runs in *polling* mode by default here, so a bot works — and answers agent questions from Telegram replies (PRD-225) — with no public URL. Slack, Discord and WhatsApp receive inbound messages only through webhooks, so their inbound needs a public URL in front of the orchestrator (a tunnel); outbound delivery works for every platform without one.

`make up` writes them for you. Its first step, `make env` (§1b), creates `.env`
from `.env.example` and gives each of the three a random value
(`openssl rand -hex 24`) while it is empty or still the `CHANGE_ME_…` placeholder
`.env.example` ships. It does that only before the first start, while the
`automatos_postgres_data` volume does not exist, because Postgres keeps the
password it was created with; it never changes any other value, nor one you
export in the shell. So the whole install is:

```bash
git clone --depth 1 https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai
make up
```

Your own values work too: set them in `.env` before the first `make up`. Keep
`.env` with your data: the database opens only with the `POSTGRES_PASSWORD` it
was created with.

**No install runs on a placeholder.** The values in `.env.example` are public, so
the backend refuses to start a command while any of the three starts with
`CHANGE_ME`, and `make env` stops on an install that has already started with
one. Both print the same message: which secrets, and how to rotate them. An
unset or empty value passes the backend's check (Helm and Railway set
`DATABASE_URL` and `REDIS_URL` instead of the two passwords); compose's own
`${VAR:?}` stops an empty one. To rotate on an install whose data you keep:

- `POSTGRES_PASSWORD`: change it in Postgres first, then put the same value in
  `.env`:

  ```bash
  new=$(openssl rand -hex 24)
  docker compose exec postgres psql -U postgres -d orchestrator_db -c "ALTER USER postgres PASSWORD '$new'"
  sed -i.bak "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$new|" .env && rm -f .env.bak
  ```

- `REDIS_PASSWORD` and `API_KEY`: a new value in `.env` alone
  (`openssl rand -hex 24`). Both are read at every start
  (`redis-server --requirepass`, and the backend's environment).

Then `make up` (or `docker compose up -d`) restarts what changed. An install with
no data to keep can start over instead: `docker compose down -v` deletes its
volumes, and the next `make up` generates all three.

**One model key** makes the agents think. The platform boots and serves without
one, but chat, agents and embeddings have no model to call until you set one of
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`
or `DEEPSEEK_API_KEY` in `.env` — or add a key later in the UI under
**Settings → API Keys**, which lists every provider in the registry (OpenAI,
Anthropic, Google, OpenRouter, NVIDIA, DeepSeek, Azure OpenAI, AWS Bedrock,
Grok / xAI, Cohere, HuggingFace), validates a key live on save, and stores it
encrypted in your local database. Models are installed *per route* in
**Marketplace → LLMs** (one tab per provider): the same model served by NVIDIA
and by OpenRouter is two cards with two prices, and an installed model is bound
to the route you chose — the runtime never reroutes a free route to a paid one.
NVIDIA's hosted endpoint (build.nvidia.com) serves its open models at no charge
under NVIDIA's own trial terms and a rate limit of about 40 requests a minute;
the key is your agreement with NVIDIA, and Automatos only routes to it
(details in [QUICKSTART.md](../../QUICKSTART.md)). While no key is stored the chat page shows a banner,
*"Add an LLM key to bring Auto to life"*, linking to that tab. The key stored
there is also what embeddings use (`PLATFORM_KEY_WORKSPACE_ID` in
`envs/api.defaults` points the platform key slot at the local workspace), so
document uploads index with it; without any embedding provider an upload is
accepted but its processing ends `failed`.

**Skills.** A fresh install ships the platform's 12 built-in skills (Auto's
platform skill and the Socials and design skills). For more, start with the free
baseline library of 140+ skills: in **Marketplace → Capabilities** press *Import from
GitHub* (the local operator is the instance's super admin, so the button is
there) and import `https://github.com/AutomatosAI/automatos-skills` — the
modal offers it as *Use baseline repo*. Every `SKILL.md` in the repo becomes a
marketplace skill you can enable per agent; re-import later to pick up updates.

## 3. What runs where

`make up` and plain `docker compose up` start the default stack: six
long-running containers, and `minio-volume-owner`, a one-shot that runs before
MinIO and exits. Nothing in it needs the internet after the images are pulled,
except the LLM provider you chose (and Composio, if you add a key).

| Service | Container | Image / build | Host port (variable) | What it does |
|---|---|---|---|---|
| `postgres` | `automatos_postgres` | `pgvector/pgvector:pg16` | 5432 (`POSTGRES_PORT`) | Relational data and the pgvector chunk store the local RAG leg searches (`S3_VECTORS_ENABLED=false`). |
| `redis` | `automatos_redis` | `redis:7-alpine` | 6379 (`REDIS_PORT`) | Cache, pub/sub, queues. `FLUSHDB`, `FLUSHALL` and `DEBUG` are disabled; 256 MB `allkeys-lru`. Compose sets the backend's `REDIS_URL` to this service, so a `REDIS_URL` for a hosted Redis in `orchestrator/.env` is not picked up inside the container. |
| `minio-volume-owner` | `automatos_minio_volume_owner` | `redis:7-alpine`, the same image as `redis` | none, and no network | One-shot, before MinIO starts: when root owns MinIO's volume, as on an install made while MinIO still ran as root, it re-owns the volume to uid 65532 once, says so in its log and exits; otherwise it exits at once. It runs as root with only the capabilities `chown` needs. MinIO starts only after it succeeds. |
| `minio` | `automatos_minio` | `cgr.dev/chainguard/minio` (Chainguard's maintained MinIO build) | 9000 API (`MINIO_PORT`), 9001 console (`MINIO_CONSOLE_PORT`) | S3-compatible object store for documents, generated outputs, plugin packages and images. Runs as the image's own user, uid 65532, never root. The backend creates the documents bucket (`S3_DOCUMENTS_BUCKET`, default `automatos-ai`) and the others the first time it uses them. |
| `backend` | `automatos_backend` | `ghcr.io/automatosai/automatos-api`, pulled: CI's build of `./orchestrator` (`production` target); `make build` builds it here | 8000 (`API_PORT`) | The FastAPI API, as CI builds and scans it: the code is inside the image and nothing reloads. `make dev` runs the `development` target instead, with the source bind-mounted and served by `uvicorn --reload`. |
| `frontend` | `automatos_frontend` | `ghcr.io/automatosai/automatos-frontend`, pulled: CI's build of `./frontend` (`production` target); built here when `.env` changes its `NEXT_PUBLIC_*` values (§1b) | 3000 (`FRONTEND_PORT`) | The Next.js UI (the standalone server). Starts once the backend health check passes. `make dev` runs the `development` target instead (`npm run dev`, source bind-mounted). |
| `workspace-worker` | `automatos_workspace_worker` | `ghcr.io/automatosai/automatos-workspace-worker`, pulled: CI's build of `./services/workspace-worker` | none — port 8081 stays inside the compose network | The Code Canvas runtime: the container in which agents read, write and run things, confined to your workspace directory (§5). |

**The core stack.** An override file runs the default stack without the
workspace worker, five containers instead of six (§1b, §5):

| Override | Command | Leaves out | Profile that brings it back |
|---|---|---|---|
| `docker-compose.core.yml` | `make up CORE=1`, or `docker compose -f docker-compose.yml -f docker-compose.core.yml up -d` | `workspace-worker` | `workspace`: name the service with the same two files (`... up -d workspace-worker`), or run `make up` without `CORE=1` |

**Optional profiles.** Each profile adds its services to the default stack:
`docker compose --profile <profile> up -d`, or name the service, as below. All
but `media` pull a published image.

| Profile | Service | Container | Image / build | Host port (variable) | What it does |
|---|---|---|---|---|---|
| `all` | `adminer` | `automatos_adminer` | `adminer` | 8080 (`ADMINER_PORT`) | Database GUI, pre-pointed at `postgres`. |
| `memory` | `qdrant` | `automatos_qdrant` | `qdrant/qdrant` | 6333 (`QDRANT_PORT`) | Durable and field memory. Also set `QDRANT_URL=http://qdrant:6333` in `envs/api.local`; without it the backend keeps memory off (§9). |
| `search` | `searxng` | `automatos_searxng` | `searxng/searxng` | 8888 (`SEARXNG_PORT`) | Web search for agents, with no key. Also set `SEARXNG_URL=http://searxng:8080` in `.env` ([§7a](#7a-web-access-for-agents-prd-240)). |
| `media` | `media-render` | `automatos_media_render` | built here from `./services/media-render`; nothing publishes it | none — port 8090 stays inside the compose network | Renders a Socials post (a template, your brand kit, the Kokoro voice and the music) into an MP4 or PNG: `docker compose --profile media up -d media-render`, which builds it the first time. See [Media rendering](#7b-media-rendering-the-media-profile). |

Every third-party image above is pinned by digest in `docker-compose.yml`, and
Dependabot moves the pins once a week. The CI test
`orchestrator/tests/test_prd262_self_host_docs.py` holds these three tables to
`docker compose config --services`, profile by profile and with the core
override.

Named volumes: `automatos_postgres_data`, `automatos_redis_data`,
`automatos_minio_data`, `automatos_backend_logs`, `automatos_workspaces_data`,
and `backend_data`, which holds the auto-generated credential-encryption key
(`CREDENTIAL_KEY_FILE` in `envs/api.defaults`), uploaded documents and the local
copy of generated documents (object storage keeps theirs). The `memory` profile
adds `automatos_qdrant_data`, the `search` profile `searxng_cache`. Your
workspace's directory is a **bind mount** of a host folder, not a volume (§5).
`automatos_workspaces_data` holds the files of every other workspace on the
stack (a second or a test workspace), so they survive recreating the
containers; `make reset` removes it with the rest.

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

1. **Images.** `make up` pulls the app images CI built, scanned and published
   for your commit (§1b). The whole default stack is about 1.33 GB compressed on
   amd64 and 1.28 GB on arm64 (measured 16 Oct 2026; the target is under 1 GB),
   so the download is the slow part: about a minute at 200 Mbit/s, three to four
   at 50 Mbit/s, plus unpacking. CI times `make up` to a working UI on every
   main publish against a 150-second bar. A later update downloads only the
   layers that changed. `make build` builds the images from your checkout
   instead (Python dependencies plus system libraries for OCR and PDF
   rendering, `npm ci`, the worker's Node and Claude Code CLI), which takes
   several minutes.
2. **Data services.** Postgres initialises an empty cluster with
   `POSTGRES_PASSWORD` — this password is baked into the volume at that moment
   (see §10). `minio-volume-owner` checks MinIO's volume and exits (§3), then
   MinIO starts; the backend creates the documents bucket the first time it
   stores a file.
   While a service starts, Docker probes its health every second
   (`start_interval: 1s` in `docker-compose.yml`, Docker Engine 25 or later), so
   a service that waits on another starts within a second or two of it being
   ready; CI holds Redis and MinIO to 2 s from start to healthy. Postgres's
   probe goes over TCP (`pg_isready -h 127.0.0.1`), so on a fresh volume it passes
   only once the real server runs, after initdb's temporary one has stopped.
3. **Backend entrypoint** (`docker-entrypoint.sh`), in order, each step
   fail-closed unless noted:
   - wait for Postgres, verify the connection;
   - **empty database?** run `python -m scripts.init_fresh_db` — builds the
     CI-proven schema (the SQLAlchemy models plus a tolerant replay of the
     migration history) and stamps Alembic at heads. Nothing is restored from
     a committed SQL snapshot; the generator is the fresh path. If that build
     is interrupted (Postgres restarts, the machine sleeps or shuts down), the
     boot stops, and the next boot runs the build again to finish it
     (`♻️ A previous fresh-database initialization was interrupted — resuming
     it`) — no reset needed;
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
4. **Frontend** starts after the backend's health check passes. The check
   probes every second while the backend starts, so the backend turns healthy
   about a second after it first answers `/health`, and compose starts the
   frontend about a second after that (CI holds this to 3 s). The check allows
   up to 300 s of start-up for a first boot (F228), then probes every 30 s.

**How long, measured.** On CI's runners on 16 Oct 2026, before the one-second
probes of step 2, the datastores turned healthy 5.5 to 6.1 s after they started,
and the backend's first boot on an empty database took 30 to 50 s from its start
to `/health` 200, 45 s at the median. A laptop can take longer: one Mac took
147 s (F228), which is why the frontend's wait allows 300 s. The targets are 2 s
for the datastores and 30 s for the backend's first boot.

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
  `docker-compose.yml`, created on first boot, gitignored) is bind-mounted as
  the local workspace's root, `/workspaces/<workspace_id>/` — read-write in
  the worker, read-only in the backend. On your machine the folder holds
  `artifacts/`, `reports/`, `content/`, `sessions/`… directly: there is no
  workspace-id folder on disk (before 2026-09-09 there was; `make up` moves
  that layout up one level, once). It is the one dial that decides what the
  agents can touch: point it at a different folder to hand them that folder.
  Recommended layout — the deliverables root beside your projects folder, so
  the Deliverables Explorer, the chat's Code mode and your Claude Code sessions
  share one place: `LOCAL_PROJECTS_DIR=/Users/you/Development`,
  `AUTOMATOS_WORKSPACE_DIR=/Users/you/Development/deliverables`. (With the
  deliverables root inside the projects folder, the Explorer also lists it
  under `projects/` — the same files twice, harmless.)
- Inside the containers each workspace is `/workspaces/<workspace_id>/`
  (the local workspace id is the `DEFAULT_WORKSPACE_ID` value in
  `envs/api.defaults`). Every Canvas tool call is re-bound to that root
  before it runs: `..` traversal, symlink escapes, null bytes and shell
  commands that reference paths outside it are rejected
  (`canvas_confinement.py`).
- Mutations still need you: a file edit or shell command pauses, the UI
  shows a permission request (a diff for edits), and the tool runs only after
  you approve (`canvas_approvals.py`). Read-only navigation is not gated.
  Local is not unguarded.
- **Model credential.** The Canvas's *Auto session* engine runs a headless
  Claude Agent SDK subprocess inside the worker, billed to an API key. It
  reads only `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` from `.env` —
  keys stored through the UI, and OpenAI/OpenRouter keys, do not reach it;
  with neither set the tab says so instead of idling. The **Runtime Canvas**
  for session agents (see *Session mode* below) is a different thing: it needs
  no key, because the terminal it shows is your own Claude Code, launched by
  the host on your machine.
- **Linux ownership note.** The worker process runs as uid 1000 (`worker`).
  Its entrypoint re-owns everything under `/workspaces` to that uid, except
  `projects/`, whenever the volume's root belongs to anyone else, and that
  includes your files under `AUTOMATOS_WORKSPACE_DIR`. On this stack it does not
  happen: a new `automatos_workspaces_data` volume takes the owner of the
  directory it is first mounted on, and both the worker's image and the
  backend's give `/workspaces` to uid 1000, so the volume is uid 1000 whichever
  starts first (the core stack, below, starts the backend without the worker).
  A volume created root-owned some other way would trigger it; then expect to
  need `sudo` to edit or delete those files, or pre-create the directory owned by
  uid 1000.
- **Running without it.** `make up CORE=1` (or
  `docker compose -f docker-compose.yml -f docker-compose.core.yml up -d`) runs
  the five other containers and not the worker (§1b). Code Canvas, the
  Deliverables file browser and agents' workspace tools then answer *Workspace
  worker not running*.
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

1. Put `COMPOSIO_KEY=…` in `.env` (free tier at app.composio.dev). That is the
   canonical name — it matches the Railway variable; `COMPOSIO_API_KEY` is
   accepted as an alias. `docker-compose.yml` forwards both.
2. Apply it: `docker compose up -d backend` (the container must be recreated;
   the key is read from the environment at process start, not from the UI).
3. On that boot, if the local catalogue (`composio_apps_cache`) is empty, the
   backend syncs the full Composio catalogue in a background thread and then
   re-binds the seeded marketplace agents to the apps they were designed for.
   A later boot with a populated catalogue only re-binds anything still
   unbound. The Tools page shows *"Integration catalogue is syncing"* while
   this runs.
4. To pull the catalogue on demand instead of waiting for a boot, press
   **Sync** on **Marketplace → Tools** (or `POST /api/tools/sync?sync_type=full`).
   It fetches every toolkit and its actions into the cache; a full catalogue is
   over 1,000 apps, so it takes a few minutes. Without a valid key the
   request returns 400 with the reason rather than reporting a sync of 0 apps.

Without a key the platform is honest rather than empty:

- the Tools page **and Marketplace → Tools** show *"Integrations are disabled —
  no Composio API key is configured."* with the fix;
- Composio tools are not offered to agents at all (excluded from discovery),
  and a direct call returns an explicit `integrations_unavailable` error —
  never a silent success;
- native platform tools (the `platform_*` actions, Playbooks, documents,
  Deliverables, Code Canvas) keep working.

`GET /api/tools/integrations/status` reports `available`, `reason`,
`key_configured`, `apps_cached`, `last_sync` and `sync_status` — the same
predicate the router uses.

## 7a. Web access for agents (PRD-240)

Agents can read and search the web as a **platform capability** — nothing in
the marketplace, nothing to connect, nothing to assign. Two platform actions
every agent has:

| Action | Needs | What it does |
|---|---|---|
| `platform_web_fetch` | nothing | GET a public URL from the backend container, TLS verified, redirects re-checked, body capped (`WEB_FETCH_MAX_BYTES`), HTML reduced to readable text. |
| `platform_web_search` | one engine (below) | `[{title, url, snippet}]` for a query, through the first engine configured. |

**Search engines, resolved in this order** (`WEB_SEARCH_PROVIDER=auto`):

1. **OpenRouter** — any OpenRouter key the platform can see (env, the
   operator workspace's key, or this workspace's Settings → API Keys). The
   action runs one cheap completion (`WEB_SEARCH_OPENROUTER_MODEL`, default
   `openai/gpt-4o-mini`) carrying OpenRouter's `openrouter:web_search` server
   tool; the engine (Exa by default) does the searching and the pages come
   back as citations. Billed per search on your OpenRouter account. An agent
   whose own route is OpenRouter also gets the server tool attached to its
   turns (`WEB_SEARCH_MAX_USES_PER_TURN`, default 3) — the model decides when
   to search and the reply ends with a **Sources** list.
2. **Composio** — `COMPOSIO_KEY` set (§7). The no-auth Composio Search toolkit
   is called as an internal engine; no app is added to the workspace.
3. **SearXNG** — `docker compose --profile search up -d` starts
   `searxng/searxng` on `127.0.0.1:${SEARXNG_PORT:-8888}` with JSON output
   enabled (`envs/searxng/settings.yml`); set `SEARXNG_URL=http://searxng:8080`.

Without any of the three, `platform_web_search` returns `{available:false, options}`
naming them; `platform_web_fetch` keeps working.

**What is always refused, in every configuration:** private, loopback,
link-local and metadata ranges (`10/8`, `172.16/12`, `192.168/16`, `127/8`,
`169.254/16` and the IPv6 equivalents), resolved through DNS
(`core/security/url_validator.py`). That is what keeps an agent out of
`postgres:5432`, `minio:9000`, `host.docker.internal` and a cloud VM's
metadata endpoint. `WEB_ACCESS_DENY` adds hosts to that (suffix match:
`example.com` also blocks `www.example.com`) — it never widens it.

**The switch.** `WEB_ACCESS` defaults to **on** when `AUTH_EDITION=local` and
**off** otherwise. Off means: no provider search tool on any request, and
both actions answer `{available:false, reason}`. In the hosted edition an
operator must set `WEB_ACCESS=on` explicitly. All of these are read at process
start — change them in `.env` and `docker compose up -d backend`.

## 7b. Media rendering (the `media` profile)

Socials (PRD-251) renders its videos and images in a separate service,
`media-render`. It assembles a finished MP4 or PNG from a template, your brand
kit, the Kokoro voice and the music; it calls no paid provider. It is an
optional compose profile because its image is the heaviest in the stack
(Chromium, ffmpeg, Hyperframes and the 310 MB Kokoro model), and the first build
takes several minutes.

```bash
docker compose --profile media up -d media-render
```

Or add the profile for good: `COMPOSE_PROFILES=media` in `.env`, then `make up` / `docker compose up -d`.
No registry publishes this image, and `make up` starts the stack with `--no-build`, so
it never builds it: run the command above once first, and again with `--build` after a
`git pull` that changes `services/media-render` (§10).

- **Wiring.** The backend already points at it: `SOCIALS_RENDER_URL=http://media-render:8090`
  in `envs/api.defaults`. An optional shared secret, `SOCIALS_RENDER_TOKEN` in `.env`,
  reaches both containers.
- **Without the profile**, rendering a post answers *"Rendering needs the media
  profile"* and the post stays as it was. Everything else in Socials works.
- **Where files go.** Rendered files are stored in MinIO under
  `social-media/<workspace>/<post>/` and appear in **Deliverables → Outputs**.
  Media inputs reach the renderer only as presigned links on your MinIO bucket
  (`MEDIA_RENDER_MEDIA_URL_PREFIXES`, default `http://minio:9000/automatos-ai/`).
- **Resources.** Compose caps it at 4 CPUs and 8 GB. It renders two videos at
  once at most, one per workspace, and queues the rest.
- **Render minutes.** The local edition has no monthly render quota. The hosted
  plans do: Basic 10, Pro 60 and Business 240 minutes a month.

### Media links and public storage

Socials hands out its media as presigned links: signed, expiring URLs to the
stored file, served inline with the file's type (`video/mp4`, `image/jpeg`,
`image/png`), so a browser plays them and a fetcher can ask for byte ranges. A
link lives `SOCIALS_MEDIA_URL_TTL_SECONDS` (default 86400, one day).

- **The approval view** shows a post's exact media through these links. Your
  browser opens them, so they are signed against `S3_PUBLIC_ENDPOINT_URL`
  (`http://localhost:9000` in `envs/api.defaults`, §6). Nothing else is needed.
- **Publishing is file-first.** LinkedIn, X, Instagram, TikTok and YouTube get
  the file itself through your Composio connection, so none of them needs to
  reach your storage.
- **Public storage** is needed only by a channel step that takes nothing but a
  URL, which the platform then fetches: today, the YouTube custom thumbnail.
  The hosted edition's AWS S3 serves such links as it is. The local edition's
  MinIO is private, so that step shows **"Needs public storage"** and is skipped
  until you set both of these in `envs/api.local`:
  - `SOCIALS_PUBLIC_MEDIA_BUCKET`: the bucket platforms fetch from. Socials
    copies the file into it under `social-media/<workspace>/<post>/<file>` and
    links it from there. MinIO creates it on first use.
  - `S3_PUBLIC_ENDPOINT_URL`: an address of your store that the internet can
    reach (an `https://` host, never `localhost`). Every presigned link is
    signed against it, so a reverse proxy in front of MinIO must pass the
    `Host` header through unchanged. It can expose that bucket's path alone.

### Publishing and scheduling posts

Socials publishes an approved post to its channels itself (PRD-251 Wave 3), through
the workspace's own Composio connections; no platform key goes in Settings. What it
needs:

- **`COMPOSIO_KEY`** (§7), each channel connected in Composio by the workspace
  (LinkedIn, X, Instagram as a professional account, TikTok, YouTube), and the action
  cache synced (Settings → Tools → Sync), so every step of a channel's sequence is
  known.
- **X: your own X API app.** Composio removed its managed X credentials in February
  2026, so X is connected in Composio with the workspace's own X app. A refused X
  credential shows this note on the channel's receipt.
- **LinkedIn images** go through the workspace's own "LinkedIn Community Management
  OAuth2 API" credential (Composio cannot upload LinkedIn images); text and video go
  through Composio.
- **The `media` profile** (above) for Instagram images: Instagram takes JPEG only and
  the stills render as PNG, so `media-render` converts them at publish time. Without
  it an Instagram image or carousel fails saying so; every other channel publishes.
- **`SOCIALS_PUBLIC_MEDIA_BUCKET`** only for the YouTube custom thumbnail (see *Media
  links and public storage*); without it the thumbnail is skipped and the receipt
  says so.
- **Scheduling** runs on the backend's scheduler (one worker holds the scheduler lock;
  with `REDIS_URL` set its jobs survive a restart). A slot missed by more than
  `SOCIALS_MISFIRE_GRACE_SECONDS` (default 1800) while the backend was down is marked
  **Missed**, with a notification, and never posted late; reschedule it or publish
  it now.
- **Tuning** (optional, `envs/api.local`): `SOCIALS_MAX_TARGET_ATTEMPTS` (3 tries of
  a channel on a transient error), `SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS` (5),
  `SOCIALS_PUBLISH_POLL_SECONDS` (5) and `SOCIALS_PUBLISH_MAX_WAIT_SECONDS` (600) for a
  platform still processing an upload, and `SOCIALS_PUBLISH_RUN_MAX_SECONDS` (1500),
  one post's whole publish, kept under `BOOT_REAPER_STALE_MINUTES`.

Agents never publish: they draft posts, a person approves them, and only the
platform's publisher reaches a channel's publish action (D14).

### media-render on Railway (SaaS)

The hosted edition runs the renderer as its own Railway service (owner sizing,
2026-09-23). The entry is `media-render` in `infrastructure/railway-manifest.json`.

1. **Create the service** in the Automatos AI project: *New → GitHub repo →
   AutomatosAI/automatos-ai*, with **root directory `services/media-render`**. It
   builds the Dockerfile with no build args (the last stage, `production`).
2. **Size it: 4 vCPU / 8 GB, one replica.** Its render lanes are in-process, so a
   second replica would double the concurrency.
3. **No public domain.** It is reached over Railway's private network only;
   `/health` on port 8090 is its health check.
4. **Set its variables:**
   - `SOCIALS_RENDER_TOKEN`: a long random secret (boot refuses production without one);
   - `ENVIRONMENT=production`, `SERVICE_NAME=media-render`;
   - `MEDIA_RENDER_MEDIA_URL_PREFIXES`: the documents bucket's HTTPS origin, e.g.
     `https://<S3_DOCUMENTS_BUCKET>.s3.<AWS_REGION>.amazonaws.com/`;
   - optional: `LOG_LEVEL`, `LOG_RELAY_ENABLED`, and
     `MEDIA_RENDER_MAX_CONCURRENT_RENDERS=2` / `MEDIA_RENDER_MAX_RENDERS_PER_WORKSPACE=1`
     (their defaults).
5. **Point the API at it** (the `automatos-ai-api` service):
   - `SOCIALS_RENDER_URL=http://media-render.railway.internal:8090`;
   - `SOCIALS_RENDER_TOKEN`: the same secret.
6. **Check it.** A render from the Socials tab moves the post to *Rendering*, then
   *Needs approval*, and the file shows in Deliverables. The tab shows
   *Render minutes: used / quota this month*.

## 8. What a fresh instance contains

Every boot in the local edition runs an idempotent first-run seed
(`orchestrator/core/seeds/seed_local_first_run.py`), scoped to the local
workspace:

- **Auto**, the assistant, through the same seeder the hosted edition uses,
  and the workspace's **Brand Designer**, which that seeder adds alongside it.
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

### What the books record

Every model call the platform makes is written to `llm_usage` with the provider
that served it (`openrouter`, `nvidia`, `anthropic`, `claude_code`, …), the lane
that asked (chat, board ticket, mission, heartbeat, embedding, rerank, …), the
agent, the execution it ran for, the tokens including prompt-cache reads and
writes, and the cost priced from that provider's route — $0 for a free NVIDIA
route or a subscription session, the provider's own reported figure when it
gives one (OpenRouter does). Analytics → LLM & Costs reads exactly that. The
vocabulary and the rules are in
[docs/ANALYTICS-COST-TRACKING.md](../ANALYTICS-COST-TRACKING.md).

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
| Durable memory and field memory (Qdrant) | Off by default: `QDRANT_URL` is empty in `envs/api.defaults`, so the backend attempts no writes and schedules no probe (PRD-238 S10). Switch it on with `docker compose --profile memory up -d` and `QDRANT_URL=http://qdrant:6333` in `envs/api.local`, or point that variable at your own Qdrant. |
| S3 Vectors RAG | Hosted-only; local RAG runs on pgvector (`S3_VECTORS_ENABLED=false`). Never enable it against MinIO. |
| Auto Live voice (Retell) | The switch and the Retell credentials live in the hosted edition's system settings; nothing in the local stack configures them. |
| Loki / Prometheus / log relay, the agent-opt worker | Hosted telemetry and prompt-optimisation services — silenced by empty URLs in `envs/api.defaults`. |

Product capability is not gated: every agent, tool, Playbook, Mission and
Deliverable feature that exists in the code runs locally.

## 10. Updating

```bash
git pull
make up                                      # 1. the new commit's images; migrations run as the new backend boots
docker compose exec backend alembic current  # 2. one revision, marked (head)
```

The default stack runs the published images, with the code inside them, so a
pull changes nothing until `make up` fetches the new commit's images (§1b) and
recreates the containers; the new backend migrates before it serves.

**OS security fixes.** Every Monday CI rebuilds the images with fresh Debian and
Alpine packages (`weekly-<yyyymmdd>`) and moves `edge` to that rebuild once it
boots. A checkout keeps the OS packages its `sha-<7>` was built with;
`git pull && make up` picks up the weekly fixes, because `make up` then runs
`edge` when it is a rebuild of your commit, or the newer commit's `sha-<7>`,
which CI built on the fresh packages. `AUTOMATOS_IMAGE_TAG=edge make up` follows
`edge` and pulls it on every call; `AUTOMATOS_IMAGE_TAG=sha-<7> make up` pins one
build.

**Under `make dev`, stop the backend before you pull.** That stack runs the
backend in reload mode with `./orchestrator` bind-mounted, so a pull under a
running backend loads the new code at once, and its startup `create_all`
builds any new tables before the new migrations run. Migrations are written to
survive that ([AGENTS.md](../../AGENTS.md) → Migrations), but stopping first
keeps the order they expect: migrations first, then the code that uses them.

```bash
docker compose stop backend   # under make dev: stop the API before the code changes
git pull
make dev
```

Building from source (`make build`)? Update with `git pull && make build`.
Nothing is bind-mounted from your checkout outside `make dev`, so there's no
reload-mode ordering to worry about; migrations run as the new backend boots.
Database migrations run on every backend boot (`alembic upgrade heads`,
fail-closed — a failing migration stops the backend rather than serving a
half-built schema), and the seeds are idempotent. If you use media
rendering, rebuild that service as well:
`docker compose --profile media up -d --build media-render` (§7b).
Never update with `down -v`: it deletes the database (§11).
Changing a value in `.env` or `envs/*` needs the container recreated
(`docker compose up -d`), not just restarted.

### The first update of an install that built its own images

Before `make up` pulled the published images, it built the three app images from
your checkout and ran the backend from your source. The first `git pull && make up`
after that change differs in four ways. Your data stays where it is.

- **`make up` pulls instead of building.** It downloads the images CI published
  for your commit (§1b; about 1.33 GB the first time, §1) and builds only one CI
  has not published. The images your checkout built stay on disk under their old
  names, `<folder>-backend`, `<folder>-frontend` and `<folder>-workspace-worker`,
  where `<folder>` is the directory you cloned into; with the default name,
  `docker image rm automatos-ai-backend automatos-ai-frontend automatos-ai-workspace-worker`
  removes them (`docker image ls` lists them). `make up-images` is gone: `make up`
  does what it did. `make build` builds from your checkout, as `make up` used to.
- **The source mount moves to `make dev`.** `make up` and plain `docker compose up`
  run the code inside the images, so an edit in your checkout changes nothing
  there; `make dev` runs the development stages with your source mounted and
  reloading (§13). Generated documents now live on the `backend_data` volume: the
  files the old backend wrote into `orchestrator/documents` in your checkout are
  no longer read, and a download falls back to MinIO's copy of a document that
  is not on the volume.
- **`.env` gains one line,** `AUTOMATOS_IMAGE_TAG=…`: the tag the last `make up` or
  `make build` ran, so later plain `docker compose` commands run the same images.
  `make` owns that line, rewrites it on every run and changes no other value in
  `.env`. Pin a tag in the shell or on the command line instead (§1b).
- **MinIO's volume is re-owned, once.** MinIO now runs as uid 65532, not root.
  On the first start `minio-volume-owner` finds the volume owned by root, changes
  its owner to 65532, logs
  `minio-volume-owner: /data was owned by 0:0; re-owned it to 65532:65532`
  (`docker compose logs minio-volume-owner`) and exits; every later start, it exits
  at once. Your buckets and objects stay as they were. Plain `docker compose up`
  does the same, so it needs no `make`.

An `.env` written before the change still runs all six containers; `make up CORE=1`
runs five (§3).

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

**`Placeholder secrets: … Each still starts with CHANGE_ME`** from `make env`, or
in the backend's log with the backend restarting. `.env` still holds a
placeholder from `.env.example`, on an install that has already started, so
nothing generated a value for it. Rotate it as §2 shows, or, with no data to
keep, `docker compose down -v` and `make up` again. `make env` changes nothing
on its own once the database volume exists.

**`Automatos needs Docker Engine 25 or later and Docker Compose 2.24.4 or later`**
from `make up` (or `make env`, `make build`, `make dev`), or **`can't set
healthcheck.start_interval as feature require Docker Engine v25 or later`**
from plain `docker compose up`. This Docker is older than the floor (§1): the
compose file probes each service every second while it starts, which needs
Engine 25. Update Docker Desktop, or on Linux install Docker Engine and its
compose plugin from Docker's repository (docs.docker.com/engine/install); a
distribution's `docker.io` package can be older. Your data volumes are kept.

**Logs.** `make logs` follows the whole stack (`make logs s=backend` for one
service). The same with plain compose: `docker compose logs -f backend` (boot
steps, migrations, seeds), `docker compose logs -f workspace-worker`,
`docker compose logs -f frontend`. `docker compose ps` shows health.

**Windows: `exec ...entrypoint.sh: no such file or directory`.** The script
may have CRLF line endings, making Linux read the shebang as `/bin/bash\r`.
The repository's `.gitattributes` keeps container runtime files at LF on new
checkouts, even with `core.autocrlf=true`. Pulling that file into an older
checkout does not rewrite unchanged files. Check the two entrypoints from
the repository root:

```bash
git ls-files --eol orchestrator/docker-entrypoint.sh services/workspace-worker/entrypoint.sh
```

The published images that `make up` pulls carry CI's copies, so this hits only
images built from your checkout (`make build`, `make dev`). If either reports
`w/crlf`, change its line-ending setting to LF in your editor and save,
preserving any local edits. Then run `make build` (or `make dev`) again.
Rebuilding alone cannot repair the backend script: the build copies your
checkout's file into the image, and `make dev` also mounts it over the image's
copy.

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

**Backend stops on boot with `DuplicateTable` or "already exists" after an
update.** New code ran before its migration (usually a pull under a running
backend, §10), and the migration named in the backend log does not tolerate a
table that `create_all` had already built. That is a bug in the migration
([AGENTS.md](../../AGENTS.md) → Migrations): open an issue with the log. Don't
edit `alembic_version` or stamp revisions by hand.

**Chat answers nothing; banner "Add an LLM key to bring Auto to life".** No
model key is stored. Settings → API Keys, or one of the keys in §2.

**Tools page: "Integrations are disabled".** No `COMPOSIO_KEY` (§7) — check the
variable actually reached the container (`docker compose exec backend env | grep
COMPOSIO`); it is read at process start, so a `.env` edit needs
`docker compose up -d backend`. Native tools keep working.

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

| | Local edition (this guide) | Hosted edition (Automatos Studio, beta opens November 2026) | Enterprise |
|---|---|---|---|
| Flag | `AUTH_EDITION=local` (backend) / `NEXT_PUBLIC_AUTH_EDITION=local` (frontend), set by `envs/*.defaults` | `AUTH_EDITION=saas` — the code default; the boot guard then **requires** Clerk (`CLERK_JWKS_URL`, `CLERK_SECRET_KEY`) and fails fast without it, so a hosted boot can never fall through to the anonymous local identity | Coming soon · [contact Automatos](https://automatos.app/contact). A Helm chart exists and is tested in CI; no `ee/` directory or licence keys yet. |
| Identity | one operator, no login | Clerk accounts, workspaces, teams, plans | — |
| Configuration | `docker-compose.yml` + `envs/*.defaults` + `.env` | Railway sets each service's environment itself and **never reads** the compose file or `envs/*` — local defaults cost the hosted edition nothing | — |
| Storage / RAG | MinIO + pgvector | AWS S3 + S3 Vectors | — |
| Tools | native tools; Composio with your own key | native tools; Composio with the platform key | — |

The same CI gates both: every change is compose-only, fresh-clone-only, or
guarded by `AUTH_EDITION`, and the fresh-clone smoke lane boots this exact
stack from an empty checkout. Contributions land in every edition under the
repository's Apache-2.0 licence with a DCO sign-off — see
[CONTRIBUTING.md](../../CONTRIBUTING.md).

### Keeping it small

Rebuilding (`make build`, `make dev`) leaves the previous image untagged and
grows the build cache, and Docker keeps both until something prunes them, so a
source-built install grows on disk with every rebuild. `make up`, `make build`
and `make dev` clean up after themselves, so this stays bounded on its own. To
reclaim at any time:

```bash
make clean     # or: docker image prune -f && docker builder prune -f --filter until=168h
```

`make clean` removes only untagged images and the build cache nothing has used
for a week. The week is deliberate: the dependency installs (the web app's
`npm install`, the API's `pip install`) run in build stages no image keeps, so
a plain `docker builder prune -f` deletes them, and the next `make build` or
`make dev` after a `git pull` would install every dependency again (several
minutes). Kept, a rebuild after a code change reuses them. A week of frequent
dependency changes can hold a few copies; `docker builder prune -f` reclaims
all of it, at the price of that reinstall. `make up` on published images builds
nothing, so it has no build cache to keep.

`make clean` never prunes volumes: a stopped stack's database volume looks
"unused" to Docker, so `docker volume prune` would delete your data. Only
`make reset` removes data, and it asks first.

On a small machine, `make up CORE=1` runs five containers instead of six and
never downloads the workspace worker's image; Code Canvas, the Deliverables
file browser and agents' workspace tools need the worker (§1b, §5).

**Docker Desktop's own cache limit.** Docker garbage-collects the build cache
only once it passes `defaultKeepStorage` — 20 GB by default, so in practice it
never fires. If you build often, lower it in Docker Desktop → Settings →
Docker Engine:

```json
{ "builder": { "gc": { "enabled": true, "defaultKeepStorage": "3GB" } } }
```

### Working on the code instead of with it

The default stack (`make up`, plain `docker compose up`) runs the published
production images, pulled from GHCR: the Next.js standalone build and a
compiler-free Python runtime, with the code inside them, because that is what a
user installs. Nothing is mounted over the code and nothing reloads. To edit the
code, run `make dev`:

```bash
make dev    # docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build --remove-orphans
```

That builds the frontend's and the backend's development stages from your
checkout and mounts your source over `/app`: the backend runs `uvicorn
--reload` and the frontend the Next.js dev server, so edits reload live. The
backend's entrypoint and healthcheck scripts are mounted from the checkout too.
It is heavier on purpose (the dev server alone holds ~3 GB of RAM).

One thing is opt-in to keep the default install small:

```bash
# headless Chromium for the workspace_html_to_png tool (+1.2 GB)
docker compose build --build-arg INSTALL_BROWSER=true workspace-worker
```

The Knowledge Graph's Leiden clustering is in every install: it runs on
graspologic-native, about 1 MB, so it needs no build option (PRD-262 US-027).

## Session mode — your own Claude Code sessions, managed (PRD-234)

Session mode lets an agent's tickets run as **your own Claude Code session on
your machine**, under your own Claude login, with Automatos as the manager above
them: Auto assigns the ticket and writes the contract, the board tracks it, the
result lands like any other agent's. API agents (BYOK, OpenRouter) keep working
exactly as before — an agent is either `api` or `cli`, and you mix them freely.

### Before you start

- Claude Code installed on this machine, and run interactively at least once
  (`claude`, then `claude login`). The host never logs in for you.
- `CLI_RUNTIME_ENABLED=true` in `.env`, then `make up` (or restart the backend).
  The flag is refused outside the local edition.
- **macOS, Linux, WSL2, or Windows 10 1809 and later.** On macOS and Linux the
  host drives sessions through a Unix pty and installs as a launchd or
  `systemd --user` service. On native Windows it uses ConPTY and a Task Scheduler
  login task, without a session sandbox and without Codex: §1a, *Session mode on
  native Windows*. Alternatively, on Windows,
  run the stack and the host inside a WSL2 distro with systemd enabled, and run
  `loginctl enable-linger <user>` so the host keeps running without a login.
  The recipe, including keeping the distro alive, is in §1a, and
  [SETUP.md](../../SETUP.md) walks through it step by step (tested in
  [issue #818](https://github.com/AutomatosAI/automatos-ai/issues/818)).
- On Linux and WSL2: `bubblewrap` and `socat` (`sudo apt-get install bubblewrap
  socat`), for the session sandbox below. macOS needs nothing. Without them the
  host does not run Claude sessions, and Settings → Session mode says why. On
  Ubuntu 24.04 and later, AppArmor also has to allow bubblewrap's user
  namespaces: [SETUP.md](../../SETUP.md) → session mode, step 2.

### The session sandbox, and what it protects

A session runs as **you**, on your machine. The host's gate judges every tool
call before it runs: file access outside the session's folders is refused, and
what it asks you about depends on the session's permission mode (below). But a gate can
only read the command line. An allowed `pytest`, `npm run build`, `python
script.py` or `git commit` runs code the session itself wrote (a `conftest.py`,
a `package.json` script, a git hook), and text a session reads (a web page, an
issue, a dependency's README) can steer what it writes.

So every Claude session's shell commands also run in Claude Code's own
operating-system sandbox (bubblewrap on Linux and WSL2, Seatbelt on macOS),
which confines those commands and everything they start:

- **Reads:** your credential stores (`~/.aws`, `~/.ssh`, `~/.config/gh`,
  `~/.git-credentials`, `~/.kube`, `~/.docker/config.json`, …), your Claude
  login, the host's own state (`~/.automatos/cli-host`) and this checkout's
  `.env` are unreadable.
- **Writes:** only the session's folders; `.git/hooks` and `.git/config` stay
  read-only.
- **Network:** only the package registries (npm, PyPI). Add a host with
  `CLI_HOST_ARGS="--session-allow-domain github.com"`. Anything else is refused.
- **No way around it:** a machine that cannot sandbox does not run the session,
  and a session cannot retry a command outside the sandbox.

What it does **not** cover: environment variables the host was started with are
passed to sessions, so keep cloud and Git tokens out of the host's environment.
It is not a virtual machine either. For the strongest isolation, run the host as
a dedicated OS user (or in a VM or container) that holds no cloud, Git or SSH
credentials. On such a host, `--no-session-sandbox` turns the sandbox off.

### Permission modes

How much a session asks before it acts. The four modes are Claude Code's own,
applied by the host's gate to every CLI:

| Mode | Edits in the session's folders | A command off the Bash allowlist |
|---|---|---|
| **Manual** | a card for each | a card for each |
| **Edit automatically** | run | a card for each |
| **Plan** | refused until you approve the plan | read-only commands run; a card for anything else |
| **Auto** | run | run |

In **Plan**, the session explores read-only and ends with a plan. Claude Code
presents it in its own plan mode; every other CLI as its final message. Only
the read-only part of the Bash allowlist runs (`git status`, `git log`, `ls`,
`cat`, `grep`, …); a commit, a test run, a redirection into a file or a
`sed -i` is a card. The plan comes to you as the **Plan card**: on the ticket, in
Command Center → Questions, the bell and Telegram. It is also saved as `plan.md`
in the ticket's deliverables, and the ticket waits on it. Then:

- **Approve** resumes the same session as Edit automatically, with the plan in
  its prompt.
- **Your own words** resume it planning again, with your feedback.
- **Reject** sends the ticket to review, with your answer as the reason.

After five rounds only Approve carries on. A Claude Code plan card you approve
while the session is still waiting lets it carry on in the same turn.

Set the workspace's default on **Settings → Session mode**; an agent can pick
its own (Agent → Model → Permission mode). The local edition defaults to
**Auto**, because the sandbox above already confines what an unlisted command
can reach. The hosted edition defaults to Edit automatically. Every mode keeps
the hard lines: `git push`, publishing and `sudo` are refused, the platform's
secrets stay unreadable, and the explicit ask-list and unresolved paths still
ask.

To fix one mode for every session on a machine, whatever the workspace says,
install the host with it: `make cli-host-install CLI_HOST_ARGS="--permission-mode manual"`.
With `--no-session-sandbox`, prefer Manual or Edit automatically: Auto then runs
unlisted commands with your full user rights.

### Pair the host, once

Settings → Session mode → **Pair a host** shows a one-time code (ten minutes).
Then, from the repository root:

```bash
make cli-host PAIR=XXXX-XXXX
```

The host exchanges the code for a token it keeps in `~/.automatos/cli-host/`
(owner-only files) and starts serving. Then make it a login service:

```bash
make cli-host-install      # launchd LaunchAgent on macOS, a systemd --user unit on Linux
make cli-host-status       # is it running, which version, which directories
make cli-host-restart      # after you change LOCAL_PROJECTS_DIR
```

The service starts at login and restarts itself when its own source or the
backend's host contract changes (a "restart requested" line in
`~/.automatos/cli-host/host.log`). `make cli-host` in a terminal still works
for a one-off session, Ctrl-C to stop. Settings → Session mode and the fleet
show the host as connected while it runs; one host per machine serves every
session agent in the workspace.

### Where sessions work

`make cli-host` registers `./workspaces` (the compose default). A ticket whose
agent names no working directory runs in
`<AUTOMATOS_WORKSPACE_DIR>/sessions/<ticket>` — the folder **Deliverables →
Explorer** shows live. When the ticket finishes, every file the session wrote
there is registered as one of the ticket's deliverables (the ticket's
Deliverables block, the Deliverables gallery) and the task report carries the
session log: model, tokens, files, refused tool calls, the last tool calls and
the `claude --resume` command.

Your own repositories: set `LOCAL_PROJECTS_DIR=/path/to/your/projects` in
`.env` — one parent folder for everything an agent may open: single
repositories, a workspace of many repositories — then `make up` and
`make cli-host-install`. The explorer shows that folder under `projects/`, the
host registers it, and each agent gets its own **workspace folder** inside it
(Agent → Model → Workspace folder; the form checks the path against the host's
registered directories and offers *Open in the Canvas*). The explorer and the
agent's Claude Code session both open there. The folder is mounted read-only
into the platform by default; sessions write to it through the host on your
machine. `LOCAL_PROJECTS_MOUNT=rw` lets the Code Canvas editor save into it
directly. Settings → Session mode chooses the default folder for agents that
name none (`projects` or the per-ticket `sessions/<ticket>` folder). A single
repository gets a git worktree per ticket; a workspace of many repositories
does not (`worktree_per_ticket` on the agent).

Approvals: on this edition a ticket you create and assign — on the board, or by
asking Auto in chat — runs without a separate approval; the workspace policy
(Command Center → Governance) still holds tickets filed by heartbeats,
schedules and channels until you approve them. Anything else needs
`CLI_HOST_ARGS="--allow /path/to/repo"`. A ticket's working directory must sit
inside a registered directory — the host refuses anything else before a process
starts, whatever the backend says. Git repositories get a `--worktree` per
session; sessions never push — you (or Auto) integrate. Files outside the
workspace folder are listed on the ticket as references (path only).

### Using it

Give an agent the runtime **Claude Code session** (`runtime: cli`, provider
`claude`, an optional model alias such as `sonnet` or `opus`) and assign it a
ticket. The ticket stays *Assigned* until the host claims it, then *In
Progress* while the session runs, then *Done* — or *Review* when the session
hit a policy denial ("could not run the tests" must never read as finished).
Cancel from the board; the session is stopped on its next event.

What the session is told is what you configured: the agent's persona and the
full text of its skills are rendered into the session's system prompt (up to
`CLI_SESSION_SKILLS_MAX_CHARS`), together with the ticket contract. Every lane
that runs agents reaches a session agent through the same ticket — a chat
message, a playbook step, a mission task, a heartbeat, a schedule, a channel —
and the lane waits for the ticket inside its own timeout. A permission
question the session asks (a command or an edit its permission mode holds) shows up as an
approval card in the Canvas and on the ticket; unanswered, it is denied after
the host's ask timeout and the ticket goes to *Review* with the reason.

**The Runtime Canvas.** Pick a session agent in the chat's agent menu (or
*Open the session in the Canvas* on its ticket) and the Canvas opens
full-screen: the explorer on the agent's workspace folder on the left, and on
the right a terminal in which the host has already launched that agent's own
Claude Code session — resumed when its transcript exists on this machine,
started fresh otherwise, with the agent's persona appended. You type alongside
it; more terminal tabs (plain shells in the same folder) up to the host's
`max_terminals`. The terminal runs on `127.0.0.1` behind single-use grants the
backend hands to the page; nothing in it passes through the platform's API
keys. A session's id is on its ticket, so `claude --resume <id>` from the
session's directory also works from any terminal of your own.

### What you will and will not see

Token usage per session, from Claude Code's own transcript — on the ticket and
in **Analytics → LLM & Costs**, where it is booked as *Claude Code ·
Subscription* at $0 beside the paid and free API routes (input, output and
cache-read tokens, per model). A resumed session reports only what the new
turn added, never its history. No dollar figures: on a subscription there is
nothing to invent. Limits are Claude Code's own — when a session reports one,
the ticket blocks with the reset time. Overage is controlled in your Anthropic
account, not here.

### The terms this design stays inside

Anthropic's Claude Code legal page permits "an end user signing in to the
unmodified Claude Code binary with their own Claude subscription", forbids
third parties from offering Claude.ai login or intermediating credentials, and
notes that usage limits "assume ordinary, individual usage". The host runs your
unmodified `claude`, never touches a token, never uses `-p` or `--bare`, and
serves one user on one machine. How many sessions you run in parallel is your
choice; the phrase above is the ceiling to keep in mind.

## Network exposure (`BIND_ADDRESS`)

Every port the stack publishes listens on `127.0.0.1` by default: the web app
(3000), the API (8000), Postgres, Redis, MinIO and the services of the optional
profiles. They are reachable from this machine only. To reach them from another
device on your network, set `BIND_ADDRESS=0.0.0.0` in `.env` deliberately. In the
local edition an anonymous request is the operator (through the web app's chat as
much as through the API), and with session mode (PRD-234) the API can start
Claude Code sessions that run commands on this machine — do not expose it to a
network you do not control.
