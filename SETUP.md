# SETUP.md — install Automatos on your own machine

This is the step-by-step install for the **local edition**, written so that you,
**or an AI coding agent working for you** (Claude Code, Codex, Cursor, Copilot…),
can follow it from start to finish on macOS, Linux or Windows.

> **Not technical?** You don't have to install anything:
> [automatos.app](https://automatos.app) runs the same platform as a hosted service.
> Or open an AI coding agent in an empty folder and say:
> *"Install Automatos by following https://github.com/AutomatosAI/automatos-ai/blob/main/SETUP.md"*.

[QUICKSTART.md](QUICKSTART.md) is the short version for people who already use
Docker. [docs/getting-started/self-hosting.md](docs/getting-started/self-hosting.md)
explains every service and setting.

---

## For agents: rules for this install

If you are an agent installing Automatos for someone, follow these rules:

1. **Ask before installing software** (Docker, WSL, git, make, Python) and before
   anything that needs administrator rights. Say what you'll install and why.
2. **Run the check after every step.** If a check fails, stop, show the person the
   exact error, and use *Troubleshooting* below. Don't guess past a failed step.
3. **Never ask the person to paste an API key or password into the chat.** You
   generate the three local secrets yourself (step 3). Model keys are entered by
   the person in the app, under **Settings → API Keys**.
4. **Never commit, upload or print `.env`.** It holds the secrets.
5. **Don't touch Docker data you didn't create.** Never run `docker system prune -a`,
   `docker volume prune` or `make reset`: they delete other projects' images and data.
6. **Finish with the checklist** in *Done* and tell the person what works and what
   was skipped.

---

## Step 0 — which path?

| Machine | Path |
|---|---|
| macOS (Intel or Apple silicon) | **Path A** |
| Linux (Ubuntu, Debian, Fedora…) | **Path A** |
| Windows 10/11 | **Path W**, then Path A inside WSL2 |

Check: `uname -s` prints `Darwin` (macOS) or `Linux`. On Windows, use PowerShell.

---

## Path W — Windows: set up WSL2 first

Automatos runs in Linux containers. On Windows the smooth, supported way is to
do **everything inside WSL2** (a real Linux environment built into Windows): the
code, Docker and, if you want it, session mode. Keep the code in the Linux home
folder (`~/…`), **not** under `C:\` or `/mnt/c/…`: files there are slow, and
session paths won't line up with Explorer.

1. **Install WSL2 and Ubuntu.** In PowerShell as Administrator:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Restart if asked, then open **Ubuntu** from the Start menu and create a Linux
   username and password.
   *Check:* in PowerShell, `wsl -l -v` shows `Ubuntu-24.04` with `VERSION 2`.
2. **Install Docker.** Either:
   - **Docker Desktop for Windows** (simplest): install it, then *Settings →
     Resources → WSL integration* → turn on **Ubuntu-24.04**; or
   - **Docker Engine inside Ubuntu**: follow Docker's *Install Docker Engine on
     Ubuntu* guide, inside the Ubuntu terminal.

   *Check:* in the Ubuntu terminal, `docker compose version` prints `v2.x` or later.
3. **Install git and make inside Ubuntu:**
   ```bash
   sudo apt update && sudo apt install -y git make python3
   ```
   *Check:* `git --version` and `make --version` both print a version.

From here, **run every command in the Ubuntu terminal** and follow Path A.
Open the app from your normal Windows browser at http://localhost:3000; WSL
forwards the ports.

---

## Path A — macOS, Linux, and Windows inside WSL2

### 1. Prerequisites

| Need | Check | Install if missing |
|---|---|---|
| Docker with Compose v2 | `docker compose version` | macOS: Docker Desktop · Linux: Docker Engine + compose plugin |
| Docker is running | `docker info` shows no error | Start Docker Desktop, or `sudo systemctl start docker` |
| git | `git --version` | macOS: `xcode-select --install` · Linux: your package manager |
| make | `make --version` | macOS: comes with the Xcode tools · Linux: `sudo apt install make` |
| ~5 GB free disk | `df -h .` | The images are about 3.7 GB |
| Free ports 3000 and 8000 | `lsof -i :3000 -i :8000` prints nothing | Pick other ports in step 3 |

On Linux, if `docker info` says *permission denied*, add the user to the `docker`
group (`sudo usermod -aG docker $USER`), then log out and back in.

### 2. Get the code

```bash
cd ~
git clone https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai
```

*Check:* `ls` shows `docker-compose.yml` and `Makefile`.

### 3. Create `.env` with the three secrets

The stack refuses to start without `POSTGRES_PASSWORD`, `REDIS_PASSWORD` and
`API_KEY`. Generate random values; nobody needs to remember them.

```bash
cp .env.example .env
for key in POSTGRES_PASSWORD REDIS_PASSWORD API_KEY; do
  value=$(openssl rand -hex 24)
  sed -i.bak "s|^${key}=.*|${key}=${value}|" .env
done
rm -f .env.bak
```

Then set where Automatos keeps its files, as an **absolute path**. This is the
folder **Deliverables → Explorer** shows, and where agents' work lands:

```bash
mkdir -p ~/automatos-deliverables
echo "AUTOMATOS_WORKSPACE_DIR=$HOME/automatos-deliverables" >> .env
```

Optional, in the same way:
- `LOCAL_PROJECTS_DIR=/absolute/path/to/your/projects`: your own code folders,
  shown read-only in Explorer under `projects/`.
- If a port is taken: `POSTGRES_PORT=5433`, `API_PORT=8001`, `FRONTEND_PORT=3001`.
  If you change `API_PORT`, also set `NEXT_PUBLIC_API_URL=http://localhost:<port>`.

*Check:* `grep -cE '^(POSTGRES_PASSWORD|REDIS_PASSWORD|API_KEY)=[0-9a-f]{48}$' .env`
prints `3`, and `grep -c CHANGE_ME .env` prints `0`.

### 4. Start it

```bash
make up
```

The first run builds the images and the database, which takes 5–15 minutes
depending on the machine. Later starts take seconds.

No `make`? Run the equivalent, from the repository folder:

```bash
mkdir -p "$HOME/automatos-deliverables/projects"
docker compose up -d --build --remove-orphans
```

*Check:* `docker compose ps` shows every service `running` or `healthy`, and
`curl -fsS http://localhost:8000/health` returns JSON. The backend can take a
couple of minutes to turn healthy on the first boot while it creates the database.

### 5. Open it

Open **http://localhost:3000**. There's no login in the local edition.

To make the agents think, the person adds **one model key** in
**Settings → API Keys**:
- **OpenRouter** (one key, 400+ models), or OpenAI / Anthropic / DeepSeek / Google…
- **NVIDIA** (build.nvidia.com): hosted open models at no charge under NVIDIA's
  trial terms; then *Marketplace → LLMs → NVIDIA*, sync, add a model.

Then try the seeded **Two-minute brief** Playbook.

---

## Optional — session mode (your own Claude Code as agents)

Session mode runs your Claude Code subscription as agents. It runs on **macOS,
Linux, or WSL2 on Windows**, not on native Windows.

1. Install Claude Code on this machine (inside Ubuntu on Windows) and log in once:
   `claude`, then `claude login`. On WSL there's no browser, so open the login
   link it prints in your Windows browser.
2. Add `CLI_RUNTIME_ENABLED=true` to `.env` and run `make up` again.
3. In the app: **Settings → Session mode → Pair a host**. Copy the code, then:
   ```bash
   make cli-host PAIR=XXXX-XXXX     # wait for "paired", then Ctrl-C
   make cli-host-install            # starts the host at login from now on
   make cli-host-status             # check: installed and running
   ```

**Extra steps on Windows (WSL2)**, tested in
[issue #818](https://github.com/AutomatosAI/automatos-ai/issues/818):
- Enable systemd in the distro: add this to `/etc/wsl.conf`, then run
  `wsl --shutdown` in PowerShell and reopen Ubuntu:
  ```ini
  [boot]
  systemd=true
  ```
- Keep the host running without an open terminal: `sudo loginctl enable-linger $USER`.
- WSL stops the distro a few seconds after its last window closes, and the host
  stops with it. To keep it alive, create a Windows **Task Scheduler** task that
  runs at logon with no time limit:
  `conhost.exe --headless wsl.exe -d Ubuntu-24.04 -u root -- sleep infinity`.

Details: [self-hosting guide → Session mode](docs/getting-started/self-hosting.md).

---

## Done — the checklist

- [ ] `docker compose ps`: all services running or healthy
- [ ] http://localhost:3000 opens, and http://localhost:8000/health returns JSON
- [ ] `.env` has real secrets (no `CHANGE_ME`) and an absolute `AUTOMATOS_WORKSPACE_DIR`
- [ ] A model key is added in Settings → API Keys (by the person)
- [ ] Optional: session mode paired, and `make cli-host-status` shows running

Day to day: `make up` starts it, `make down` stops it (your data is kept), and
`make status` shows what's running.

---

## Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `exec /usr/local/bin/docker-entrypoint.sh: no such file or directory` | The code was checked out on Windows with CRLF line endings | Clone inside WSL2 (Path W). An old Windows checkout: re-clone inside Ubuntu |
| `make: command not found` | make isn't installed | `sudo apt install make`, or use the plain `docker compose` command in step 4 |
| `port is already allocated` / `address already in use` | Something else uses 5432, 3000 or 8000 (often a local Postgres) | Set `POSTGRES_PORT`, `FRONTEND_PORT` or `API_PORT` in `.env` (step 3) |
| `required variable POSTGRES_PASSWORD is missing` | `.env` is missing or the secrets are empty | Redo step 3 |
| Backend stays `unhealthy` | Usually a slow first boot | Wait 2–3 minutes, then `docker compose logs backend --tail 50` |
| Session files don't appear under their session in Explorer | `AUTOMATOS_WORKSPACE_DIR` is relative (`./workspaces`), or the code is under `/mnt/c` while Docker sees `C:\` | Use an absolute Linux path (step 3) and keep everything inside WSL2 |
| `Session mode needs macOS, Linux or WSL2` | The host was started on native Windows | Run it inside Ubuntu (WSL2) |
| Disk filling up | Old image layers from rebuilds | `make clean`: safe, never touches your data |

Still stuck? Open an issue with the error and your OS:
https://github.com/AutomatosAI/automatos-ai/issues/new/choose
