# SETUP.md — install Automatos on your own machine

This is the step-by-step install for the **local edition**, written so that you,
**or an AI coding agent working for you** (Claude Code, Codex, Cursor, Copilot…),
can follow it from start to finish on macOS, Linux or Windows.

> **Not technical?** Automatos Studio, the same platform as a hosted service,
> opens as a beta in November 2026: [join the waitlist](https://studio.automatos.app/#waitlist).
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

   *Check:* in the Ubuntu terminal, `docker version --format '{{.Server.Version}}'`
   prints 25 or later, and `docker compose version --short` prints 2.24.4 or later.
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
| Docker Engine 25 or later with Compose 2.24.4 or later | `docker version --format '{{.Server.Version}}'` prints 25 or later; `docker compose version --short` prints 2.24.4 or later | macOS: Docker Desktop (update it if older) · Linux: Docker Engine + compose plugin from Docker's repository (docs.docker.com/engine/install); a distribution's `docker.io` package can be older |
| Docker is running | `docker info` shows no error | Start Docker Desktop, or `sudo systemctl start docker` |
| git | `git --version` | macOS: `xcode-select --install` · Linux: your package manager |
| make | `make --version` | macOS: comes with the Xcode tools · Linux: `sudo apt install make` |
| ~5 GB free disk | `df -h .` | The first `make up` downloads about 1.33 GB of images (amd64, compressed; 1.28 GB on arm64), which Docker unpacks to more; your data comes on top |
| Free ports 3000, 8000, 5432, 6379, 9000, 9001 | `lsof -i :3000 -i :8000 -i :5432 -i :6379 -i :9000 -i :9001` prints nothing | Pick other ports in step 3 |

On Linux, if `docker info` says *permission denied*, add the user to the `docker`
group (`sudo usermod -aG docker $USER`), then log out and back in.

### 2. Get the code

```bash
cd ~
git clone --depth 1 https://github.com/AutomatosAI/automatos-ai.git
cd automatos-ai
```

*Check:* `ls` shows `docker-compose.yml` and `Makefile`.

### 3. Create `.env` with the three secrets

The stack refuses to start without `POSTGRES_PASSWORD`, `REDIS_PASSWORD` and
`API_KEY`, and refuses the `CHANGE_ME` placeholders `.env.example` ships.
`make env` generates random values; nobody needs to remember them. (`make up`
runs it first anyway; running it now lets you add the settings below.)

```bash
make env
```

It is safe to run again: it creates `.env` from `.env.example` only if it's
missing, and fills in a secret only while it is still empty or a `CHANGE_ME`
placeholder, and only before the first start. **Never replace a secret on an
install that has already started**: the database keeps the password it was
created with, and the stack stops connecting. On such an install `make env`
changes nothing; if it finds a placeholder there it stops and prints how to
rotate it.

No `make`? The same by hand:

```bash
[ -f .env ] || { cp .env.example .env && chmod 600 .env; }
docker volume inspect automatos_postgres_data > /dev/null 2>&1 || for key in POSTGRES_PASSWORD REDIS_PASSWORD API_KEY; do
  if grep -qE "^${key}=(CHANGE_ME.*)?$" .env; then
    sed -i.bak "s|^${key}=.*|${key}=$(openssl rand -hex 24)|" .env && rm -f .env.bak
  fi
done
```

Then set where Automatos keeps its files, as an **absolute path**. This is the
folder **Deliverables → Explorer** shows, and where agents' work lands:

```bash
mkdir -p ~/automatos-deliverables
grep -q '^AUTOMATOS_WORKSPACE_DIR=' .env || echo "AUTOMATOS_WORKSPACE_DIR=$HOME/automatos-deliverables" >> .env
```

Optional, in the same way:
- `LOCAL_PROJECTS_DIR=/absolute/path/to/your/projects`: your own code folders,
  shown read-only in Explorer under `projects/`.
- If a port is taken, set its variable to a free port: `FRONTEND_PORT` (3000),
  `API_PORT` (8000), `POSTGRES_PORT` (5432), `REDIS_PORT` (6379), `MINIO_PORT`
  (9000), `MINIO_CONSOLE_PORT` (9001). Two of them need a companion setting:
  - `API_PORT`: also set `NEXT_PUBLIC_API_URL=http://localhost:<port>` and
    `NEXT_PUBLIC_WS_URL=ws://localhost:<port>/ws` in `.env`.
  - `MINIO_PORT`: also set `NEXT_PUBLIC_MEDIA_ORIGINS=http://localhost:<port>` in
    `.env`, and `S3_PUBLIC_ENDPOINT_URL=http://localhost:<port>` in
    `envs/api.local` (create the file if it doesn't exist).

  The published web app is built for ports 8000 and 9000, so with these set
  `make up` builds the web app on this machine instead (a few minutes), and says
  so. A moved port without its companion setting stops `make up` with the line
  to add.

  The rest of this guide uses the defaults, 3000 and 8000; use your ports
  wherever it shows those.

*Check:* `grep -cE '^(POSTGRES_PASSWORD|REDIS_PASSWORD|API_KEY)=[0-9a-f]{48}$' .env`
prints `3`, and `grep -c CHANGE_ME .env` prints `0`.

### 4. Start it

```bash
make up
```

`make up` downloads the images CI publishes for this commit instead of
building them: about 1.3 GB the first time, so about a minute at 200 Mbit/s or
three to four at 50 Mbit/s, plus the first boot, which builds the database. It
prints the image tag it picked and why. Later starts take seconds.

No `make`? Run the equivalent, from the repository folder. It runs the `edge`
images, the newest main build that booted in CI:

```bash
mkdir -p "$HOME/automatos-deliverables/projects"
docker compose up -d --remove-orphans
```

*Check:* `docker compose ps` shows every service `running` or `healthy`, and
`curl -fsS http://localhost:8000/health` returns JSON (use your `API_PORT` if you changed it). The backend can take a
couple of minutes to turn healthy on the first boot while it creates the database.

### 5. Open it

Open **http://localhost:3000** (or your `FRONTEND_PORT`). There's no login in the local edition.

To make the agents think, the person adds **one model key** in
**Settings → API Keys**:
- **OpenRouter** (one key, 400+ models), or OpenAI / Anthropic / DeepSeek / Google…
- **NVIDIA** (build.nvidia.com): hosted open models at no charge under NVIDIA's
  trial terms; then *Marketplace → LLMs → NVIDIA*, sync, add a model.

Then try the seeded **Two-minute brief** Playbook.

---

## Optional — session mode (your own Claude Code as agents)

Session mode runs your Claude Code subscription as agents. It runs on **macOS,
Linux, or WSL2 on Windows**, and these steps use WSL2 on Windows. It also runs
natively on Windows 10 1809 or later, with differences: see
[the self-hosting guide](docs/getting-started/self-hosting.md), §1a, *Session
mode on native Windows*.

0. **Windows (WSL2) only, first:** the host installs as a `systemd --user`
   service, so enable systemd and lingering before step 4 (tested in
   [issue #818](https://github.com/AutomatosAI/automatos-ai/issues/818)):
   - Add this to `/etc/wsl.conf`, then run `wsl --shutdown` in PowerShell and
     reopen Ubuntu:
     ```ini
     [boot]
     systemd=true
     ```
     *Check:* `systemctl --user status` prints a status, not an error.
   - Keep the host running without an open terminal: `sudo loginctl enable-linger $USER`.
1. Install Claude Code on this machine (inside Ubuntu on Windows) and log in once:
   `claude`, then `claude login`. On WSL there's no browser, so open the login
   link it prints in your Windows browser.
2. **Linux and WSL2 only:** install the tools the session sandbox needs (macOS
   needs nothing). Without them the host won't run Claude sessions, and Settings →
   Session mode says why.
   ```bash
   sudo apt-get install -y bubblewrap socat
   ```
   On Ubuntu 24.04 or later, check `sysctl kernel.apparmor_restrict_unprivileged_userns`.
   If it prints `1`, allow bubblewrap to create its sandbox (this profile applies to
   `bwrap` itself, not the commands it runs):
   ```bash
   sudo tee /etc/apparmor.d/bwrap > /dev/null <<'EOF'
   abi <abi/4.0>,
   include <tunables/global>

   profile bwrap /usr/bin/bwrap flags=(unconfined) {
     userns,
     include if exists <local/bwrap>
   }
   EOF
   sudo systemctl reload apparmor
   ```
   *Check:* `bwrap --ro-bind / / true` exits without an error.
3. Add `CLI_RUNTIME_ENABLED=true` to `.env` and run `make up` again.
4. In the app: **Settings → Session mode → Pair a host**. Copy the code, then:
   ```bash
   make cli-host PAIR=XXXX-XXXX     # wait for "paired", then Ctrl-C
   make cli-host-install            # starts the host at login from now on
   make cli-host-status             # check: installed and running
   ```

Sessions run in **Auto** mode by default: they edit and run commands without
asking, inside the session sandbox, and only risky actions become a question.
To be asked more, pick **Manual**, **Edit automatically** or **Plan** under
**Settings → Session mode**.

**Windows (WSL2), after step 4:**
- WSL stops the distro a few seconds after its last window closes, and the host
  stops with it. To keep it alive, create a Windows **Task Scheduler** task that
  runs at logon with no time limit:
  `conhost.exe --headless wsl.exe -d Ubuntu-24.04 -u root -- sleep infinity`.
  Task Scheduler stops a task after 72 hours unless you turn that off. The
  PowerShell commands that create the task are in the
  [self-hosting guide → Windows](docs/getting-started/self-hosting.md#1a-windows).

Details: [self-hosting guide → Session mode](docs/getting-started/self-hosting.md).

---

## Reach your install from your phone (optional)

For Automatos Go on a phone, or a browser on another machine. Do this only when the
person asks for it: it puts the install on a network. With `LOCAL_REMOTE_ACCESS=paired`
a request with no credential is refused (`401 device_pairing_required`), from the
browser on this machine too, so each phone and each browser pairs once with a
one-time code. Turn that on before the install is reachable (step 1), then put TLS in
front (step 2).

1. **Turn on paired mode and give the stack its public addresses.** Agree two https
   addresses with the person, one for the web app and one for the API (below, Caddy's
   `studio.example.com` and `api.example.com`; Tailscale's are in step 2). Add to `.env`:
   ```bash
   LOCAL_REMOTE_ACCESS=paired
   NEXT_PUBLIC_API_URL=https://api.example.com
   NEXT_PUBLIC_WS_URL=wss://api.example.com/ws
   ```
   and to `envs/api.local`, creating it if it is missing (`.env` reaches only the
   variables compose names; this file reaches every backend setting):
   ```bash
   PUBLIC_API_HOST=api.example.com
   FRONTEND_URL=https://studio.example.com
   CORS_ALLOW_ORIGINS=https://studio.example.com,http://localhost:3000
   ```
   A phone is handed the API as `https://<PUBLIC_API_HOST>`. When the proxy will run
   on another machine, also publish the API and the web app, and only those:
   `API_PUBLISH_ADDRESS=0.0.0.0` and `WEB_PUBLISH_ADDRESS=0.0.0.0` in `.env`. Leave
   `BIND_ADDRESS` unset, so Postgres, Redis, MinIO and the other datastores stay on
   `127.0.0.1`. Then run `make up`: it builds the web app here, because its API
   address changed.

   *Check:* `curl -s http://localhost:8000/.well-known/automatos.json` (or your
   `API_PORT`) shows `"pairing":{"enabled":true}` and the API's https address as
   `api_url`.
2. **Put TLS in front** (ask before installing Caddy or Tailscale, rule 1):
   - **Caddy** on this machine, with both DNS names pointing at it (it fetches the
     certificates itself):
     ```
     studio.example.com {
         reverse_proxy 127.0.0.1:3000
     }
     api.example.com {
         reverse_proxy 127.0.0.1:8000
     }
     ```
   - **Tailscale**, for devices on the person's tailnet only: `tailscale serve --bg 3000`
     serves the web app at `https://<machine>.<tailnet>.ts.net`, and
     `tailscale serve --bg --https=8443 8000` the API on port 8443. In step 1 the web
     address is then `https://<machine>.<tailnet>.ts.net` and `PUBLIC_API_HOST` is
     `<machine>.<tailnet>.ts.net:8443`.

   A proxy on this machine reaches the ports on `127.0.0.1`: publish nothing more.

   *Check:* the web app's https address opens it, and `https://<api address>/health`
   returns JSON.
3. **Pair the first browser.** Every unpaired browser now shows *Pair this browser* in
   place of the app. Mint its code on the server:
   ```bash
   docker compose exec backend python -m scripts.mint_pairing_code
   ```
   It prints a code (it works once, for ten minutes) and its pairing link. The person
   types the code into *Pair this browser*, with a name for that browser.

   *Check:* the app shows after the code.
4. **Pair phones from Studio.** In the paired browser: **Settings → Phones & devices →
   Get a pairing code** shows a code, its ten-minute countdown and the pairing link;
   the person enters the code in Automatos Go. Another browser opens the web address
   and enters a code minted there.

   *Check:* the phone is listed under **Settings → Phones & devices**, where Revoke
   cuts it off.

**Notifications.** Automatos Go's notifications go through Expo's push service: each
one leaves the install for Expo, which hands it to the phone. Nothing leaves until the
operator agrees, and that answer is the person's to give, never yours: Studio asks once,
the first time they open *Pair a phone*, and the app asks when it pairs; they change it
later under **Settings → Phones & devices**. Tell them what it means: a question's text
and its options travel in its notification unless the workspace redacts questions (they
are then sent as a generic line, with Open only); every other notification is a generic
line. Self-hosted notifications rely on Expo's "enhanced push security" staying off on
the Automatos EAS project: with it on, Expo refuses every send that does not carry the
project's own access token.

*Check:* `curl -s http://localhost:8000/.well-known/automatos.json` shows
`"push":{"expo":true}` once they agreed, and `"push":{"expo":false}` until then.

**An install that already set `BIND_ADDRESS=0.0.0.0`** publishes the API and the web
app beyond this machine, so while `LOCAL_REMOTE_ACCESS` is unset (`off`) the backend
refuses to start, and `docker compose logs backend` says why and names the fix. Show
the person the message and let them choose: `LOCAL_REMOTE_ACCESS=paired` (above);
`LOCAL_REMOTE_ACCESS=open`, which keeps the old behaviour (whoever reaches the install
is its operator) and logs a warning at every boot; or removing `BIND_ADDRESS`, which
keeps the install on this machine. A tunnel or proxy on this machine is invisible to
that check, so behind one run `paired`. Details:
[self-hosting guide → Network exposure](docs/getting-started/self-hosting.md#network-exposure-bind_address).

---

## Done — the checklist

- [ ] `docker compose ps`: all services running or healthy
- [ ] http://localhost:3000 opens, and http://localhost:8000/health returns JSON (or your `FRONTEND_PORT` / `API_PORT`)
- [ ] `.env` has real secrets (no `CHANGE_ME`) and an absolute `AUTOMATOS_WORKSPACE_DIR`
- [ ] A model key is added in Settings → API Keys (by the person)
- [ ] Optional: session mode paired, and `make cli-host-status` shows running
- [ ] Optional: reached from a phone; discovery shows `"pairing":{"enabled":true}` and the phone is listed under Settings → Phones & devices

Day to day: `make up` starts it, `make down` stops it (your data is kept), and
`make status` shows what's running. To update: `git pull && make up`, which also
brings the weekly rebuild with fresh OS packages.

---

## Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `exec /usr/local/bin/docker-entrypoint.sh: no such file or directory` | The code was checked out on Windows with CRLF line endings | Clone inside WSL2 (Path W). An old Windows checkout: re-clone inside Ubuntu |
| `make: command not found` | make isn't installed | `sudo apt install make`, or use the plain `docker compose` command in step 4 |
| `Automatos needs Docker Engine 25 or later and Docker Compose 2.24.4 or later` (from `make up`), or `can't set healthcheck.start_interval as feature require Docker Engine v25 or later` (from `docker compose up`) | Docker is older than the floor | Update Docker Desktop, or install Docker Engine and its compose plugin from Docker's repository (step 1), then run `make up` again |
| `port is already allocated` / `address already in use` | Something else uses 5432, 3000 or 8000 (often a local Postgres) | Set `POSTGRES_PORT`, `FRONTEND_PORT` or `API_PORT` in `.env` (step 3) |
| `required variable POSTGRES_PASSWORD is missing` | `.env` is missing or the secrets are empty | Redo step 3 |
| `Placeholder secrets: … Each still starts with CHANGE_ME` (from `make env`, or the backend keeps restarting with it in its log) | An install that already started on the `.env.example` placeholders | Rotate them as it prints ([self-hosting guide §2](docs/getting-started/self-hosting.md#2-the-three-secrets-and-the-one-key)). On the install you just made, with nothing in it yet: `docker compose down -v`, then `make up` |
| Backend stays `unhealthy` | Usually a slow first boot | Wait 2–3 minutes, then `docker compose logs backend --tail 50` |
| Session files don't appear under their session in Explorer | `AUTOMATOS_WORKSPACE_DIR` is relative (`./workspaces`), or the code is under `/mnt/c` while Docker sees `C:\` | Use an absolute Linux path (step 3) and keep everything inside WSL2 |
| `Session mode on Windows needs Windows 10 version 1809 or later` | The host was started on an older Windows | Run it inside Ubuntu (WSL2) |
| Session mode says `claude_sandbox_unavailable` | `bubblewrap` / `socat` missing, or blocked by AppArmor on Ubuntu 24.04+ | Session mode, step 2 |
| Disk filling up | Old image layers from rebuilds | `make clean`: safe, never touches your data |

Still stuck? Open an issue with the error and your OS:
https://github.com/AutomatosAI/automatos-ai/issues/new/choose
