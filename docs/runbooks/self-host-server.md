# Runbook — the local edition on a server of your own

**Scope:** running the local edition on one Linux server you rent (a VPS) and reach privately, so the stack and
your agents' Claude Code sessions run around the clock and your computer only shows the app. It complements
[the self-hosting guide](../getting-started/self-hosting.md), which stays the reference for every service and
setting. This page is the server-specific part: who runs what, how you reach it, how it updates, what to back up,
themes from Automatos Cloud, and the problems we hit doing it.

**What it was verified on (31 Oct 2026):** Ubuntu 24.04 LTS, Docker Engine 29.9 with Compose 5.6, 12 vCPUs and
48 GB of memory, the server reachable only over Tailscale (no public port open, SSH included). Every command
below was run there. Placeholders: `<server>` is the server's name on your private network, `<admin>` a user with
`sudo`, `<agent-user>` the user your agents' Claude Code sessions run as, `<deliverables>` and `<projects>` the
absolute paths of the two shared folders.

---

## 1. The layout

| What | Where, and who owns it | Why |
|---|---|---|
| The checkout | `/opt/automatos`, owned by root | Sessions can read the session host's code but can't change what root runs. |
| Docker | Docker Engine from Docker's own apt repository; the normal daemon, started as a system service | Rootless Docker can't run the stack (section 2). |
| Starting and updating the stack | `<admin>` runs `sudo make up` in the checkout | `<agent-user>` is not in the `docker` group: membership of that group is root. |
| `<deliverables>` and `<projects>` | owned by `<agent-user>`, created **before** the first `make up` | The worker then writes them as `<agent-user>`'s uid (the guide's §5 *Linux ownership note*). A folder `make up` creates as root stays root's, and the worker can't write it. |
| `.env` | in the checkout, mode 600, root's | `make env` creates it readable by its owner alone. |
| The ports | `127.0.0.1` only (`BIND_ADDRESS`'s default), with the firewall dropping all public inbound traffic | You reach the app through a tunnel (section 4), so nothing is published. |

## 2. Why not rootless Docker

Under rootless Docker, the user who runs Docker is root inside every container. The workspace worker runs as uid
1000; it sees your deliverables folder as root's, can't write it, and stays uid 1000 (the guide's §5 says what it
does then). With the normal daemon it sees the folder's real owner and runs as that uid, so what agents write is
`<agent-user>`'s, as it is on macOS and Windows. Its log says which it chose:

```
[entrypoint] uid 1000 cannot write /workspaces/<workspace id>: the worker runs as its owner, <uid>:<gid>
```

## 3. Install

As `<admin>`, after installing Docker Engine from Docker's repository
(docs.docker.com/engine/install/ubuntu):

```bash
sudo apt-get install -y make
sudo systemctl enable --now docker
sudo git clone https://github.com/AutomatosAI/automatos-ai.git /opt/automatos
sudo -u <agent-user> mkdir -p <deliverables>/projects     # before make up, so root never owns it
cd /opt/automatos
sudo make env                                              # .env with the three random secrets
```

Then set in `.env` (as root): `CLI_RUNTIME_ENABLED=true` for session mode,
`AUTOMATOS_WORKSPACE_DIR=<deliverables>`, `LOCAL_PROJECTS_DIR=<projects>`. Leave the `NEXT_PUBLIC_*` lines at their
defaults (section 4). Start it:

```bash
sudo make up
```

The first `make up` downloads about 1.3 GB of images and needs no build. For session mode on Ubuntu 24.04, also
follow [SETUP.md](../../SETUP.md) → session mode, step 2: AppArmor blocks bubblewrap's user namespaces until its
profile is added.

## 4. Reaching it from your computer

Open an SSH tunnel on the **same port numbers**:

```bash
ssh -N -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000 -L 9000:127.0.0.1:9000 <admin>@<server>
```

Then open `http://localhost:3000`. The published web app has `http://localhost:8000` (the API) and
`http://localhost:9000` (files) built in, so with the same numbers on your side it works as published, the backend's
default CORS origin (`http://localhost:3000`) matches, and updates never build it (section 5).

**Keeping the tunnel up.** On macOS, a LaunchAgent restarts it after sleep, a network change or a server reboot.
Save as `~/Library/LaunchAgents/<label>.plist` and load it with
`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/<label>.plist`:

```xml
<plist version="1.0"><dict>
  <key>Label</key><string><label></string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/ssh</string><string>-N</string>
    <string>-o</string><string>BatchMode=yes</string>
    <string>-o</string><string>ExitOnForwardFailure=yes</string>
    <string>-o</string><string>ServerAliveInterval=30</string>
    <string>-L</string><string>3000:127.0.0.1:3000</string>
    <string>-L</string><string>8000:127.0.0.1:8000</string>
    <string>-L</string><string>9000:127.0.0.1:9000</string>
    <string><admin>@<server></string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
</dict></plist>
```

`BatchMode=yes` needs a login that asks nothing (an SSH key, or Tailscale SSH). A local stack of your own uses the
same three ports: stop it while the tunnel runs.

**Other port numbers** work, at a cost. Set `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_WS_URL` and
`NEXT_PUBLIC_MEDIA_ORIGINS` in `.env`, and `CORS_ALLOW_ORIGINS` and `S3_PUBLIC_ENDPOINT_URL` in `envs/api.local`, to
the addresses your browser uses. `make up` then builds the web app on the server, on every update: 10 to 15
minutes on the server above.

For a phone, follow the guide's [Connect a phone](../getting-started/self-hosting.md#connect-a-phone-automatos-go):
its Tailscale recipe applies to a server unchanged.

## 5. Updating

```bash
cd /opt/automatos
sudo git fetch origin
sudo git checkout --detach <commit>     # or origin/main
sudo make up
```

With the defaults above, an update downloads the new commit's images and recreates the containers; nothing is
built. Measured: an update across nine merges that changed 177 web app files took 200 seconds.

**Only move to a commit whose images CI has published.** `make up` builds an image it can't download, which on a
server is slow. Check all three first:

```bash
for img in api frontend workspace-worker; do
  docker manifest inspect ghcr.io/automatosai/automatos-$img:sha-<7-char sha> >/dev/null && echo "$img ok"
done
```

CI publishes them a few minutes after a merge. A merge whose commit message contains `[skip ci]` anywhere (a
squash merge lists every commit's message) starts no workflow on `main`, so nothing is published for it until
the next push to `main` or a manual run of the images workflow.

### An updater that never builds, and rolls back

This script moves to `origin/main` or a given commit, refuses a commit with no published images, waits for the stack
to be healthy and goes back to the previous commit if it isn't. Going back is best effort: a migration the newer
backend already ran is not undone. Install it as `/usr/local/sbin/automatos-update`, owned by root:

```bash
#!/usr/bin/env bash
set -euo pipefail
STACK_DIR=/opt/automatos
LOG=/var/log/automatos-update.log
IMAGES=(api frontend workspace-worker)
HEALTH_TIMEOUT_S=300
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

target="${1:-origin/main}"
if [[ "$target" != origin/main && ! "$target" =~ ^[0-9a-f]{7,40}$ ]]; then
  echo "usage: automatos-update [commit-sha]" >&2; exit 2
fi
exec 9>/run/automatos-update.lock
flock -n 9 || { echo "another automatos-update is running" >&2; exit 1; }

cd "$STACK_DIR"
git fetch -q origin
new="$(git rev-parse --short=7 "${target}^{commit}")"
old="$(git rev-parse --short=7 HEAD)"

published() {
  local img
  for img in "${IMAGES[@]}"; do
    docker manifest inspect "ghcr.io/automatosai/automatos-$img:sha-$1" >/dev/null 2>&1 || return 1
  done
}
healthy() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_S)) statuses
  while (( SECONDS < deadline )); do
    statuses="$(docker ps --format '{{.Status}}')"
    if curl -fsS -o /dev/null --max-time 5 http://127.0.0.1:8000/health \
      && curl -fsS -o /dev/null --max-time 5 http://127.0.0.1:3000 \
      && ! grep -qE 'unhealthy|Restarting|starting' <<<"$statuses"; then
      return 0
    fi
    sleep 5
  done
  return 1
}
deploy() { git checkout -q --detach "$1" && make up >>"$LOG" 2>&1; }

[ "$new" = "$old" ] && { log "already on $new"; exit 0; }
published "$new" || { log "CI has not published the sha-$new images yet: staying on $old"; exit 3; }

log "updating $old -> $new"
if deploy "$new" && healthy; then log "OK: running $new"; exit 0; fi
log "FAIL: $new is not healthy; rolling back to $old"
if deploy "$old" && healthy; then log "rolled back: running $old"; else log "ROLLBACK FAILED"; fi
exit 1
```

Run it nightly with a systemd timer (`/etc/systemd/system/automatos-update.service` and `.timer`, then
`sudo systemctl enable --now automatos-update.timer`):

```ini
# automatos-update.service
[Unit]
Description=Update the Automatos stack (published images only)
After=docker.service network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/automatos-update
TimeoutStartSec=30min

# automatos-update.timer
[Unit]
Description=Nightly Automatos update
[Timer]
OnCalendar=*-*-* 05:00:00
Persistent=true
[Install]
WantedBy=timers.target
```

`sudo automatos-update <sha>` pins one commit by hand; `/var/log/automatos-update.log` says what each run did.

## 6. Backups

Five things, all needed together. Every command was run on the server above, from `/opt/automatos`.

| What | Command | Notes |
|---|---|---|
| The database | `sudo docker compose exec -T postgres pg_dump -U postgres -d orchestrator_db -Fc > db.dump` | Consistent while the stack runs. About 0.9 MB on a fresh install. |
| The credential key and stored brand files | `sudo docker cp automatos_backend:/app/data - \| gzip > backend_data.tar.gz` | Holds `.credential_key` and `documents/`. Keep it with the database: without the key, every API key stored through the UI is undecryptable (the guide's §11). |
| Object storage | inside the MinIO container: `mc alias set b http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"`, then `mc mirror b/automatos-ai <folder>` | Brand logos, generated documents and media. Run with `docker compose exec -T minio sh -c '…'` so the credentials come from the container's own environment. |
| `.env` | copy it | The database opens only with its `POSTGRES_PASSWORD`. |
| `<deliverables>` | copy it | What agents wrote. A bind mount, so it is not in any volume. |

**Prove a dump restores** without touching the live database: restore it into a scratch database, compare, drop
it.

```bash
sudo docker compose exec -T postgres createdb -U postgres restore_check
sudo docker compose exec -T postgres pg_restore -U postgres -d restore_check --no-owner < db.dump
sudo docker compose exec -T postgres psql -U postgres -d restore_check -c "select count(*) from document_templates"
sudo docker compose exec -T postgres dropdb -U postgres restore_check
```

Copy the five off the server: a backup on the same disk is lost with it. A full restore and its targets are in
[DR-postgres.md](DR-postgres.md).

## 7. Themes from Automatos Cloud

A local install pulls the official themes with **Sync with Automatos Cloud** (Marketplace header, or Settings ›
Automatos Cloud): one anonymous read of Automatos Cloud's public catalogue, from the server, so the server needs
outbound HTTPS and nothing inbound. What it sends, checks and stores is in
[Sync with Automatos Cloud](../community-marketplace/sync-with-automatos-cloud.md). What it did on the server above:

- **The first Sync** took about 2 seconds and listed 10 signed versions of 4 themes, each card marked
  **Official · From Automatos Cloud**.
- **Adding one** (**Add to my workspace**) downloaded its package from Automatos Cloud and checked its signature
  again before installing it: 12 to 20 seconds.
- **Updates:** an install not pinned to a version moves to a newer listed version at the next Sync. An older HUD
  installed on purpose was offered the newer one at once, and the next Sync moved it.

## 8. Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| `make up`: "Docker does not answer (docker version failed) … permission denied while trying to connect to the docker API at unix:///var/run/docker.sock" | The user isn't in the `docker` group, by design (section 1). | `sudo make up`. |
| `make up`: "`docker pull ghcr.io/automatosai/automatos-api:…` failed (exit 1)"; a manual `docker pull` ends "read tcp [<IPv6>]:…->[2606:50c0:…]:443: read: connection reset by peer" | The server's IPv6 route to GitHub's package downloads (`2606:50c0::/32`) drops large transfers partway. The address in the error tells you it is IPv6. | Refuse IPv6 to that range, so the connection fails at once and Docker uses IPv4 instead: `sudo ufw reject out to 2606:50c0::/32`. |
| The worker's log: "uid 1000 cannot write …" and agents' files can't be written | Rootless Docker (section 2), or a shared folder created by root. | The normal daemon, and folders owned by `<agent-user>`. |
| Every update rebuilds the web app (10–15 minutes) | `.env` sets `NEXT_PUBLIC_*` values that differ from the published web app's. | The defaults, and a tunnel on the same port numbers (section 4). |
| The tunnel: "bind [127.0.0.1]:9000: Address already in use … Could not request local forwarding" | Something on your computer already uses the port, often a local stack. | Stop it. The web app needs 8000 and 9000 to be the tunnel's. |
| `make up` builds the app images; the updater says "CI has not published the sha-… images yet" | CI is still building that commit's images, or its merge carried `[skip ci]` (section 5). | Wait, or stay on the last published commit. |
| Session mode: `claude_sandbox_unavailable` | Ubuntu 24.04's AppArmor blocks bubblewrap's user namespaces. | [SETUP.md](../../SETUP.md) → session mode, step 2. |
