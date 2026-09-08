# Automatos CLI host (PRD-234 Session Mode)

The local process that runs `runtime: cli` tickets as **your own Claude Code
sessions on your machine**, with Automatos (the local edition, in Docker) as the
manager above them. Standard-library Python 3.9+, nothing to install.

```
make cli-host PAIR=XXXX-XXXX   # first time — the code comes from Settings → Session mode
make cli-host                  # afterwards; Ctrl-C to stop
make cli-host-install          # or: run it as a login service — always on (below)
```

## Always on: the host as a login service

One host per machine serves every `runtime: cli` agent of the workspace, so
"my agents are always available" means "this process is always running". A
terminal window is not that. `make cli-host-install` registers the host with
your login session manager — a launchd LaunchAgent on macOS, a `systemd --user`
unit on Linux — with the same directories `make cli-host` would use:

- starts at login, restarts within 15 s whenever it exits non-zero (a crash,
  the backend not being up yet, or a deliberate drift restart);
- logs to `~/.automatos/cli-host/host.log`;
- `make cli-host-status` / `make cli-host-restart` / `make cli-host-uninstall`.

A clean exit stays down on purpose: that is the host telling you it needs you
(not paired yet, no directory allowed). Pair first with `make cli-host PAIR=…`.

**It keeps itself current.** Every ticket spawns a fresh `claude`, so a Claude
Code update is used by the next session with no restart. The host's own code
is different: it is loaded at start. So the host drains and exits for a restart
when (a) its own files changed on disk (you switched branches or pulled), or
(b) the backend answers a heartbeat with a different host contract (the app was
rebuilt). `make up` also sends it a nudge (`SIGHUP`) after every rebuild.
"Drain" means: no new claims, running sessions finish, then exit `75` — the
service manager restarts it on the new code. In a terminal, `make cli-host`
simply exits; start it again.

## What it does, in one turn

1. Claims a ticket the board holds for a `cli` agent (same exactly-once claim the
   dispatcher uses; the ticket arrives with a pre-assigned session id).
2. Resolves the ticket's working directory against **this host's own allowlist**
   (`--allow DIR`; `make cli-host` registers `./workspaces`). Anything outside is
   refused before a process starts.
3. Writes the session's files under `~/.automatos/cli-host/sessions/<ticket>/`
   (the ticket, a stable system prompt, a hooks-only `settings.json`) — never into
   your repository — and records the folder-trust decision where Claude Code reads
   it (`~/.claude.json`, one flag, backup kept).
4. Spawns **your** `claude`, interactively, under a pseudo-terminal it only drains:
   `--session-id`, `--permission-mode acceptEdits`, `--append-system-prompt-file`,
   `--settings`, `--setting-sources user`, `--strict-mcp-config`, `--add-dir`,
   `--name`, `--model` when the agent has one, `--worktree` for git repositories,
   and a short pointer prompt. Never `-p`, never `--bare`.
5. Hooks carry the turn: `PreToolUse` is the policy gate (file tools inside the
   directory, a Bash allowlist, never `git push`), `PostToolUse` the files touched,
   `Stop` the end of the turn with the final text. A permission prompt that would
   reach the TUI is denied — nobody is watching it.
6. On `Stop` it reads the transcript for token usage, terminates the process and
   posts the result. Any denial lands the ticket in `review`, never `done`.

## The invariant it keeps (PRD-234 §Terms)

- the unmodified `claude` from your login-shell `PATH`; nothing bundled or patched;
- your own login (`claude login`); no credential is ever read, copied or set;
  `CLAUDE_CONFIG_DIR` is never overridden;
- no identity games: `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_ENTRYPOINT`
  and every `CLAUDE*` session marker are stripped from the session environment;
- interactive sessions only — the surface Anthropic keeps on your plan;
- one user, one machine; the host refuses any backend that is not the local
  edition with `CLI_RUNTIME_ENABLED=true`.

## Files

| Path | Purpose |
|---|---|
| `~/.automatos/cli-host/host.json` (0600) | the host token minted at pairing — the only secret |
| `~/.automatos/cli-host/allowlist.json` | directories sessions may work in |
| `~/.automatos/cli-host/sessions.json` | process table (killed on the next start if left behind) |
| `~/.automatos/cli-host/hooks.sock` | the loopback socket hooks talk to |
| `~/.automatos/cli-host/sessions/<ticket>/` | ticket, system prompt, settings for one session |

## Tests

`pytest -q tests` (from this directory). `tests/fake_claude.py` stands in for the
CLI: it refuses forbidden arguments, fires the hooks from the settings file,
writes a transcript where Claude Code would, and idles until terminated — so the
whole loop runs in CI without a real session or a subscription.
