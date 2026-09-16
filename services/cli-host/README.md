# Automatos CLI host (PRD-234 Session Mode)

The local process that runs `runtime: cli` tickets as **your own CLI sessions on
your machine** — Claude Code today, the others as their adapters land — with
Automatos (the local edition, in Docker) as the manager above them.
Standard-library Python 3.9+, nothing to install.

**Which CLI is a parameter of the ticket.** The pattern is
`docs/architecture/CLI-RUNTIME-ADAPTER-DESIGN.md`; in this package it is two
files and a folder:

| Where | What |
|---|---|
| `presets.py` | one `CliPreset` row per CLI — how it is *spelled*: binary, flags, hook events and timeout literal, the env it must never inherit, the arguments that break the subscription or our gate, how to tell "logged in with the operator's own plan" |
| `adapters/base.py` | the seven-method contract every CLI runs through; `PresetAdapter` implements all of it from the preset |
| `adapters/<cli>.py` | only what a preset cannot say — where a config home lives, a translating hook shim, a different transcript |

Served today: **Claude Code** (native tier) and **Codex** (hooks tier — a per-agent
`CODEX_HOME` under the host's state dir with your `~/.codex/auth.json` linked in,
hooks as `config.toml` tables, `codex resume <id>` for continuity; `codex login`
with your ChatGPT plan is required — an API-key login is refused before spawn).

The host announces every CLI in the registry with `served: true/false` and why;
the backend claims a ticket for a host only when that host serves the ticket's
CLI. A CLI the registry knows but this host cannot run is an honest line on the
ticket, never a silent fallback to another CLI.

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
4. Spawns **your** CLI, interactively, under a pseudo-terminal it only drains,
   with the argv its preset spells — for Claude Code: `--session-id`,
   `--permission-mode acceptEdits`, `--append-system-prompt-file`, `--settings`,
   `--setting-sources user`, `--strict-mcp-config`, `--add-dir`, `--name`,
   `--model` when the agent has one, `--worktree` for git repositories, and a
   short pointer prompt. Never `-p`, never `--bare` (the preset's forbidden list
   is asserted on every command line).
5. Hooks carry the turn over one bus, whatever the CLI: `PreToolUse` is the
   policy gate — it reads what the call *does* (read, write, run a shell) so the
   rules are the same for every CLI: file tools inside the directory, a shell
   allowlist, never `git push`; `PostToolUse` the files touched; `Stop` the end
   of the turn with the final text. A permission prompt that would reach the TUI
   is denied — nobody is watching it.
6. On `Stop` it reads the transcript for token usage, terminates the process and
   posts the result. Any denial lands the ticket in `review`, never `done`.

## The invariant it keeps (PRD-234 §Terms)

- the unmodified CLI from your login-shell `PATH` (or `--cli-binary claude=/path`);
  nothing bundled or patched;
- your own login (`claude login`, `codex login`); no credential is ever read,
  copied or set; `CLAUDE_CONFIG_DIR` is never overridden;
- no identity games: each preset names the keys and session markers stripped from
  its session environment (`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
  `CLAUDE_CODE_ENTRYPOINT`, every `CLAUDE*` marker for Claude Code); the Canvas
  terminal's shell gets the union over every CLI;
- interactive sessions only — the surface each vendor keeps on your plan; no
  headless mode, ever (each preset names the flags that would be one);
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
whole loop runs in CI without a real session or a subscription. Every CLI gets
such a fake, in *its* vocabulary and transcript shape, when its adapter lands.

## Adding a CLI

The checklist is §10 of the design doc. In short: classify its tier and how a
turn ends; check it has a first-party login for the operator's own plan; add its
`CliPreset` row; write `adapters/<cli>.py` only if its preset cannot say where its
hook config lives or how its payloads translate; add `tests/fake_<cli>.py`; add
its id to `orchestrator/core/cli_presets.py` (a parity test keeps the two in step).
Nothing in `host.py`, nothing in the backend lane, nothing in the frontend.
