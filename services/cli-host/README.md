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
   (the ticket — which names its deliverables folder — a stable system prompt that
   says what the session can reach and how to ask, a hooks-only `settings.json`)
   — never into your repository — and records the folder-trust decision where
   Claude Code reads it (`~/.claude.json`, one flag, backup kept).
4. Spawns **your** CLI, interactively, under a pseudo-terminal it only drains,
   with the argv its preset spells — for Claude Code: `--session-id`,
   `--permission-mode acceptEdits`, `--append-system-prompt-file`, `--settings`,
   `--setting-sources user`, `--strict-mcp-config`, `--add-dir`, `--name`,
   `--model` when the agent has one, `--worktree` for git repositories, and a
   short pointer prompt. Never `-p`, never `--bare` (the preset's forbidden list
   is asserted on every command line).
5. Hooks carry the turn over one bus, whatever the CLI: `PreToolUse` is the
   policy gate — it reads what the call *does* (read, write, run a shell) so the
   rules are the same for every CLI: file tools inside the directory, the Bash
   gate below, never `git push`; `PostToolUse` the files touched; `Stop` the end
   of the turn with the final text. A permission prompt that would reach the TUI
   is denied — nobody is watching it.
6. On `Stop` it reads the transcript for token usage, terminates the process,
   copies what the session wrote in its own ticket folder into the deliverables
   folder (below) and posts the result. Only a held command forces `review`
   (below).

## The Bash gate (PRD-245)

A shell command is read the way a shell reads it, then judged by every simple
command in it:

- **Segments.** The line is tokenised quote-aware (`shlex`, POSIX rules) and cut
  only on unquoted `&&`, `||`, `;`, `|`, `&`, parentheses and newlines — a `|`
  in a grep pattern or a `;` in a commit message is a character. An unbalanced
  quote is held for you, never allowed on a guess.
- **Verbs.** Each segment is judged by its verb once shell keywords (`for … in`,
  `do`, `done`, `if`, `then`, `while`, …) and leading `NAME=value` assignments
  are peeled off. Assignments and loop variables are remembered for the rest of
  the line, so `D=…/deliverables; ls "$D/tasks"` is judged on the real path;
  `$(…)` and backtick substitutions are judged as command lines of their own;
  `git -C <path> <sub…>` is judged as `git <sub…>` with `<path>` inside the
  roots. The default allowlist is read-only git (`status`, `diff`, `log`,
  `ls-files`, `blame`, `rev-parse`, …), the text verbs (`ls`, `cat`, `grep`,
  `rg`, `find`, `sort`, `uniq`, `cut`, `tr`, `sed`, `awk`, `date`, `jq`, `diff`,
  `stat`, …) and the build/test verbs (`pytest`, `npm test`, `make test`, `tsc`,
  …); an interpreter may run a file inside the roots (never `-c`/`-e`); the
  agent's `allowed_tools` add to the list. `xargs`, `env`, `sh`, `bash`, `eval`
  and `sudo` are never on it. `git push`, `git remote add`, `gh pr create`,
  `sudo`, `rm -rf /` and `curl … | sh` are denied on sight, in every spelling
  the gate can read (`git -C x push`, a newline, `$(git push)`, a subshell).
- **Every global option is peeled first.** `git -c k=v push`,
  `git --git-dir=… push` and a repeated `-C` all reach the never-allowed list as
  `git push`; `gh -R owner/name pr create` likewise. A path a global names is
  confined like any argument.
- **A program is read, not just its verb.** `awk` and `sed` stay on the list
  because a ticket needs them, but their program is refused when it can run a
  command of its own (`awk 'BEGIN{system(…)}'`, a pipe to a command, `print >
  file`, sed's `e` command or `s///e` flag) or when the gate cannot see it at
  all (`-f progfile`). A program is not a path: `sed '/foo/d'` opens with a
  regex address, so program text is left out of the path check.
- **What `find` would run is judged too.** `-exec`/`-execdir`/`-ok`/`-okdir`
  hand their command to the same rules (`-exec cat {} \;` runs, `-exec rm {} +`
  and `-exec sh -c …` are held), and `-delete`/`-fprint…` are held for you.
- **Process substitution is a command, not a redirection.** `<(cmd)` and
  `>(cmd)` run `cmd` whether or not the outer command reads the result, so the
  parenthesis is always its own token and `cmd` is judged (`echo <(git push)` is
  denied).
- **A here-document's body is data — unless its delimiter says otherwise.**
  `<<'EOF'` bodies are inert and never judged, so a file may contain the text
  `$(git push)`. An unquoted `<<EOF` expands as the shell reads it, so the
  substitutions inside that body are judged as command lines of their own.
- **Verbs that only print a path** (`echo`, `printf`, `basename`, `dirname`,
  `date`, `true`) may name one — `echo "see /etc/hosts"` is a string. The
  exemption stops at the top level: inside `$(…)` the output becomes another
  command's argument, so `cat $(echo /etc/hosts)` is denied, and a redirection
  target is always confined.
- **Read confinement.** Every absolute or `~` path an allowed verb names — also
  as an option value (`--file=/etc/x`) — must resolve inside the session's roots
  (the working folder, its worktree, the ticket folder). Outside is **denied**,
  never held: `cat ~/.automatos/cli-host/host.json` is refused. A glob is judged
  by the directory of its literal prefix; a path built from a reference the line
  never defined (`$HOME/x`, `${HOME:-/etc}/x`, `$1/x`) is held. `cd` follows the
  same rule. Relative paths resolve under the working folder, and `..` is
  refused wherever it appears in one (`./../x`, `a/../../x`) — a git range
  (`main..HEAD`) is not a traversal.
- **Redirections.** The targets of `>`, `>>`, `<`, `2>`, `&>` follow the same
  rule; `/dev/null`, `/dev/stdout` and `/dev/stderr` are always fine.
- **Everything else** — a verb outside the allowlist — is **held**: the session
  waits (`--ask-timeout`, 120 s by default) while the command is shown as a card
  on the ticket's Canvas and, once the PRD-245 backend lane lands, in the
  Questions tab, the bell and on Telegram. No answer in time is a deny.

A guardrail against accidents on your own machine, not a sandbox: what a
command reads through data it fetched at run time (`$(cat list)`, a `for` over
the output of a command) is beyond a static gate.

## Review, and where held commands appear

Only a held command forces review (PRD-245 S0.3, built in the backend lane): a
ticket lands in `review` when a hold expired or you denied it. A refused read
outside the roots, an unknown tool or a denied TUI prompt is recorded on the
ticket and does not force review. Held commands appear on the ticket's Canvas
card today and, once the backend lane lands, in the Questions tab, the bell and
the Telegram bridge — answered from either place, the same hold resolves.

## Deliverables

The ticket names the folder: `<default root>/sessions/<ticket>/` — the
`--default-root`, which `make cli-host` sets to `AUTOMATOS_WORKSPACE_DIR`, the
folder the backend maps into the workspace volume. On the turn's end the host
copies whatever the session wrote inside its own ticket folder
(`~/.automatos/cli-host/sessions/<ticket>/`, minus the host's own `ticket.md`,
`system_prompt.md`, `settings.json`, `terminal.log` and `mcp.json`) into that
folder — names kept, an earlier copy overwritten, one failed copy a warning —
and reports the copies in `files_touched`, so the backend registers them as
Deliverables exactly as it does for files written there directly. A host with
no default root copies nothing.

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
| `~/.automatos/cli-host/sessions/<ticket>/` | ticket, system prompt, settings, terminal log for one session; what the session writes here is copied to the deliverables folder |
| `<default root>/sessions/<ticket>/` | the ticket's deliverables folder (the backend registers what lands here) |

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
