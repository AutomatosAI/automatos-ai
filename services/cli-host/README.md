# Automatos CLI host (PRD-234 Session Mode)

The local process that runs `runtime: cli` tickets as **your own CLI sessions on
your machine** — Claude Code, Codex and GitHub Copilot today, the others as their adapters land — with
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
with your ChatGPT plan is required — an API-key login is refused before spawn)
and **GitHub Copilot** (hooks tier — `copilot -p` per turn in a per-agent
`COPILOT_HOME`, Claude-format hooks; `copilot login` with your Copilot seat, see
[GitHub Copilot CLI](#github-copilot-cli-prd-253) below).

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
unit on Linux, a Task Scheduler login task on Windows (`--install`; the task runs
the host under a small supervisor, `winservice.py`, because Task Scheduler only
restarts a task that fails to start) — with the same directories `make cli-host`
would use:

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

## Native Windows (#818)

The host runs on Windows 10 1809 and later as well as macOS and Linux, standard
library only. The guide's §1a (*Session mode on native Windows*) is the operator's
view; underneath:

| Unix | Windows | Where |
|---|---|---|
| a pty; stop = SIGHUP/SIGTERM/SIGKILL to the process group | ConPTY through ctypes; each session in a kill-on-close job object; stop = close the console, then end the job | `ptyproc.py`, `conpty.py`, `winjob.py` |
| hooks over `hooks.sock`, peer-PID checked | a named pipe, random per start; an HMAC handshake on each connection's own thread with a key handed to the session; the shim checks the server PID and denies rather than wait | `hook_pipe.py`, `hook_shim.py` |
| `--nudge` = SIGHUP | a restart-request file the host watches; `--uninstall` leaves a stop request | `lifecycle.py` |
| launchd / systemd | a Task Scheduler login task running a supervisor | `winservice.py` |
| `flock` | `msvcrt.locking` | `filelock.py` |
| the login shell | PowerShell (`pwsh`, then `powershell`, then `cmd.exe`) | `terminal_server.py` |

Refused on Windows, each with a sentence: an older Windows (`__main__.py`), a
`.cmd` or `.bat` launcher (`ptyproc.windows_command_line`), a sandboxed session
(`sandbox.WINDOWS_HINT`) and Codex (`codex_windows`).

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
4. Spawns **your** CLI under a pseudo-terminal it only drains — interactively
   for Claude Code and Codex, `copilot -p` for GitHub Copilot (below) — with the
   argv its preset spells — for Claude Code: `--session-id`,
   `--permission-mode acceptEdits`, `--append-system-prompt-file`, `--settings`,
   `--setting-sources user`, `--strict-mcp-config`, `--add-dir`, `--name`,
   `--model` when the agent has one, `--worktree` for git repositories, and a
   short pointer prompt. Never `-p`, never `--bare` for Claude Code (each preset's
   forbidden list is asserted on every command line).
5. Hooks carry the turn over one bus, whatever the CLI: `PreToolUse` is the
   policy gate — it reads what the call *does* (read, write, run a shell) so the
   rules are the same for every CLI: file tools inside the directory, the Bash
   gate below, never `git push`; `PostToolUse` the files touched; `Stop` the end
   of the turn with the final text. A permission prompt that would reach the TUI
   is denied — nobody is watching it. (GitHub Copilot can ask again for a call the
   gate already allowed, for a path or URL check of its own: that request gets the
   gate's verdict on the same call, at once.)
6. On `Stop` (for a CLI whose turn is the process, its exit) it reads the
   transcript for token usage, terminates the process,
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
- **`gh` reads, and only reads.** Sessions never publish, so every `gh`
  subcommand not known to be a read is refused — an issue, a pull request, a
  release, a workflow run, an SSH key, an extension, an alias — and so is `gh api`
  with a body or any method but `GET`/`HEAD` (PRD-253 S0.3).
- **What the gate cannot read is a card, even in Auto.** A shell's `-c` or an
  `eval` is judged as the command line it runs (`bash -c 'git push'` is denied).
  A command whose name is decided only when it runs (`$CMD`, `git${IFS}push`),
  code an interpreter takes inline (`python3 -c`, `node -e`) and a shell reading
  its commands from input (`echo … | sh`, `bash -s`, `source /dev/stdin`) are held
  for you — in Auto mode too, where an unlisted verb otherwise runs. A write tool
  that names no file is refused in every mode.
- **Plan is read-only, on every CLI.** In Plan mode only the read-only part of
  the allowlist runs (`PLAN_BASH_ALLOW`: no git write, test runner or `npm run`),
  and a redirection into a file or an in-place edit (`sed -i`) is a card. A CLI
  with no plan mode of its own has nothing else holding it read-only. Its ticket
  file says to end the turn with the plan, which reaches the backend as a
  `PlanReady` event in the final flush. The result always waits for that flush
  (PRD-253 Wave P).
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
- **Everything else** — a verb outside the allowlist — is **held** (in Auto
  mode it runs, with its paths still judged; see *Permission modes* in
  `docs/getting-started/self-hosting.md`): the session
  waits while the command is shown as a card on the ticket's Canvas, in the
  Questions tab, the bell and on Telegram. No answer in time is a deny. The wait
  is `--ask-timeout` (an hour by default), but never more than 530 s inside a
  turn: a CLI's hook waits at most 560 s for the host, and the host answers
  first (PRD-253 D4).

A guardrail against accidents on your own machine, not a sandbox: what a
command reads through data it fetched at run time (`$(cat list)`, a `for` over
the output of a command) is beyond a static gate, and so is what an allowed
command RUNS — `python x.py`, `npm run x`, `pytest` (a `conftest.py`) and
`git commit` (a hook) execute code the session wrote.

## The session sandbox (`sandbox.py`)

That is the operating system's job. Every Claude session's `--settings` file
also switches on Claude Code's own Bash sandbox (bubblewrap on Linux and WSL2,
Seatbelt on macOS), which confines every shell command and every process it
starts. Credential stores (`~/.aws`, `~/.ssh`, `~/.config/gh`,
`~/.git-credentials`, …), the Claude login, the host's state dir and the
platform checkout's `.env` family are unreadable; writes stay in the session's
folders (`.git/hooks` and `.git/config` stay read-only); the network is the
package registries plus `--session-allow-domain HOST`, and any other host is
refused, never asked. `failIfUnavailable` and `allowUnsandboxedCommands: false`
leave no unsandboxed path, and `autoAllowBashIfSandboxed: false` keeps the gate
deciding every call first. On Linux the host checks for `bwrap` and `socat` and,
without them, does not serve Claude (`claude_sandbox_unavailable`, with the
install command). On native Windows there is no sandbox to configure, so a
sandboxed session is refused with the same reason and the way out.
`--no-session-sandbox` turns it off for a host that is already isolated (a VM, a
container, a dedicated user with no credentials).
Codex sandboxes itself (`-s workspace-write`). GitHub Copilot sessions get
Copilot's own sandbox, configured the same way ([below](#github-copilot-cli-prd-253)).

## GitHub Copilot CLI (PRD-253)

**Install and log in.** GitHub Copilot CLI 1.0.70 or later, the first version
whose hooks fail closed: `brew install copilot-cli` (a standalone binary) or
`npm install -g @github/copilot`. Then run `copilot login` with the account that
holds your Copilot seat. A `gh auth login` to that account also works: Copilot
asks `gh` for the token itself.

Two kinds of login are refused, because sessions never carry a token and the host
never copies one:
- a login that exists only as an environment token (`COPILOT_GITHUB_TOKEN`,
  `GH_TOKEN`, `GITHUB_TOKEN`);
- a plaintext token in `~/.copilot/config.json`.

Settings → Session mode shows the account each host runs Copilot as (`login`) and
how it logs in (`copilot`: its own login, with the token in the OS credential
store; or `gh`).

**How a turn runs.** Each turn is one `copilot -p` process on the host's
pseudo-terminal, and the process exit ends the turn. GitHub documents `-p` for
programmatic use and bills it the same way as an interactive prompt (premium
requests, or AI credits).

```
copilot -p "<pointer>" (--session-id <uuid> | --resume <id>) --add-dir <session dir> [--add-dir <deliverables>]
        [--model M] [--name "automatos #N"] [--worktree automatos-N] [--additional-mcp-config @<session>/mcp.json]
        --no-ask-user --disable-builtin-mcps --no-remote --no-auto-update
```

- **No allow flag, ever.** In `-p`, Copilot itself refuses any call that would
  need approval, so a call runs only when the gate allows it. A missing hook, a
  crashed shim or a timed-out hook therefore ends in a refusal, never in an
  allowed call. Every allow flag (`--allow-all-tools`, `--yolo`, `--allow-tool`,
  …) is on the preset's forbidden list, and `COPILOT_ALLOW_ALL` is stripped.
- **The agent's own home.** `COPILOT_HOME` is
  `~/.automatos/cli-host/agents/<agent>/.copilot`, rebuilt on every spawn. It
  holds three files:
  - `config.json`: your account pointer, and no trusted folder. In `-p`, an
    untrusted folder keeps repository hooks, workspace MCP servers and
    extensions out.
  - `settings.json`: memory, auto-update, tips and the silent model switch off,
    plus your own co-author and proxy settings.
  - `hooks/automatos.json`: our hooks, in Claude's format. Copilot answers them
    in Claude's shape.

  Your own `~/.copilot` is never written, and your MCP servers never come along.
  The home is per agent, not per ticket, because Copilot keeps its session index
  there and `--resume` needs it.
- **The soul and the ticket** ride the first prompt (`UserPromptSubmit` →
  `additionalContext`), because Copilot has no system-prompt flag.
- **The gate reads Copilot's tools** (`bash`, `view`, `create`, `edit`,
  `apply_patch`, `grep`, `glob`, `web_fetch`, …) the way it reads Claude's.
  Copilot's own plan mode, subagents and other agent tools are denied. Plan is
  the plan turn, as on every CLI.
- **Held calls.** The shim waits up to 560 s for your answer, and Copilot's hook
  timeout is 600 s. A Copilot hook that times out lets the call through to
  Copilot's own check, so the shim always answers first.
- **The Automatos tools.** They come in as an HTTP MCP server, from a 0600 file
  in the session folder that is removed when the turn ends.
  - GitHub's own MCP server is off (`--disable-builtin-mcps`): it would let a
    session write to GitHub with your token, outside the gate.
  - If your organisation's MCP policy allows only registry servers, Copilot
    refuses ours. The ticket then says the session ran without the Automatos
    tools.
- **Usage.** The host reads tokens per model, AI credits and premium requests
  from the session record (`session-state/<id>/events.jsonl`). They are booked
  as your plan's usage, never as a price. Running out of AI credits, or hitting
  a rate limit, pauses the host for Copilot and puts the ticket back in the
  queue.
- **Taking over.** The ticket's Runtime Canvas terminal runs
  `copilot --resume <id>` in the agent's home, with you at the keyboard. Our
  hooks stand aside there.

**The sandbox.** Copilot sandboxes its own shell commands (Seatbelt on macOS,
bubblewrap on Linux; still experimental in 1.0.91). When the host sandboxes, the
agent's `settings.json` turns that sandbox on, with these limits:
- writes go only to the working folder and the session's folders;
- the credential stores, the platform's secrets and the host's state are
  unreadable;
- there is no bypass, no git or gh credentials and no keychain;
- outbound network reaches only the allowed hosts, never the local network.

The host refuses Copilot (`copilot_sandbox_unavailable`) on a machine without
the prerequisites:
- macOS: `sandbox-exec`;
- Linux: bubblewrap 0.5 or later, slirp4netns, util-linux 2.35 or later,
  iptables and `/dev/net/tun`.

`--no-session-sandbox` turns the sandbox off on a host that is already isolated.

**Refusals.** Settings shows each one with its sentence. A Copilot ticket waits
for a host that serves Copilot.

| Code | When | What to do |
|---|---|---|
| `copilot_missing` | No `copilot` on your PATH | Install it (above) |
| `copilot_too_old` | Older than 1.0.70 | `copilot update` |
| `copilot_not_logged_in` | No Copilot login and no `gh` login | `copilot login`, or `gh auth login` with an account that has a Copilot seat |
| `copilot_plaintext_token` | The login is a plaintext token in `config.json`, and `gh` is not logged in | Turn on the OS credential store (the macOS Keychain; on Linux or WSL2, a Secret Service such as gnome-keyring) and run `copilot login` again |
| `copilot_managed_hooks_only` | A managed policy (`/etc/github-copilot/policy.d/*.json`) sets `allowManagedHooksOnly` | Ask your Copilot administrator to allow user hooks |
| `copilot_sandbox_unavailable` | The host sandboxes and a prerequisite is missing | Install it, or use `--no-session-sandbox` on an isolated host |
| `copilot_hooks_disabled_here` | Per ticket: the repository's `.github/copilot/settings.json` (or `settings.local.json`) sets `disableAllHooks` | Remove the setting, or give the agent another folder |

**What an organisation administrator must allow:**
- **Copilot CLI** for the users (the Copilot CLI policy).
- **The models** the agents name, enabled at the enterprise or organisation
  level. "No model available. Check policy enablement…" is an error on the
  ticket, not a pause.
- **User hooks.** Under `allowManagedHooksOnly` no hook of ours loads, so the
  host refuses Copilot.
- **The Automatos MCP server,** where the MCP policy allows only registry
  servers. Otherwise sessions run without the Automatos tools.
- **The sandbox prerequisites** above, on Linux machines.

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
- your own login (`claude login`, `codex login`, `copilot login` or `gh`); no
  credential is ever read, copied or set; `CLAUDE_CONFIG_DIR` is never overridden;
- no identity games: each preset names the keys and session markers stripped from
  its session environment (`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`,
  `CLAUDE_CODE_ENTRYPOINT`, every `CLAUDE*` marker for Claude Code); the Canvas
  terminal's shell gets the union over every CLI;
- only the surface each vendor keeps on your plan: interactive sessions for
  Claude Code and Codex, `copilot -p` for GitHub Copilot (GitHub documents it for
  programmatic use and bills it like an interactive prompt). No other headless
  mode, server or remote control, ever: each preset names the flags that would be
  one;
- one user, one machine; the host refuses any backend that is not the local
  edition with `CLI_RUNTIME_ENABLED=true`.

## Files

| Path | Purpose |
|---|---|
| `~/.automatos/cli-host/host.json` (0600) | the host token minted at pairing — the only secret |
| `~/.automatos/cli-host/allowlist.json` | directories sessions may work in |
| `~/.automatos/cli-host/sessions.json` | process table (killed on the next start if left behind) |
| `~/.automatos/cli-host/hooks.sock` | the loopback socket hooks talk to (on Windows: a named pipe with a random name, both ends holding a per-start key) |
| `~/.automatos/cli-host/sessions/<ticket>/` | ticket, system prompt, settings, terminal log for one session; what the session writes here is copied to the deliverables folder |
| `~/.automatos/cli-host/agents/<agent>/.codex`, `.copilot` | an agent's own config home for Codex and GitHub Copilot: rebuilt on every spawn, never holding a token; the CLI's session index lives here |
| `<default root>/sessions/<ticket>/` | the ticket's deliverables folder (the backend registers what lands here) |

## Tests

`pytest -q tests` (from this directory). `tests/fake_claude.py` stands in for the
CLI: it refuses forbidden arguments, fires the hooks from the settings file,
writes a transcript where Claude Code would, and idles until terminated — so the
whole loop runs in CI without a real session or a subscription. Every CLI gets
such a fake, in *its* vocabulary and transcript shape, when its adapter lands:
`tests/fake_codex.py`, and `tests/fake_copilot.py` (`copilot -p` with Claude-format
hooks that fail open on a timeout, as the real CLI's do, and the `events.jsonl`
session record).

## Adding a CLI

The checklist is §10 of the design doc. In short: classify its tier and how a
turn ends; check it has a first-party login for the operator's own plan; add its
`CliPreset` row; write `adapters/<cli>.py` only if its preset cannot say where its
hook config lives or how its payloads translate; add `tests/fake_<cli>.py`; add
its id to `orchestrator/core/cli_presets.py` (a parity test keeps the two in step).
Nothing in `host.py`, nothing in the backend lane, nothing in the frontend.
