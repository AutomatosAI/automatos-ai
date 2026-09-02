# Automatos CLI host (PRD-234 Session Mode)

The local process that runs `runtime: cli` tickets as **your own Claude Code
sessions on your machine**, with Automatos (the local edition, in Docker) as the
manager above them. Standard-library Python 3.9+, nothing to install.

```
make cli-host PAIR=XXXX-XXXX   # first time — the code comes from Settings → Session mode
make cli-host                  # afterwards; Ctrl-C to stop
```

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
