# PRD-253: GitHub Copilot as a session CLI, and all four permission modes on every session CLI

> **Status:** IN BUILD 2026-10-02 — W0 [#855](https://github.com/AutomatosAI/automatos-ai/pull/855), Wave P [#856](https://github.com/AutomatosAI/automatos-ai/pull/856), W1–W3 in one PR stacked on #856. Each is a draft until the owner's local test.
>
> **The owner's words:** "We wrote a pattern for loading runtime agents like claude code, codex, copilot and so on … I would like to look at adding copilot next as we have a possible bank poc which would be huge." The same day, on [#845](https://github.com/AutomatosAI/automatos-ai/pull/845) (Claude Code's four permission modes for sessions, merged 2026-09-30): "User should be able to do this for all runtime models."
>
> **Grounded @** `automatos-ai` `origin/main` `19a16c722` (cli-host 0.9.0) and the CLI Runtime Adapter design rev 2 ([`docs/architecture/CLI-RUNTIME-ADAPTER-DESIGN.md`](../architecture/CLI-RUNTIME-ADAPTER-DESIGN.md)).
>
> Copilot facts come from **GitHub Copilot CLI 1.0.91** (npm `@github/copilot`, published 2026-10-01). They were read out of the package without running it:
> - the option table in `prebuilds/<platform>/cli-native.node`;
> - the `copilot help <topic>` texts and launcher code in the bundled `app.js`;
> - `changelog.json`, `schemas/session-events.schema.json` and the bundled `copilot-sdk` typings.
>
> Also used: the docs.github.com pages on hooks, the CLI command reference, authentication, ACP and enterprise administration. Copilot is **not installed** on this machine, and nothing was run. Every fact that needs a real session is listed under *Verify at build*.
>
> **Corrects the design doc.** Its Copilot row (§3.1 "seed tier", §9.1 "GitHub (?)", §14 wave 5) came from munder's code and was never checked against the CLI. Copilot is a **hooks-tier** CLI, and its events are already in Claude's shape. On 2026-09-11 you decided D-3 as "Copilot/Cursor open the gates for now". Copilot no longer needs that: it runs gated, like Codex. D-3 still applies to Cursor.

---

## Framing (CLAUDE.md §3)

**Extension.** The adapter seam (#725/#727/#731) and the session bridge (PRD-245) already run Claude Code and Codex. Copilot follows the design's §10 checklist and changes nothing above the bus.

What it adds:
- one preset row (`presets.py`);
- one adapter (`adapters/copilot.py`);
- one backend registry row (`core/cli_presets.py`);
- one fake CLI (`tests/fake_copilot.py`).

It also makes five small changes outside the adapter. Copilot exposes the gaps they close, and three of them protect Claude and Codex sessions too (Wave 0).

**Extension of #845 (Wave P).** #845 gave sessions Claude Code's four permission modes. Three of them (Manual, Edit automatically, Auto) are the host gate's own verdicts and already work on any CLI. Plan works only on Claude Code. Wave P makes Plan one flow on every CLI by reusing PRD-245 W2's park-and-resume path. Codex gets it first, and Copilot ships with all four modes.

No migration, table, route or dependency is added. The CLI picker needs no change for Copilot, because it renders from the registry (`runtime-section.tsx:76`). Wave P's frontend work is deleting the Plan fallback.

## What changed since the design doc (Copilot CLI 1.0.91)

| Design rev 2 said | 1.0.91 does | Where it says so |
|---|---|---|
| Tier **seed**: "none; `-p` print mode only" | Lifecycle hooks since 0.0.362. `preToolUse` can deny (0.0.396) and answer `ask` (1.0.4). Stop/SubagentStop came in 0.0.401, PermissionRequest in 1.0.16, Notification in 1.0.18, PreCompact in 1.0.5. A `preToolUse` hook that errors or exits non-zero **denies** the call (1.0.57, 1.0.70). Hooks fire in `-p`; repo hooks fire only for a trusted folder (1.0.10, 1.0.40, 1.0.49, launcher code). | bundled `changelog.json`; hooks reference |
| Event vocabulary: none | Configure hooks under **PascalCase** names, and events arrive as Claude/VS Code-shaped snake_case JSON (`hook_event_name`, `session_id`, `tool_name`, `tool_input`, `transcript_path`, `stop_hook_active`). `PreToolUse` uses Claude's tool names (`Bash`, `Read`, `Write`, `Edit`, `Grep`, `Glob`, `WebFetch`, `WebSearch`, `Agent`). Input keys stay Copilot's. | changelog 1.0.6, 1.0.21, 1.0.62; hooks reference |
| Isolation lever: "none needed" | `COPILOT_HOME` moves config **and** state, including the session index. User hooks load from `$COPILOT_HOME/hooks/*.json`. `--config-dir` is deprecated in its favour. | `copilot help environment`; option table; hooks reference |
| Turn end: process exit | `-p` "exits after completion". `Stop` (agentStop) fires at the turn's natural end, including through `task_complete` (1.0.45). `SessionEnd` fires once per completed turn in `-p` (1.0.78). | option table; changelog |
| Session id: — | `--session-id`: "Resume an existing session or task by ID, or set the UUID for a new session" | option table |
| Own-plan login: "GitHub", probe "(?)" | The OAuth device login is kept in the OS keychain. `COPILOT_GITHUB_TOKEN`, then `GH_TOKEN`, then `GITHUB_TOKEN` silently override it. `gh auth token` is the last fallback. The config file holds only the account pointer (`loggedInUsers`, `lastLoggedInUser`), unless `storeTokenPlaintext` is on. | `copilot help environment` and `help config`; docs (authentication); runtime strings |
| Strip `GITHUB_TOKEN` "(?)" | Also `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `COPILOT_ALLOW_ALL` and `COPILOT_PROVIDER_*`. `COPILOT_ALLOW_ALL` set to exactly "true" also trusts the folder and loads its hooks and MCP servers. `COPILOT_PROVIDER_*` turns on BYOK: "GitHub authentication is not required". | `copilot help environment`, `help providers` |
| Ungated (D-3) | Run `-p` **without** `--allow-all-tools`, and a call that needs approval but gets no hook decision is refused: "Permission denied and could not request permission from user". | option table; runtime strings |

## What exists today (main @ 19a16c722)

- **The seam serves two CLIs.** `adapter_for()` serves `claude` and `codex` (`adapters/__init__.py`).
  - A preset row with no adapter is announced as `served: false`, and the claim filter holds its tickets back (`host_capabilities`, `session.py:822`).
  - The parity test (`orchestrator/tests/test_cli_presets_parity.py`) stays red until the host row and the backend row both exist.
- **The turn-end check only covers Claude-style CLIs.** Every CLI runs on a PTY, and its turn ends on the preset's `turn_end` (`_wait_for_turn`, `session.py:652`). `process_exit` exists, but it is built for the seed tier:
  - the `no_session_start` proof (that the gate loaded) runs only for `stop_hook` CLIs;
  - a `process_exit` CLI that exits counts as `completed`, whatever its exit code;
  - its stdout is taken as the answer (`session.py:735`).
- **A file write that names no file is allowed.** `_decide_files` (`policy.py:1331`) allows a `FILE_WRITE` with no paths in `edits` and `auto` mode.
  - This is harmless for Claude, because Edit and Write always carry `file_path`.
  - It is not harmless for a patch-shaped edit, whose paths live inside the patch text. Copilot's `apply_patch` reports as `Edit`; Codex's `apply_patch` hits it whenever `patch_paths` cannot read a header.
- **Most `gh` writes pass as ordinary verbs.** `NEVER_ALLOWED_BASH` (`policy.py:50`) refuses `git push`, `git remote add|set-url` and `gh pr|release create|merge|edit`. Every other `gh` write is an ordinary verb, for example `gh api -X POST`, `gh issue create|comment`, `gh repo create` and `gh workflow run`.
- **The shim cannot take an event name from argv.** It reads the name from the payload only (`hook_shim.py:64`). The design's `event_name_source: argv` (§5) is a preset field with no implementation.
- **Permission modes (#845).** Sessions run as Manual, Edit automatically, Plan or Auto. The mode is set three ways: a workspace default, a per-agent override, and the host's `--permission-mode` flag. The claim carries it (host contract 0.9.0). Plan exists only where the preset has a `plan_stance`, which today means Claude only:
  - The host's `session_mode` (`permission_modes.py`) turns Plan into Edit automatically for any CLI without a plan mode, and for any resumed session.
  - The frontend says so with `PLAN_MODE_CLIS = ['claude']` / `runsPlanAsEdits` ("Plan needs Claude Code", `PermissionModePicker.tsx`).
  - Edits refused in Plan say "present it with ExitPlanMode first", which is Claude's tool.
  - Claude's plan card is an in-turn hold. If nobody answers it within `--ask-timeout` (120 s by default), it is denied, and an expired hold sends the ticket to review.
- **The report's take-over line is Claude-only.** The task report prints `claude --resume <id>` for every session (`services/session_report.py:111`). That is already wrong for Codex.
- **Settings shows only Claude.** Each host line shows only Claude's version (`frontend/components/settings/SessionModeSections.tsx:97`), though hosts announce every CLI under `capabilities.clis`.
- **Session mode is local and single-user.**
  - It runs only in the local edition: `config.py:2107` aborts any other edition that sets `CLI_RUNTIME_ENABLED`.
  - The host's default backend is `http://127.0.0.1:8000` (`cli-host/config.py:38`). `--url` exists, but nothing has been built or tested for a remote backend, with or without TLS.
  - The MCP bridge is reached at that same address.
  - `cli_hosts` has no owning user (`core/models/cli_hosts.py:35`).

  See *Bank PoC*.

## Goals

- **Same runtime as the other CLIs.** A `runtime: cli` agent with provider `copilot` runs each ticket as the operator's own Copilot CLI session, on their own seat. It uses the same gate, approvals, questions, Automatos tools, deliverables and usage booking as Claude Code and Codex.
- **Every permission mode on every CLI.** Manual, Edit automatically, Plan and Auto work on Claude Code, Codex, Copilot and any CLI the adapter pattern adds later. They are set the same way (#845). No CLI falls back from Plan to Edit automatically, and "Plan needs Claude Code" goes away.
- **No ungated run:**
  - no allow flag is ever passed;
  - a missing or timed-out hook ends in Copilot's own refusal;
  - every run proves the gate loaded;
  - the known ways to switch hooks off are refused before spawn.
- **No borrowed credentials.** No API key, PAT or BYOK provider ever reaches a session, and Settings names the login a session will use.
- **Discovered, not listed.** Copilot shows up in the picker and in Settings because a host announced it. Nobody edits a list.
- **Bank PoC.** Copilot works in the local edition, and the central (AKS) setup gets an explicit decision (O1).

## Decisions (adopted in this draft; each reversible in its PR)

- **D1 · Transport: `copilot -p`, on the host's existing PTY.**

  Claude went interactive for billing reasons (PRD-234). Codex went interactive because `codex exec` fires no hooks (design §6.6). Neither reason applies to Copilot:
  - user hooks fire in `-p`;
  - GitHub bills a CLI prompt the same way in every mode. Legacy billing counts premium requests per prompt; current billing counts AI credits by usage. There is no separate `-p` pool like Anthropic's paused `claude -p` change;
  - GitHub documents `-p` and the SDK for programmatic use.

  `-p` is also the safer floor:
  - **No allow flag means a refusal, not a stall.** An approval nobody can answer is refused.
  - **An untrusted folder keeps repo config out.** `-p` then loads no repo hooks, workspace MCP servers or project extensions. The launcher checks `COPILOT_ALLOW_ALL==="true" || GITHUB_COPILOT_PROMPT_MODE_REPO_HOOKS==="true" || folderTrustIsTrusted(...)`, and the session strips both variables.
  - **No dialog waits for a keypress.** There is no TUI prompt (folder trust, rate-limit model switch, update, login) that nobody will answer.

  Interactive mode is used only for the operator's own take-over terminal in the Runtime Canvas.

  **Not chosen:**
  - **ACP** (`copilot --acp`, public preview since 2026-01-28): a second transport in the host for one CLI.
  - **The SDK** (`@github/copilot-sdk`, GA 2026-06-02): a new Node or Python dependency in a stdlib-only host, over a wire protocol GitHub does not publish as a spec.

  Both are recorded as O8.

- **D2 · One `COPILOT_HOME` per agent per host.** This is §6.1's reason again: the session index lives in the home (`session-state/`, `session-store.db`), so a home per ticket would break resume.
  - The operator's `~/.copilot` is never written.
  - The home's `config.json` is rebuilt on every spawn from a **whitelist**, never copied, because this file can hold a plaintext token.

- **D3 · No allow flag, ever; the gate's `allow` is the only lift.**
  - Forbidden: `--allow-all-tools`, `--allow-all`/`--yolo`, `--allow-all-paths`, `--allow-all-urls`, `--allow-tool` and `--assisted-approval`.
  - Stripped: `COPILOT_ALLOW_ALL` and `COPILOT_ASSISTED_APPROVAL`.
  - A missing hook, a crashed shim or a timed-out hook therefore ends in Copilot's own refusal, never in an allowed call.

- **D4 · Hook timeouts sit above our hold.** Copilot's hook timeouts **fail open**: a timed-out hook "lets the tool call proceed through the normal permission flow" (hooks reference).
  - `timeoutSec` is 600 for PreToolUse and permissionRequest.
  - The shim answers by 560 s (`AUTOMATOS_HOOK_WAIT_SECONDS`), and the host's hold ends by `--ask-timeout`.
  - Every other event gets 60.
  - A preset-invariant test pins this order for every preset.

- **D5 · What a session never gets:**

  | Flag | What it keeps out |
  |---|---|
  | `--disable-builtin-mcps` | GitHub's own MCP server. It would let a session write to GitHub with the operator's token, outside the shell gate (§9.3). |
  | `--no-remote` | Remote control from github.com |
  | ~~`--no-auto-login`~~ | Not used (F233, 2 Oct): it switches off the stored login and the `gh` fallback, the only ways a session signs in, and `-p` never waits on a login prompt anyway |
  | `--no-auto-update` and `COPILOT_AUTO_UPDATE=false` | A version change mid-run; the host runs the version it announced |
  | `--no-ask-user` | Questions inside Copilot; they go through `ask_human` (PRD-245 W2) |

  Copilot memory also stays off. It is off by default in `-p`, `--enable-memory` is forbidden, and `memory: false` is set in config.

- **D6 · No token ever passes through us.** The session authenticates in one of two ways:
  - through the operator's keychain login, found via the account pointer we copy;
  - through Copilot's own `gh auth token` fallback, which Copilot runs itself.

  Env tokens and BYOK provider variables are stripped. If the operator's only Copilot login is a plaintext token in `config.json`, the host is refused; the token is never copied.

- **D7 · Automatos tools come in through a file.** The flag is `--additional-mcp-config @<session>/mcp.json` (mode 0600), with the token written in literally.
  - Same file name and same shred at turn end as Claude (`CREDENTIAL_SESSION_FILES`, `session.py:167`).
  - The token never goes into the home, because the home is per agent and would outlive the ticket.

- **D8 · Usage comes from the session record.** The record is `$COPILOT_HOME/session-state/<id>/events.jsonl`.
  - Each `assistant.usage` event carries the model, the token counts (input, output, cache read, cache write, reasoning) and `copilotUsage.totalNanoAiu`.
  - We book tokens and the plan's own AI-credit units, never a price.

- **D9 · All four modes on every CLI; Plan is one flow (Wave P).**

  **Manual, Edit automatically and Auto** are the gate's own verdicts and already CLI-neutral. #845 tests them on Codex in `tests/test_permission_modes.py`. Copilot gets them through its tool mapping (S1.4), with nothing else to build.

  **Plan** runs in four steps on every CLI:
  1. The session explores under the gate's plan verdicts: no edits; only the allowlist's read-only verbs run; anything else is a card.
  2. The plan reaches the operator as the Plan card in the Questions tab, the bell and Telegram, and `plan.md` lands in Deliverables.
  3. If the operator answers while the session is still waiting, the session carries on as Edit automatically in the same turn. That needs a CLI with its own plan tool that the gate can hold, which today means Claude's `ExitPlanMode` (#845).
  4. Otherwise the turn ends with the plan and the ticket parks on it, using PRD-245 W2's park. That covers every CLI without a plan tool, and a Claude card nobody answered inside `--ask-timeout`. The operator then picks one of three answers (PRD-252's review words):
     - **Approve:** resumes the same session as Edit automatically.
     - **Discuss:** resumes it in Plan, with the feedback.
     - **Reject:** sends the ticket to review, with the reason.

  The backend decides each claim's mode from the ticket's plan state (`runtime_ref.plan`). These are deleted:
  - the host's two shortcuts: a CLI with no plan mode runs Edit automatically, and a resumed session runs Edit automatically;
  - `PLAN_MODE_CLIS`, `runsPlanAsEdits` and the "Plan needs Claude Code" note.

  Copilot's own `--plan` is not used. In this flow the host gate holds the session read-only on every CLI, and a CLI's plan mode is used only where its plan tool can be held in-turn.

## Stories

### Wave 0 — gaps Copilot exposes, closed for every CLI (one PR)

**S0.1 · A file write with no file named is refused (S)**
In `_decide_files`, a `FILE_WRITE` with empty `intent.paths` now returns `deny`, with the reason "a write that names no file — the gate cannot tell where it lands". Reads and searches with no path still work in the cwd, as the existing branch intends.
**Files:** `services/cli-host/automatos_cli_host/policy.py`; `tests/test_units.py`, `tests/test_adapters.py`.
**Acceptance:**
- [ ] A path-less `FILE_WRITE` is denied in manual, edits, plan and auto modes.
- [ ] A Codex `apply_patch` with no `*** Add|Update|Delete File:` header is denied. A patch with headers inside the roots is still allowed, as before.
- [ ] `Grep` and `Glob` with no path still allow in the cwd. The rest of the policy suite is unchanged.
**Editions:** local only (the host).

**S0.2 · The gate proof applies to every gated CLI, however its turn ends (S)**
`_wait_for_turn` now runs the `no_session_start` check for every preset whose tier has hooks (`native`, `hooks`, `proxy`), not just `stop_hook` ones.

For a gated `process_exit` CLI, `_collect` (`session.py:723`) reads the exit like this:
- exit 0 after both `SessionStart` and `Stop` → `completed`;
- no `SessionStart` → `error`, `exit_reason="ungated_exit"`, empty result text. The message: "<CLI> ran without Automatos' gate — no SessionStart hook arrived; nothing it produced is reported";
- a non-zero exit, or no `Stop` → `exited_before_stop`, with the tail;
- still running 30 s after `SessionEnd` → terminated; the turn still counts.

The seed tier keeps "stdout is the answer".

A new optional preset field, `startup_timeout_seconds`, overrides the host's 180 s. Copilot's is 30 s, because `-p` shows no dialogs.
**Files:** `presets.py`, `session.py`; `tests/test_units.py`, plus a fake-CLI case in S1.7.
**Acceptance:**
- [ ] A gated `process_exit` fake that never fires SessionStart ends as `error/ungated_exit`, and its stdout is not used as the ticket's result.
- [ ] Exit 3 after Stop → error with the tail. Exit 0 after SessionStart and Stop → success.
- [ ] The Claude and Codex suites pass untouched.
**Editions:** local only.

**S0.3 · Sessions never publish through `gh` either (S)**
`NEVER_ALLOWED_BASH` gains the `gh` write forms:

| Group | Refused forms |
|---|---|
| `gh issue`, `gh pr` | `create`, `comment`, `edit`, `close`, `reopen`, `merge`, `review`, `ready`, `lock` |
| `gh release` | `create`, `upload`, `edit`, `delete` |
| `gh repo` | `create`, `edit`, `delete`, `fork`, `rename`, `archive`, `sync` |
| `gh workflow` | `run`, `enable`, `disable` |
| `gh run` | `rerun`, `cancel` |
| `gh secret`, `gh variable` | `set`, `delete` |
| `gh gist` | `create`, `edit`, `delete` |
| `gh api` | `-X`/`--method` other than `GET`, or any of `-f`, `-F`, `--field`, `--raw-field`, `--input` (each makes `gh api` POST) |

These are matched the same way as `git push`: on the raw line, on every simple command, and with wrappers peeled off. Read forms (`gh pr view|list|diff|checks`, `gh issue view|list`, `gh api` GET) keep today's verdict.

The session rules already say this (`session.py:118`: "never push, publish or open pull requests"). Copilot is GitHub's own agent and reaches for `gh` first.
**Files:** `policy.py`; `tests/test_units.py`.
**Acceptance:**
- [ ] Every row is denied, including `xargs gh issue create …`, `env X=1 gh api -X POST …` and `gh api repos/a/b -f x=1`.
- [ ] `gh pr view 1` and `gh api repos/a/b` keep today's verdict.
**Editions:** local only.

**S0.4 · The shim takes the event name from argv when the payload has none (S)**
`hook_shim.main` accepts `--event <BusName>`. If the payload has no `hook_event_name`, the shim uses the argv name; if the payload has one, the payload wins. This implements §5's `event_name_source: argv`.
- Copilot's `permissionRequest` and `notification` exist only in camelCase, and their payloads carry no snake_case event name.
- For `copilot`, the offline deny comes in two shapes: the Claude shape for PreToolUse, and the shape Copilot reads for `permissionRequest` (verify at build).
- No new imports, so start-up cost does not change.
**Files:** `hook_shim.py`; `tests/test_hooks_roundtrip.py`.
**Acceptance:**
- [ ] A payload with no name, sent with `--event PermissionRequest`, reaches the host as PermissionRequest. With the host down, the shim writes the deny.
- [ ] A payload that carries a name ignores argv.

**S0.5 · The take-over line comes from the registry (S)**
`CliPresetInfo` gains a `takeover` template:
- claude: `claude --resume {session_id}`
- codex: `codex resume {session_id}`
- copilot: `copilot --resume={session_id}`

`session_report.py:111` renders the template for the ticket's provider. For CLIs whose home is per agent (codex, copilot), the line adds: "or open the ticket's Runtime Canvas terminal, which starts it in the agent's own home".
**Files:** `orchestrator/core/cli_presets.py`, `orchestrator/services/session_report.py`; `tests/test_prd234_s2_session_report.py`.
**Acceptance:**
- [ ] A Codex ticket's report prints `codex resume <id>`.
- [ ] Claude's line is unchanged.
- [ ] An unknown provider prints no take-over line.
**Editions:** both. The report renders in both editions; only local runs sessions.

### Wave P — every permission mode on every CLI (one PR; does not depend on Copilot)

**SP.1 · The ticket's plan state decides each claim's mode (S, backend)**
Add `runtime_ref.plan = {state, version, text_path, feedback, decided_at}`, where `state` is one of `planning`, `awaiting`, `approved`, `discussing`. `_claim_ref` carries it across claims, the way it already carries `session_asks`.

`core/session_permission_modes.py` gains `claim_permission_mode(ticket_mode, plan_state)`, which `_claim_one` passes to `_claim_payload`:
- When the ticket's mode (agent override, else workspace default) is `plan`, the claim sends `plan` until the state is `approved`, then `edits`.
- Every other mode passes through unchanged.

This replaces the host's guess "a resumed session already presented its plan": the backend knows whether it did. An in-turn approval reported by the host (SP.2) sets `approved` as well.
**Files:** `orchestrator/core/session_permission_modes.py`, `orchestrator/services/cli_host_service.py` (`_claim_ref`, `_claim_one`); `orchestrator/tests/test_session_permission_modes.py`.
**Acceptance:**
- [ ] A Plan agent's first claim sends `plan`; after Approve it sends `edits`; after Discuss it sends `plan` again.
- [ ] The plan state survives a claim.
- [ ] Manual, Edit automatically and Auto agents are unchanged.
- [ ] `claim_for_host` carries the mode end to end, as the #845 test does.
**Editions:** local only.

**SP.2 · The host runs Plan the same way on every CLI (S, host)**
`permission_modes.session_mode` loses its `resuming` and `can_plan` fallbacks, because the claim now decides (the host's `--permission-mode` override still wins). In `plan`:
- **Edits** are refused. The reason says how *this* CLI presents a plan: `ExitPlanMode` for a preset with a `plan_stance`; "your final message, then end your turn" for every other preset. Today's text names Claude's tool for every CLI.
- **Shell commands:** only `PLAN_BASH_ALLOW` runs, the read-only part of the allowlist:
  - allowed: `git status|diff|log|show|branch|ls-files|rev-parse|blame|describe|shortlog|remote -v|worktree list|stash list`, `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `rg`, `find` (its `-exec`/`-delete` are already judged), `pwd`, `which`, the text filters except where they write (`sed -i` and the sed/awk file-writing forms are cards), `jq`, `tree`, `du`, `stat`, `diff`;
  - a card: everything else, including `git add|commit|stash|restore|checkout -b|switch -c`, test runners and `npm run`.

  A CLI without its own plan mode has nothing else holding it read-only. The backend's rendering copy (`SESSION_BASH_VERBS`) gains the subset, kept in step by `test_cli_presets_parity.py`.
- **The ticket file** gains a Plan section: explore read-only, make no changes, end with the plan as your final message (or present it with `ExitPlanMode` in Claude Code).
- **At turn end,** a plan-mode outcome carries `plan: {text, approved_in_turn}`. The text is the `ExitPlanMode` text when one was presented, otherwise the final message. It is saved as `plan.md` with the existing `save_plan`.

**Claude, in-turn:** an `ExitPlanMode` card the operator approves inside `--ask-timeout` works exactly as in #845 and reports `approved_in_turn: true`.

**Claude, expired:** an `ExitPlanMode` card that expires is answered "your plan is with the operator; end your turn now; you resume when they approve". It is recorded as `refused`, not a `hold`, so it does not force review. The plan then parks like everyone else's (SP.3). This is the one behaviour change to #845: today an unanswered plan card sends the ticket to review.
**Files:** `services/cli-host/automatos_cli_host/permission_modes.py`, `policy.py` (`PLAN_BASH_ALLOW`, the per-CLI reason), `session.py` (the Plan section, the outcome's `plan`), `orchestrator/core/cli_presets.py` (the verbs mirror); `tests/test_permission_modes.py`, `tests/test_session_fake_codex.py`, `orchestrator/tests/test_cli_presets_parity.py`.
**Acceptance:**
- [ ] A fake Codex in `plan`: an edit is denied with the final-message wording; `git log` runs; `git commit` and `pytest` are cards; the outcome's plan is the final message and `plan.md` is saved.
- [ ] Claude, in-turn approval: unchanged (the #845 tests pass untouched).
- [ ] Claude, expired card: the turn ends, the outcome carries the plan, and no hold denial is recorded.
**Editions:** local only.

**SP.3 · A plan parks the ticket on the Plan card, and the answer resumes the same session (M, backend)**
**When a plan arrives:** `apply_result` receives an outcome with a `plan` that was not approved in-turn. It then:
- sets `plan.state = awaiting` and moves `version` on;
- registers `plan.md` as a Deliverable;
- files the Plan card through PRD-225's shared `stage_question`: subject = this ticket, question "Approve the plan for ticket #N?", options Approve / Discuss / Reject, `details={"cli_plan": {task_id, version}}`;
- parks the ticket `blocked` ("Waiting for your approval of its plan"), through `_park_for_answer`'s path, so `_mark_resumable` keeps the session id.

The Questions tab, the bell and Telegram need nothing new.

**When the operator answers:** the answer route (`api/approval_grants.py`, beside the `cli_ask` and `cli_permission` branches) calls `answer_session_plan`:

| Answer | State | Ticket |
|---|---|---|
| **Approve** | `approved` | `assigned` + resume |
| **Discuss** (with the operator's text) | `discussing` + feedback | `assigned` + resume |
| **Reject** | — | `review` with "plan rejected: <text>" |

A card for an older plan version is refused as stale.

**On the next claim,** `_plan_fold_in(task)` (beside `_answers_fold_in`, and marked folded the same way) puts one of two sections in the prompt:
- "## Your plan was approved — implement it now", with the plan;
- "## Feedback on your plan — revise it and present it again", with the feedback.

**Limits:** Plan cards do not count against `MAX_ASKS_PER_TICKET`. A ticket on its sixth plan version goes to review instead (O6).
**Files:** `orchestrator/services/cli_host_service.py` (`apply_result`, `answer_session_plan`, `_plan_fold_in`, `_ticket_prompt`), `orchestrator/api/approval_grants.py`; `orchestrator/tests/test_session_plan_turn.py` (new, real DB, seeded like `test_prd234_s1a_cli_hosts_realdb.py`).
**Acceptance:**
- [ ] A result carrying a plan → `blocked`, one question row, one `plan.md` Deliverable.
- [ ] Approve → `assigned`, with the resume id. The next claim's mode is `edits`, and its prompt carries the approved plan.
- [ ] Discuss → the next claim's mode is `plan`, and its prompt carries the feedback.
- [ ] Reject → `review`, with the reason.
- [ ] A Telegram answer reaches the same path. A re-POSTed result files no second card.
**Editions:** local only.

**SP.4 · Every CLI offers every mode (S, frontend + docs)**
- **Delete:** `PLAN_MODE_CLIS`, `runsPlanAsEdits` and the "Plan needs Claude Code" note (`PermissionModePicker.tsx`). The Plan description stays the same for every CLI: "Explores and presents a plan; edits start once you approve it."
- **The ticket's `SessionBlock`:** while `plan.state` is `awaiting`, it shows "Waiting for your approval of its plan", with a link to `plan.md`.
- **Tests:** the vitest that pinned `PLAN_MODE_CLIS` to the host presets becomes "every registry CLI offers all four modes".
- **Docs:** the self-hosting guide's Permission modes section and the cli-host README describe Plan on every CLI.
**Files:** `frontend/components/settings/PermissionModePicker.tsx`, `frontend/components/agents/runtime-section.tsx`, the `SessionBlock` component, `frontend/components/settings/__tests__/permission-mode-picker.test.tsx`, `docs/getting-started/self-hosting.md`, `services/cli-host/README.md`.
**Acceptance:**
- [ ] vitest, `tsc`, lint and the changed-lines checks are green.
- [ ] The owner sees no fallback note on a Codex agent's form.
**Editions:** local only (`isLocal`).

**As built (Wave P, 2026-10-02).** Where the build differs from the stories above, and why:
- **The plan travels as an event, not on the result.** `apply_result` is 114 lines, and touching it
  would fail the changed-lines function-length check. The host sends `PlanReady` in the turn's final
  event flush, and `record_events` files the card (`services/session_plans.py`). The result's
  `_park_for_answer` then parks on it. The host now holds a session's result until its last events
  are accepted: a failed final flush used to be dropped.
- **The ledger is `runtime_ref.session_plans`**, one entry per plan version, not a single
  `runtime_ref.plan`. The state is read from the latest entry (`plan_state`).
- **The card offers Approve and Reject.** The operator's own words are the third answer.
  A *Discuss* button would send the word "Discuss" with no feedback, because the Questions
  tab submits a chip as the answer.
- **`SESSION_BASH_VERBS` does not gain the Plan subset.** The system prompt is stable per agent
  (the prompt-cache invariant). A Plan turn's rules are in the ticket file, and the gate enforces
  `PLAN_BASH_ALLOW`.
- **The §10 design-doc note on permission modes** (S3.2) landed with Wave P, which it describes.
- **policy.py was split** (`shell_text.py`, unchanged code) so Wave P does not grow it past main's size.

### Wave 1 — Copilot runs, gated (two PRs: the rows, then the adapter)

**S1.1 · The preset row and the registry row (S)**

```python
COPILOT = CliPreset(
    id="copilot", label="GitHub Copilot", binary="copilot",
    tier=TIER_HOOKS, turn_end=TURN_END_PROCESS_EXIT,          # -p exits after the turn; S0.2 proves the gate
    startup_timeout_seconds=30,
    model_flag="--model",
    session_id_flag="--session-id",                           # a new session with the backend's pre-assigned uuid
    resume_flag="--resume",                                   # an unknown id fails; --session-id would silently start fresh
    add_dir_flag="--add-dir",
    mcp_config_flag="--additional-mcp-config",                # value: "@<session>/mcp.json"
    worktree_args=("--worktree",), worktree_takes_name=True,  # resume interplay: verify (S1.3)
    system_prompt_flag=None,                                  # the soul rides UserPromptSubmit → additionalContext (§6.9)
    initial_prompt=PROMPT_FLAG, initial_prompt_flag="-p",
    name_flag="--name",
    ungated_stance=(),                                        # nothing: our hook's allow is the only lift (D3)
    plan_stance=(),                                           # Plan = the plan turn (D9, Wave P); Copilot's own --plan is not used
    required_args=("--no-ask-user", "--disable-builtin-mcps", "--no-remote", "--no-auto-update"),   # never --no-auto-login (F233)
    hook_events=BUS_EVENTS - {"PostCompact"},
    hook_timeouts={"*": 60, "PreToolUse": 600, "PermissionRequest": 600},   # SECONDS (timeoutSec); timeouts fail OPEN
    config_home_env="COPILOT_HOME", config_home_scope=SCOPE_PER_AGENT,
    strip_env=frozenset({"COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "COPILOT_ALLOW_ALL",
                         "COPILOT_ASSISTED_APPROVAL", "COPILOT_MODEL", "COPILOT_OFFLINE", "COPILOT_HOOK_ALLOW_LOCALHOST",
                         "GITHUB_COPILOT_PROMPT_MODE_REPO_HOOKS", "GITHUB_COPILOT_PROMPT_MODE_WORKSPACE_MCP",
                         "GITHUB_COPILOT_PROMPT_MODE_EXTENSIONS", "COPILOT_CLI"}),
    strip_env_prefixes=("COPILOT_PROVIDER_",),                # BYOK never reaches a session
    keep_env=frozenset({"GH_HOST", "COPILOT_GH_HOST", "COPILOT_PROXY_KERBEROS_SPN"}),
    extra_env={"COPILOT_AUTO_UPDATE": "false"},
    forbidden_args=("--allow-all-tools", "--allow-all", "--yolo", "--allow-all-paths", "--allow-all-urls",
                    "--allow-tool", "--assisted-approval", "--enable-memory", "--config-dir", "--share-gist",
                    "--remote", "--remote-export", "--cloud", "--connect", "--acp", "--server", "--headless",
                    "-i", "--interactive", "--continue", "--mcp-github-auth"),
    auth_probe=AuthProbe(kind="copilot_login", code="copilot_not_logged_in",
        refusal="GitHub Copilot is not logged in on this machine. Run `copilot login` (or `gh auth login` with an account that has a Copilot seat), then retry."),
    install_hint="GitHub Copilot CLI is not installed on this machine (no `copilot` on your PATH). Install it (`brew install copilot-cli`, or `npm install -g @github/copilot`) and run `copilot login`.",
    docs_url="https://docs.github.com/copilot/concepts/agents/about-copilot-cli",
)
```

**The backend row:**
- `PROVIDER_COPILOT = "copilot"`, label "GitHub Copilot", usage slug `copilot_cli`.
- The model rule is permissive, as for Codex (D-2): `^[a-z0-9][a-z0-9.\-]*$`. `auto`, `claude-sonnet-4.6` and `gpt-5.4` pass; `openai/gpt-5` and `GPT 5` fail.
- The hint: "A model your Copilot plan and your organisation's policy enable in Copilot CLI (for example auto, claude-sonnet-4.6 or gpt-5.4). Blank = the CLI's default."

Until S1.2 lands, the host announces `copilot: served:false` with the reason. Codex did the same (`presets.py:190`).
**Files:** `presets.py`, `orchestrator/core/cli_presets.py`; `tests/test_adapters.py` (preset invariants), `orchestrator/tests/test_cli_presets_parity.py`, `orchestrator/tests/test_prd234_s1a_cli_runtime.py`, the `providerOptions` vitest.
**Acceptance:**
- [ ] The preset invariants cover copilot:
  - no forbidden arg is ever in the launch argv;
  - every required arg always is;
  - `build_session_env` drops every stripped name and every `COPILOT_PROVIDER_*`;
  - `GH_HOST` survives.
- [ ] For **every** preset, the held-event timeout > the shim's wait (560) > the host's longest hold (D4).
- [ ] `validate_runtime_configuration` accepts and refuses the model examples above.
- [ ] The analytics label for `copilot_cli` is "GitHub Copilot".
- [ ] The picker lists GitHub Copilot, marked not served until a host announces it, with no frontend change.

**S1.2 · The per-agent home: config, hooks, login (M)**
`CopilotAdapter.prepare()` builds the home and returns `{"COPILOT_HOME": …}`:

- **The home:** `<state>/agents/<agent_id>/.copilot`, mode 0700.
- **`config.json`** (mode 0600), rebuilt on every spawn. It reads the operator's `~/.copilot/config.json` as JSON and takes **only** these keys:
  - `loggedInUsers` and `lastLoggedInUser`: which account (host and login, never a token);
  - `includeCoAuthoredBy`, `proxyUrl`, `proxyKerberosServicePrincipal`.

  It then sets these fixed values:

  | Key | Value | Why |
  |---|---|---|
  | `trustedFolders` | `[]` | In `-p`, an untrusted folder keeps repo hooks, workspace MCP servers and project extensions out (D1). This also undoes any "remember this folder" given in the take-over terminal. |
  | `banner` | `"never"` | |
  | `autoUpdate` | `false` | |
  | `memory` | `false` | |
  | `showTipsOnStartup` | `false` | |
  | `ide` | `{autoConnect: false}` | |
  | `continueOnAutoMode` | `false` | A rate limit never silently swaps the agent's model. |
  | `storeTokenPlaintext` | `false` | |

  No other key is ever copied or logged.
- **`hooks/automatos.json`** (mode 0600) is the only hooks file. It contains:
  - every bus event Copilot has, under its **PascalCase** name, in Claude's nested shape: `{"hooks": [{"type": "command", "command": <hook_command()>, "timeoutSec": N}]}`. This makes Copilot send the Claude-compatible payload;
  - `permissionRequest` and `notification`, which are camelCase only, with `--event` on the command (S0.4).
- **No `mcp-config.json`** is placed in the home, so the operator's MCP servers never come along (PRD-245 D9).

**Login (`logged_in()`)** never reads a credential. It tries three routes in order:
1. **Keychain.** The operator's config has `lastLoggedInUser` and no plaintext token: `storeTokenPlaintext` is not true, and there is no token-bearing key (the exact key is confirmed at build; the probe checks only whether the key exists). → served, logging in from the keychain through the copied pointer.
2. **`gh`.** Otherwise, if `gh auth status --hostname <host>` reports a logged-in account → served. Copilot runs `gh auth token` itself, so the token never passes through the host. `--show-token` is never passed.
3. **Refuse.**

`detect()` adds `login` (`<login>@<host>`) and `login_route` (`keychain` or `gh`). Neither is secret. Settings shows both, so the operator sees whose seat will run the tickets.
**Files:** `adapters/copilot.py` (new), `adapters/__init__.py`; `tests/test_adapters.py`.
**Acceptance:**
- [ ] After a session, the (fake) operator `~/.copilot/` is byte-identical.
- [ ] The seeded config holds exactly the whitelisted and fixed keys.
- [ ] A fixture config whose token-bearing key holds a marker string leaves that marker nowhere: not in the agent home, `host.log`, the result or the events.
- [ ] Two tickets of one agent share the home.
- [ ] A hooks file from an earlier spawn is replaced, not appended.
- [ ] `trustedFolders` is `[]` again after the operator trusted the folder from the Canvas terminal.
- [ ] Route 1, route 2 and the refusal each yield the right `served`, `reason` and `login_route`.
- [ ] An operator whose only login is an env token is refused, with a sentence saying sessions never carry tokens.

**S1.3 · Launch (S)**
A new session:
`copilot -p "<pointer>" --session-id <uuid> --add-dir <session dir> [--add-dir <deliverables folder>] [--model M] [--name "automatos #N"] [--worktree automatos-N] [--additional-mcp-config @<session>/mcp.json] --no-ask-user --disable-builtin-mcps --no-remote --no-auto-update`

A resumed session uses `--resume <cli_session_id>` in place of `--session-id`.

- **PTY:** it spawns on the host's PTY like every CLI. The drain and the terminal log are unchanged.
- **`--add-dir`:** passed for every folder the gate grants: the session dir, and the deliverables folder when it is not the cwd. Copilot's own path check ("cwd + temp") then sees the same folders.
- **The prompt:** the pointer is the same short line every CLI gets. The soul and the ticket ride `UserPromptSubmit` → `additionalContext`, which Copilot puts into the model-facing prompt (1.0.65).
- **Telemetry:** `OTEL_*` is kept for Copilot's own export (O1). `OTEL_EXPORTER_OTLP_HEADERS`, when set, is also named in `--secret-env-vars`, so session shells never see its value.
- **Resume:** if verify shows `--worktree` cannot combine with `--resume`, set `worktree_excludes_resume=True`. If `--name` fails on resume, the adapter drops it there.
**Files:** `adapters/copilot.py`; `tests/test_adapters.py`.
**Acceptance:**
- [ ] Golden argv for the new, resumed, worktree, model and Automatos-tools cases.
- [ ] `assert_args_honour_invariant` and `assert_secret_not_in_args` pass.
- [ ] No allow flag appears in any case.

**S1.4 · Tools: what a Copilot call does (M)**
PascalCase payloads use Claude's tool names, but Copilot's input keys are not remapped (hooks reference). The mapping therefore reuses the Claude adapter's path keys (`file_path`, `notebook_path`, `path`) and the Codex adapter's `patch_paths`.

| `tool_name` in the payload (Copilot tool) | Input | ToolIntent |
|---|---|---|
| `Bash` (`bash`, `powershell`) | `command` | SHELL |
| `Read` (`view`) | `path` | FILE_READ |
| `Write` (`create`) | `path` | FILE_WRITE |
| `Edit` (`edit`, `str_replace_editor`, `apply_patch`) | `path`; or patch text in `input`/`patch` → `patch_paths()`; or `command: "view"` → a read | FILE_WRITE (FILE_READ for view) |
| `Grep` (`grep`, `rg`), `Glob` (`glob`) | `pattern`, `path`, `glob` | FILE_READ, with globs |
| `WebFetch`, `WebSearch` | `url`, `query` | WEB |
| `TodoWrite` (`update_todo`); `AskUserQuestion` (`ask_user`, not offered under `--no-ask-user`) | — | BENIGN |
| `read_bash`, `stop_bash`, `list_bash`, `read_powershell`, `report_intent`, `task_complete`, `fetch_copilot_cli_documentation` | — | BENIGN |
| `Agent` / `Task` (`task`) | — | UNKNOWN → deny (same as Claude today) |
| The Automatos server's tools, spelled any of `mcp__automatos__x`, `automatos-x`, `automatos/x`, `automatos__x`, `automatos.x` | — | PLATFORM (the name must be on the ticket's list) |
| Anything else, including Copilot's built-in Computer Use, workflow and background-agent tools, and `exit_plan_mode` (Copilot's own plan mode is not used: D9) | — | UNKNOWN → deny |

Native lowercase names map the same way, in case a payload carries them.

**PermissionRequest.** Verify at build whether Copilot raises `permissionRequest` after a PreToolUse `allow`, for example on a path or URL check.
- If it does, the preset gets `permission_request="rejudge"`, and the host answers with the gate's verdict on the same call: allow or deny, never ask.
- If not, the existing deny stays ("sessions are policy-gated, not prompted").
**Files:** `adapters/copilot.py`, `presets.py` (the field, if needed), `session.py` (`_reply_for`, if needed); `tests/test_adapters.py`, `tests/test_units.py`.
**Acceptance:**
- [ ] Recorded payload shapes map as in the table.
- [ ] An `Edit` whose patch names `../../.ssh/authorized_keys` is denied.
- [ ] A `view` of `~/.aws/credentials` is denied by the secret guard.
- [ ] `github-mcp-server/create_issue` and `mcp__other__x` are denied.
- [ ] Re-running the existing policy suite through the Copilot adapter gives the same verdicts for the shared tools (design §11, policy neutrality).

**S1.5 · The record: result and usage (S)**
**Where the record is:** `$COPILOT_HOME/session-state/<id>/events.jsonl`, or the payload's `transcript_path` when it carries one.

**`read_usage`** adds up the `assistant.usage` events per model:

| Copilot field | Booked as |
|---|---|
| `inputTokens` | `input_tokens` |
| `outputTokens` | `output_tokens` |
| `cacheReadTokens` | `cache_read_input_tokens` |
| `cacheWriteTokens` | `cache_creation_input_tokens` |
| `reasoningTokens` | `reasoning_output_tokens` (additional) |
| `copilotUsage.totalNanoAiu` | `ai_credits` (additional; unit confirmed at build) |
| `session.shutdown.totalPremiumRequests` | `premium_requests` (additional; legacy billing) |

It never books a price. The resume snapshot on SessionStart (`usage_delta`) does not change.

**`last_text`** takes the first of: Stop's `last_assistant_message`, the record's last `assistant.message`, the stdout tail.
**Files:** `adapters/copilot.py`; `tests/test_adapters.py`.
**Acceptance:**
- [ ] A two-model fixture books two `per_model` buckets with the right totals.
- [ ] A resumed run reports only its own calls.
- [ ] The `llm_usage` row has `provider=copilot_cli`, `billing=subscription` and no cost.

**S1.6 · Preflight refusals and limits (S)**

| Code | When | The sentence says |
|---|---|---|
| `copilot_missing` | No binary | The install hint |
| `copilot_too_old` | Below the floor confirmed at build: at least 1.0.70. Before 1.0.57 a crashing preToolUse hook **allowed** the call, and 1.0.70 made exit 2 deny. | Update with `copilot update` |
| `copilot_not_logged_in` | No route in S1.2 | Run `copilot login`, or `gh auth login` |
| `copilot_plaintext_token` | The Copilot login is a plaintext token in config.json, and `gh` is not logged in | Turn on the OS keychain (on Linux/WSL2, a Secret Service such as gnome-keyring) and run `copilot login` again; Automatos never copies a credential |
| `copilot_managed_hooks_only` | A managed policy file (`/etc/github-copilot/policy.d/*.json`) sets `allowManagedHooksOnly` | Your organisation's Copilot policy runs only administrator-deployed hooks, so Automatos' gate cannot load (O4) |
| `copilot_hooks_disabled_here` | Per ticket: `.github/copilot/settings.json` or `settings.local.json` in the ticket's folder or git root sets `disableAllHooks` | That repository switches every hook off |

**Limits (F083).** `usage_limit` learns Copilot's limit sentences, taken from the binary:
- "You've run out of your AI credits";
- "…your included AI credits for the month";
- "You've hit your rate limit";
- "…the rate limit for this model";
- "…your session rate limit";
- "You've reached your weekly rate limit".

Each one pauses the host for Copilot and puts the ticket back in the queue. "No model available. Check policy enablement under GitHub Settings > Copilot" is an error with that sentence: it comes from policy, so it is not a pause.
**Files:** `adapters/copilot.py`, `usage_limit.py`, `session.py` (the per-ticket check); `tests/test_usage_limit.py`, `tests/test_adapters.py`.
**Acceptance:**
- [ ] Each code is produced by its fixture.
- [ ] The six limit sentences pause.
- [ ] The model-policy sentence errors.
- [ ] The Claude and Codex limit tests pass unchanged.

**S1.7 · `fake_copilot` and the end-to-end test (M)**
`tests/fake_copilot.py` mirrors `fake_codex.py`. The fake:
- exits 64 on any forbidden arg or when `-p` is missing;
- requires `COPILOT_HOME`;
- loads `$COPILOT_HOME/hooks/*.json`;
- fires the PascalCase events with snake_case payloads and Claude tool names, plus camelCase `permissionRequest`;
- honours a deny;
- treats a hook timeout as **fail-open**, like the real CLI, so the test proves the shim answers first;
- writes `session-state/<id>/events.jsonl` (`session.start`, `assistant.usage`, `assistant.message`, `session.shutdown`);
- follows `--session-id` new versus existing, and exits non-zero on `--resume` with an unknown id;
- exits 0 after Stop and SessionEnd.

`tests/test_session_fake_copilot.py` runs the whole turn: claim → gated calls → result, model, usage, files touched and denials → `done`.
**Acceptance:**
- [ ] Green in the `cli-host-tests` lane.
- [ ] All four modes, through the Copilot adapter:
  - Manual: an edit is a card;
  - Edit automatically: an edit runs and an unlisted command is a card;
  - Auto: both run;
  - Plan: an edit is refused, the outcome carries the plan, and an Approve resumes the same session id as Edit automatically (Wave P).
- [ ] A fake started with its hooks file ignored ends as `ungated_exit`.
- [ ] The operator's home is untouched.
- [ ] A resume by the same agent continues the session id.

### Wave 2 — Automatos tools and the sandbox

**S2.1 · The MCP bridge entry (S)**
`<session>/mcp.json` (mode 0600) is `{"mcpServers": {"automatos": {"type": "http", "url": …, "headers": {"Authorization": "Bearer …"}, "tools": ["*"]}}}`. It is passed as `--additional-mcp-config @<path>`.
- The token is written literally (D7), and the file is shredded at turn end.
- `--disable-builtin-mcps` keeps GitHub's server out.
- The home has no `mcp-config.json`, so the operator's servers stay out too.

**When the organisation blocks the server.** If the organisation's MCP policy is registry-only and the Automatos server is not on it, Copilot refuses the server. The host reads this from the session record (`session.mcp_servers_loaded` / `session.mcp_server_status_changed`). The ticket then says: "your organisation's Copilot MCP policy blocked the Automatos tools — this session ran without them".
**Files:** `adapters/copilot.py`, `session.py` (the blocked-server note); `tests/test_adapters.py`, `tests/test_session_fake_copilot.py`.
**Acceptance:**
- [ ] The fake receives the flag and the file. The file is mode 0600, and gone after the turn.
- [ ] The token is nowhere else.
- [ ] The blocked case puts the sentence on the ticket.

**S2.2 · Copilot's command sandbox under the session (M) — subject to O3**
When the sandbox is on:
- **The launch adds `--sandbox`.** The flag needs `experimental: true` in the seeded config, unless a managed policy forces sandboxing.
- **A seeded `$COPILOT_HOME/settings.json`** gets a `sandbox` block built from `sandbox.py`'s values:

| Setting | Value |
|---|---|
| `enabled`, `addCurrentWorkingDirectory` | true |
| `readwritePaths` | The session dir and the deliverables folder |
| `deniedPaths` | `CREDENTIAL_PATHS` (`sandbox.py:47`), the platform's secret roots and the host's state dir |
| `allowBypass` | false (no per-command escape hatch) |
| `auth.git`, `auth.gh` | false (no credentials in a session, as the Claude sandbox denies `~/.config/gh`) |
| `userPolicy.seatbelt.keychainAccess` | false |
| `sandboxMcpServers` | true |
| network | Outbound only to `DEFAULT_ALLOWED_DOMAINS` and `--session-allow-domain`, if Copilot's host rules can express that (verify). Otherwise `allowOutbound: true` and `allowLocalNetwork: false`, and the ticket says so. |

**A host that cannot sandbox is refused** with `copilot_sandbox_unavailable`. The refusal names Copilot's own prerequisites:
- macOS: `sandbox-exec`;
- Linux: `bwrap` 0.5 or later, `slirp4netns`, util-linux 2.35 or later, iptables and `/dev/net/tun`.

`--no-session-sandbox` turns all of this off on an isolated host, as it does for Claude.
**Files:** `adapters/copilot.py`, `sandbox.py` (shared constants); `tests/test_session_sandbox.py`, `tests/test_adapters.py`.
**Acceptance:**
- [ ] The seeded settings match the golden file.
- [ ] A missing prerequisite (fake probe) is refused with the sentence.
- [ ] `--no-session-sandbox` drops both `--sandbox` and the block.

### Wave 3 — the operator sees it

**S3.1 · Settings lists every CLI a host announces (S, frontend)**
Each host line (`SessionModeSections.tsx:97`) renders every `capabilities.clis.<id>`:
- the label from the registry;
- the version;
- served, or the reason it is not;
- for Copilot, the `login`.

The ticket's `SessionBlock` shows `ai_credits` and `premium_requests` when present ("plan usage, no cost").
**Files:** `frontend/components/settings/SessionModeSections.tsx`, the `SessionBlock` component, their vitest files.
**Acceptance:**
- [ ] vitest renders three CLIs, including one not-served reason and a Copilot login.
- [ ] `tsc`, lint and the changed-lines checks are green.
- [ ] The owner checks it in the local edition.
**Editions:** local only (`isLocal`).

**S3.2 · Docs (S)**
**The design doc:**
- Correct the §3.1 Copilot row: hooks tier, PascalCase identity bus, per-agent `COPILOT_HOME`, `-p` with exit as the turn end, login by keychain or `gh`.
- Correct the §9.1 and §9.2 rows.
- Annotate §12 D-3: it now applies to Cursor only.
- In §14, give Copilot its own wave after Codex.
- Add to §10 step 1: "classify from the vendor's current binary, not a reference implementation". Copilot stayed misclassified for three weeks because the row was read from munder's code.
- Add to §10: "Permission modes need nothing per CLI. Manual, Edit automatically and Auto are the gate's verdicts on `ToolIntent`. Plan is the plan turn (Wave P). Give a preset a `plan_stance` only if its plan tool can be held in-turn and an approval continues the same turn, as Claude's `ExitPlanMode` does."

**The cli-host README:** how to install, `copilot login`, the refusal table, and what an organisation admin must allow.
**Files:** `docs/architecture/CLI-RUNTIME-ADAPTER-DESIGN.md`, `services/cli-host/README.md`.

**As built (W1–W3, 2026-10-02).** One PR, stacked on Wave P. Where the build differs from the stories above, and why:
- **The hooks file is Claude's own format.** Copilot 1.0.91 reads a Claude-format file in
  `$COPILOT_HOME/hooks/` unmodified and answers it with Claude's snake_case payloads and Claude's
  tool names. The decision is read from `hookSpecificOutput`, so the shim and the gate need no
  translation.
  - Every event is written under its PascalCase bus name, with `timeout` (not `timeoutSec`).
  - The tool events carry `matcher: "*"`.
  - `PermissionRequest` and `Notification` also name themselves with `--event` on the command line.
- **`config.json` and `settings.json` are two files.** Since 1.0.91, `config.json` is Copilot's
  state, and its settings are in `settings.json`. The agent's home gets:
  - `config.json`: the account pointer and `trustedFolders: []`;
  - `settings.json`: the fixed values, plus `askUser: false` and `disableAllHooks: false`, the
    operator's co-author and proxy settings (from their `settings.json`, else the legacy
    `config.json`), and the sandbox block.
- **The sandbox is switched on by settings, not a flag.** 1.0.91 has no `--sandbox`. A saved
  `sandbox.enabled`, with `experimental: true`, turns it on.
  - The block uses 1.0.91's `userPolicy` schema (`adapters/copilot_sandbox.py`).
  - A non-empty `allowedHosts` blocks every other host, so the package registries and
    `--session-allow-domain` are the only outbound hosts.
  - The prerequisites are checked on the login-shell `PATH`, plus `/dev/net/tun` on Linux.
- **`permission_request="rejudge"` was built before the live check (verify 10).** Denying there
  would refuse a call the gate had allowed. Copilot's re-ask gets an allow when the gate allowed the
  identical call earlier in the turn (the last 64 calls) or would allow it now. Anything else is
  denied, and it is never a card. Claude Code and Codex keep the deny.
- **The soul reaches every CLI without a system-prompt flag.** Design §6.9 put the soul on
  `UserPromptSubmit`, but only the ticket ever rode it, so Codex sessions never saw their system
  prompt. `_turn_context` now puts the system prompt ahead of the ticket for any preset with no
  `system_prompt_flag`. That covers Codex as well as Copilot.
- **D4 holds for every preset, at both ends.**
  - The held-hook timeouts were below the shim's 560 s wait: Claude Code at 540 s, Codex at 30 s.
    With Codex's approvals off, a hook it timed out left the call to Codex. Every preset now uses
    `HELD_HOOK_TIMEOUT_SECONDS` (600).
  - The host's own hold is capped at `MAX_HOLD_SECONDS` (530) inside a turn. `--ask-timeout`
    defaults to an hour (night 1). Every CLI's hook died first, so a hold never really lasted past
    about nine minutes. An answer after that was recorded as approved for a call the CLI had
    already been told was denied ("host is unreachable").
  - One test pins the order for every preset: hook timeout > shim wait > host hold.
- **The Canvas take-over opens the agent's own home** (W0's note on `codex resume` described this,
  but the terminal opened the operator's home).
  - The terminal launch now carries `agent_id`.
  - For a per-agent CLI whose home exists, the host starts it with
    `/usr/bin/env CODEX_HOME|COPILOT_HOME=<agent home> PYTHONPATH=<host package> …`, and reads the
    turn's usage from that home.
  - Our hooks are installed there, so the shim stands aside when `AUTOMATOS_TERMINAL` is set and
    there is no host socket.
  - Host contract 0.11.0.
- **`session.py` was split** (`session_files.py`, unchanged code), so this wave keeps it under
  800 lines.
- **The login route is `copilot` or `gh`, not `keychain`.** The host's source guard rejects
  the word "keychain" in the package's code (no credential handling). The one exempt literal
  is the sandbox's `"keychainAccess": False`, which denies the keychain to sandboxed commands.
  The guard asserts that the deny is present.
- **Waiting on the live run (Verify at build):**
  - the plaintext-token key: the probe checks `storeTokenPlaintext` and the presence of five
    candidate keys, and never reads a value;
  - the MCP tool-name spelling: six spellings are accepted;
  - `--worktree` combined with `--resume` (`worktree_excludes_resume` stays false);
  - the `totalNanoAiu` unit (booked as `/1e9` AI credits);
  - the version floor (1.0.70).
- **Found by the first live runs (build 5, 2 Oct night):**
  - **F233.** `--no-auto-login` switched off the stored login and the `gh` fallback, the only
    ways a session can sign in, so every session failed "No authentication information found".
    It was dropped. Copilot's `config.json` carries `//` header lines, and the account pointer
    is now read past them.
  - **F234.** Copilot runs its hooks inside its own sandbox. The seatbelt profile lets a process
    reach a Unix socket only at a read-write path, so the shim could not reach the host and
    every call was denied. The hook socket itself is now in `readwritePaths`; nothing else of
    the host's state is. The live rerun (1269-1271) showed that the profile writes its
    `deniedPaths` after its grants ("override broader allow rules"): the denied state dir beat
    the socket's grant and the session dir's (`ticket.md`). So the state is now denied entry by
    entry around this session's folder and the socket (`deny_all_but`). A read-write socket
    path also lets a sandboxed command unlink it and bind its own, so the shim checks that
    the peer is the host's PID (`AUTOMATOS_HOST_PID`; macOS
    `LOCAL_PEERPID`, Linux `SO_PEERCRED`). Under Linux's bubblewrap the socket is a bind mount
    that cannot be unlinked.

## Bank PoC

**What Copilot sessions give the PoC.** Automatos runs tickets on the Copilot seats the bank already licenses, as each developer's own session, behind Automatos' gate.
- Every write, command and URL is decided call by call.
- Held calls are answered from the Questions tab, the bell or Telegram.
- Results, Deliverables and usage (tokens and AI credits) land on the ticket.

**The bank's own Copilot controls still apply,** according to GitHub's admin docs:
- the Copilot CLI policy (enabled, disabled, or each organisation decides);
- enterprise model enablement ("users can only access AI models that are enabled at the enterprise level");
- content exclusions, which apply to Copilot CLI;
- the MCP server allowlist;
- the enterprise audit log, which records policy changes.

**Data residency.**
- Copilot CLI targets a GitHub Enterprise Cloud data-residency tenant through `GH_HOST`/`COPILOT_GH_HOST`, and the session keeps both.
- If the code lives on GitHub Enterprise Server, Copilot still authenticates against github.com or the GHE.com tenant.
- Which residency regions GitHub offers for Copilot, and whether one is in the UK, is for the bank's GitHub account team to confirm.

**Telemetry.** Copilot's own OpenTelemetry export can point at the bank's collector through the operator's `OTEL_*` settings, which the session keeps. The export follows the GenAI conventions, includes token metrics and offers mTLS options.

**Bank-side settings the PoC relies on (for your list):**
- Copilot CLI is enabled for the PoC users.
- The MCP server policy allows the Automatos tools server, or the PoC users are not on a registry-only allowlist. If neither holds, sessions run without Automatos tools and the ticket says so.
- `allowManagedHooksOnly` is not set for PoC users, or O4 is decided.
- The models the PoC agents name are enabled, or the agents use `auto`.
- Each developer has a seat, has run `copilot login` (keychain) or `gh auth login`, and has a paired host. The host runs on macOS or Linux, or on Windows through WSL2 only.

**What session mode does not do yet.** This is true for every CLI, not just Copilot:
- it runs in the local edition only (`config.py:2107`);
- hosts and the MCP bridge have only ever run against a backend on the same machine;
- a host has no owning user. Any host in a workspace claims any ticket it serves, so developer A's seat could run developer B's ticket.

With this PRD alone, the PoC setup is the local edition on each developer's machine. A central Automatos in the bank's AKS managing the developers' Copilot sessions needs four more pieces:
1. session mode in the enterprise (OIDC) edition;
2. hosts that pair and talk over TLS;
3. the bridge reachable over TLS from the developer's machine;
4. host-to-user binding at claim.

That is O1.

## Owner decisions

- **O1 · The central PoC setup.** Build "session mode outside the local edition" as its own PRD, next to the OIDC edition, or as Wave 4 here? It means the enterprise edition, remote hosts over TLS, the bridge over TLS, and host-to-user binding (a host claims only tickets of agents its pairing user owns or was assigned).
  - *Recommended:* its own PRD. Claude and Codex need it too, and it moves auth and tenancy, an area where AGENTS.md says ask first.
  - Without it, the PoC demo runs on the local edition, one per developer.
- **O2 · Windows desktops.** If the bank's developers are on Windows without WSL2, the host (a POSIX PTY) does not run there, although Copilot itself does. A native Windows host is not in this PRD. Is WSL2 available on the PoC machines?
- **O3 · Copilot's command sandbox is experimental** (`--experimental`, MXC).
  - *Recommended:* on by default (S2.2). This gives Copilot sessions Claude's rule: a machine that cannot sandbox never runs a session unsandboxed. The bank's admins can also force sandboxing org-wide.
  - The alternative: wait for GA, and rely on the host gate alone until then.
- **O4 · Orgs that allow managed hooks only.**
  - Either ship the shim as an administrator-deployed policy hook (`/etc/github-copilot/policy.d/`) that answers nothing outside Automatos sessions (munder's env-scoped pattern);
  - or keep refusing such hosts.
  - *Recommended:* refuse in v1, and revisit if the bank's policy requires managed-only hooks.
- **O5 · A per-ticket AI-credit cap.** Copilot itself enforces `--max-ai-credits <n>` as a soft cap, which answers munder #288 ("budgets never stop anything"). Add an agent field and pass it through?
  - *Recommended:* yes for the PoC: one validated field, plus the review path for the "spending limit for this session" sentence.
- **O6 · How many plan rounds** before a ticket goes to review (SP.3). *Recommended:* 5, meaning Discuss can send a plan back four times. Also whether **Reject** should cancel the ticket instead of sending it to review. *Recommended:* review, so the work and the plan stay on the record.
- **O7 · Subagents.** `Agent`/`Task` stays denied, matching Claude. Copilot fires hooks for a subagent's tool calls (1.0.49), so allowing it is safe in principle. It is still a policy change for both CLIs.
- **O8 · ACP as a future tier.** `copilot --acp`, Gemini CLI and Zed's adapters for Claude and Codex all send permission requests to the client as JSON-RPC. One ACP bridge could serve several CLIs. Record it in the design doc; nothing is built here.

**Not this product:**
- **Microsoft 365 Copilot and Copilot Studio.** Different products; this PRD is GitHub Copilot CLI.
- **The Copilot coding agent.** It works issues in GitHub Actions, not as a session on the operator's machine.
- **Copilot BYOK** (`COPILOT_PROVIDER_*` pointed at Azure OpenAI, Anthropic or Foundry Local). That is an API-key route, and the subscription rule strips it from sessions. A bank that wants Azure OpenAI in a UK data zone already has the API runtime for that.

## Verify at build

**No spend** (the binary, a throwaway `COPILOT_HOME`, the headless server):
1. **Login.** Does a fresh `COPILOT_HOME` holding only `loggedInUsers`/`lastLoggedInUser` log in from the keychain? Check with the headless server's auth-status call (`copilot --headless --stdio`; method names in the bundled `schemas/api.schema.json`; `authType` is `user` or `gh-cli`). No model call. If this fails, route 2 (`gh`) carries the design. Also confirm which config key holds a plaintext token.
2. **Policy keys.** The exact key for `allowManagedHooksOnly` in a `policy.d` file. Which repo settings (`.github/copilot/settings.json`, including `disableAllHooks`) Copilot reads for an untrusted folder in `-p`.
3. **Version floor.** The version these checks pass on becomes the floor.
4. **Sandbox.** Does `--sandbox` work in `-p`? What schema do the network host rules use?

**First live ticket** (spends AI credits on the operator's seat):
5. **Hooks load.** Hooks load from `$COPILOT_HOME/hooks/` in `-p` for an untrusted folder, and SessionStart fires before the first model call.
6. **Payload shape.** The PascalCase payload carries `session_id`, `cwd`, `transcript_path` (SessionStart), `last_assistant_message` (Stop), and `tool_input` with Copilot's keys. Hook commands inherit the CLI's environment (`AUTOMATOS_HOST_SOCK`, `AUTOMATOS_TASK_ID`, `PYTHONPATH`).
7. **Resume.** `-p --resume <id>` continues the session and appends to the same `events.jsonl`. `--session-id <new uuid>` creates a new one.
8. **Worktree.** Where `--worktree <name>` puts the worktree, and whether `--worktree` and `--name` combine with `--resume`.
9. **MCP config.** `--additional-mcp-config` accepts the HTTP entry (does it need `tools`?), and MCP tool names arrive in one of the S1.4 spellings.
10. **permissionRequest.** Does it fire after a PreToolUse `allow`, for a path outside the cwd or a URL? That decides `permission_request="rejudge"`. Also its camelCase output shape.
11. **AI-credit unit.** The unit of `totalNanoAiu`.
12. **The floor holds.** With the hooks file removed, a `-p` ticket's writes and commands are refused ("could not request permission from user"), and the run ends as `ungated_exit`.

## Test plan (owner, local edition)

0. **Set up.** You need a Copilot plan on your GitHub account. Run `brew install copilot-cli` (the standalone binary; no Node needed), then `copilot login`. Settings → Session mode should show "GitHub Copilot 1.0.x · <login> · served".
1. **After W0.** A Codex ticket's report shows `codex resume <id>`. `gh issue create` is refused in any session.
1b. **After Wave P.** Set a Codex agent to Plan and give it a ticket. Expect:
   - It explores, makes no change, and ends with its plan.
   - The plan arrives as a card in the Questions tab and on Telegram, and `plan.md` is in Deliverables.
   - **Discuss** with a note → a revised plan.
   - **Approve** → the same session resumes and makes the change.
   - A Claude Code agent in Plan whose card you leave unanswered for over 2 minutes now waits for you. It no longer lands in review.
2. **After W1.** File a ticket for a Copilot agent: "summarise the open TODOs in this repo into a CSV in the deliverables folder". Expect:
   - the ticket ends `done`;
   - it lists the gated calls and what the host decided for each;
   - a command off the allowlist, answered from the Questions tab, runs;
   - the report shows tokens and AI credits.
3. **Ungated attempt.** Put `{"disableAllHooks": true}` in a scratch repo's `.github/copilot/settings.json` and file a ticket there. It is refused with the sentence and never spawns.
4. **After W2.** The ticket calls `board_summary` and `submit_report`, and none of your own MCP servers appear. With the sandbox on, `cat ~/.ssh/id_ed25519` fails inside the session.
5. **After W3.** Settings shows Claude, Codex and Copilot for each host, each with its state.

## Success metrics

- The owner test plan passes in the local edition before the bank demo.
- All four permission modes work on Claude Code, Codex and Copilot. No form or setting says "Plan needs Claude Code", and no CLI quietly runs Plan as Edit automatically.
- Every Copilot run in the CI fixtures either ends with the gate proven (SessionStart seen) or as `ungated_exit` with no result.
- No session argv or environment contains a forbidden flag, an env token or a `COPILOT_PROVIDER_*` variable (tests).
- Copilot appears in the picker, Settings and analytics without anyone editing a frontend list.
- The first live Copilot ticket has zero false refusals of read-only work.

## Merge notes

- **Order:** W0 and Wave P → W1a (the rows; the host announces `copilot: served:false`) → W1b (the adapter) → W2 → W3.
  - W0 and Wave P are independent of each other and of Copilot.
  - Wave P is what makes Plan work on Codex, so it can ship first.
  - Each wave is green in CI and tested by the owner in the local edition before the next one starts.
- **Wave P changes the claim's meaning, not its shape.** `permission_mode` was the configured mode; it is now the mode for this turn. The keys are unchanged. If the outcome's new `plan` field counts as a contract change, bump the host version and `EXPECTED_CLI_HOST_VERSION` together. Otherwise an older host reports no plan, and a Plan ticket on it ends as an ordinary result.
- **No migration, no route.** The claim's wire shape does not move, so `EXPECTED_CLI_HOST_VERSION` stays. If a wave starts depending on a new capability field, move both pins together: `test_prd235_host_always_on.py:25` and `test_prd239_session_prompt.py:314`.
- **The rows land together.** The backend row and the host row ship in the same PR (W1a); otherwise the parity test goes red.
- **Commits:** DCO sign-off on every commit, and `git add` with explicit paths.
