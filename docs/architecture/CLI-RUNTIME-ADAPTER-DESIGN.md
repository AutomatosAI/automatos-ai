# CLI Runtime Adapter — Design

> **The pattern for running a ticket as the operator's own subscription CLI session, for any CLI.**
> Session mode works today on Claude Code (PRD-234 → PRD-239). The runtime that drives it is
> hardwired to one binary. This is the design that makes the CLI a **parameter** instead of an
> assumption, so the twelve CLIs on the roadmap share one integration path instead of twelve.
>
> **Status:** Design, rev 2 — reviewed 2026-09-11 against munder's implementation of all twelve
> CLIs and a real Codex rollout. No code until green-lit.
> **Closes:** PRD-239 D-5. **Supersedes:** PRD-235 S20 (`codex exec` — wrong transport, §6.6).
> **Reference implementation:** `chaitanyagiri/munder-difflin` — `src/shared/agentProvider.ts`
> (the preset table), `src/main/hive.ts` (the per-CLI bridges). The platform's host was already
> ported from munder (`env.py`, `transcript.py`); this extends the borrowing to its provider model,
> and departs from it in two places, both named (§6.1, §13).
> **Roadmap (owner, 2026-09-11):** Claude Code · Codex · Grok · Kimi Code · Antigravity · Qwen ·
> Gemini CLI · OpenCode · Crush · Pi · Copilot · Cursor. Codex and Grok are next.
> **Verified against:** `automatos-ai` @ `028e3a74e`, cli-host `0.6.0`; Codex CLI `0.154.0` (binary,
> `codex doctor`, a real rollout under `~/.codex/sessions`); `cursor-agent 2026.01.23`; munder @
> `9ce27e7`. On this machine: `claude`, `codex`, `cursor-agent` only.
>
> **Rev 2 changes (review findings):** Codex home is per agent, not per session (§6.1 — resume
> broke otherwise); translation is bidirectional (§5); a turn-end model for hookless CLIs (§3);
> claim-time provider filter, not post-claim refusal (§8.2); per-preset hook events + timeout
> literal (§4.1); Codex worktrees are feature-flagged and exclude resume (§6.7); the rollout reader
> spec from a real file (§6.8); the twelve-CLI matrix (§3.1); subscription legitimacy per CLI
> (§9.1); Grok, the next CLI (§13); a rollout order (§14).

---

## 1. What exists, and what is actually wrong

Session mode is built and working. `runtime: cli` agents get their tickets claimed by a paired host
which runs them as supervised interactive Claude Code sessions, gated by hooks over a loopback
socket, with the transcript as the record. That machinery is sound and **none of it is the problem.**

The problem is that "which CLI" was never a variable:

| Layer | State |
|---|---|
| `core/cli_runtime.py` | Already multi-CLI: `PROVIDER_CODEX`, per-provider model rules, `codex` usage slug |
| `cli_host_service.claim` | Ships `provider` on every claimed ticket |
| `runtime-section.tsx` | Offers Codex in the picker, hardcoded, labelled "not yet served" |
| **cli-host** | **Ignores `provider` entirely.** `host.py:308` logs it; `session.py:386` resolves `claude` unconditionally |

So the contract is already provider-shaped end to end, and the last mile discards it. Two consequences:

1. **Choosing Codex today silently runs Claude Code** and books the spend to `llm_usage` as `codex`.
   Nothing validates the agent's provider against the host's announced `capabilities.providers`
   (which `session.py:566` hardcodes to `["claude"]` anyway).
2. Every CLI-specific fact — argv, hook wiring, trust file, transcript layout, tool names — sits
   inline in `session.py`, so a second CLI means either branching that file or forking it.

**This design fixes (2) and (1) falls out of it.** The guard in (1) is independently shippable and
should land first regardless of when the rest is built.

---

## 2. The insight: the hook payload is the bus

The temptation is to abstract "the CLI". That is the wrong seam, and munder's twelve-provider
experience shows why: the CLIs do not differ meaningfully in *what they do*, they differ in **how
their lifecycle reaches you**. Three surfaces, in descending order of cooperation:

- Claude Code hands you lifecycle events through `--settings` hooks.
- Codex, Gemini, Grok, Antigravity, OpenCode, Pi hand you the same events through a config-file
  hook shim — with a translation step for some of them.
- Qwen and Crush hand you nothing, so you observe their LLM traffic and *synthesize* the events.

In all three cases the harness above wants **the identical event stream**. So the abstraction is:

> **The hook payload contract is the host's internal event bus.** Claude and Codex speak it
> natively. Weaker CLIs get a translator — in both directions. CLIs with no hook surface get a
> proxy that fabricates it, or are declared ungated. Nothing above the bus ever learns which CLI
> is running.

Concretely, everything in this list stays exactly as written and is **out of scope** for any CLI
work, now or for CLI #12: `hook_server.py`, `Session.handle_hook` (`session.py:249`),
`Session._pre_tool_use` (`:297`), the ask/approval round trip, `allowlist.py`, `state.py`,
`api.py`, `terminal_log.py`, and the whole claim/lease/flush loop in `host.py`.

The seam goes **below** the bus, and nowhere else.

---

## 3. The four tiers, and how a turn ends

Every CLI lands in exactly one tier. The tier decides how much work it is; the preset row is
identical in shape regardless.

| Tier | How events arrive | Turn ends on | Gate | Work to add one |
|---|---|---|---|---|
| **`native`** | The CLI's own hooks + a real `--append-system-prompt` | `Stop` hook | ours | preset row |
| **`hooks`** | A per-agent config home carrying hook entries that point at **our shim** | `Stop` (or the CLI's idle/end event, translated) | ours | preset row + `prepare()`; translator if not Claude-shaped |
| **`proxy`** | No hook surface — a loopback reverse proxy watches LLM traffic and synthesizes events | synthesized `Stop` | ours, on synthesized events | preset row + a sidecar |
| **`seed`** | No lifecycle at all | **process exit** (print mode) or **PTY idle** (interactive) | **none** — the CLI's own sandbox only | preset row; degraded |

**The turn-end model was missing from rev 1**, and without it every hookless CLI would invent its
own. It is a preset field, `turn_end ∈ {stop_hook, process_exit, pty_idle}`, and the session loop
(`session.py:463-484`, which today waits only on `self.stopped`) switches on it. `pty_idle` is
munder's "pty-quiescence fallback" — the weakest signal there is, and honest about it.

Two consequences worth stating plainly:

- **Codex is tier-2, and tier-2 is cheap.** Its hook payload and response contract are already
  Claude-shaped, so the shim is reused *verbatim* — munder does exactly this
  (`agentProvider.ts:93`). The work is where the config file lives, not what goes in it.
- **A `seed`-tier CLI cannot be policy-gated.** No `PreToolUse` means no gate, which means the
  session's blast radius is whatever the CLI's own sandbox gives it. That is a product decision, not
  a technical one: either such CLIs are refused for session agents, or they run with a loud,
  persistent "ungated" marker on the ticket. **Open decision D-3 (§12).**

### 3.1 The twelve, classified

Every cell is sourced: **(m)** = read from munder's code; **(v)** = verified on this machine;
**(?)** = unverified, must be confirmed at build. munder itself marks OpenCode, Pi and Crush
`LIVE-UNVERIFIED`.

| CLI | Tier | Isolation lever | Event vocabulary | Turn end | Own-plan login | Here |
|---|---|---|---|---|---|---|
| Claude Code | native | `--settings <file>` (v) | Claude | `Stop` | Anthropic OAuth in the unmodified binary (v) | ✓ |
| **Codex** | hooks | `CODEX_HOME` per agent + `config.toml` tables (m, v) | Claude-shaped, identity (v) | `Stop` | ChatGPT login; **this machine is on an API key** (v) | ✓ |
| **Grok** | hooks | `GROK_HOME` exists (m); munder writes a *global* `~/.grok/hooks/` file instead (m) — §13 | Claude events, **camelCase fields**, snake_case names; extras `PostToolUseFailure`, `PermissionDenied`, `StopFailure`, `Notification` (m) | `Stop` | xAI login (?) | ✗ |
| Kimi Code | hooks (?) | (?) — munder ships no bridge; "supports lifecycle hooks" (m) | (?) | `Stop` (?) | Moonshot login (?) | ✗ |
| Antigravity | hooks | none in munder — writes *global* `~/.gemini/config/hooks.json` + `antigravity-cli/hooks.json`, merged under a key (m) | `PreToolUse, PostToolUse, PreInvocation, PostInvocation, Stop`; **event name passed as argv**, payload carries none (m) | `Stop` | Google login (m) | ✗ |
| Gemini CLI | hooks | `GEMINI_CLI_SYSTEM_SETTINGS_PATH` per agent; auth stays in the normal home, settings are layered (m) | `SessionStart, BeforeAgent, BeforeTool, AfterTool, AfterAgent` → translated (m) | `AfterAgent`→`Stop` | Google login (m) | ✗ |
| OpenCode | hooks | `OPENCODE_CONFIG_DIR` per agent; plugin in `plugin/` **and** `plugins/` (m) | `tool.execute.before/after`, `session.idle` (m) | `session.idle`→`Stop` | BYOK, or OAuth into *another vendor's* plan — see §9.4 | ✗ |
| Pi | hooks | `PI_CODING_AGENT_DIR` per agent; extension; copies `models.json` (m) | `tool_call, tool_result, agent_end` (m) | `agent_end`→`Stop` | BYOK (m) | ✗ |
| Qwen | proxy | `OPENAI_BASE_URL` → loopback (m, TODO-verify in munder) | synthesized | synthesized | OAuth ended 2026-04-15 → BYOK | ✗ |
| Crush | proxy | `CRUSH_GLOBAL_CONFIG` + `CRUSH_GLOBAL_DATA` per agent (m) | synthesized | synthesized | BYOK (m) | ✗ |
| Copilot | seed | none needed | **none**; `-p` print mode only (m) | `process_exit` | GitHub login (m) | ✗ |
| Cursor | seed | none needed | **none** (v: no hook flag in `--help`) ; `-p --output-format stream-json` exists (v) | `process_exit` (print) / `pty_idle` (TUI) | Cursor login (v: `models … for this account`) | ✓ |

Read the two right-hand columns together with §9.1 and §14: **hook surface × legitimate own-plan
login** is the order to build in, and it is not the roadmap order.

---

## 4. The contract

Two objects, deliberately split: **data you add** and **behaviour you override**. A well-behaved CLI
needs only the first.

### 4.1 `CliPreset` — pure data, one row per CLI

New file: `services/cli-host/automatos_cli_host/presets.py`. No imports beyond stdlib, no I/O — so
it can be read by tests, by the backend mirror (§8.1), and by a future manifest without dragging the
host in.

```python
@dataclass(frozen=True)
class CliPreset:
    id: str                          # "claude" | "codex" | …  (matches core/cli_runtime PROVIDERS)
    label: str                       # "Claude Code" — what the picker shows
    binary: str                      # default executable name; overridable per host
    tier: Tier                       # native | hooks | proxy | seed
    turn_end: TurnEnd                # stop_hook | process_exit | pty_idle          (rev 2)

    # ── launch ──────────────────────────────────────────────────────────────
    model_flag: Optional[str]                 # "--model" / "-m"; None = no model selection
    session_id_flag: Optional[str]            # "--session-id"; None = learn it from SessionStart
    resume_flag: Optional[str]                # "--resume <id>" (claude, grok, copilot, cursor)
    resume_subcommand: Optional[str]          # "resume" — Codex resumes by SUBCOMMAND, not a flag
    cwd_flag: Optional[str]                   # None = spawn with cwd=; "-C" for Codex
    add_dir_flag: Optional[str]               # "--add-dir"
    worktree_args: Tuple[str, ...]            # ("--worktree",) / ("--enable","worktrees","--worktree")
    worktree_excludes_resume: bool            # Codex: --worktree cannot resume (§6.7)
    system_prompt_flag: Optional[str]         # "--append-system-prompt-file"; None ⇒ ride the bus (§6.9)
    initial_prompt: PromptDelivery            # positional | flag("-i") | type_into_tui
    ungated_stance: Tuple[str, ...]           # argv that means "don't prompt, we gate at PreToolUse"
    required_args: Tuple[str, ...]            # e.g. Codex's --dangerously-bypass-hook-trust (§6.4)

    # ── hooks ───────────────────────────────────────────────────────────── (rev 2)
    hook_events: FrozenSet[str]               # the bus events THIS CLI can deliver
    hook_timeout_literal: Any                 # what the CLI's config expects: 540 (s) / 30 (s) / 30000 (ms)
    allow_is_silence: bool                    # agy: any stdout object = a decision; allow = write nothing
    event_name_source: EventNameSource        # payload | argv   (agy passes the event on argv)

    # ── environment ─────────────────────────────────────────────────────────
    config_home_env: Optional[str]            # "CODEX_HOME", "GROK_HOME", "OPENCODE_CONFIG_DIR", …
    config_home_scope: Scope                  # per_agent (default) | per_session | none
    strip_env: FrozenSet[str]                 # credentials/redirection this CLI must never inherit
    keep_env: FrozenSet[str]                  # operator config that IS forwarded
    extra_env: Mapping[str, str]              # e.g. CODEX_NON_INTERACTIVE=1

    # ── invariants ──────────────────────────────────────────────────────────
    forbidden_args: Tuple[str, ...]           # per-CLI; the subscription + gate guarantees (§9)

    # ── identity ────────────────────────────────────────────────────────────
    auth_probe: AuthProbe                     # how to tell "logged in with the operator's own plan" (§9.1)
    install_hint: Optional[str]               # shown when the binary is missing; never auto-run
    docs_url: Optional[str]
```

`config_home_scope` defaults to **per agent per host**, not per session — see §6.1 for why per
session is wrong.

### 4.2 `CliAdapter` — seven methods, all defaulted

New file: `services/cli-host/automatos_cli_host/adapters/base.py`. `PresetAdapter` implements all
seven from the preset alone; a CLI only subclasses for what its preset cannot express.

```python
class CliAdapter(Protocol):
    preset: CliPreset

    def preflight(self) -> Optional[str]:
        """None = ready. Otherwise the operator-facing sentence that lands on the
        ticket: binary missing, never run interactively, or logged in the wrong way
        (an API key where the runtime requires the operator's own plan)."""

    def prepare(self, ctx: LaunchContext) -> Prepared:
        """Write everything the session needs before spawn and return
        (env_additions, extra_args). Tier-2's whole weight lives here: the config
        home, the hook entries pointing at our shim, the trust record."""

    def launch_args(self, ctx: LaunchContext) -> List[str]:
        """Full argv. Must satisfy assert_args_honour_invariant(preset)."""

    def normalize_event(self, raw: Mapping[str, Any]) -> Optional[CanonicalEvent]:
        """Raw hook payload → the bus shape (§5). Identity for claude and codex;
        a key/name map for grok and gemini. None = an event this CLI emits that the
        bus does not model — dropped, not an error."""

    def render_response(self, event: str, reply: CanonicalReply) -> Optional[str]:   # rev 2
        """The bus's answer (allow / deny+reason / block+reason / additionalContext /
        continue=false) → the bytes THIS CLI expects on the hook's stdout. None =
        write nothing. Claude and Codex: the Claude wire shape. Gemini/agy/grok:
        {decision, reason}. agy: allow MUST be None (§5)."""

    def tool_intent(self, tool_name: str, tool_input: Mapping[str, Any]) -> ToolIntent:
        """What the call DOES, not what it is called: a class plus the fields the
        policy needs. This is what lets policy.py stop hardcoding "Bash"."""

    def read_usage(self, transcript: Path) -> Dict[str, Any]:
        """Normalized token counts for llm_usage. Never a price — a subscription
        session has no dollar figure to invent."""
```

`ToolIntent` is the piece that earns its keep. The current `policy.decide` (`policy.py:165`)
switches on Claude's tool *names* and reaches into Claude's tool-input *keys*. Codex's shell tool is
`unified_exec` and its edit tool is `apply_patch` with a different input shape — so name-mapping
alone is not enough:

```python
class ToolClass(Enum):
    FILE_READ, FILE_WRITE, SHELL, WEB, BENIGN, UNKNOWN

@dataclass(frozen=True)
class ToolIntent:
    cls: ToolClass
    paths: Tuple[str, ...] = ()      # everything this call would touch
    command: Optional[str] = None    # the shell command, if any
```

`policy.decide(intent, ctx)` then switches on `intent.cls` and reads `intent.paths` /
`intent.command`. Every rule already in `policy.py` — the never-allowed list, the traversal check,
the compound-command split, `_runs_own_code`, the ask-the-operator fallback — is **unchanged and
CLI-neutral**. Only the adapter knows that `apply_patch` writes files.

---

## 5. The bus: canonical event, canonical reply

Already defined in practice by `compact_event` (`session.py:178`). Formalizing it as the contract:

```
event  { event, at, session_id, transcript_path, cwd,
         tool_name, subject, notification_type, message }

reply  allow | deny(reason) | block(reason) | context(text) | stop(reason)
```

`event` ∈ `SessionStart · UserPromptSubmit · PreToolUse · PermissionRequest · PostToolUse ·
Notification · Stop · SubagentStop · PreCompact · PostCompact · SessionEnd`.

**Translation is bidirectional, and rev 1 modelled only the inbound half.** munder's translating
shims (`hive.ts` — `GROK_HOOK_SHIM`, `GEMINI_HOOK_SHIM`, `AGY_HOOK_SHIM`) each carry a `done()`
that maps the harness's Claude-shaped reply back into the CLI's decision vocabulary. Three things
those shims teach, now preset fields:

- **Gemini and Grok** want `{decision: "deny", reason}` where Claude wants
  `hookSpecificOutput.permissionDecision`. A name map, nothing more.
- **Antigravity fail-closes on any stdout object.** An empty or decision-less JSON object is a
  DENY. So `allow` must be *no output at all* → `allow_is_silence: true`, and `render_response`
  returns `None`. Get this wrong and every tool call in an agy session is refused.
- **Antigravity's payload carries no event name** — munder passes it as argv to the shim →
  `event_name_source: argv`.

Where the translation lives: **in the host, in the adapter — not in per-CLI shims.** munder ships
one shim per CLI because its shim is the translator. Ours (`hook_shim.py`) forwards bytes and
answers bytes; it stays one file. `Session.handle_hook` calls `adapter.normalize_event` on the way
in and `adapter.render_response` on the way out. The shim keeps exactly one CLI-specific thing:
its **offline fail-closed deny** for the gated events (`hook_shim.py:25-41`) is Claude-shaped
today; it takes the CLI id from `AUTOMATOS_CLI` in the session env and picks the deny shape from a
three-entry table. That is the entire per-CLI footprint in the shim.

Claude → canonical is identity. **Codex → canonical is also identity** for every event the bus
models; Codex additionally emits `turn_id`, `agent_type`, `tool_use_id` (dropped) and `Interrupt`
(dropped — nobody is at the TUI to interrupt). Codex has **no `Notification` event**; the host
collects notifications today (`session.py:286`) but never reports them, so nothing is lost.

`subject` extraction (`_subject_of`, `session.py:203`) currently probes Claude's input keys. It
moves behind `ToolIntent`: `subject = intent.command or intent.paths[0]`.

---

## 6. Codex, concretely

Every fact below is either read out of Codex `0.154.0` on this machine — binary strings, `codex
doctor`, and a real rollout file — or taken from munder's production implementation
(`hive.ts:2060 installCodexHooks`). Where munder's empirical result contradicts the binary's own
schema strings, munder wins — noted inline.

### 6.1 One config home per agent per host, with the operator's login linked in

`CODEX_HOME` = `<host state dir>/agents/<agent_id>/.codex`. **Per agent, not per session.**

Rev 1 said per session, and that is wrong: Codex keeps its session index in
`$CODEX_HOME/state_5.sqlite` (`codex doctor`: "rollout DB thread inventory"), and `codex resume
<id>` resolves the id through that index. A fresh home per ticket would mean **a resumed chat
ticket never finds its predecessor** — the exact continuity PRD-239 D1 built. munder's scope is per
worker for the same reason. Per-ticket state (`ticket.md`, the terminal log) stays in the session
dir as now; only the CLI's own home is per agent. Concurrent tickets of one agent share the home
the way concurrent terminals of one user do — that is Codex's normal case.

The operator's `~/.codex` is **never written**. The login is carried in by **symlinking
`~/.codex/auth.json`** into the agent home (copy fallback where symlinks need privilege). This is
the subscription invariant expressed for Codex: the session authenticates *as the operator, through
the operator's own login file*. We do not read the token, do not copy it into our own storage, do
not put it on a command line, and do not intermediate it — the same posture that makes the
unmodified-`claude`-with-your-own-login pattern acceptable. `~/.codex/packages` is linked too (the
managed app-server daemon is rooted there).

munder goes one step further — `exposeCodexDataDirs` moves the isolated home's `sessions/` under a
namespaced folder in the operator's *global* `~/.codex/sessions/` and links it back, so the
operator's own `codex resume` picker and usage tools see hive sessions. That is a write into the
operator's global tree, however namespaced. **Off by default for us; D-7.**

### 6.2 Hooks go in `config.toml`, not `hooks.json`

**This contradicts the binary's own schema, and the binary loses.** `$CODEX_HOME/hooks.json` is
*plugin-scoped* — discovered only when referenced from a plugin manifest — and never fires for a
plain config dir. munder verified this empirically (`hive.ts:2085-2090`). The surface Codex actually
scans is `config.toml`:

```toml
[[hooks.PreToolUse]]
[[hooks.PreToolUse.hooks]]
type = "command"
command = "<our existing shim>"
timeout = 30
```

…one group per event, **all pointing at the same shim we already ship** (`hook_shim.py`), because
Codex's payload and response contract are Claude-shaped.

The file is **seeded from the operator's own `config.toml`** before the hook tables are appended, so
their model, provider and trust settings carry into the session. Regenerated per spawn, idempotent —
so an operator's model change propagates to the next ticket.

### 6.3 `timeout` is SECONDS, `0` is a trap, and the unit differs per CLI

Claude's settings take `timeout: 0` as "no timeout". Codex parses the key as `timeout_sec` and
normalizes it `timeout_sec.unwrap_or(600).max(1)` — so **`0` is floored to one second**, the
shortest budget there is. munder shipped that through v0.3.7 and every worker died on
`SessionStart hook (failed) — hook timed out after 1s`, because the shim cold-start measured
0.08–0.16 s idle but 0.6–0.7 s under eight concurrent spawns.

| CLI | Key | Unit | Value to write | Why |
|---|---|---|---|---|
| Claude | `timeout` | s | 540 (gated events), 60 | hold-open for the approvals inbox, under the CLI's 600 s default |
| Codex | `timeout` | s | **30** | clears cold-start by two orders; caps a wedged shim well under 600 |
| Gemini | `timeout` | **ms** | 30000 | munder's value |
| Antigravity | `timeout` | — | 0 | munder writes 0 and it works there; treat as unverified |

Hence `hook_timeout_literal` on the preset: the value the CLI's own config expects, verbatim, not a
number the host converts. Verifiable for Codex with no model spend: `codex app-server` →
`initialize` → `hooks/list` reports the normalized `timeoutSec` per event.

### 6.4 `--dangerously-bypass-hook-trust` is required, and is not a gate bypass

Codex refuses to run hooks from a config dir without persisted hook trust — normally an interactive
gate. Without the flag **the hooks silently never fire**, which is the worst possible failure: the
session runs completely ungated and looks fine.

This will read alarming next to `FORBIDDEN_ARGS` (`session.py:54`). It is a different axis:

- `--dangerously-skip-permissions` / `--permission-mode bypassPermissions` / `-p` / `--bare` remove
  **our** gate or break the subscription invariant. Permanently banned.
- `--dangerously-bypass-hook-trust` trusts **our own hook file, which we just wrote, inside a config
  home we own**. It is the precondition for the gate existing at all.

That is precisely Codex's documented use ("automation that already vets hook sources"), and it is
the direct analogue of what the host already does for Claude in `claude_settings.record_directory_trust`
— recording the operator's trust decision where the CLI reads it, minimally and backup-first.
Hence `required_args` and `forbidden_args` are both **per-preset** (§9.2).

### 6.5 Permission stance: approvals off, sandbox on

`-a never -s workspace-write`, plus `--add-dir <session dir>` so the session can write its own
scratch alongside the workspace. **Not** `--dangerously-bypass-approvals-and-sandbox`: munder used
to, purely because a worker needed to write outside cwd, and correctly reclassified that as a
path-layout problem rather than a reason to drop the OS sandbox. Keeping Codex's sandbox is free
defence in depth *underneath* our own policy gate.

`CODEX_NON_INTERACTIVE=1` suppresses the first-run trust/installer prompts that a supervised session
cannot answer.

### 6.6 The transport question, settled

Codex hooks fire in **interactive** sessions. They do **not** fire under headless `codex exec`.

PRD-235 S20 specs `codex exec`. That is wrong twice over: it forfeits every lifecycle event (no
policy gate, no live ticket log, no approvals), and it is the same headless posture that was
withdrawn for Claude on billing grounds in the PRD-234 rewrite. The PTY-drained interactive session
this host already runs is the correct and only transport. **S20 is superseded by this document.**

### 6.7 Session identity, resume, worktrees

**Two ids, and the backend already models both.** The backend pre-assigns `runtime_ref.session_id`
at claim (`cli_host_service.py:676`) and Claude is told to use it (`--session-id`). Codex has no
such flag — it mints its own. The backend already handles that: an event carrying `session_id`
stores it as `runtime_ref.cli_session_id` (`:794`, `:862`), and resume prefers it
(`:512`: `cli_session_id or session_id`). So: **the pre-assigned id is the correlation key** (host
heartbeat, lease re-attach at `:585`), **`cli_session_id` is the CLI's own id** (resume, the
transcript, the Canvas `claude --resume` / `codex resume` line). `session_id_flag: None` on the
preset means "correlation only; the real id arrives on `SessionStart`". No backend change.

**Resume is a subcommand:** `codex resume <SESSION_ID>` — `--resume` does not exist (munder:
"restarts used to silently start a brand-new session instead of continuing").

**Worktrees are feature-flagged and exclude resume.** From the binary: `--worktree requires the
worktrees feature; enable it with --enable worktrees`; `--worktree cannot be combined with
--ignore-user-config` or `--ephemeral`; and **`--worktree cannot resume an existing session`**. So
`worktree_args = ("--enable", "worktrees", "--worktree")`, and a resumed ticket omits them — the
resumed session is already in its worktree. **Where Codex puts a managed worktree is not
determinable from the binary** (`worktree/src/paths.rs`); if it lands under `$CODEX_HOME`, the
per-agent home (§6.1) keeps it alive across tickets, which is the second reason per-session was
wrong. **Verify at build (§6.10).**

### 6.8 The rollout file — `read_usage` from a real one

Read from `~/.codex/sessions/2026/04/30/rollout-…-019de017-….jsonl` on this machine (cli
`0.126.0`; confirm the shape holds on `0.154.0` at build). Line-delimited JSON, `{timestamp, type,
payload}`:

| Record | What the reader takes |
|---|---|
| `session_meta` (first line) | `payload.id` = the session id; `cwd`, `cli_version`, `model_provider` |
| `turn_context` (per turn) | `payload.model` (`"gpt-5.5"`), `sandbox_policy`, `approval_policy` — **the model is here, not on the usage record** |
| `event_msg` / `token_count` | `payload.info.total_token_usage` = **cumulative** `{input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens, total_tokens}`; `last_token_usage` = the turn's own; `rate_limits` (null on an API key — a plan-window signal on a ChatGPT login Claude's transcript never gives us) |
| `event_msg` / `agent_message` | the assistant text; the last one is the result — the `Stop` hook's `last_assistant_message` is the primary, this the cross-check, exactly as `transcript.py` does for Claude |
| `response_item` / `function_call`, `custom_tool_call` | tool calls (`unified_exec`, `apply_patch`) — not needed for usage; the hooks already report them live |

So `read_usage` for Codex = the **last** `token_count.total_token_usage`, model from the **last**
`turn_context`, per-model by attributing each `token_count` delta to the `turn_context` before it.
`usage_delta` (`transcript.py`) applies unchanged for resume, **if** `codex resume` appends to the
same rollout — assumed, verify at build. Normalized key map: `cached_input_tokens` →
`cache_read_input_tokens`; `reasoning_output_tokens` is reported additionally (Claude has no
equivalent; the analytics page ignores unknown keys).

### 6.9 The soul rides the bus — no argv exposure

Codex has no `--append-system-prompt-file`, which looks like it forces the agent's persona and
skills (PRD-239 S1) into argv — visible in `ps`, against the rule stated at `session.py:162`.

It does not. **The host already injects the ticket through `UserPromptSubmit` →
`additionalContext`** (`session.py:268`), and Codex implements that same hook with that same output
field. So the soul travels the identical channel on both CLIs, Codex's positional prompt stays the
same short pointer Claude gets, and PRD-239 S1's stable-per-agent prompt-cache invariant is
preserved unchanged.

`system_prompt_flag: None` therefore means *"deliver via the bus"*, not *"unsupported"* — and that
is the default for every tier-2/3 CLI, so no future CLI reopens this question. (Tier-4 has no bus;
the soul goes into the seed prompt, and that *is* argv-visible unless it is a file pointer — the
preset's `initial_prompt` must be a pointer for the same reason.)

### 6.10 Verify at build — no spend, no live session

**As built (Wave C, 2026-09-11):** `adapters/codex.py` + `tests/fake_codex.py` cover the mechanics
in CI — the per-agent home, the linked login, the seeded `config.toml` with `[[hooks.<Event>]]`
tables at `timeout = 30`, hooks firing only with `--dangerously-bypass-hook-trust`, the gate on
`exec_command`/`apply_patch`, the id learned on `SessionStart`, the rollout reader on the §6.8
shape, `codex resume <id>` into the same home and rollout with this turn's usage only. The six
items below remain what a **live** run on the operator's machine must confirm; the login probe
reads `auth.json`'s `auth_mode` label and whether `tokens` exist — never a value.

1. **Auth mode.** `codex doctor` → `stored ChatGPT tokens: true` after `codex login`. Until then
   `preflight()` refuses (§9.1).
2. **Hooks fire.** `codex app-server` → `initialize` → `hooks/list` lists our eight entries with
   `timeoutSec: 30`.
3. **Worktree location** with `--enable worktrees --worktree` in a throwaway repo: where the
   checkout lands, and that it survives the session dir being removed.
4. **Resume appends** to the same rollout file (for `usage_delta`).
5. **Rollout shape on 0.154.0** matches §6.8 (the sample is from 0.126.0).
6. **`hooks.json` is inert** for a plain config dir (re-confirm munder's finding on this version).

---

## 7. Where the code changes

| File | Change | Size |
|---|---|---|
| `presets.py` | **new** — the preset table | ~260 |
| `adapters/base.py` | **new** — `CliAdapter`, `PresetAdapter`, `ToolIntent`, `LaunchContext`, `CanonicalReply` | ~220 |
| `adapters/claude.py` | **new** — today's behaviour, moved not rewritten | ~120 |
| `adapters/codex.py` | **new** — `prepare()` (§6.1–6.5), `read_usage()` (§6.8) | ~180 |
| `session.py` | resolve the adapter from `ticket["provider"]`; `build_args`/preflight/`_collect` call through it; `handle_hook` wraps in/out translation; the wait loop switches on `turn_end` | −180 / +70 |
| `policy.py` | `decide(intent, ctx)`; delete `FILE_TOOLS`/`"Bash"` literals. Rules unchanged | ~30 |
| `hook_shim.py` | offline-deny shape from `AUTOMATOS_CLI` (three entries) | ~15 |
| `claude_settings.py` | → `adapters/claude.py`; `write_settings` becomes its `prepare()` | move |
| `transcript.py` | keep Claude's reader; the Codex rollout reader beside it | +80 |
| `env.py` | `STRIPPED_EXACT`/`CLAUDE_CONFIG_KEEP` come from the preset | ~25 |
| `config.py` | `--claude` → `--cli-binary <id>=<path>` (repeatable); keep `--claude` as an alias | ~20 |
| `session.host_capabilities` | announce **detected** providers, per-CLI version + login state | ~40 |

`session.py` gets **smaller**. Nothing in `hook_server.py`, `allowlist.py`, `state.py`, `api.py` or
`terminal_log.py` is touched; `host.py` changes in two lines (the fingerprint walks the subpackage;
the terminal server takes `cli_binaries`).

**As built (Wave 0/B, 2026-09-11) — four decisions made while building, recorded here:**

- **Moved, not wrapped.** `build_args`, `claude_settings.py`, `_subject_of`, `FILE_TOOLS` and friends
  moved into `adapters/claude.py`; no module-level Claude names survive in `session.py` or
  `policy.py` (repo rule: no dual paths). Tests changed imports and signatures only — every verdict
  and every assertion is the same.
- **`--claude <path>` is gone; `--cli-binary ID=PATH` (repeatable) and `AUTOMATOS_CLI_BINARIES`
  replace it.** The login service reproduces it. `make cli-host` never passed `--claude`, so no
  installed service carries the old flag.
- **The Canvas terminal's plain shell strips the union of every preset's credentials and
  markers** (`build_shell_env`); a *launched* session gets its own CLI's hygiene. The terminal
  injects the soul only for a CLI with a system-prompt flag — a terminal has no hooks, so for the
  others the operator gets their plain CLI, said plainly in `command_for`.
- **Capabilities changed shape** (host 0.7.0): every registry CLI under `clis.<id>` with
  `{path, version, served, reason, tier}`; `providers` = the served ids. `capabilities.claude` is
  gone; the Settings tab reads `clis.claude`. A preset without an adapter (Codex until its wave)
  is announced `served: false` — the picker can show *why*, and the claim filter keeps its
  tickets for a host that can.
- **The refusal code is data.** `AuthProbe.code` is the ticket's `exit_reason`
  (`claude_not_onboarded`, `codex_api_key_login`); the base adapter emits `<id>_missing` for an
  absent binary. Nothing in the backend interprets the codes beyond display.

`terminal_server.py` has its own hardcoded `launch.kind != "claude"` refusal (`:462`) for the Canvas
takeover terminal. Same treatment, same preset: `codex resume <id>` in the session's effective cwd,
with `CODEX_HOME` pointed at the agent home so the id resolves.

---

## 8. Backend and UI — where "plug and play" is actually won

The host work makes CLI #7 *possible*. These three make it *free*.

### 8.1 One registry, two renderings

`core/cli_runtime.py` already holds a second copy of the CLI list (`CLI_PROVIDERS`,
`USAGE_PROVIDER_SLUGS`, `_CLAUDE_MODEL_ALIASES`, `_CODEX_MODEL_RE`). Two copies drift.

The backend does not need argv or hook wiring — only id, label, model rule and usage slug. So:
`orchestrator/core/cli_presets.py` holds that subset, and **a test asserts the id set matches the
host's `presets.py`**. No build step, no generated code, and drift fails CI. The host stays the
single owner of anything operational.

### 8.2 Honest capabilities, and a claim-time filter

`host_capabilities` announces what it actually detected — per CLI: path, version, and whether
`preflight()` would pass. **A provider counts as served only if preflight passes**: a host with
`codex` on an API key announces `codex: {present: true, served: false, reason: …}`, so the
operator sees why in Settings instead of every Codex ticket bouncing to review.

The guard must be a **claim-time SQL filter, not a post-claim refusal.** `claim_tasks`
(`board_dispatcher.py:91`) takes `runtime=` and `workspace_id=` today; it gains `providers=` and
adds `agents.configuration->>'provider' IN (…)` to the claim statement. Rev 1 said "refuses a
ticket" — with two hosts of different CLIs that is a claim/release loop every poll. Filtered at
claim, a Codex ticket simply waits for a Codex host, with `blocked_reason` naming the missing CLI
the same way `NO_HOST_REASON` does today. The filter reads `capabilities.providers` strictly: a host
that **never announced the key** gets no filter (the pre-field behaviour — the real-DB suite pairs
hosts that way); a host that announced **an empty list** claims nothing.

This is the fix for §1(1) and it does not depend on any of the rest. **Ship it first.**

### 8.3 The picker renders from capabilities

`runtime-section.tsx:226` hardcodes two `SelectItem`s and Claude-specific model help text. It
should render from the paired hosts' announced providers — label, model hint and placeholder all
from the same registry.

That is the whole point: **adding CLI #7 makes it appear in the picker because a host announced it,
with no frontend change.** A provider no host serves shows as unavailable with the reason, instead
of a hand-maintained "not yet served" string that goes stale — as the current Codex one already has.

---

## 9. Invariants, per CLI

The subscription and safety guarantees are the reason this runtime exists. They are **preset fields
with per-CLI values**, not global constants — because the same guarantee is spelled differently on
each binary, and a global constant silently means "unchecked" for CLI #2.

### 9.1 "The operator's own plan, never a key"

The policy, in one line: **a CLI is admissible when its vendor ships a first-party login for the
operator's own plan in the unmodified binary; BYOK-only CLIs, and OAuth into a *different* vendor's
plan, are not.** `auth_probe` per preset, checked in `preflight()`, refusing with a plain sentence.

| CLI | Own-plan login | Probe | Strip from the session env |
|---|---|---|---|
| Claude | Anthropic OAuth | `~/.claude.json` `hasCompletedOnboarding` | `ANTHROPIC_API_KEY/AUTH_TOKEN/BASE_URL` (done) |
| **Codex** | ChatGPT login | `auth.json` exists **and** mode ≠ `api_key` (`codex doctor` reports it) | `OPENAI_API_KEY`, `CODEX_API_KEY`, `OPENAI_BASE_URL` |
| Grok | xAI — **verify** | (?) | `XAI_API_KEY` (?) |
| Kimi | Moonshot — **verify** | (?) | (?) |
| Antigravity | Google | in `~/.gemini` (m) | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| Gemini CLI | Google | in `GEMINI_CLI_HOME` (m) | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| Copilot | GitHub | (?) | `GITHUB_TOKEN` (?) |
| Cursor | Cursor | `cursor-agent models` lists "for this account" (v) | `CURSOR_API_KEY` (?) |
| OpenCode | **none of its own** — BYOK, or OAuth into Anthropic/OpenAI plans (§9.4) | — | — |
| Pi | BYOK (m) | — | — |
| Qwen | OAuth ended 2026-04-15 → BYOK | — | — |
| Crush | BYOK (m) | — | — |

**This bites today.** `codex doctor` on this machine reports `stored auth mode: api_key`,
`stored ChatGPT tokens: false`. The symlink pattern (§6.1) is correct either way, but until
`codex login` runs, a Codex session spends an API key — the exact thing the runtime exists to avoid.
`preflight()` must refuse that, in words, rather than run it.

### 9.2 "The gate is ours, and it is never bypassed"

`forbidden_args` per preset, still asserted at spawn (`assert_args_honour_invariant`, `:167`), now
against the preset's tuple:

| CLI | Forbidden | Required | Reason |
|---|---|---|---|
| `claude` | `-p`, `--print`, `--bare`, `--dangerously-skip-permissions`, `--permission-mode bypassPermissions` | — | headless billing posture; removes our gate |
| `codex` | `--dangerously-bypass-approvals-and-sandbox`, `exec`, `--ephemeral` | `--dangerously-bypass-hook-trust` | drops the OS sandbox; `exec` has no hooks; ephemeral persists nothing |
| `grok` | `--permission-mode bypassPermissions` (?) | — | same axis as Claude |
| seed-tier | (none — there is no gate to protect) | — | see D-3 |

### 9.3 "Sessions never push"

CLI-neutral — it lives in `policy.NEVER_ALLOWED_BASH` and applies to any `ToolClass.SHELL` call
regardless of what the CLI calls its shell tool. Unchanged. **Unenforceable on seed-tier** — a third
reason for D-3.

### 9.4 The proxy tier is a key tier

A proxy bridge works by pointing the CLI's upstream base URL at a loopback sidecar. That
presupposes the CLI is talking to an OpenAI/Anthropic-wire endpoint **with a key** — which is
BYOK by construction. Qwen and Crush therefore cannot satisfy §9.1 against a cloud upstream.

They *can* against a **local model** (Ollama, LM Studio): no key, no plan, no policy question, and
the sidecar still synthesizes the bus. That is the only admissible form of the proxy tier for this
platform, and the preset says so (`auth_probe: local_upstream_only`). **D-6.**

OpenCode is the sharp case: it ships OAuth flows into Anthropic's and OpenAI's plans. Anthropic's
terms forbid third parties routing requests through plan credentials; that is the forbidden
pattern regardless of which binary does the routing. OpenCode is admissible BYOK-to-a-local-model
only, same as the proxy tier.

---

## 10. Adding CLI #N — the checklist

The deliverable this whole design exists to produce.

1. **Classify** (§3): hook surface / config-file hooks / nothing. Then **turn end**: `Stop` hook,
   process exit, or PTY idle.
2. **Check admissibility** (§9.1): first-party own-plan login in the unmodified binary? If not,
   local-upstream only, or not at all.
3. **Add the preset row.** Launch flags, resume shape, `hook_events`, `hook_timeout_literal`,
   `allow_is_silence`, `event_name_source`, env to strip, forbidden/required args, auth probe.
4. **Tier-2 only:** implement `prepare()` — where its config home lives (**per agent**), how hook
   entries are written, how trust is recorded. Point them at the shim we already ship.
   *Not Claude-shaped:* `normalize_event` + `render_response` — a name map each way.
   *Tier-3:* the sidecar. *Tier-4:* declare it ungated and accept D-3's answer.
5. **Map its tools** in `tool_intent()` — the shell tool, the edit tool, the read tools.
6. **Map its usage** in `read_usage()` — where the counts are, which record carries the model,
   cumulative or per-turn.
7. **Add a fake-CLI fixture** mirroring `tests/fake_claude.py`: refuse the forbidden args, drive the
   hooks in *its* vocabulary, write a transcript in *its* shape.
8. **Add its id** to `orchestrator/core/cli_presets.py`. The parity test (§8.1) will tell you if you
   forgot.
9. Nothing in the frontend. Nothing in `host.py`. Nothing in the backend lane.

Steps 1–3, 7, 8 are a row and a fixture. Step 4 is the only real code, and only for tier 2+.

---

## 11. Test plan

CI-only, no spend, no live CLI (workspace rule).

- **Preset invariants** (table-driven, every preset): `forbidden_args` never intersects the argv
  `launch_args` produces; `required_args` always does; every preset id resolves to an adapter;
  `strip_env` covers that CLI's credential vars; `turn_end` is set.
- **Registry parity**: host `presets.py` ids == `orchestrator/core/cli_presets.py` ids.
- **`ToolIntent` mapping**, per CLI: its shell tool → `SHELL` with the command; its edit tool →
  `FILE_WRITE` with the paths.
- **Policy neutrality**: the existing `policy` suite re-run through `ToolIntent` for both CLIs —
  identical verdicts. This is the proof the rules did not change.
- **Translation round trip**: a recorded payload per event per CLI → canonical → the reply →
  `render_response` → the bytes that CLI documents. **agy allow → `None`.** Gemini deny →
  `{decision, reason}`.
- **Turn end**: `stop_hook` ends on `Stop`; `process_exit` ends on exit and reads stdout;
  `pty_idle` ends after the quiet window and marks the ticket `ungated`.
- **`fake_codex` end-to-end** (mirroring `test_session_fake_claude.py`): refuses `exec` and the
  sandbox bypass, fires hooks, writes a rollout in the §6.8 shape; the session reports usage
  (from the last cumulative record), the model (from `turn_context`), files touched and denials.
- **`prepare()` isolation**: after a Codex session, `~/.codex/config.toml` is byte-identical;
  `auth.json` was linked, never copied into our state dir, never read; the agent home persists
  across two tickets and the second resumes the first's session id.
- **Preflight refusals**: api-key-mode Codex → the refusal sentence, not a spawn; the host
  announces `served: false` with that reason.
- **Claim filter**: a `codex` ticket is not claimed by a claude-only host and its `blocked_reason`
  names the missing CLI; it is claimed the moment a codex host heartbeats.

---

## 12. Open decisions

- **D-1 · Build order.** The §8.2 filter first (small, fixes a live silent-wrong-CLI bug), then the
  seam refactor with Claude as the only adapter (behaviour-preserving, provable by the existing
  suite), then Codex on top. Three stacked PRs. *Recommended* — each is independently revertible and
  the middle one changes no behaviour at all.
- **D-2 · Codex model validation.** `_CODEX_MODEL_RE` currently accepts any lowercase token. Either
  keep it permissive (the CLI refuses a bad model honestly) or seed a real list. *Recommended:*
  permissive — a hardcoded list rots, and PRD-223's lesson was about the *route* validating nothing,
  which the preset now fixes. PRD-223's quarantine list, if it is a governance rule and not an
  Auto-route rule, applies to CLI models too — say which.
- **D-3 · Seed-tier CLIs** (Copilot, Cursor, and any tier-2 CLI whose bridge is unverified).
  Refuse them for session agents, or allow them with a permanent "ungated" marker on the ticket?
  No `PreToolUse` gate, no never-push rule, no approvals. Not a blocker for Codex or Grok; must be
  answered before Copilot/Cursor.
- **D-4 · Where `--worktree` truth lives.** Both CLIs have a native worktree mode; PRD-239's
  per-agent `worktree_per_ticket` already gates it. The preset supplies only the *spelling*
  (`worktree_args`) and the exclusion (`worktree_excludes_resume`); the decision stays the agent's.
- **D-5 · Grok isolation** (§13). Isolate via `GROK_HOME` like Codex, or accept munder's scoped
  global hook file? *Recommended:* `GROK_HOME` + linked login, falling back to munder's approach
  only if `GROK_HOME` breaks the login. Needs the binary to decide.
- **D-6 · Proxy tier admissibility** (§9.4). Local-upstream only, or not at all? *Recommended:*
  local-upstream only, marked on the ticket. Not needed before Qwen/Crush.
- **D-7 · `exposeCodexDataDirs`** (§6.1). Write a namespaced `sessions/` folder into the operator's
  global `~/.codex` so their own `codex resume` picker sees Automatos sessions? *Recommended:* off
  by default, an operator opt-in — it is a global write, however tidy.

---

## 13. Grok — the next CLI, what is known and what is not

Nothing here is verified on this machine: `grok` is not installed. Every fact is munder-sourced
(`agentProvider.ts` grok preset, `hive.ts:2382 installGrokHooks`, `GROK_HOOK_SHIM`).

**What munder established:** Grok supports the Claude hook events and the Claude decision
vocabulary (`decision: block/deny`, `continue: false`, `additionalContext` pass-through), but its
**stdin payload is camelCase** (`hookEventName`, `sessionId`, `toolName`, `toolInput`,
`cwd|workspaceRoot`, `stopHookActive`, `notificationType`) with snake_case event *values*
(`pre_tool_use`). It adds four events the bus does not model — `post_tool_use_failure`,
`permission_denied`, `stop_failure`, `notification` — the first three dropped, the last mapped.
Positional initial prompt; `--resume <id>`; `--model`; auto mode is `--permission-mode
bypassPermissions` (Claude's spelling — which puts it on **our forbidden list**; the ungated stance
for Grok is whatever its `acceptEdits` equivalent is — **verify**).

**Where munder and this design part ways.** munder installs the hook file into the operator's
**global** `~/.grok/hooks/munder-hive.json` and keeps it inert for ordinary sessions by scoping the
shim on an `AGENT_ID` env var — its stated reason being that global hooks are trusted and
sessions/resume stay in the operator's normal `GROK_HOME`. That is a permanent write into the
operator's global config, which §6.1 rules out for Codex. munder's own comment says `GROK_HOME`
exists; the open question is whether pointing it at a per-agent home keeps the login (as
`CODEX_HOME` does with `auth.json` linked) or loses it. **D-5**, and it needs the binary: install
Grok, find where it stores the login, try the isolated home first.

**Adapter shape, if D-5 goes the recommended way:** tier `hooks`; `config_home_env: GROK_HOME`;
`prepare()` = link the login file + write the hook entries in Grok's config format (**verify the
file and its timeout unit**); `normalize_event` = the camelCase→snake_case key map plus the event
name map; `render_response` = deny → `{decision: "deny", reason}`, everything else Claude-shaped;
`tool_intent` = **verify Grok's tool names**; `read_usage` = **verify Grok's transcript**. Five
verifies — that is the honest size of Grok today.

---

## 14. Rollout order — by evidence, not by list position

The owner's roadmap is Claude · Codex · Grok · Kimi · Antigravity · Qwen · Gemini · OpenCode ·
Crush · Pi · Copilot · Cursor. Sorting by *hook surface × admissible own-plan login × what can be
verified*, the order that keeps every step shippable is:

| Wave | CLIs | Why here |
|---|---|---|
| 0 | the §8.2 filter, the seam with Claude alone | fixes a live bug; proves the refactor is behaviour-neutral |
| 1 | **Codex** | tier-2, identity translation, every fact verified except the six in §6.10; binary present |
| 2 | **Gemini CLI**, **Grok** | tier-2 with a name-map translation each; Gemini's isolation story is the cleanest of the twelve (munder); Grok needs D-5 and the binary |
| 3 | **Antigravity** | tier-2 but global-config write + argv event names + fail-closed allow — three sharp edges, all modelled by rev 2 fields |
| 4 | **Kimi** | tier-2 *if* its hooks are as munder says; nobody has written the bridge — a genuine spike |
| 5 | **Copilot**, **Cursor** | seed tier; blocked on D-3; Cursor is on this machine and its `--output-format stream-json` print mode is worth a look as a structured-stdout bridge before declaring it ungated |
| 6 | **OpenCode**, **Pi** | tier-2 mechanically, but BYOK and `LIVE-UNVERIFIED` even in munder; OpenCode's OAuth-into-Anthropic path is the forbidden pattern (§9.4) |
| 7 | **Qwen**, **Crush** | proxy tier; local-upstream only (D-6); Crush's `type-into-tui` seed delivery is the last new mechanism |

Grok moves from third to a shared second because the D-5 question needs the binary before any
code, and Gemini's bridge is the better-understood of the two translating shims to write first —
the second translation is then a copy with a different map.
