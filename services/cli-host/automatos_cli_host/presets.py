"""The preset table — one row per CLI, pure data (CLI adapter design §4.1).

A ``CliPreset`` says how a CLI is *spelled*: its binary, the flags that select a
model / resume a session / add a directory, the hook events it can deliver and
the timeout literal its config expects, the environment it must never inherit,
the arguments that would break the subscription or the gate, and how to tell
"logged in with the operator's own plan". Nothing here runs anything — the
adapter (``adapters/``) reads a preset and does the work; a well-behaved CLI is
a row in this file and nothing else.

Stdlib only, no I/O: tests, the backend's registry mirror
(``orchestrator/core/cli_presets.py``, kept in step by a parity test) and a
future manifest can all read it without dragging the host in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from .permission_request import PERMISSION_REQUEST_MODES

# ── vocabularies ─────────────────────────────────────────────────────────────
TIER_NATIVE = "native"      # the CLI's own hooks + a real system-prompt flag (claude)
TIER_HOOKS = "hooks"        # a config-file hook shim in a per-agent config home (codex, gemini, grok, …)
TIER_PROXY = "proxy"        # no hook surface: a loopback proxy synthesizes the events (qwen, crush)
TIER_SEED = "seed"          # no lifecycle at all: spawn, seed a prompt, read the end (cursor)
TIERS = (TIER_NATIVE, TIER_HOOKS, TIER_PROXY, TIER_SEED)

TURN_END_STOP_HOOK = "stop_hook"        # the turn ends on the Stop hook (tiers 1–3)
TURN_END_PROCESS_EXIT = "process_exit"  # print-mode CLIs: the turn is the process lifetime
TURN_END_PTY_IDLE = "pty_idle"          # an interactive TUI with no hooks: quiet output = done (weakest)
TURN_ENDS = (TURN_END_STOP_HOOK, TURN_END_PROCESS_EXIT, TURN_END_PTY_IDLE)

PROMPT_POSITIONAL = "positional"        # ``cli "<pointer>"``
PROMPT_FLAG = "flag"                    # ``cli -i "<pointer>"`` — the flag is ``initial_prompt_flag``
PROMPT_TYPE_INTO_TUI = "type_into_tui"  # the CLI reads a positional as a subcommand (crush, cursor)

EVENT_NAME_PAYLOAD = "payload"          # ``hook_event_name`` is in the JSON (claude, codex, gemini, grok)
EVENT_NAME_ARGV = "argv"                # the payload carries no name; the hook command is given it (agy)

SCOPE_NONE = "none"                     # the CLI's own config is not relocated (claude: ``--settings``)
SCOPE_PER_AGENT = "per_agent"           # one config home per agent per host — the default for tier 2
SCOPE_PER_SESSION = "per_session"       # never for a CLI whose session index lives in the home (§6.1)

# PRD-253 D4: how long the hook shim waits for the host's answer to a held call
# (``AUTOMATOS_HOOK_WAIT_SECONDS``) before it denies on its own. Every preset's
# timeout for a held event sits ABOVE it: a CLI that kills a hook first decides
# the call itself, and a CLI whose timed-out hook fails open (Copilot) would run it.
HOOK_WAIT_SECONDS = 560
HELD_HOOK_TIMEOUT_SECONDS = 600
# ...and the host's own hold ends BELOW it: ``--ask-timeout`` is capped here, so
# the host answers every held call itself. An answer that came after the shim gave
# up would be recorded for a call the CLI had already been told was denied.
MAX_HOLD_SECONDS = HOOK_WAIT_SECONDS - 30
UNSET_HOLD_SECONDS = 120.0


def hold_seconds(ask_timeout: Optional[float]) -> float:
    """How long a held call waits for the operator inside a turn: ``--ask-timeout``
    (``UNSET_HOLD_SECONDS`` when unset), never past ``MAX_HOLD_SECONDS``."""
    return min(float(ask_timeout or UNSET_HOLD_SECONDS), MAX_HOLD_SECONDS)

# The bus (design §5): every event the host models. A preset lists the subset its CLI delivers.
BUS_EVENTS: FrozenSet[str] = frozenset({
    "SessionStart", "UserPromptSubmit", "PreToolUse", "PermissionRequest", "PostToolUse",
    "Notification", "Stop", "SubagentStop", "PreCompact", "PostCompact", "SessionEnd",
})


@dataclass(frozen=True)
class AuthProbe:
    """How ``preflight()`` tells the operator is logged in with their OWN plan.
    ``kind`` is interpreted by the adapter; ``code`` is the ticket's exit_reason
    and ``refusal`` the sentence it shows when the probe fails. Never a
    credential read — a file's presence or a mode."""
    kind: str
    code: str
    refusal: str


@dataclass(frozen=True)
class CliPreset:
    id: str                                   # matches the backend registry (core/cli_presets.py)
    label: str                                # what the picker shows
    binary: str                               # default executable; --cli-binary id=path overrides per host
    tier: str                                 # TIERS
    turn_end: str                             # TURN_ENDS

    # ── launch ──────────────────────────────────────────────────────────────
    model_flag: Optional[str] = None          # "--model" / "-m"; None = no model selection
    session_id_flag: Optional[str] = None     # "--session-id"; None = the CLI mints one, learned on SessionStart
    resume_flag: Optional[str] = None         # "--resume <id>"
    resume_subcommand: Optional[str] = None   # "resume" — Codex resumes by SUBCOMMAND, never a flag
    cwd_flag: Optional[str] = None            # None = spawn with cwd=; "-C" for Codex
    add_dir_flag: Optional[str] = None        # "--add-dir"
    mcp_config_flag: Optional[str] = None     # "--mcp-config" (PRD-245 W1; None ⇒ this CLI takes MCP elsewhere)
    worktree_args: Tuple[str, ...] = ()       # ("--worktree",) / ("--enable", "worktrees", "--worktree")
    worktree_excludes_resume: bool = False    # Codex: --worktree cannot resume (§6.7)
    worktree_takes_name: bool = False         # Claude: ``--worktree <name>``; Codex names its own
    system_prompt_flag: Optional[str] = None  # "--append-system-prompt-file"; None ⇒ the soul rides the bus (§6.9)
    initial_prompt: str = PROMPT_POSITIONAL
    initial_prompt_flag: Optional[str] = None
    name_flag: Optional[str] = None           # "--name" — how the session shows in the CLI's own UI
    ungated_stance: Tuple[str, ...] = ()      # "don't prompt, we gate at PreToolUse"
    plan_stance: Tuple[str, ...] = ()         # Plan mode's start instead; () = this CLI has no plan mode
    required_args: Tuple[str, ...] = ()       # always on the command line (narrowing, hook trust)

    # ── hooks ───────────────────────────────────────────────────────────────
    hook_events: FrozenSet[str] = frozenset() # the bus events THIS CLI can deliver
    hook_timeouts: Mapping[str, Any] = field(default_factory=dict)  # {"*": literal, "<Event>": literal} — the CLI's own unit
    allow_is_silence: bool = False            # agy: any stdout object is a decision; allow = write nothing
    event_name_source: str = EVENT_NAME_PAYLOAD
    # A PermissionRequest hook: "deny" (a prompt reached the TUI, nobody watches it) or
    # "rejudge" (the CLI re-asks after the gate allowed — the gate's verdict on the same
    # call, never a card; permission_request.py, PRD-253 S1.4).
    permission_request: str = "deny"
    # How long a gated session may take to prove its gate loaded (its SessionStart
    # hook); None = the host's --startup-timeout. A print-mode CLI shows no login
    # screen or dialog, so its window can be short (turn_end.py, PRD-253 S0.2).
    startup_timeout_seconds: Optional[float] = None

    # ── environment ─────────────────────────────────────────────────────────
    config_home_env: Optional[str] = None     # "CODEX_HOME", "GROK_HOME", "OPENCODE_CONFIG_DIR", …
    config_home_scope: str = SCOPE_NONE
    strip_env: FrozenSet[str] = frozenset()   # credentials / redirection this CLI must never inherit
    strip_env_prefixes: Tuple[str, ...] = ()  # session markers (a nested CLI must not think it is a child)
    keep_env: FrozenSet[str] = frozenset()    # the operator's own configuration that IS forwarded
    extra_env: Mapping[str, str] = field(default_factory=dict)

    # ── invariants ──────────────────────────────────────────────────────────
    forbidden_args: Tuple[str, ...] = ()      # per CLI: what breaks the subscription posture or our gate

    # ── identity ────────────────────────────────────────────────────────────
    auth_probe: Optional[AuthProbe] = None
    install_hint: Optional[str] = None        # shown when the binary is missing; never auto-run
    docs_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"{self.id}: tier must be one of {TIERS}, got {self.tier!r}")
        if self.turn_end not in TURN_ENDS:
            raise ValueError(f"{self.id}: turn_end must be one of {TURN_ENDS}, got {self.turn_end!r}")
        if self.tier == TIER_SEED and self.turn_end == TURN_END_STOP_HOOK:
            raise ValueError(f"{self.id}: a seed-tier CLI has no Stop hook to end a turn on")
        unknown = set(self.hook_events) - BUS_EVENTS
        if unknown:
            raise ValueError(f"{self.id}: hook_events not on the bus: {sorted(unknown)}")
        if self.initial_prompt == PROMPT_FLAG and not self.initial_prompt_flag:
            raise ValueError(f"{self.id}: initial_prompt=flag needs initial_prompt_flag")
        if self.permission_request not in PERMISSION_REQUEST_MODES:
            raise ValueError(f"{self.id}: permission_request must be one of {PERMISSION_REQUEST_MODES}")

    @property
    def hold_events(self) -> Tuple[str, ...]:
        """The hooks the host may HOLD while the operator answers (the gate)."""
        return tuple(e for e in ("PreToolUse", "PermissionRequest") if e in self.hook_events)

    def hook_timeout(self, event: str) -> Any:
        return self.hook_timeouts.get(event, self.hook_timeouts.get("*"))


# ── the rows ─────────────────────────────────────────────────────────────────

CLAUDE = CliPreset(
    id="claude",
    label="Claude Code",
    binary="claude",
    tier=TIER_NATIVE,
    turn_end=TURN_END_STOP_HOOK,
    model_flag="--model",
    session_id_flag="--session-id",
    resume_flag="--resume",
    add_dir_flag="--add-dir",
    # PRD-245 W1: Automatos' own MCP server, passed per session. ``--strict-mcp-config``
    # (required_args) means this is the ONLY server the session loads — the
    # operator's own stay out of an unattended ticket.
    mcp_config_flag="--mcp-config",
    worktree_args=("--worktree",),
    worktree_takes_name=True,
    system_prompt_flag="--append-system-prompt-file",
    initial_prompt=PROMPT_POSITIONAL,
    name_flag="--name",
    # acceptEdits: no prompt for edits; everything else reaches PreToolUse, where WE decide.
    ungated_stance=("--permission-mode", "acceptEdits"),
    # Plan mode: Claude Code's own. It explores, presents the plan with ExitPlanMode
    # (a card), and leaves plan mode when the gate approves it.
    plan_stance=("--permission-mode", "plan"),
    # The operator's user-scope settings only, no repo .claude/, no MCP from the folder.
    required_args=("--setting-sources", "user", "--strict-mcp-config"),
    hook_events=BUS_EVENTS,
    # PreToolUse may HOLD while the approvals inbox answers: the shim answers by
    # HOOK_WAIT_SECONDS, before Claude's own timeout would decide the call (D4). Seconds.
    hook_timeouts={"*": 60, "PreToolUse": HELD_HOOK_TIMEOUT_SECONDS, "PermissionRequest": HELD_HOOK_TIMEOUT_SECONDS},
    config_home_scope=SCOPE_NONE,              # per-session ``--settings``; CLAUDE_CONFIG_DIR stays the operator's
    strip_env=frozenset({
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_ENTRYPOINT",   # no identity games
        "CLAUDE_CODE_OAUTH_TOKEN",  # the CLI reads its own login; we never carry a token
    }),
    # This host is often started from inside a Claude Code terminal; an inherited
    # CLAUDE_CODE_CHILD_SESSION silently disables transcript saving (breaks --resume).
    strip_env_prefixes=("CLAUDECODE", "CLAUDE_"),
    keep_env=frozenset({"CLAUDE_CONFIG_DIR", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"}),
    # Headless posture (billing) or a bypass of our gate. ``--bare`` never reads OAuth.
    forbidden_args=("-p", "--print", "--bare", "--dangerously-skip-permissions", "--permission-mode bypassPermissions"),
    auth_probe=AuthProbe(
        kind="claude_onboarding",
        code="claude_not_onboarded",
        refusal="Claude Code has never been run interactively on this machine. Run `claude` once in your terminal and log in, then retry.",
    ),
    install_hint="Claude Code is not installed on this machine (no `claude` on your PATH). Install it and run `claude login`.",
    docs_url="https://docs.claude.com/en/docs/claude-code",
)

# Codex: the row is here so the registry (and the backend's mirror) know the CLI;
# the adapter that runs it lands with the Codex wave. Until then the host announces
# ``codex: served=false`` with the reason, and the claim filter keeps its tickets
# waiting for a host that can. Every value below is from design §6 (Codex 0.154.0 +
# munder's production bridge); the six §6.10 checks are verified when the adapter lands.
CODEX = CliPreset(
    id="codex",
    label="Codex",
    binary="codex",
    tier=TIER_HOOKS,
    turn_end=TURN_END_STOP_HOOK,
    model_flag="--model",
    session_id_flag=None,                      # Codex mints the id; learned on SessionStart
    resume_subcommand="resume",                # ``codex resume <id>`` — no --resume flag exists
    cwd_flag="-C",
    add_dir_flag="--add-dir",
    worktree_args=("--enable", "worktrees", "--worktree"),
    worktree_excludes_resume=True,
    system_prompt_flag=None,                   # the soul rides UserPromptSubmit → additionalContext
    initial_prompt=PROMPT_POSITIONAL,
    ungated_stance=("-a", "never", "-s", "workspace-write"),   # approvals off, OS sandbox on
    required_args=("--dangerously-bypass-hook-trust",),        # trusts OUR hook file; not a gate bypass (§6.4)
    hook_events=BUS_EVENTS - {"Notification"},
    # SECONDS, and 0 floors to 1 s (§6.3). A held call outlasts the shim's wait (D4):
    # with approvals off, a hook Codex timed out first would leave the call to Codex.
    hook_timeouts={"*": 30, "PreToolUse": HELD_HOOK_TIMEOUT_SECONDS, "PermissionRequest": HELD_HOOK_TIMEOUT_SECONDS},
    config_home_env="CODEX_HOME",
    config_home_scope=SCOPE_PER_AGENT,         # the session index lives in the home (§6.1)
    strip_env=frozenset({"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL"}),
    extra_env={"CODEX_NON_INTERACTIVE": "1"},
    forbidden_args=("--dangerously-bypass-approvals-and-sandbox", "exec", "--ephemeral"),
    auth_probe=AuthProbe(
        kind="codex_chatgpt_login",
        code="codex_api_key_login",
        refusal="Codex is logged in with an API key, not your ChatGPT plan. Run `codex login` (ChatGPT) on this machine, then retry.",
    ),
    install_hint="Codex is not installed on this machine (no `codex` on your PATH). Install it and run `codex login`.",
    docs_url="https://github.com/openai/codex",
)

# GitHub Copilot CLI (PRD-253): the hooks tier — the design doc's "seed" row was read
# from munder's code, not the binary. ``copilot -p``: hooks fire in print mode, the
# turn is the process, and with no allow flag a call the gate did not allow is
# refused by Copilot itself (D1, D3). Facts from the 1.0.91 bundle; what only a
# live run proves is listed under the PRD's "Verify at build".
COPILOT = CliPreset(
    id="copilot",
    label="GitHub Copilot",
    binary="copilot",
    tier=TIER_HOOKS,
    turn_end=TURN_END_PROCESS_EXIT,
    startup_timeout_seconds=30,                # no login screen or dialog in -p (turn_end.py)
    model_flag="--model",
    session_id_flag="--session-id",            # a new session with the backend's pre-assigned uuid
    resume_flag="--resume",                    # an unknown id fails; --session-id would silently start fresh
    add_dir_flag="--add-dir",
    mcp_config_flag="--additional-mcp-config", # value: "@<session>/mcp.json"
    worktree_args=("--worktree",),
    worktree_takes_name=True,
    system_prompt_flag=None,                   # the soul rides UserPromptSubmit → additionalContext (§6.9)
    initial_prompt=PROMPT_FLAG,
    initial_prompt_flag="-p",
    name_flag="--name",
    ungated_stance=(),                         # nothing: the gate's allow is the only lift (D3)
    plan_stance=(),                            # Plan is the plan turn (Wave P); Copilot's own --plan is not used
    permission_request="rejudge",              # its own path/URL checks may re-ask after the gate allowed (S1.4)
    # Never --no-auto-login: it switches off the stored login AND the gh fallback, the
    # only ways a session signs in (env tokens are stripped), so every session failed
    # "No authentication information found" (F233). Without a credential, -p exits
    # with that error; it never waits on a login prompt.
    required_args=("--no-ask-user", "--disable-builtin-mcps", "--no-remote", "--no-auto-update"),
    hook_events=BUS_EVENTS - {"PostCompact"},
    # SECONDS (``timeoutSec``). A timed-out Copilot hook FAILS OPEN, so a held call
    # must be answered by the shim first (D4).
    hook_timeouts={"*": 60, "PreToolUse": HELD_HOOK_TIMEOUT_SECONDS, "PermissionRequest": HELD_HOOK_TIMEOUT_SECONDS},
    config_home_env="COPILOT_HOME",
    config_home_scope=SCOPE_PER_AGENT,         # the session index lives in the home (§6.1)
    strip_env=frozenset({
        "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",          # a session never carries a token (D6)
        "COPILOT_ALLOW_ALL", "COPILOT_ASSISTED_APPROVAL",            # allow flags by another name (D3)
        "COPILOT_MODEL", "COPILOT_OFFLINE", "COPILOT_HOOK_ALLOW_LOCALHOST",
        "GITHUB_COPILOT_PROMPT_MODE_REPO_HOOKS", "GITHUB_COPILOT_PROMPT_MODE_WORKSPACE_MCP",
        "GITHUB_COPILOT_PROMPT_MODE_EXTENSIONS",                    # repo hooks, workspace MCP, extensions in -p
        "COPILOT_CLI",
    }),
    strip_env_prefixes=("COPILOT_PROVIDER_",), # BYOK never reaches a session
    keep_env=frozenset({"GH_HOST", "COPILOT_GH_HOST", "COPILOT_PROXY_KERBEROS_SPN"}),   # GHE.com data residency, proxy
    extra_env={"COPILOT_AUTO_UPDATE": "false"},
    forbidden_args=(
        "--allow-all-tools", "--allow-all", "--yolo", "--allow-all-paths", "--allow-all-urls", "--allow-tool",
        "--assisted-approval", "--enable-memory", "--config-dir", "--share-gist", "--remote", "--remote-export",
        "--cloud", "--connect", "--acp", "--server", "--headless", "-i", "--interactive", "--continue",
        "--mcp-github-auth",
    ),
    auth_probe=AuthProbe(
        kind="copilot_login",
        code="copilot_not_logged_in",
        refusal="GitHub Copilot is not logged in on this machine. Run `copilot login` (or `gh auth login` with an "
                "account that has a Copilot seat), then retry.",
    ),
    install_hint="GitHub Copilot CLI is not installed on this machine (no `copilot` on your PATH). Install it "
                 "(`brew install copilot-cli`, or `npm install -g @github/copilot`) and run `copilot login`.",
    docs_url="https://docs.github.com/copilot/concepts/agents/about-copilot-cli",
)

REGISTRY: Dict[str, CliPreset] = {p.id: p for p in (CLAUDE, CODEX, COPILOT)}
DEFAULT_CLI = CLAUDE.id


class UnknownCli(KeyError):
    """A ticket or a grant named a CLI this host has no preset for."""


def preset_for(cli_id: Optional[str]) -> CliPreset:
    """The preset for a CLI id; ``None``/empty = the default (what every session
    agent ran on before the field existed — the backend's claim uses the same default)."""
    key = (cli_id or DEFAULT_CLI).strip().lower()
    try:
        return REGISTRY[key]
    except KeyError:
        raise UnknownCli(f"no CLI preset named {cli_id!r}; known: {sorted(REGISTRY)}") from None


def union_strip_env() -> Tuple[FrozenSet[str], Tuple[str, ...], FrozenSet[str]]:
    """What the operator's own shell (the Canvas terminal) must never inherit:
    every preset's credentials and markers, keeping every preset's configuration."""
    strip: set = set()
    prefixes: list = []
    keep: set = set()
    for p in REGISTRY.values():
        strip |= set(p.strip_env)
        prefixes += [x for x in p.strip_env_prefixes if x not in prefixes]
        keep |= set(p.keep_env)
    return frozenset(strip), tuple(prefixes), frozenset(keep)
