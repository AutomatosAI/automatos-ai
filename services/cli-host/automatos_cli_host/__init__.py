"""Automatos CLI host — PRD-234 S1b.

The local process that runs ``runtime: cli`` tickets as the user's OWN CLI
sessions on their machine — Claude Code today, the CLIs the preset table names
as their adapters land — with Automatos (the local edition, in Docker) as the
manager above: it pairs once, claims tickets, runs each as a supervised
interactive session, streams the session's hook events to the board and posts
one result per attempt. Which CLI is a parameter of the ticket (CLI adapter
design, ``docs/architecture/CLI-RUNTIME-ADAPTER-DESIGN.md``): ``presets.py``
says how a CLI is spelled, ``adapters/`` does what a preset cannot say.

Invariants (PRD-234 §Terms — every one has a source guard in the tests):

* the ``claude`` binary is the user's own, unmodified, found on their login-shell
  ``PATH``; nothing is bundled or patched;
* login is the user's own (``claude login``); this process never reads, copies,
  forwards or sets a credential and never overrides ``CLAUDE_CONFIG_DIR``;
* no identity games: no ``CLAUDE_CODE_ENTRYPOINT``, no ``ANTHROPIC_BASE_URL``,
  no ``--bare`` (it disables OAuth by design);
* agent turns run Claude Code's normal INTERACTIVE mode (the surface Anthropic
  keeps on the plan) — ``claude -p`` and the Agent SDK are not used;
* one user, one machine: the host serves the local instance's single operator.

Standard library only, Python 3.9+, so ``make cli-host`` needs no virtualenv.
"""

__version__ = "0.12.0"  # 0.12.0: a claim carries ``brand_files``, the brand kit's uploaded logo and logo mark; they are written under brand/ in the ticket folder and the ticket file names them (F332, brand_files.py). 0.11.0: GitHub Copilot CLI is a session CLI (PRD-253 W1–W3) — `copilot -p` with a per-agent COPILOT_HOME and Claude-format hooks; the Canvas terminal opens a per-agent CLI (Codex, Copilot) in the agent's own home; a CLI's own permission request after the gate allowed is answered with the gate's verdict. 0.10.0: Plan runs on every CLI (PRD-253 Wave P) — the claim's ``permission_mode`` is THIS turn's mode (``edits`` once the ticket's plan is approved) and it carries ``plan_approved``; a turn that made a plan reports it as a ``PlanReady`` event, and a finished session's last events reach the backend before its result. 0.9.0: a claim carries ``permission_mode`` (manual | edits | plan | auto), the agent's or the workspace's; ``--permission-mode`` on this host overrides it (permission_modes.py). 0.8.0: a claim carries this ticket's Automatos tools — ``session_tools`` (name/description/schema), ``session_tools_path`` and a per-ticket ``session_token``; the adapter writes them as an MCP server the session can call (PRD-245). 0.7.0: the CLI is a parameter — presets + adapters, capabilities announce every CLI with served/reason, --cli-binary ID=PATH (CLI adapter design); 0.6.0: a ticket with no folder runs in <root>/sessions/<ticket>; 0.5.0: results and TerminalClosed carry the turn's token usage; 0.3.0: persona + skills in the session prompt (PRD-239)
