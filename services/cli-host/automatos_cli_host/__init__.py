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

__version__ = "0.7.0"  # 0.7.0: the CLI is a parameter — presets + adapters, capabilities announce every CLI with served/reason, --cli-binary ID=PATH (CLI adapter design); 0.6.0: a ticket with no folder runs in <root>/sessions/<ticket>; 0.5.0: results and TerminalClosed carry the turn's token usage; 0.3.0: persona + skills in the session prompt (PRD-239)
