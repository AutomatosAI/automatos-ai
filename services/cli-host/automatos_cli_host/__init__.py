"""Automatos CLI host — PRD-234 S1b.

The local process that runs ``runtime: cli`` tickets as the user's OWN Claude
Code sessions on their machine, with Automatos (the local edition, in Docker)
as the manager above: it pairs once, claims tickets, runs each as a supervised
interactive ``claude`` session, streams the session's hook events to the board
and posts one result per attempt.

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

__version__ = "0.1.0"
