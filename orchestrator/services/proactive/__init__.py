"""Proactive Auto (PRD-268): the services that read the database for a check-in.

The rules are pure functions in :mod:`modules.proactive`; these services read what they
decide on.

- :mod:`services.proactive.signals`: what changed in a workspace over a check-in's
  window, the signals Auto's workspace check-in starts from (§6.2.1).
- :mod:`services.proactive.runner`: the two check-ins, Auto's workspace check-in and an
  agent's, each one run in the read-only lane (§6.2.1, §6.9). It uses
  :mod:`~services.proactive.switches` (the platform mode, the workspace switch, the
  person's pause), :mod:`~services.proactive.gates` (what refuses a run's start),
  :mod:`~services.proactive.runs` (the run rows), :mod:`~services.proactive.prompts`,
  :mod:`~services.proactive.loop` (the tool loop) and :mod:`~services.proactive.reads`
  (the run's read log, which a suggestion's evidence must come from).
"""
