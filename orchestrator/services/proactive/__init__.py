"""Proactive Auto (PRD-268): the services that read the database for a check-in.

The rules are pure functions in :mod:`modules.proactive`; these services read what they
decide on.

- :mod:`services.proactive.signals`: what changed in a workspace over a check-in's
  window, the signals Auto's workspace check-in starts from (§6.2.1).
- :mod:`services.proactive.runner`: the two check-ins, Auto's workspace check-in and an
  agent's, each one run in the read-only lane (§6.2.1, §6.9). It uses
  :mod:`~services.proactive.switches` (the platform mode, the workspace switch, the
  person's pause), :mod:`~services.proactive.limits` (the workspace's caps over the config
  defaults, and what they count), :mod:`~services.proactive.gates` (what refuses a run's start),
  :mod:`~services.proactive.runs` (the run rows), :mod:`~services.proactive.prompts`,
  :mod:`~services.proactive.loop` (the tool loop) and :mod:`~services.proactive.reads`
  (the run's read log, which a suggestion's evidence must come from).
  :mod:`~services.proactive.failing_streaks` keeps an agent's failing streak: one card,
  a slower check-in from the third failure, the reset (§6.2.2), with
  :mod:`~services.proactive.failures` (the error class) and
  :mod:`~services.proactive.streak_text` (what the card says).
- :mod:`services.proactive.delivery`: in live mode, a run's new suggestions posted in the
  addressee's Auto thread and one ``suggestion_ready`` bell per person (§6.5.1, K8, K9), in the
  words of :mod:`~services.proactive.delivery_text`.
- :mod:`services.proactive.heartbeat_bridge`: the jobs the heartbeat scheduler runs, today's
  tick, a check-in or both, by the proactive mode (§6.2.1, K7, K10); a live check-in rings
  the heartbeat bell only through :mod:`~services.proactive.heartbeat_bell`.
- :mod:`services.proactive.decisions`: a person's Start or Hide of a suggestion (§6.2.5), the
  ticket a Start files in :mod:`~services.proactive.start_ticket`, and what the routes say in
  :mod:`~services.proactive.decision_text`; :mod:`~services.proactive.listing` lists the
  suggestions and the runs for ``api/proactive.py``, and :mod:`~services.proactive.settings`
  reads and writes the workspace switch, its caps and a person's pause for it (§6.9).
- :mod:`services.proactive.display_state`: one status vocabulary (§6.11.1): where a row
  stands, its next step from :mod:`~services.proactive.strings` in the person's language,
  and what Try again does; the needs-you rows, the Missions list and the board's ticket
  reads carry them.
- :mod:`services.proactive.retention`: the run rows (``proactive_runs``) kept
  ``PROACTIVE_RUNS_RETENTION_DAYS`` (90), the older ones deleted once a day on the leader (§6.2).
"""
